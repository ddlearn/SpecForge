"""Image-only Qwen-VL prompt preparation for online server capture.

The adapter deliberately keeps the control plane tensor-free.  SpecForge does
the same CPU image preprocessing as the target ``AutoProcessor`` so it can
filter on the final expanded sequence before tasks enter the controller.  The
SGLang request still carries the original (unexpanded) token IDs and shared
image sources. Correctness therefore requires validating local/SGLang processor
parity in the final deployment environment before training.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from specforge.config import Config
from specforge.runtime.contracts import PromptTask

from .prompt_builder import _iter_records
from .template import TEMPLATE_REGISTRY

logger = logging.getLogger(__name__)

_MAX_REJECT_EXAMPLES = 5


def _source_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_image_identities(
    rows: Sequence[tuple[int, Mapping[str, Any]]], *, dataset_dir: Path
) -> list[dict[str, Any]]:
    """Return cheap invalidation metadata without reading image contents."""

    identities: list[dict[str, Any]] = []
    for _line_number, record in rows:
        conversations = record.get("conversations")
        if not isinstance(conversations, list):
            continue
        for message in conversations:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content", message.get("value", ""))
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                if isinstance(image_url, Mapping):
                    image_url = image_url.get("url")
                if not isinstance(image_url, str) or not image_url:
                    continue
                parsed = urlparse(image_url)
                if parsed.scheme in ("http", "https") or (
                    parsed.scheme and parsed.scheme != "file"
                ):
                    continue
                path = (
                    Path(unquote(parsed.path))
                    if parsed.scheme == "file"
                    else Path(image_url)
                )
                if not path.is_absolute():
                    path = dataset_dir / path
                resolved = path.resolve()
                try:
                    stat = resolved.stat()
                except OSError:
                    identities.append({"path": str(resolved), "missing": True})
                else:
                    identities.append(
                        {
                            "path": str(resolved),
                            "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                        }
                    )
    return identities


def _map_cache_path(
    config: Config,
    processor: Any,
    tokenizer: Any,
    *,
    source_path: str,
    raw_rows: Sequence[tuple[int, Mapping[str, Any]]],
    dataset_dir: Path,
    min_loss_tokens: int,
) -> Path | None:
    cache_dir = getattr(config.data, "cache_dir", None)
    if not cache_dir:
        return None
    image_processor = getattr(processor, "image_processor", None)
    image_processor_to_dict = getattr(image_processor, "to_dict", None)
    image_processor_config = (
        image_processor_to_dict() if callable(image_processor_to_dict) else None
    )
    identity = {
        "namespace": getattr(config.data, "cache_key", None),
        "source_path": str(Path(source_path).resolve()),
        "source_digest": _source_digest(source_path),
        "local_images": _local_image_identities(
            raw_rows, dataset_dir=dataset_dir
        ),
        "target_model": config.model.target_model_path,
        "processor_type": (
            f"{type(processor).__module__}.{type(processor).__qualname__}"
        ),
        "processor_chat_template": getattr(processor, "chat_template", None),
        "image_processor_config": image_processor_config,
        "tokenizer_type": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "tokenizer_init_kwargs": getattr(tokenizer, "init_kwargs", None),
        "tokenizer_size": len(tokenizer),
        "image_token_id": _image_token_id(processor, tokenizer),
        "merge_size": _spatial_merge_size(processor),
        "chat_template": config.data.chat_template,
        "train_only_last_turn": config.data.train_only_last_turn,
        "max_length": config.data.max_length,
        "min_pixels": config.data.min_pixels,
        "max_pixels": config.data.max_pixels,
        "min_loss_tokens": min_loss_tokens,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"qwen3_vl-{digest}.arrow"


class _ProcessorTokenizerProxy:
    """Let the existing loss-mask parsers render through ``AutoProcessor``."""

    def __init__(self, processor: Any, tokenizer: Any) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.last_rendered: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        processor_kwargs = dict(kwargs.pop("processor_kwargs", None) or {})
        if "add_special_tokens" in kwargs:
            processor_kwargs["add_special_tokens"] = kwargs.pop(
                "add_special_tokens"
            )
        if processor_kwargs:
            kwargs["processor_kwargs"] = processor_kwargs
        self.last_rendered = self.processor.apply_chat_template(messages, **kwargs)
        return self.last_rendered


def _safe_source_label(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return source


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _flatten_ids(value: Any, *, field: str) -> list[int]:
    value = _as_list(value)
    if len(value) == 1 and isinstance(value[0], Sequence):
        value = _as_list(value[0])
    if any(isinstance(item, Sequence) for item in value):
        raise ValueError(f"processor field {field} must be one-dimensional")
    return [int(item) for item in value]


def _processor_value(output: Any, key: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(key)
    return getattr(output, key, None)


def _normalize_grid(value: Any, *, expected_images: int) -> list[list[int]]:
    if value is None:
        raise ValueError("processor omitted image_grid_thw")
    rows = _as_list(value)
    if rows and not isinstance(rows[0], Sequence):
        rows = [rows]
    grid = [[int(item) for item in _as_list(row)] for row in rows]
    if len(grid) != expected_images or any(len(row) != 3 for row in grid):
        raise ValueError(
            "processor image_grid_thw shape mismatch: "
            f"expected ({expected_images}, 3), got {grid!r}"
        )
    if any(item <= 0 for row in grid for item in row):
        raise ValueError(f"processor returned non-positive image_grid_thw: {grid!r}")
    return grid


def _image_token_id(processor: Any, tokenizer: Any) -> int:
    for owner in (processor, getattr(processor, "image_processor", None)):
        value = getattr(owner, "image_token_id", None) if owner is not None else None
        if value is not None:
            return int(value)
    value = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if value is None or int(value) < 0:
        raise ValueError("Qwen processor/tokenizer does not define <|image_pad|>")
    return int(value)


def _spatial_merge_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    for name in ("merge_size", "spatial_merge_size"):
        value = getattr(image_processor, name, None)
        if value is not None:
            value = int(value)
            if value > 0:
                return value
    raise ValueError("Qwen image processor does not expose a positive merge_size")


def _expand_image_tokens(
    input_ids: Sequence[int],
    loss_mask: Sequence[int],
    *,
    image_token_id: int,
    token_counts: Sequence[int],
) -> tuple[list[int], list[int]]:
    if len(input_ids) != len(loss_mask):
        raise ValueError("base input_ids/loss_mask lengths differ")
    placeholders = sum(token == image_token_id for token in input_ids)
    if placeholders != len(token_counts):
        raise ValueError(
            f"conversation contains {placeholders} image placeholder(s), "
            f"but {len(token_counts)} image(s) were loaded",
        )
    expanded_ids: list[int] = []
    expanded_mask: list[int] = []
    image_index = 0
    for token, mask in zip(input_ids, loss_mask):
        if token != image_token_id:
            expanded_ids.append(int(token))
            expanded_mask.append(int(mask))
            continue
        count = int(token_counts[image_index])
        image_index += 1
        expanded_ids.extend([image_token_id] * count)
        # MVP accepts images only in user messages, so visual tokens never train.
        expanded_mask.extend([0] * count)
    return expanded_ids, expanded_mask


def _load_shared_image(
    source: str, *, dataset_dir: Path
) -> tuple[str, Any | None, tuple[str, str] | None]:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        import requests

        try:
            response = requests.get(source, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            return source, None, (
                "image_read_error",
                f"failed to download {_safe_source_label(source)!r}: {exc}",
            )
        request_source = source
        data = response.content

    elif parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        request_source = str(path.resolve())
    elif parsed.scheme:
        raise ValueError(
            "images must be an HTTP(S) URL or a filesystem path shared by "
            f"SpecForge and SGLang; got scheme {parsed.scheme!r}",
        )
    else:
        path = Path(source)
        if not path.is_absolute():
            path = dataset_dir / path
        request_source = str(path.resolve())

    if parsed.scheme not in ("http", "https"):
        try:
            data = Path(request_source).read_bytes()
        except OSError as exc:
            return request_source, None, (
                "image_read_error",
                f"failed to read shared image {request_source!r}: {exc}",
            )
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.load()
    except (OSError, ValueError) as exc:
        return request_source, None, (
            "image_decode_error",
            f"image {_safe_source_label(source)!r} cannot be decoded: {exc}",
        )
    return request_source, image, None


def _part_image_source(part: Mapping[str, Any]) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, Mapping):
        image_url = image_url.get("url")
    if not isinstance(image_url, str) or not image_url:
        raise ValueError("image_url part must contain a non-empty URL")
    return image_url


def _normalize_messages(
    conversations: Any, *, dataset_dir: Path
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[Any],
    tuple[str, str] | None,
]:
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")
    messages: list[dict[str, Any]] = []
    image_sources: list[str] = []
    images: list[Any] = []
    for message_index, raw_message in enumerate(conversations):
        if not isinstance(raw_message, Mapping):
            raise ValueError(f"message {message_index} must be an object")
        role = raw_message.get("role", raw_message.get("from"))
        role = {"human": "user", "gpt": "assistant", "model": "assistant"}.get(
            role, role
        )
        if role not in ("system", "user", "assistant"):
            raise ValueError(
                f"message {message_index} has unsupported role {role!r}",
            )
        content = raw_message.get("content", raw_message.get("value", ""))
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError(
                f"message {message_index} content must be a string or parts list",
            )
        normalized_parts: list[dict[str, Any]] = []
        for part_index, part in enumerate(content):
            if not isinstance(part, Mapping):
                raise ValueError(
                    f"message {message_index} part {part_index} must be an object",
                )
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError("text part must contain text")
                normalized_parts.append({"type": "text", "text": text})
                continue
            if part_type == "image_url":
                if role != "user":
                    raise ValueError(
                        f"images are supported only in user messages, got {role!r}",
                    )
                source = _part_image_source(part)
                request_source, image, image_failure = _load_shared_image(
                    source, dataset_dir=dataset_dir
                )
                if image_failure is not None:
                    return [], [], [], image_failure
                image_sources.append(request_source)
                images.append(image)
                normalized_parts.append(
                    {"type": "image", "image": request_source}
                )
                continue
            if part_type in ("video_url", "input_audio", "audio"):
                raise ValueError(f"MVP supports images only, got {part_type!r}")
            raise ValueError(
                f"unsupported content part {part_type!r}"
            )
        messages.append({"role": role, "content": normalized_parts})
    return messages, image_sources, images, None


def _parse_base_tokens(
    processor: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    chat_template: str,
    train_only_last_turn: bool,
) -> tuple[str, list[int], list[int]]:
    try:
        template = TEMPLATE_REGISTRY.get(chat_template)
    except KeyError as exc:
        raise ValueError(f"unknown chat template {chat_template!r}") from exc
    proxy = _ProcessorTokenizerProxy(processor, tokenizer)
    if template.parser_type == "general":
        from .parse import GeneralParser

        parser = GeneralParser(proxy, template)
    elif template.parser_type == "thinking":
        from .parse import ThinkingParser

        parser = ThinkingParser(proxy, template)
    else:
        raise ValueError(
            "Qwen-VL MVP supports general/thinking chat parsers, got "
            f"{template.parser_type!r}"
        )
    # Do not truncate here.  Length rejection happens after image-pad expansion.
    input_ids, loss_mask = parser.parse(
        messages,
        max_length=1 << 30,
        train_only_last_turn=train_only_last_turn,
    )
    if proxy.last_rendered is None:
        raise ValueError("Qwen processor did not render the conversation")
    return (
        proxy.last_rendered,
        _flatten_ids(input_ids, field="input_ids"),
        _flatten_ids(loss_mask, field="loss_mask"),
    )


def _process_images(
    processor: Any,
    rendered: str,
    images: list[Any],
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[list[list[int]], list[int], int]:
    kwargs: dict[str, Any] = {
        "text": [rendered],
        "images": images,
        "padding": True,
        "return_tensors": "pt",
        # Matches SGLang's Qwen3-VL/Qwen3.5 processor call. It is a no-op for
        # image-only rows but keeps the invocation contract identical.
        "do_sample_frames": False,
    }
    if min_pixels is not None:
        kwargs["images_kwargs"] = {
            "size": {
                "shortest_edge": int(min_pixels),
                "longest_edge": int(max_pixels),
            }
        }
    output = processor(**kwargs)
    grid = _normalize_grid(
        _processor_value(output, "image_grid_thw"), expected_images=len(images)
    )
    merge_size = _spatial_merge_size(processor)
    counts = [row[0] * row[1] * row[2] // (merge_size**2) for row in grid]
    if any(count <= 0 for count in counts):
        raise ValueError(f"processor produced invalid image token counts: {counts!r}")
    if _processor_value(output, "input_ids") is None:
        raise ValueError("processor omitted input_ids")
    return grid, counts, sum(counts)


def _filtered_dataset_row(
    record_id: str, *, reason: str, detail: str
) -> dict[str, Any]:
    return {
        "reject_reason": reason,
        "reject_detail": detail,
        "task_id": record_id,
        "input_ids": [],
        "loss_mask": [],
        "server_input_ids": [],
        "image_sources": [],
        "num_tokens": 0,
    }


def _prepare_dataset_row(
    example: Mapping[str, Any],
    *,
    processor: Any,
    tokenizer: Any,
    dataset_dir: str,
    chat_template: str,
    train_only_last_turn: bool,
    min_pixels: int | None,
    max_pixels: int | None,
    max_length: int,
    min_loss_tokens: int,
) -> dict[str, Any]:
    record = json.loads(str(example["record_json"]))
    source_label = f"line {int(example['line_number'])}"
    raw_id = record.get("id")
    record_id = str(raw_id) if raw_id is not None else source_label
    try:
        if (
            not isinstance(raw_id, (str, int))
            or isinstance(raw_id, bool)
            or not record_id
        ):
            raise ValueError("record id must be a non-empty string or integer")
        if record.get("images"):
            raise ValueError(
                "top-level images are ambiguous; place ordered image_url parts "
                "inside user message content"
            )
        messages, image_sources, images, image_failure = _normalize_messages(
            record.get("conversations"),
            dataset_dir=Path(dataset_dir),
        )
        if image_failure is not None:
            reason, detail = image_failure
            return _filtered_dataset_row(
                record_id, reason=reason, detail=detail
            )
        rendered, base_ids, base_mask = _parse_base_tokens(
            processor,
            tokenizer,
            messages,
            chat_template=chat_template,
            train_only_last_turn=train_only_last_turn,
        )
        if images:
            _grid, counts, processor_image_tokens = _process_images(
                processor,
                rendered,
                images,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image_token_id = _image_token_id(processor, tokenizer)
            final_ids, final_mask = _expand_image_tokens(
                base_ids,
                base_mask,
                image_token_id=image_token_id,
                token_counts=counts,
            )
            if final_ids.count(image_token_id) != processor_image_tokens:
                raise ValueError(
                    "locally expanded image token count does not match processor"
                )
        else:
            final_ids, final_mask = base_ids, base_mask
    except ValueError as exc:
        raise ValueError(
            f"Qwen-VL preprocessing failed for {record_id!r} at {source_label}: {exc}"
        ) from exc

    if not final_ids:
        return _filtered_dataset_row(
            record_id,
            reason="empty_prompt",
            detail="prompt token sequence is empty",
        )
    if len(final_ids) > max_length:
        return _filtered_dataset_row(
            record_id,
            reason="over_max_length",
            detail=f"expanded length {len(final_ids)} exceeds max length {max_length}",
        )
    loss_tokens = sum(final_mask)
    if loss_tokens < min_loss_tokens:
        return _filtered_dataset_row(
            record_id,
            reason="too_few_loss_tokens",
            detail=f"loss tokens {loss_tokens} < required {min_loss_tokens}",
        )
    return {
        "reject_reason": "",
        "reject_detail": "",
        "task_id": record_id,
        "input_ids": final_ids,
        "loss_mask": final_mask,
        "server_input_ids": base_ids,
        "image_sources": image_sources,
        "num_tokens": len(final_ids),
    }

class Qwen3VLServerInputAdapter:
    """Server-capture input adapter for Qwen3-VL/Qwen3.5 images."""

    def load_input_tools(self, config: Config) -> Any:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            config.model.target_model_path,
            cache_dir=config.model.cache_dir,
            trust_remote_code=config.model.trust_remote_code,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("target AutoProcessor does not expose a tokenizer")
        if config.model.tokenizer_pad_token_id is not None:
            tokenizer.pad_token_id = config.model.tokenizer_pad_token_id
        elif tokenizer.pad_token_id is None:
            fallback_id = tokenizer.eos_token_id
            if isinstance(fallback_id, (list, tuple)):
                fallback_id = fallback_id[0] if fallback_id else None
            if fallback_id is None:
                fallback_id = tokenizer.unk_token_id
            if fallback_id is None:
                raise ValueError(
                    "target tokenizer has no pad, EOS, or unknown token ID; set "
                    "model.tokenizer_pad_token_id explicitly"
                )
            tokenizer.pad_token_id = fallback_id
        return processor

    def prepare_prompts(
        self, config: Config, input_tools: Any, *, draft_config: Any
    ) -> list[dict[str, Any]]:
        if config.data.prompts_path:
            raise ValueError(
                "Qwen-VL online MVP requires data.train_data_path with "
                "id + conversations; pre-tokenized prompts cannot preserve images"
            )
        source_path = config.data.train_data_path
        if not source_path:
            raise ValueError("Qwen-VL prompt preparation requires data.train_data_path")
        processor = input_tools
        tokenizer = processor.tokenizer
        from specforge.algorithms.model_providers import dflash_min_loss_tokens

        min_loss_tokens = dflash_min_loss_tokens(config, draft_config)
        deployment = getattr(config, "deployment", None)
        disaggregated = getattr(deployment, "disaggregated", None)
        if getattr(disaggregated, "managed_local", None) is None:
            logger.warning(
                "Qwen-VL uses an external SGLang server: use the same "
                "--mm-process-config pixel bounds as SpecForge; processor "
                "parity is not checked at runtime"
            )
        dataset_dir = Path(source_path).resolve().parent
        limit = (
            None
            if config.data.max_prompts in (None, 0)
            else config.data.max_prompts
        )

        try:
            from datasets import (
                Dataset,
                Features,
                Sequence as HFSequence,
                Value,
            )
        except ImportError as exc:  # pragma: no cover - production dependency
            raise ImportError("Qwen-VL prompt preparation requires datasets") from exc

        raw_rows = list(_iter_records(source_path))
        if not raw_rows:
            raise ValueError(f"Qwen-VL dataset {source_path!r} is empty")
        num_proc = min(config.data.build_dataset_num_proc, len(raw_rows))
        cache_path = _map_cache_path(
            config,
            processor,
            tokenizer,
            source_path=source_path,
            raw_rows=raw_rows,
            dataset_dir=dataset_dir,
            min_loss_tokens=min_loss_tokens,
        )
        rejected: Counter[str] = Counter()
        examples: list[str] = []
        raw_dataset = Dataset.from_dict(
            {
                "line_number": [line_number for line_number, _ in raw_rows],
                # Keep heterogeneous OpenAI content parts opaque to Arrow.
                "record_json": [json.dumps(record) for _, record in raw_rows],
            }
        )
        output_features = Features(
            {
                "reject_reason": Value("string"),
                "reject_detail": Value("string"),
                "task_id": Value("string"),
                "input_ids": HFSequence(Value("int64")),
                "loss_mask": HFSequence(Value("int64")),
                "server_input_ids": HFSequence(Value("int64")),
                "image_sources": HFSequence(Value("string")),
                "num_tokens": Value("int64"),
            }
        )
        if num_proc > 1:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if cache_path is not None:
            logger.info("Qwen-VL map cache: %s", cache_path)
        processed_dataset = raw_dataset.map(
            _prepare_dataset_row,
            fn_kwargs={
                "processor": processor,
                "tokenizer": tokenizer,
                "dataset_dir": str(dataset_dir),
                "chat_template": config.data.chat_template,
                "train_only_last_turn": config.data.train_only_last_turn,
                "min_pixels": config.data.min_pixels,
                "max_pixels": config.data.max_pixels,
                "max_length": config.data.max_length,
                "min_loss_tokens": min_loss_tokens,
            },
            num_proc=num_proc if num_proc > 1 else None,
            remove_columns=raw_dataset.column_names,
            features=output_features,
            load_from_cache_file=cache_path is not None,
            cache_file_name=str(cache_path) if cache_path is not None else None,
            desc="Preparing Qwen-VL prompts",
        )
        accepted: list[dict[str, Any]] = []
        accepted_ids: set[str] = set()
        for row in processed_dataset:
            reason = str(row["reject_reason"])
            if reason:
                rejected[reason] += 1
                if len(examples) < _MAX_REJECT_EXAMPLES:
                    examples.append(f"{row['task_id']}: {row['reject_detail']}")
                continue
            record_id = str(row["task_id"])
            if record_id in accepted_ids:
                raise ValueError(f"duplicate Qwen-VL record id {record_id!r}")
            accepted_ids.add(record_id)
            if limit is not None and len(accepted) >= limit:
                continue
            accepted.append(
                {
                    "task_id": record_id,
                    "source_id": source_path,
                    "max_length": config.data.max_length,
                    "chat_template": config.data.chat_template,
                    "payload": {
                        "input_ids": [int(token) for token in row["input_ids"]],
                        "loss_mask": [int(token) for token in row["loss_mask"]],
                        "server_input_ids": [
                            int(token) for token in row["server_input_ids"]
                        ],
                        "image_sources": [
                            str(source) for source in row["image_sources"]
                        ],
                    },
                    "metadata": {
                        "num_tokens": int(row["num_tokens"]),
                        "tokenizer_version": str(config.model.target_model_path),
                    },
                }
            )

        if not accepted_ids:
            raise ValueError(
                "Qwen-VL prompt preparation rejected every sample; "
                f"reasons={dict(sorted(rejected.items()))}, examples={examples}"
            )

        logger.info(
            "Qwen-VL prompt preparation accepted=%d rejected=%d "
            "reasons=%s examples=%s",
            len(accepted),
            sum(rejected.values()),
            dict(sorted(rejected.items())),
            examples,
        )
        return accepted

    def build_request_inputs(
        self, tasks: Sequence[PromptTask]
    ) -> dict[str, Any]:
        input_ids: list[list[int]] = []
        image_data: list[list[str] | None] = []
        for task in tasks:
            server_ids = task.payload.get("server_input_ids")
            if server_ids is None:
                raise ValueError(
                    f"task {task.task_id}: Qwen-VL payload omits server_input_ids"
                )
            input_ids.append(list(server_ids))
            sources = list(task.payload.get("image_sources") or [])
            image_data.append(sources or None)
        return {"input_ids": input_ids, "image_data": image_data}


__all__ = ["Qwen3VLServerInputAdapter"]
