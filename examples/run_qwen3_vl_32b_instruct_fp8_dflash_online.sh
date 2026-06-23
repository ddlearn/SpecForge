#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:?"Error: TARGET_MODEL_PATH is required"}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flex_attention}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-vl-32b-instruct-fp8-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/allava4v_train_frac_1_12.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-vl-32b-instruct-fp8-dflash \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 262144 \
    --chat-template qwen3-instruct \
    --attention-backend $ATTENTION_BACKEND \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 50000 \
    --report-to tensorboard \
    --target-model-backend sglang \
    --block-size 16 \
    --num-anchors 512 \
    --embedding-key model.language_model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --sglang-mem-fraction-static 0.5 \
    --sglang-cuda-graph-max-bs 1 \
    --is-vlm \
    --min-pixels 65536 \
    --max-pixels 16777216 \
    --trust-remote-code \
    --dist-timeout 120