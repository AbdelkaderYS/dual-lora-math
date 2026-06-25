#!/bin/bash

MODEL_NAME_OR_PATH="/mnt/gs21/scratch/djagbapr/lora_outputs/merged-dual-lora-llama33-math-dual-lora-mp"
DATA="math"
SPLIT="test"
PROMPT_TYPE="mp"
NUM_TEST_SAMPLE=-1

cd /mnt/gs21/scratch/djagbapr/dual-lora-math/unified/Math
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false \
python -m infer.inference \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA} --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_func_call 1 \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 --temperature 0 --n_sampling 1 \
    --top_p 0.95 --start 0 --end -1
