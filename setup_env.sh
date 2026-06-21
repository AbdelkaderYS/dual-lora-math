#!/bin/bash
set -e

ENV_NAME="dual_lora_mp"
ENV_FILE="/mnt/scratch/djagbapr/dual-lora-mp-project/setup_env.sh"

if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda env: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.11

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    pip install transformers==4.44.2 datasets==2.21.0
    pip install vllm==0.6.1
    pip install peft==0.12.0 bitsandbytes==0.44.0
    pip install accelerate==0.34.2 sentencepiece protobuf
    pip install einops

    echo "Environment $ENV_NAME ready"
else
    echo "Environment $ENV_NAME already exists"
fi
