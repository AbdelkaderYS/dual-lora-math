#!/bin/bash
set -e

PROJECT_DIR="/mnt/scratch/djagbapr/dual-lora-mp-project"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate dual_lora_mp

echo "========== STEP 1: MATH data preparation =========="
python prepare_data_math.py

echo ""
echo "========== STEP 2: GSM8K trace generation =========="
python generate_traces_gsm8k.py

echo ""
echo "========== STEP 3: GSM8K data preparation =========="
python prepare_data_gsm8k.py

echo ""
echo "========== STEP 4: Dual LoRA MATH training =========="
python train_math_mp.py

echo ""
echo "========== STEP 5: Dual LoRA GSM8K training =========="
python train_gsm8k_mp.py

echo ""
echo "========== STEP 6: Dual LoRA 4-bit MATH training =========="
python train_math_mp_4bit.py

echo ""
echo "========== STEP 7: Evaluation =========="
python evaluate.py

echo ""
echo "========== ALL DONE =========="
