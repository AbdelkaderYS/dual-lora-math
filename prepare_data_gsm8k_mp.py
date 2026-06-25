#!/usr/bin/env python3
import json
import os
import random
from pathlib import Path
from transformers import AutoTokenizer

MODEL_PATH   = "meta-llama/Llama-3.3-70B-Instruct"
HF_TOKEN     = os.environ.get("HF_TOKEN", None)
INPUT_FILE   = (
    "/mnt/home/djagbapr/Saley/ABIA_EUNICE/meta-prompting/Math/outputs/"
    "models/Llama-3.3-70B-Instruct/gsm8k/"
    "train_mp_-1_seed0_t0.0_s0_e7473_05-XX_XX-XX_Unknown.jsonl"
)
OUTPUT_DIR   = "/mnt/gs21/scratch/djagbapr/lora_data_gsm8k_mp"
TRAIN_RATIO  = 0.9
SEED         = 42
MAX_SEQ_LEN  = 2048

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def main():
    print("="*70)
    print("GSM8K Meta-Prompting Data Preparation")
    print("="*70)

    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        print("Check the actual filename and update INPUT_FILE in this script.")
        return

    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading data from {INPUT_FILE}...")
    data = load_jsonl(INPUT_FILE)
    print(f"  Loaded {len(data)} examples")

    print("Filtering to correct predictions only...")
    correct = []
    for e in data:
        pred = e.get('pred', [])
        gt = str(e.get('gt', '')).strip()
        if not pred:
            continue
        if str(pred[-1]).strip() != gt:
            continue
        code = e.get('code', [])
        solution = code[0] if isinstance(code, list) and code else str(code) if code else ""
        if not solution.strip():
            continue
        correct.append({
            "question": e.get('question', e.get('problem', '')).strip(),
            "solution": solution.strip(),
            "gt": gt,
            "idx": e.get('idx', -1),
        })
    print(f"  Kept {len(correct)} correct ({100*len(correct)/len(data):.1f}%)")

    print("Formatting with chat template and filtering by length...")
    processed = []
    skipped_long = 0
    for ex in correct:
        messages = [
            {"role": "user", "content": f"Solve the following math problem step by step.\n\nProblem: {ex['question']}"},
            {"role": "assistant", "content": ex['solution']},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if token_count > MAX_SEQ_LEN:
            skipped_long += 1
            continue
        processed.append({"text": text, "token_count": token_count, "idx": ex["idx"]})

    print(f"  Formatted {len(processed)} examples")
    print(f"  Skipped {skipped_long} exceeding {MAX_SEQ_LEN} tokens")
    if not processed:
        print("ERROR: No valid training examples!")
        return

    random.seed(SEED)
    random.shuffle(processed)
    split_idx = int(len(processed) * TRAIN_RATIO)
    train_data = processed[:split_idx]
    val_data = processed[split_idx:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname, split in [("train.jsonl", train_data), ("val.jsonl", val_data)]:
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, 'w') as f:
            for item in split:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(split)} examples -> {path}")

if __name__ == "__main__":
    main()
