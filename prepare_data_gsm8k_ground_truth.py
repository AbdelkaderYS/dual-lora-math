"""
prepare_data_gsm8k.py
---------------------
Reads train.json (JSONL) from the meta-prompting GSM8K dataset,
applies the Llama-3.3-Instruct chat template to each example,
splits 90/10 into train/val, and saves both splits to disk.

Run once before training:
    python prepare_data_gsm8k_ground_truth.py

Output:
    /mnt/scratch/djagbapr/lora_data_gsm8k/train.jsonl
    /mnt/scratch/djagbapr/lora_data_gsm8k/val.jsonl
"""

import json
import os
import random
from pathlib import Path
from transformers import AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "meta-llama/Llama-3.3-70B-Instruct"
HF_TOKEN     = os.environ.get("HF_TOKEN", None)
INPUT_FILE   = "/mnt/home/djagbapr/Saley/ABIA_EUNICE/meta-prompting/Math/data/gsm8k/train.json"
OUTPUT_DIR   = "/mnt/scratch/djagbapr/lora_data_gsm8k"
VAL_FRACTION = 0.10
SEED         = 42
MAX_SEQ_LEN  = 2048          # examples longer than this are discarded (not truncated)

# ── Load tokenizer ─────────────────────────────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token   # Llama-3.3 has no pad token by default

# ── Load raw data ──────────────────────────────────────────────────────────────
print(f"Reading {INPUT_FILE} ...")
examples = []
with open(INPUT_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            examples.append(json.loads(line))

print(f"Total examples loaded: {len(examples)}")

# ── Build prompt+completion strings using the chat template ────────────────────
def build_text(example: dict) -> str:
    """
    Format one GSM8K example as a full Llama-3.3 instruct conversation.
    The user message contains the problem; the assistant message is the solution.
    The EOS token at the end tells the model where generation should stop.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following math problem. "
                "Show your reasoning step by step, then give the final answer.\n\n"
                f"Problem: {example['question']}"
            ),
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]
    # apply_chat_template with add_generation_prompt=False includes the full
    # assistant turn (we want the model to learn to produce the solution).
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text

# ── Filter by length ───────────────────────────────────────────────────────────
print("Building and filtering examples ...")
processed = []
skipped   = 0

for ex in examples:
    text = build_text(ex)
    token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if token_count > MAX_SEQ_LEN:
        skipped += 1
        continue
    processed.append({
        "text":       text,
        "idx":        ex.get("idx", -1),
        "token_count": token_count,
    })

print(f"Examples kept : {len(processed)}")
print(f"Examples skipped (>{MAX_SEQ_LEN} tokens): {skipped}")

# ── Token length statistics ────────────────────────────────────────────────────
lengths = [e["token_count"] for e in processed]
print(f"Token length  — min: {min(lengths)}  max: {max(lengths)}  "
      f"mean: {sum(lengths)/len(lengths):.0f}")

# ── Train / val split ──────────────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(processed)
n_val   = int(len(processed) * VAL_FRACTION)
n_train = len(processed) - n_val
train_data = processed[n_val:]    # first n_train after shuffle
val_data   = processed[:n_val]    # last n_val

print(f"Train: {len(train_data)}  |  Val: {len(val_data)}")

# ── Save to disk ───────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
val_path   = os.path.join(OUTPUT_DIR, "val.jsonl")

for path, split in [(train_path, train_data), (val_path, val_data)]:
    with open(path, "w") as f:
        for item in split:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(split)} examples → {path}")

print("\nData preparation complete.")
print(f"  Train: {train_path}")
print(f"  Val  : {val_path}")