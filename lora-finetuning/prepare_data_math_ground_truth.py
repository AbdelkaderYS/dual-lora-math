"""
prepare_data_mp.py
------------------
Builds a fine-tuning dataset from the model's OWN correct meta-prompting
reasoning traces. Only examples where the model predicted the right answer
are included — this reinforces exactly the behavior the inference pipeline
expects, rather than conflicting with it.

Source:
    train_mp_-1_seed0_t0.0_s0_e7500_05-04_16-14_Unknown.jsonl
    (7500 training problems run through the meta-prompting pipeline)

Output:
    /mnt/scratch/djagbapr/lora_data_mp/train.jsonl   (~6100 examples)
    /mnt/scratch/djagbapr/lora_data_mp/val.jsonl     (~680 examples)

Run:
    python prepare_data_mp.py
"""

import json
import os
import random
from transformers import AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/mnt/scratch/djagbapr/models/Llama-3.3-70B-Instruct"
INPUT_JSONL  = (
    "/mnt/home/djagbapr/Saley/ABIA_EUNICE/meta-prompting/Math/outputs/"
    "models/Llama-3.3-70B-Instruct/math/"
    "train_mp_-1_seed0_t0.0_s0_e7500_05-04_16-14_Unknown.jsonl"
)
OUTPUT_DIR   = "/mnt/scratch/djagbapr/lora_data_mp"
VAL_FRACTION = 0.10
SEED         = 42
MAX_SEQ_LEN  = 4096

# ── Tokenizer ──────────────────────────────────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ── Load and filter correct predictions only ───────────────────────────────────
print(f"Reading {INPUT_JSONL} ...")
correct_examples = []
total = 0

with open(INPUT_JSONL) as f:
    for line in f:
        d = json.loads(line.strip())
        total += 1

        pred = d.get("pred", [])
        gt   = str(d.get("gt", "")).strip()
        if not pred:
            continue
        if str(pred[0]).strip() != gt:
            continue

        # Extract the reasoning trace
        code = d["code"]
        reasoning = code[0] if isinstance(code, list) else code
        if not reasoning or not reasoning.strip():
            continue

        correct_examples.append({
            "prompt":    d["prompt"],       # meta-prompting instruction + problem
            "reasoning": reasoning.strip(), # model's correct step-by-step trace
            "gt":        gt,
            "level":     d.get("level", ""),
            "type":      d.get("type", ""),
            "idx":       d.get("idx", -1),
        })

print(f"Total problems  : {total}")
print(f"Correct examples: {len(correct_examples)}")
print(f"Accuracy        : {len(correct_examples)/total*100:.1f}%")

# ── Build formatted text using Llama-3.3 chat template ────────────────────────
print("\nFormatting with chat template and filtering by length ...")

processed = []
skipped   = 0

for ex in correct_examples:
    messages = [
        {"role": "user",      "content": ex["prompt"]},
        {"role": "assistant", "content": ex["reasoning"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])

    if token_count > MAX_SEQ_LEN:
        skipped += 1
        continue

    processed.append({
        "text":        text,
        "token_count": token_count,
        "level":       ex["level"],
        "type":        ex["type"],
        "idx":         ex["idx"],
    })

print(f"Examples kept   : {len(processed)}")
print(f"Skipped (>{MAX_SEQ_LEN} tokens): {skipped}")

lengths = [e["token_count"] for e in processed]
print(f"Token length — min: {min(lengths)}  max: {max(lengths)}  "
      f"mean: {sum(lengths)/len(lengths):.0f}")

# ── Train / val split ──────────────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(processed)

n_val      = int(len(processed) * VAL_FRACTION)
val_data   = processed[:n_val]
train_data = processed[n_val:]

print(f"\nTrain: {len(train_data)}  |  Val: {len(val_data)}")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname, split in [("train.jsonl", train_data), ("val.jsonl", val_data)]:
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w") as f:
        for item in split:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(split)} examples → {path}")

print("\nData preparation complete.")
print(f"  Train : {os.path.join(OUTPUT_DIR, 'train.jsonl')}")
print(f"  Val   : {os.path.join(OUTPUT_DIR, 'val.jsonl')}")
print(f"\nNext step: update DATA_DIR in train_lora.py to:")
print(f"  DATA_DIR = \"{OUTPUT_DIR}\"")
print(f"Then run: CUDA_VISIBLE_DEVICES=0,1 python train_lora.py 2>&1 | tee training_mp.log")