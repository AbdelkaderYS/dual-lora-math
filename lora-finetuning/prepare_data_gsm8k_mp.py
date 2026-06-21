#!/usr/bin/env python3
"""
prepare_data_gsm8k_mp.py
────────────────────────────────────────────────────────────────────────
Prepares GSM8K meta-prompting outputs for LoRA fine-tuning.

Usage:
    python prepare_data_gsm8k_mp.py

Reads the baseline model's correct meta-prompting outputs on GSM8K train set,
formats them for instruction fine-tuning, and saves train/val splits.
"""

import json
import os
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
# baseline model's meta-prompting outputs on GSM8K training set
INPUT_FILE = "outputs/models/Llama-3.3-70B-Instruct/gsm8k/train_mp_-1_seed0_t0.0_s0_e7473_05-XX_XX-XX_Unknown.jsonl"

# Output directory
OUTPUT_DIR = "/mnt/scratch/djagbapr/lora_data_gsm8k_mp"

# Train/val split ratio
TRAIN_RATIO = 0.9

# Max sequence length 
MAX_SEQ_LEN = 2048

# ── Helper functions ───────────────────────────────────────────────────────
def load_jsonl(path):
    """Load JSONL file into list of dicts."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def is_correct(entry):
    """Check if the model's prediction matches ground truth."""
    pred = entry.get('pred', [])
    gt = str(entry.get('gt', '')).strip()
    if not pred:
        return False
    return str(pred[-1]).strip() == gt

def format_for_training(entry):
    """
    Format a single GSM8K example for instruction fine-tuning.
    
    Returns dict with 'prompt' and 'completion' fields in Llama-3 chat format.
    """
    question = entry.get('question', entry.get('problem', '')).strip()
    
    # Use the model's meta-prompting output (full reasoning trace)
    code = entry.get('code', [])
    if isinstance(code, list) and code:
        solution = code[0]
    else:
        solution = str(code) if code else ""
    
    if not solution:
        return None
    
    # Llama-3 chat format
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Solve the following math problem step by step.\n\n"
        f"Problem: {question}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    completion = solution + "<|eot_id|>"
    
    return {
        "prompt": prompt,
        "completion": completion,
        "idx": entry.get('idx', -1),
        "gt": entry.get('gt', ''),
    }

def count_tokens_approx(text):
    """Rough token count (4 chars per token)."""
    return len(text) // 4

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("GSM8K Meta-Prompting Output Data Preparation for LoRA Fine-Tuning")
    print("="*70)
    
    # Check input file
    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        print("\nYou need to run the baseline model on GSM8K train set first:")
        print("  python run.py --model Llama-3.3-70B-Instruct --prompt mp --split train")
        print("\nThen update INPUT_FILE in this script with the actual filename.")
        return
    
    print(f"\nInput file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train/val split: {TRAIN_RATIO:.0%} / {100*(1-TRAIN_RATIO):.0%}")
    print(f"Max sequence length: {MAX_SEQ_LEN} tokens")
    
    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    data = load_jsonl(INPUT_FILE)
    print(f"  Loaded {len(data)} examples")
    
    # Filter to correct predictions only
    print("\nFiltering to correct predictions only...")
    correct_data = [e for e in data if is_correct(e)]
    print(f"  Kept {len(correct_data)} correct predictions ({100*len(correct_data)/len(data):.1f}%)")
    print(f"  Discarded {len(data) - len(correct_data)} incorrect predictions")
    
    # Format for training
    print("\nFormatting examples...")
    formatted = []
    skipped_empty = 0
    skipped_long = 0
    
    for entry in correct_data:
        formatted_entry = format_for_training(entry)
        
        if formatted_entry is None:
            skipped_empty += 1
            continue
        
        # Check sequence length
        total_text = formatted_entry['prompt'] + formatted_entry['completion']
        approx_tokens = count_tokens_approx(total_text)
        
        if approx_tokens > MAX_SEQ_LEN:
            skipped_long += 1
            continue
        
        formatted.append(formatted_entry)
    
    print(f"  Formatted {len(formatted)} examples")
    print(f"  Skipped {skipped_empty} empty outputs")
    print(f"  Skipped {skipped_long} examples exceeding {MAX_SEQ_LEN} tokens")
    
    if len(formatted) == 0:
        print("\nERROR: No valid training examples after filtering!")
        return
    
    # Split train/val
    split_idx = int(len(formatted) * TRAIN_RATIO)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]
    
    print(f"\nTrain/val split:")
    print(f"  Training: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save train set
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"\nSaved training data → {train_path}")
    
    # Save validation set
    val_path = os.path.join(OUTPUT_DIR, "val.jsonl")
    with open(val_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved validation data → {val_path}")
    
    # Print sample
    print("\n" + "="*70)
    print("Sample training example:")
    print("="*70)
    sample = train_data[0]
    print("PROMPT:")
    print(sample['prompt'][:500] + "..." if len(sample['prompt']) > 500 else sample['prompt'])
    print("\nCOMPLETION:")
    print(sample['completion'][:500] + "..." if len(sample['completion']) > 500 else sample['completion'])
    
    print("\n" + "="*70)
    print("Data preparation complete!")
    print("="*70)
    print(f"\nNext step: Run the training script:")
    print(f"  sbatch train_lora_gsm8k_mp.sb")

if __name__ == "__main__":
    main()
