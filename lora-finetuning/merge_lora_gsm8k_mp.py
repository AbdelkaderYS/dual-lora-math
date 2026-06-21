#!/usr/bin/env python3
"""
merge_lora_gsm8k_mp.py
────────────────────────────────────────────────────────────────────────
Merges the LoRA adapter with the base model for GSM8K meta-prompting.

Usage:
    python merge_lora_gsm8k_mp.py

Loads the base Llama-3.3-70B-Instruct model, applies the trained LoRA
adapter, merges them into a single model, and saves the result.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Base model
BASE_MODEL = "/mnt/scratch/djagbapr/models/Llama-3.3-70B-Instruct"

# LoRA adapter (trained model)
ADAPTER_DIR = "/mnt/scratch/djagbapr/lora_outputs/llama33-gsm8k-lora-mp/final_adapter"

# Output directory for merged model
OUTPUT_DIR = "/mnt/scratch/djagbapr/lora_outputs/llama33-gsm8k-lora-mp-merged"

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

print("="*70)
print("LoRA Adapter Merge: GSM8K Meta-Prompting")
print("="*70)

print(f"\nBase model: {BASE_MODEL}")
print(f"LoRA adapter: {ADAPTER_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Check paths exist
if not os.path.exists(BASE_MODEL):
    print(f"\nERROR: Base model not found at {BASE_MODEL}")
    exit(1)

if not os.path.exists(ADAPTER_DIR):
    print(f"\nERROR: LoRA adapter not found at {ADAPTER_DIR}")
    print("Make sure training completed successfully.")
    exit(1)

# ══════════════════════════════════════════════════════════════════════════
# LOAD BASE MODEL
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Loading base model...")
print("="*70)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Base model loaded: {base_model.__class__.__name__}")
print(f"Model dtype: {base_model.dtype}")

# ══════════════════════════════════════════════════════════════════════════
# LOAD AND MERGE ADAPTER
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Loading LoRA adapter...")
print("="*70)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("LoRA adapter loaded.")
print("\nMerging adapter with base model...")

merged_model = model.merge_and_unload()

print("Merge complete!")

# ══════════════════════════════════════════════════════════════════════════
# SAVE MERGED MODEL
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Saving merged model...")
print("="*70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

merged_model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=True,
    max_shard_size="5GB"
)

print(f"Merged model saved to: {OUTPUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# SAVE TOKENIZER
# ══════════════════════════════════════════════════════════════════════════

print("\nSaving tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Tokenizer saved to: {OUTPUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Merge complete!")
print("="*70)

print(f"\nMerged model location: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Evaluate on GSM8K test set:")
print("     python run.py --model llama33-gsm8k-lora-mp-merged --prompt mp --split test")
print("  2. Cross-evaluate on MATH:")
print("     python run.py --model llama33-gsm8k-lora-mp-merged --prompt mp --split test --dataset math")

# Print model info
import subprocess
result = subprocess.run(['du', '-sh', OUTPUT_DIR], capture_output=True, text=True)
if result.returncode == 0:
    size = result.stdout.split()[0]
    print(f"\nMerged model size: {size}")

print("\nDone.")
