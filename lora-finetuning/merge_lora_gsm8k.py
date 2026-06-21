"""
merge_lora_gsm8k.py
-------------------
Merges the trained LoRA adapter back into the Llama-3.3-70B-Instruct base weights,
producing a standalone model you can run directly with vLLM (no PEFT needed).

Run AFTER training is complete:
    python merge_lora_gsm8k.py

Output (~140 GB):
    /mnt/scratch/djagbapr/lora_outputs/llama33-gsm8k-lora-merged/
"""

import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_MODEL_PATH  = "/mnt/scratch/djagbapr/models/Llama-3.3-70B-Instruct"
ADAPTER_PATH     = "/mnt/scratch/djagbapr/lora_outputs/llama33-gsm8k-lora/final_adapter"
MERGED_OUTPUT    = "/mnt/scratch/djagbapr/lora_outputs/llama33-gsm8k-lora-merged"

# ── Disk space check ───────────────────────────────────────────────────────────
print("Checking disk space (need ~140 GB free) ...")
os.system("df -h /mnt/scratch/djagbapr")

# ── Load base model ────────────────────────────────────────────────────────────
print("\nLoading base model ...")
print("This will take ~3–5 minutes ...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ── Load and merge adapter ─────────────────────────────────────────────────────
print(f"Loading adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging adapter weights into base model ...")
model = model.merge_and_unload()
print("Merge complete.")

# ── Save merged model ──────────────────────────────────────────────────────────
os.makedirs(MERGED_OUTPUT, exist_ok=True)
print(f"\nSaving merged model to: {MERGED_OUTPUT}")
print("This will take ~10–15 minutes (writing ~140 GB) ...")

model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)

print("Saving tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.save_pretrained(MERGED_OUTPUT)

print(f"\nMerge complete. Merged model at: {MERGED_OUTPUT}")
print("You can now run inference with vLLM pointing to this directory.")