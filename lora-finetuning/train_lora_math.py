"""
train_lora.py
-------------
Fine-tunes Llama-3.3-70B-Instruct on the MATH dataset using LoRA (no quantization).
Uses device_map="auto" to spread the 140 GB model across both H200 GPUs in a
single process — no DDP, no SLURM, runs directly from a Jupyter terminal.

Prerequisites:
    python prepare_data.py          # run once first

Run:
    CUDA_VISIBLE_DEVICES=0,1 python train_lora.py 2>&1 | tee training.log

Outputs (adapters only, ~500 MB):
    /mnt/scratch/djagbapr/lora_outputs/llama33-math-lora/
"""

import os
import json
import torch
from typing import Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from datasets import Dataset


# ── Custom Trainer ─────────────────────────────────────────────────────────────
class PeftTrainer(Trainer):
    """
    Overrides compute_loss to avoid the cross-device num_items_in_batch
    bug in transformers 4.46 when using device_map='auto' across GPUs.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "/mnt/scratch/djagbapr/models/Llama-3.3-70B-Instruct"
#DATA_DIR   = "/mnt/scratch/djagbapr/lora_data"
DATA_DIR = "/mnt/scratch/djagbapr/lora_data_mp"
#OUTPUT_DIR = "/mnt/scratch/djagbapr/lora_outputs/llama33-math-lora"
OUTPUT_DIR = "/mnt/scratch/djagbapr/lora_outputs/llama33-math-lora-mp"
LOG_DIR    = "/mnt/scratch/djagbapr/lora_outputs/logs"

# ── LoRA hyperparameters ───────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training hyperparameters ───────────────────────────────────────────────────
MAX_SEQ_LEN           = 4046
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8       # effective batch = 1 × 8 = 8
NUM_EPOCHS            = 2
LEARNING_RATE         = 2e-4
WARMUP_RATIO          = 0.03
LR_SCHEDULER          = "cosine"
SAVE_STRATEGY         = "epoch"
EVAL_STRATEGY         = "epoch"
LOGGING_STEPS         = 10
SEED                  = 42
MAX_GRAD_NORM         = 1.0


# ── GPU info ───────────────────────────────────────────────────────────────────
print(f"Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
print(f"Torch sees {torch.cuda.device_count()} GPU(s)")
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name}  ({mem:.1f} GB)")


# ── Tokenizer ──────────────────────────────────────────────────────────────────
print("\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"


# ── Model ──────────────────────────────────────────────────────────────────────
print("Loading model with device_map='auto' (splits across both GPUs) ...")
print("This will take ~3-5 minutes ...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

# Must be set before applying LoRA and before training
model.config.use_cache = False
model.enable_input_require_grads()


# ── LoRA ───────────────────────────────────────────────────────────────────────
print("Applying LoRA adapters ...")
lora_config = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = LORA_TARGET_MODULES,
    task_type      = TaskType.CAUSAL_LM,
    bias           = "none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ── Load datasets ──────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)

print("\nLoading prepared data ...")
train_dataset = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
val_dataset   = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
print(f"Train: {len(train_dataset)} examples")
print(f"Val  : {len(val_dataset)} examples")


# ── Tokenize ───────────────────────────────────────────────────────────────────
def tokenize(example: Dict) -> Dict:
    """
    Tokenize the full formatted text from prepare_data.py.
    Masks prompt tokens with -100 so loss is computed on assistant tokens only.
    """
    tokenized = tokenizer(
        example["text"],
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]
    labels    = input_ids.copy()

    # Find the last assistant header and mask everything before it
    assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
    header_len = len(header_ids)

    prompt_end = None
    for i in range(len(input_ids) - header_len, -1, -1):
        if input_ids[i : i + header_len] == header_ids:
            prompt_end = i + header_len
            break

    if prompt_end is not None:
        labels[:prompt_end] = [-100] * prompt_end
    else:
        # Fallback: mask first half (should not happen with correct data)
        labels[:len(labels) // 2] = [-100] * (len(labels) // 2)

    tokenized["labels"] = labels
    return tokenized


print("Tokenizing train set ...")
train_tokenized = train_dataset.map(
    tokenize,
    remove_columns=train_dataset.column_names,
    num_proc=4,
    desc="Tokenizing train",
)

print("Tokenizing val set ...")
val_tokenized = val_dataset.map(
    tokenize,
    remove_columns=val_dataset.column_names,
    num_proc=4,
    desc="Tokenizing val",
)


# ── Data collator ──────────────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)


# ── Training arguments ─────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size  = PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps = GRADIENT_ACCUMULATION,
    learning_rate               = LEARNING_RATE,
    lr_scheduler_type           = LR_SCHEDULER,
    warmup_ratio                = WARMUP_RATIO,
    bf16                        = True,
    fp16                        = False,
    gradient_checkpointing      = True,
    max_grad_norm               = MAX_GRAD_NORM,
    eval_strategy               = EVAL_STRATEGY,
    save_strategy               = SAVE_STRATEGY,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    logging_dir                 = LOG_DIR,
    logging_steps               = LOGGING_STEPS,
    save_total_limit            = 2,
    seed                        = SEED,
    report_to                   = "none",
    ddp_find_unused_parameters  = False,
    dataloader_num_workers      = 0,
    remove_unused_columns       = False,
)


# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = PeftTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = train_tokenized,
    eval_dataset     = val_tokenized,
    processing_class = tokenizer,
    data_collator    = data_collator,
)


# ── Train ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Starting LoRA fine-tuning ...")
print(f"  Train examples : {len(train_tokenized)}")
print(f"  Val examples   : {len(val_tokenized)}")
print(f"  Epochs         : {NUM_EPOCHS}")
print(f"  Effective batch: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  Learning rate  : {LEARNING_RATE}")
print(f"  Output dir     : {OUTPUT_DIR}")
print("=" * 60 + "\n")

trainer.train()


# ── Save final adapter ─────────────────────────────────────────────────────────
final_adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
model.save_pretrained(final_adapter_path)
tokenizer.save_pretrained(final_adapter_path)

print(f"\nTraining complete.")
print(f"Final adapter saved to: {final_adapter_path}")
print(f"Run merge_lora.py to merge adapter into the base model.")