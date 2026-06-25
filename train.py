import argparse
import json
import os
import sys

import torch
from typing import Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from datasets import Dataset

from methods import DualLoRAConfig, apply_dual_lora, apply_dual_lora_4bit, set_model_step


METHODS = ["lora", "qlora", "dual_lora", "q_dual_lora"]
DATASETS = ["math", "gsm8k"]
DATA_TYPES = ["mp", "gt"]

MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

DATA_DIRS = {
    ("math", "mp"): "/mnt/gs21/scratch/djagbapr/lora_data_mp",
    ("math", "gt"): "/mnt/gs21/scratch/djagbapr/lora_data",
    ("gsm8k", "mp"): "/mnt/gs21/scratch/djagbapr/lora_data_gsm8k_mp",
    ("gsm8k", "gt"): "/mnt/gs21/scratch/djagbapr/lora_data_gsm8k",
}

OUTPUT_BASE = "/mnt/gs21/scratch/djagbapr/lora_outputs"
LOG_BASE = "/mnt/gs21/scratch/djagbapr/lora_outputs/logs"

MAX_SEQ_LEN = {"math": 4096, "gsm8k": 2048}
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_R1 = 16
LORA_R2 = 16
WARMUP_STEPS = 20
PER_DEVICE_BATCH_SIZE = 1
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0
SEED = 42

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class PeftTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class WarmupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            set_model_step(model, state.global_step)
        return control


def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def build_tokenize_fn(tokenizer, max_seq_len):
    def tokenize(example: Dict) -> Dict:
        tokenized = tokenizer(
            example["text"],
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        header_len = len(header_ids)
        prompt_end = None
        for i in range(len(input_ids) - header_len, -1, -1):
            if input_ids[i:i + header_len] == header_ids:
                prompt_end = i + header_len
                break
        if prompt_end is not None:
            labels[:prompt_end] = [-100] * prompt_end
        else:
            labels[:len(labels) // 2] = [-100] * (len(labels) // 2)
        tokenized["labels"] = labels
        return tokenized
    return tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-type", default="mp", choices=DATA_TYPES)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    grad_accum = 8 if n_gpus <= 1 else (4 if n_gpus == 2 else 2)
    effective_batch = PER_DEVICE_BATCH_SIZE * grad_accum * max(n_gpus, 1)

    print(f"Method: {args.method}  Dataset: {args.dataset}  Data type: {args.data_type}")
    print(f"GPUs: {n_gpus}  Grad accum: {grad_accum}  Effective batch: {effective_batch}")

    data_dir = DATA_DIRS[(args.dataset, args.data_type)]
    method_tag = args.method.replace("_", "-")
    dataset_tag = args.dataset
    data_tag = "mp" if args.data_type == "mp" else "gt"
    output_dir = args.output_dir or os.path.join(OUTPUT_BASE, f"llama33-{dataset_tag}-{method_tag}-{data_tag}")
    log_dir = os.path.join(LOG_BASE, f"{dataset_tag}_{args.method}_{args.data_type}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_seq_len = MAX_SEQ_LEN[args.dataset]

    if args.method in ("qlora", "q_dual_lora"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",
            token=HF_TOKEN,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            token=HF_TOKEN,
        )

    model.config.use_cache = False
    model.enable_input_require_grads()

    callbacks = []
    if args.method in ("lora", "qlora"):
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        dual_config = DualLoRAConfig(
            r1=LORA_R1, r2=LORA_R2,
            lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES, warmup_steps=WARMUP_STEPS,
        )
        if args.method == "dual_lora":
            model = apply_dual_lora(model, dual_config)
        else:
            model = apply_dual_lora_4bit(model, dual_config)
            model.is_loaded_in_4bit = False
        callbacks = [WarmupCallback()]

    train_dataset = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_dataset = load_jsonl(os.path.join(data_dir, "val.jsonl"))
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    tokenize_fn = build_tokenize_fn(tokenizer, max_seq_len)
    train_tok = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names, num_proc=4)
    val_tok = val_dataset.map(tokenize_fn, remove_columns=val_dataset.column_names, num_proc=4)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=MAX_GRAD_NORM,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=log_dir,
        logging_steps=10,
        save_total_limit=2,
        seed=SEED,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = PeftTrainer(
        model=model, args=training_args,
        train_dataset=train_tok, eval_dataset=val_tok,
        processing_class=tokenizer, data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()

    adapter_path = os.path.join(output_dir, "final_adapter")
    os.makedirs(adapter_path, exist_ok=True)

    if args.method in ("lora", "qlora"):
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        save_cfg = {
            "method": args.method,
            "r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "dataset": args.dataset, "data_type": args.data_type,
        }
    else:
        adapter_state = {k: v.cpu() for k, v in model.state_dict().items()
                         if "dual_lora" in k or "lora_A" in k or "lora_B" in k
                         or "lora_C" in k or "lora_D" in k}
        torch.save(adapter_state, os.path.join(adapter_path, "dual_lora_adapters.pt"))
        save_cfg = {
            "method": args.method,
            "r1": LORA_R1, "r2": LORA_R2,
            "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES, "warmup_steps": WARMUP_STEPS,
            "dataset": args.dataset, "data_type": args.data_type,
        }
        tokenizer.save_pretrained(adapter_path)

    with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
        json.dump(save_cfg, f, indent=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Done. Adapter: {adapter_path}  Method: {args.method}  Trainable: {trainable:,}")


if __name__ == "__main__":
    main()
