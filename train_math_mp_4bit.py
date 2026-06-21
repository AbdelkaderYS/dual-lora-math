import os
import json
import torch
import torch.nn as nn
from typing import Dict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback,
)
from datasets import Dataset
from dual_lora import DualLoRAConfig, DualLoRALinear, set_model_step
from peft import prepare_model_for_kbit_training
import config


class DualLoRATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class WarmupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            set_model_step(model, state.global_step)
        return control


def apply_dual_lora_4bit(model: nn.Module, dual_config: DualLoRAConfig) -> nn.Module:
    try:
        from bitsandbytes.nn import Linear4bit
        accepted = (nn.Linear, Linear4bit)
    except ImportError:
        accepted = (nn.Linear,)

    replaced = 0
    for name, module in list(model.named_modules()):
        for target in dual_config.target_modules:
            if not name.endswith(target):
                continue
            if not isinstance(module, accepted):
                continue
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]
            setattr(parent, child_name, DualLoRALinear(module, dual_config))
            replaced += 1
            break
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[DualLoRA-4bit] Replaced {replaced} layers | "
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    return model


def main():
    DATA_DIR = config.MATH_PREP_DIR
    OUTPUT_DIR = config.MATH_4BIT_OUTPUT

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        grad_accum = 8
    elif n_gpus == 2:
        grad_accum = 4
    else:
        grad_accum = 2

    print(f"GPUs: {n_gpus}  Grad accum: {grad_accum}  "
          f"Effective batch: {config.PER_DEVICE_BATCH_SIZE * grad_accum * max(n_gpus, 1)}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        token=config.HF_TOKEN,
    )

    model.config.use_cache = False
    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model)

    dual_config = DualLoRAConfig(
        r1=config.LORA_R1, r2=config.LORA_R2,
        lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET, warmup_steps=config.WARMUP_STEPS,
    )
    model = apply_dual_lora_4bit(model, dual_config)
    model.gradient_checkpointing_enable()

    def load_jsonl(path):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return Dataset.from_list(records)

    train_dataset = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_dataset = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    def tokenize(example):
        tokenized = tokenizer(
            example["text"], max_length=config.MAX_SEQ_LEN,
            truncation=True, padding=False, return_tensors=None,
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

    train_tok = train_dataset.map(tokenize, remove_columns=train_dataset.column_names, num_proc=4)
    val_tok = val_dataset.map(tokenize, remove_columns=val_dataset.column_names, num_proc=4)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=config.MAX_GRAD_NORM,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        save_total_limit=2,
        seed=config.SEED,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = DualLoRATrainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        processing_class=tokenizer, data_collator=collator,
        callbacks=[WarmupCallback()],
    )

    trainer.train()

    adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    os.makedirs(adapter_path, exist_ok=True)

    adapter_state = {k: v.cpu() for k, v in model.state_dict().items()
                     if "dual_lora" in k or "lora_A" in k or "lora_B" in k
                     or "lora_C" in k or "lora_D" in k}
    torch.save(adapter_state, os.path.join(adapter_path, "dual_lora_adapters.pt"))

    save_cfg = {
        "r1": config.LORA_R1, "r2": config.LORA_R2,
        "lora_alpha": config.LORA_ALPHA, "lora_dropout": config.LORA_DROPOUT,
        "target_modules": config.LORA_TARGET, "warmup_steps": config.WARMUP_STEPS,
        "base_model": config.MODEL_PATH, "data": "math_mp",
        "quantization": "4bit_nf4",
    }
    with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
        json.dump(save_cfg, f, indent=2)

    tokenizer.save_pretrained(adapter_path)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MATH 4-bit Dual LoRA done. Adapter: {adapter_path}  Trainable: {trainable:,}")


if __name__ == "__main__":
    main()
