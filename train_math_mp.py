import os, json, torch
from typing import Dict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, TrainerCallback,
)
from datasets import Dataset
from dual_lora import DualLoRAConfig, apply_dual_lora, set_model_step


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


MODEL_PATH = "/mnt/scratch/djagbapr/models/Llama-3.3-70B-Instruct"
DATA_DIR = "/mnt/scratch/djagbapr/lora_data_mp"
OUTPUT_DIR = "/mnt/scratch/djagbapr/lora_outputs/dual_lora_math_mp"
LOG_DIR = "/mnt/scratch/djagbapr/lora_outputs/logs"

LORA_R1 = 16
LORA_R2 = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
WARMUP_STEPS = 20

MAX_SEQ_LEN = 4046
PER_DEVICE_BATCH_SIZE = 1
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
LOGGING_STEPS = 10
SEED = 42
MAX_GRAD_NORM = 1.0

n_gpus = torch.cuda.device_count()
if n_gpus <= 1:
    GRADIENT_ACCUMULATION = 8
elif n_gpus == 2:
    GRADIENT_ACCUMULATION = 4
else:
    GRADIENT_ACCUMULATION = 2

print(f"GPUs: {n_gpus}  Grad accum: {GRADIENT_ACCUMULATION}  "
      f"Effective batch: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION * max(n_gpus, 1)}")

for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name}  ({mem:.1f} GB)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

model.config.use_cache = False
model.enable_input_require_grads()

config = DualLoRAConfig(
    r1=LORA_R1, r2=LORA_R2, lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET,
    warmup_steps=WARMUP_STEPS,
)
model = apply_dual_lora(model, config)
model.gradient_checkpointing_enable()


def load_jsonl(path: str) -> Dataset:
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


def tokenize(example: Dict) -> Dict:
    tokenized = tokenizer(
        example["text"], max_length=MAX_SEQ_LEN,
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

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    max_grad_norm=MAX_GRAD_NORM,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=LOG_DIR,
    logging_steps=LOGGING_STEPS,
    save_total_limit=2,
    seed=SEED,
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
    "r1": LORA_R1, "r2": LORA_R2, "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT, "target_modules": LORA_TARGET,
    "warmup_steps": WARMUP_STEPS, "base_model": MODEL_PATH,
    "data": "math_mp",
}
with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
    json.dump(save_cfg, f, indent=2)

tokenizer.save_pretrained(adapter_path)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Adapter saved: {adapter_path}  Trainable params: {trainable:,}")
