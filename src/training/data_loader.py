import os
import numpy as np
from typing import Optional, Tuple
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{problem}\n\n"
    "### Response: Let's think step by step. {response}"
)

INFERENCE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{problem}\n\n"
    "### Response: Let's think step by step."
)


def format_inference_prompt(problem: str) -> str:
    return INFERENCE.format(problem=problem.strip())


def _tokenize(
    ex: dict,
    tokenizer: PreTrainedTokenizer,
    max_len: int,
) -> dict:
    full = PROMPT.format(
        problem=ex["query"].strip(),
        response=ex["response"].strip(),
    )
    tok = tokenizer(
        full, max_length=max_len, truncation=True, padding=False, return_tensors=None,
    )
    prompt_ids = tokenizer(
        INFERENCE.format(problem=ex["query"].strip()),
        max_length=max_len, truncation=True, padding=False, return_tensors=None,
    )["input_ids"]

    labels = tok["input_ids"].copy()
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

    tok["labels"] = np.array(labels, dtype=np.int64)
    tok["input_ids"] = np.array(tok["input_ids"], dtype=np.int64)
    tok["attention_mask"] = np.array(tok["attention_mask"], dtype=np.int64)
    return tok


def load_metamath_dataset(
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    num_train_samples: Optional[int] = None,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    print("[DataLoader] Loading MetaMathQA...")
    ds = load_dataset("meta-math/MetaMathQA", split="train")

    if num_train_samples:
        ds = ds.shuffle(seed=seed).select(range(num_train_samples))
        print(f"[DataLoader] Debug: {num_train_samples} samples")

    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    train_raw = split["train"]
    val_raw = split["test"]

    print(f"[DataLoader] Train: {len(train_raw):,}  Val: {len(val_raw):,}")

    def tok_ds(d, desc):
        out = d.map(
            lambda ex: _tokenize(ex, tokenizer, max_seq_length),
            batched=False,
            remove_columns=d.column_names,
            desc=desc,
            num_proc=min(4, max(1, os.cpu_count() // 2)),
        )
        out.set_format("torch")
        return out

    train_ds = tok_ds(train_raw, "Tokenizing train")
    val_ds = tok_ds(val_raw, "Tokenizing val")
    return train_ds, val_ds


def load_gsm8k_eval(split: str = "test") -> Dataset:
    ds = load_dataset("gsm8k", "main", split=split)
    print(f"[DataLoader] GSM8K: {len(ds):,}")
    return ds


def load_math_eval(split: str = "test") -> Dataset:
    ds = load_dataset("lighteval/MATH", split=split, trust_remote_code=True)
    print(f"[DataLoader] MATH: {len(ds):,}")
    return ds
