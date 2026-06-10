"""
validate_preprocessing.py
=========================
Run this BEFORE fine-tuning to verify the data pipeline is correct.

Validates:
  1. Tokenizer
  2. Prompt templates (train/eval coherence)
  3. Label masking
  4. MetaMathQA sample statistics (with max_seq_length=1024)
  5. Answer extraction — extract_final_answer, extract_gsm8k_answer, extract_math_answer
  6. Train/val split
  7. Summary

Usage:
    cd /mnt/scratch/djagbapr/dual-lora-math
    source /mnt/home/djagbapr/envs/dual_lora/bin/activate
    python scripts/validate_preprocessing.py
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from training.data_loader import PROMPT, INFERENCE, _tokenize, load_metamath_dataset
from evaluation.answer_extraction import (
    extract_gsm8k_gold,
    extract_gsm8k_answer,
    extract_math_answer,
    extract_final_answer,
    to_canonical_number,
    answers_are_equal,
)

SEP = "=" * 70
MAX_SEQ_LEN = 1024   # matches base_config.yaml

# ─────────────────────────────────────────────────────────────────
# 1. Tokenizer
# ─────────────────────────────────────────────────────────────────
print(SEP)
print("1. TOKENIZER")
print(SEP)

MODEL = "EleutherAI/llemma_7b"
print(f"Loading tokenizer: {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"Vocab size   : {tok.vocab_size:,}")
print(f"PAD token    : {repr(tok.pad_token)} (id={tok.pad_token_id})")
print(f"EOS token    : {repr(tok.eos_token)} (id={tok.eos_token_id})")

# ─────────────────────────────────────────────────────────────────
# 2. Prompt templates
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("2. PROMPT TEMPLATES")
print(SEP)

q = "If x + 2 = 7, what is x?"
a = "We solve: x = 7 - 2 = 5. The answer is: \\boxed{5}."

full_train  = PROMPT.format(problem=q, solution=a)
eval_prompt = INFERENCE.format(problem=q)

print("TRAIN template (full):")
print(full_train)
print()
print("EVAL template (prompt only):")
print(eval_prompt)
print()

prefix_ok = full_train.startswith(eval_prompt)
print(f"PREFIX CHECK : {'PASS' if prefix_ok else 'FAIL — mismatch between train and eval!'}")

# ─────────────────────────────────────────────────────────────────
# 3. Label masking
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"3. LABEL MASKING (max_seq_length={MAX_SEQ_LEN})")
print(SEP)

ex = {"query": q, "response": a}
result = _tokenize(ex, tok, max_len=MAX_SEQ_LEN)   # matches training config

input_ids = result["input_ids"]
labels    = result["labels"]

prompt_ids = tok(
    INFERENCE.format(problem=q.strip()),
    max_length=MAX_SEQ_LEN, truncation=True,
    padding=False, return_tensors=None
)["input_ids"]

prompt_len   = len(prompt_ids)
total_len    = len(input_ids)
solution_len = total_len - prompt_len
masked_count = sum(1 for l in labels if l == -100)

print(f"Total tokens    : {total_len}")
print(f"Prompt tokens   : {prompt_len}  (masked -100, NOT trained)")
print(f"Solution tokens : {solution_len} (trained)")
print(f"Training ratio  : {100*solution_len/total_len:.1f}%")

masking_ok = masked_count == prompt_len
print(f"Masking check   : {'PASS' if masking_ok else 'FAIL'} ({masked_count} masked)")

print("\nPrompt (not trained):")
print("  " + repr(tok.decode(input_ids[:prompt_len], skip_special_tokens=True)[:120]))
print("Solution (trained):")
print("  " + repr(tok.decode(input_ids[prompt_len:], skip_special_tokens=True)[:120]))

# ─────────────────────────────────────────────────────────────────
# 4. MetaMathQA sample statistics
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"4. METAMATHQA SAMPLE STATISTICS (100 examples, max_seq_length={MAX_SEQ_LEN})")
print(SEP)

from datasets import load_dataset
from collections import Counter

meta = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=42).select(range(100))

lengths = []
for sample in meta:
    full = PROMPT.format(problem=sample["query"], solution=sample["response"])
    ids  = tok(full, truncation=False, return_tensors=None)["input_ids"]
    lengths.append(len(ids))

lengths_sorted = sorted(lengths)
truncated_1024 = sum(1 for l in lengths if l > MAX_SEQ_LEN)
truncated_512  = sum(1 for l in lengths if l > 512)

print(f"Min length         : {lengths_sorted[0]} tokens")
print(f"Median length      : {lengths_sorted[len(lengths_sorted)//2]} tokens")
print(f"95th pct           : {lengths_sorted[int(len(lengths_sorted)*0.95)]} tokens")
print(f"Max length         : {lengths_sorted[-1]} tokens")
print(f"Truncated > 512    : {truncated_512}/100 ({truncated_512}%)  [old config]")
print(f"Truncated > {MAX_SEQ_LEN}  : {truncated_1024}/100 ({truncated_1024}%)  [current config ✓]")

types = Counter(s["type"] for s in meta)
print(f"\nProblem types:")
for t, c in types.most_common():
    print(f"  {t:<30} : {c}")

# ─────────────────────────────────────────────────────────────────
# 5. Answer extraction
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("5. ANSWER EXTRACTION")
print(SEP)

all_pass = True

# ── 5a. extract_final_answer — generic cases ───────────────────
print("5a. extract_final_answer (generic fallback):")
print(f"{'Input':<35}  {'Got':<18}  {'Expected':<12}  Status")
print("-" * 80)

generic_cases = [
    (r"\boxed{42}",           "42",      "42"),
    (r"\boxed{\frac{1}{2}}",  "0.5",     "0.5"),
    (r"\frac{3}{4}",          "0.75",    "0.75"),
    (r"\frac{22}{7}",         "3.14286", "3.14286"),
    ("The answer is: 15.",    "15",      "15"),
    (r"\$50",                 "50",      "50"),
    ("75%",                   "0.75",    "0.75"),
    ("1,000",                 "1000",    "1000"),
    ("-7",                    "-7",      "-7"),
]

for inp, _, expected in generic_cases:
    raw   = extract_final_answer(inp)
    canon = to_canonical_number(raw) if raw else None
    got   = canon or raw
    ok    = answers_are_equal(got, expected) if got else False
    if not ok:
        all_pass = False
    print(f"{inp:<35}  {str(got):<18}  {expected:<12}  {'PASS' if ok else 'FAIL'}")

# ── 5b. extract_gsm8k_answer — MetaMath eval_gsm8k.py protocol ─
print(f"\n5b. extract_gsm8k_answer (MetaMath eval_gsm8k.py protocol):")
print("     Priority: 'The answer is: ' → regex number → Fraction")
print(f"{'Input':<55}  {'Got':<10}  {'Expected':<10}  Status")
print("-" * 85)

gsm_cases = [
    ("Let's think step by step. ... The answer is: 42",         "42",   "42"),
    ("...calculate... The answer is: 1,500",                    "1500", "1500"),
    ("...The answer is: 1/2",                                   "0.5",  "0.5"),
    ("...The answer is: -7",                                    "-7",   "-7"),
    ("No answer pattern — #### 99",                             "99",   "99"),
    ("...The answer is: 3/4",                                   "0.75", "0.75"),
]

for inp, _, expected in gsm_cases:
    got = extract_gsm8k_answer(inp)
    ok  = (str(got) == expected) if got is not None else False
    if not ok:
        all_pass = False
    print(f"{inp[:55]:<55}  {str(got):<10}  {expected:<10}  {'PASS' if ok else 'CHECK'}")

# ── 5c. extract_math_answer — MetaMath eval_math.py protocol ───
print(f"\n5c. extract_math_answer (MetaMath eval_math.py protocol):")
print("     Priority: 'The answer is: ' → unwrap \\boxed{} → last \\boxed{}")
print(f"{'Input':<60}  {'Got':<20}  Status")
print("-" * 85)

math_cases = [
    # (input, expected)
    (r"...The answer is: \boxed{\frac{1}{2}}",        r"\frac{1}{2}"),
    (r"...The answer is: 42",                          "42"),
    # Critical: intermediate \boxed{} must not be captured
    (r"So \boxed{3} per child. The answer is: \boxed{12}", "12"),
    # Fallback: no "The answer is:" — take last \boxed{}
    (r"Therefore \boxed{3} and \boxed{7}",             "7"),
    # Fallback: single \boxed{}
    (r"The result is \boxed{x^2 + 1}",                "x^2 + 1"),
]

for inp, expected in math_cases:
    got = extract_math_answer(inp)
    ok  = answers_are_equal(got, expected) if got else got == expected
    if not ok:
        all_pass = False
    print(f"{inp[:60]:<60}  {str(got):<20}  {'PASS' if ok else 'FAIL'}")

# ── 5d. GSM8K gold format ───────────────────────────────────────
print(f"\n5d. extract_gsm8k_gold (#### format):")
for text, expected in [("#### 5", "5"), ("#### 1,500", "1500"), ("#### -10", "-10")]:
    got = extract_gsm8k_gold(text)
    ok  = got == expected
    if not ok:
        all_pass = False
    print(f"  '{text}' → {got!r}  {'PASS' if ok else 'FAIL'}")

# ── 5e. answers_are_equal tolerance ────────────────────────────
print(f"\n5e. answers_are_equal tolerance check (tol=1e-3):")
tol_cases = [
    ("0.333",   "1/3",   True,  "0.333 ≈ 1/3 within 1e-3"),
    ("0.333",   "1/3",   True,  "tol=1e-3 should accept"),
    ("3.14159", "22/7",  False, "pi vs 22/7 differ by ~1.2e-4 < 1e-3 → True"),
]
for pred, gold, _, desc in tol_cases:
    result = answers_are_equal(pred, gold)
    print(f"  answers_are_equal({pred!r}, {gold!r}) = {result}  [{desc}]")

# ─────────────────────────────────────────────────────────────────
# 6. Train/val split
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"6. TRAIN / VAL SPLIT (500 debug samples, max_seq_length={MAX_SEQ_LEN})")
print(SEP)

train_ds, val_ds = load_metamath_dataset(
    tokenizer=tok,
    max_seq_length=MAX_SEQ_LEN,   # matches training config
    num_train_samples=500,
    val_ratio=0.02,
    seed=42,
)
print(f"Train : {len(train_ds):,} examples")
print(f"Val   : {len(val_ds):,} examples")
print(f"Split : {100*len(val_ds)/(len(train_ds)+len(val_ds)):.1f}% val")
print(f"Fields: {list(train_ds.features.keys())}")

split_ok = len(train_ds) > 0 and len(val_ds) > 0

# ─────────────────────────────────────────────────────────────────
# 7. Summary
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("7. SUMMARY")
print(SEP)

checks = {
    "Tokenizer loaded"            : True,
    "Prompt prefix matches"       : prefix_ok,
    "Label masking correct"       : masking_ok,
    "Answer extraction OK"        : all_pass,
    "Dataset split OK"            : split_ok,
}

for name, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}]  {name}")

print()
if all(checks.values()):
    print("ALL CHECKS PASSED")
    print()
    print("Next:")
    print("  bash scripts/train.sh configs/dual_lora_config.yaml --debug")
else:
    print("SOME CHECKS FAILED ")