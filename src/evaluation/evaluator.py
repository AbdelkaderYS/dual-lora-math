"""
evaluator.py
============
Evaluation runner for GSM8K and MATH benchmarks.

Aligned with MetaMath official eval protocol:
  - eval_gsm8k.py : split on "The answer is: ", extract number via regex
  - eval_math.py  : split on "The answer is: ", symbolic comparison via is_equiv
  - max_new_tokens=512 for GSM8K, 1024 for MATH (MetaMath uses 2048)
  - stop tokens to prevent multi-answer generation
  - max_length=1024 matches training max_seq_length

Usage:
    python src/evaluation/evaluator.py \
        --checkpoint_path experiments/results/dual_lora_seed42 \
        --dataset both \
        --batch_size 32
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    StoppingCriteria, StoppingCriteriaList,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_loader import format_inference_prompt, load_gsm8k_eval, load_math_eval
from evaluation.answer_extraction import (
    extract_gsm8k_answer, extract_math_answer,
    extract_gsm8k_gold, extract_math_gold,
    answers_are_equal,
)
from evaluation.metrics import compute_accuracy, compute_per_category_accuracy
from utils.logging_utils import setup_logging


# 
# Stop tokens — aligned with MetaMath eval official
# Prevents model from generating a second problem after its answer
# 

STOP_STRINGS = ["Question:", "USER:", "ASSISTANT:", "Instruction:", "Response:"]


class StopOnStrings(StoppingCriteria):
    """Stop generation when any stop string is produced."""
    def __init__(self, stop_strings: List[str], tokenizer):
        self.stop_ids = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in stop_strings
        ]

    def __call__(self, input_ids, scores, **kwargs):
        for stop in self.stop_ids:
            if len(stop) > 0 and input_ids[0][-len(stop):].tolist() == stop:
                return True
        return False


# 
# Model loading
# 

def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Load checkpoint correctly per method.
    - dual_lora : rebuild DualLoRALinear layers, load adapter weights
    - lora/qlora: PEFT PeftModel → merge_and_unload
    - full_ft   : from_pretrained directly
    """
    config_path = os.path.join(checkpoint_path, "experiment_config.json")
    method = "unknown"
    base_model_name = "EleutherAI/llemma_7b"

    if os.path.exists(config_path):
        with open(config_path) as f:
            exp_config = json.load(f)
        method          = exp_config.get("experiment", {}).get("method", "unknown")
        base_model_name = exp_config.get("model",      {}).get("name", base_model_name)

    print(f"[Evaluator] Method     : {method}")
    print(f"[Evaluator] Base model : {base_model_name}")
    print(f"[Evaluator] Device     : {device}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import transformers as _tf
    dtype_kwarg = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"

    if method == "dual_lora":
        print("[Evaluator] Rebuilding DualLoRA structure...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **{dtype_kwarg: torch.bfloat16},
            device_map=None,
        )
        from models.dual_lora import DualLoRAConfig, apply_dual_lora
        lora_cfg = exp_config.get("lora", {})
        dual_cfg = exp_config.get("dual_lora", {})
        adapter_config = DualLoRAConfig(
            r1=lora_cfg.get("r1", 16),
            r2=lora_cfg.get("r2", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32.0),
            lora_dropout=0.0,
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            init_std=dual_cfg.get("init_std", 0.02),
        )
        model = apply_dual_lora(model, adapter_config)

        adapter_path = os.path.join(checkpoint_path, "dual_lora_adapters.pt")
        if os.path.exists(adapter_path):
            # weights_only=True : sécurité PyTorch 2.6+
            state = torch.load(adapter_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[Evaluator] Adapter loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
        else:
            print("[WARN] dual_lora_adapters.pt not found — trying pytorch_model.bin fallback")
            sd_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(sd_path):
                state = torch.load(sd_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=False)

        model = model.to(device).to(dtype=torch.bfloat16)

    elif method in ("lora", "qlora"):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **{dtype_kwarg: torch.bfloat16},
            device_map=None,
        )
        model = PeftModel.from_pretrained(base, checkpoint_path)
        model = model.merge_and_unload()
        model = model.to(device).to(dtype=torch.bfloat16)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            **{dtype_kwarg: torch.bfloat16},
            device_map=None,
        )
        model = model.to(device).to(dtype=torch.bfloat16)

    model.eval()
    return model, tokenizer


# 
# Generation
# 

def generate_answers(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 32,
    max_new_tokens: int = 512,
    max_input_length: int = 1024,
    device: str = "cuda",
) -> List[str]:
    """
    max_input_length=1024 : matches training max_seq_length.
    Stop tokens prevent multi-answer generation (MetaMath eval protocol).
    """
    model.eval()
    all_outputs = []
    stopping_criteria = StoppingCriteriaList([
        StopOnStrings(STOP_STRINGS, tokenizer)
    ])

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,   # FIX: was hardcoded 512
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        for out in output_ids:
            input_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            all_outputs.append(decoded)

    return all_outputs


# 
# Evaluation runners
# 

def evaluate_gsm8k(
    model, tokenizer,
    batch_size: int = 32,
    max_new_tokens: int = 512,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> dict:
    """
    GSM8K evaluation aligned with MetaMath eval_gsm8k.py:
    - extract_gsm8k_answer: split on 'The answer is: ', extract number via regex
    - gold: extracted via #### pattern
    """
    dataset = load_gsm8k_eval()
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    problems     = [ex["question"] for ex in dataset]
    gold_answers = [extract_gsm8k_gold(ex["answer"]) for ex in dataset]
    prompts      = [format_inference_prompt(p) for p in problems]

    outputs     = generate_answers(
        model, tokenizer, prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_input_length=1024,
        device=device,
    )
    predictions = [extract_gsm8k_answer(out) for out in outputs]
    metrics     = compute_accuracy(predictions, gold_answers)

    return {
        "dataset":      "gsm8k",
        "metrics":      metrics,
        "predictions":  predictions,
        "gold_answers": gold_answers,
    }


def evaluate_math(
    model, tokenizer,
    batch_size: int = 32,
    max_new_tokens: int = 1024,   # FIX: 1024 for MATH level 4-5 long solutions
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> dict:
    """
    MATH evaluation aligned with MetaMath eval_math.py:
    - extract_math_answer: split on 'The answer is: ', symbolic comparison
    - gold: extracted from \boxed{} in solution
    - max_new_tokens=1024 (MetaMath uses 2048, 1024 is a reasonable compromise)
    """
    dataset = load_math_eval()
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    problems    = [ex["problem"] for ex in dataset]
    gold_raw    = [ex["solution"] for ex in dataset]
    categories  = [ex.get("type", "unknown") for ex in dataset]

    prompts      = [format_inference_prompt(p) for p in problems]
    gold_answers = [extract_math_gold(g) for g in gold_raw]

    outputs     = generate_answers(
        model, tokenizer, prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_input_length=1024,
        device=device,
    )
    predictions  = [extract_math_answer(out) for out in outputs]
    overall      = compute_accuracy(predictions, gold_answers)
    per_category = compute_per_category_accuracy(predictions, gold_answers, categories)

    return {
        "dataset":      "math",
        "metrics":      overall,
        "per_category": per_category,
        "predictions":  predictions,
        "gold_answers": gold_answers,
    }


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset",        choices=["gsm8k", "math", "both"], default="both")
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens for GSM8K. MATH always uses max(this, 1024).")
    parser.add_argument("--max_samples",    type=int, default=None,
                        help="Limit samples for quick testing. None = full dataset.")
    args = parser.parse_args()

    logger = setup_logging("evaluator")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Evaluator] Checkpoint : {args.checkpoint_path}")
    print(f"[Evaluator] Dataset    : {args.dataset}")
    print(f"[Evaluator] Batch size : {args.batch_size}")
    print(f"[Evaluator] Device     : {device}")

    model, tokenizer = load_checkpoint(args.checkpoint_path, device)
    results = {"checkpoint": args.checkpoint_path}

    if args.dataset in ("gsm8k", "both"):
        logger.info("Evaluating on GSM8K (max_new_tokens=512)...")
        r = evaluate_gsm8k(
            model, tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            device=device,
        )
        results["gsm8k"]             = r["metrics"]
        results["gsm8k_predictions"] = r["predictions"]    # needed for error analysis
        results["gsm8k_gold"]        = r["gold_answers"]   # needed for error analysis
        logger.info(f"GSM8K: {r['metrics']['accuracy_pct']:.2f}%  "
                    f"({r['metrics']['correct']}/{r['metrics']['total']})")

    if args.dataset in ("math", "both"):
        math_tokens = max(args.max_new_tokens, 1024)
        logger.info(f"Evaluating on MATH (max_new_tokens={math_tokens})...")
        r = evaluate_math(
            model, tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=math_tokens,
            max_samples=args.max_samples,
            device=device,
        )
        results["math"]              = r["metrics"]
        results["math_per_category"] = r["per_category"]
        results["math_predictions"]  = r["predictions"]    # needed for error analysis
        results["math_gold"]         = r["gold_answers"]   # needed for error analysis
        logger.info(f"MATH : {r['metrics']['accuracy_pct']:.2f}%  "
                    f"({r['metrics']['correct']}/{r['metrics']['total']})")

    out_path = os.path.join(args.checkpoint_path, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    if "gsm8k" in results:
        r = results["gsm8k"]
        print(f"GSM8K : {r['accuracy_pct']:.2f}%  ({r['correct']}/{r['total']})")
    if "math" in results:
        r = results["math"]
        print(f"MATH  : {r['accuracy_pct']:.2f}%  ({r['correct']}/{r['total']})")
    print("=" * 50)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()