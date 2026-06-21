import argparse
import json
import os
import re
import shutil
import sys

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

import config
from dual_lora import DualLoRAConfig, DualLoRALinear, apply_dual_lora


MATH_TEMPLATE = """\
Integrate step-by-step reasoning to solve mathematical problems under following structure:

{{
    "Problem": "[question to be answered]",
    "Solution": {{
        "Step 1": "Begin the response with \\"Let's think step by step.\\"",
        "Step 2": "Follow with the reasoning steps, ensuring the solution process is broken down clearly and logically.",
        "Step 3": "End the solution with the final answer encapsulated in a LaTeX-formatted box, $\\boxed{{...}}$, for clarity and emphasis."
    }},
    "Final Answer": "[final answer to the problem]"
}}

---

Problem: {problem}
Solution:"""

GSM8K_TEMPLATE = """\
**Problem Statement**:
- **Problem**: {question}

**Solution Structure**:
1. Begin the response with "Let's think step by step."
2. Follow with the reasoning steps, ensuring the solution process is broken down clearly and logically.
3. End the solution with the final answer encapsulated in a LaTeX-formatted box, $\\boxed{{...}}$, for clarity and emphasis.
4. Finally, state "The answer is [final answer to the problem].", with the final answer presented in LaTeX notation.

---

Problem: {question}
Solution:"""


def extract_boxed(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"The answer is \$?([0-9,\.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip().split("\n")[-1].strip()


def extract_gsm8k_answer(text: str) -> str:
    m = re.search(r"####\s*(\S+)", text)
    if m:
        return m.group(1).strip()
    return ""


def unwrap_dual_lora(model: nn.Module) -> nn.Module:
    for name, module in list(model.named_modules()):
        if isinstance(module, DualLoRALinear):
            linear = module.unwrap()
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], linear)
    return model


def load_adapter_and_merge(model: nn.Module, adapter_path: str, dual_config: DualLoRAConfig):
    model = apply_dual_lora(model, dual_config)
    state = torch.load(os.path.join(adapter_path, "dual_lora_adapters.pt"), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = unwrap_dual_lora(model)
    return model


def evaluate_on_dataset(model_path: str, dataset_name: str, output_file: str):
    if dataset_name == "math":
        data_file = os.path.join(config.MATH_DATA_DIR, "test.json")
        template = MATH_TEMPLATE
        max_tokens = config.VLLM_MAX_TOKENS
    else:
        data_file = os.path.join(config.GSM8K_DATA_DIR, "test.json")
        template = GSM8K_TEMPLATE
        max_tokens = config.VLLM_MAX_TOKENS

    problems = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    print(f"Evaluating {dataset_name} on merged model: {len(problems)} samples")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for p in problems:
        if dataset_name == "math":
            question = p.get("problem", "")
            prompt_text = template.format(problem=question)
        else:
            question = p.get("question", "")
            prompt_text = template.format(question=question)
        messages = [{"role": "user", "content": prompt_text}]
        tokenizer_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(tokenizer_input)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=config.VLLM_TENSOR_PARALLEL,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=config.VLLM_TEMPERATURE,
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    correct = 0
    for i, (p, out) in enumerate(zip(problems, outputs)):
        generated = out.outputs[0].text.strip()
        pred = extract_boxed(generated)

        if dataset_name == "math":
            gt_sol = p.get("solution", "")
            gt_ans = extract_boxed(gt_sol)
        else:
            gt_raw = p.get("answer", "")
            gt_ans = extract_gsm8k_answer(gt_raw)

        is_correct = pred == gt_ans
        if is_correct:
            correct += 1

        results.append({
            "idx": i,
            "pred": pred,
            "gt": gt_ans,
            "correct": is_correct,
        })

        if i % 500 == 0:
            print(f"  [{i}/{len(problems)}] pred={pred} gt={gt_ans} correct={is_correct}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    acc = 100 * correct / len(problems)
    print(f"\n{'='*50}")
    print(f"  {dataset_name.upper()} ACCURACY: {acc:.1f}% ({correct}/{len(problems)})")
    print(f"{'='*50}")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to adapter folder (with dual_lora_adapters.pt + adapter_config.json)")
    parser.add_argument("--keep-model", action="store_true", help="Keep merged model on disk after eval")
    args = parser.parse_args()

    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        cfg = json.load(f)

    dual_config = DualLoRAConfig(
        r1=cfg.get("r1", config.LORA_R1),
        r2=cfg.get("r2", config.LORA_R2),
        lora_alpha=cfg.get("lora_alpha", config.LORA_ALPHA),
        lora_dropout=cfg.get("lora_dropout", config.LORA_DROPOUT),
        target_modules=cfg.get("target_modules", config.LORA_TARGET),
        warmup_steps=cfg.get("warmup_steps", config.WARMUP_STEPS),
    )

    merged_path = os.path.join(args.adapter, "merged_model")
    if not os.path.exists(merged_path):
        print(f"Loading base model: {config.MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            token=config.HF_TOKEN,
        )

        print("Applying Dual LoRA and merging adapter...")
        model = load_adapter_and_merge(model, args.adapter, dual_config)

        print(f"Saving merged model to {merged_path}")
        model.save_pretrained(merged_path, max_shard_size="10GB")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, token=config.HF_TOKEN)
        tokenizer.save_pretrained(merged_path)
        print("Merged model saved.")
    else:
        print(f"Merged model already exists: {merged_path}")

    math_acc = evaluate_on_dataset(merged_path, "math", os.path.join(args.adapter, "eval_math.jsonl"))
    gsm8k_acc = evaluate_on_dataset(merged_path, "gsm8k", os.path.join(args.adapter, "eval_gsm8k.jsonl"))

    print(f"\n{'='*50}")
    print(f"  FINAL RESULTS")
    print(f"  MATH:  {math_acc:.1f}%")
    print(f"  GSM8K: {gsm8k_acc:.1f}%")
    print(f"{'='*50}")

    if not args.keep_model:
        print(f"Cleaning up merged model: {merged_path}")
        shutil.rmtree(merged_path, ignore_errors=True)


if __name__ == "__main__":
    main()
