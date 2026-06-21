import json
import os
import re
import sys

from vllm import LLM, SamplingParams

import config


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


def evaluate(dataset_name: str, tokenizer):
    if dataset_name == "math":
        data_dir = config.MATH_DATA_DIR
        template = MATH_TEMPLATE
        output_file = config.MATH_EVAL_OUT
        max_tokens = config.VLLM_MAX_TOKENS
    elif dataset_name == "gsm8k":
        data_dir = config.GSM8K_DATA_DIR
        template = GSM8K_TEMPLATE
        output_file = config.GSM8K_EVAL_OUT
        max_tokens = config.VLLM_MAX_TOKENS
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    problems = []
    with open(os.path.join(data_dir, "test.json")) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    print(f"Evaluating {dataset_name}: {len(problems)} test samples")

    prompts = []
    for p in problems:
        if dataset_name == "math":
            question = p.get("problem", "")
            prompt_text = template.format(problem=question)
        else:
            question = p.get("question", "")
            prompt_text = template.format(question=question)
        messages = [
            {"role": "user", "content": prompt_text},
        ]
        tokenizer_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(tokenizer_input)

    llm = LLM(
        model=config.MODEL_PATH,
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
            gt = p.get("solution", "")
            gt_ans = extract_boxed(gt)
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
            "generated": generated,
        })

        if i % 500 == 0:
            print(f"  [{i}/{len(problems)}] pred={pred} gt={gt_ans} correct={is_correct}")

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    acc = 100 * correct / len(problems)
    print(f"\n{'='*50}")
    print(f"  {dataset_name.upper()} TEST ACCURACY: {acc:.1f}% ({correct}/{len(problems)})")
    print(f"{'='*50}")
    return acc


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    math_acc = evaluate("math", tokenizer)
    gsm8k_acc = evaluate("gsm8k", tokenizer)
    print(f"\nFinal results:")
    print(f"  MATH:  {math_acc:.1f}%")
    print(f"  GSM8K: {gsm8k_acc:.1f}%")


if __name__ == "__main__":
    main()
