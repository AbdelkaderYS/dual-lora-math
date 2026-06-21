import json
import os
import re
import sys

from vllm import LLM, SamplingParams

import config


MATH_PROMPT_TEMPLATE = """\
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

Problem: {question}
Solution:"""

GSM8K_PROMPT_TEMPLATE = """\
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


def extract_gsm8k_answer(text: str) -> str:
    m = re.search(r"####\s*(\S+)", text)
    if m:
        return m.group(1).strip()
    return ""


def extract_boxed(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"The answer is \$?([0-9,]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(config.GSM8K_TRACES_DIR, exist_ok=True)

    problems = []
    train_file = os.path.join(config.GSM8K_DATA_DIR, "train.json")
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    print(f"Loaded {len(problems)} GSM8K problems from {train_file}")

    prompts = []
    chat_prompts = []
    for p in problems:
        question = p.get("question", "")
        prompt_text = GSM8K_PROMPT_TEMPLATE.format(question=question)
        messages = [
            {"role": "user", "content": prompt_text},
        ]
        tokenizer_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_prompts.append(tokenizer_prompt)
        prompts.append(prompt_text)

    llm = LLM(
        model=config.MODEL_PATH,
        tensor_parallel_size=config.VLLM_TENSOR_PARALLEL,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=config.VLLM_TEMPERATURE,
        max_tokens=config.VLLM_MAX_TOKENS,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    outputs = llm.generate(chat_prompts, sampling_params)

    results = []
    for i, (p, out) in enumerate(zip(problems, outputs)):
        generated = out.outputs[0].text.strip()
        pred = extract_boxed(generated)
        gt_raw = p.get("answer", "")
        gt = extract_gsm8k_answer(gt_raw)

        results.append({
            "idx": i,
            "question": p.get("question", ""),
            "gt_cot": gt_raw,
            "gt": gt,
            "prompt": prompts[i],
            "code": [generated],
            "pred": [pred],
        })

        if i % 500 == 0:
            print(f"  [{i}/{len(problems)}] pred={pred} gt={gt}")

    with open(config.GSM8K_TRACES_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    correct = sum(1 for r in results if r["pred"][0] == r["gt"])
    print(f"GSM8K traces saved: {config.GSM8K_TRACES_FILE}")
    print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")


if __name__ == "__main__":
    main()
