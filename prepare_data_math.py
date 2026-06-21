import json
import os
import random
import re
import sys

import config


def extract_answer(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    return ""


def main():
    os.makedirs(config.MATH_PREP_DIR, exist_ok=True)

    records = []
    with open(config.MATH_TRACES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} MATH traces from {config.MATH_TRACES_FILE}")

    correct = []
    for r in records:
        pred = r.get("pred", [None])[0]
        gt = r.get("gt", "")
        if pred is not None and str(pred).strip() == str(gt).strip():
            correct.append(r)
        else:
            pred = extract_answer(r.get("code", [""])[0])
            gt_ans = extract_answer(r.get("gt_cot", ""))
            if pred and gt_ans and pred == gt_ans:
                correct.append(r)

    print(f"Correct samples: {len(correct)} / {len(records)}")

    random.seed(config.SEED)
    random.shuffle(correct)
    split = int(len(correct) * 0.9)
    train = correct[:split]
    val = correct[split:]

    def format_sample(r):
        prompt = r.get("prompt", "")
        completion = r.get("code", [""])[0] if r.get("code") else ""
        text = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{completion}<|eot_id|>"
        )
        return {"text": text}

    with open(os.path.join(config.MATH_PREP_DIR, "train.jsonl"), "w") as f:
        for s in train:
            f.write(json.dumps(format_sample(s)) + "\n")

    with open(os.path.join(config.MATH_PREP_DIR, "val.jsonl"), "w") as f:
        for s in val:
            f.write(json.dumps(format_sample(s)) + "\n")

    print(f"MATH data prepared: {len(train)} train, {len(val)} val -> {config.MATH_PREP_DIR}")


if __name__ == "__main__":
    main()
