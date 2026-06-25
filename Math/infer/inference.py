import os
import argparse
from tqdm import tqdm

from vllm import LLM, SamplingParams

from eval.evaluate import evaluate
from utils.utils import set_seed, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data


Llama3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = '<|begin_of_text|>' + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_type", type=str, default="mp")
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--num_test_sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens_per_call", type=int, default=2048)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--use_train_prompt_format", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    return parser.parse_args()


def is_llama_instruct(model_name):
    name = model_name.lower()
    return "llama" in name and "instruct" in name


def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        chat_template=Llama3_CHAT_TEMPLATE,
    )


def main():
    args = parse_args()
    if args.temperature == 0:
        args.top_p = 1
    set_seed(args.seed)

    print(f"Loading data: {args.data_name} ({args.split})")
    examples = load_data(args.data_name, args.split)
    if args.num_test_sample > 0:
        examples = examples[:args.num_test_sample]
    if args.end == -1:
        args.end = len(examples)
    examples = examples[args.start:args.end]
    print(f"Examples: {len(examples)}")

    print(f"Loading model: {args.model_name_or_path}")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()

    IS_LLAMA_INSTRUCT = is_llama_instruct(args.model_name_or_path)

    ans_split = "\n\nProblem: "
    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "---", "```output", ans_split.strip()]
    if not IS_LLAMA_INSTRUCT:
        stop_tokens = ["</s>", "---", "```output", ans_split.strip()]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens_per_call,
        n=args.n_sampling,
        stop=stop_tokens,
    )

    results = []
    for example in tqdm(examples, desc="Generating"):
        idx = example["idx"]

        example["question"] = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)

        prompt = construct_prompt(args, example)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = apply_chat_template(tokenizer, messages) if IS_LLAMA_INSTRUCT else prompt

        outputs = llm.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        answer = extract_answer(generated_text)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt": gt_ans,
            "gt_cot": gt_cot,
            "pred": [answer],
            "code": [],
            "prompt": formatted_prompt,
            "generated_text": generated_text,
        }
        for key in ["level", "type", "subject", "solution"]:
            if key in example:
                sample[key] = example[key]
        results.append(sample)

    output_dir = f"outputs/{args.data_name}/{args.prompt_type}"
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"{args.data_name}_{args.prompt_type}_seed{args.seed}_t{args.temperature}"
    results_file = f"{output_dir}/{base_name}.jsonl"
    save_jsonl(results, results_file)

    eval_result = evaluate(
        samples=results,
        data_name=args.data_name,
        prompt_type=args.prompt_type,
    )
    print(eval_result)
    print("Done.")


if __name__ == "__main__":
    main()
