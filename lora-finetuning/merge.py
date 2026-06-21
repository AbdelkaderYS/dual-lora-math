import argparse
import json
import os
import subprocess

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from methods import DualLoRAConfig, apply_dual_lora, apply_dual_lora_4bit, unwrap_dual_lora


MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
OUTPUT_BASE = "/mnt/scratch/djagbapr/lora_outputs"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def merge_lora(adapter_path, output_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
    tokenizer.save_pretrained(output_path)
    return output_path


def merge_dual_lora(adapter_path, output_path, use_4bit=False):
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        cfg = json.load(f)

    dual_config = DualLoRAConfig(
        r1=cfg.get("r1", 16), r2=cfg.get("r2", 16),
        lora_alpha=cfg.get("lora_alpha", 32.0),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules", TARGET_MODULES),
        warmup_steps=cfg.get("warmup_steps", 20),
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )

    if use_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb_config,
            device_map="auto", token=HF_TOKEN,
        )
        base_model = prepare_model_for_kbit_training(base_model)
        base_model = apply_dual_lora_4bit(base_model, dual_config)
    else:
        base_model = apply_dual_lora(base_model, dual_config)

    state = torch.load(os.path.join(adapter_path, "dual_lora_adapters.pt"), map_location="cpu")
    base_model.load_state_dict(state, strict=False)
    merged = unwrap_dual_lora(base_model)
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
    tokenizer.save_pretrained(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["lora", "qlora", "dual_lora", "q_dual_lora"])
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.adapter):
        print(f"ERROR: adapter not found: {args.adapter}")
        sys.exit(1)

    tag = args.method.replace("_", "-")
    output_path = args.output_dir or os.path.join(OUTPUT_BASE, f"merged-{tag}-{os.path.basename(os.path.dirname(args.adapter))}")
    os.makedirs(output_path, exist_ok=True)

    print(f"Merging {args.method} adapter: {args.adapter}")
    print(f"Output: {output_path}")

    if args.method in ("lora", "qlora"):
        merge_lora(args.adapter, output_path)
    else:
        use_4bit = args.method == "q_dual_lora"
        merge_dual_lora(args.adapter, output_path, use_4bit=use_4bit)

    result = subprocess.run(["du", "-sh", output_path], capture_output=True, text=True)
    size = result.stdout.split()[0] if result.returncode == 0 else "?"
    print(f"Merged model: {output_path}  ({size})")
    print("Done.")


if __name__ == "__main__":
    main()
