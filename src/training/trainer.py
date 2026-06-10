"""
trainer.py  —  Modular trainer: dual_lora | lora | qlora | full_ft
Any HuggingFace CausalLM. Resources auto-detected at startup.
"""
import argparse, os, sys, warnings
from pathlib import Path
import torch, yaml
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
    TrainingArguments, Trainer, set_seed,
)
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.dual_lora import DualLoRAConfig, apply_dual_lora
from models.lora_baseline import LoRABaselineConfig, apply_lora_baseline
from training.data_loader import load_metamath_dataset
from utils.hardware import detect_hardware, build_accelerate_config, get_optimal_batch_size
from utils.logging_utils import setup_logging, log_config
from utils.checkpoint import save_experiment_config

# ── Suppress known non-critical warnings ────────────────────
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays")
warnings.filterwarnings("ignore", message="warmup_ratio is deprecated")
warnings.filterwarnings("ignore", message="torch_dtype is deprecated")


# ── HuggingFace auto-login ───────────────────────────────────
def _hf_login():
    #token = os.environ.get("HF_TOKEN", "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")
    token = os.environ.get("HF_TOKEN", "")

    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass

_hf_login()


def load_config(path: str) -> dict:
    import copy
    config_dir = Path(path).parent
    with open(path) as f:
        config = yaml.safe_load(f)
    defaults = config.pop("defaults", [])
    base = {}
    for d in defaults:
        base_name = d if isinstance(d, str) else list(d.values())[0]
        base_path = config_dir / f"{base_name}.yaml"
        if base_path.exists():
            with open(base_path) as f:
                base = yaml.safe_load(f)
            base.pop("defaults", None)
    return _deep_merge(base, config)


def _deep_merge(base: dict, override: dict) -> dict:
    import copy
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_model_and_tokenizer(config, hw):
    name = config["model"]["name"]
    method = config["experiment"]["method"]
    dtype = getattr(torch, hw.torch_dtype)
    print(f"[Trainer] Loading {name} (method={method}, dtype={hw.torch_dtype})")

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    kw = {}
    if method == "qlora":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        kw["device_map"] = "auto"
        print("[Trainer] 4-bit QLoRA quantization")
    else:
        import transformers as _tf
        dtype_key = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
        kw[dtype_key] = dtype
        if hw.n_gpus <= 1:
            kw["device_map"] = "auto"

    # Utilisation du Flash Attention natif de PyTorch (SDPA) au lieu de la bibliothèque externe
    if method != "full_ft":
        kw["attn_implementation"] = "sdpa"
        print("[Trainer] Using PyTorch SDPA (native Flash Attention)")

    model = AutoModelForCausalLM.from_pretrained(name, **kw)

    if method == "full_ft":
        model.gradient_checkpointing_enable()
        total = sum(p.numel() for p in model.parameters())
        print(f"[Trainer] Full FT — {total:,} trainable params")
    return model, tok


def apply_adapter(model, config):
    method = config["experiment"]["method"]
    lcfg = config.get("lora", {})
    if method == "dual_lora":
        dcfg = config.get("dual_lora", {})
        ac = DualLoRAConfig(
            r1=lcfg["r1"], r2=lcfg["r2"], lora_alpha=lcfg["lora_alpha"],
            lora_dropout=lcfg.get("lora_dropout", 0.05),
            target_modules=lcfg["target_modules"],
            init_std=dcfg.get("init_std", 0.02),
            sign_ste_threshold=dcfg.get("sign_ste_threshold", 0.0))
        return apply_dual_lora(model, ac)
    elif method in ("lora", "qlora"):
        if method == "qlora":
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        ac = LoRABaselineConfig(
            r=lcfg["r"], lora_alpha=lcfg["lora_alpha"],
            lora_dropout=lcfg.get("lora_dropout", 0.05),
            target_modules=lcfg["target_modules"])
        return apply_lora_baseline(model, ac)
    elif method == "full_ft":
        return model
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: dual_lora | lora | qlora | full_ft")


def train(config, debug=False, output_dir_override=None, seed_override=None):
    # ── Set PYTORCH_CUDA_ALLOC_CONF to reduce memory fragmentation ──
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision('high')

    # ── Auto-detect hardware ────────────────────────────────
    hw = detect_hardware()
    build_accelerate_config(hw, output_path="configs/accelerate_runtime.yaml")

    set_seed(config["training"].get("seed", 42))
    # CLI seed overrides config seed (used by run_experiments.sh)
    if seed_override is not None:
        config["training"]["seed"] = seed_override
        set_seed(seed_override)
        print(f"[Trainer] Seed override : {seed_override}")
    logger = setup_logging(config["experiment"]["name"])
    run_name = config["training"].get("run_name", config["experiment"]["name"])
    out_dir = output_dir_override or os.path.join(config["training"]["output_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    log_config(logger, config)
    save_experiment_config(config, out_dir)

    model, tok = load_model_and_tokenizer(config, hw)
    model = apply_adapter(model, config)
    # torch.compile est incompatible avec Dual LoRA (SignSTE custom) en distribué
    if hasattr(torch, 'compile') and config["experiment"]["method"] != "dual_lora":
        print("[Trainer] Enabling torch.compile...")
        model = torch.compile(model)

    model.enable_input_require_grads()      #avant gradient_checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False   

    #model = apply_adapter(model, config)
    #model.gradient_checkpointing_enable()


    tcfg = config["training"]
    bs = tcfg.get("per_device_train_batch_size") or get_optimal_batch_size(hw)
    num_epochs = 1 if debug else tcfg["num_epochs"]
    num_samples = 500 if debug else None

    train_ds, val_ds = load_metamath_dataset(
        tokenizer=tok,
        max_seq_length=config["data"]["max_seq_length"],
        num_train_samples=num_samples,
        val_ratio=config["data"].get("val_ratio", 0.02),
        seed=tcfg.get("seed", 42),
    )

    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True, pad_to_multiple_of=8)

    use_wandb = config["logging"].get("use_wandb", False) and not debug

    # ── Compute warmup_steps instead of warmup_ratio (avoids deprecation warning) ──
    # Estimate total steps: (train_size / (bs * grad_accum * n_gpus)) * epochs
    n_gpus = max(hw.n_gpus, 1)
    grad_accum = tcfg.get("gradient_accumulation_steps", 4)
    steps_per_epoch = len(train_ds) // (bs * grad_accum * n_gpus)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * tcfg.get("warmup_ratio", 0.05))
    warmup_steps = max(warmup_steps, 10)  # at least 10 steps
    print(f"[Trainer] warmup_steps={warmup_steps} (from ratio {tcfg.get('warmup_ratio', 0.05)}, total={total_steps} steps)")

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,           # replaces deprecated warmup_ratio
        bf16=hw.has_bf16,
        fp16=(not hw.has_bf16 and hw.n_gpus > 0),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        optim=tcfg.get("optimizer", "adamw_torch"),
        weight_decay=tcfg.get("weight_decay", 0.01),
        eval_strategy=tcfg.get("eval_strategy", "epoch"),
        eval_steps=tcfg.get("eval_steps", None) if tcfg.get("eval_strategy") == "steps" else None,
        save_strategy=tcfg.get("save_strategy", "epoch"),
        save_steps=tcfg.get("save_steps", 500) if tcfg.get("save_strategy") == "steps" else 500,
        save_total_limit=tcfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=tcfg.get("logging_steps", 50),
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        dataloader_pin_memory=(hw.n_gpus > 0),
        dataloader_num_workers=min(4, max(1, os.cpu_count() // 2)),
        remove_unused_columns=False,
        seed=tcfg.get("seed", 42),
    )

    callbacks = [] if debug else [EarlyStoppingCallback(early_stopping_patience=2)]

    # Handle transformers 4.x vs 5.x Trainer API
    import transformers as _tf
    _trainer_kwargs = dict(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, callbacks=callbacks,
    )
    if int(_tf.__version__.split(".")[0]) >= 5:
        _trainer_kwargs["processing_class"] = tok
    else:
        _trainer_kwargs["tokenizer"] = tok

    trainer = Trainer(**_trainer_kwargs)
    logger.info(f"Training: {run_name} | {config['experiment']['method']} | {hw.n_gpus} GPU(s)")

    # ── Resume from checkpoint if one exists ───────────────────
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = None
    if os.path.isdir(out_dir) and not debug:
        last_checkpoint = get_last_checkpoint(out_dir)
        if last_checkpoint:
            logger.info(f"[Trainer] Checkpoint détecté → reprise depuis {last_checkpoint}")
            print(f"[Trainer] Reprise depuis : {last_checkpoint}")
        else:
            logger.info("[Trainer] Aucun checkpoint — démarrage from scratch")

    trainer.train(resume_from_checkpoint=last_checkpoint)



    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    # Save Dual LoRA adapter weights separately for correct evaluation loading
    if config["experiment"]["method"] == "dual_lora":
        adapter_path = os.path.join(out_dir, "dual_lora_adapters.pt")
        adapter_state = {
            k: v.cpu() for k, v in model.state_dict().items()
            if "dual_lora" in k or "lora_A" in k or "lora_B" in k
               or "lora_C" in k or "lora_D" in k
        }
        torch.save(adapter_state, adapter_path)
        logger.info(f"Dual LoRA adapter weights saved → {adapter_path} ({len(adapter_state)} tensors)")

    logger.info(f"Done → {out_dir}")
    return out_dir


def push_to_hub(checkpoint_dir, repo_id, private=True):
    """Push trained model to HuggingFace Hub."""
    print(f"[Hub] Pushing {checkpoint_dir} → {repo_id}")
    tok = AutoTokenizer.from_pretrained(checkpoint_dir)
    import transformers as _tf
    dtype_key = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **{dtype_key: torch.bfloat16})
    tok.push_to_hub(repo_id, private=private)
    model.push_to_hub(repo_id, private=private)
    cfg = os.path.join(checkpoint_dir, "experiment_config.json")
    if os.path.exists(cfg):
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=cfg, path_in_repo="experiment_config.json", repo_id=repo_id)
    print(f"[Hub] Done → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)          # <-- déplacé ICI
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--hub_repo_id", type=str, default=None)
    p.add_argument("--hub_private", action="store_true", default=True)
    a = p.parse_args()
    if a.push_to_hub:
        push_to_hub(a.checkpoint_path, a.hub_repo_id, a.hub_private)
    else:
        train(load_config(a.config), debug=a.debug,
              output_dir_override=a.output_dir, seed_override=a.seed)

