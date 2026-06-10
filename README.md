# Dual LoRA for Mathematical Reasoning

> First application of Dual LoRA to mathematical reasoning using LLEMMA-7B.
> Paper contribution: novel weight decomposition (magnitude × direction) for PEFT.

## Formula

```
ΔW = (α / √(r₁·r₂)) × ReLU(B·A) ⊙ Sign(D·C)
```

| Component | Role |
|-----------|------|
| `ReLU(BA)` | Magnitude group : controls update scale |
| `Sign(DC)` | Direction group : controls update sign via SignSTE |

## Quick start (VS Code → HPCC)

```bash
# 1. Setup (once)
bash scripts/setup_env.sh

# 2. Debug local (no GPU needed)
bash scripts/train.sh configs/dual_lora_config.yaml --debug

# 3. Run all experiments for the paper (3 seeds × 2 methods)
bash scripts/run_experiments.sh

# 4. After all jobs complete — generate paper table
python scripts/paper_results.py

# 5. Deploy best checkpoint to HuggingFace
bash scripts/deploy_to_hub.sh \
    experiments/results/dual_lora_seed42 \
    yourname/dual-lora-llemma-7b
```

## Switch method or model: one line in the YAML

```yaml
# configs/dual_lora_config.yaml
experiment:
  method: "dual_lora"   # or: lora | qlora | full_ft

model:
  name: "EleutherAI/llemma_7b"  # or any HF CausalLM
```

```bash
sbatch scripts/train.sh configs/dual_lora_config.yaml
sbatch scripts/train.sh configs/full_ft_config.yaml
sbatch scripts/train.sh configs/qlora_config.yaml
```

## Project structure

```
dual-lora-math/
├── configs/
│   ├── base_config.yaml           # shared hyperparams
│   ├── dual_lora_config.yaml      # Dual LoRA (this paper)
│   ├── lora_baseline_config.yaml  # LoRA standard baseline
│   ├── qlora_config.yaml          # QLoRA (memory-efficient)
│   ├── full_ft_config.yaml        # Full fine-tuning
│   └── accelerate_config.yaml     # fallback — auto-generated at runtime
│
├── src/
│   ├── models/
│   │   ├── dual_lora.py           # DualLoRALayer, SignSTE, apply_dual_lora
│   │   └── lora_baseline.py       # Standard LoRA via PEFT
│   ├── training/
│   │   ├── trainer.py             # Modular trainer (all methods)
│   │   └── data_loader.py         # MetaMathQA + train/val split
│   ├── evaluation/
│   │   ├── evaluator.py           # GSM8K + MATH runner
│   │   ├── answer_extraction.py   # Robust answer parsing
│   │   └── metrics.py             # Accuracy + per-category
│   └── utils/
│       ├── hardware.py            # Auto GPU detection + accelerate config
│       ├── logging_utils.py       # W&B + file logging
│       └── checkpoint.py         # Save/load configs
│
├── scripts/
│   ├── train.sh                   # Generic SLURM launcher (all methods)
│   ├── evaluate.sh                # Evaluation + optional HF push
│   ├── run_experiments.sh         # Launch all paper runs (3 seeds × 2 methods)
│   ├── deploy_to_hub.sh           # Full HuggingFace deployment
│   ├── generate_model_card.py     # Auto model card with results
│   ├── paper_results.py           # LaTeX table + CSV from all runs
│   └── setup_env.sh               # One-shot environment setup
│
├── tests/
│   ├── test_dual_lora.py          # Unit tests: SignSTE, DualLoRALayer
│   └── test_answer_extraction.py  # Unit tests: parsing, normalization
│
└── notebooks/
    └── analysis.ipynb             # Plots and analysis
```

## Hardware auto-detection

`src/utils/hardware.py` detects at startup:
- Number of GPUs on the allocated node
- BF16 support (Ampere+)
- Flash Attention 2 availability
- Optimal batch size from VRAM

No manual configuration needed. Works on 1 GPU, 2 GPUs, 4 GPUs, or CPU.

## Datasets

| Dataset | Split | Size | Use |
|---------|-------|------|-----|
| MetaMathQA | train (98%) | ~387K | Training |
| MetaMathQA | val (2%) | ~7.9K | Early stopping |
| GSM8K | test | 1.3K | Benchmark |
| MATH | test | 5K | Benchmark |

## Citation

```bibtex
@misc{dual_lora_math_2025,
  title={Dual LoRA for Mathematical Reasoning},
  author={...},
  year={2026}
}
```
