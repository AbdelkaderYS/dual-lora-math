from dataclasses import dataclass, field
from typing import List
from peft import LoraConfig, TaskType, get_peft_model
import torch.nn as nn


@dataclass
class LoRABaselineConfig:
    r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"


def apply_lora_baseline(model: nn.Module, config: LoRABaselineConfig) -> nn.Module:
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
