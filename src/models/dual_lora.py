import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DualLoRAConfig:
    r1: int = 16
    r2: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    init_std: float = 0.02
    sign_ste_threshold: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        x = ctx.saved_tensors[0]
        grad = grad_output.clone()
        grad[grad_output.abs() <= ctx.threshold] = 0.0
        return grad, None


def sign_ste(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    return SignSTE.apply(x, threshold)


class DualLoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r1: int = 16,
        r2: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        init_std: float = 0.02,
        sign_ste_threshold: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r1 = r1
        self.r2 = r2
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / math.sqrt(r1 * r2)
        self.sign_ste_threshold = sign_ste_threshold

        self.lora_A = nn.Linear(in_features, r1, bias=False)
        self.lora_B = nn.Linear(r1, out_features, bias=False)
        self.lora_C = nn.Linear(in_features, r2, bias=False)
        self.lora_D = nn.Linear(r2, out_features, bias=False)

        if lora_dropout > 0.0:
            self.dropout = nn.Dropout(lora_dropout)
        else:
            self.dropout = nn.Identity()

        self._init_weights(init_std)

    def _init_weights(self, std: float = 0.02):
        for layer in [self.lora_A, self.lora_B, self.lora_C, self.lora_D]:
            nn.init.normal_(layer.weight, mean=0.0, std=std)

    def compute_delta_w(self, x: torch.Tensor) -> torch.Tensor:
        x_drop = self.dropout(x)
        magnitude = F.relu(self.lora_B(self.lora_A(x_drop)))
        direction = sign_ste(self.lora_D(self.lora_C(x_drop)), self.sign_ste_threshold)
        return self.scaling * magnitude * direction

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        return base_output + self.compute_delta_w(x)


class DualLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, config: DualLoRAConfig):
        super().__init__()
        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad = False
        self.dual_lora = DualLoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            r1=config.r1,
            r2=config.r2,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            init_std=config.init_std,
            sign_ste_threshold=config.sign_ste_threshold,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        return self.dual_lora(x, base_out)


def apply_dual_lora(model: nn.Module, config: DualLoRAConfig) -> nn.Module:
    replaced = 0
    for name, module in list(model.named_modules()):
        for target in config.target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                child_name = parts[-1]
                setattr(parent, child_name, DualLoRALinear(module, config))
                replaced += 1
                break

    print(f"[DualLoRA] Replaced {replaced} Linear layers with DualLoRALinear.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[DualLoRA] Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    return model


def get_trainable_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100 * trainable / total,
    }
