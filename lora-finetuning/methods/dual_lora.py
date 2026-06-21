import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


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
    warmup_steps: int = 0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        grad = torch.clamp(grad_output, -1.0, 1.0)
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
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r1 = r1
        self.r2 = r2
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / math.sqrt(r1 * r2)
        self.sign_ste_threshold = sign_ste_threshold
        self.warmup_steps = warmup_steps
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

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
        warmup_scale = 1.0
        if self.warmup_steps > 0:
            warmup_scale = min(1.0, self._step.item() / self.warmup_steps)
        x_drop = self.dropout(x)
        magnitude = F.relu(self.lora_B(self.lora_A(x_drop)))
        direction = sign_ste(self.lora_D(self.lora_C(x_drop)), self.sign_ste_threshold)
        return warmup_scale * self.scaling * magnitude * direction

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        return base_output + self.compute_delta_w(x)

    def set_step(self, step: int):
        self._step.fill_(max(0, int(step)))


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
            warmup_steps=config.warmup_steps,
        )
        self.dual_lora.to(device=self.base_linear.weight.device, dtype=self.base_linear.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        return self.dual_lora(x, base_out)

    def set_step(self, step: int):
        self.dual_lora.set_step(step)

    def merge_weights(self):
        with torch.no_grad():
            BA = self.dual_lora.lora_B.weight @ self.dual_lora.lora_A.weight
            DC = self.dual_lora.lora_D.weight @ self.dual_lora.lora_C.weight
            delta_W = self.dual_lora.scaling * F.relu(BA) * torch.sign(DC)
            self.base_linear.weight.add_(delta_W.to(self.base_linear.weight.dtype))

    def unmerge_weights(self):
        with torch.no_grad():
            BA = self.dual_lora.lora_B.weight @ self.dual_lora.lora_A.weight
            DC = self.dual_lora.lora_D.weight @ self.dual_lora.lora_C.weight
            delta_W = self.dual_lora.scaling * F.relu(BA) * torch.sign(DC)
            self.base_linear.weight.sub_(delta_W.to(self.base_linear.weight.dtype))

    def unwrap(self) -> nn.Linear:
        self.merge_weights()
        return self.base_linear


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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[DualLoRA] Replaced {replaced} Linear layers | "
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    return model


def apply_dual_lora_4bit(model: nn.Module, config: DualLoRAConfig) -> nn.Module:
    try:
        from bitsandbytes.nn import Linear4bit
        accepted = (nn.Linear, Linear4bit)
    except ImportError:
        accepted = (nn.Linear,)
    replaced = 0
    for name, module in list(model.named_modules()):
        for target in config.target_modules:
            if not name.endswith(target):
                continue
            if not isinstance(module, accepted):
                continue
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]
            setattr(parent, child_name, DualLoRALinear(module, config))
            replaced += 1
            break
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[DualLoRA-4bit] Replaced {replaced} layers | "
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    return model


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


def get_trainable_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100 * trainable / total,
    }


def set_model_step(model: nn.Module, step: int):
    for module in model.modules():
        if hasattr(module, "set_step"):
            module.set_step(step)
