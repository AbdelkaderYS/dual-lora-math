import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareProfile:
    n_gpus: int = 0
    gpu_names: list = field(default_factory=list)
    total_vram_gb: float = 0.0
    has_bf16: bool = False
    has_flash_attn: bool = False
    torch_dtype: str = "float32"
    device: str = "cpu"
    n_gpus_effective: int = 0
    distributed: bool = False


def detect_hardware() -> HardwareProfile:
    try:
        import torch
        if not torch.cuda.is_available():
            return _cpu_profile()
        n_gpus = torch.cuda.device_count()
        gpu_names = []
        total_vram = 0
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            gpu_names.append(props.name)
            total_vram += props.total_memory
        total_vram_gb = total_vram / (1024**3)
        has_bf16 = torch.cuda.is_bf16_supported()
        has_flash = _flash_sdpa_available()
        profile = HardwareProfile(
            n_gpus=n_gpus,
            gpu_names=gpu_names,
            total_vram_gb=total_vram_gb,
            has_bf16=has_bf16,
            has_flash_attn=has_flash,
            torch_dtype="bfloat16" if has_bf16 else "float16",
            device="cuda",
            n_gpus_effective=n_gpus,
            distributed=n_gpus > 1,
        )
        _log_profile(profile)
        return profile
    except ImportError:
        return _cpu_profile()


def _cpu_profile() -> HardwareProfile:
    print("[Hardware] No GPU detected — using CPU")
    return HardwareProfile(
        n_gpus=0, device="cpu", torch_dtype="float32",
        n_gpus_effective=1, distributed=False,
    )


def _flash_sdpa_available() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        return torch.backends.cuda.flash_sdp_enabled()
    except (ImportError, AttributeError):
        return False


def _log_profile(profile: HardwareProfile):
    print(f"[Hardware] Detected {profile.n_gpus} GPU(s):")
    for i, name in enumerate(profile.gpu_names):
        print(f"  GPU {i}: {name}")
    print(f"[Hardware] Total VRAM    : {profile.total_vram_gb:.0f} GB")
    print(f"[Hardware] BF16 support  : {profile.has_bf16}")
    print(f"[Hardware] Flash Attn    : {profile.has_flash_attn}")
    print(f"[Hardware] dtype         : {profile.torch_dtype}")
    print(f"[Hardware] Distributed   : {profile.distributed} ({profile.n_gpus} processes)")


def build_accelerate_config(profile: HardwareProfile, output_path: str = "configs/accelerate_runtime.yaml"):
    import yaml
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU" if profile.distributed else "NO",
        "gpu_ids": "all" if profile.n_gpus > 0 else None,
        "main_training_function": "main",
        "mixed_precision": "bf16" if profile.has_bf16 else "fp16",
        "num_processes": profile.n_gpus if profile.n_gpus > 0 else 1,
        "use_cpu": profile.device == "cpu",
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[Hardware] Accelerate config written to {output_path}")


def get_optimal_batch_size(profile: HardwareProfile, model_size_b: float = 7.0) -> int:
    if profile.n_gpus == 0:
        return 2
    vram_per_gpu = profile.total_vram_gb / profile.n_gpus
    if vram_per_gpu >= 80:
        return 24
    elif vram_per_gpu >= 40:
        return 12
    elif vram_per_gpu >= 16:
        return 8
    else:
        return 4
