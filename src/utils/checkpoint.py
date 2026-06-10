import json
import os
from typing import Any, Dict, List, Optional


def save_experiment_config(config: dict, output_dir: str):
    path = os.path.join(output_dir, "experiment_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_experiment_config(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, "experiment_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def list_checkpoints(results_dir: str) -> List[str]:
    if not os.path.exists(results_dir):
        return []
    return sorted(os.listdir(results_dir))
