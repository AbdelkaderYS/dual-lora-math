import logging
import os
import sys
from typing import Optional


def setup_logging(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def log_config(logger: logging.Logger, config: dict):
    import yaml
    logger.info(f"Config:\n{yaml.dump(config, default_flow_style=False)}")


def init_wandb(config: dict):
    try:
        import wandb
        wandb.init(
            project=config.get("logging", {}).get("project", "dual-lora-math"),
            config=config,
        )
    except ImportError:
        print("[W&B] wandb not installed — skipping")
    except Exception as e:
        print(f"[W&B] Error: {e}")
