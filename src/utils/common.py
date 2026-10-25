"""Shared utilities: config loading, seeding, device, logging."""
from __future__ import annotations
import os
import json
import random
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
class Config(dict):
    """dict with attribute access and dotted-key override support.

    Nested dicts are converted to Config *in place* at construction, so
    `cfg.model.video.backbone = 'x'` mutates the real stored dict (not a
    throwaway wrapper). This correctness matters: silent no-op assignment
    would make every in-code override (epochs, backbone, ...) ineffective.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, Config):
                self[k] = Config(v)

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = Config(v) if (isinstance(v, dict) and not isinstance(v, Config)) else v

    def set_dotted(self, dotted: str, value: Any):
        """cfg.set_dotted('train.epochs', 10) with type coercion."""
        keys = dotted.split(".")
        d = self
        for k in keys[:-1]:
            d = d[k]
        old = d.get(keys[-1])
        # coerce the string from CLI to the existing field's type
        if isinstance(old, bool):
            value = str(value).lower() in ("1", "true", "yes")
        elif isinstance(old, int):
            value = int(value)
        elif isinstance(old, float):
            value = float(value)
        d[keys[-1]] = value


def to_plain(obj):
    """Recursively convert Config/nested dicts to plain dict (picklable, no custom
    class). Used before torch.save so checkpoints load under weights_only=True."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_plain(v) for v in obj)
    return obj


def load_config(path: str = "configs/default.yaml", overrides: list[str] | None = None) -> Config:
    with open(path) as f:
        cfg = Config(yaml.safe_load(f))
    for ov in overrides or []:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        cfg.set_dotted(k.strip(), v.strip())
    return cfg


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # determinism-friendly (small perf cost)
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def get_device(pref: str = "auto"):
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    if pref == "mps" or (pref == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = "vq", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger
