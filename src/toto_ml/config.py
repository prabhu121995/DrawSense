"""Configuration loading, merging and hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("TOTO_ML_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

N_NUMBERS = 49
PICK = 6
BASELINE_P = PICK / N_NUMBERS  # 0.12245 theoretical marginal probability per number


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path or os.environ.get("TOTO_ML_CONFIG", DEFAULT_CONFIG_PATH))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return deep_merge(cfg, overrides)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
