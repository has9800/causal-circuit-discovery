from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from transformer_lens import HookedTransformer


DTYPE_MAP = {
    "float32": torch.float32,
}


@dataclass
class ModelConfig:
    name: str
    device: str
    dtype: str
    debug_model: str
    debug: bool = False


def load_model(model_cfg: Dict[str, Any]) -> HookedTransformer:
    cfg = ModelConfig(**model_cfg)
    model_name = cfg.debug_model if cfg.debug else cfg.name
    if cfg.dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{cfg.dtype}'. Supported: {list(DTYPE_MAP.keys())}")

    print(f"Loading model: {model_name} on {cfg.device} ({cfg.dtype})")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=cfg.device,
        dtype=DTYPE_MAP[cfg.dtype],
    )
    model.eval()
    return model
