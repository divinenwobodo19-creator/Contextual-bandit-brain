from __future__ import annotations
from typing import Optional
from .decision_service import create_app
from ..brain.config import LinUCBConfig


def build_app(num_actions: int, d: int, alpha: float = 0.5):
    if create_app is None:
        raise RuntimeError("FastAPI not available")
    cfg = LinUCBConfig(num_actions=num_actions, d=d, alpha=alpha)
    return create_app(cfg)
