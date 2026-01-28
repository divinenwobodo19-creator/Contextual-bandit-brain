from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LinUCBConfig:
    num_actions: int
    d: int
    alpha: float = 0.5
