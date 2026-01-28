from __future__ import annotations
from typing import Dict
from ..bis.scoring import compute_bis, DEFAULT_WEIGHTS_EDU


def bis(metrics: Dict[str, float]) -> float:
    return float(compute_bis(metrics, DEFAULT_WEIGHTS_EDU))
