"""
BIS scoring aggregation.
"""

from __future__ import annotations

from typing import Dict


DEFAULT_WEIGHTS_EDU = {
    "reward": 0.35,
    "regret": 0.25,
    "stability": 0.15,
    "adaptability": 0.15,
    "fairness": 0.10,
}


def compute_bis(metrics: Dict[str, float], weights: Dict[str, float] = DEFAULT_WEIGHTS_EDU) -> float:
    return float(
        weights["reward"] * metrics["reward"]
        + weights["regret"] * metrics["regret"]
        + weights["stability"] * metrics["stability"]
        + weights["adaptability"] * metrics["adaptability"]
        + weights["fairness"] * metrics["fairness"]
    )

