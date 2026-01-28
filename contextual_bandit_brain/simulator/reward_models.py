from __future__ import annotations
from typing import Callable
import numpy as np


def logistic_reward(theta: np.ndarray, x: np.ndarray, noise_std: float, rng: np.random.Generator) -> float:
    noise = rng.normal(0.0, noise_std)
    raw = float(x.dot(theta) + noise)
    r = 1.0 / (1.0 + np.exp(-raw))
    return float(np.clip(r, 0.0, 1.0))


def linear_clipped_reward(theta: np.ndarray, x: np.ndarray, noise_std: float, rng: np.random.Generator) -> float:
    noise = rng.normal(0.0, noise_std)
    r = float(x.dot(theta) + noise)
    return float(np.clip(r, 0.0, 1.0))
