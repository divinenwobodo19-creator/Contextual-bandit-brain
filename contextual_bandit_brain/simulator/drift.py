"""
Environment drift utilities.

Simulates non-stationarity by changing hidden arm parameters.
"""

from __future__ import annotations

import numpy as np


def apply_drift(theta_true: np.ndarray, scale: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Return drifted parameters by mixing current with random direction.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, 1.0, size=theta_true.shape).astype(float)
    noise /= np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-12)
    new = (1.0 - scale) * theta_true + scale * noise
    return new

