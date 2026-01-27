"""
Normalization utilities for BIS metrics.
"""

from __future__ import annotations

import numpy as np


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def safe_ratio(num: float, den: float) -> float:
    if den <= 1e-12:
        return 0.0
    return float(num / den)

