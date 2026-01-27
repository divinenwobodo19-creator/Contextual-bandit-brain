"""
LinUCB algorithm utilities.

Provides action scoring across a set of arms.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .arm import LinUCBArm


def score_actions(arms: List[LinUCBArm], x: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute estimated rewards, uncertainties, and UCBs for all arms.
    """
    est = []
    unc = []
    ucb = []
    for a in arms:
        e, u, s = a.score(x, alpha)
        est.append(e)
        unc.append(u)
        ucb.append(s)
    return np.asarray(est, dtype=float), np.asarray(unc, dtype=float), np.asarray(ucb, dtype=float)

