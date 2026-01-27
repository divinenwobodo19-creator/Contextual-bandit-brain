"""
LinUCB arm maintaining per-action parameters.

State:
- A ∈ R^{d×d}, initialized to I_d
- b ∈ R^{d}, initialized to 0
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np


class LinUCBArm:
    """
    Per-arm LinUCB parameters and operations.
    """

    def __init__(self, d: int) -> None:
        if d <= 0:
            raise ValueError("feature dimension d must be positive")
        self._d = int(d)
        self._A = np.eye(self._d, dtype=float)
        self._b = np.zeros(self._d, dtype=float)

    @property
    def d(self) -> int:
        return self._d

    def reset(self) -> None:
        self._A = np.eye(self._d, dtype=float)
        self._b = np.zeros(self._d, dtype=float)

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self._A, self._b)

    def score(self, x: np.ndarray, alpha: float) -> Tuple[float, float, float]:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        theta_hat = np.linalg.solve(self._A, self._b)
        est = float(x.dot(theta_hat))
        Ax_inv_x = float(x.dot(np.linalg.solve(self._A, x)))
        unc = float(alpha * np.sqrt(max(Ax_inv_x, 0.0)))
        ucb = est + unc
        return est, unc, ucb

    def update(self, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        self._A += np.outer(x, x)
        self._b += float(reward) * x

    def to_dict(self) -> Dict[str, Any]:
        return {"d": self._d, "A": self._A.tolist(), "b": self._b.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LinUCBArm:
        d = int(data["d"])
        obj = cls(d)
        obj._A = np.asarray(data["A"], dtype=float)
        obj._b = np.asarray(data["b"], dtype=float)
        return obj

