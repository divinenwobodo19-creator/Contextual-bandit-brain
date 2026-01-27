"""
Action-level LinUCB parameters and operations.

Intent:
- Encapsulate per-action state for LinUCB (A and b).
- Provide mathematically clear operations: parameter estimation, UCB scoring, update, reset.
- Remain domain-agnostic: only numeric vectors and rewards are handled.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np


class LinUCBAction:
    """
    Per-action model for LinUCB.

    Design:
    - A ∈ R^{d×d} initialized to I_d to ensure invertibility at t=0.
    - b ∈ R^{d} initialized to 0, accumulating reward-weighted contexts.
    - θ̂ derived by solving A θ̂ = b for numerical stability.
    - Uncertainty computed via xᵀ A⁻¹ x using a linear solve, avoiding explicit matrix inverse.
    """

    def __init__(self, d: int) -> None:
        if d <= 0:
            raise ValueError("Feature dimension d must be positive")
        self._d = int(d)
        self._A = np.eye(self._d, dtype=float)
        self._b = np.zeros(self._d, dtype=float)

    @property
    def d(self) -> int:
        return self._d

    def reset(self) -> None:
        """Reinitialize A and b to their priors."""
        self._A = np.eye(self._d, dtype=float)
        self._b = np.zeros(self._d, dtype=float)

    def theta(self) -> np.ndarray:
        """
        Return θ̂ solving A θ̂ = b.

        Intent: provide the maximum-likelihood estimate under a ridge-like prior from A=I.
        """
        return np.linalg.solve(self._A, self._b)

    def score(self, x: np.ndarray, alpha: float) -> Tuple[float, float, float]:
        """
        Compute LinUCB components for context x.

        Returns:
        - estimated_reward: xᵀ θ̂
        - uncertainty: α √(xᵀ A⁻¹ x)
        - ucb: estimated_reward + uncertainty
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"Context dimension {x.shape[0]} != d={self._d}")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")

        theta_hat = np.linalg.solve(self._A, self._b)
        estimated_reward = float(x.dot(theta_hat))
        Ax_inv_x = x.dot(np.linalg.solve(self._A, x))
        uncertainty = float(alpha * np.sqrt(max(Ax_inv_x, 0.0)))
        ucb = estimated_reward + uncertainty
        return estimated_reward, uncertainty, ucb

    def update(self, x: np.ndarray, reward: float) -> None:
        """
        Update A and b with context x and scalar reward.
        A ← A + x xᵀ
        b ← b + r x
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"Context dimension {x.shape[0]} != d={self._d}")
        self._A += np.outer(x, x)
        self._b += float(reward) * x

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable snapshot of this action's state."""
        return {
            "d": self._d,
            "A": self._A.tolist(),
            "b": self._b.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LinUCBAction:
        """Rehydrate an action from a serialized snapshot."""
        d = int(data["d"])
        obj = cls(d)
        obj._A = np.asarray(data["A"], dtype=float)
        obj._b = np.asarray(data["b"], dtype=float)
        return obj

