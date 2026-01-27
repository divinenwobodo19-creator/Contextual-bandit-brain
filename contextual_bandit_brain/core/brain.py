"""
LinUCB decision engine.

Interface:
- select_action(context) -> int
- update(action, reward, context) -> None
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np

from .arm import LinUCBArm
from .linucb import score_actions


class LinUCBBrian:  # intentional name in doc but use LinUCBBrain class below
    pass


class LinUCBBrain:
    """
    Standalone LinUCB decision engine.
    """

    def __init__(self, num_actions: int, alpha: float, d: int) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if d <= 0:
            raise ValueError("feature dimension d must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self._num_actions = int(num_actions)
        self._alpha = float(alpha)
        self._d = int(d)
        self._arms: List[LinUCBArm] = [LinUCBArm(self._d) for _ in range(self._num_actions)]
        self._last_decision: Optional[Dict[str, Any]] = None

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def d(self) -> int:
        return self._d

    @property
    def alpha(self) -> float:
        return self._alpha

    def reset(self) -> None:
        for a in self._arms:
            a.reset()
        self._last_decision = None

    def select_action(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        est, unc, ucb = score_actions(self._arms, x, self._alpha)
        chosen = int(np.argmax(ucb))
        best_est = int(np.argmax(est))
        mode = "exploitation" if chosen == best_est else "exploration"
        self._last_decision = {
            "action": chosen,
            "estimated_reward": float(est[chosen]),
            "uncertainty": float(unc[chosen]),
            "ucb": float(ucb[chosen]),
            "mode": mode,
            "context": x.tolist(),
        }
        return chosen

    def update(self, action: int, reward: float, context: np.ndarray) -> None:
        if not (0 <= action < self._num_actions):
            raise ValueError(f"action {action} out of range [0, {self._num_actions})")
        if not np.isfinite(reward):
            raise ValueError("reward must be a finite number")
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        r = float(np.clip(reward, 0.0, 1.0))
        self._arms[action].update(x, r)

    def explain_last(self) -> Dict[str, Any]:
        if self._last_decision is None:
            raise RuntimeError("No decision to explain")
        return dict(self._last_decision)

