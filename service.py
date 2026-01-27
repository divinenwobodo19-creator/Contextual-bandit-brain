"""
Standalone LinUCB Brain service exposing a minimal interface.

Interface:
- select_action(context) -> int
- update(action, reward, context) -> None

Intent:
- Provide a production-ready API independent of any UI or LMS.
- Maintain per-arm parameters with adjustable alpha.
- Ensure immediate online learning from rewards.
"""

from __future__ import annotations

from typing import List
import numpy as np

from actions import LinUCBAction


class LinUCBService:
    """
    Minimal LinUCB service class.

    Behavior:
    - Holds independent per-arm LinUCB parameters.
    - Computes UCB for a given context and returns the best arm.
    - Updates the specified arm using the provided reward and context.
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
        self._actions: List[LinUCBAction] = [LinUCBAction(self._d) for _ in range(self._num_actions)]

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def d(self) -> int:
        return self._d

    @property
    def alpha(self) -> float:
        return self._alpha

    def select_action(self, context: np.ndarray) -> int:
        """
        Select the action with maximal UCB value for the given context.
        """
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        scores = []
        for a in self._actions:
            _, _, ucb = a.score(x, self._alpha)
            scores.append(ucb)
        return int(np.argmax(scores))

    def update(self, action: int, reward: float, context: np.ndarray) -> None:
        """
        Update the specified action immediately using reward and context.

        A ← A + x xᵀ
        b ← b + r x, with r clamped to [0, 1]
        """
        if not (0 <= action < self._num_actions):
            raise ValueError(f"action {action} out of range [0, {self._num_actions})")
        if not np.isfinite(reward):
            raise ValueError("reward must be a finite number")
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        r = float(np.clip(reward, 0.0, 1.0))
        self._actions[action].update(x, r)

    def reset(self) -> None:
        """Reset all arms to their priors."""
        for a in self._actions:
            a.reset()

