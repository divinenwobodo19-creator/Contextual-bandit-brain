"""
Lightweight simulator for validating the LinUCB brain.

Intent:
- Generate random contexts and hidden linear reward models per action.
- Produce scalar rewards in [0, 1] via a logistic mapping with noise.
- Remain application-agnostic and reproducible via seeds.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ContextualBanditSimulator:
    """
    Synthetic environment for contextual bandits.

    Construction:
    - θ_true[a] ∈ R^d for each action a, drawn from N(0, I) and optionally scaled.
    - Contexts x ∈ R^d drawn from N(0, I).
    - Reward r ∈ [0, 1] computed as σ(xᵀ θ_true[a] + ε), where ε ~ N(0, noise_std²).
    """

    def __init__(self, d: int, num_actions: int, noise_std: float = 0.1, seed: Optional[int] = None) -> None:
        if d <= 0:
            raise ValueError("feature dimension d must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        self._d = int(d)
        self._num_actions = int(num_actions)
        self._noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)
        # Hidden parameters per action; scaled to moderate signal-to-noise
        self._theta_true = self._rng.normal(loc=0.0, scale=1.0, size=(self._num_actions, self._d)).astype(float)
        self._theta_true /= np.maximum(np.linalg.norm(self._theta_true, axis=1, keepdims=True), 1e-12)
        self._theta_true *= 2.0
        self._last_context: Optional[np.ndarray] = None

    @property
    def d(self) -> int:
        return self._d

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset RNG and discard any stored context."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_context = None

    def generate_context(self) -> np.ndarray:
        """Draw a random context x ∈ R^d from N(0, I)."""
        x = self._rng.normal(loc=0.0, scale=1.0, size=self._d).astype(float)
        self._last_context = x
        return x

    def get_reward(self, action: int, context: np.ndarray) -> float:
        """
        Compute reward r ∈ [0, 1] for a given action and context.
        r = σ(xᵀ θ_true[a] + ε), clipped to [0, 1].
        """
        if not (0 <= action < self._num_actions):
            raise ValueError(f"action {action} out of range [0, {self._num_actions})")
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        noise = self._rng.normal(loc=0.0, scale=self._noise_std)
        raw = float(x.dot(self._theta_true[action]) + noise)
        r = 1.0 / (1.0 + np.exp(-raw))
        return float(np.clip(r, 0.0, 1.0))

