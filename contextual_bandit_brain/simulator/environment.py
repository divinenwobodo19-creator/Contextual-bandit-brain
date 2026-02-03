"""
Synthetic environment for contextual bandit evaluation.

Hidden per-arm parameters define reward tendencies.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from .student_model import StudentModel
from .drift import apply_drift
from .reward_model import expected_reward as exp_logistic
from .reward_model import sample_reward as samp_logistic
from .reward_model import expected_reward_linear as exp_linear
from .reward_model import sample_reward_linear as samp_linear


class Environment:
    """
    Stationary and non-stationary environment with logistic reward mapping.
    """

    def __init__(self, d: int, num_actions: int, noise_std: float = 0.1, seed: Optional[int] = None, reward_mode: str = "logistic") -> None:
        if d <= 0:
            raise ValueError("feature dimension d must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if reward_mode not in ("logistic", "linear"):
            raise ValueError("reward_mode must be 'logistic' or 'linear'")
        self._d = int(d)
        self._num_actions = int(num_actions)
        self._noise_std = float(noise_std)
        try:
            self._rng = np.random.default_rng(seed)
        except AttributeError:
            self._rng = np.random.RandomState(seed)
        self._student = StudentModel(d=d, seed=seed)
        self._theta_true = self._rng.normal(0.0, 1.0, size=(self._num_actions, self._d)).astype(float)
        self._theta_true /= np.maximum(np.linalg.norm(self._theta_true, axis=1, keepdims=True), 1e-12)
        self._theta_true *= 2.0
        self._reward_mode = reward_mode

    @property
    def d(self) -> int:
        return self._d

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def context(self) -> np.ndarray:
        return self._student.sample_context()

    def reward(self, action: int, context: np.ndarray) -> float:
        if not (0 <= action < self._num_actions):
            raise ValueError(f"action {action} out of range [0, {self._num_actions})")
        x = np.asarray(context, dtype=float).reshape(-1)
        if self._reward_mode == "logistic":
            return float(samp_logistic(self._theta_true[action], x, self._noise_std, self._rng))
        else:
            return float(samp_linear(self._theta_true[action], x, self._noise_std, self._rng))

    def expected_reward(self, action: int, context: np.ndarray) -> float:
        if not (0 <= action < self._num_actions):
            raise ValueError(f"action {action} out of range [0, {self._num_actions})")
        x = np.asarray(context, dtype=float).reshape(-1)
        if self._reward_mode == "logistic":
            return float(exp_logistic(self._theta_true[action], x))
        else:
            return float(exp_linear(self._theta_true[action], x))

    def drift(self, scale: float = 0.5) -> None:
        self._theta_true = apply_drift(self._theta_true, scale=scale, rng=self._rng)

    def theta(self) -> np.ndarray:
        return np.asarray(self._theta_true, dtype=float)
