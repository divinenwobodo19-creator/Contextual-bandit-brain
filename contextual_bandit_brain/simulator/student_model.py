"""
Student model generating context vectors.

Features can encode profile, performance, topic difficulty, and temporal signals.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


class StudentModel:
    """
    Synthetic student context generator.
    """

    def __init__(self, d: int, seed: Optional[int] = None) -> None:
        if d <= 0:
            raise ValueError("feature dimension d must be positive")
        self._d = int(d)
        try:
            self._rng = np.random.default_rng(seed)
        except AttributeError:
            self._rng = np.random.RandomState(seed)

    @property
    def d(self) -> int:
        return self._d

    def sample_context(self) -> np.ndarray:
        base = self._rng.normal(0.0, 1.0, size=self._d).astype(float)
        time = self._rng.normal(0.0, 0.5)
        base[-1] = time
        return base
