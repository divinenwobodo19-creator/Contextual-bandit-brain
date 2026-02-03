from __future__ import annotations
import numpy as np


def expected_reward(theta: np.ndarray, x: np.ndarray) -> float:
    raw = float(np.asarray(x, dtype=float).reshape(-1).dot(np.asarray(theta, dtype=float).reshape(-1)))
    return float(1.0 / (1.0 + np.exp(-raw)))


def sample_reward(theta: np.ndarray, x: np.ndarray, noise_std: float, rng: np.random.Generator) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    raw = float(x.dot(theta) + rng.normal(0.0, float(noise_std)))
    r = float(1.0 / (1.0 + np.exp(-raw)))
    return float(np.clip(r, 0.0, 1.0))


def expected_reward_linear(theta: np.ndarray, x: np.ndarray) -> float:
    raw = float(np.asarray(x, dtype=float).reshape(-1).dot(np.asarray(theta, dtype=float).reshape(-1)))
    return float(np.clip(raw, 0.0, 1.0))


def sample_reward_linear(theta: np.ndarray, x: np.ndarray, noise_std: float, rng: np.random.Generator) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    raw = float(x.dot(theta) + rng.normal(0.0, float(noise_std)))
    return float(np.clip(raw, 0.0, 1.0))
