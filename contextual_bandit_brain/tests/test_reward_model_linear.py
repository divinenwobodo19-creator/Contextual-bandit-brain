import numpy as np
import pytest

from contextual_bandit_brain.simulator.reward_model import expected_reward_linear, sample_reward_linear


def test_expected_reward_linear_clips_and_matches_dot():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=5).astype(float)
    theta = rng.normal(0.0, 1.0, size=5).astype(float)
    raw = float(x.dot(theta))
    exp = expected_reward_linear(theta, x)
    assert exp == float(np.clip(raw, 0.0, 1.0))


def test_sample_reward_linear_zero_noise_equals_expected():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=4).astype(float)
    theta = rng.normal(0.0, 1.0, size=4).astype(float)
    r = sample_reward_linear(theta, x, noise_std=0.0, rng=rng)
    exp = expected_reward_linear(theta, x)
    assert r == exp
