import numpy as np
import pytest

from contextual_bandit_brain.core.brain import LinUCBBrain
from contextual_bandit_brain.simulator.environment import Environment


def run_once(seed: int, d: int = 6, num_actions: int = 4, alpha: float = 0.5, steps: int = 500):
    env = Environment(d=d, num_actions=num_actions, noise_std=0.1, seed=seed)
    brain = LinUCBBrain(num_actions=num_actions, alpha=alpha, d=d)
    rewards = []
    actions = []
    for _ in range(steps):
        x = env._rng.normal(0.0, 1.0, size=d).astype(float)
        a = brain.select_action(x)
        r = env.reward(a, x)
        brain.update(a, r, x)
        rewards.append(r)
        actions.append(a)
    return np.asarray(rewards, dtype=float), np.asarray(actions, dtype=int)


def test_deterministic_under_fixed_seed():
    r1, a1 = run_once(seed=0)
    r2, a2 = run_once(seed=0)
    assert np.allclose(r1, r2)
    assert np.array_equal(a1, a2)
