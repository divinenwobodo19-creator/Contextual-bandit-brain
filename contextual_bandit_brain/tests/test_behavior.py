import numpy as np
import pytest

from contextual_bandit_brain.core.brain import LinUCBBrain


def test_exploitation_after_update_with_low_alpha():
    d, num_actions = 2, 2
    brain = LinUCBBrain(num_actions=num_actions, alpha=0.1, d=d)
    x = np.array([1.0, 0.0], dtype=float)
    a0 = brain.select_action(x)
    info0 = brain.explain_last()
    assert a0 == 0
    assert info0["mode"] in ("exploration", "exploitation")
    brain.update(a0, 1.0, x)
    a1 = brain.select_action(x)
    info1 = brain.explain_last()
    assert a1 == 0
    assert info1["mode"] == "exploitation"


def test_exploration_with_high_alpha_due_to_uncertainty():
    d, num_actions = 2, 2
    brain = LinUCBBrain(num_actions=num_actions, alpha=10.0, d=d)
    x = np.array([1.0, 0.0], dtype=float)
    a0 = brain.select_action(x)
    brain.update(a0, 1.0, x)
    a1 = brain.select_action(x)
    info = brain.explain_last()
    assert a1 == 1
    assert info["mode"] == "exploration"


def test_explain_last_contains_expected_keys():
    d, num_actions = 3, 3
    brain = LinUCBBrain(num_actions=num_actions, alpha=0.5, d=d)
    x = np.array([0.2, -0.1, 0.5], dtype=float)
    _ = brain.select_action(x)
    info = brain.explain_last()
    for key in ("action", "estimated_reward", "uncertainty", "ucb", "mode", "context"):
        assert key in info
