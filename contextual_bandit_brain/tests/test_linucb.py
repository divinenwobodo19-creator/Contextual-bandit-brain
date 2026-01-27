import numpy as np
import pytest

from contextual_bandit_brain.core.arm import LinUCBArm
from contextual_bandit_brain.core.brain import LinUCBBrain


def test_arm_initialization_and_update():
    d = 4
    arm = LinUCBArm(d)
    assert arm.d == d
    assert np.allclose(arm._A, np.eye(d))
    assert np.allclose(arm._b, np.zeros(d))
    x = np.array([1.0, 0.0, -1.0, 0.5])
    arm.update(x, 1.0)
    assert np.allclose(arm._A, np.eye(d) + np.outer(x, x))
    assert np.allclose(arm._b, x)


def test_brain_selection_logic():
    d, num_actions, alpha = 3, 3, 0.5
    brain = LinUCBBrain(num_actions=num_actions, alpha=alpha, d=d)
    x = np.array([0.2, 0.2, 0.2])
    a = brain.select_action(x)
    assert a == 0
    info = brain.explain_last()
    assert "ucb" in info and "mode" in info

