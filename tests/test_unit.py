import numpy as np
import pytest

from actions import LinUCBAction
from brain import LinUCBBandit
from tests.conftest import manual_ucb


def test_action_initialization_identity_and_zero():
    d = 4
    a = LinUCBAction(d)
    assert a.d == d
    theta = a.theta()
    assert np.allclose(theta, np.zeros(d))
    # Access private state via name mangling to verify numerically
    A = a._A
    b = a._b
    assert np.allclose(A, np.eye(d))
    assert np.allclose(b, np.zeros(d))


def test_dimensions_match_feature_size(default_params):
    brain = LinUCBBandit(num_actions=default_params["num_actions"], alpha=default_params["alpha"], d=default_params["d"])
    assert brain.d == default_params["d"]
    for a in brain._actions:
        assert a.d == default_params["d"]
        assert a._A.shape == (default_params["d"], default_params["d"])
        assert a._b.shape == (default_params["d"],)


def test_independent_parameters_per_action(default_params):
    brain = LinUCBBandit(num_actions=default_params["num_actions"], alpha=default_params["alpha"], d=default_params["d"])
    x = np.ones(default_params["d"])
    brain.observe(x)
    a_id = brain.choose_action()
    brain.receive_reward(a_id, 1.0)
    # Only one action should have non-zero b after a single update with non-zero reward
    nonzero_bs = [np.linalg.norm(a._b) > 1e-12 for a in brain._actions]
    assert sum(nonzero_bs) == 1


def test_ucb_calculation_deterministic():
    d = 3
    alpha = 0.4
    x = np.array([1.0, -2.0, 0.5])
    action = LinUCBAction(d)
    est, unc, ucb = action.score(x, alpha)
    manual = manual_ucb(action._A, action._b, x, alpha)
    assert np.isclose(ucb, manual)
    assert np.isclose(est + unc, ucb)


def test_action_selection_tie_breaking():
    d = 3
    brain = LinUCBBandit(num_actions=4, alpha=0.3, d=d)
    x = np.array([0.2, 0.2, 0.2])
    brain.observe(x)
    a = brain.choose_action()
    # All actions equal at initialization; deterministic np.argmax chooses index 0
    assert a == 0


def test_update_rule_numeric():
    d = 3
    brain = LinUCBBandit(num_actions=2, alpha=0.3, d=d)
    x = np.array([1.0, 2.0, 3.0])
    r = 0.5
    brain.observe(x)
    a = brain.choose_action()
    # Capture pre-update A and b for chosen action
    A_prev = brain._actions[a]._A.copy()
    b_prev = brain._actions[a]._b.copy()
    brain.receive_reward(a, r)
    A_new = brain._actions[a]._A
    b_new = brain._actions[a]._b
    assert np.allclose(A_new, A_prev + np.outer(x, x))
    assert np.allclose(b_new, b_prev + r * x)


def test_reward_validation_and_edge_cases():
    d = 3
    brain = LinUCBBandit(num_actions=2, alpha=0.3, d=d)
    x = np.array([1.0, 0.0, -1.0])

    # Reward 0
    brain.observe(x)
    a = brain.choose_action()
    b_prev = brain._actions[a]._b.copy()
    brain.receive_reward(a, 0.0)
    assert np.allclose(brain._actions[a]._b, b_prev)  # no change when r=0

    # Reward 1
    brain.observe(x)
    a = brain.choose_action()
    b_prev = brain._actions[a]._b.copy()
    brain.receive_reward(a, 1.0)
    assert np.allclose(brain._actions[a]._b, b_prev + x)

    # Reward < 0 gets clipped to 0
    brain.observe(x)
    a = brain.choose_action()
    b_prev = brain._actions[a]._b.copy()
    brain.receive_reward(a, -5.0)
    assert np.allclose(brain._actions[a]._b, b_prev)  # clipped to 0

    # Reward > 1 gets clipped to 1
    brain.observe(x)
    a = brain.choose_action()
    b_prev = brain._actions[a]._b.copy()
    brain.receive_reward(a, 3.0)
    assert np.allclose(brain._actions[a]._b, b_prev + x)  # clipped to 1

    # NaN reward is rejected
    brain.observe(x)
    a = brain.choose_action()
    with pytest.raises(ValueError):
        brain.receive_reward(a, float("nan"))

