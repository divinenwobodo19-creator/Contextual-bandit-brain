import numpy as np
import pytest

from brain import LinUCBBandit


def test_missing_context_raises():
    brain = LinUCBBandit(num_actions=3, alpha=0.5, d=4)
    with pytest.raises(RuntimeError):
        brain.choose_action()


def test_dimension_mismatch_raises():
    brain = LinUCBBandit(num_actions=3, alpha=0.5, d=4)
    x = np.ones(5)
    with pytest.raises(ValueError):
        brain.observe(x)


def test_reset_behavior_correctness():
    brain = LinUCBBandit(num_actions=2, alpha=0.3, d=3)
    x = np.array([1.0, -1.0, 0.5])
    brain.observe(x)
    a = brain.choose_action()
    brain.receive_reward(a, 1.0)
    # Ensure parameters changed
    assert not np.allclose(brain._actions[a]._A, np.eye(3))
    assert np.linalg.norm(brain._actions[a]._b) > 0
    # Reset
    brain.reset()
    for act in brain._actions:
        assert np.allclose(act._A, np.eye(3))
        assert np.allclose(act._b, np.zeros(3))
    assert brain._t == 0
    with pytest.raises(RuntimeError):
        brain.choose_action()  # context cleared by reset


def test_wrong_action_reward_raises():
    brain = LinUCBBandit(num_actions=3, alpha=0.5, d=4)
    x = np.ones(4)
    brain.observe(x)
    chosen = brain.choose_action()
    wrong = (chosen + 1) % 3
    with pytest.raises(ValueError):
        brain.receive_reward(wrong, 0.5)


def test_pending_reward_blocks_next_choose():
    brain = LinUCBBandit(num_actions=3, alpha=0.5, d=4)
    x = np.ones(4)
    brain.observe(x)
    _ = brain.choose_action()
    with pytest.raises(RuntimeError):
        brain.choose_action()
    brain.receive_reward(0, 0.1)  # deliver reward for last action
    # Now can proceed with next observe/choose
    brain.observe(x)
    _ = brain.choose_action()

