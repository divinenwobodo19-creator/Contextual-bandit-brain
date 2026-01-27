import numpy as np
import os
import tempfile

from brain import LinUCBBandit
from simulator import ContextualBanditSimulator


def test_save_load_continue_learning(tmp_path):
    d, num_actions, alpha = 8, 5, 0.5
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=0.1, seed=7)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    steps = 1000
    for _ in range(steps):
        x = env.generate_context()
        brain.observe(x)
        a = brain.choose_action()
        r = env.get_reward(a, x)
        brain.receive_reward(a, r)

    state_file = os.path.join(tmp_path, "linucb_state_test.json")
    brain.save_state(state_file)

    brain2 = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    brain2.load_state(state_file)

    # Parameters unchanged after reload
    for a1, a2 in zip(brain._actions, brain2._actions):
        assert np.allclose(a1._A, a2._A)
        assert np.allclose(a1._b, a2._b)

    # Decisions remain consistent on a fixed context
    x = env.generate_context()
    brain.observe(x)
    a1 = brain.choose_action()
    brain2.observe(x)
    a2 = brain2.choose_action()
    assert a1 == a2
    # Deliver rewards to clear pending actions
    r1 = env.get_reward(a1, x)
    brain.receive_reward(a1, r1)
    r2 = env.get_reward(a2, x)
    brain2.receive_reward(a2, r2)

    # Continue learning
    rewards = []
    for _ in range(1000):
        x = env.generate_context()
        brain2.observe(x)
        a = brain2.choose_action()
        r = env.get_reward(a, x)
        brain2.receive_reward(a, r)
        rewards.append(r)
    rewards = np.array(rewards)
    final_avg = float(np.mean(rewards[-200:]))
    initial_avg = float(np.mean(rewards[:200]))
    assert final_avg >= (initial_avg - 0.02)
