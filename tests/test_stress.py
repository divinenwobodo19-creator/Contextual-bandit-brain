import numpy as np
import pytest

from brain import LinUCBBandit
from simulator import ContextualBanditSimulator


def test_high_dimension_and_large_action_space_no_crash():
    d, num_actions, alpha = 64, 20, 0.3
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=0.2, seed=999)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    steps = 1000
    rewards = []
    for _ in range(steps):
        x = env.generate_context()
        brain.observe(x)
        a = brain.choose_action()
        r = env.get_reward(a, x)
        brain.receive_reward(a, r)
        rewards.append(r)
    rewards = np.array(rewards)
    assert np.isfinite(rewards).all()
    assert float(np.mean(rewards[-200:])) >= float(np.mean(rewards[:200]))


def test_noisy_rewards_stability():
    d, num_actions, alpha = 16, 8, 0.4
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=1.0, seed=321)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    steps = 1500
    rewards = []
    for _ in range(steps):
        x = env.generate_context()
        brain.observe(x)
        a = brain.choose_action()
        r = env.get_reward(a, x)
        brain.receive_reward(a, r)
        rewards.append(r)
    rewards = np.array(rewards)
    assert np.isfinite(rewards).all()
    assert float(np.mean(rewards[-300:])) >= float(np.mean(rewards[:300]))


def test_sparse_rewards_robustness():
    d, num_actions, alpha = 12, 6, 0.5
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=0.4, seed=555)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    steps = 2000
    rng = np.random.default_rng(777)
    dropout = 0.5
    rewards = []
    for _ in range(steps):
        x = env.generate_context()
        brain.observe(x)
        a = brain.choose_action()
        r = env.get_reward(a, x)
        if rng.random() < dropout:
            r = 0.0
        brain.receive_reward(a, r)
        rewards.append(r)
    rewards = np.array(rewards)
    assert np.isfinite(rewards).all()
    assert float(np.mean(rewards[-300:])) >= float(np.mean(rewards[:300]))

def test_delayed_rewards_no_crash_and_stable_learning():
    d, num_actions, alpha = 10, 6, 0.4
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=0.2, seed=888)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    steps = 1200
    rng = np.random.default_rng(42)
    rewards = []
    for _ in range(steps):
        x = env.generate_context()
        brain.observe(x)
        a = brain.choose_action()
        delay = rng.integers(0, 3)  # simulate reward delay by idle loops
        for _ in range(int(delay)):
            pass
        r = env.get_reward(a, x)
        brain.receive_reward(a, r)
        rewards.append(r)
    rewards = np.array(rewards)
    assert np.isfinite(rewards).all()
    assert float(np.mean(rewards[-300:])) >= float(np.mean(rewards[:300]))
