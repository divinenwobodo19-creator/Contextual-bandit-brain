import numpy as np
import pytest

from brain import LinUCBBandit
from simulator import ContextualBanditSimulator


def run_episode(env: ContextualBanditSimulator, brain: LinUCBBandit, steps: int, track_exploration: bool = False):
    rewards = []
    exploration = 0
    for _ in range(steps):
        x = env.generate_context()
        brain.observe(x)
        a = brain.choose_action()
        if track_exploration:
            info = brain.explain_last_decision()
            if info["mode"] == "exploration":
                exploration += 1
        r = env.get_reward(a, x)
        brain.receive_reward(a, r)
        rewards.append(r)
    return np.array(rewards), exploration


def expected_reward(env: ContextualBanditSimulator, action: int, x: np.ndarray) -> float:
    theta = env._theta_true[action]
    raw = float(x.dot(theta))
    return 1.0 / (1.0 + np.exp(-raw))


def test_learning_and_exploration_behavior():
    d, num_actions, alpha = 8, 5, 0.5
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=0.1, seed=0)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    steps = 2000
    rewards, exploration = run_episode(env, brain, steps, track_exploration=True)
    win = 300
    initial_avg = float(np.mean(rewards[:win]))
    final_avg = float(np.mean(rewards[-win:]))
    assert final_avg > initial_avg
    assert exploration / steps > 0.0  # early exploration exists

    # Exploration decreases: compare early vs late ratios using a sliding window
    _, exploration2 = run_episode(env, brain, steps, track_exploration=True)
    early_ratio = exploration / steps
    late_ratio = exploration2 / steps
    assert late_ratio < early_ratio


def test_regret_reduction_trend():
    d, num_actions, alpha = 8, 5, 0.5
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=0.1, seed=1)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    steps = 1500
    regrets = []
    for _ in range(steps):
        x = env.generate_context()
        brain.observe(x)
        a = brain.choose_action()
        expected_rewards = [expected_reward(env, i, x) for i in range(num_actions)]
        best_exp = max(expected_rewards)
        chosen_exp = expected_rewards[a]
        regrets.append(best_exp - chosen_exp)
        r = env.get_reward(a, x)
        brain.receive_reward(a, r)
    regrets = np.array(regrets)
    win = 300
    initial_avg_regret = float(np.mean(regrets[:win]))
    final_avg_regret = float(np.mean(regrets[-win:]))
    assert final_avg_regret < initial_avg_regret

