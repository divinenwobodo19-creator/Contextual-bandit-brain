import numpy as np
import pytest

from contextual_bandit_brain.core.brain import LinUCBBrain
from contextual_bandit_brain.simulator.environment import Environment


def test_reward_improves_and_exploration_nonzero():
    d, num_actions, alpha, steps = 8, 5, 0.5, 2000
    env = Environment(d=d, num_actions=num_actions, noise_std=0.1, seed=0)
    brain = LinUCBBrain(num_actions=num_actions, alpha=alpha, d=d)
    rewards = []
    exploration_flags = []
    for t in range(steps):
        x = env._rng.normal(0.0, 1.0, size=d).astype(float)
        a = brain.select_action(x)
        info = brain.explain_last()
        exploration_flags.append(info["mode"] == "exploration")
        r = env.reward(a, x)
        brain.update(a, r, x)
        rewards.append(r)
        if t == steps // 2:
            env.drift(scale=0.6)
    rewards = np.asarray(rewards, dtype=float)
    exploration_flags = np.asarray(exploration_flags, dtype=bool)
    phase1 = steps // 2
    first = np.mean(rewards[: phase1 // 4])
    last = np.mean(rewards[phase1 - phase1 // 4 : phase1])
    assert last >= first
    ratio_early = float(np.mean(exploration_flags[: phase1 // 4].astype(float)))
    ratio_late = float(np.mean(exploration_flags[phase1 - phase1 // 4 : phase1].astype(float)))
    assert ratio_late <= ratio_early
    overall_ratio = float(np.mean(exploration_flags.astype(float)))
    assert overall_ratio > 0.0


def test_cumulative_regret_sublinear_tendency():
    d, num_actions, alpha, steps = 8, 5, 0.5, 2000
    env = Environment(d=d, num_actions=num_actions, noise_std=0.1, seed=1)
    brain = LinUCBBrain(num_actions=num_actions, alpha=alpha, d=d)
    regrets = []
    for t in range(steps):
        x = env._rng.normal(0.0, 1.0, size=d).astype(float)
        a = brain.select_action(x)
        r = env.reward(a, x)
        brain.update(a, r, x)
        exp_rewards = [1.0 / (1.0 + np.exp(-float(x.dot(env.theta()[i])))) for i in range(num_actions)]
        regrets.append(max(exp_rewards) - exp_rewards[a])
        if t == steps // 2:
            env.drift(scale=0.6)
    phase1 = steps // 2
    cum = np.cumsum(np.asarray(regrets[:phase1], dtype=float))
    slope_first = float((cum[phase1 // 2] - cum[phase1 // 4]) / (phase1 // 4))
    slope_last = float((cum[-1] - cum[phase1 - phase1 // 4]) / (phase1 // 4))
    assert slope_last <= slope_first
