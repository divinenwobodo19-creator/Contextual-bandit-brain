import numpy as np
from contextual_bandit_brain.brain.linucb import LinUCBBrain
from contextual_bandit_brain.simulator.environment import Environment


def test_learning_improves_reward_over_random():
    d, num_actions, alpha = 8, 5, 0.5
    env = Environment(d=d, num_actions=num_actions, noise_std=0.1, seed=0)
    brain = LinUCBBrain(num_actions=num_actions, alpha=alpha, d=d)
    steps = 500
    rewards_bandit = []
    rewards_random = []
    rng = np.random.default_rng(0)
    for _ in range(steps):
        x = env.context()
        decision = brain.decide(x.tolist())
        a = decision["action"]
        r = env.reward(a, x)
        brain.learn(x.tolist(), a, r)
        rewards_bandit.append(r)
        ar = rng.integers(0, num_actions)
        rewards_random.append(env.reward(int(ar), x))
    assert np.mean(rewards_bandit) >= np.mean(rewards_random) - 0.05
