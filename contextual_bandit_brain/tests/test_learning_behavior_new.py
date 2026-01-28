import numpy as np
from contextual_bandit_brain.brain.linucb import LinUCBBrain
from contextual_bandit_brain.simulator.environment import Environment


def test_regret_decreases_over_time():
    d, num_actions, alpha = 8, 5, 0.5
    env = Environment(d=d, num_actions=num_actions, noise_std=0.1, seed=1)
    brain = LinUCBBrain(num_actions=num_actions, alpha=alpha, d=d)
    steps = 800
    regrets = []
    for t in range(steps):
        x = env.context()
        decision = brain.decide(x.tolist())
        a = decision["action"]
        r = env.reward(a, x)
        brain.learn(x.tolist(), a, r)
        best = max(env.reward(k, x) for k in range(num_actions))
        regrets.append(best - r)
    regrets = np.asarray(regrets, dtype=float)
    first_half = float(np.mean(regrets[: steps // 2]))
    second_half = float(np.mean(regrets[steps // 2 :]))
    assert second_half <= first_half + 0.02
