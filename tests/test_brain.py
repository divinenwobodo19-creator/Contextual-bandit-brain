import numpy as np
from contextual_bandit_brain.brain.linucb import LinUCBBrain


def test_decide_and_learn_flow():
    d, num_actions, alpha = 6, 4, 0.4
    brain = LinUCBBrain(num_actions=num_actions, alpha=alpha, d=d)
    x = np.random.randn(d).astype(float).tolist()
    decision = brain.decide(x)
    assert 0 <= decision["action"] < num_actions
    assert "ucb" in decision and "uncertainty" in decision and "mode" in decision
    brain.learn(x, decision["action"], reward=0.7)
