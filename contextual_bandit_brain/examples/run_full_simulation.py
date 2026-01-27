"""
Run full simulation using the library components.
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from contextual_bandit_brain.core.brain import LinUCBBrain
from contextual_bandit_brain.simulator.environment import Environment


def expected_reward(theta: np.ndarray, x: np.ndarray) -> float:
    raw = float(x.dot(theta))
    return 1.0 / (1.0 + np.exp(-raw))


def sampler_normal(rng: np.random.Generator, d: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=d).astype(float)


def run(seed: int = 0) -> dict:
    d = 8
    num_actions = 5
    alpha = 0.5
    steps = 3000
    env = Environment(d=d, num_actions=num_actions, noise_std=0.1, seed=seed)
    brain = LinUCBBrain(num_actions=num_actions, alpha=alpha, d=d)
    rewards = []
    exploration = 0
    for t in range(steps):
        x = sampler_normal(env._rng, d)
        a = brain.select_action(x)
        info = brain.explain_last()
        if info["mode"] == "exploration":
            exploration += 1
        r = env.reward(a, x)
        brain.update(a, r, x)
        rewards.append(r)
        if t == steps // 2:
            env.drift(scale=0.6)
    return {"avg_reward": float(np.mean(rewards)), "exploration_ratio": exploration / max(steps, 1)}


if __name__ == "__main__":
    out = run()
    path = os.path.join(PROJECT_ROOT, "examples", "full_sim_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

