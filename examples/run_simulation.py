"""
Example: validate LinUCBBandit learning in a synthetic environment.

Intent:
- Demonstrate the strict loop: observe → choose_action → receive_reward.
- Track learning via average reward and exploration frequency.
- Exercise persistence by saving/loading the brain state mid-run.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np

# Ensure parent directory is importable when running this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brain import LinUCBBandit
from simulator import ContextualBanditSimulator


def run(seed: int = 0) -> Tuple[float, float]:
    d = 8
    num_actions = 5
    alpha = 0.5
    steps_phase1 = 2000
    steps_phase2 = 2000

    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=0.1, seed=seed)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)

    rewards = []
    exploration_count = 0

    for _ in range(steps_phase1):
        x = env.generate_context()
        brain.observe(x)
        a = brain.choose_action()
        info = brain.explain_last_decision()
        if info["mode"] == "exploration":
            exploration_count += 1
        r = env.get_reward(a, x)
        brain.receive_reward(a, r)
        rewards.append(r)

    avg_reward_phase1 = float(np.mean(rewards)) if rewards else 0.0
    exploration_ratio_phase1 = exploration_count / max(len(rewards), 1)

    state_path = os.path.join(PROJECT_ROOT, "linucb_state.json")
    brain.save_state(state_path)

    brain2 = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    brain2.load_state(state_path)

    rewards2 = []
    exploration_count2 = 0
    for _ in range(steps_phase2):
        x = env.generate_context()
        brain2.observe(x)
        a = brain2.choose_action()
        info = brain2.explain_last_decision()
        if info["mode"] == "exploration":
            exploration_count2 += 1
        r = env.get_reward(a, x)
        brain2.receive_reward(a, r)
        rewards2.append(r)

    avg_reward_phase2 = float(np.mean(rewards2)) if rewards2 else 0.0
    exploration_ratio_phase2 = exploration_count2 / max(len(rewards2), 1)

    print(f"Phase 1: average reward={avg_reward_phase1:.4f}, exploration={exploration_ratio_phase1:.3f}")
    print(f"Phase 2 (after load): average reward={avg_reward_phase2:.4f}, exploration={exploration_ratio_phase2:.3f}")

    return avg_reward_phase1, avg_reward_phase2


if __name__ == "__main__":
    avg1, avg2 = run()
    metrics = {
        "phase1_avg_reward": float(avg1),
        "phase2_avg_reward": float(avg2),
    }
    out_path = os.path.join(PROJECT_ROOT, "examples", "metrics.json")
    try:
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f)
        print(f"Metrics written to {out_path}")
    except Exception as e:
        print(f"Failed to write metrics: {e}")
