from __future__ import annotations

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brain import LinUCBBandit
from simulator import ContextualBanditSimulator


def expected_reward(theta: np.ndarray, x: np.ndarray) -> float:
    raw = float(x.dot(theta))
    return 1.0 / (1.0 + np.exp(-raw))


def run_clean(seed: int = 0, d: int = 8, num_actions: int = 5, alpha: float = 0.5, steps: int = 2000, noise_std: float = 0.1):
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=noise_std, seed=seed)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)

    rewards = []
    explorations = []
    regrets = []

    for _ in range(steps):
        x = env.generate_context()
        brain.observe(x)
        a = brain.choose_action()
        info = brain.explain_last_decision()
        explorations.append(info["mode"] == "exploration")
        exp_rewards = [expected_reward(env._theta_true[i], x) for i in range(num_actions)]
        best_exp = max(exp_rewards)
        chosen_exp = exp_rewards[a]
        regrets.append(best_exp - chosen_exp)
        r = env.get_reward(a, x)
        brain.receive_reward(a, r)
        rewards.append(r)

    rewards = np.asarray(rewards, dtype=float)
    explorations = np.asarray(explorations, dtype=bool)
    regrets = np.asarray(regrets, dtype=float)

    win = max(200, steps // 10)
    avg_reward_initial = float(np.mean(rewards[:win]))
    avg_reward_final = float(np.mean(rewards[-win:]))
    exploration_initial = float(np.mean(explorations[:win]))
    exploration_final = float(np.mean(explorations[-win:]))
    cum_regret = float(np.sum(regrets))
    avg_regret_per_step = float(cum_regret / steps)

    conv_win = max(50, steps // 20)
    rolling_regret = np.convolve(regrets, np.ones(conv_win) / conv_win, mode="valid")
    conv_step = steps
    for idx, val in enumerate(rolling_regret):
        if val <= 0.05:
            conv_step = idx + conv_win - 1
            break

    report = {
        "config": {"seed": seed, "d": d, "num_actions": num_actions, "alpha": alpha, "steps": steps, "noise_std": noise_std},
        "metrics": {
            "avg_reward_initial": avg_reward_initial,
            "avg_reward_final": avg_reward_final,
            "exploration_initial": exploration_initial,
            "exploration_final": exploration_final,
            "cumulative_regret": cum_regret,
            "avg_regret_per_step": avg_regret_per_step,
            "convergence_step": conv_step,
        },
    }

    print("======================================")
    print("LinUCB Clean Test")
    print("======================================")
    print(f"seed={seed} | d={d} | actions={num_actions} | alpha={alpha} | steps={steps} | noise={noise_std}")
    print("--------------------------------------")
    print(f"avg_reward_initial: {avg_reward_initial:.4f}")
    print(f"avg_reward_final:   {avg_reward_final:.4f}")
    print(f"exploration_initial:{exploration_initial:.3f}")
    print(f"exploration_final:  {exploration_final:.3f}")
    print(f"cumulative_regret:  {cum_regret:.4f}")
    print(f"avg_regret_per_step:{avg_regret_per_step:.4f}")
    print(f"convergence_step:   {conv_step}")
    print("======================================")

    out_path = os.path.join(PROJECT_ROOT, "examples", "clean_ui_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    run_clean()

