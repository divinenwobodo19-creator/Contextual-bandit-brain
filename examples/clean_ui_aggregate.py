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


def sampler_uniform(rng: np.random.Generator, d: int) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=d).astype(float)


def sampler_normal(rng: np.random.Generator, d: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=d).astype(float)


def sampler_laplace(rng: np.random.Generator, d: int) -> np.ndarray:
    return rng.laplace(0.0, 1.0, size=d).astype(float)


def run_domain(domain: str, sampler, seeds, d: int, num_actions: int, alpha: float, steps: int, noise_std: float):
    metrics = []
    for seed in seeds:
        env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=noise_std, seed=seed)
        brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
        rewards = []
        explorations = []
        regrets = []
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            x = sampler(rng, d)
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
        metrics.append(
            {
                "seed": seed,
                "avg_reward_initial": avg_reward_initial,
                "avg_reward_final": avg_reward_final,
                "exploration_initial": exploration_initial,
                "exploration_final": exploration_final,
                "cumulative_regret": cum_regret,
                "avg_regret_per_step": avg_regret_per_step,
                "convergence_step": int(conv_step),
            }
        )
    agg = {
        "avg_reward_initial": float(np.mean([m["avg_reward_initial"] for m in metrics])),
        "avg_reward_final": float(np.mean([m["avg_reward_final"] for m in metrics])),
        "exploration_initial": float(np.mean([m["exploration_initial"] for m in metrics])),
        "exploration_final": float(np.mean([m["exploration_final"] for m in metrics])),
        "cumulative_regret": float(np.mean([m["cumulative_regret"] for m in metrics])),
        "avg_regret_per_step": float(np.mean([m["avg_regret_per_step"] for m in metrics])),
        "convergence_step": int(np.mean([m["convergence_step"] for m in metrics])),
    }
    return {"domain": domain, "config": {"d": d, "num_actions": num_actions, "alpha": alpha, "steps": steps, "noise_std": noise_std, "seeds": seeds}, "aggregate": agg, "runs": metrics}


def main():
    d = 8
    num_actions = 5
    alpha = 0.5
    steps = 2000
    noise_std = 0.1
    seeds = [0, 1, 2, 3, 4]
    domains = [
        ("education", sampler_uniform),
        ("finance", sampler_laplace),
        ("abstract", sampler_normal),
    ]
    results = []
    print("======================================")
    print("LinUCB Clean Aggregate Test")
    print("======================================")
    print(f"d={d} | actions={num_actions} | alpha={alpha} | steps={steps} | noise={noise_std} | seeds={seeds}")
    for name, sampler in domains:
        res = run_domain(name, sampler, seeds, d, num_actions, alpha, steps, noise_std)
        results.append(res)
        agg = res["aggregate"]
        print("--------------------------------------")
        print(f"domain: {name}")
        print(f"avg_reward_initial: {agg['avg_reward_initial']:.4f}")
        print(f"avg_reward_final:   {agg['avg_reward_final']:.4f}")
        print(f"exploration_initial:{agg['exploration_initial']:.3f}")
        print(f"exploration_final:  {agg['exploration_final']:.3f}")
        print(f"cumulative_regret:  {agg['cumulative_regret']:.4f}")
        print(f"avg_regret_per_step:{agg['avg_regret_per_step']:.4f}")
        print(f"convergence_step:   {agg['convergence_step']}")
    out_path = os.path.join(PROJECT_ROOT, "examples", "clean_ui_aggregate_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print("======================================")
    print(f"Aggregate report written to {out_path}")


if __name__ == "__main__":
    main()

