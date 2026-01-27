"""
BIS benchmark pipeline for LinUCB service.

Outputs:
- JSON report to /artifacts/bis_report.json
- PNG plots:
  * cumulative reward
  * regret over time
  * exploration ratio
  * arm distribution
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulator import ContextualBanditSimulator
from service import LinUCBService


def expected_reward(theta: np.ndarray, x: np.ndarray) -> float:
    raw = float(x.dot(theta))
    return 1.0 / (1.0 + np.exp(-raw))


def sampler_normal(rng: np.random.Generator, d: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=d).astype(float)


def sampler_laplace(rng: np.random.Generator, d: int) -> np.ndarray:
    return rng.laplace(0.0, 1.0, size=d).astype(float)


def run_episode(env: ContextualBanditSimulator, service: LinUCBService, steps: int, sampler, track_exploration: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rewards = []
    regrets = []
    chosen_actions = []
    exploration_flags = []
    est_reward_argmax = []
    rng = env._rng
    for _ in range(steps):
        x = sampler(rng, env.d)
        scores = []
        ests = []
        for a in service._actions:
            est, unc, ucb = a.score(x, service.alpha)
            scores.append(ucb)
            ests.append(est)
        action = int(np.argmax(scores))
        if track_exploration:
            est_argmax = int(np.argmax(ests))
            est_reward_argmax.append(est_argmax)
            exploration_flags.append(action != est_argmax)
        chosen_actions.append(action)
        r = env.get_reward(action, x)
        rewards.append(r)
        exp_rewards = [expected_reward(env._theta_true[i], x) for i in range(service.num_actions)]
        regret = max(exp_rewards) - exp_rewards[action]
        regrets.append(regret)
        service.update(action, r, x)
    return (
        np.asarray(rewards, dtype=float),
        np.asarray(regrets, dtype=float),
        np.asarray(chosen_actions, dtype=int),
        np.asarray(exploration_flags, dtype=bool),
        np.asarray(est_reward_argmax, dtype=int),
    )


def compute_metrics(rewards: np.ndarray, regrets: np.ndarray, chosen_actions: np.ndarray, env: ContextualBanditSimulator, contexts: List[np.ndarray], num_actions: int) -> Dict[str, float]:
    cum_reward = float(np.sum(rewards))
    best_exp_series = []
    for x in contexts:
        exp_rewards = [expected_reward(env._theta_true[i], x) for i in range(num_actions)]
        best_exp_series.append(max(exp_rewards))
    best_exp_sum = float(np.sum(best_exp_series))
    normalized_cum_reward = float(0.0 if best_exp_sum <= 1e-12 else np.clip(cum_reward / best_exp_sum, 0.0, 1.0))
    regret_efficiency = float(0.0 if best_exp_sum <= 1e-12 else np.clip(1.0 - (float(np.sum(regrets)) / best_exp_sum), 0.0, 1.0))
    stability = float(np.clip(1.0 - (float(np.std(rewards)) / 0.25), 0.0, 1.0))
    counts = np.bincount(chosen_actions, minlength=num_actions).astype(float)
    probs = counts / max(float(len(chosen_actions)), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(probs * np.log(probs + 1e-12))
    fairness = float(np.clip(entropy / np.log(max(num_actions, 2)), 0.0, 1.0))
    return {
        "normalized_cumulative_reward": normalized_cum_reward,
        "regret_efficiency": regret_efficiency,
        "stability": stability,
        "fairness": fairness,
    }


def adaptability_score(rewards_phase2: np.ndarray, env2: ContextualBanditSimulator, contexts_phase2: List[np.ndarray], num_actions: int, conv_step: int, steps_phase2: int) -> float:
    best_exp_series = []
    for x in contexts_phase2:
        exp_rewards = [expected_reward(env2._theta_true[i], x) for i in range(num_actions)]
        best_exp_series.append(max(exp_rewards))
    best_exp_avg = float(np.mean(best_exp_series[-max(100, steps_phase2 // 10):]))
    avg_reward_last = float(np.mean(rewards_phase2[-max(100, steps_phase2 // 10):]))
    normalized_last = float(0.0 if best_exp_avg <= 1e-12 else np.clip(avg_reward_last / best_exp_avg, 0.0, 1.0))
    conv_component = float(np.clip(1.0 - (conv_step / steps_phase2), 0.0, 1.0))
    return float(np.clip(0.5 * normalized_last + 0.5 * conv_component, 0.0, 1.0))


def plot_series(series: Dict[str, np.ndarray], out_dir: str) -> None:
    plt.figure(figsize=(8, 4))
    for label, y in series.items():
        plt.plot(y, label=label)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_reward.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    for label, y in series.items():
        if "regret" in label:
            plt.plot(y, label=label)
    plt.xlabel("step")
    plt.ylabel("regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regret_over_time.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    for label, y in series.items():
        if "exploration" in label:
            plt.plot(y.astype(float), label=label)
    plt.xlabel("step")
    plt.ylabel("exploration_flag")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exploration_ratio.png"))
    plt.close()


def plot_arm_distribution(counts: np.ndarray, out_dir: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(counts)), counts)
    plt.xlabel("arm")
    plt.ylabel("pulls")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "arm_distribution.png"))
    plt.close()


def run_benchmark(d: int = 8, num_actions: int = 5, alpha: float = 0.5, steps_phase1: int = 2000, steps_phase2: int = 2000, seeds: List[int] = [0, 1, 2]) -> Dict[str, Any]:
    base_noise = 0.1
    metrics_list = []
    rng = np.random.default_rng(seeds[0])
    artifacts_seed = seeds[0]
    for seed in seeds:
        env1 = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=base_noise, seed=seed)
        env2 = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=base_noise, seed=seed + 1000)
        service = LinUCBService(num_actions=num_actions, alpha=alpha, d=d)
        rewards1, regrets1, chosen1, exploration1, est_argmax1 = run_episode(env1, service, steps_phase1, sampler_normal, track_exploration=True)
        contexts1 = []
        env1._rng = np.random.default_rng(seed)  # ensure reproducibility
        service2 = LinUCBService(num_actions=num_actions, alpha=alpha, d=d)
        rewards2, regrets2, chosen2, exploration2, est_argmax2 = run_episode(env2, service, steps_phase2, sampler_laplace, track_exploration=True)
        exploration_ratio_series = exploration1.astype(float)
        conv_win = max(50, steps_phase2 // 20)
        rolling_regret2 = np.convolve(regrets2, np.ones(conv_win) / conv_win, mode="valid")
        conv_step2 = steps_phase2
        for idx, val in enumerate(rolling_regret2):
            if val <= 0.05:
                conv_step2 = idx + conv_win - 1
                break
        contexts1 = [sampler_normal(env1._rng, d) for _ in range(steps_phase1)]
        contexts2 = [sampler_laplace(env2._rng, d) for _ in range(steps_phase2)]
        m1 = compute_metrics(rewards1, regrets1, chosen1, env1, contexts1, num_actions)
        adapt = adaptability_score(rewards2, env2, contexts2, num_actions, conv_step2, steps_phase2)
        metrics_list.append({
            "normalized_cumulative_reward": m1["normalized_cumulative_reward"],
            "regret_efficiency": m1["regret_efficiency"],
            "stability": m1["stability"],
            "fairness": m1["fairness"],
            "adaptability": adapt,
        })
        if seed == artifacts_seed:
            cum_reward1 = np.cumsum(rewards1)
            cum_reward2 = np.cumsum(rewards2)
            series = {
                "cum_reward_phase1": cum_reward1,
                "cum_reward_phase2": cum_reward2,
                "regret_phase1": regrets1,
                "regret_phase2": regrets2,
                "exploration_phase1": exploration1,
                "exploration_phase2": exploration2,
            }
            plot_series(series, ARTIFACTS_DIR)
            counts_total = np.bincount(np.concatenate([chosen1, chosen2]), minlength=num_actions).astype(int)
            plot_arm_distribution(counts_total, ARTIFACTS_DIR)
    agg = {key: float(np.mean([m[key] for m in metrics_list])) for key in metrics_list[0].keys()}
    weights = {
        "reward": 0.35,
        "regret": 0.25,
        "stability": 0.15,
        "adaptability": 0.15,
        "fairness": 0.10,
    }
    bis = float(
        weights["reward"] * agg["normalized_cumulative_reward"]
        + weights["regret"] * agg["regret_efficiency"]
        + weights["stability"] * agg["stability"]
        + weights["adaptability"] * agg["adaptability"]
        + weights["fairness"] * agg["fairness"]
    )
    return {"metrics": agg, "BIS": bis}


def run_random_baseline(d: int, num_actions: int, alpha: float, steps: int, seed: int = 0) -> Dict[str, Any]:
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=0.1, seed=seed)
    rng = np.random.default_rng(seed)
    rewards = []
    regrets = []
    chosen = []
    contexts = []
    for _ in range(steps):
        x = sampler_normal(rng, d)
        a = int(rng.integers(0, num_actions))
        r = env.get_reward(a, x)
        rewards.append(r)
        exp_rewards = [expected_reward(env._theta_true[i], x) for i in range(num_actions)]
        regrets.append(max(exp_rewards) - exp_rewards[a])
        chosen.append(a)
        contexts.append(x)
    rewards = np.asarray(rewards, dtype=float)
    regrets = np.asarray(regrets, dtype=float)
    chosen = np.asarray(chosen, dtype=int)
    m = compute_metrics(rewards, regrets, chosen, env, contexts, num_actions)
    return {"metrics": m}


def main() -> int:
    d = int(os.environ.get("LINUCB_D", "8"))
    num_actions = int(os.environ.get("LINUCB_ACTIONS", "5"))
    alpha = float(os.environ.get("LINUCB_ALPHA", "0.5"))
    steps1 = int(os.environ.get("LINUCB_STEPS1", "2000"))
    steps2 = int(os.environ.get("LINUCB_STEPS2", "2000"))
    seeds = [0, 1, 2]
    result = run_benchmark(d=d, num_actions=num_actions, alpha=alpha, steps_phase1=steps1, steps_phase2=steps2, seeds=seeds)
    baseline = run_random_baseline(d=d, num_actions=num_actions, alpha=alpha, steps=steps1, seed=0)
    status = "PASS" if (result["BIS"] >= 0.75) else "FAIL"
    report = {
        "algorithm": "LinUCB",
        "domain": "education",
        "metrics": result["metrics"],
        "BIS": result["BIS"],
        "baseline_metrics": baseline["metrics"],
        "status": status,
    }
    out_path = os.path.join(ARTIFACTS_DIR, "bis_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

