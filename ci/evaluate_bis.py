from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brain import LinUCBBandit
from simulator import ContextualBanditSimulator


def expected_reward(theta: np.ndarray, x: np.ndarray) -> float:
    raw = float(x.dot(theta))
    return 1.0 / (1.0 + np.exp(-raw))


def evaluate_single_run(seed: int, d: int, num_actions: int, alpha: float, steps: int, noise_std: float) -> Dict[str, Any]:
    env = ContextualBanditSimulator(d=d, num_actions=num_actions, noise_std=noise_std, seed=seed)
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    rewards: List[float] = []
    explorations: List[bool] = []
    regrets: List[float] = []
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
    rewards_arr = np.asarray(rewards, dtype=float)
    explorations_arr = np.asarray(explorations, dtype=bool)
    regrets_arr = np.asarray(regrets, dtype=float)
    early_win = max(100, steps // 6)
    late_win = max(100, steps // 6)
    conv_win = max(50, steps // 20)
    early_ratio = float(np.mean(explorations_arr[:early_win]))
    late_ratio = float(np.mean(explorations_arr[-late_win:]))
    rolling_regret = np.convolve(regrets_arr, np.ones(conv_win) / conv_win, mode="valid")
    conv_step = None
    threshold = 0.05
    for idx, val in enumerate(rolling_regret):
        if val <= threshold:
            conv_step = idx + conv_win - 1
            break
    if conv_step is None:
        conv_step = steps
    avg_reward_last = float(np.mean(rewards_arr[-late_win:]))
    cum_regret = float(np.sum(regrets_arr))
    return {
        "cum_regret": cum_regret,
        "avg_reward_last": avg_reward_last,
        "early_ratio": early_ratio,
        "late_ratio": late_ratio,
        "conv_step": int(conv_step),
        "steps": int(steps),
    }


def normalize_metrics(run_metrics: List[Dict[str, Any]], run_metrics_noisy: List[Dict[str, Any]]) -> Dict[str, float]:
    avg_regret_per_step = np.mean([m["cum_regret"] / m["steps"] for m in run_metrics])
    regret_efficiency = float(np.clip(1.0 - avg_regret_per_step, 0.0, 1.0))
    average_reward = float(np.mean([m["avg_reward_last"] for m in run_metrics]))
    early_ratios = [m["early_ratio"] for m in run_metrics]
    late_ratios = [m["late_ratio"] for m in run_metrics]
    early_avg = float(np.mean(early_ratios))
    late_avg = float(np.mean(late_ratios))
    early_score = float(np.clip(early_avg / 0.05, 0.0, 1.0))
    late_score = float(np.clip(1.0 - late_avg, 0.0, 1.0))
    exploration_health = float(np.clip(0.5 * early_score + 0.5 * late_score, 0.0, 1.0))
    conv_steps = [m["conv_step"] for m in run_metrics]
    steps = [m["steps"] for m in run_metrics]
    css_values = []
    for cs, T in zip(conv_steps, steps):
        css_values.append(np.clip(1.0 - (cs / T), 0.0, 1.0))
    convergence_speed = float(np.mean(css_values))
    base_late = float(np.mean([m["avg_reward_last"] for m in run_metrics]))
    noisy_late = float(np.mean([m["avg_reward_last"] for m in run_metrics_noisy]))
    robustness = float(0.0 if base_late <= 1e-12 else np.clip(noisy_late / base_late, 0.0, 1.0))
    return {
        "regret_efficiency": regret_efficiency,
        "average_reward": average_reward,
        "exploration_health": exploration_health,
        "convergence_speed": convergence_speed,
        "robustness": robustness,
    }


def compute_bis(metrics: Dict[str, float]) -> float:
    res = metrics["regret_efficiency"]
    ars = metrics["average_reward"]
    ehs = metrics["exploration_health"]
    css = metrics["convergence_speed"]
    rs = metrics["robustness"]
    return float(0.30 * res + 0.20 * ars + 0.15 * ehs + 0.20 * css + 0.15 * rs)


def main() -> int:
    d = int(os.environ.get("LINUCB_D", "8"))
    num_actions = int(os.environ.get("LINUCB_ACTIONS", "5"))
    alpha = float(os.environ.get("LINUCB_ALPHA", "0.5"))
    steps = int(os.environ.get("LINUCB_STEPS", "2000"))
    seeds = [0, 1, 2]
    base_noise = 0.1
    noisy_noise = 1.0
    run_metrics = []
    for s in seeds:
        run_metrics.append(evaluate_single_run(s, d, num_actions, alpha, steps, base_noise))
    run_metrics_noisy = []
    for s in seeds:
        run_metrics_noisy.append(evaluate_single_run(s, d, num_actions, alpha, steps, noisy_noise))
    metrics = normalize_metrics(run_metrics, run_metrics_noisy)
    bis = compute_bis(metrics)
    status = "PASS"
    if bis < 0.65 or min(metrics.values()) < 0.4:
        status = "FAIL"
    report = {
        "algorithm": "LinUCB",
        "domain": "education",
        "metrics": metrics,
        "BIS": float(bis),
        "status": status,
        "runs": {
            "seeds": seeds,
            "d": d,
            "num_actions": num_actions,
            "alpha": alpha,
            "steps": steps,
            "base_noise": base_noise,
            "noisy_noise": noisy_noise,
        },
    }
    out_dir = os.path.join(PROJECT_ROOT, "ci")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bis_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

