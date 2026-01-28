from __future__ import annotations
import os
import sys
import yaml
import numpy as np
from ..brain.config import LinUCBConfig
from ..api.decision_service import DecisionService
from ..api.schemas import DecideRequest, LearnRequest
from ..simulator.environment import Environment
from ..evaluation.metrics import compute_all_metrics
from ..evaluation.bis import bis
from ..evaluation.plots import generate_all_plots
from ..evaluation.reports import write_reports


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(os.path.dirname(root), "ci", "thresholds.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    steps = int(cfg.get("steps", 1000))
    alpha = float(cfg.get("alpha", 0.5))
    num_actions = int(cfg.get("num_actions", 5))
    d = int(cfg.get("d", 8))
    noise_std = float(cfg.get("noise_std", 0.1))
    threshold = float(cfg.get("bis_threshold", 0.75))

    env = Environment(d=d, num_actions=num_actions, noise_std=noise_std, seed=42)
    svc = DecisionService(LinUCBConfig(num_actions=num_actions, d=d, alpha=alpha))

    rewards = []
    regrets = []
    chosen_actions = []
    exploration_flags = []
    best_exp_series = []
    best_exp_sum = 0.0
    conv_step = steps

    for t in range(steps):
        x = env.context()
        decision = svc.decide(DecideRequest(context=context_to_list(x)))
        a = int(decision.action)
        r = env.reward(a, x)
        svc.learn(LearnRequest(context=context_to_list(x), action=a, reward=r))
        rewards.append(float(r))
        chosen_actions.append(int(a))
        exploration_flags.append(1.0 if decision.diagnostics["mode"] == "exploration" else 0.0)
        best_estimated = np.max([env.reward(k, x) for k in range(num_actions)])
        regrets.append(float(best_estimated - r))
        best_exp_series.append(float(best_estimated))
        best_exp_sum += float(best_estimated)
        if t == steps // 2:
            env.drift(scale=0.5)
            conv_step = t

    rewards = np.asarray(rewards, dtype=float)
    regrets = np.asarray(regrets, dtype=float)
    chosen_actions = np.asarray(chosen_actions, dtype=int)
    exploration_flags = np.asarray(exploration_flags, dtype=float)
    best_exp_series = np.asarray(best_exp_series, dtype=float)

    metrics = compute_all_metrics(
        rewards=rewards,
        regrets=regrets,
        chosen_actions=chosen_actions,
        num_actions=num_actions,
        best_exp_sum=best_exp_sum,
        best_exp_series=best_exp_series,
        conv_step=conv_step,
        steps=steps,
    )
    score = bis(metrics)

    out_dir = os.path.join(os.path.dirname(root), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    generate_all_plots(out_dir, rewards, regrets, exploration_flags, chosen_actions, score)
    write_reports(out_dir, score, metrics)

    if score < threshold:
        return 1
    return 0


def context_to_list(x: np.ndarray) -> list[float]:
    return np.asarray(x, dtype=float).reshape(-1).tolist()


if __name__ == "__main__":
    sys.exit(main())
