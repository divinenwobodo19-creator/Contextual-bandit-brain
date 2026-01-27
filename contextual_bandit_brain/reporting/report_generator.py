"""
Report generation orchestrating BIS metrics and plots.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any
import numpy as np

from ..reporting.plots import (
    plot_cumulative_reward,
    plot_regret,
    plot_exploration_ratio,
    plot_arm_distribution,
    plot_bis_gauge,
)


def write_json_report(out_dir: str, bis_score: float, metrics: Dict[str, float]) -> str:
    data = {
        "bis_score": float(bis_score),
        "reward": float(metrics["reward"]),
        "regret": float(metrics["regret"]),
        "stability": float(metrics["stability"]),
        "adaptability": float(metrics["adaptability"]),
        "fairness": float(metrics["fairness"]),
        "pass": bool(bis_score >= 0.75),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bis_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def generate_plots(out_dir: str, rewards: np.ndarray, regrets: np.ndarray, exploration_flags: np.ndarray, chosen_actions: np.ndarray, bis_score: float) -> None:
    cum_reward = np.cumsum(rewards)
    plot_cumulative_reward(cum_reward, out_dir)
    plot_regret(regrets, out_dir)
    plot_exploration_ratio(exploration_flags, window=max(50, len(exploration_flags) // 50), out_dir=out_dir)
    counts = np.bincount(chosen_actions, minlength=int(np.max(chosen_actions) + 1)).astype(int)
    plot_arm_distribution(counts, out_dir)
    plot_bis_gauge(bis_score, out_dir)

