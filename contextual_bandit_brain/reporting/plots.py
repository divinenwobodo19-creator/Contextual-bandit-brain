"""
Plot generation for contextual bandit evaluation.
"""

from __future__ import annotations

import os
from typing import Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_cumulative_reward(cum_reward: np.ndarray, out_dir: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(cum_reward, color="#2a9d8f")
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_reward.png"))
    plt.close()


def plot_regret(regret: np.ndarray, out_dir: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(regret, color="#e76f51")
    plt.xlabel("step")
    plt.ylabel("regret")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regret_over_time.png"))
    plt.close()


def plot_exploration_ratio(exploration_flags: np.ndarray, window: int, out_dir: str) -> None:
    ratio = []
    n = len(exploration_flags)
    for i in range(0, n - window + 1):
        w = exploration_flags[i : i + window]
        ratio.append(float(np.mean(w)))
    ratio = np.asarray(ratio, dtype=float)
    plt.figure(figsize=(8, 4))
    plt.plot(ratio, color="#264653")
    plt.xlabel("step")
    plt.ylabel("exploration ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exploration_ratio.png"))
    plt.close()


def plot_arm_distribution(counts: np.ndarray, out_dir: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(counts)), counts, color="#457b9d")
    plt.xlabel("arm")
    plt.ylabel("pulls")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "arm_distribution.png"))
    plt.close()


def plot_bis_gauge(score: float, out_dir: str) -> None:
    plt.figure(figsize=(6, 2))
    plt.barh([0], [score], color="#4caf50")
    plt.xlim(0.0, 1.0)
    plt.yticks([])
    plt.xlabel("BIS")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bis_gauge.png"))
    plt.close()

