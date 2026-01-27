"""
BIS metrics for contextual bandits.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np

from .normalization import clamp01, safe_ratio


def entropy(probs: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.nansum(probs * np.log(probs + 1e-12)))


def compute_reward_score(rewards: np.ndarray, best_exp_sum: float) -> float:
    cum_reward = float(np.sum(rewards))
    return clamp01(safe_ratio(cum_reward, best_exp_sum))


def compute_regret_efficiency(regrets: np.ndarray, best_exp_sum: float) -> float:
    cum_regret = float(np.sum(regrets))
    return clamp01(1.0 - safe_ratio(cum_regret, best_exp_sum))


def compute_stability(chosen_actions: np.ndarray, num_actions: int, window: int = 200) -> float:
    counts = np.bincount(chosen_actions, minlength=num_actions).astype(float)
    probs = counts / max(len(chosen_actions), 1)
    rolling = []
    for i in range(0, len(chosen_actions) - window + 1, window):
        c = np.bincount(chosen_actions[i : i + window], minlength=num_actions).astype(float)
        p = c / max(window, 1)
        rolling.append(p)
    if not rolling:
        rolling.append(probs)
    rolling = np.asarray(rolling, dtype=float)
    var = float(np.mean(np.var(rolling, axis=1)))
    return clamp01(1.0 - min(var * num_actions, 1.0))


def compute_adaptability(rewards: np.ndarray, best_exp_series: np.ndarray, conv_step: int, steps: int) -> float:
    last_win = max(100, steps // 10)
    avg_best_last = float(np.mean(best_exp_series[-last_win:]))
    avg_reward_last = float(np.mean(rewards[-last_win:]))
    normalized_last = clamp01(safe_ratio(avg_reward_last, avg_best_last))
    conv_component = clamp01(1.0 - safe_ratio(conv_step, steps))
    return clamp01(0.5 * normalized_last + 0.5 * conv_component)


def compute_fairness(chosen_actions: np.ndarray, num_actions: int) -> float:
    counts = np.bincount(chosen_actions, minlength=num_actions).astype(float)
    probs = counts / max(len(chosen_actions), 1)
    ent = entropy(probs)
    max_ent = np.log(max(num_actions, 2))
    return clamp01(safe_ratio(ent, max_ent))

