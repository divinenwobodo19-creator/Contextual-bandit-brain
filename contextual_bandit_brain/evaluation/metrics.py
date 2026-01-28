from __future__ import annotations
from typing import Dict
import numpy as np
from ..bis.metrics import (
    compute_reward_score,
    compute_regret_efficiency,
    compute_stability,
    compute_adaptability,
    compute_fairness,
)


def compute_all_metrics(
    rewards: np.ndarray,
    regrets: np.ndarray,
    chosen_actions: np.ndarray,
    num_actions: int,
    best_exp_sum: float,
    best_exp_series: np.ndarray,
    conv_step: int,
    steps: int,
) -> Dict[str, float]:
    return {
        "reward": float(compute_reward_score(rewards, best_exp_sum)),
        "regret": float(compute_regret_efficiency(regrets, best_exp_sum)),
        "stability": float(compute_stability(chosen_actions, num_actions)),
        "adaptability": float(compute_adaptability(rewards, best_exp_series, conv_step, steps)),
        "fairness": float(compute_fairness(chosen_actions, num_actions)),
    }
