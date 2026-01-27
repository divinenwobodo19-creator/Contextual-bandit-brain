import numpy as np
import pytest

from contextual_bandit_brain.bis.metrics import (
    compute_reward_score,
    compute_regret_efficiency,
    compute_stability,
    compute_adaptability,
    compute_fairness,
)
from contextual_bandit_brain.bis.normalization import clamp01, safe_ratio


def test_safe_ratio_and_clamp01():
    assert safe_ratio(5.0, 0.0) == 0.0
    assert clamp01(-0.2) == 0.0
    assert clamp01(0.5) == 0.5
    assert clamp01(1.5) == 1.0


def test_reward_and_regret_scores():
    rewards = np.ones(100, dtype=float) * 0.5  # sum = 50
    best_exp_sum = 100.0
    r_score = compute_reward_score(rewards, best_exp_sum)
    assert np.isclose(r_score, 0.5, atol=1e-6)

    regrets = np.ones(100, dtype=float) * 0.3  # sum = 30
    reg_eff = compute_regret_efficiency(regrets, best_exp_sum)
    assert np.isclose(reg_eff, 0.7, atol=1e-6)


def test_stability_uniform_vs_concentrated():
    num_actions = 5
    uniform = np.tile(np.arange(num_actions), 200).astype(int)  # length 1000, evenly distributed
    concentrated = np.zeros(1000, dtype=int)  # all pulls to arm 0
    stab_uniform = compute_stability(uniform, num_actions=num_actions, window=100)
    stab_conc = compute_stability(concentrated, num_actions=num_actions, window=100)
    assert 0.0 <= stab_uniform <= 1.0
    assert 0.0 <= stab_conc <= 1.0
    assert stab_uniform > stab_conc


def test_adaptability_conv_step_influence():
    steps = 1000
    rewards = np.ones(steps, dtype=float) * 0.9
    best_exp_series = np.ones(steps, dtype=float) * 1.0
    adapt_low_conv = compute_adaptability(rewards, best_exp_series, conv_step=50, steps=steps)
    adapt_high_conv = compute_adaptability(rewards, best_exp_series, conv_step=900, steps=steps)
    assert 0.0 <= adapt_low_conv <= 1.0
    assert 0.0 <= adapt_high_conv <= 1.0
    assert adapt_low_conv > adapt_high_conv


def test_fairness_balanced_vs_unbalanced():
    num_actions = 4
    balanced = np.tile(np.arange(num_actions), 250).astype(int)  # equal pulls
    unbalanced = np.zeros(1000, dtype=int)
    fair_bal = compute_fairness(balanced, num_actions=num_actions)
    fair_unbal = compute_fairness(unbalanced, num_actions=num_actions)
    assert 0.0 <= fair_bal <= 1.0
    assert 0.0 <= fair_unbal <= 1.0
    assert fair_bal > fair_unbal
