import numpy as np
from contextual_bandit_brain.evaluation.metrics import compute_all_metrics


def test_metric_accuracy_shapes_and_ranges():
    rewards = np.random.rand(1000).astype(float)
    regrets = np.random.rand(1000).astype(float)
    chosen = np.random.randint(0, 5, size=1000).astype(int)
    best_exp_series = np.random.rand(1000).astype(float)
    metrics = compute_all_metrics(
        rewards=rewards,
        regrets=regrets,
        chosen_actions=chosen,
        num_actions=5,
        best_exp_sum=float(np.sum(best_exp_series)),
        best_exp_series=best_exp_series,
        conv_step=500,
        steps=1000,
    )
    for v in metrics.values():
        assert 0.0 <= float(v) <= 1.0
