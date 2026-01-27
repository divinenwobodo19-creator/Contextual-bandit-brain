import numpy as np
import pytest

from ci.bis_benchmark import run_benchmark, run_random_baseline


def test_bis_threshold_and_baseline_superiority():
    d, num_actions, alpha = 8, 5, 0.5
    steps1, steps2 = 1000, 1000
    result = run_benchmark(d=d, num_actions=num_actions, alpha=alpha, steps_phase1=steps1, steps_phase2=steps2, seeds=[0, 1])
    assert result["BIS"] >= 0.75

    baseline = run_random_baseline(d=d, num_actions=num_actions, alpha=alpha, steps=steps1, seed=0)
    assert result["metrics"]["normalized_cumulative_reward"] >= baseline["metrics"]["normalized_cumulative_reward"]
    assert result["metrics"]["regret_efficiency"] >= baseline["metrics"]["regret_efficiency"]
