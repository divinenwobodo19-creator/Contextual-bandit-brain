import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brain import LinUCBBandit
from simulator import ContextualBanditSimulator
from actions import LinUCBAction


@pytest.fixture
def default_params():
    return {"d": 5, "num_actions": 3, "alpha": 0.5, "seed": 42}


@pytest.fixture
def brain(default_params):
    return LinUCBBandit(num_actions=default_params["num_actions"], alpha=default_params["alpha"], d=default_params["d"])


@pytest.fixture
def env(default_params):
    return ContextualBanditSimulator(
        d=default_params["d"], num_actions=default_params["num_actions"], noise_std=0.1, seed=default_params["seed"]
    )


def manual_ucb(A: np.ndarray, b: np.ndarray, x: np.ndarray, alpha: float) -> float:
    theta = np.linalg.solve(A, b)
    est = float(x.dot(theta))
    Ax_inv_x = float(x.dot(np.linalg.solve(A, x)))
    return est + float(alpha * np.sqrt(max(Ax_inv_x, 0.0)))
