import numpy as np
import pytest

from contextual_bandit_brain.core.arm import LinUCBArm


def test_A_remains_positive_definite():
    d = 8
    arm = LinUCBArm(d)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        x = rng.normal(0.0, 1.0, size=d).astype(float)
        r = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))
        arm.update(x, r)
    # Check SPD via eigenvalues (should be strictly > 0)
    vals = np.linalg.eigvalsh(arm._A)
    assert float(np.min(vals)) > 1e-10
