import os
import json
import numpy as np
import pytest

from contextual_bandit_brain.core.brain import LinUCBBrain


def test_brain_save_and_load(tmp_path):
    path = os.path.join(tmp_path, "brain_state.json")
    brain = LinUCBBrain(num_actions=3, alpha=0.5, d=4)
    x = np.array([0.1, -0.2, 0.3, 0.4], dtype=float)
    a = brain.select_action(x)
    r = 0.8
    brain.update(a, r, x)
    brain.save_state(path)
    brain2 = LinUCBBrain.load_state(path)
    y = np.array([0.1, -0.2, 0.3, 0.4], dtype=float)
    a2 = brain2.select_action(y)
    assert isinstance(a2, int)
