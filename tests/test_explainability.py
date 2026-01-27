import numpy as np

from brain import LinUCBBandit


def test_explainability_numeric_components_and_mode():
    d, num_actions, alpha = 5, 3, 0.6
    brain = LinUCBBandit(num_actions=num_actions, alpha=alpha, d=d)
    x = np.array([1.0, 0.5, -0.5, 0.0, 2.0])

    # Precondition: update action 0 several times to reduce its uncertainty
    for _ in range(10):
        brain.observe(x)
        a = brain.choose_action()
        brain.receive_reward(a, 0.0)  # zero reward, only A changes

    # Fresh context and decision
    x2 = np.array([-0.2, 1.3, 0.0, -0.7, 0.9])
    brain.observe(x2)
    a = brain.choose_action()
    info = brain.explain_last_decision()

    # Numeric components consistent
    assert isinstance(info["estimated_reward"], float)
    assert isinstance(info["uncertainty"], float)
    assert isinstance(info["ucb"], float)
    assert np.isclose(info["estimated_reward"] + info["uncertainty"], info["ucb"])

    # Mode classification present and valid
    assert info["mode"] in ("exploration", "exploitation")

    # Make an action with lower uncertainty more likely to be exploitation
    # Update chosen action with non-zero reward to shift theta
    brain.receive_reward(a, 1.0)
    brain.observe(x2)
    a2 = brain.choose_action()
    info2 = brain.explain_last_decision()
    assert info2["mode"] in ("exploration", "exploitation")

