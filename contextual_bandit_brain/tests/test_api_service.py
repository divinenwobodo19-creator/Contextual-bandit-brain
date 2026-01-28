from contextual_bandit_brain.api.decision_service import DecisionService
from contextual_bandit_brain.brain.config import LinUCBConfig
from contextual_bandit_brain.api.schemas import DecideRequest, LearnRequest


def test_service_flow_basic():
    svc = DecisionService(LinUCBConfig(num_actions=3, d=4, alpha=0.5))
    x = [0.1, -0.2, 0.3, 0.4]
    resp = svc.decide(DecideRequest(context=x))
    assert 0 <= resp.action < 3
    assert 0.0 <= resp.confidence <= 1.0
    svc.learn(LearnRequest(context=x, action=int(resp.action), reward=0.7))
    assert "arms" in svc.get_state().state
