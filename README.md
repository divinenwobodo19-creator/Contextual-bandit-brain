# Contextual Bandit Brain (LinUCB)

## What It Is
- Production-ready, standalone decision brain implementing LinUCB
- Operates via Python import or REST (FastAPI optional)
- Includes simulator, metrics, BIS benchmark, headless plots, and CI integration

## Run Standalone
```python
from contextual_bandit_brain.brain.linucb import LinUCBBrain
brain = LinUCBBrain(num_actions=5, alpha=0.5, d=8)
x = [0.1, -0.2, 0.3, 0.4, 0.0, 0.2, -0.1, 0.5]
decision = brain.decide(x)
brain.learn(x, decision["action"], reward=0.8)
```

## Decision Service API
- POST /decide: {context:[...], metadata?:{}} → {action, confidence, diagnostics}
- POST /learn: {context:[...], action, reward} → {status:ok}
- GET /state → model state
- POST /reset → reset brain

Start REST (optional):
```python
from contextual_bandit_brain.api.router import build_app
app = build_app(num_actions=5, d=8, alpha=0.5)
# run with uvicorn app:app --host 0.0.0.0 --port 8000
```

## Simulation & BIS
Headless CI benchmark:
```bash
python -m contextual_bandit_brain.ci.run_benchmark
```
Outputs to ./artifacts:
- bis_report.json
- cumulative_reward.png, regret_over_time.png, exploration_ratio.png
- arm_distribution.png, bis_gauge.png

## Determinism and Tests
- Seeds respected across simulator
- Tests under contextual_bandit_brain/tests and ./tests validate behavior and metrics

## Documentation
- docs/math.md: LinUCB equations
- docs/architecture.md: system layout
- docs/bis_spec.md: BIS computation and artifacts
