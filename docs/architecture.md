# Bandit Brain Architecture

Repository layout:

contextual_bandit_brain/
- brain/: base interface, LinUCB core, config, state
- api/: DecisionService with callable and FastAPI router, schemas
- simulator/: synthetic environment, reward models, drift
- evaluation/: metrics, BIS, plots, reports
- ci/: headless benchmark runner and thresholds
- tests/: unit and behavior tests

Design principles:
- Pure math core independent of application domain
- Clear API: decide, learn, state, reset
- Deterministic seeds and reproducible simulations
- Headless visualization and JSON reports for CI
- Persistence via JSON snapshots

Integration:
- Import DecisionService or LinUCBBrain directly
- Optional FastAPI app via api.router.build_app
- CI enforces BIS via contextual_bandit_brain/ci/run_benchmark.py
