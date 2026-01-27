# LinUCB Brain Testing Suite

## Purpose
- Establish trust in a project-agnostic LinUCB decision brain through deterministic, failure-oriented tests.
- Validate mathematical correctness, API discipline, learning behavior, explainability, persistence, robustness, and domain agnosticism.

## Structure
- Unit: test_unit.py — validates algorithmic primitives and updates.
- API: test_api.py — enforces strict loop and interface invariants.
- Simulation: test_simulation.py — verifies learning, exploration dynamics, and regret reduction.
- Cross-Domain: test_cross_domain.py — confirms domain-agnostic behavior across distributions.
- Stress: test_stress.py — high d/N, noisy and sparse reward scenarios.
- Persistence: test_persistence.py — save/load equivalence and continuity.
- Explainability: test_explainability.py — numeric consistency and mode classification.

## What Each Test Proves
- Initialization: A=I_d, b=0, dimensions correct, independent per-action parameters.
- UCB Calculation: Matches xᵀθ̂ + α√(xᵀA⁻¹x) with manual linear solves.
- Action Selection: Deterministic tie-breaking via argmax; repeatable results.
- Update Rule: A ← A + xxᵀ, b ← b + rx numeric equality.
- Reward Validation: [0,1] clamp; rejects NaN; edge-case handling.
- Learning Behavior: Average reward increases; exploration decreases over time; regret trend downward.
- Domain Agnosticism: Learning improves across uniform/normal/laplace contexts; no domain assumptions.
- API Discipline: observe→choose→receive enforced; dimension mismatch and wrong-action rewards rejected; reset clears state.
- Robustness: No crashes; stable learning with noisy/sparse rewards; non-decreasing performance windows.
- Persistence: Parameters unchanged on reload; decisions consistent on fixed context; continued learning post-reload.
- Explainability: Estimated reward, uncertainty, UCB additivity; valid exploration/exploitation classification.

## Running
- Requirements: Python, numpy, pytest
- Commands:
  - `python -m pytest -q`
  - `python -m pytest -q --junitxml=tests/report.xml` (produces JUnit report)

## Philosophy
- Fail loudly and early; prioritize mathematical invariants and strict loop discipline.
- Control randomness via seeds; tolerate stochastic variance via small numeric windows where appropriate.
