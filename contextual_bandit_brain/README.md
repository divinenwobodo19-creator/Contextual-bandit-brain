# contextual_bandit_brain

## Overview
`contextual_bandit_brain` is a standalone, research-grade LinUCB decision engine with a synthetic education-focused simulator, a formal Bandit Intelligence Score (BIS), and CI-ready validation producing quantitative and visual artifacts.

## Components
- Core decision engine: LinUCB arms and brain
- Simulator: student contexts, stationary and drift environments
- BIS: reward, regret efficiency, stability, adaptability, fairness
- Reporting: publication-quality plots and JSON scorecards
- CI: `run_bis_ci.py` generates artifacts and enforces BIS thresholds

## Algorithm
LinUCB maintains per-arm parameters `A` and `b`, computes `θ = A^{-1}b` via linear solves, and selects actions by maximizing `x^Tθ + α√(x^TA^{-1}x)`. Parameters update online: `A ← A + xx^T`, `b ← b + rx`, with rewards normalized to `[0,1]`.

## BIS
Metrics in `[0,1]`:
- Reward: normalized cumulative reward vs best-expected trajectories
- Regret: inverted normalized cumulative regret
- Stability: low variance of arm distribution across time windows
- Adaptability: recovery speed and late-window normalized reward after drift
- Fairness: entropy of arm pulls relative to maximum

Education weights: Reward=0.35, Regret=0.25, Stability=0.15, Adaptability=0.15, Fairness=0.10. BIS is the weighted sum. CI fails if BIS < 0.75.

## Usage
### Decision engine
```python
import numpy as np
from contextual_bandit_brain.core.brain import LinUCBBrain

brain = LinUCBBrain(num_actions=5, alpha=0.5, d=8)
x = np.random.randn(8)
a = brain.select_action(x)
r = 0.7
brain.update(a, r, x)
```

### CI BIS
```bash
python contextual_bandit_brain/ci/run_bis_ci.py
```

Artifacts saved to `/artifacts`:
- `bis_report.json`
- `cumulative_reward.png`
- `regret_over_time.png`
- `exploration_ratio.png`
- `arm_distribution.png`
- `bis_gauge.png`

## Simulations
Run `examples/run_full_simulation.py` to validate end-to-end behavior under drift.

## Integration
Integrate `LinUCBBrain` in any backend:
- Provide numeric context vectors
- Normalize rewards to `[0,1]`
- Configure `d`, `num_actions`, and `alpha`

## Roadmap
- Additional algorithms (Thompson Sampling, Neural Bandits)
- Richer student models and domain shift patterns
- Packaging and distribution via PyPI
