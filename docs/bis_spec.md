# Bandit Intelligence Score (BIS) Spec

BIS = 0.35·R + 0.25·(1−Ĝ) + 0.15·S + 0.15·A + 0.10·F

Components (normalized to [0,1]):
- R: cumulative reward normalized by best-expected sum
- Ĝ: cumulative regret normalized by best-expected sum
- S: stability via variance penalty of arm distribution over windows
- A: adaptability under drift combining late-window normalized reward and convergence speed
- F: fairness measured by entropy of arm pulls vs log(K)

Normalization:
- safe_ratio(num, den) with den≥1e−12, else 0
- clamp01 after each metric

Thresholds:
- CI fails if BIS < configurable threshold in ci/thresholds.yaml

Artifacts:
- PNG plots: reward, regret, exploration ratio, arm histogram, BIS gauge
- JSON: bis_report.json
