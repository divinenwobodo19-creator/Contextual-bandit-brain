# LinUCB Math

Let x ∈ R^d be the context vector and a ∈ {0,…,K−1} index arms.

Per arm a maintain:
- A_a = I_d + ∑ x x^T
- b_a = ∑ r x

Parameter estimate:
- θ_a = A_a^{-1} b_a solved via A_a θ_a = b_a

Upper Confidence Bound:
- p_a = θ_a^T x + α √(x^T A_a^{-1} x)

Update on (x, a, r):
- A_a ← A_a + x x^T
- b_a ← b_a + r x

Notes:
- Use linear solves instead of explicit inverse for numerical stability
- Rewards are clamped to [0,1] to decouple external semantics
