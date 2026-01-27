"""
Standalone LinUCB Decision Brain.

Intent:
- Provide a project-agnostic contextual bandit brain using LinUCB.
- Expose a minimal decision API: observe → choose_action → receive_reward → reset.
- Maintain per-action parameters and learn online from scalar rewards.
- Offer inspectability and optional persistence without domain assumptions.
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from actions import LinUCBAction


class LinUCBBandit:
    """
    Contextual bandit brain implementing LinUCB across N discrete actions.

    Design decisions:
    - Each action maintains its own LinUCB state (A, b), ensuring independence.
    - θ̂ is obtained via linear solve (A θ̂ = b) for numerical stability.
    - Uncertainty term uses xᵀ A⁻¹ x computed with a linear solve, avoiding explicit inverse.
    - Reward inputs are normalized to [0, 1] by clamping, keeping semantics external.
    - Strict decision loop enforced by clearing the stored context after each update.
    """

    def __init__(self, num_actions: int, alpha: float, d: int) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if d <= 0:
            raise ValueError("feature dimension d must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self._num_actions = int(num_actions)
        self._alpha = float(alpha)
        self._d = int(d)
        self._actions: List[LinUCBAction] = [LinUCBAction(self._d) for _ in range(self._num_actions)]
        self._context: Optional[np.ndarray] = None
        self._last_decision: Optional[Dict[str, Any]] = None
        self._last_action: Optional[int] = None
        self._t: int = 0

    @property
    def d(self) -> int:
        return self._d

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def alpha(self) -> float:
        return self._alpha

    def reset(self) -> None:
        """Reinitialize all action states and clear internal counters and context."""
        for a in self._actions:
            a.reset()
        self._context = None
        self._last_decision = None
        self._last_action = None
        self._t = 0

    def observe(self, context: np.ndarray) -> None:
        """
        Capture the current context vector x ∈ R^d.

        Validation:
        - 1-D float array of length d.
        - Stored until a reward is applied, then cleared to honor the strict loop.
        """
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        self._context = x

    def choose_action(self) -> int:
        """
        Evaluate UCB scores for all actions and select the argmax.

        Returns:
        - Integer action ID in [0, num_actions).

        Explainability:
        - Stores estimated reward, uncertainty, and exploration/exploitation mode for inspection.
        """
        if self._context is None:
            raise RuntimeError("observe(context) must be called before choose_action()")
        if self._last_action is not None:
            raise RuntimeError("Pending reward for previous action; call receive_reward() before choosing again")

        x = self._context
        est_rewards: List[float] = []
        uncertainties: List[float] = []
        ucbs: List[float] = []
        for a in self._actions:
            est, unc, ucb = a.score(x, self._alpha)
            est_rewards.append(est)
            uncertainties.append(unc)
            ucbs.append(ucb)

        chosen = int(np.argmax(ucbs))
        best_estimated = int(np.argmax(est_rewards))
        mode = "exploitation" if chosen == best_estimated else "exploration"

        self._last_decision = {
            "action": chosen,
            "estimated_reward": float(est_rewards[chosen]),
            "uncertainty": float(uncertainties[chosen]),
            "ucb": float(ucbs[chosen]),
            "mode": mode,
            "context": x.tolist(),
            "t": self._t,
        }
        self._last_action = chosen
        return chosen

    def receive_reward(self, action: int, reward: float) -> None:
        """
        Apply scalar reward for the previously chosen action and update its parameters.

        Loop discipline:
        - Updates happen immediately: A ← A + x xᵀ, b ← b + r x.
        - Reward is clamped to [0, 1] to keep semantics external.
        - Clears the stored context to enforce the observe→act→update cadence.
        """
        if self._context is None:
            raise RuntimeError("No context to update on; call observe() and choose_action() first")
        if not (0 <= action < self._num_actions):
            raise ValueError(f"action {action} out of range [0, {self._num_actions})")
        if self._last_action is None:
            raise RuntimeError("No pending action to update; call choose_action() first")
        if int(action) != int(self._last_action):
            raise ValueError("Received reward for an action different from the last chosen action")
        if not np.isfinite(reward):
            raise ValueError("reward must be a finite number")

        r = float(np.clip(reward, 0.0, 1.0))
        self._actions[action].update(self._context, r)
        self._t += 1
        self._context = None
        self._last_action = None

    def explain_last_decision(self) -> Dict[str, Any]:
        """
        Return inspectable details of the most recent decision:
        - chosen action
        - estimated reward
        - uncertainty term
        - exploration vs exploitation mode
        """
        if self._last_decision is None:
            raise RuntimeError("No decision to explain; call choose_action() first")
        return dict(self._last_decision)

    def save_state(self, file_path: str) -> None:
        """
        Persist brain state to a JSON file.

        Contents:
        - alpha, d, num_actions, t
        - per-action A and b
        - last context (if present) and last decision metadata
        """
        snapshot = {
            "alpha": self._alpha,
            "d": self._d,
            "num_actions": self._num_actions,
            "t": self._t,
            "actions": [a.to_dict() for a in self._actions],
            "context": None if self._context is None else self._context.tolist(),
            "last_decision": self._last_decision,
            "last_action": self._last_action,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)

    def load_state(self, file_path: str) -> None:
        """
        Load brain state from a JSON file.

        Intent:
        - Rehydrate actions and metadata exactly as saved.
        - Validate structural consistency (d and num_actions).
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        d = int(data["d"])
        num_actions = int(data["num_actions"])
        if d != self._d or num_actions != self._num_actions:
            raise ValueError("Loaded state incompatible with current brain configuration")
        self._alpha = float(data["alpha"])
        self._t = int(data["t"])
        self._actions = [LinUCBAction.from_dict(a) for a in data["actions"]]
        ctx = data.get("context", None)
        self._context = None if ctx is None else np.asarray(ctx, dtype=float).reshape(-1)
        self._last_decision = data.get("last_decision", None)
        last_action = data.get("last_action", None)
        self._last_action = None if last_action is None else int(last_action)
