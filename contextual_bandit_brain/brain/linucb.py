from __future__ import annotations
from typing import List, Dict, Any, Sequence, Tuple, Optional
import numpy as np
from .base_bandit import BaseBandit


class LinUCBArm:
    def __init__(self, d: int) -> None:
        if d <= 0:
            raise ValueError("feature dimension d must be positive")
        self._d = int(d)
        self._A = np.eye(self._d, dtype=float)
        self._b = np.zeros(self._d, dtype=float)

    @property
    def d(self) -> int:
        return self._d

    def reset(self) -> None:
        self._A = np.eye(self._d, dtype=float)
        self._b = np.zeros(self._d, dtype=float)

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self._A, self._b)

    def score(self, x: np.ndarray, alpha: float) -> Tuple[float, float, float]:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        theta_hat = np.linalg.solve(self._A, self._b)
        est = float(x.dot(theta_hat))
        Ax_inv_x = float(x.dot(np.linalg.solve(self._A, x)))
        unc = float(alpha * np.sqrt(max(Ax_inv_x, 0.0)))
        ucb = est + unc
        return est, unc, ucb

    def update(self, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        self._A += np.outer(x, x)
        self._b += float(reward) * x

    def to_dict(self) -> Dict[str, Any]:
        return {"d": self._d, "A": self._A.tolist(), "b": self._b.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinUCBArm":
        d = int(data["d"])
        obj = cls(d)
        obj._A = np.asarray(data["A"], dtype=float)
        obj._b = np.asarray(data["b"], dtype=float)
        return obj


def score_actions(arms: List[LinUCBArm], x: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    est = []
    unc = []
    ucb = []
    for a in arms:
        e, u, s = a.score(x, alpha)
        est.append(e)
        unc.append(u)
        ucb.append(s)
    return np.asarray(est, dtype=float), np.asarray(unc, dtype=float), np.asarray(ucb, dtype=float)


class LinUCBBrain(BaseBandit):
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
        self._arms: List[LinUCBArm] = [LinUCBArm(self._d) for _ in range(self._num_actions)]
        self._last_decision: Optional[Dict[str, Any]] = None

    def decide(self, context: Sequence[float]) -> Dict[str, Any]:
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        est, unc, ucb = score_actions(self._arms, x, self._alpha)
        chosen = int(np.argmax(ucb))
        best_est = int(np.argmax(est))
        mode = "exploitation" if chosen == best_est else "exploration"
        self._last_decision = {
            "action": chosen,
            "estimated_reward": float(est[chosen]),
            "uncertainty": float(unc[chosen]),
            "ucb": float(ucb[chosen]),
            "mode": mode,
            "context": x.tolist(),
        }
        return dict(self._last_decision)

    def learn(self, context: Sequence[float], action: int, reward: float) -> None:
        if not (0 <= action < self._num_actions):
            raise ValueError(f"action {action} out of range [0, {self._num_actions})")
        if not np.isfinite(reward):
            raise ValueError("reward must be a finite number")
        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self._d:
            raise ValueError(f"context dimension {x.shape[0]} != d={self._d}")
        r = float(np.clip(reward, 0.0, 1.0))
        self._arms[action].update(x, r)

    def reset(self) -> None:
        for a in self._arms:
            a.reset()
        self._last_decision = None

    def get_state(self) -> Dict[str, Any]:
        return {
            "alpha": float(self._alpha),
            "d": int(self._d),
            "num_actions": int(self._num_actions),
            "arms": [a.to_dict() for a in self._arms],
            "last_decision": None if self._last_decision is None else dict(self._last_decision),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "LinUCBBrain":
        obj = cls(num_actions=int(state["num_actions"]), alpha=float(state["alpha"]), d=int(state["d"]))
        obj._arms = [LinUCBArm.from_dict(a) for a in state["arms"]]
        ld = state.get("last_decision", None)
        obj._last_decision = None if ld is None else dict(ld)
        return obj
