from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence


class BaseBandit(ABC):
    @abstractmethod
    def decide(self, context: Sequence[float]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def learn(self, context: Sequence[float], action: int, reward: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError
