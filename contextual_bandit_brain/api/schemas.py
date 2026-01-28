from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


SCHEMA_VERSION = "1.0"


@dataclass
class DecideRequest:
    context: List[float]
    metadata: Optional[Dict[str, Any]] = None
    version: str = SCHEMA_VERSION


@dataclass
class DecideResponse:
    action: int
    confidence: float
    diagnostics: Dict[str, Any]
    version: str = SCHEMA_VERSION


@dataclass
class LearnRequest:
    context: List[float]
    action: int
    reward: float
    version: str = SCHEMA_VERSION


@dataclass
class StateResponse:
    state: Dict[str, Any]
    version: str = SCHEMA_VERSION


def validate_context(x: List[float]) -> None:
    if not isinstance(x, list):
        raise ValueError("context must be a list of floats")
    if len(x) == 0:
        raise ValueError("context cannot be empty")
    for v in x:
        if not isinstance(v, (int, float)):
            raise ValueError("context elements must be numeric")
