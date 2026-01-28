from __future__ import annotations
from typing import Dict, Any
import json


class StateManager:
    def save(self, path: str, state: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
