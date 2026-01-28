from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from ..brain.linucb import LinUCBBrain
from ..brain.config import LinUCBConfig
from ..brain.state import StateManager
from .schemas import DecideRequest, DecideResponse, LearnRequest, StateResponse, validate_context


class DecisionService:
    def __init__(self, config: LinUCBConfig) -> None:
        self._brain = LinUCBBrain(num_actions=config.num_actions, alpha=config.alpha, d=config.d)
        self._state = StateManager()

    def decide(self, req: DecideRequest) -> DecideResponse:
        validate_context(req.context)
        decision = self._brain.decide(req.context)
        conf = 1.0 / (1.0 + float(decision["uncertainty"]))
        return DecideResponse(action=int(decision["action"]), confidence=float(conf), diagnostics=decision)

    def learn(self, req: LearnRequest) -> None:
        validate_context(req.context)
        self._brain.learn(req.context, int(req.action), float(req.reward))

    def get_state(self) -> StateResponse:
        return StateResponse(state=self._brain.get_state())

    def reset(self) -> None:
        self._brain.reset()

    def save(self, path: str) -> None:
        self._state.save(path, self._brain.get_state())

    def load(self, path: str) -> None:
        st = self._state.load(path)
        self._brain = LinUCBBrain.from_state(st)


try:
    from fastapi import FastAPI
    from fastapi import APIRouter
    from pydantic import BaseModel

    class DecideModel(BaseModel):
        context: list[float]
        metadata: Optional[dict[str, Any]] = None

    class LearnModel(BaseModel):
        context: list[float]
        action: int
        reward: float

    def create_app(config: LinUCBConfig) -> FastAPI:
        svc = DecisionService(config)
        app = FastAPI()
        router = APIRouter()

        @router.post("/decide")
        def decide(req: DecideModel) -> Dict[str, Any]:
            resp = svc.decide(DecideRequest(context=req.context, metadata=req.metadata))
            return {"action": resp.action, "confidence": resp.confidence, "diagnostics": resp.diagnostics}

        @router.post("/learn")
        def learn(req: LearnModel) -> Dict[str, Any]:
            svc.learn(LearnRequest(context=req.context, action=req.action, reward=req.reward))
            return {"status": "ok"}

        @router.get("/state")
        def state() -> Dict[str, Any]:
            return svc.get_state().state

        @router.post("/reset")
        def reset() -> Dict[str, Any]:
            svc.reset()
            return {"status": "ok"}

        app.include_router(router)
        return app
except Exception:
    create_app = None
