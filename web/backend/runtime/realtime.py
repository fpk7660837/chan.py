"""
Application-level orchestration for the real-time alert pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from fastapi import FastAPI

from ..schemas.alerts import KLineTick
from ..services.chan_trigger import ChanTriggerSession, LEVEL_TO_KL_TYPE
from ..services.state_cache import StateCache
from ..services.alert_dispatcher import AlertDispatcher
from ..strategies import StrategyContext, StrategyRegistry, SimpleBreakoutStrategy


class RealTimeOrchestrator:
    """Coordinates Chan trigger sessions, strategy evaluations, and alert fan-out."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ChanTriggerSession] = {}
        self.state_cache = StateCache()
        self.alert_dispatcher = AlertDispatcher()
        self.strategy_registry = StrategyRegistry([SimpleBreakoutStrategy()])

    def _session_key(self, symbol: str) -> str:
        return symbol

    def _ensure_session(self, symbol: str, level: str) -> ChanTriggerSession:
        level_enum = LEVEL_TO_KL_TYPE.get(level)
        if level_enum is None:
            raise ValueError(f"Unsupported level {level}")
        key = self._session_key(symbol)
        session = self._sessions.get(key)
        if session is None:
            session = ChanTriggerSession(symbol=symbol, lv_list=[level_enum])
            self._sessions[key] = session
        return session

    async def handle_tick(self, payload: KLineTick) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        """Feed a tick, run strategies, and return snapshot plus generated alerts."""
        session = self._ensure_session(payload.symbol, payload.level)
        raw_tick = payload.model_dump()
        snapshot = session.feed(payload.level, raw_tick)

        tick_json = payload.model_dump(mode="json")
        await self.state_cache.append(payload.symbol, payload.level, tick_json, snapshot)
        recent = await self.state_cache.get_recent(payload.symbol, payload.level, limit=50)

        ctx = StrategyContext(
            symbol=payload.symbol,
            level=payload.level,
            tick=tick_json,
            recent_ticks=recent,
            snapshot=snapshot,
        )
        alerts_models = self.strategy_registry.evaluate(ctx)
        serialized_alerts = []
        for alert in alerts_models:
            serialized_alerts.append(alert.model_dump())
            await self.alert_dispatcher.publish(alert)

        return snapshot, serialized_alerts

    async def fetch_state(self, symbol: str, level: str) -> Dict[str, object]:
        snapshot = await self.state_cache.get_snapshot(symbol, level)
        recent = await self.state_cache.get_recent(symbol, level, limit=50)
        return {
            "symbol": symbol,
            "level": level,
            "snapshot": snapshot,
            "recent": recent,
        }

    def list_strategies(self) -> List[Dict[str, object]]:
        return self.strategy_registry.list_strategies()


def setup_realtime(app: FastAPI) -> None:
    """Attach orchestrator to FastAPI application lifecycle."""
    orchestrator = RealTimeOrchestrator()
    app.state.realtime = orchestrator

    @app.on_event("shutdown")
    async def _cleanup() -> None:
        await orchestrator.alert_dispatcher.reset()


def get_realtime(app: FastAPI) -> RealTimeOrchestrator:
    orchestrator = getattr(app.state, "realtime", None)
    if orchestrator is None:
        raise RuntimeError("Real-time orchestrator not initialized")
    return orchestrator

