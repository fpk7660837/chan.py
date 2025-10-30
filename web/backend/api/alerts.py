"""
Alert management endpoints and websocket stream.
"""
from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect

from ..runtime.realtime import get_realtime
from ..schemas.alerts import KLineTick

router = APIRouter()


@router.post("/feed")
async def feed_tick(payload: KLineTick, request: Request) -> Dict[str, object]:
    orchestrator = get_realtime(request.app)
    snapshot, alerts = await orchestrator.handle_tick(payload)
    return {"snapshot": snapshot, "alerts": alerts}


@router.get("/state/{symbol}/{level}")
async def fetch_state(symbol: str, level: str, request: Request) -> Dict[str, object]:
    orchestrator = get_realtime(request.app)
    return await orchestrator.fetch_state(symbol, level)


@router.get("/strategies")
async def list_strategies(request: Request) -> Dict[str, object]:
    orchestrator = get_realtime(request.app)
    return {"items": orchestrator.list_strategies()}


@router.get("/history")
async def get_history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, object]:
    orchestrator = get_realtime(request.app)
    return {"items": orchestrator.alert_dispatcher.history(limit)}


@router.websocket("/stream")
async def stream_alerts(websocket: WebSocket) -> None:
    orchestrator = get_realtime(websocket.app)
    await websocket.accept()
    queue = await orchestrator.alert_dispatcher.register()
    try:
        await websocket.send_json({"type": "history", "data": orchestrator.alert_dispatcher.history(50)})
        while True:
            alert = await queue.get()
            await websocket.send_json({"type": "alert", "data": alert.model_dump()})
    except WebSocketDisconnect:
        pass
    finally:
        await orchestrator.alert_dispatcher.unregister(queue)

