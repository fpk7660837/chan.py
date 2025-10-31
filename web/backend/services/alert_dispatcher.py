"""
Alert dispatcher responsible for buffering strategy signals and broadcasting them to listeners.
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Iterable, List, Set

from strategies.base import AlertSignal


class AlertDispatcher:
    """Fan-out dispatcher with history retention and simple cooldown."""

    def __init__(self, history_size: int = 200, cooldown_seconds: int = 30) -> None:
        self._history: Deque[AlertSignal] = deque(maxlen=history_size)
        self._listeners: Set[asyncio.Queue[AlertSignal]] = set()
        self._last_emit: Dict[tuple[str, str], datetime] = {}
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._lock = asyncio.Lock()

    async def publish(self, signal: AlertSignal) -> bool:
        """
        Publish a signal to all listeners.
        Returns True when the signal is dispatched, False if dropped due to cooldown.
        """
        now = datetime.now(timezone.utc)
        key = (signal.symbol, signal.strategy_id)
        async with self._lock:
            last_emit = self._last_emit.get(key)
            if last_emit and now - last_emit < self._cooldown:
                return False
            self._last_emit[key] = now
            self._history.appendleft(signal)
            listeners = list(self._listeners)

        for queue in listeners:
            await queue.put(signal)
        return True

    async def register(self) -> asyncio.Queue[AlertSignal]:
        """Register a new listener and return its queue."""
        queue: asyncio.Queue[AlertSignal] = asyncio.Queue()
        async with self._lock:
            self._listeners.add(queue)
        return queue

    async def unregister(self, queue: asyncio.Queue[AlertSignal]) -> None:
        """Remove a listener queue."""
        async with self._lock:
            self._listeners.discard(queue)

    def history(self, limit: int = 50) -> List[Dict[str, object]]:
        """Return recent alerts as serializable dicts."""
        items: Iterable[AlertSignal]
        items = list(self._history)[:limit]
        return [signal.model_dump() for signal in items]

    async def reset(self) -> None:
        """Clear internal buffers (mostly useful for tests)."""
        async with self._lock:
            self._history.clear()
            self._listeners.clear()
            self._last_emit.clear()
