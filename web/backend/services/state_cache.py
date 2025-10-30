"""
In-memory state cache for real-time K-line streams.
Keeps a rolling window of recent ticks and the latest Chan snapshot per (symbol, level).
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Deque, Dict, List, Tuple


class StateCache:
    """Thread-safe cache storing recent ticks and Chan snapshots."""

    def __init__(self, max_points: int = 500) -> None:
        self.max_points = max_points
        self._ticks: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = {}
        self._snapshots: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def append(
        self,
        symbol: str,
        level: str,
        tick: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> None:
        """Append a new tick and update the cached snapshot."""
        key = (symbol, level)
        async with self._lock:
            series = self._ticks.setdefault(key, deque(maxlen=self.max_points))
            series.append(tick)
            self._snapshots[key] = snapshot

    async def get_recent(self, symbol: str, level: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return at most `limit` recent ticks for the given symbol/level."""
        key = (symbol, level)
        async with self._lock:
            items = list(self._ticks.get(key, ()))
        if limit <= 0:
            return items
        return items[-limit:]

    async def get_snapshot(self, symbol: str, level: str) -> Dict[str, Any] | None:
        """Return the latest Chan snapshot or None if absent."""
        key = (symbol, level)
        async with self._lock:
            snapshot = self._snapshots.get(key)
        if snapshot is None:
            return None
        return dict(snapshot)

    async def tracked_symbols(self) -> List[str]:
        """List symbols currently tracked in the cache."""
        async with self._lock:
            symbols = {symbol for symbol, _ in self._ticks.keys()}
        return sorted(symbols)

