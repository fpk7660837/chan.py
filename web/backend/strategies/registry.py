"""
Registry to manage strategy lifecycle and evaluation.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .base import AlertSignal, StrategyBase, StrategyContext


class StrategyRegistry:
    """Stores registered strategies and coordinates evaluation."""

    def __init__(self, strategies: Sequence[StrategyBase]) -> None:
        self._strategies: Dict[str, StrategyBase] = {strategy.id: strategy for strategy in strategies}

    def evaluate(self, ctx: StrategyContext) -> List[AlertSignal]:
        alerts: List[AlertSignal] = []
        for strategy in self._strategies.values():
            alerts.extend(strategy.evaluate(ctx))
        return alerts

    def list_strategies(self) -> List[Dict[str, object]]:
        return [strategy.describe() for strategy in self._strategies.values()]

    def __iter__(self) -> Iterable[StrategyBase]:
        return iter(self._strategies.values())

