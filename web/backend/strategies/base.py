"""
Strategy primitives for the real-time alert engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Sequence

from pydantic import BaseModel, Field


class AlertSignal(BaseModel):
    """Serializable alert data returned to clients."""

    symbol: str
    level: str
    strategy_id: str
    strategy_name: str
    message: str
    severity: Literal["info", "warning", "critical"] = "info"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class StrategyContext:
    """Execution context passed to strategies each time a new tick arrives."""

    symbol: str
    level: str
    tick: Dict[str, Any]
    recent_ticks: Sequence[Dict[str, Any]]
    snapshot: Dict[str, Any]


class StrategyBase:
    """Abstract base class for concrete alert strategies."""

    id: str = "strategy.base"
    name: str = "Base Strategy"
    description: str = ""

    def evaluate(self, ctx: StrategyContext) -> List[AlertSignal]:
        """
        Evaluate a strategy with the provided context and return zero or more alerts.
        Subclasses must override this method.
        """
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """Return public metadata for front-end configuration displays."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }

