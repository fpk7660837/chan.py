"""
Baseline breakout strategy for demonstration purposes.
"""
from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Dict, List

from .base import AlertSignal, StrategyBase, StrategyContext


class SimpleBreakoutStrategy(StrategyBase):
    """
    Emits alerts when the latest close deviates from the moving average by a configurable ratio.
    """

    id = "strategy.simple_breakout.v1"
    name = "简单突破策略"
    description = "最新收盘价相对最近均线超过阈值时触发报警。"

    def __init__(self, window: int = 12, threshold: float = 0.006) -> None:
        self.window = max(3, window)
        self.threshold = abs(threshold)

    def evaluate(self, ctx: StrategyContext) -> List[AlertSignal]:
        recent = list(ctx.recent_ticks)
        if len(recent) < self.window:
            return []

        closes: List[float] = []
        for entry in recent[-self.window :]:
            close_value = entry.get("close")
            if close_value is None:
                return []
            closes.append(float(close_value))
        latest_close = closes[-1]
        avg_close = mean(closes[:-1]) if len(closes) > 1 else closes[-1]
        if avg_close == 0:
            return []

        deviation = (latest_close - avg_close) / avg_close
        timestamp_raw = recent[-1].get("time")
        timestamp = self._parse_timestamp(timestamp_raw)

        alerts: List[AlertSignal] = []
        if deviation >= self.threshold:
            alerts.append(
                AlertSignal(
                    symbol=ctx.symbol,
                    level=ctx.level,
                    strategy_id=self.id,
                    strategy_name=self.name,
                    severity="warning",
                    message=f"价格突破均线 {latest_close:.2f} > {avg_close:.2f} (+{deviation*100:.2f}%)",
                    metadata={
                        "threshold": self.threshold,
                        "window": self.window,
                        "timestamp": timestamp.isoformat() if timestamp else None,
                    },
                )
            )
        elif deviation <= -self.threshold:
            alerts.append(
                AlertSignal(
                    symbol=ctx.symbol,
                    level=ctx.level,
                    strategy_id=self.id,
                    strategy_name=self.name,
                    severity="info",
                    message=f"价格跌破均线 {latest_close:.2f} < {avg_close:.2f} ({deviation*100:.2f}%)",
                    metadata={
                        "threshold": self.threshold,
                        "window": self.window,
                        "timestamp": timestamp.isoformat() if timestamp else None,
                    },
                )
            )
        return alerts

    @staticmethod
    def _parse_timestamp(value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                if value.endswith("Z"):
                    value = value.replace("Z", "+00:00")
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

