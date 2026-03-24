from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .SignalSnapshot import SignalSnapshot


class ChanSignalBacktest:
    """纯信号事件回测，不引入 ML。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.holding_period = int(self.config.get("holding_period", 20))
        self.entry_price = self.config.get("entry_price", "next_open")
        self.exit_price = self.config.get("exit_price", "close")
        self.allow_overlap_positions = bool(self.config.get("allow_overlap_positions", False))
        self.stop_loss_pct = self.config.get("stop_loss_pct", None)
        self.initial_capital = float(self.config.get("initial_capital", 100000))
        self.position_size = float(self.config.get("position_size", 1.0))
        self.commission_rate = float(self.config.get("commission_rate", 0.0003))
        self.slippage = float(self.config.get("slippage", 0.001))

    def run(self, snapshots: List[SignalSnapshot]) -> Dict[str, Any]:
        trades: List[Dict[str, Any]] = []
        next_available_by_code: Dict[str, int] = {}

        ordered = sorted(snapshots, key=lambda item: (item.signal_time.ts, item.signal_idx, item.code))
        for snapshot in ordered:
            trade = self._execute_trade(snapshot)
            if trade is None:
                continue
            if not self.allow_overlap_positions:
                next_available_idx = next_available_by_code.get(snapshot.code, -1)
                if trade["entry_idx"] <= next_available_idx:
                    continue
                next_available_by_code[snapshot.code] = trade["exit_idx"]
            trades.append(trade)

        metrics = self._calculate_research_metrics(trades, total_signals=len(snapshots))
        return {"metrics": metrics, "trades": trades}

    def _execute_trade(self, snapshot: SignalSnapshot) -> Optional[Dict[str, Any]]:
        signal_klu = snapshot.signal_klu
        if signal_klu is None:
            return None
        entry_klu = self._get_entry_klu(signal_klu)
        if entry_klu is None:
            return None
        entry_value = self._get_entry_value(entry_klu, snapshot.direction)
        if entry_value is None or entry_value == 0:
            return None

        window = self._forward_window(entry_klu, self.holding_period)
        if len(window) < self.holding_period:
            return None

        exit_klu = window[-1]
        exit_value = self._get_exit_value(exit_klu, snapshot.direction)
        if exit_value is None:
            return None

        realized_exit_klu = exit_klu
        realized_exit_value = exit_value
        exit_reason = "time_exit"
        stop_loss_hit = False
        if self.stop_loss_pct is not None:
            stop_klu, stop_value = self._scan_stop_loss(window, entry_value, snapshot.direction)
            if stop_klu is not None and stop_value is not None:
                realized_exit_klu = stop_klu
                realized_exit_value = stop_value
                exit_reason = "stop_loss"
                stop_loss_hit = True

        if snapshot.direction == "buy":
            gross_return = (realized_exit_value - entry_value) / entry_value
        else:
            gross_return = (entry_value - realized_exit_value) / entry_value
        net_return = gross_return - 2 * self.commission_rate
        capital = self.initial_capital * self.position_size
        profit = capital * net_return

        return {
            "code": snapshot.code,
            "source": snapshot.source,
            "signal_time": snapshot.signal_time_str,
            "signal_idx": snapshot.signal_idx,
            "entry_time": entry_klu.time.to_str(),
            "entry_idx": entry_klu.idx,
            "entry_price": float(entry_value),
            "exit_time": realized_exit_klu.time.to_str(),
            "exit_idx": realized_exit_klu.idx,
            "exit_price": float(realized_exit_value),
            "direction": snapshot.direction,
            "signal_kind": snapshot.signal_kind,
            "bsp_type": snapshot.bsp_type,
            "holding_bars": int(realized_exit_klu.idx - entry_klu.idx + 1),
            "exit_reason": exit_reason,
            "stop_loss_hit": int(stop_loss_hit),
            "return": float(net_return),
            "profit": float(profit),
        }

    @staticmethod
    def _calculate_research_metrics(trades: List[Dict[str, Any]], *, total_signals: int) -> Dict[str, float]:
        returns = np.array([float(trade["return"]) for trade in trades], dtype=float) if trades else np.array([], dtype=float)
        if returns.size == 0:
            return {
                "total_signals": float(total_signals),
                "executed_trades": 0.0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "median_return": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "profit_loss_ratio": 0.0,
                "stop_loss_rate": 0.0,
                "avg_holding_bars": 0.0,
            }

        equity_curve = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = 1 - np.divide(
            equity_curve,
            running_max,
            out=np.ones_like(equity_curve),
            where=running_max != 0,
        )

        winners = returns[returns > 0]
        losers = returns[returns < 0]
        avg_profit = float(np.mean(winners)) if winners.size else 0.0
        avg_loss = abs(float(np.mean(losers))) if losers.size else 0.0

        return {
            "total_signals": float(total_signals),
            "executed_trades": float(len(trades)),
            "win_rate": float(np.mean(returns > 0)),
            "avg_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
            "total_return": float(equity_curve[-1] - 1),
            "max_drawdown": float(np.max(drawdown)) if drawdown.size else 0.0,
            "profit_loss_ratio": avg_profit / avg_loss if avg_loss > 0 else 0.0,
            "stop_loss_rate": float(np.mean([trade.get("stop_loss_hit", 0) for trade in trades])),
            "avg_holding_bars": float(np.mean([trade.get("holding_bars", 0) for trade in trades])),
        }

    def _get_entry_klu(self, signal_klu):
        if self.entry_price == "signal_close":
            return signal_klu
        if self.entry_price in {"next_open", "next_close"}:
            return getattr(signal_klu, "next", None)
        raise ValueError(f"Unsupported entry_price: {self.entry_price}")

    @staticmethod
    def _forward_window(entry_klu, horizon: int) -> List[Any]:
        result = []
        cursor = entry_klu
        while cursor is not None and len(result) < horizon:
            result.append(cursor)
            cursor = getattr(cursor, "next", None)
        return result

    def _get_entry_value(self, entry_klu, direction: str) -> Optional[float]:
        base_price = getattr(entry_klu, "open" if self.entry_price == "next_open" else "close", None)
        if base_price is None:
            return None
        slippage_factor = 1 + self.slippage if direction == "buy" else 1 - self.slippage
        return float(base_price) * slippage_factor

    def _get_exit_value(self, exit_klu, direction: str) -> Optional[float]:
        base_price = getattr(exit_klu, self.exit_price, None)
        if base_price is None:
            return None
        if direction == "buy":
            return float(base_price) * (1 - self.slippage)
        return float(base_price) * (1 + self.slippage)

    def _scan_stop_loss(self, window: List[Any], entry_price: float, direction: str):
        for klu in window:
            if direction == "buy":
                low = float(getattr(klu, "low", entry_price))
                if (entry_price - low) / entry_price >= float(self.stop_loss_pct):
                    return klu, entry_price * (1 - float(self.stop_loss_pct))
            else:
                high = float(getattr(klu, "high", entry_price))
                if (high - entry_price) / entry_price >= float(self.stop_loss_pct):
                    return klu, entry_price * (1 + float(self.stop_loss_pct))
        return None, None
