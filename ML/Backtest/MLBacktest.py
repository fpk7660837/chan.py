"""
机器学习单信号回测引擎

默认使用更接近真实交易的规则：
信号生成后下一根K线入场，固定持有期后按配置价格出场，并可禁止同一标的重叠持仓。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Chan import CChan
from ..Evaluation.Metrics import Metrics
from ..Prediction.Predictor import Predictor


class MLBacktest:
    """机器学习单信号回测引擎"""

    def __init__(self, predictor: Predictor, config: Dict[str, Any] = None):
        self.predictor = predictor
        self.config = config or {}

        self.score_threshold = float(self.config.get('score_threshold', 0.7))
        self.holding_period = int(self.config.get('holding_period', 20))
        self.trade_direction = self.config.get('trade_direction', 'buy')
        self.entry_price = self.config.get('entry_price', 'next_open')
        self.exit_price = self.config.get('exit_price', 'close')
        self.allow_overlap_positions = bool(self.config.get('allow_overlap_positions', False))
        self.use_extreme_price_exit = bool(self.config.get('use_extreme_price_exit', False))
        self.allow_partial_window = bool(self.config.get('allow_partial_window', False))
        self.initial_capital = float(self.config.get('initial_capital', 100000))
        self.commission_rate = float(self.config.get('commission_rate', 0.0003))
        self.slippage = float(self.config.get('slippage', 0.001))
        self.max_position = float(self.config.get('max_position', 1.0))
        self.periods_per_year = int(self.config.get('periods_per_year', 252))

    def run(self, chan_list: List[CChan]) -> Dict[str, Any]:
        print("Starting backtest...")

        all_trades = []
        all_returns = []

        for i, chan in enumerate(chan_list):
            print(f"Processing {chan.code} ({i + 1}/{len(chan_list)})...")
            bsp_signals = self.predictor.filter_bsp_by_threshold(
                chan,
                threshold=self.score_threshold,
                direction=self.trade_direction,
            )
            trades = self._simulate_trades(chan, bsp_signals)
            all_trades.extend(trades)
            all_returns.extend(trade['return'] for trade in trades)

        metrics = Metrics.calculate_trading_metrics(
            np.array(all_returns, dtype=float),
            all_trades,
            periods_per_year=self.periods_per_year,
        )
        metrics['total_signals'] = float(len(all_trades))

        print("\nBacktest completed!")
        print(f"Total signals: {int(metrics['total_signals'])}")
        Metrics.print_metrics(metrics, title="Backtest Results")
        return metrics

    def _simulate_trades(self, chan: CChan, bsp_signals: List[Tuple]) -> List[Dict[str, Any]]:
        trades = []
        next_available_entry_idx = -1

        ordered_signals = sorted(bsp_signals, key=lambda item: item[0].klu.idx if item[0].klu is not None else -1)
        for bsp, score in ordered_signals:
            trade = self._execute_trade(chan.code, bsp, score)
            if trade is None:
                continue
            if not self.allow_overlap_positions and trade['entry_idx'] <= next_available_entry_idx:
                continue
            trades.append(trade)
            if not self.allow_overlap_positions:
                next_available_entry_idx = trade['exit_idx']

        return trades

    def _execute_trade(self, code: str, bsp, score: float) -> Optional[Dict[str, Any]]:
        if bsp.klu is None:
            return None

        entry_klu = self._get_entry_klu(bsp.klu)
        if entry_klu is None:
            return None

        exit_klu, forward_window = self._get_exit_klu(entry_klu)
        if exit_klu is None:
            return None

        if not self.allow_partial_window and len(forward_window) < self.holding_period:
            return None

        entry_price = self._get_entry_value(entry_klu, bsp.is_buy)
        exit_price = self._get_exit_value(exit_klu, forward_window, bsp.is_buy)
        if entry_price is None or exit_price is None or entry_price == 0:
            return None

        if bsp.is_buy:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        net_return = gross_return - 2 * self.commission_rate
        position_size = self.initial_capital * self.max_position
        profit = position_size * net_return

        return {
            'code': code,
            'signal_time': bsp.klu.time if hasattr(bsp.klu, 'time') else None,
            'entry_time': entry_klu.time if hasattr(entry_klu, 'time') else None,
            'exit_time': exit_klu.time if hasattr(exit_klu, 'time') else None,
            'signal_idx': bsp.klu.idx,
            'entry_idx': entry_klu.idx,
            'exit_idx': exit_klu.idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': 'buy' if bsp.is_buy else 'sell',
            'score': float(score),
            'return': float(net_return),
            'profit': float(profit),
            'holding_period': int(len(forward_window)),
        }

    def _get_entry_klu(self, signal_klu):
        if self.entry_price == 'signal_close':
            return signal_klu
        if self.entry_price in {'next_open', 'next_close'}:
            return getattr(signal_klu, 'next', None)
        raise ValueError(f"Unsupported entry_price: {self.entry_price}")

    def _get_exit_klu(self, entry_klu) -> Tuple[Optional[Any], List[Any]]:
        if self.holding_period <= 0:
            return entry_klu, [entry_klu]

        forward_window = [entry_klu]
        current = entry_klu
        for _ in range(self.holding_period - 1):
            nxt = getattr(current, 'next', None)
            if nxt is None:
                break
            current = nxt
            forward_window.append(current)

        if len(forward_window) < self.holding_period and not self.allow_partial_window:
            return None, forward_window
        return forward_window[-1], forward_window

    def _get_entry_value(self, entry_klu, is_buy: bool) -> Optional[float]:
        if self.entry_price == 'signal_close':
            base_price = getattr(entry_klu, 'close', None)
        elif self.entry_price == 'next_close':
            base_price = getattr(entry_klu, 'close', None)
        else:
            base_price = getattr(entry_klu, 'open', None)

        if base_price is None:
            return None
        slippage_factor = 1 + self.slippage if is_buy else 1 - self.slippage
        return float(base_price) * slippage_factor

    def _get_exit_value(self, exit_klu, forward_window: List[Any], is_buy: bool) -> Optional[float]:
        if self.use_extreme_price_exit and forward_window:
            if is_buy:
                base_price = max(float(klu.high) for klu in forward_window)
            else:
                base_price = min(float(klu.low) for klu in forward_window)
        elif self.exit_price == 'open':
            base_price = getattr(exit_klu, 'open', None)
        elif self.exit_price == 'high':
            base_price = getattr(exit_klu, 'high', None)
        elif self.exit_price == 'low':
            base_price = getattr(exit_klu, 'low', None)
        else:
            base_price = getattr(exit_klu, 'close', None)

        if base_price is None:
            return None

        if is_buy:
            return float(base_price) * (1 - self.slippage)
        return float(base_price) * (1 + self.slippage)

    def run_with_validation(self, train_chan_list: List[CChan], test_chan_list: List[CChan]) -> Dict[str, Any]:
        print("Running backtest on training set...")
        train_metrics = self.run(train_chan_list)

        print("\nRunning backtest on test set...")
        test_metrics = self.run(test_chan_list)

        return {'train': train_metrics, 'test': test_metrics}
