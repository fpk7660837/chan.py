"""
横截面选股组合回测

在每个调仓点，对股票池内最近有效的买点信号做排序，选出前N只股票等权持有。
"""

from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Chan import CChan
from ..Evaluation.Metrics import Metrics
from ..Prediction.Predictor import Predictor


class CrossSectionBacktest:
    """横截面股票池组合回测"""

    def __init__(self, predictor: Predictor, config: Dict[str, Any] = None):
        self.predictor = predictor
        self.config = config or {}

        self.top_k = int(self.config.get('top_k', 5))
        self.score_threshold = float(self.config.get('score_threshold', 0.6))
        self.rebalance_bars = int(self.config.get('rebalance_bars', 5))
        self.signal_lookback_bars = int(self.config.get('signal_lookback_bars', 20))
        self.holding_period = self.config.get('holding_period')
        self.min_positions = int(self.config.get('min_positions', 1))
        self.direction = self.config.get('direction', 'buy')
        self.entry_price = self.config.get('entry_price', 'next_open')
        self.exit_price = self.config.get('exit_price', 'close')
        self.allow_partial_window = bool(self.config.get('allow_partial_window', False))
        self.initial_capital = float(self.config.get('initial_capital', 100000))
        self.commission_rate = float(self.config.get('commission_rate', 0.0003))
        self.slippage = float(self.config.get('slippage', 0.001))
        self.weighting = self.config.get('weighting', 'equal')
        self.periods_per_year = int(self.config.get('periods_per_year', 52))

    def run(self, chan_list: List[CChan]) -> Dict[str, Any]:
        if not chan_list:
            raise ValueError("chan_list cannot be empty")

        bar_cache = self._build_bar_cache(chan_list)
        signal_cache = self._build_signal_cache(chan_list)
        calendar = self._build_calendar(bar_cache)
        if len(calendar) <= self.rebalance_bars:
            raise ValueError("Not enough bars to run cross-sectional backtest")

        equity = self.initial_capital
        period_returns: List[float] = []
        trades: List[Dict[str, Any]] = []
        portfolio_history: List[Dict[str, Any]] = []

        rebalance_indices = list(range(0, len(calendar) - self.rebalance_bars, self.rebalance_bars))
        for rebalance_idx in rebalance_indices:
            rebalance_ts, rebalance_time = calendar[rebalance_idx]
            next_idx = rebalance_idx + self.rebalance_bars
            next_rebalance_ts, next_rebalance_time = calendar[next_idx]

            candidates = self._select_candidates(
                chan_list=chan_list,
                bar_cache=bar_cache,
                signal_cache=signal_cache,
                rebalance_ts=rebalance_ts,
            )
            if len(candidates) < self.min_positions:
                continue

            selected = candidates[:self.top_k]
            if self.weighting != 'equal':
                raise ValueError(f"Unsupported weighting scheme: {self.weighting}")

            weight = 1.0 / len(selected)
            period_trade_records = []
            stock_returns = []

            for candidate in selected:
                trade = self._build_trade(
                    candidate=candidate,
                    bar_cache=bar_cache,
                    rebalance_ts=rebalance_ts,
                    next_rebalance_ts=next_rebalance_ts,
                    allocation=equity * weight,
                )
                if trade is None:
                    continue
                period_trade_records.append(trade)
                stock_returns.append(trade['return'])

            if len(period_trade_records) < self.min_positions:
                continue

            period_return = float(np.mean(stock_returns))
            equity *= (1 + period_return)
            period_returns.append(period_return)
            trades.extend(period_trade_records)
            portfolio_history.append({
                'rebalance_time': rebalance_time,
                'exit_time': next_rebalance_time,
                'positions': len(period_trade_records),
                'period_return': period_return,
                'equity': equity,
                'codes': [trade['code'] for trade in period_trade_records],
            })

        metrics = Metrics.calculate_trading_metrics(
            np.array(period_returns, dtype=float),
            trades=trades,
            periods_per_year=self.periods_per_year,
        )
        metrics['periods'] = float(len(period_returns))
        metrics['avg_positions'] = float(np.mean([item['positions'] for item in portfolio_history])) if portfolio_history else 0.0
        metrics['portfolio_history'] = portfolio_history
        metrics['trades'] = trades

        Metrics.print_metrics(metrics, title="Cross-Section Backtest Results")
        return metrics

    def _build_bar_cache(self, chan_list: List[CChan]) -> Dict[str, Dict[str, Any]]:
        cache = {}
        for chan in chan_list:
            bars = list(chan[0].klu_iter())
            ts_list = [bar.time.ts for bar in bars]
            cache[chan.code] = {
                'chan': chan,
                'bars': bars,
                'ts': ts_list,
            }
        return cache

    def _build_signal_cache(self, chan_list: List[CChan]) -> Dict[str, List[Dict[str, Any]]]:
        cache: Dict[str, List[Dict[str, Any]]] = {}
        for chan in chan_list:
            scored = self.predictor.score_bsp(chan, direction=self.direction)
            scored.sort(key=lambda item: item[0].klu.idx if item[0].klu is not None else -1)
            cache[chan.code] = [
                {
                    'bsp': bsp,
                    'score': float(score),
                    'signal_idx': bsp.klu.idx,
                    'signal_ts': bsp.klu.time.ts,
                }
                for bsp, score in scored
                if bsp.klu is not None
            ]
        return cache

    def _build_calendar(self, bar_cache: Dict[str, Dict[str, Any]]) -> List[Tuple[float, Any]]:
        calendar_map = {}
        for data in bar_cache.values():
            for bar in data['bars']:
                calendar_map[bar.time.ts] = bar.time
        return sorted(calendar_map.items(), key=lambda item: item[0])

    def _select_candidates(
        self,
        chan_list: List[CChan],
        bar_cache: Dict[str, Dict[str, Any]],
        signal_cache: Dict[str, List[Dict[str, Any]]],
        rebalance_ts: float,
    ) -> List[Dict[str, Any]]:
        candidates = []
        for chan in chan_list:
            latest_bar = self._get_bar_at_or_before(bar_cache[chan.code], rebalance_ts)
            if latest_bar is None:
                continue
            signal = self._get_latest_signal(signal_cache[chan.code], latest_bar.idx, rebalance_ts)
            if signal is None:
                continue
            candidates.append({
                'code': chan.code,
                'score': signal['score'],
                'signal': signal,
                'latest_bar': latest_bar,
            })

        candidates = [candidate for candidate in candidates if candidate['score'] >= self.score_threshold]
        candidates.sort(key=lambda item: item['score'], reverse=True)
        return candidates

    def _get_latest_signal(
        self,
        signals: List[Dict[str, Any]],
        latest_bar_idx: int,
        rebalance_ts: float,
    ) -> Optional[Dict[str, Any]]:
        for signal in reversed(signals):
            if signal['signal_ts'] > rebalance_ts:
                continue
            if latest_bar_idx - signal['signal_idx'] > self.signal_lookback_bars:
                break
            return signal
        return None

    def _build_trade(
        self,
        candidate: Dict[str, Any],
        bar_cache: Dict[str, Dict[str, Any]],
        rebalance_ts: float,
        next_rebalance_ts: float,
        allocation: float,
    ) -> Optional[Dict[str, Any]]:
        code = candidate['code']
        current_bar = self._get_bar_at_or_before(bar_cache[code], rebalance_ts)
        if current_bar is None:
            return None

        entry_bar = self._get_entry_bar(current_bar)
        if entry_bar is None:
            return None

        exit_bar = self._get_exit_bar(bar_cache[code], entry_bar, next_rebalance_ts)
        if exit_bar is None:
            return None

        entry_price = self._get_entry_value(entry_bar)
        exit_price = self._get_exit_value(exit_bar)
        if entry_price is None or exit_price is None or entry_price == 0:
            return None

        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - 2 * self.commission_rate
        profit = allocation * net_return

        return {
            'code': code,
            'signal_time': candidate['signal']['bsp'].klu.time,
            'entry_time': entry_bar.time,
            'exit_time': exit_bar.time,
            'score': float(candidate['score']),
            'entry_idx': entry_bar.idx,
            'exit_idx': exit_bar.idx,
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'return': float(net_return),
            'profit': float(profit),
        }

    def _get_bar_at_or_before(self, cache_entry: Dict[str, Any], target_ts: float):
        idx = bisect_right(cache_entry['ts'], target_ts) - 1
        if idx < 0:
            return None
        return cache_entry['bars'][idx]

    def _get_entry_bar(self, current_bar):
        if self.entry_price == 'signal_close':
            return current_bar
        if self.entry_price in {'next_open', 'next_close'}:
            return getattr(current_bar, 'next', None)
        raise ValueError(f"Unsupported entry_price: {self.entry_price}")

    def _get_exit_bar(self, cache_entry: Dict[str, Any], entry_bar, next_rebalance_ts: float):
        if self.holding_period is not None:
            current = entry_bar
            bars_seen = 1
            for _ in range(int(self.holding_period) - 1):
                nxt = getattr(current, 'next', None)
                if nxt is None:
                    break
                current = nxt
                bars_seen += 1
            if bars_seen < int(self.holding_period) and not self.allow_partial_window:
                return None
            return current

        exit_bar = self._get_bar_at_or_before(cache_entry, next_rebalance_ts)
        if exit_bar is None or exit_bar.idx < entry_bar.idx:
            return None
        return exit_bar

    def _get_entry_value(self, entry_bar) -> Optional[float]:
        if self.entry_price == 'signal_close':
            base_price = getattr(entry_bar, 'close', None)
        elif self.entry_price == 'next_close':
            base_price = getattr(entry_bar, 'close', None)
        else:
            base_price = getattr(entry_bar, 'open', None)
        if base_price is None:
            return None
        return float(base_price) * (1 + self.slippage)

    def _get_exit_value(self, exit_bar) -> Optional[float]:
        if self.exit_price == 'open':
            base_price = getattr(exit_bar, 'open', None)
        elif self.exit_price == 'high':
            base_price = getattr(exit_bar, 'high', None)
        elif self.exit_price == 'low':
            base_price = getattr(exit_bar, 'low', None)
        else:
            base_price = getattr(exit_bar, 'close', None)
        if base_price is None:
            return None
        return float(base_price) * (1 - self.slippage)
