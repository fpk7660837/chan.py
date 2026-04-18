"""
标签构建器

默认采用可交易的固定持有期未来收益率打标签：
信号产生后，下一个可交易K线入场，持有固定窗口后按配置价格出场。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from BuySellPoint.BS_Point import CBS_Point


class LabelBuilder:
    """标签构建器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.strategy = self.config.get('label_strategy', 'forward_return')
        self.lookforward_bars = int(self.config.get('lookforward_bars', 20))
        self.min_future_bars = int(self.config.get('min_future_bars', self.lookforward_bars))
        self.threshold_pct = float(self.config.get('threshold_pct', 0.05))
        self.round_trip_cost_pct = float(self.config.get('round_trip_cost_pct', 0.0))
        self.entry_price = self.config.get('entry_price', 'next_open')
        self.exit_price = self.config.get('exit_price', 'close')
        self.allow_partial_window = bool(self.config.get('allow_partial_window', False))
        self.use_highest_for_buy = bool(self.config.get('use_highest_for_buy', False))
        self.use_lowest_for_sell = bool(self.config.get('use_lowest_for_sell', False))
        self.exit_warning_confirmation_horizon_30m = int(
            self.config.get('exit_warning_confirmation_horizon_30m', 8)
        )

    def build_labels(self, bsp_list: List[CBS_Point]) -> Tuple[np.ndarray, np.ndarray]:
        if self.strategy == 'forward_return':
            return self._forward_return_labels(bsp_list)
        if self.strategy == 'future_return':
            return self._legacy_future_return_labels(bsp_list)
        raise ValueError(f"Unknown label strategy: {self.strategy}")

    def _forward_return_labels(self, bsp_list: List[CBS_Point]) -> Tuple[np.ndarray, np.ndarray]:
        labels = []
        returns = []

        for bsp in bsp_list:
            label, ret = self._calculate_forward_return(bsp)
            labels.append(label)
            returns.append(ret)

        return np.array(labels, dtype=int), np.array(returns, dtype=float)

    def _legacy_future_return_labels(self, bsp_list: List[CBS_Point]) -> Tuple[np.ndarray, np.ndarray]:
        labels = []
        returns = []

        for bsp in bsp_list:
            label, ret = self._calculate_legacy_future_return(bsp)
            labels.append(label)
            returns.append(ret)

        return np.array(labels, dtype=int), np.array(returns, dtype=float)

    def _calculate_forward_return(self, bsp: CBS_Point) -> Tuple[int, float]:
        if bsp.klu is None:
            return 0, 0.0

        entry_klu = self._get_entry_klu(bsp.klu)
        if entry_klu is None:
            return 0, 0.0

        exit_klu, bars_available = self._get_exit_klu(entry_klu)
        if exit_klu is None:
            return 0, 0.0

        if not self.allow_partial_window and bars_available < self.min_future_bars:
            return 0, 0.0

        entry_value = self._resolve_price(entry_klu, self.entry_price, fallback='open')
        exit_value = self._resolve_price(exit_klu, self.exit_price, fallback='close')
        if entry_value is None or exit_value is None or entry_value == 0:
            return 0, 0.0

        if bsp.is_buy:
            ret = (exit_value - entry_value) / entry_value
        else:
            ret = (entry_value - exit_value) / entry_value

        return (1 if ret >= self.threshold_pct else 0), float(ret)

    def _calculate_legacy_future_return(self, bsp: CBS_Point) -> Tuple[int, float]:
        if bsp.klu is None:
            return 0, 0.0

        current_price = float(bsp.klu.close)
        future_klus = self._get_forward_window(bsp.klu, self.lookforward_bars)
        if not future_klus:
            return 0, 0.0

        if bsp.is_buy:
            if self.use_highest_for_buy:
                future_price = max(float(klu.high) for klu in future_klus)
            else:
                future_price = float(future_klus[-1].close)
            ret = (future_price - current_price) / current_price
        else:
            if self.use_lowest_for_sell:
                future_price = min(float(klu.low) for klu in future_klus)
            else:
                future_price = float(future_klus[-1].close)
            ret = (current_price - future_price) / current_price

        return (1 if ret >= self.threshold_pct else 0), float(ret)

    def _get_entry_klu(self, signal_klu):
        if self.entry_price == 'signal_close':
            return signal_klu
        if self.entry_price in {'next_open', 'next_close'}:
            return getattr(signal_klu, 'next', None)
        raise ValueError(f"Unsupported entry_price: {self.entry_price}")

    def _get_exit_klu(self, entry_klu) -> Tuple[Optional[Any], int]:
        if self.lookforward_bars <= 0:
            return entry_klu, 0

        current = entry_klu
        bars_available = 1
        for _ in range(self.lookforward_bars - 1):
            nxt = getattr(current, 'next', None)
            if nxt is None:
                break
            current = nxt
            bars_available += 1

        if bars_available < self.lookforward_bars and not self.allow_partial_window:
            return None, bars_available
        return current, bars_available

    def _get_forward_window(self, start_klu, periods: int) -> List[Any]:
        window = []
        current = start_klu
        for _ in range(periods):
            current = getattr(current, 'next', None)
            if current is None:
                break
            window.append(current)
        return window

    @staticmethod
    def _resolve_price(klu, price_mode: str, fallback: str = 'close') -> Optional[float]:
        if klu is None:
            return None

        mapping = {
            'signal_close': 'close',
            'next_open': 'open',
            'next_close': 'close',
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
        }
        attr_name = mapping.get(price_mode, mapping.get(fallback, 'close'))
        value = getattr(klu, attr_name, None)
        return None if value is None else float(value)

    def get_label_distribution(self, labels: np.ndarray) -> Dict[str, Any]:
        total = len(labels)
        positive = int(np.sum(labels == 1))
        negative = int(np.sum(labels == 0))

        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'positive_ratio': positive / total if total > 0 else 0.0,
            'negative_ratio': negative / total if total > 0 else 0.0,
        }

    def label_buy_entry(self, event: Any) -> Optional[Tuple[int, float]]:
        entry_klu = getattr(event, 'entry_klu', None)
        exit_confirm_bsp = getattr(event, 'exit_confirm_bsp', None)
        exit_klu = getattr(event, 'exit_klu', None)

        if entry_klu is None or exit_confirm_bsp is None or exit_klu is None:
            return None

        entry_price = getattr(entry_klu, 'open', None)
        exit_price = getattr(exit_klu, 'open', None)
        if entry_price is None or exit_price is None or float(entry_price) == 0.0:
            return None

        gross_return = (float(exit_price) - float(entry_price)) / float(entry_price)
        net_return = gross_return - self.round_trip_cost_pct
        return (1 if net_return > 0.0 else 0), float(net_return)

    def label_exit_warning(self, event: Any) -> Optional[int]:
        confirm_delay_bars_30m = getattr(event, 'confirm_delay_bars_30m', None)
        confirm_bsp = getattr(event, 'confirm_bsp', None)

        if confirm_bsp is None or confirm_delay_bars_30m is None:
            return 0
        return 1 if int(confirm_delay_bars_30m) <= self.exit_warning_confirmation_horizon_30m else 0

    def build_event_labels(self, events: List[Any], *, task_name: str) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        kept_events: List[Any] = []
        labels: List[int] = []
        aux_values: List[float] = []

        if task_name == 'buy_entry':
            for event in events:
                labeled = self.label_buy_entry(event)
                if labeled is None:
                    continue
                label, net_return = labeled
                kept_events.append(event)
                labels.append(label)
                aux_values.append(net_return)
            return kept_events, np.array(labels, dtype=int), np.array(aux_values, dtype=float)

        if task_name == 'exit_warning':
            for event in events:
                label = self.label_exit_warning(event)
                if label is None:
                    continue
                kept_events.append(event)
                labels.append(int(label))
                confirm_delay = getattr(event, 'confirm_delay_bars_30m', None)
                aux_values.append(float(confirm_delay) if confirm_delay is not None else -1.0)
            return kept_events, np.array(labels, dtype=int), np.array(aux_values, dtype=float)

        raise ValueError(f"Unsupported event labeling task: {task_name}")
