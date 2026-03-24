from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .SignalSnapshot import SignalSnapshot


class SignalLabeler:
    """给标准化事件样本补充未来表现标签。"""

    def label_snapshots(
        self,
        snapshots: List[SignalSnapshot],
        *,
        holding_periods: Sequence[int] = (5, 10, 20),
        stop_loss_pct: Optional[float] = 0.05,
        entry_price: str = "next_open",
        exit_price: str = "close",
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for snapshot in snapshots:
            record = snapshot.to_dict()
            record.update(
                self.label_snapshot(
                    snapshot,
                    holding_periods=holding_periods,
                    stop_loss_pct=stop_loss_pct,
                    entry_price=entry_price,
                    exit_price=exit_price,
                )
            )
            records.append(record)
        return records

    def label_snapshot(
        self,
        snapshot: SignalSnapshot,
        *,
        holding_periods: Sequence[int],
        stop_loss_pct: Optional[float],
        entry_price: str,
        exit_price: str,
    ) -> Dict[str, Any]:
        signal_klu = snapshot.signal_klu
        if signal_klu is None:
            return {}

        entry_klu = self._get_entry_klu(signal_klu, entry_price)
        if entry_klu is None:
            return {"is_eligible": 0}

        entry_value = self._get_price(entry_klu, entry_price)
        if entry_value is None or entry_value == 0:
            return {"is_eligible": 0}

        result: Dict[str, Any] = {
            "entry_time": entry_klu.time.to_str(),
            "entry_idx": entry_klu.idx,
            "entry_price": float(entry_value),
            "is_eligible": 1,
        }
        result.update(self._build_environment_metrics(signal_klu))

        for horizon in holding_periods:
            window = self._get_forward_window(entry_klu, horizon)
            prefix = f"h{horizon}"
            result[f"{prefix}_window"] = len(window)
            if len(window) < horizon:
                result[f"{prefix}_available"] = 0
                continue

            exit_klu = window[-1]
            exit_value = self._get_price(exit_klu, exit_price)
            if exit_value is None:
                result[f"{prefix}_available"] = 0
                continue

            mfe, mae = self._calc_excursions(
                entry_value=float(entry_value),
                window=window,
                is_buy=snapshot.is_buy,
            )
            stop_loss_idx = self._scan_stop_loss(
                entry_value=float(entry_value),
                window=window,
                is_buy=snapshot.is_buy,
                stop_loss_pct=stop_loss_pct,
            )
            reverse_signal_idx = self._find_reverse_signal_idx(
                snapshot,
                max_signal_idx=window[-1].idx,
            )
            stop_loss_hit = stop_loss_idx is not None
            reverse_signal_hit = reverse_signal_idx is not None
            stop_loss_first = stop_loss_hit and (
                reverse_signal_idx is None or stop_loss_idx <= reverse_signal_idx
            )
            reverse_signal_first = reverse_signal_hit and (
                stop_loss_idx is None or reverse_signal_idx < stop_loss_idx
            )

            result[f"{prefix}_available"] = 1
            result[f"{prefix}_exit_time"] = exit_klu.time.to_str()
            result[f"{prefix}_exit_idx"] = exit_klu.idx
            result[f"{prefix}_exit_price"] = float(exit_value)
            result[f"{prefix}_return"] = self._calc_return(
                entry_price=float(entry_value),
                exit_price=float(exit_value),
                is_buy=snapshot.is_buy,
            )
            result[f"{prefix}_mfe"] = mfe
            result[f"{prefix}_mae"] = mae
            result[f"{prefix}_stop_loss_hit"] = int(stop_loss_hit)
            result[f"{prefix}_stop_loss_first"] = int(stop_loss_first)
            result[f"{prefix}_reverse_signal_hit"] = int(reverse_signal_hit)
            result[f"{prefix}_reverse_signal_first"] = int(reverse_signal_first)

        return result

    def _build_environment_metrics(self, signal_klu) -> Dict[str, Any]:
        history_20 = self._get_history_window(signal_klu, 20)
        history_10 = self._get_history_window(signal_klu, 10)
        result: Dict[str, Any] = {
            "env_trend_return_20": None,
            "env_ma_gap_20": None,
            "env_avg_range_10": None,
        }

        if len(history_20) >= 20:
            closes = [float(getattr(klu, "close", 0.0) or 0.0) for klu in reversed(history_20)]
            start_close = closes[0]
            last_close = closes[-1]
            if start_close != 0:
                result["env_trend_return_20"] = (last_close - start_close) / start_close
            ma20 = sum(closes) / len(closes)
            if ma20 != 0:
                result["env_ma_gap_20"] = (last_close - ma20) / ma20

        if len(history_10) >= 10:
            ranges = []
            for klu in history_10:
                close = float(getattr(klu, "close", 0.0) or 0.0)
                if close == 0:
                    continue
                high = float(getattr(klu, "high", close))
                low = float(getattr(klu, "low", close))
                ranges.append((high - low) / close)
            if ranges:
                result["env_avg_range_10"] = sum(ranges) / len(ranges)

        return result

    @staticmethod
    def _get_entry_klu(signal_klu, entry_price: str):
        if entry_price == "signal_close":
            return signal_klu
        if entry_price in {"next_open", "next_close"}:
            return getattr(signal_klu, "next", None)
        raise ValueError(f"Unsupported entry_price: {entry_price}")

    @staticmethod
    def _get_history_window(end_klu, lookback: int) -> List[Any]:
        window = []
        cursor = end_klu
        while cursor is not None and len(window) < lookback:
            window.append(cursor)
            cursor = getattr(cursor, "pre", None)
        return window

    @staticmethod
    def _get_forward_window(entry_klu, horizon: int) -> List[Any]:
        window = []
        cursor = entry_klu
        while cursor is not None and len(window) < horizon:
            window.append(cursor)
            cursor = getattr(cursor, "next", None)
        return window

    @staticmethod
    def _get_price(klu, price_key: str) -> Optional[float]:
        if price_key == "signal_close":
            price_key = "close"
        elif price_key == "next_open":
            price_key = "open"
        elif price_key == "next_close":
            price_key = "close"
        return float(getattr(klu, price_key, 0.0) or 0.0)

    @staticmethod
    def _calc_return(entry_price: float, exit_price: float, is_buy: bool) -> float:
        if is_buy:
            return (exit_price - entry_price) / entry_price
        return (entry_price - exit_price) / entry_price

    @staticmethod
    def _calc_excursions(
        entry_value: float,
        window: Iterable[Any],
        is_buy: bool,
    ) -> tuple[float, float]:
        favorable = []
        adverse = []
        for klu in window:
            high = float(getattr(klu, "high", entry_value))
            low = float(getattr(klu, "low", entry_value))
            if is_buy:
                favorable.append(max((high - entry_value) / entry_value, 0.0))
                adverse.append(max((entry_value - low) / entry_value, 0.0))
            else:
                favorable.append(max((entry_value - low) / entry_value, 0.0))
                adverse.append(max((high - entry_value) / entry_value, 0.0))
        return (max(favorable) if favorable else 0.0, max(adverse) if adverse else 0.0)

    @staticmethod
    def _scan_stop_loss(
        *,
        entry_value: float,
        window: List[Any],
        is_buy: bool,
        stop_loss_pct: Optional[float],
    ) -> Optional[int]:
        if stop_loss_pct is None:
            return None

        for klu in window:
            if is_buy:
                low = float(getattr(klu, "low", entry_value))
                if (entry_value - low) / entry_value >= float(stop_loss_pct):
                    return int(getattr(klu, "idx", -1))
            else:
                high = float(getattr(klu, "high", entry_value))
                if (high - entry_value) / entry_value >= float(stop_loss_pct):
                    return int(getattr(klu, "idx", -1))
        return None

    @staticmethod
    def _find_reverse_signal_idx(
        snapshot: SignalSnapshot,
        max_signal_idx: int,
    ) -> Optional[int]:
        if snapshot.chan is None:
            return None

        for bsp in snapshot.chan[snapshot.lv_idx].bs_point_lst.getSortedBspList():
            bsp_idx = getattr(getattr(bsp, "klu", None), "idx", -1)
            if bsp_idx <= snapshot.signal_idx:
                continue
            if bsp_idx > max_signal_idx:
                break
            if snapshot.is_buy and not bsp.is_buy:
                return int(bsp_idx)
            if not snapshot.is_buy and bsp.is_buy:
                return int(bsp_idx)
        return None
