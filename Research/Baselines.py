from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional

from Chan import CChan

from .SignalSnapshot import SignalSnapshot


class BaselineSignalGenerator:
    """用于和缠论信号做最基本的对照。"""

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def generate_random_signals(
        self,
        chan_list: List[CChan],
        *,
        target_counts: Optional[Dict[str, int]] = None,
        direction: str = "buy",
        signal_lv_idx: int = 0,
        min_history_bars: int = 30,
        min_future_bars: int = 20,
    ) -> List[SignalSnapshot]:
        snapshots: List[SignalSnapshot] = []
        if target_counts is None:
            target_counts = {chan.code: 20 for chan in chan_list}

        for chan in chan_list:
            candidate_klus = [
                klu for klu in chan[signal_lv_idx].klu_iter()
                if klu.idx >= min_history_bars and self._has_future_bars(klu, min_future_bars)
            ]
            if not candidate_klus:
                continue

            sample_size = min(target_counts.get(chan.code, 0), len(candidate_klus))
            if sample_size <= 0:
                continue

            for klu in self.random.sample(candidate_klus, sample_size):
                snapshots.append(
                    SignalSnapshot.from_klu(
                        chan,
                        klu,
                        lv_idx=signal_lv_idx,
                        source="baseline_random",
                        direction=direction,
                        signal_kind="random_entry",
                        note="random baseline",
                    )
                )
        snapshots.sort(key=lambda item: (item.signal_time.ts, item.signal_idx, item.code))
        return snapshots

    def generate_momentum_signals(
        self,
        chan_list: List[CChan],
        *,
        target_counts: Optional[Dict[str, int]] = None,
        direction: str = "buy",
        signal_lv_idx: int = 0,
        lookback: int = 20,
        min_future_bars: int = 20,
    ) -> List[SignalSnapshot]:
        snapshots: List[SignalSnapshot] = []
        if target_counts is None:
            target_counts = Counter()
            for chan in chan_list:
                target_counts[chan.code] = 20

        for chan in chan_list:
            limit = int(target_counts.get(chan.code, 0))
            if limit <= 0:
                continue

            emitted = 0
            for klu in chan[signal_lv_idx].klu_iter():
                if klu.idx < lookback or not self._has_future_bars(klu, min_future_bars):
                    continue
                strength = self._momentum_breakout_strength(klu, lookback, direction)
                if strength is None:
                    continue
                snapshots.append(
                    SignalSnapshot.from_klu(
                        chan,
                        klu,
                        lv_idx=signal_lv_idx,
                        source="baseline_momentum",
                        direction=direction,
                        signal_kind="momentum_breakout",
                        note=f"{lookback}-bar breakout (time-ordered)",
                        metadata={"strength": strength, "lookback": lookback},
                    )
                )
                emitted += 1
                if emitted >= limit:
                    break

        snapshots.sort(key=lambda item: (item.signal_time.ts, item.signal_idx, item.code))
        return snapshots

    @staticmethod
    def count_by_code(snapshots: List[SignalSnapshot]) -> Dict[str, int]:
        return dict(Counter(snapshot.code for snapshot in snapshots))

    @staticmethod
    def _has_future_bars(klu, min_future_bars: int) -> bool:
        cursor = klu
        steps = 0
        while cursor is not None and steps < min_future_bars:
            cursor = getattr(cursor, "next", None)
            steps += 1
        return steps >= min_future_bars

    @staticmethod
    def _momentum_breakout_strength(klu, lookback: int, direction: str) -> Optional[float]:
        prev_closes = []
        cursor = getattr(klu, "pre", None)
        while cursor is not None and len(prev_closes) < lookback:
            prev_closes.append(float(cursor.close))
            cursor = getattr(cursor, "pre", None)
        if len(prev_closes) < lookback:
            return None
        prev_close = prev_closes[0]
        history = list(reversed(prev_closes))
        if direction == "buy":
            highest = max(history)
            previous_highest = max(history[:-1]) if len(history) > 1 else highest
            if float(klu.close) <= highest or prev_close > previous_highest:
                return None
            return (float(klu.close) - highest) / highest if highest else None

        lowest = min(history)
        previous_lowest = min(history[:-1]) if len(history) > 1 else lowest
        if float(klu.close) >= lowest or prev_close < previous_lowest:
            return None
        return (lowest - float(klu.close)) / lowest if lowest else None
