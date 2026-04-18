"""
多级别训练事件构造。

把外部 loader 提供的 `day/30m/5m` 上下文转换成可训练的入场和退出预警事件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .MultiLevelDataLoader import InstrumentMultiLevelData


@dataclass(frozen=True)
class BuyEntryEvent:
    position_id: str
    code: str
    name: str
    bsp_type: str
    entry_bsp: Any
    entry_klu: Optional[Any]
    exit_warning_bsp: Optional[Any]
    exit_confirm_bsp: Optional[Any]
    exit_klu: Optional[Any]
    context: Dict[str, Any]


@dataclass(frozen=True)
class ExitWarningEvent:
    position_id: str
    code: str
    name: str
    warning_bsp: Any
    confirm_bsp: Optional[Any]
    confirm_delay_bars_30m: Optional[int]
    warning_klu: Any
    context: Dict[str, Any]
    entry_event: BuyEntryEvent


class MultiLevelSampleBuilder:
    def __init__(
        self,
        *,
        day_level: str = "day",
        decision_level: str = "30m",
        execution_level: str = "5m",
    ) -> None:
        self.day_level = day_level
        self.decision_level = decision_level
        self.execution_level = execution_level

    def build_buy_entry_events(self, instruments: Sequence[InstrumentMultiLevelData]) -> List[BuyEntryEvent]:
        events: List[BuyEntryEvent] = []
        for instrument in instruments:
            day_bsps = self._sort_points(instrument.level_bsps.get(self.day_level, []))
            decision_bsps = self._sort_points(instrument.level_bsps.get(self.decision_level, []))
            execution_bsps = self._sort_points(instrument.level_bsps.get(self.execution_level, []))
            execution_bars = self._sort_bars(instrument.level_bars.get(self.execution_level, []))

            for entry_bsp in decision_bsps:
                if not getattr(entry_bsp, "is_buy", False):
                    continue

                anchor_time = self._time_value(entry_bsp)
                entry_klu = self._first_bar_after(execution_bars, anchor_time)
                exit_confirm_bsp = self._first_reverse_bsp(
                    decision_bsps,
                    after_time=anchor_time,
                    reverse_is_buy=False,
                )
                exit_klu = self._first_bar_after(execution_bars, self._time_value(exit_confirm_bsp))
                exit_warning_bsp = self._first_reverse_bsp(
                    execution_bsps,
                    after_time=self._time_value(entry_klu),
                    reverse_is_buy=False,
                    end_time=self._time_value(exit_confirm_bsp),
                    include_same_time=True,
                )
                day_context = self._latest_bsp_at_or_before(day_bsps, anchor_time)
                execution_context = self._latest_same_direction_bsp_at_or_before(
                    execution_bsps,
                    anchor_time=anchor_time,
                    is_buy=True,
                )

                position_id = f"{instrument.code}:{anchor_time}:{self._primary_bsp_type(entry_bsp)}"
                events.append(
                    BuyEntryEvent(
                        position_id=position_id,
                        code=instrument.code,
                        name=instrument.name,
                        bsp_type=self._primary_bsp_type(entry_bsp),
                        entry_bsp=entry_bsp,
                        entry_klu=entry_klu,
                        exit_warning_bsp=exit_warning_bsp,
                        exit_confirm_bsp=exit_confirm_bsp,
                        exit_klu=exit_klu,
                        context={
                            self.day_level: day_context,
                            self.decision_level: entry_bsp,
                            self.execution_level: execution_context,
                        },
                    )
                )
        return events

    def build_exit_warning_events(self, instruments: Sequence[InstrumentMultiLevelData]) -> List[ExitWarningEvent]:
        warnings: List[ExitWarningEvent] = []
        for instrument in instruments:
            decision_bars = self._sort_bars(instrument.level_bars.get(self.decision_level, []))
            for entry_event in self.build_buy_entry_events([instrument]):
                if entry_event.exit_warning_bsp is None:
                    continue
                warnings.append(
                    ExitWarningEvent(
                        position_id=entry_event.position_id,
                        code=entry_event.code,
                        name=entry_event.name,
                        warning_bsp=entry_event.exit_warning_bsp,
                        confirm_bsp=entry_event.exit_confirm_bsp,
                        confirm_delay_bars_30m=self._count_bars_between(
                            decision_bars,
                            start_time=self._time_value(entry_event.exit_warning_bsp),
                            end_time=self._time_value(entry_event.exit_confirm_bsp),
                        ),
                        warning_klu=getattr(entry_event.exit_warning_bsp, "klu", None),
                        context={
                            self.day_level: entry_event.context.get(self.day_level),
                            self.decision_level: entry_event.entry_bsp,
                            self.execution_level: entry_event.exit_warning_bsp,
                        },
                        entry_event=entry_event,
                    )
                )
        return warnings

    @staticmethod
    def _sort_points(points: Sequence[Any]) -> List[Any]:
        return sorted(points, key=MultiLevelSampleBuilder._sort_key)

    @staticmethod
    def _sort_bars(bars: Sequence[Any]) -> List[Any]:
        return sorted(bars, key=MultiLevelSampleBuilder._sort_key)

    @staticmethod
    def _sort_key(item: Any) -> tuple:
        return (
            MultiLevelSampleBuilder._time_value(item),
            int(getattr(getattr(item, "klu", item), "idx", -1) or -1),
        )

    @staticmethod
    def _time_value(item: Any) -> int:
        if item is None:
            return -1

        klu = getattr(item, "klu", item)
        time_value = getattr(klu, "time", None)
        if hasattr(time_value, "ts"):
            return int(time_value.ts)
        if isinstance(time_value, (int, float)):
            return int(time_value)
        if hasattr(klu, "idx"):
            return int(klu.idx)
        return -1

    @staticmethod
    def _primary_bsp_type(bsp: Any) -> str:
        bsp_type = getattr(bsp, "type", "")
        if isinstance(bsp_type, list):
            primary = bsp_type[0] if bsp_type else ""
        else:
            primary = bsp_type
        return getattr(primary, "value", str(primary))

    @staticmethod
    def _latest_bsp_at_or_before(points: Sequence[Any], anchor_time: int) -> Optional[Any]:
        latest = None
        for point in points:
            point_time = MultiLevelSampleBuilder._time_value(point)
            if point_time <= anchor_time:
                latest = point
            else:
                break
        return latest

    @staticmethod
    def _latest_same_direction_bsp_at_or_before(points: Sequence[Any], *, anchor_time: int, is_buy: bool) -> Optional[Any]:
        latest = None
        for point in points:
            point_time = MultiLevelSampleBuilder._time_value(point)
            if point_time > anchor_time:
                break
            if getattr(point, "is_buy", None) == is_buy:
                latest = point
        return latest

    @staticmethod
    def _first_bar_after(bars: Sequence[Any], anchor_time: Optional[int]) -> Optional[Any]:
        if anchor_time is None or anchor_time < 0:
            return None
        for bar in bars:
            if MultiLevelSampleBuilder._time_value(bar) > anchor_time:
                return bar
        return None

    @staticmethod
    def _first_reverse_bsp(
        points: Sequence[Any],
        *,
        after_time: Optional[int],
        reverse_is_buy: bool,
        end_time: Optional[int] = None,
        include_same_time: bool = False,
    ) -> Optional[Any]:
        if after_time is None or after_time < 0:
            return None
        for point in points:
            point_time = MultiLevelSampleBuilder._time_value(point)
            if include_same_time:
                if point_time < after_time:
                    continue
            elif point_time <= after_time:
                continue
            if end_time is not None and end_time >= 0 and point_time >= end_time:
                break
            if getattr(point, "is_buy", None) == reverse_is_buy:
                return point
        return None

    @staticmethod
    def _count_bars_between(
        bars: Sequence[Any],
        *,
        start_time: Optional[int],
        end_time: Optional[int],
    ) -> Optional[int]:
        if start_time is None or start_time < 0 or end_time is None or end_time < 0:
            return None
        count = 0
        for bar in bars:
            bar_time = MultiLevelSampleBuilder._time_value(bar)
            if bar_time <= start_time:
                continue
            if bar_time > end_time:
                break
            count += 1
        return count
