"""
Wrapper around CChan that leverages trigger_load for incremental K-line updates.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_FIELD, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Common.func_util import check_kltype_order
from KLine.KLine_Unit import CKLine_Unit


LEVEL_TO_KL_TYPE: Dict[str, KL_TYPE] = {
    "1m": KL_TYPE.K_1M,
    "3m": KL_TYPE.K_3M,
    "5m": KL_TYPE.K_5M,
    "15m": KL_TYPE.K_15M,
    "30m": KL_TYPE.K_30M,
    "60m": KL_TYPE.K_60M,
    "day": KL_TYPE.K_DAY,
    "week": KL_TYPE.K_WEEK,
    "mon": KL_TYPE.K_MON,
}

KL_TYPE_TO_LEVEL: Dict[KL_TYPE, str] = {value: key for key, value in LEVEL_TO_KL_TYPE.items()}


class ChanTriggerSession:
    """Maintains a CChan instance configured for trigger_load based incremental updates."""

    def __init__(
        self,
        symbol: str,
        lv_list: Sequence[KL_TYPE] | None = None,
        chan_config: Mapping[str, object] | None = None,
        *,
        autype: AUTYPE = AUTYPE.NONE,
    ) -> None:
        if lv_list is None or len(lv_list) == 0:
            lv_list = [KL_TYPE.K_1M]
        check_kltype_order(list(lv_list))

        config_dict: MutableMapping[str, object] = dict(chan_config or {})
        config_dict["trigger_step"] = True
        self.config = CChanConfig(config_dict)
        self.symbol = symbol
        self.lv_list = list(lv_list)

        self.chan = CChan(
            code=symbol,
            data_src=DATA_SRC.CSV,  # placeholder, no historical loading when trigger_step=True
            lv_list=self.lv_list,
            config=self.config,
            autype=autype,
        )

    def feed(self, level: str, tick: Mapping[str, object]) -> Dict[str, object]:
        """
        Feed a single tick into the Chan engine and return lightweight snapshot statistics.

        Args:
            level: textual level such as '1m' or 'day'
            tick: mapping containing at least time/open/high/low/close fields. Time should be datetime.
        """
        level_enum = LEVEL_TO_KL_TYPE.get(level)
        if level_enum is None:
            raise ValueError(f"Unsupported level {level!r}")
        if level_enum not in self.lv_list:
            raise ValueError(f"Level {level} not registered in session (expected one of {self.lv_list})")

        klu = CKLine_Unit(self._to_chan_payload(tick), autofix=True)
        self.chan.trigger_load({level_enum: [klu]})

        kl_list = self.chan[level_enum]
        snapshot = {
            "symbol": self.symbol,
            "level": level,
            "time": klu.time.to_str(),
            "close": klu.close,
            "klc_count": len(kl_list),
            "bi_count": len(kl_list.bi_list),
            "seg_count": len(kl_list.seg_list),
            "zs_count": len(kl_list.zs_list),
            "bsp_count": len(kl_list.bs_point_lst),
        }
        return snapshot

    def _to_chan_payload(self, tick: Mapping[str, object]) -> Dict[str, object]:
        """
        Convert inbound tick mapping into chan-compatible payload.
        Expected keys: time, open, high, low, close, volume?, turnover?, turnover_rate?
        """
        try:
            timestamp: datetime = tick["time"]  # type: ignore[index]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError("Tick payload missing 'time'") from exc
        if not isinstance(timestamp, datetime):
            raise TypeError(f"tick['time'] must be datetime, got {type(timestamp)}")

        ctime = CTime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
        )

        payload: Dict[str, object] = {
            DATA_FIELD.FIELD_TIME: ctime,
            DATA_FIELD.FIELD_OPEN: float(tick["open"]),  # type: ignore[index]
            DATA_FIELD.FIELD_HIGH: float(tick["high"]),  # type: ignore[index]
            DATA_FIELD.FIELD_LOW: float(tick["low"]),  # type: ignore[index]
            DATA_FIELD.FIELD_CLOSE: float(tick["close"]),  # type: ignore[index]
        }

        optional_fields: Iterable[str] = (
            DATA_FIELD.FIELD_VOLUME,
            DATA_FIELD.FIELD_TURNOVER,
            DATA_FIELD.FIELD_TURNRATE,
        )
        for field in optional_fields:
            value = tick.get(field)  # type: ignore[arg-type]
            if value is not None:
                payload[field] = float(value)
        return payload

