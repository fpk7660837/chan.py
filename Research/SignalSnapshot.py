from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from Common.CTime import CTime


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _serialize_zs_seq(zs_list: Iterable[Any]) -> str:
    parts = []
    for zs in zs_list:
        begin_bi = getattr(zs, "begin_bi", None)
        end_bi = getattr(zs, "end_bi", None)
        begin_idx = _safe_int(getattr(begin_bi, "idx", None), -1)
        end_idx = _safe_int(getattr(end_bi, "idx", None), -1)
        high = _safe_float(getattr(zs, "high", 0.0), 0.0)
        low = _safe_float(getattr(zs, "low", 0.0), 0.0)
        is_sure = int(bool(getattr(zs, "is_sure", False)))
        parts.append(f"{begin_idx}-{end_idx}:{low:.4f}-{high:.4f}:{is_sure}")
    return ";".join(parts)


@dataclass
class SignalSnapshot:
    code: str
    level: str
    source: str
    signal_time: CTime
    signal_idx: int
    price: float
    is_buy: bool
    lv_idx: int = 0
    bsp_type: str = ""
    bsp_type_count: int = 0
    signal_kind: str = "bsp"
    is_seg_bsp: bool = False
    bi_idx: int = -1
    seg_idx: int = -1
    is_sure: bool = False
    bi_is_sure: bool = False
    seg_is_sure: bool = False
    divergence_rate: float = 0.0
    zs_count: int = 0
    zs_seq: str = ""
    last_zs_high: float = 0.0
    last_zs_low: float = 0.0
    relate_bsp1_idx: Optional[int] = None
    parent_level: str = ""
    parent_trend_return_10: Optional[float] = None
    parent_trend_state: str = ""
    parent_seg_idx: int = -1
    parent_seg_dir: str = ""
    parent_seg_is_sure: bool = False
    parent_seg_zs_count: int = 0
    parent_seg_multi_zs_count: int = 0
    parent_seg_position: str = ""
    parent_bsp_type: str = ""
    parent_bsp_family: str = ""
    parent_bsp_time: str = ""
    parent_bias_state: str = ""
    parent_context: str = ""
    child_level: str = ""
    child_confirmed: bool = False
    child_interval_state: str = ""
    child_confirm_type: str = ""
    child_confirm_time: str = ""
    child_confirm_idx: int = -1
    child_confirm_delay: int = -1
    child_opposite_first: bool = False
    child_first_signal_side: str = ""
    child_first_signal_type: str = ""
    child_first_signal_family: str = ""
    child_first_signal_delay: int = -1
    child_same_first_type: str = ""
    child_same_first_family: str = ""
    child_opposite_first_type: str = ""
    child_opposite_first_family: str = ""
    child_same_signal_count: int = 0
    child_opposite_signal_count: int = 0
    child_interval_pattern: str = ""
    joint_setup_key: str = ""
    joint_setup_label: str = ""
    note: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    chan: Optional[CChan] = field(default=None, repr=False)
    bsp: Optional[CBS_Point] = field(default=None, repr=False)
    signal_klu: Any = field(default=None, repr=False)

    @property
    def signal_time_str(self) -> str:
        return self.signal_time.to_str()

    @property
    def direction(self) -> str:
        return "buy" if self.is_buy else "sell"

    @property
    def is_seg_signal(self) -> bool:
        return self.is_seg_bsp

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "code": self.code,
            "level": self.level,
            "source": self.source,
            "is_buy": int(self.is_buy),
            "direction": self.direction,
            "signal_time": self.signal_time_str,
            "signal_idx": self.signal_idx,
            "price": self.price,
            "lv_idx": self.lv_idx,
            "bsp_type": self.bsp_type,
            "bsp_type_count": self.bsp_type_count,
            "signal_kind": self.signal_kind,
            "is_seg_bsp": int(self.is_seg_bsp),
            "is_seg_signal": int(self.is_seg_signal),
            "bi_idx": self.bi_idx,
            "seg_idx": self.seg_idx,
            "is_sure": int(self.is_sure),
            "bi_is_sure": int(self.bi_is_sure),
            "seg_is_sure": int(self.seg_is_sure),
            "divergence_rate": self.divergence_rate,
            "zs_count": self.zs_count,
            "zs_seq": self.zs_seq,
            "last_zs_high": self.last_zs_high,
            "last_zs_low": self.last_zs_low,
            "relate_bsp1_idx": self.relate_bsp1_idx,
            "parent_level": self.parent_level,
            "parent_trend_return_10": self.parent_trend_return_10,
            "parent_trend_state": self.parent_trend_state,
            "parent_seg_idx": self.parent_seg_idx,
            "parent_seg_dir": self.parent_seg_dir,
            "parent_seg_is_sure": int(self.parent_seg_is_sure),
            "parent_seg_zs_count": self.parent_seg_zs_count,
            "parent_seg_multi_zs_count": self.parent_seg_multi_zs_count,
            "parent_seg_position": self.parent_seg_position,
            "parent_bsp_type": self.parent_bsp_type,
            "parent_bsp_family": self.parent_bsp_family,
            "parent_bsp_time": self.parent_bsp_time,
            "parent_bias_state": self.parent_bias_state,
            "parent_context": self.parent_context,
            "child_level": self.child_level,
            "child_confirmed": int(self.child_confirmed),
            "child_interval_state": self.child_interval_state,
            "child_confirm_type": self.child_confirm_type,
            "child_confirm_time": self.child_confirm_time,
            "child_confirm_idx": self.child_confirm_idx,
            "child_confirm_delay": self.child_confirm_delay,
            "child_opposite_first": int(self.child_opposite_first),
            "child_first_signal_side": self.child_first_signal_side,
            "child_first_signal_type": self.child_first_signal_type,
            "child_first_signal_family": self.child_first_signal_family,
            "child_first_signal_delay": self.child_first_signal_delay,
            "child_same_first_type": self.child_same_first_type,
            "child_same_first_family": self.child_same_first_family,
            "child_opposite_first_type": self.child_opposite_first_type,
            "child_opposite_first_family": self.child_opposite_first_family,
            "child_same_signal_count": self.child_same_signal_count,
            "child_opposite_signal_count": self.child_opposite_signal_count,
            "child_interval_pattern": self.child_interval_pattern,
            "joint_setup_key": self.joint_setup_key,
            "joint_setup_label": self.joint_setup_label,
            "note": self.note,
        }
        for key, value in sorted(self.metadata.items()):
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[f"meta_{key}"] = value
        return result

    @classmethod
    def from_bsp(
        cls,
        chan: CChan,
        bsp: CBS_Point,
        *,
        lv_idx: int = 0,
        is_seg_signal: bool = False,
        source: str = "chan_bsp",
        note: str = "",
    ) -> "SignalSnapshot":
        seg = getattr(bsp.bi, "parent_seg", None) if bsp.bi is not None else None
        zs_list = getattr(seg, "zs_lst", []) if seg is not None else []
        last_zs = zs_list[-1] if zs_list else None
        divergence_rate = 0.0
        if getattr(bsp, "features", None) is not None:
            for key, value in bsp.features.items():
                if key == "divergence_rate":
                    divergence_rate = _safe_float(value, 0.0)
                    break

        return cls(
            code=chan.code,
            level=chan.lv_list[lv_idx].name,
            source=source,
            signal_time=bsp.klu.time,
            signal_idx=bsp.klu.idx,
            price=_safe_float(bsp.klu.close),
            is_buy=bool(bsp.is_buy),
            lv_idx=lv_idx,
            bsp_type=bsp.type2str(),
            bsp_type_count=len(bsp.type),
            signal_kind="seg_bsp" if is_seg_signal else "bsp",
            is_seg_bsp=is_seg_signal,
            bi_idx=getattr(getattr(bsp, "bi", None), "idx", -1),
            seg_idx=getattr(seg, "idx", -1),
            is_sure=bool(getattr(seg if is_seg_signal else getattr(bsp, "bi", None), "is_sure", False)),
            bi_is_sure=bool(getattr(getattr(bsp, "bi", None), "is_sure", False)),
            seg_is_sure=bool(getattr(seg, "is_sure", False)),
            divergence_rate=divergence_rate,
            zs_count=len(zs_list),
            zs_seq=_serialize_zs_seq(zs_list),
            last_zs_high=_safe_float(getattr(last_zs, "high", 0.0), 0.0),
            last_zs_low=_safe_float(getattr(last_zs, "low", 0.0), 0.0),
            relate_bsp1_idx=getattr(getattr(getattr(bsp, "relate_bsp1", None), "bi", None), "idx", None),
            note=note,
            chan=chan,
            bsp=bsp,
            signal_klu=bsp.klu,
        )

    @classmethod
    def from_klu(
        cls,
        chan: CChan,
        klu: Any,
        *,
        lv_idx: int = 0,
        source: str,
        direction: str,
        signal_kind: str,
        note: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SignalSnapshot":
        return cls(
            code=chan.code,
            level=chan.lv_list[lv_idx].name,
            source=source,
            signal_time=klu.time,
            signal_idx=klu.idx,
            price=_safe_float(getattr(klu, "close", 0.0), 0.0),
            is_buy=direction == "buy",
            lv_idx=lv_idx,
            bsp_type=signal_kind,
            bsp_type_count=1,
            signal_kind=signal_kind,
            is_seg_bsp=False,
            note=note,
            metadata=metadata or {},
            chan=chan,
            signal_klu=klu,
        )
