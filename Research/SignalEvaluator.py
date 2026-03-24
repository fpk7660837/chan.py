from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .SignalLabeler import SignalLabeler
from .SignalSnapshot import SignalSnapshot


class SignalEvaluator:
    """针对已打标事件做分组统计。"""

    BSP_TYPE_ORDER = {
        "1": 0,
        "1p": 1,
        "2": 2,
        "2s": 3,
        "3a": 4,
        "3b": 5,
        "2,3b": 6,
        "2s,3b": 7,
    }

    def __init__(self, labeler: Optional[SignalLabeler] = None):
        self.labeler = labeler or SignalLabeler()

    def evaluate_signals(
        self,
        snapshots: List[SignalSnapshot],
        *,
        holding_periods: Sequence[int] = (5, 10, 20),
        stop_loss_pct: Optional[float] = 0.05,
        entry_price: str = "next_open",
        exit_price: str = "close",
    ) -> List[Dict[str, Any]]:
        return self.labeler.label_snapshots(
            snapshots,
            holding_periods=holding_periods,
            stop_loss_pct=stop_loss_pct,
            entry_price=entry_price,
            exit_price=exit_price,
        )

    def summarize_by(
        self,
        records: List[Dict[str, Any]],
        *,
        group_field: str,
        holding_period: int = 20,
    ) -> List[Dict[str, Any]]:
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for record in records:
            key = record.get(group_field, "UNKNOWN")
            groups.setdefault(key, []).append(record)

        summaries = []
        for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
            summaries.append(self._build_summary_row(key, group, group_field, holding_period))
        return summaries

    def summarize_overall(
        self,
        records: List[Dict[str, Any]],
        *,
        holding_period: int = 20,
        group_name: str = "ALL",
    ) -> Dict[str, Any]:
        return self._build_summary_row(group_name, records, "group", holding_period)

    def summarize_joint_type_assessments(
        self,
        records: List[Dict[str, Any]],
        *,
        holding_period: int = 20,
    ) -> List[Dict[str, Any]]:
        core_rows = self.summarize_by(records, group_field="bsp_type", holding_period=holding_period)
        core_rows.sort(key=lambda row: self._bsp_type_sort_key(row.get("bsp_type")))

        assessments: List[Dict[str, Any]] = []
        for core_row in core_rows:
            bsp_type = core_row.get("bsp_type")
            type_records = [record for record in records if record.get("bsp_type") == bsp_type]
            assessments.append({
                "bsp_type": bsp_type,
                "core_summary": core_row,
                "joint_setup_summary": self._summarize_joint_rows(
                    type_records,
                    holding_period=holding_period,
                ),
            })
        return assessments

    @classmethod
    def _bsp_type_sort_key(cls, value: Any):
        text = "" if value is None else str(value)
        return (cls.BSP_TYPE_ORDER.get(text, len(cls.BSP_TYPE_ORDER)), text)

    def _summarize_joint_rows(
        self,
        records: List[Dict[str, Any]],
        *,
        holding_period: int,
    ) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            key = str(record.get("joint_setup_key") or "UNKNOWN")
            groups.setdefault(key, []).append(record)

        rows: List[Dict[str, Any]] = []
        total_count = len(records) or 1
        for key, group in groups.items():
            row = self._build_summary_row(key, group, "joint_setup_key", holding_period)
            sample = group[0]
            row["parent_context"] = sample.get("parent_context", "")
            row["child_interval_pattern"] = sample.get("child_interval_pattern", "")
            row["joint_setup_label"] = sample.get("joint_setup_label", "")
            row["share_of_type"] = len(group) / total_count
            row["sure_share"] = float(np.mean([1 if item.get("is_sure") else 0 for item in group])) if group else 0.0
            rows.append(row)

        rows.sort(
            key=lambda row: (
                int(row.get("count", 0)),
                float(row.get("avg_return", 0.0)),
                float(row.get("win_rate", 0.0)),
            ),
            reverse=True,
        )
        return rows

    @staticmethod
    def _build_summary_row(
        key: Any,
        records: List[Dict[str, Any]],
        group_field: str,
        holding_period: int,
    ) -> Dict[str, Any]:
        prefix = f"h{holding_period}"
        available = [record for record in records if record.get(f"{prefix}_available") == 1]
        returns = np.array([record.get(f"{prefix}_return", 0.0) for record in available], dtype=float) if available else np.array([], dtype=float)
        mfes = np.array([record.get(f"{prefix}_mfe", 0.0) for record in available], dtype=float) if available else np.array([], dtype=float)
        maes = np.array([record.get(f"{prefix}_mae", 0.0) for record in available], dtype=float) if available else np.array([], dtype=float)
        return {
            group_field: key,
            "count": len(records),
            "eligible_count": len(available),
            "win_rate": float(np.mean(returns > 0)) if returns.size else 0.0,
            "avg_return": float(np.mean(returns)) if returns.size else 0.0,
            "median_return": float(np.median(returns)) if returns.size else 0.0,
            "avg_mfe": float(np.mean(mfes)) if mfes.size else 0.0,
            "avg_mae": float(np.mean(maes)) if maes.size else 0.0,
            "stop_loss_rate": float(np.mean([record.get(f"{prefix}_stop_loss_hit", 0) for record in available])) if available else 0.0,
            "reverse_signal_rate": float(np.mean([record.get(f"{prefix}_reverse_signal_hit", 0) for record in available])) if available else 0.0,
        }
