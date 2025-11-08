"""
In-memory backtest service coordinating Chan calculations for historical analysis.
"""
from __future__ import annotations

import bisect
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from services.chan_service import ChanService
from schemas.backtest import (
    BacktestEnqueueRequest,
    BacktestFilters,
    BacktestJob,
    BacktestJobStatus,
    BacktestRunOptions,
    BacktestSummary,
    StrengthSummary,
)


@dataclass
class _BacktestJobRecord:
    """Internal representation of a queued backtest job."""

    id: str
    stock_code: str
    level: str
    data_src: str
    begin_time: Optional[str]
    end_time: Optional[str]
    chan_config: Dict[str, Any] = field(default_factory=dict)
    indicator_overrides: Dict[str, Any] = field(default_factory=dict)
    filters: BacktestFilters = field(default_factory=BacktestFilters)
    limit_kline_count: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: BacktestJobStatus = BacktestJobStatus.PENDING
    last_run_at: Optional[datetime] = None
    last_summary: Optional[BacktestSummary] = None
    error_message: Optional[str] = None


class _CandleContext:
    """Helper encapsulating candle-derived aggregates for range calculations."""

    def __init__(
        self,
        timestamps: List[int],
        volume_prefix: List[float],
        macd_abs_prefix: List[float],
    ) -> None:
        self._timestamps = timestamps
        self._volume_prefix = volume_prefix
        self._macd_abs_prefix = macd_abs_prefix

    def _find_index(self, timestamp: int) -> Optional[int]:
        if not self._timestamps:
            return None
        idx = bisect.bisect_left(self._timestamps, timestamp)
        if idx <= 0:
            return 0
        if idx >= len(self._timestamps):
            return len(self._timestamps) - 1
        before = self._timestamps[idx - 1]
        after = self._timestamps[idx]
        return idx if abs(after - timestamp) <= abs(timestamp - before) else idx - 1

    def calc_range(self, start_ts: Optional[int], end_ts: Optional[int]) -> Dict[str, Optional[float]]:
        if start_ts is None or end_ts is None or not self._timestamps:
            return {"volume": None, "macd_abs": None}
        start_idx = self._find_index(start_ts)
        end_idx = self._find_index(end_ts)
        if start_idx is None or end_idx is None:
            return {"volume": None, "macd_abs": None}
        low = min(start_idx, end_idx)
        high = max(start_idx, end_idx)
        volume = None
        if self._volume_prefix:
            volume = self._volume_prefix[high]
            if low > 0:
                volume -= self._volume_prefix[low - 1]
        macd_abs = None
        if self._macd_abs_prefix:
            macd_abs = self._macd_abs_prefix[high]
            if low > 0:
                macd_abs -= self._macd_abs_prefix[low - 1]
        return {"volume": volume, "macd_abs": macd_abs}


class BacktestService:
    """Facade for queueing and running Chan backtests."""

    def __init__(self) -> None:
        self._chan_service = ChanService()
        self._jobs: Dict[str, _BacktestJobRecord] = {}
        self._lock = threading.Lock()

    # ------------------------- Public API ------------------------- #
    def list_jobs(self) -> List[BacktestJob]:
        with self._lock:
            records = list(self._jobs.values())
        records.sort(key=lambda job: job.created_at, reverse=True)
        return [self._serialize(record) for record in records]

    def enqueue(self, payload: BacktestEnqueueRequest) -> BacktestJob:
        record = _BacktestJobRecord(
            id=uuid4().hex,
            stock_code=payload.stockCode,
            level=payload.level,
            data_src=payload.dataSrc,
            begin_time=payload.beginTime,
            end_time=payload.endTime,
            chan_config=dict(payload.chanConfig or {}),
            indicator_overrides=dict(payload.indicatorOverrides or {}),
            filters=payload.filters,
            limit_kline_count=payload.limitKlineCount,
        )
        with self._lock:
            self._jobs[record.id] = record

        try:
            summary = self._generate_summary(record)
            record.last_summary = summary
        except Exception as exc:  # pylint: disable=broad-except
            record.status = BacktestJobStatus.FAILED
            record.error_message = str(exc)
            raise

        return self._serialize(record)

    def get_job(self, job_id: str) -> Optional[BacktestJob]:
        with self._lock:
            record = self._jobs.get(job_id)
        if not record:
            return None
        return self._serialize(record)

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def run_job(self, job_id: str, options: Optional[BacktestRunOptions] = None) -> BacktestJob:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                raise KeyError(f"Job {job_id} not found")
            record.status = BacktestJobStatus.RUNNING
            record.error_message = None

        try:
            summary = self._generate_summary(record, force_refresh=bool(options and options.forceRefresh))
        except Exception as exc:  # pylint: disable=broad-except
            with self._lock:
                record.status = BacktestJobStatus.FAILED
                record.error_message = str(exc)
            raise

        with self._lock:
            record.status = BacktestJobStatus.COMPLETED
            record.last_run_at = datetime.now(timezone.utc)
            record.last_summary = summary
        return self._serialize(record)

    # ------------------------- Internal helpers ------------------------- #
    def _serialize(self, record: _BacktestJobRecord) -> BacktestJob:
        return BacktestJob(
            id=record.id,
            stockCode=record.stock_code,
            level=record.level,
            dataSrc=record.data_src,
            beginTime=record.begin_time,
            endTime=record.end_time,
            filters=record.filters,
            chanConfig=record.chan_config,
            indicatorOverrides=record.indicator_overrides,
            createdAt=record.created_at,
            status=record.status,
            lastRunAt=record.last_run_at,
            lastSummary=record.last_summary,
            errorMessage=record.error_message,
        )

    def _build_analysis_params(self, record: _BacktestJobRecord) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "code": record.stock_code,
            "lv_list": [record.level],
            "data_src": record.data_src,
            "chan_config": record.chan_config or {},
            "plot_bi": True,
            "plot_seg": True,
            "plot_zs": True,
            "plot_bsp": True,
            "plot_macd": True,
        }
        if record.begin_time:
            params["begin_time"] = record.begin_time
        if record.end_time:
            params["end_time"] = record.end_time
        if record.limit_kline_count:
            params["limit_kl_count"] = record.limit_kline_count
        for key, value in (record.indicator_overrides or {}).items():
            if key in {"code", "lv_list", "chan_config"}:
                continue
            params[key] = value
        # Ensure critical outputs stay enabled even if overrides differ
        for required in ("plot_bi", "plot_seg", "plot_zs", "plot_bsp"):
            params[required] = True
        return params

    def _generate_summary(self, record: _BacktestJobRecord, force_refresh: bool = False) -> BacktestSummary:
        params = self._build_analysis_params(record)
        if force_refresh:
            params["_force_refresh"] = True  # hint for future caching layers
        result = self._chan_service.calculate(params)
        summary_payload = self._summarize_result(result, record.filters, record.chan_config)
        return BacktestSummary(**summary_payload)

    # ------------------ Result summarisation helpers ------------------ #
    def _summarize_result(
        self,
        analysis_result: Dict[str, Any],
        filters: BacktestFilters,
        chan_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not analysis_result:
            return {
                "status": "missing",
                "signalCount": 0,
                "sampleSignal": None,
                "strength": None,
                "generatedAt": datetime.now(timezone.utc),
                "signals": [],
                "resultInfo": None,
            }

        signal_filters = self._normalize_filters(filters)
        bsp_list = analysis_result.get("bsp_list") or []
        filtered = self._filter_signals(bsp_list, signal_filters)
        sorted_signals = self._sort_signals(filtered)
        sample_signal = dict(sorted_signals[0]) if sorted_signals else None

        strength_summary = self._derive_strength_summary(analysis_result, chan_config)

        formatted_signals = [self._format_signal(entry) for entry in sorted_signals]
        return {
            "status": "ready",
            "signalCount": len(filtered),
            "sampleSignal": sample_signal,
            "strength": strength_summary,
            "generatedAt": datetime.now(timezone.utc),
            "signals": formatted_signals,
            "resultInfo": self._build_result_info(analysis_result),
        }

    def _normalize_filters(self, filters: BacktestFilters) -> Dict[str, Any]:
        return {
            "directions": [direction.value for direction in filters.directions] or ["buy", "sell"],
            "types": filters.types or [],
        }

    def _filter_signals(self, bsp_list: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not bsp_list:
            return []
        directions = set(filters.get("directions") or ["buy", "sell"])
        type_whitelist = set(filters.get("types") or [])
        filtered: List[Dict[str, Any]] = []
        for item in bsp_list:
            if not isinstance(item, dict):
                continue
            direction = "buy" if item.get("is_buy") else "sell"
            if direction not in directions:
                continue
            if type_whitelist:
                type_key = self._normalize_bsp_type(item.get("type"))
                if type_key not in type_whitelist:
                    continue
            filtered.append(item)
        return filtered

    def _sort_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sorted_signals: List[Dict[str, Any]] = []
        for item in signals:
            timestamp = self._convert_time(item.get("time"))
            if timestamp is None:
                continue
            enriched = dict(item)
            enriched["timestamp"] = timestamp
            enriched["typeKey"] = self._normalize_bsp_type(item.get("type"))
            sorted_signals.append(enriched)
        sorted_signals.sort(key=lambda entry: entry["timestamp"], reverse=True)
        return sorted_signals

    def _format_signal(self, item: Dict[str, Any]) -> Dict[str, Any]:
        formatted = {
            "timestamp": item.get("timestamp"),
            "type": item.get("type"),
            "typeKey": item.get("typeKey"),
            "price": self._safe_float(item.get("price")),
            "code": item.get("code"),
            "level": item.get("level"),
            "is_buy": bool(item.get("is_buy")),
        }
        for key in ("time", "idx", "comment", "extra"):
            if key in item:
                formatted[key] = item[key]
        return formatted

    def _build_result_info(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        meta = analysis_result.get("meta") if isinstance(analysis_result.get("meta"), dict) else {}
        return {
            "klineCount": len(analysis_result.get("kline_data") or []),
            "rawKlineCount": len(analysis_result.get("raw_kline_data") or []),
            "biCount": len(analysis_result.get("bi_list") or []),
            "segCount": len(analysis_result.get("seg_list") or []),
            "zsCount": len(analysis_result.get("zs_list") or []),
            "bspCount": len(analysis_result.get("bsp_list") or []),
            "meta": meta,
        }

    def _normalize_bsp_type(self, value: Any) -> str:
        if value is None:
            return "未标注"
        text = str(value).strip()
        return text if text else "未标注"

    def _safe_float(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value) if math.isfinite(value) else None
        if value is None:
            return None
        try:
            parsed = float(str(value))
        except (TypeError, ValueError):
            return None
        return float(parsed) if math.isfinite(parsed) else None

    def _convert_time(self, value: Any) -> Optional[int]:
        if not value:
            return None
        text = str(value).strip().replace("/", "-")
        if "T" not in text and " " in text:
            text = text.replace(" ", "T", 1)
        if "." in text:
            # remove fractional seconds that break fromisoformat when timezone suffix absent
            head, _, tail = text.partition(".")
            if tail and ("+" in tail or "Z" in tail):
                tail = tail.split("+")[0].split("Z")[0]
                text = f"{head}.{tail}"
            else:
                text = head
        if len(text) == 16 and text.count(":") == 1:
            text = f"{text}:00"
        candidates = [
            text,
            text.replace("T", " "),
        ]
        for candidate in candidates:
            try:
                dt = datetime.fromisoformat(candidate)
                return int(dt.timestamp() * 1000)
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(text.replace("T", " "), fmt)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        return None

    def _calculate_macd_histogram(self, values: List[float], fast: int, slow: int, signal: int) -> List[float]:
        if not values:
            return []

        def calc_ema(series: List[float], period: int) -> List[float]:
            if not series:
                return []
            smoothing = 2 / (period + 1)
            ema = series[0]
            ema_values = [ema]
            for value in series[1:]:
                ema = (value * smoothing) + ema * (1 - smoothing)
                ema_values.append(ema)
            return ema_values

        ema_fast = calc_ema(values, max(fast, 1))
        ema_slow = calc_ema(values, max(slow, 1))
        if len(ema_fast) != len(ema_slow):
            return []
        macd_line = [fast_val - slow_val for fast_val, slow_val in zip(ema_fast, ema_slow)]
        ema_signal = calc_ema(macd_line, max(signal, 1))
        if len(ema_signal) != len(macd_line):
            return []
        return [macd - signal_val for macd, signal_val in zip(macd_line, ema_signal)]

    def _derive_strength_summary(self, analysis_result: Dict[str, Any], chan_config: Dict[str, Any]) -> Optional[StrengthSummary]:
        candle_series = analysis_result.get("kline_data") or []
        if not candle_series:
            return None

        candles: List[Dict[str, Any]] = []
        volume_prefix: List[float] = []
        total_volume = 0.0
        for item in candle_series:
            timestamp = self._convert_time(item.get("time") or item.get("time_begin"))
            close = self._safe_float(item.get("close"))
            volume = self._safe_float(item.get("volume")) or 0.0
            if timestamp is None or close is None:
                continue
            total_volume += max(volume, 0.0)
            volume_prefix.append(total_volume)
            candles.append({"timestamp": timestamp, "close": close})
        if not candles:
            return None

        timestamps = [item["timestamp"] for item in candles]
        macd_config = chan_config.get("macd") if isinstance(chan_config, dict) else {}
        macd_fast = int(macd_config.get("fast", 12) or 12)
        macd_slow = int(macd_config.get("slow", 26) or 26)
        macd_signal = int(macd_config.get("signal", 9) or 9)
        macd_histogram = self._calculate_macd_histogram(
            [item["close"] for item in candles],
            macd_fast,
            macd_slow,
            macd_signal,
        )
        macd_abs_prefix: List[float] = []
        macd_total = 0.0
        for value in macd_histogram:
            macd_total += abs(value) if math.isfinite(value) else 0.0
            macd_abs_prefix.append(macd_total)

        candle_context = _CandleContext(timestamps, volume_prefix, macd_abs_prefix)

        bi_metrics = self._build_strength_metrics(
            analysis_result.get("bi_list") or [],
            candle_context,
            label_prefix="笔",
        )
        seg_metrics = self._build_strength_metrics(
            analysis_result.get("seg_list") or [],
            candle_context,
            label_prefix="段",
        )

        bi_strength = self._compute_strength_comparison(bi_metrics)
        seg_strength = self._compute_strength_comparison(seg_metrics)

        if not bi_strength and not seg_strength:
            return None
        return StrengthSummary(biStrength=bi_strength, segStrength=seg_strength)

    def _build_strength_metrics(
        self,
        items: List[Dict[str, Any]],
        candle_context: _CandleContext,
        label_prefix: str,
    ) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        for idx, item in enumerate(items or []):
            start_ts = self._convert_time(item.get("begin_time"))
            end_ts = self._convert_time(item.get("end_time"))
            begin_price = self._safe_float(item.get("begin_price"))
            end_price = self._safe_float(item.get("end_price"))
            direction = item.get("dir") or item.get("direction")

            if start_ts is None or end_ts is None:
                continue
            if begin_price is None or end_price is None:
                continue
            duration_minutes = abs(end_ts - start_ts) / 60000.0
            if duration_minutes <= 0:
                continue
            amplitude = abs(end_price - begin_price)
            if amplitude <= 0:
                continue

            if not direction:
                if end_price > begin_price:
                    direction = "up"
                elif end_price < begin_price:
                    direction = "down"
                else:
                    continue
            direction_key = "down" if str(direction).lower().startswith("down") else "up"

            extras = candle_context.calc_range(start_ts, end_ts)
            efficiency = amplitude / duration_minutes if duration_minutes else None

            metrics.append(
                {
                    "idx": item.get("idx", idx),
                    "startTs": start_ts,
                    "endTs": end_ts,
                    "beginPrice": begin_price,
                    "endPrice": end_price,
                    "dir": direction_key,
                    "duration": duration_minutes,
                    "amplitude": amplitude,
                    "efficiency": efficiency,
                    "volume": extras.get("volume"),
                    "macdAbs": extras.get("macd_abs"),
                    "label": item.get("label") or f"{label_prefix}#{item.get('idx') or idx + 1}",
                }
            )
        return metrics

    def _compute_strength_comparison(self, metrics: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if len(metrics) < 2:
            return None
        ordered = sorted(metrics, key=lambda entry: entry["endTs"])
        latest = ordered[-1]
        previous = None
        for candidate in reversed(ordered[:-1]):
            if candidate["dir"] == latest["dir"]:
                previous = candidate
                break
        if not previous:
            return None

        amplitude_ratio = (
            latest["amplitude"] / previous["amplitude"]
            if previous["amplitude"]
            else None
        )
        duration_ratio = (
            latest["duration"] / previous["duration"]
            if previous["duration"]
            else None
        )
        efficiency_ratio = (
            (latest.get("efficiency") or 0) / previous["efficiency"]
            if previous.get("efficiency")
            else None
        )
        volume_ratio = (
            (latest.get("volume") or 0) / previous["volume"]
            if previous.get("volume")
            else None
        )
        macd_ratio = (
            (latest.get("macdAbs") or 0) / previous["macdAbs"]
            if previous.get("macdAbs")
            else None
        )

        volume_weakening = volume_ratio is not None and volume_ratio < 1
        macd_weakening = macd_ratio is not None and macd_ratio < 1
        strong_momentum = (
            (amplitude_ratio is not None and amplitude_ratio > 1.1)
            or (macd_ratio is not None and macd_ratio > 1.1)
        )
        weakening = (
            amplitude_ratio is not None
            and amplitude_ratio < 1
            and (
                (efficiency_ratio is not None and efficiency_ratio < 1)
                or volume_weakening
                or macd_weakening
            )
        )

        hint_notes: List[str] = []
        if volume_weakening:
            hint_notes.append("量能缩减")
        elif volume_ratio is not None and volume_ratio > 1.1:
            hint_notes.append("量能放大")
        if macd_weakening:
            hint_notes.append("MACD走弱")
        elif macd_ratio is not None and macd_ratio > 1.1:
            hint_notes.append("MACD走强")

        hint = ""
        hint_type = None
        if strong_momentum:
            hint = " / ".join(hint_notes) if hint_notes else "力度增强"
            hint_type = "strong"
        elif weakening:
            hint = " / ".join(hint_notes) if hint_notes else "力度减弱，关注背驰"
            hint_type = "weak"
        elif hint_notes:
            hint = " / ".join(hint_notes)

        return {
            "dir": latest["dir"],
            "current": latest,
            "previous": previous,
            "amplitudeRatio": amplitude_ratio,
            "durationRatio": duration_ratio,
            "efficiencyRatio": efficiency_ratio,
            "volumeRatio": volume_ratio,
            "macdRatio": macd_ratio,
            "volumeWeakening": volume_weakening,
            "macdWeakening": macd_weakening,
            "weakening": weakening,
            "strongMomentum": strong_momentum,
            "hint": hint,
            "hintType": hint_type,
        }
