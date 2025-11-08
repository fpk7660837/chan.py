"""
Pydantic models for backtest API endpoints.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class BacktestDirection(str, Enum):
    """Supported direction filters for buy/sell points."""

    BUY = "buy"
    SELL = "sell"


class BacktestFilters(BaseModel):
    """Signal filter options applied before computing summaries."""

    directions: List[BacktestDirection] = Field(
        default_factory=lambda: [BacktestDirection.BUY, BacktestDirection.SELL],
        description="Signal directions to include.",
    )
    types: List[str] = Field(default_factory=list, description="Specific BSP type keys to include.")

    @validator("directions", pre=True)
    def _sanitize_directions(cls, value: Any) -> List[BacktestDirection]:
        if not value:
            return [BacktestDirection.BUY, BacktestDirection.SELL]
        if isinstance(value, (str, BacktestDirection)):
            value = [value]
        if not isinstance(value, list):
            return [BacktestDirection.BUY, BacktestDirection.SELL]
        normalized: List[BacktestDirection] = []
        for item in value:
            if isinstance(item, BacktestDirection):
                normalized.append(item)
            else:
                candidate = str(item).strip().lower()
                if candidate == "sell":
                    normalized.append(BacktestDirection.SELL)
                else:
                    normalized.append(BacktestDirection.BUY)
        # remove duplicates while preserving order
        unique = []
        seen = set()
        for item in normalized:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique or [BacktestDirection.BUY, BacktestDirection.SELL]

    @validator("types", pre=True)
    def _sanitize_types(cls, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        normalized: List[str] = []
        seen: set[str] = set()
        for item in value:
            if item is None:
                continue
            candidate = str(item).strip()
            if candidate and candidate not in seen:
                normalized.append(candidate)
                seen.add(candidate)
        return normalized


class BacktestJobStatus(str, Enum):
    """Lifecycle status for a queued backtest job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StrengthSummary(BaseModel):
    """Optional strength comparison data for bi / segment."""

    biStrength: Optional[Dict[str, Any]] = Field(default=None, description="Latest bi strength snapshot.")
    segStrength: Optional[Dict[str, Any]] = Field(default=None, description="Latest segment strength snapshot.")


class BacktestSummary(BaseModel):
    """Lightweight summary returned after analysis."""

    status: str = Field("ready", description="Summary status flag.")
    signalCount: int = Field(0, ge=0, description="Number of signals after filters.")
    sampleSignal: Optional[Dict[str, Any]] = Field(default=None, description="Most recent signal payload.")
    strength: Optional[StrengthSummary] = Field(default=None, description="Strength comparison data.")
    generatedAt: datetime = Field(default_factory=datetime.utcnow, description="Timestamp summary was generated.")


class BacktestJob(BaseModel):
    """Serialized job returned to clients."""

    id: str
    stockCode: str
    level: str
    dataSrc: str = "BAO_STOCK"
    beginTime: Optional[str] = None
    endTime: Optional[str] = None
    filters: BacktestFilters = Field(default_factory=BacktestFilters)
    chanConfig: Dict[str, Any] = Field(default_factory=dict)
    indicatorOverrides: Dict[str, Any] = Field(default_factory=dict)
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    status: BacktestJobStatus = BacktestJobStatus.PENDING
    lastRunAt: Optional[datetime] = None
    lastSummary: Optional[BacktestSummary] = None
    errorMessage: Optional[str] = None


class BacktestEnqueueRequest(BaseModel):
    """Request payload when enqueuing a new backtest run."""

    stockCode: str = Field(..., description="Symbol code to analyze.")
    level: str = Field(..., description="K-line level, e.g. 5m/15m/day.")
    dataSrc: str = Field("BAO_STOCK", description="Underlying data source.")
    beginTime: Optional[str] = Field(None, description="Optional analysis start date.")
    endTime: Optional[str] = Field(None, description="Optional analysis end date.")
    chanConfig: Dict[str, Any] = Field(default_factory=dict, description="Chan configuration overrides.")
    indicatorOverrides: Dict[str, Any] = Field(default_factory=dict, description="Visualization indicator toggles.")
    filters: BacktestFilters = Field(default_factory=BacktestFilters, description="Signal filters to apply.")
    limitKlineCount: Optional[int] = Field(
        None,
        ge=10,
        le=5000,
        description="Optional limit for returned K-line count to reduce load.",
    )


class BacktestRunOptions(BaseModel):
    """Options passed when executing an existing job."""

    forceRefresh: bool = Field(False, description="Force underlying analysis data to refresh.")
