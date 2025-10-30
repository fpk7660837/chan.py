"""
Pydantic schemas shared by alert-related modules.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class KLineTick(BaseModel):
    """Incoming K线 tick payload used for trigger_load feeding."""

    symbol: str = Field(..., description="证券代码，例如 sh.600000")
    level: str = Field(default="1m", description="K线周期，例如 1m / 5m / day")
    time: datetime = Field(..., description="K线结束时间")
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    turnover: Optional[float] = None
    turnover_rate: Optional[float] = Field(default=None, alias="turnoverRate")
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)
