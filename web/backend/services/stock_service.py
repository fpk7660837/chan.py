"""
Stock search helper that proxies BaoStock listings with lightweight caching.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class StockInfo:
    code: str
    name: str
    status: str
    ipo_date: Optional[str] = None
    out_date: Optional[str] = None

    def as_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "name": self.name,
            "status": self.status,
            "ipoDate": self.ipo_date,
            "outDate": self.out_date,
        }


class StockService:
    """Pull and cache BaoStock listings, expose simple keyword search."""

    def __init__(self, cache_ttl_seconds: int = 3600) -> None:
        self._lock = threading.RLock()
        self._cache: List[StockInfo] = []
        self._last_refresh = 0.0
        self._cache_ttl = cache_ttl_seconds

    def search(self, keyword: str, limit: int = 20) -> List[Dict[str, str]]:
        normalized = (keyword or "").strip().lower()
        if not normalized:
            return []
        limit = max(1, min(limit or 20, 100))
        records = self._ensure_cache()
        results: List[StockInfo] = []
        for item in records:
            if normalized in item.code.lower() or normalized in item.name.lower():
                results.append(item)
                if len(results) >= limit:
                    break
        return [record.as_dict() for record in results]

    # -------------------- internal helpers -------------------- #
    def _ensure_cache(self) -> List[StockInfo]:
        with self._lock:
            is_stale = (time.time() - self._last_refresh) > self._cache_ttl
            if self._cache and not is_stale:
                return list(self._cache)
        refreshed = self._refresh_cache()
        if refreshed:
            return refreshed
        with self._lock:
            return list(self._cache)

    def _refresh_cache(self) -> List[StockInfo]:
        try:
            stocks = self._fetch_from_baostock()
        except Exception:  # pylint: disable=broad-except
            return []
        with self._lock:
            self._cache = stocks
            self._last_refresh = time.time()
            return list(self._cache)

    def _fetch_from_baostock(self) -> List[StockInfo]:
        try:
            import baostock as bs
        except ImportError as exc:  # pragma: no cover - missing optional dependency
            raise RuntimeError("baostock package is required for stock search") from exc

        login_result = bs.login()
        if login_result.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login_result.error_msg}")

        records: List[StockInfo] = []
        try:
            rs = bs.query_stock_basic()
            if rs.error_code != "0":
                raise RuntimeError(f"BaoStock query failed: {rs.error_msg}")
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                record = StockInfo(
                    code=row[0],
                    name=row[1],
                    ipo_date=row[2] or None,
                    out_date=row[3] or None,
                    status=row[5] or "",
                )
                records.append(record)
        finally:
            bs.logout()
        records.sort(key=lambda item: item.code)
        return records


stock_service = StockService()
