"""
同步 A 股前复权日线到本地 SQLite。

示例:
    python3.11 App/sync_a_share_daily_to_sqlite.py --universe hs300 --begin 2018-01-01
    python3.11 App/sync_a_share_daily_to_sqlite.py --codes 600519,000333 --begin 2020-01-01 --end 2026-04-11
    python3.11 App/sync_a_share_daily_to_sqlite.py --codes-file ./hs300_codes.txt --begin 2018-01-01
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    ak = None
    pd = None

from App.generate_stock_recommendations import load_universe
from DataAPI.SQLiteDailyBarAPI import connect_local_db, ensure_daily_bar_schema, resolve_local_db_path, upsert_daily_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步 A 股前复权日线到本地 SQLite")
    parser.add_argument("--begin", required=True, help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="结束日期，格式 YYYY-MM-DD，默认今天",
    )
    parser.add_argument("--codes", default=None, help="逗号分隔的股票代码列表")
    parser.add_argument("--codes-file", default=None, help="股票代码文件，支持 txt/csv/json")
    parser.add_argument("--universe", default=None, help="命名股票池，目前支持 hs300")
    parser.add_argument("--limit", type=int, default=None, help="仅同步前 N 只股票，便于试跑")
    parser.add_argument("--db-path", default=None, help="SQLite 数据库路径，默认 ./data/market_data.sqlite3")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="每只股票之间的等待秒数")
    return parser.parse_args()


def _format_ak_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")


def _fetch_daily_qfq_bars(code: str, begin_time: str, end_time: str) -> List[Dict[str, float | str]]:
    if ak is None or pd is None:
        raise RuntimeError("akshare and pandas are required to sync local market data")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=_format_ak_date(begin_time),
        end_date=_format_ak_date(end_time),
        adjust="qfq",
    )
    if df is None or df.empty:
        return []

    rows: List[Dict[str, float | str]] = []
    for _, row in df.iterrows():
        trade_date = row["日期"]
        if isinstance(trade_date, pd.Timestamp):
            trade_date_value = trade_date.strftime("%Y-%m-%d")
        else:
            trade_date_value = str(trade_date)
        rows.append(
            {
                "trade_date": trade_date_value,
                "open": float(row["开盘"]),
                "high": float(row["最高"]),
                "low": float(row["最低"]),
                "close": float(row["收盘"]),
                "volume": float(row.get("成交量", 0.0) or 0.0),
                "amount": float(row.get("成交额", 0.0) or 0.0),
            }
        )
    return rows


def _resolve_sync_universe(args: argparse.Namespace) -> Sequence[Tuple[str, str]]:
    return load_universe(
        SimpleNamespace(
            codes=args.codes,
            codes_file=args.codes_file,
            universe=args.universe,
            limit=args.limit,
        )
    )


def main() -> int:
    args = parse_args()
    db_path = resolve_local_db_path(args.db_path)
    universe = _resolve_sync_universe(args)
    if not universe:
        raise RuntimeError("Universe is empty")

    conn = connect_local_db(db_path)
    ensure_daily_bar_schema(conn)

    synced_count = 0
    total_rows = 0
    failures: List[Tuple[str, str]] = []
    try:
        for idx, (code, name) in enumerate(universe, 1):
            print(f"[{idx}/{len(universe)}] syncing {code} {name}".rstrip())
            try:
                rows = _fetch_daily_qfq_bars(code, args.begin, args.end)
                written = upsert_daily_bars(conn, code=code, rows=rows, source="akshare")
                total_rows += written
                synced_count += 1
                print(f"  upserted {written} rows")
            except Exception as exc:
                failures.append((code, str(exc)))
                print(f"  failed: {exc}")

            if idx < len(universe) and args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        conn.close()

    print("\nSync Summary")
    print("-" * 80)
    print(f"db_path: {db_path}")
    print(f"universe_size: {len(universe)}")
    print(f"synced_count: {synced_count}")
    print(f"failed_count: {len(failures)}")
    print(f"upserted_rows: {total_rows}")
    if failures:
        for code, error in failures[:20]:
            print(f"failed {code}: {error}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
