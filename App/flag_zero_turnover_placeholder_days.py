from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.flag_delisting_status_windows import apply_schema  # noqa: E402
from App.import_stock_bar_1m_raw_to_clickhouse import (  # noqa: E402
    DEFAULT_COMPOSE_FILE,
    DEFAULT_ENV_FILE,
    _sql_literal,
    run_clickhouse_query,
)


DEFAULT_SOURCE = "derived_zero_turnover_placeholder_2015_2025_v1"
DEFAULT_NOTE = "derived from 241-bar zero-turnover flat day rule"


def year_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    windows: list[tuple[str, str]] = []
    year = start.year
    while year <= end.year:
        window_start = start if year == start.year else date(year, 1, 1)
        window_end = end if year == end.year else date(year, 12, 31)
        windows.append((window_start.isoformat(), window_end.isoformat()))
        year += 1
    return windows


def delete_placeholder_flags_query(*, level: str, source: str) -> str:
    return f"""
ALTER TABLE market.bar_status_flags DELETE
WHERE level = {_sql_literal(level)}
  AND status = 'zero_turnover_placeholder'
  AND source = {_sql_literal(source)}
SETTINGS mutations_sync = 2
""".strip()


def insert_placeholder_flags_query(*, level: str, start_date: str, end_date: str, source: str) -> str:
    return f"""
INSERT INTO market.bar_status_flags
    (symbol, level, trade_date, status, note, source)
SELECT
    symbol,
    level,
    trade_date,
    'zero_turnover_placeholder' AS status,
    {_sql_literal(DEFAULT_NOTE)} AS note,
    {_sql_literal(source)} AS source
FROM
(
    SELECT
        symbol,
        level,
        trade_date,
        count() AS day_bar_count,
        sum(volume) AS day_volume_sum,
        sum(amount) AS day_amount_sum,
        uniqExact(open) AS day_open_values,
        uniqExact(high) AS day_high_values,
        uniqExact(low) AS day_low_values,
        uniqExact(close) AS day_close_values
    FROM market.bars
    WHERE level = {_sql_literal(level)}
      AND trade_date >= toDate({_sql_literal(start_date)})
      AND trade_date <= toDate({_sql_literal(end_date)})
    GROUP BY symbol, level, trade_date
)
WHERE day_bar_count = 241
  AND day_volume_sum = 0
  AND day_amount_sum = 0
  AND day_open_values = 1
  AND day_high_values = 1
  AND day_low_values = 1
  AND day_close_values = 1
""".strip()


def summarize_placeholder_flags_query(*, level: str, source: str) -> str:
    return f"""
SELECT
    count() AS rows,
    uniqExact(symbol) AS symbols,
    min(trade_date) AS first_day,
    max(trade_date) AS last_day
FROM market.bar_status_flags
WHERE level = {_sql_literal(level)}
  AND status = 'zero_turnover_placeholder'
  AND source = {_sql_literal(source)}
FORMAT JSON
""".strip()


def apply_placeholder_flags(
    *,
    level: str,
    start_date: str,
    end_date: str,
    source: str,
    env_file: Path,
    compose_file: Path,
    execute: bool,
) -> dict:
    if not execute:
        output = run_clickhouse_query(
            summarize_placeholder_flags_query(level=level, source=source),
            env_file=env_file,
            compose_file=compose_file,
        )
        return {"applied": False, "existing": json.loads(output)}

    apply_schema(env_file=env_file, compose_file=compose_file)
    run_clickhouse_query(delete_placeholder_flags_query(level=level, source=source), env_file=env_file, compose_file=compose_file)
    for window_start, window_end in year_windows(start_date, end_date):
        run_clickhouse_query(
            insert_placeholder_flags_query(level=level, start_date=window_start, end_date=window_end, source=source),
            env_file=env_file,
            compose_file=compose_file,
        )
    output = run_clickhouse_query(
        summarize_placeholder_flags_query(level=level, source=source),
        env_file=env_file,
        compose_file=compose_file,
    )
    return {"applied": True, "summary": json.loads(output)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag zero-turnover placeholder symbol-days into market.bar_status_flags")
    parser.add_argument("--level", default="1m", help="Bar level to scan")
    parser.add_argument("--start-date", default="2015-01-05", help="Inclusive start date")
    parser.add_argument("--end-date", default="2025-12-31", help="Inclusive end date")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source label for inserted flags")
    parser.add_argument("--execute", action="store_true", help="Apply flags to ClickHouse")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = apply_placeholder_flags(
        level=args.level,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
        env_file=Path(args.env_file).expanduser(),
        compose_file=Path(args.compose_file).expanduser(),
        execute=args.execute,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
