from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.import_stock_bar_1m_raw_to_clickhouse import (  # noqa: E402
    DEFAULT_COMPOSE_FILE,
    DEFAULT_ENV_FILE,
    _sql_literal,
    run_clickhouse_query,
)


def default_view_configs() -> dict[str, dict[str, object]]:
    return {
        "market.bars_1m_train_balanced": {
            "level": "1m",
            "start_date": "2015-01-05",
            "end_date": "2024-12-31",
            "include_delisting_arrangement": False,
            "include_pre_delist_below_1yuan": False,
        },
        "market.bars_1m_holdout_2025_strict": {
            "level": "1m",
            "start_date": "2025-01-02",
            "end_date": "2025-12-31",
            "include_delisting_arrangement": False,
            "include_pre_delist_below_1yuan": False,
        },
        "market.bars_1m_holdout_2025_research": {
            "level": "1m",
            "start_date": "2025-01-02",
            "end_date": "2025-12-31",
            "include_delisting_arrangement": True,
            "include_pre_delist_below_1yuan": True,
        },
    }


def default_aggregate_view_configs() -> dict[str, dict[str, str]]:
    return {
        "market.bars_5m_train_balanced": {
            "source_view": "market.bars_1m_train_balanced",
            "target_level": "5m",
        },
        "market.bars_30m_train_balanced": {
            "source_view": "market.bars_1m_train_balanced",
            "target_level": "30m",
        },
        "market.bars_day_train_balanced": {
            "source_view": "market.bars_1m_train_balanced",
            "target_level": "day",
        },
        "market.bars_week_train_balanced": {
            "source_view": "market.bars_1m_train_balanced",
            "target_level": "week",
        },
        "market.bars_5m_holdout_2025_strict": {
            "source_view": "market.bars_1m_holdout_2025_strict",
            "target_level": "5m",
        },
        "market.bars_30m_holdout_2025_strict": {
            "source_view": "market.bars_1m_holdout_2025_strict",
            "target_level": "30m",
        },
        "market.bars_day_holdout_2025_strict": {
            "source_view": "market.bars_1m_holdout_2025_strict",
            "target_level": "day",
        },
        "market.bars_week_holdout_2025_strict": {
            "source_view": "market.bars_1m_holdout_2025_strict",
            "target_level": "week",
        },
    }


def _excluded_statuses(*, include_delisting_arrangement: bool, include_pre_delist_below_1yuan: bool) -> list[str]:
    excluded = ["zero_turnover_placeholder", "post_delist_placeholder"]
    if not include_delisting_arrangement:
        excluded.append("delisting_arrangement")
    if not include_pre_delist_below_1yuan:
        excluded.append("pre_delist_below_1yuan")
    return excluded


def _filtered_select_sql(
    *,
    level: str,
    start_date: str,
    end_date: str,
    include_delisting_arrangement: bool,
    include_pre_delist_below_1yuan: bool,
) -> str:
    excluded_statuses = _excluded_statuses(
        include_delisting_arrangement=include_delisting_arrangement,
        include_pre_delist_below_1yuan=include_pre_delist_below_1yuan,
    )
    status_values = ", ".join(_sql_literal(value) for value in excluded_statuses)
    return f"""
SELECT
    b.symbol AS symbol,
    b.level AS level,
    b.trade_time AS trade_time,
    b.trade_date AS trade_date,
    b.open AS open,
    b.high AS high,
    b.low AS low,
    b.close AS close,
    b.volume AS volume,
    b.amount AS amount,
    b.source AS source
FROM market.bars AS b
WHERE b.level = {_sql_literal(level)}
  AND b.trade_date >= toDate({_sql_literal(start_date)})
  AND b.trade_date <= toDate({_sql_literal(end_date)})
  AND (b.symbol, b.level, b.trade_time) NOT IN
      (SELECT symbol, level, trade_time FROM market.bar_anomalies WHERE level = {_sql_literal(level)})
  AND (b.symbol, b.level, b.trade_date) NOT IN
      (
          SELECT symbol, level, trade_date
          FROM market.bar_status_flags
          WHERE level = {_sql_literal(level)}
            AND trade_date >= toDate({_sql_literal(start_date)})
            AND trade_date <= toDate({_sql_literal(end_date)})
            AND status IN ({status_values})
      )
""".strip()


def dataset_filter_query(
    *,
    level: str,
    start_date: str,
    end_date: str,
    include_delisting_arrangement: bool,
    include_pre_delist_below_1yuan: bool,
) -> str:
    return f"""
{_filtered_select_sql(
    level=level,
    start_date=start_date,
    end_date=end_date,
    include_delisting_arrangement=include_delisting_arrangement,
    include_pre_delist_below_1yuan=include_pre_delist_below_1yuan,
)}
ORDER BY b.symbol, b.trade_time
""".strip()


def create_view_sql(
    *,
    view_name: str,
    level: str,
    start_date: str,
    end_date: str,
    include_delisting_arrangement: bool,
    include_pre_delist_below_1yuan: bool,
) -> str:
    return f"""
CREATE OR REPLACE VIEW {view_name} AS
{_filtered_select_sql(
    level=level,
    start_date=start_date,
    end_date=end_date,
    include_delisting_arrangement=include_delisting_arrangement,
    include_pre_delist_below_1yuan=include_pre_delist_below_1yuan,
)}
""".strip()


def _minute_bucket_expression(minutes: int) -> str:
    return f"""
toStartOfInterval(
    if(
        (toHour(source_trade_time) = 15) AND (toMinute(source_trade_time) = 0),
        source_trade_time - toIntervalMinute({minutes}),
        source_trade_time
    ),
    INTERVAL {minutes} MINUTE
)
""".strip()


def create_aggregate_view_sql(*, view_name: str, source_view: str, target_level: str) -> str:
    if target_level == "5m":
        bucket_expression = _minute_bucket_expression(5)
        return f"""
CREATE OR REPLACE VIEW {view_name} AS
WITH prepared AS (
    SELECT
        symbol,
        trade_time AS source_trade_time,
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount,
        source,
        {bucket_expression} AS bucket_time
    FROM {source_view}
)
SELECT
    symbol AS symbol,
    '5m' AS level,
    bucket_time AS trade_time,
    toDate(bucket_time) AS trade_date,
    argMin(open, source_trade_time) AS open,
    max(high) AS high,
    min(low) AS low,
    argMax(close, source_trade_time) AS close,
    sum(volume) AS volume,
    sum(amount) AS amount,
    any(source) AS source
FROM prepared
GROUP BY symbol, bucket_time
ORDER BY symbol, trade_time
""".strip()

    if target_level == "30m":
        bucket_expression = _minute_bucket_expression(30)
        return f"""
CREATE OR REPLACE VIEW {view_name} AS
WITH prepared AS (
    SELECT
        symbol,
        trade_time AS source_trade_time,
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount,
        source,
        {bucket_expression} AS bucket_time
    FROM {source_view}
)
SELECT
    symbol AS symbol,
    '30m' AS level,
    bucket_time AS trade_time,
    toDate(bucket_time) AS trade_date,
    argMin(open, source_trade_time) AS open,
    max(high) AS high,
    min(low) AS low,
    argMax(close, source_trade_time) AS close,
    sum(volume) AS volume,
    sum(amount) AS amount,
    any(source) AS source
FROM prepared
GROUP BY symbol, bucket_time
ORDER BY symbol, trade_time
""".strip()

    if target_level == "day":
        return f"""
CREATE OR REPLACE VIEW {view_name} AS
WITH prepared AS (
    SELECT
        symbol,
        trade_time AS source_trade_time,
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount,
        source
    FROM {source_view}
)
SELECT
    symbol AS symbol,
    'day' AS level,
    toDateTime64(trade_date, 3, 'Asia/Shanghai') AS trade_time,
    trade_date AS trade_date,
    argMin(open, source_trade_time) AS open,
    max(high) AS high,
    min(low) AS low,
    argMax(close, source_trade_time) AS close,
    sum(volume) AS volume,
    sum(amount) AS amount,
    any(source) AS source
FROM prepared
GROUP BY symbol, trade_date
ORDER BY symbol, trade_time
""".strip()

    if target_level == "week":
        return f"""
CREATE OR REPLACE VIEW {view_name} AS
WITH prepared AS (
    SELECT
        symbol,
        trade_time AS source_trade_time,
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount,
        source,
        toMonday(trade_date) AS week_trade_date
    FROM {source_view}
)
SELECT
    symbol AS symbol,
    'week' AS level,
    toDateTime64(week_trade_date, 3, 'Asia/Shanghai') AS trade_time,
    week_trade_date AS trade_date,
    argMin(open, source_trade_time) AS open,
    max(high) AS high,
    min(low) AS low,
    argMax(close, source_trade_time) AS close,
    sum(volume) AS volume,
    sum(amount) AS amount,
    any(source) AS source
FROM prepared
GROUP BY symbol, week_trade_date
ORDER BY symbol, trade_time
""".strip()

    raise ValueError(f"Unsupported target_level: {target_level}")


def profile_query(
    *,
    level: str,
    start_date: str,
    end_date: str,
    include_delisting_arrangement: bool,
    include_pre_delist_below_1yuan: bool,
) -> str:
    excluded_statuses = _excluded_statuses(
        include_delisting_arrangement=include_delisting_arrangement,
        include_pre_delist_below_1yuan=include_pre_delist_below_1yuan,
    )
    explicit_excluded_statuses = [status for status in excluded_statuses if status != "zero_turnover_placeholder"]
    explicit_values = ", ".join(_sql_literal(value) for value in explicit_excluded_statuses) or _sql_literal("__none__")
    return f"""
WITH filtered AS (
    {_filtered_select_sql(
        level=level,
        start_date=start_date,
        end_date=end_date,
        include_delisting_arrangement=include_delisting_arrangement,
        include_pre_delist_below_1yuan=include_pre_delist_below_1yuan,
    )}
)
SELECT
    count() AS kept_rows,
    uniqExact(fs.symbol) AS kept_symbols,
    uniqExact(fs.trade_date) AS kept_trade_dates,
    (
        SELECT uniqExact(symbol, trade_date)
        FROM market.bar_status_flags
        WHERE level = {_sql_literal(level)}
          AND trade_date >= toDate({_sql_literal(start_date)})
          AND trade_date <= toDate({_sql_literal(end_date)})
          AND status = 'zero_turnover_placeholder'
    ) AS excluded_placeholder_days,
    (
        SELECT uniqExact(symbol, trade_date)
        FROM market.bar_status_flags
        WHERE level = {_sql_literal(level)}
          AND trade_date >= toDate({_sql_literal(start_date)})
          AND trade_date <= toDate({_sql_literal(end_date)})
          AND status IN ({explicit_values})
    ) AS excluded_status_flag_days
FROM filtered AS fs
FORMAT JSON
""".strip()


def build_profiles(*, env_file: Path, compose_file: Path) -> dict:
    windows = {
        "balanced_2015_2024": default_view_configs()["market.bars_1m_train_balanced"],
        "holdout_2025_strict": default_view_configs()["market.bars_1m_holdout_2025_strict"],
        "holdout_2025_research": default_view_configs()["market.bars_1m_holdout_2025_research"],
    }
    results = {}
    for name, config in windows.items():
        output = run_clickhouse_query(
            profile_query(**config),
            env_file=env_file,
            compose_file=compose_file,
        )
        results[name] = json.loads(output)
    return results


def apply_views(*, env_file: Path, compose_file: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    for view_name, config in default_view_configs().items():
        run_clickhouse_query(
            create_view_sql(view_name=view_name, **config),
            env_file=env_file,
            compose_file=compose_file,
        )
        results[view_name] = "applied"
    for view_name, config in default_aggregate_view_configs().items():
        run_clickhouse_query(
            create_aggregate_view_sql(view_name=view_name, **config),
            env_file=env_file,
            compose_file=compose_file,
        )
        results[view_name] = "applied"
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable training dataset filter profiles for market.bars")
    parser.add_argument("--apply-views", action="store_true", help="Create or replace the default training dataset views")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env_file = Path(args.env_file).expanduser()
    compose_file = Path(args.compose_file).expanduser()
    view_results = None
    if args.apply_views:
        view_results = apply_views(env_file=env_file, compose_file=compose_file)
    results = build_profiles(
        env_file=env_file,
        compose_file=compose_file,
    )
    payload: dict[str, object] = {"profiles": results}
    if view_results is not None:
        payload["views"] = view_results
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
