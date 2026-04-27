from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.import_stock_bar_1m_raw_to_clickhouse import (  # noqa: E402
    DEFAULT_COMPOSE_FILE,
    DEFAULT_ENV_FILE,
    DEFAULT_MARKET_DATA_ROOT,
    _sql_literal,
    parse_years,
    run_clickhouse_query,
)


DEFAULT_REPORT_DIR = DEFAULT_MARKET_DATA_ROOT / "reports" / "bar_quality"


def month_bounds(year_month: str) -> Tuple[str, str]:
    year = int(year_month[:4])
    month = int(year_month[4:6])
    start = f"{year:04d}-{month:02d}-01 00:00:00"
    if month == 12:
        end = f"{year + 1:04d}-01-01 00:00:00"
    else:
        end = f"{year:04d}-{month + 1:02d}-01 00:00:00"
    return start, end


def schema_sql() -> str:
    return """
CREATE TABLE IF NOT EXISTS market.bar_anomalies
(
    symbol LowCardinality(String),
    level LowCardinality(String),
    trade_time DateTime64(3, 'Asia/Shanghai'),
    trade_date Date,
    rule LowCardinality(String),
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    amount Float64,
    source LowCardinality(String) DEFAULT '',
    updated_at DateTime64(3, 'Asia/Shanghai') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY (level, toYYYYMM(trade_time))
ORDER BY (level, symbol, trade_time, rule)
SETTINGS index_granularity = 8192;

CREATE OR REPLACE VIEW market.bars_1m_clean AS
SELECT *
FROM market.bars
WHERE level = '1m'
  AND open > 0
  AND high > 0
  AND low > 0
  AND close > 0
  AND high >= greatest(open, low, close)
  AND low <= least(open, high, close)
  AND volume >= 0
  AND amount >= 0
  AND NOT isNaN(open)
  AND NOT isNaN(high)
  AND NOT isNaN(low)
  AND NOT isNaN(close)
  AND NOT isNaN(volume)
  AND NOT isNaN(amount)
  AND open <= 10000
  AND high <= 10000
  AND low <= 10000
  AND close <= 10000
  AND (symbol, level, trade_time) NOT IN
      (SELECT symbol, level, trade_time FROM market.bar_anomalies WHERE level = '1m');
""".strip()


def anomaly_rules_expression() -> str:
    return """
arrayFilter(rule -> rule != '', [
    if(open <= 0 OR high <= 0 OR low <= 0 OR close <= 0, 'non_positive_price', ''),
    if(high < greatest(open, low, close) OR low > least(open, high, close), 'bad_ohlc', ''),
    if(volume < 0 OR amount < 0, 'negative_volume_amount', ''),
    if(isNaN(open) OR isNaN(high) OR isNaN(low) OR isNaN(close) OR isNaN(volume) OR isNaN(amount), 'nan_value', ''),
    if(open > 10000 OR high > 10000 OR low > 10000 OR close > 10000, 'stock_price_gt_10000', ''),
    if(volume > 0 AND amount >= close * 100 AND close > 0 AND (amount / (volume * close * 100) < 0.01 OR amount / (volume * close * 100) > 100), 'amount_price_volume_mismatch', ''),
    if(volume = 0 AND amount >= greatest(open, high, low, close) * 100, 'zero_volume_positive_amount', ''),
    if(volume > 0 AND amount = 0, 'positive_volume_zero_amount', '')
])
""".strip()


def insert_anomalies_query(*, level: str, year_month: str) -> str:
    start, end = month_bounds(year_month)
    rules = anomaly_rules_expression()
    return f"""
INSERT INTO market.bar_anomalies
    (symbol, level, trade_time, trade_date, rule, open, high, low, close, volume, amount, source)
SELECT
    symbol,
    level,
    trade_time,
    trade_date,
    arrayJoin(rules) AS rule,
    open,
    high,
    low,
    close,
    volume,
    amount,
    source
FROM
(
    SELECT
        symbol,
        level,
        trade_time,
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount,
        source,
        {rules} AS rules
    FROM market.bars
    WHERE level = {_sql_literal(level)}
      AND trade_time >= toDateTime64({_sql_literal(start)}, 3, 'Asia/Shanghai')
      AND trade_time < toDateTime64({_sql_literal(end)}, 3, 'Asia/Shanghai')
)
WHERE length(rules) > 0
""".strip()


def list_months(*, level: str, years: Optional[Set[str]], env_file: Path, compose_file: Path) -> List[str]:
    filters = [f"level = {_sql_literal(level)}"]
    if years:
        start_year = min(int(year) for year in years)
        end_year = max(int(year) for year in years) + 1
        filters.append(
            f"trade_time >= toDateTime64({_sql_literal(f'{start_year:04d}-01-01 00:00:00')}, 3, 'Asia/Shanghai')"
        )
        filters.append(
            f"trade_time < toDateTime64({_sql_literal(f'{end_year:04d}-01-01 00:00:00')}, 3, 'Asia/Shanghai')"
        )
    query = (
        "SELECT toYYYYMM(trade_time) AS year_month "
        "FROM market.bars "
        f"WHERE {' AND '.join(filters)} "
        "GROUP BY year_month ORDER BY year_month FORMAT TSV"
    )
    output = run_clickhouse_query(query, env_file=env_file, compose_file=compose_file)
    months = [line.strip() for line in output.splitlines() if line.strip()]
    if years:
        months = [month for month in months if month[:4] in years]
    return months


def delete_month_anomalies_query(*, level: str, year_month: str) -> str:
    return f"""
ALTER TABLE market.bar_anomalies DELETE
WHERE level = {_sql_literal(level)}
  AND toYYYYMM(trade_time) = {int(year_month)}
SETTINGS mutations_sync = 2
""".strip()


def apply_schema(*, env_file: Path, compose_file: Path) -> None:
    run_clickhouse_query(schema_sql(), env_file=env_file, compose_file=compose_file)


def scan_anomalies(
    *,
    level: str,
    years: Optional[Set[str]],
    env_file: Path,
    compose_file: Path,
    truncate: bool,
    month_limit: Optional[int] = None,
    replace_existing_months: bool = False,
) -> Dict[str, int | float]:
    apply_schema(env_file=env_file, compose_file=compose_file)
    if truncate:
        run_clickhouse_query("TRUNCATE TABLE market.bar_anomalies", env_file=env_file, compose_file=compose_file)
        print("truncated market.bar_anomalies", flush=True)
    months = list_months(level=level, years=years, env_file=env_file, compose_file=compose_file)
    if month_limit is not None:
        months = months[:month_limit]
    started_at = time.monotonic()
    stats: Dict[str, int | float] = {"months": len(months), "scanned": 0, "seconds": 0.0}
    for month in months:
        month_start = time.monotonic()
        if replace_existing_months and not truncate:
            run_clickhouse_query(
                delete_month_anomalies_query(level=level, year_month=month),
                env_file=env_file,
                compose_file=compose_file,
            )
        run_clickhouse_query(insert_anomalies_query(level=level, year_month=month), env_file=env_file, compose_file=compose_file)
        stats["scanned"] = int(stats["scanned"]) + 1
        elapsed = time.monotonic() - month_start
        count = run_clickhouse_query(
            f"SELECT count() FROM market.bar_anomalies WHERE level = {_sql_literal(level)} AND toYYYYMM(trade_time) = {month}",
            env_file=env_file,
            compose_file=compose_file,
        )
        print(f"scanned_month {month} anomaly_rows={count.strip()} seconds={elapsed:.1f}", flush=True)
    stats["seconds"] = time.monotonic() - started_at
    return stats


def report_paths(report_dir: Path) -> Dict[str, Path]:
    return {
        "summary": report_dir / "bars_1m_quality_summary.json",
        "by_rule": report_dir / "bars_1m_anomalies_by_rule.csv",
        "by_symbol_day": report_dir / "bars_1m_anomalies_by_symbol_day.csv",
        "samples": report_dir / "bars_1m_anomaly_samples.csv",
        "clean_summary": report_dir / "bars_1m_clean_summary.json",
    }


def _write_tsv_as_csv(output: str, path: Path) -> None:
    lines = [line for line in output.splitlines() if line.strip()]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        for line in lines:
            writer.writerow(line.split("\t"))


def write_reports(*, report_dir: Path, env_file: Path, compose_file: Path) -> Dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = report_paths(report_dir)

    summary_query = """
SELECT
    count() AS anomaly_rows,
    uniqExact(symbol, trade_time) AS anomalous_bars,
    uniqExact(symbol) AS symbols,
    uniqExact(trade_date) AS dates,
    min(trade_time) AS min_time,
    max(trade_time) AS max_time
FROM market.bar_anomalies
WHERE level = '1m'
FORMAT JSON
""".strip()
    clean_summary_query = """
SELECT
    (SELECT count() FROM market.bars WHERE level = '1m') AS raw_rows,
    (SELECT uniqExact(symbol, level, trade_time) FROM market.bar_anomalies WHERE level = '1m') AS anomalous_bars,
    raw_rows - anomalous_bars AS clean_rows_after_anomaly_exclusion
FORMAT JSON
""".strip()
    by_rule_query = """
SELECT rule, count() AS anomaly_rows, uniqExact(symbol, trade_time) AS anomalous_bars, uniqExact(symbol) AS symbols, uniqExact(trade_date) AS dates
FROM market.bar_anomalies
WHERE level = '1m'
GROUP BY rule
ORDER BY anomaly_rows DESC
FORMAT TSVWithNames
""".strip()
    by_symbol_day_query = """
SELECT symbol, trade_date, count() AS anomaly_rows, groupUniqArray(rule) AS rules, min(trade_time) AS first_time, max(trade_time) AS last_time
FROM market.bar_anomalies
WHERE level = '1m'
GROUP BY symbol, trade_date
ORDER BY anomaly_rows DESC, symbol, trade_date
LIMIT 10000
FORMAT TSVWithNames
""".strip()
    samples_query = """
SELECT symbol, trade_time, rule, open, high, low, close, volume, amount, source
FROM market.bar_anomalies
WHERE level = '1m'
ORDER BY trade_time, symbol, rule
LIMIT 20000
FORMAT TSVWithNames
""".strip()

    paths["summary"].write_text(run_clickhouse_query(summary_query, env_file=env_file, compose_file=compose_file), encoding="utf-8")
    paths["clean_summary"].write_text(
        run_clickhouse_query(clean_summary_query, env_file=env_file, compose_file=compose_file),
        encoding="utf-8",
    )
    _write_tsv_as_csv(run_clickhouse_query(by_rule_query, env_file=env_file, compose_file=compose_file), paths["by_rule"])
    _write_tsv_as_csv(
        run_clickhouse_query(by_symbol_day_query, env_file=env_file, compose_file=compose_file),
        paths["by_symbol_day"],
    )
    _write_tsv_as_csv(run_clickhouse_query(samples_query, env_file=env_file, compose_file=compose_file), paths["samples"])
    return {name: str(path) for name, path in paths.items()}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build market.bars anomaly table and clean 1m view")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    parser.add_argument("--level", default="1m", help="Bar level to scan")
    parser.add_argument("--years", default=None, help="Year filter, e.g. 2024 or 2020-2024")
    parser.add_argument("--truncate", action="store_true", help="TRUNCATE market.bar_anomalies before scanning")
    parser.add_argument("--month-limit", type=int, default=None, help="Limit months for smoke tests")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Quality report directory")
    parser.add_argument("--schema-only", action="store_true", help="Only create table/view")
    parser.add_argument("--reports-only", action="store_true", help="Only write reports from existing anomaly table")
    parser.add_argument(
        "--replace-existing-months",
        action="store_true",
        help="Delete anomaly rows for each scanned month before re-inserting them",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    env_file = Path(args.env_file).expanduser()
    compose_file = Path(args.compose_file).expanduser()
    report_dir = Path(args.report_dir).expanduser()

    if args.schema_only:
        apply_schema(env_file=env_file, compose_file=compose_file)
        print("schema applied", flush=True)
        return 0
    if not args.reports_only:
        stats = scan_anomalies(
            level=args.level,
            years=parse_years(args.years),
            env_file=env_file,
            compose_file=compose_file,
            truncate=args.truncate,
            month_limit=args.month_limit,
            replace_existing_months=args.replace_existing_months,
        )
        print(f"scan_summary: {stats}", flush=True)
    paths = write_reports(report_dir=report_dir, env_file=env_file, compose_file=compose_file)
    print(f"reports: {json.dumps(paths, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
