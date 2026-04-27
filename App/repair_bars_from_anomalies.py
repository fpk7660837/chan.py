from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.import_stock_bar_1m_raw_to_clickhouse import (  # noqa: E402
    DEFAULT_COMPOSE_FILE,
    DEFAULT_ENV_FILE,
    _sql_literal,
    parse_years,
    run_clickhouse_query,
)


REPAIRABLE_RULES = (
    "non_positive_price",
    "bad_ohlc",
    "negative_volume_amount",
    "nan_value",
    "stock_price_gt_10000",
    "amount_price_volume_mismatch",
    "zero_volume_positive_amount",
    "positive_volume_zero_amount",
)


def _repairable_rule_sql() -> str:
    return ", ".join(_sql_literal(rule) for rule in REPAIRABLE_RULES)


def list_months_with_repairable_anomalies(
    *,
    level: str,
    years: Optional[Set[str]],
    env_file: Path,
    compose_file: Path,
) -> List[str]:
    filters = [f"level = {_sql_literal(level)}", f"rule IN ({_repairable_rule_sql()})"]
    if years:
        year_list = ", ".join(str(int(year)) for year in sorted(years))
        filters.append(f"toYear(trade_time) IN ({year_list})")
    query = (
        "SELECT toString(toYYYYMM(trade_time)) AS year_month "
        "FROM market.bar_anomalies "
        f"WHERE {' AND '.join(filters)} "
        "GROUP BY year_month ORDER BY year_month FORMAT TSV"
    )
    output = run_clickhouse_query(query, env_file=env_file, compose_file=compose_file)
    return [line.strip() for line in output.splitlines() if line.strip()]


def summarize_repairable_anomalies_query(*, level: str, year_month: str) -> str:
    return f"""
SELECT
    rule,
    count() AS anomaly_rows,
    uniqCombined64(symbol, trade_time) AS bars,
    uniqCombined64(symbol) AS symbols
FROM market.bar_anomalies
WHERE level = {_sql_literal(level)}
  AND toYYYYMM(trade_time) = {int(year_month)}
  AND rule IN ({_repairable_rule_sql()})
GROUP BY rule
ORDER BY anomaly_rows DESC, rule
FORMAT JSON
""".strip()


def delete_repairable_bars_query(*, level: str, year_month: str) -> str:
    return f"""
ALTER TABLE market.bars DELETE
WHERE level = {_sql_literal(level)}
  AND toYYYYMM(trade_time) = {int(year_month)}
  AND (symbol, level, trade_time) IN
      (
          SELECT symbol, level, trade_time
          FROM market.bar_anomalies
          WHERE level = {_sql_literal(level)}
            AND toYYYYMM(trade_time) = {int(year_month)}
            AND rule IN ({_repairable_rule_sql()})
      )
SETTINGS mutations_sync = 2
""".strip()


def delete_repairable_anomalies_query(*, level: str, year_month: str) -> str:
    return f"""
ALTER TABLE market.bar_anomalies DELETE
WHERE level = {_sql_literal(level)}
  AND toYYYYMM(trade_time) = {int(year_month)}
  AND rule IN ({_repairable_rule_sql()})
SETTINGS mutations_sync = 2
""".strip()


def apply_repairs(
    *,
    level: str,
    years: Optional[Set[str]],
    env_file: Path,
    compose_file: Path,
    execute: bool,
    month_limit: Optional[int] = None,
) -> Dict[str, object]:
    months = list_months_with_repairable_anomalies(
        level=level,
        years=years,
        env_file=env_file,
        compose_file=compose_file,
    )
    if month_limit is not None:
        months = months[:month_limit]

    result: Dict[str, object] = {"months": months, "applied": execute, "summaries": {}}
    for month in months:
        summary = run_clickhouse_query(
            summarize_repairable_anomalies_query(level=level, year_month=month),
            env_file=env_file,
            compose_file=compose_file,
        )
        result["summaries"][month] = json.loads(summary)
        print(f"repairable_month {month} summary={summary}", flush=True)
        if not execute:
            continue
        run_clickhouse_query(
            delete_repairable_bars_query(level=level, year_month=month),
            env_file=env_file,
            compose_file=compose_file,
        )
        run_clickhouse_query(
            delete_repairable_anomalies_query(level=level, year_month=month),
            env_file=env_file,
            compose_file=compose_file,
        )
        print(f"applied_month {month}", flush=True)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete repairable market.bars rows based on market.bar_anomalies")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    parser.add_argument("--level", default="1m", help="Bar level to repair")
    parser.add_argument("--years", default=None, help="Year filter, e.g. 2025 or 2000,2002,2003,2006")
    parser.add_argument("--month-limit", type=int, default=None, help="Limit repair months for smoke tests")
    parser.add_argument("--execute", action="store_true", help="Apply ALTER DELETE mutations")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    apply_repairs(
        level=args.level,
        years=parse_years(args.years),
        env_file=Path(args.env_file).expanduser(),
        compose_file=Path(args.compose_file).expanduser(),
        execute=args.execute,
        month_limit=args.month_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
