from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.import_stock_bar_1m_raw_to_clickhouse import (  # noqa: E402
    DEFAULT_COMPOSE_FILE,
    DEFAULT_ENV_FILE,
    DEFAULT_MARKET_DATA_ROOT,
    _sql_literal,
    run_clickhouse_query,
)


DEFAULT_TRADE_DATES: Tuple[str, ...] = ("2010-11-01", "2011-02-16")
DEFAULT_BACKUP_DIR = DEFAULT_MARKET_DATA_ROOT / "backup" / "clickhouse" / "bar_repairs"


def parse_trade_dates(trade_dates: Optional[str]) -> Tuple[str, ...]:
    if not trade_dates:
        return DEFAULT_TRADE_DATES
    parsed = tuple(part.strip() for part in trade_dates.split(",") if part.strip())
    if not parsed:
        raise ValueError("no trade dates provided")
    return parsed


def _trade_date_sql(trade_dates: Sequence[str]) -> str:
    return ", ".join(_sql_literal(trade_date) for trade_date in trade_dates)


def candidate_where_sql(*, level: str, trade_dates: Sequence[str]) -> str:
    trade_date_sql = _trade_date_sql(trade_dates)
    return f"""
level = {_sql_literal(level)}
  AND trade_date IN ({trade_date_sql})
  AND (symbol, trade_date) IN
      (
          SELECT symbol, trade_date
          FROM market.bars
          WHERE level = {_sql_literal(level)}
            AND trade_date IN ({trade_date_sql})
          GROUP BY symbol, trade_date
          HAVING count() = 241
             AND max(high) / min(low) > 10
      )
  AND toHour(trade_time) = 15
  AND toMinute(trade_time) = 0
  AND close > 0
  AND open / close > 9
  AND open / close < 11
  AND high / close > 9
  AND high / close < 11
  AND low / close > 0.8
  AND low / close < 1.2
""".strip()


def summarize_candidates_query(*, level: str, trade_dates: Sequence[str]) -> str:
    where_sql = candidate_where_sql(level=level, trade_dates=trade_dates)
    return f"""
SELECT
    toString(trade_date) AS trade_date,
    count() AS candidate_rows,
    uniqExact(symbol) AS symbols,
    min(open / close) AS min_open_close_ratio,
    max(open / close) AS max_open_close_ratio
FROM market.bars
WHERE {where_sql}
GROUP BY trade_date
ORDER BY trade_date
FORMAT JSONEachRow
""".strip()


def export_candidates_query(*, level: str, trade_dates: Sequence[str]) -> str:
    where_sql = candidate_where_sql(level=level, trade_dates=trade_dates)
    return f"""
SELECT
    symbol,
    level,
    toString(trade_time) AS trade_time_text,
    open,
    high,
    low,
    close,
    volume,
    amount,
    source,
    open / 10.0 AS repaired_open,
    high / 10.0 AS repaired_high
FROM market.bars
WHERE {where_sql}
ORDER BY trade_date, symbol, trade_time
FORMAT CSVWithNames
""".strip()


def update_candidates_query(*, level: str, trade_dates: Sequence[str]) -> str:
    where_sql = candidate_where_sql(level=level, trade_dates=trade_dates)
    return f"""
ALTER TABLE market.bars UPDATE
    open = open / 10.0,
    high = high / 10.0
WHERE {where_sql}
SETTINGS mutations_sync = 2
""".strip()


def remaining_candidates_query(*, level: str, trade_dates: Sequence[str]) -> str:
    where_sql = candidate_where_sql(level=level, trade_dates=trade_dates)
    return f"""
SELECT count() AS candidate_rows
FROM market.bars
WHERE {where_sql}
FORMAT TSV
""".strip()


def _load_json_each_row(output: str) -> List[Dict[str, object]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def default_backup_path(*, trade_dates: Sequence[str], backup_dir: Path) -> Path:
    suffix = "_".join(trade_date.replace("-", "") for trade_date in trade_dates)
    return backup_dir / f"bars_tail_auction_x10_{suffix}_{date.today():%Y%m%d}.csv"


def apply_repairs(
    *,
    level: str,
    trade_dates: Sequence[str],
    env_file: Path,
    compose_file: Path,
    execute: bool,
    backup_path: Optional[Path],
) -> Dict[str, object]:
    summary_rows = _load_json_each_row(
        run_clickhouse_query(
            summarize_candidates_query(level=level, trade_dates=trade_dates),
            env_file=env_file,
            compose_file=compose_file,
        )
    )
    candidate_rows = sum(int(row["candidate_rows"]) for row in summary_rows)
    result: Dict[str, object] = {
        "level": level,
        "trade_dates": list(trade_dates),
        "candidate_rows": candidate_rows,
        "summary": summary_rows,
        "applied": execute,
    }
    if candidate_rows == 0:
        return result
    if not execute:
        return result

    export_csv = run_clickhouse_query(
        export_candidates_query(level=level, trade_dates=trade_dates),
        env_file=env_file,
        compose_file=compose_file,
    )
    target_backup_path = backup_path or default_backup_path(trade_dates=trade_dates, backup_dir=DEFAULT_BACKUP_DIR)
    target_backup_path.parent.mkdir(parents=True, exist_ok=True)
    target_backup_path.write_text(export_csv + "\n", encoding="utf-8")

    run_clickhouse_query(
        update_candidates_query(level=level, trade_dates=trade_dates),
        env_file=env_file,
        compose_file=compose_file,
    )
    remaining = run_clickhouse_query(
        remaining_candidates_query(level=level, trade_dates=trade_dates),
        env_file=env_file,
        compose_file=compose_file,
    ).strip()
    result["backup_path"] = str(target_backup_path)
    result["remaining_candidate_rows"] = int(remaining or "0")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair 15:00 tail-auction open/high x10 rows in market.bars")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    parser.add_argument("--level", default="1m", help="Bar level to repair")
    parser.add_argument("--trade-dates", default=",".join(DEFAULT_TRADE_DATES), help="Comma-separated trade dates")
    parser.add_argument("--backup-path", default=None, help="Optional CSV backup path for the repaired rows")
    parser.add_argument("--execute", action="store_true", help="Apply the UPDATE mutation")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = apply_repairs(
        level=args.level,
        trade_dates=parse_trade_dates(args.trade_dates),
        env_file=Path(args.env_file).expanduser(),
        compose_file=Path(args.compose_file).expanduser(),
        execute=args.execute,
        backup_path=Path(args.backup_path).expanduser() if args.backup_path else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
