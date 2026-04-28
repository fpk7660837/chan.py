from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.import_stock_bar_1m_raw_to_clickhouse import (  # noqa: E402
    DEFAULT_COMPOSE_FILE,
    DEFAULT_ENV_FILE,
    _sql_literal,
    run_clickhouse_query,
)


DEFAULT_SOURCE = "manual_review_2026_04_28"


@dataclass(frozen=True)
class StatusWindow:
    symbol: str
    level: str
    start_date: str
    end_date: str
    status: str
    note: str


DEFAULT_WINDOWS: Tuple[StatusWindow, ...] = (
    StatusWindow(
        symbol="600804.SH",
        level="1m",
        start_date="2025-06-10",
        end_date="2025-06-30",
        status="delisting_arrangement",
        note="reviewed 2025 low-price window; delisting arrangement trading period",
    ),
    StatusWindow(
        symbol="600804.SH",
        level="1m",
        start_date="2025-07-01",
        end_date="2025-07-02",
        status="post_delist_placeholder",
        note="reviewed 2025 low-price window; post-delist zero-turnover placeholder days",
    ),
    StatusWindow(
        symbol="002336.SZ",
        level="1m",
        start_date="2025-06-17",
        end_date="2025-07-03",
        status="delisting_arrangement",
        note="reviewed 2025 low-price window; delisting arrangement trading period",
    ),
    StatusWindow(
        symbol="000584.SZ",
        level="1m",
        start_date="2025-06-20",
        end_date="2025-07-10",
        status="delisting_arrangement",
        note="reviewed 2025 low-price window; delisting arrangement trading period",
    ),
    StatusWindow(
        symbol="600462.SH",
        level="1m",
        start_date="2025-06-24",
        end_date="2025-07-14",
        status="delisting_arrangement",
        note="reviewed 2025 low-price window; delisting arrangement trading period",
    ),
    StatusWindow(
        symbol="600462.SH",
        level="1m",
        start_date="2025-07-15",
        end_date="2025-07-16",
        status="post_delist_placeholder",
        note="reviewed 2025 low-price window; post-delist zero-turnover placeholder days",
    ),
    StatusWindow(
        symbol="000622.SZ",
        level="1m",
        start_date="2025-06-25",
        end_date="2025-07-15",
        status="delisting_arrangement",
        note="reviewed 2025 low-price window; delisting arrangement trading period",
    ),
    StatusWindow(
        symbol="600190.SH",
        level="1m",
        start_date="2025-06-30",
        end_date="2025-07-04",
        status="delisting_arrangement",
        note="reviewed 2025 low-price window; delisting arrangement trading period",
    ),
    StatusWindow(
        symbol="300208.SZ",
        level="1m",
        start_date="2025-06-30",
        end_date="2025-07-16",
        status="delisting_arrangement",
        note="reviewed 2025 low-price window; delisting arrangement trading period",
    ),
    StatusWindow(
        symbol="300280.SZ",
        level="1m",
        start_date="2025-09-19",
        end_date="2025-10-13",
        status="delisting_arrangement",
        note="reviewed 2025 low-price window; delisting arrangement trading period",
    ),
    StatusWindow(
        symbol="600200.SH",
        level="1m",
        start_date="2025-12-09",
        end_date="2025-12-29",
        status="delisting_arrangement",
        note="reviewed 2025 low-price window; delisting arrangement trading period",
    ),
    StatusWindow(
        symbol="300108.SZ",
        level="1m",
        start_date="2025-04-17",
        end_date="2025-04-23",
        status="pre_delist_below_1yuan",
        note="reviewed 2025 low-price window; pre-delist sub-1-yuan trading period",
    ),
    StatusWindow(
        symbol="000851.SZ",
        level="1m",
        start_date="2025-09-19",
        end_date="2025-09-26",
        status="pre_delist_below_1yuan",
        note="reviewed 2025 low-price window; pre-delist sub-1-yuan trading period",
    ),
)


def schema_sql() -> str:
    return """
CREATE TABLE IF NOT EXISTS market.bar_status_flags
(
    symbol LowCardinality(String),
    level LowCardinality(String),
    trade_date Date,
    status LowCardinality(String),
    note String,
    source LowCardinality(String) DEFAULT '',
    updated_at DateTime64(3, 'Asia/Shanghai') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY (level, toYYYYMM(trade_date))
ORDER BY (level, symbol, trade_date, status)
SETTINGS index_granularity = 8192
""".strip()


def _expand_dates(start_date: str, end_date: str) -> Iterable[str]:
    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    while current <= end:
        yield current.isoformat()
        current += timedelta(days=1)


def expand_window_rows(
    windows: Sequence[StatusWindow],
    *,
    source: str = DEFAULT_SOURCE,
    allowed_dates: Optional[Set[Tuple[str, str]]] = None,
) -> List[Tuple[str, str, str, str, str, str]]:
    rows: List[Tuple[str, str, str, str, str, str]] = []
    for window in windows:
        for trade_date in _expand_dates(window.start_date, window.end_date):
            if allowed_dates is not None and (window.symbol, trade_date) not in allowed_dates:
                continue
            rows.append((window.symbol, window.level, trade_date, window.status, window.note, source))
    return rows


def delete_existing_flags_query(*, level: str, source: str) -> str:
    return f"""
ALTER TABLE market.bar_status_flags DELETE
WHERE level = {_sql_literal(level)}
  AND source = {_sql_literal(source)}
SETTINGS mutations_sync = 2
""".strip()


def insert_flags_query(rows: Sequence[Tuple[str, str, str, str, str, str]]) -> str:
    if not rows:
        raise ValueError("rows must not be empty")
    values = ",\n".join(
        "("
        + ", ".join(
            [
                _sql_literal(symbol),
                _sql_literal(level),
                f"toDate({_sql_literal(trade_date)})",
                _sql_literal(status),
                _sql_literal(note),
                _sql_literal(source),
            ]
        )
        + ")"
        for symbol, level, trade_date, status, note, source in rows
    )
    return f"""
INSERT INTO market.bar_status_flags
    (symbol, level, trade_date, status, note, source)
VALUES
{values}
""".strip()


def apply_schema(*, env_file: Path, compose_file: Path) -> None:
    run_clickhouse_query(schema_sql(), env_file=env_file, compose_file=compose_file)


def existing_trade_dates(
    windows: Sequence[StatusWindow],
    *,
    env_file: Path,
    compose_file: Path,
) -> Set[Tuple[str, str]]:
    if not windows:
        return set()
    clauses = []
    for window in windows:
        clauses.append(
            "("
            + " AND ".join(
                [
                    f"symbol = {_sql_literal(window.symbol)}",
                    f"level = {_sql_literal(window.level)}",
                    f"trade_date >= toDate({_sql_literal(window.start_date)})",
                    f"trade_date <= toDate({_sql_literal(window.end_date)})",
                ]
            )
            + ")"
        )
    query = f"""
SELECT symbol, toString(trade_date) AS trade_date_text
FROM market.bars
WHERE {' OR '.join(clauses)}
GROUP BY symbol, trade_date
ORDER BY symbol, trade_date
FORMAT TSVWithNames
""".strip()
    output = run_clickhouse_query(query, env_file=env_file, compose_file=compose_file)
    rows = [line.split("\t") for line in output.splitlines()[1:] if line.strip()]
    return {(symbol, trade_date) for symbol, trade_date in rows}


def summarize_rows(rows: Sequence[Tuple[str, str, str, str, str, str]]) -> dict:
    status_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}
    for symbol, _level, _trade_date, status, _note, _source in rows:
        status_counts[status] = status_counts.get(status, 0) + 1
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    return {
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "symbol_counts": dict(sorted(symbol_counts.items())),
    }


def apply_flags(
    *,
    windows: Sequence[StatusWindow],
    env_file: Path,
    compose_file: Path,
    source: str,
    execute: bool,
) -> dict:
    allowed_dates = existing_trade_dates(windows, env_file=env_file, compose_file=compose_file)
    rows = expand_window_rows(windows, source=source, allowed_dates=allowed_dates)
    summary = summarize_rows(rows)
    summary["applied"] = False
    if not execute:
        return summary

    apply_schema(env_file=env_file, compose_file=compose_file)
    run_clickhouse_query(delete_existing_flags_query(level="1m", source=source), env_file=env_file, compose_file=compose_file)
    run_clickhouse_query(insert_flags_query(rows), env_file=env_file, compose_file=compose_file)
    inserted_count = run_clickhouse_query(
        (
            "SELECT count() FROM market.bar_status_flags "
            f"WHERE level = '1m' AND source = {_sql_literal(source)} FORMAT TSV"
        ),
        env_file=env_file,
        compose_file=compose_file,
    ).strip()
    summary["applied"] = True
    summary["inserted_rows"] = int(inserted_count)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag reviewed 2025 delisting-related 1m bar windows")
    parser.add_argument("--execute", action="store_true", help="Apply flags to ClickHouse")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source label for inserted flags")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = apply_flags(
        windows=DEFAULT_WINDOWS,
        env_file=Path(args.env_file).expanduser(),
        compose_file=Path(args.compose_file).expanduser(),
        source=args.source,
        execute=args.execute,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
