from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.import_stock_bar_1m_raw_to_clickhouse import (  # noqa: E402
    DEFAULT_COMPOSE_FILE,
    DEFAULT_ENV_FILE,
    DEFAULT_MARKET_DATA_ROOT,
    _compose_clickhouse_command,
    _sql_literal,
    parse_years,
    run_clickhouse_query,
)


DEFAULT_PARQUET_ROOT = DEFAULT_MARKET_DATA_ROOT / "parquet"
DEFAULT_STATUS_DB = DEFAULT_MARKET_DATA_ROOT / "manifests" / "bars_parquet_export_status.sqlite3"
EXPORT_COLUMNS = [
    "symbol",
    "level",
    "trade_time",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "updated_at",
]


def month_bounds(year_month: str) -> Tuple[str, str]:
    year = int(year_month[:4])
    month = int(year_month[4:6])
    start = f"{year:04d}-{month:02d}-01 00:00:00"
    if month == 12:
        end = f"{year + 1:04d}-01-01 00:00:00"
    else:
        end = f"{year:04d}-{month + 1:02d}-01 00:00:00"
    return start, end


def partition_output_path(parquet_root: Path, *, level: str, year_month: str) -> Path:
    return parquet_root / "bars" / f"level={level}" / f"year={year_month[:4]}" / f"month={year_month[4:6]}" / "part.parquet"


def export_query(*, level: str, year_month: str) -> str:
    start, end = month_bounds(year_month)
    columns = ", ".join(EXPORT_COLUMNS)
    return (
        f"SELECT {columns} "
        "FROM market.bars "
        f"WHERE level = {_sql_literal(level)} "
        f"AND trade_time >= toDateTime64({_sql_literal(start)}, 3, 'Asia/Shanghai') "
        f"AND trade_time < toDateTime64({_sql_literal(end)}, 3, 'Asia/Shanghai') "
        "FORMAT Parquet"
    )


def ensure_status_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bars_parquet_exports (
            level TEXT NOT NULL,
            year_month TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            bytes_written INTEGER NOT NULL,
            output_path TEXT NOT NULL,
            seconds REAL NOT NULL,
            status TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '',
            exported_at TEXT NOT NULL,
            PRIMARY KEY (level, year_month)
        )
        """
    )
    conn.commit()


def is_completed(conn: sqlite3.Connection, *, level: str, year_month: str, expected_rows: int) -> bool:
    row = conn.execute(
        """
        SELECT status, row_count, output_path FROM bars_parquet_exports
        WHERE level = ? AND year_month = ?
        """,
        (level, year_month),
    ).fetchone()
    if not row:
        return False
    status, row_count, output_path = row
    return bool(status == "completed" and int(row_count) == expected_rows and Path(output_path).exists())


def mark_completed(
    conn: sqlite3.Connection,
    *,
    level: str,
    year_month: str,
    row_count: int,
    bytes_written: int,
    output_path: str,
    seconds: float,
) -> None:
    conn.execute(
        """
        INSERT INTO bars_parquet_exports (
            level, year_month, row_count, bytes_written, output_path, seconds, status, error, exported_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'completed', '', ?)
        ON CONFLICT(level, year_month) DO UPDATE SET
            row_count = excluded.row_count,
            bytes_written = excluded.bytes_written,
            output_path = excluded.output_path,
            seconds = excluded.seconds,
            status = excluded.status,
            error = excluded.error,
            exported_at = excluded.exported_at
        """,
        (
            level,
            year_month,
            row_count,
            bytes_written,
            output_path,
            seconds,
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
        ),
    )
    conn.commit()


def mark_failed(conn: sqlite3.Connection, *, level: str, year_month: str, error: str) -> None:
    conn.execute(
        """
        INSERT INTO bars_parquet_exports (
            level, year_month, row_count, bytes_written, output_path, seconds, status, error, exported_at
        ) VALUES (?, ?, 0, 0, '', 0, 'failed', ?, ?)
        ON CONFLICT(level, year_month) DO UPDATE SET
            status = excluded.status,
            error = excluded.error,
            exported_at = excluded.exported_at
        """,
        (level, year_month, error[:1000], datetime.utcnow().isoformat(timespec="seconds") + "Z"),
    )
    conn.commit()


def list_partitions(
    *,
    level: str,
    years: Optional[Set[str]],
    env_file: Path,
    compose_file: Path,
) -> List[Dict[str, int | str]]:
    filters = [f"level = {_sql_literal(level)}"]
    if years:
        start_year = min(int(year) for year in years)
        end_year = max(int(year) for year in years) + 1
        filters.append(
            "trade_time >= "
            f"toDateTime64({_sql_literal(f'{start_year:04d}-01-01 00:00:00')}, 3, 'Asia/Shanghai')"
        )
        filters.append(
            "trade_time < "
            f"toDateTime64({_sql_literal(f'{end_year:04d}-01-01 00:00:00')}, 3, 'Asia/Shanghai')"
        )
    query = (
        "SELECT toYYYYMM(trade_time) AS year_month, count() AS rows "
        "FROM market.bars "
        f"WHERE {' AND '.join(filters)} "
        "GROUP BY year_month "
        "ORDER BY year_month FORMAT TSV"
    )
    output = run_clickhouse_query(query, env_file=env_file, compose_file=compose_file)
    partitions: List[Dict[str, int | str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        year_month, row_count = line.split("\t", 1)
        if years and year_month[:4] not in years:
            continue
        partitions.append({"year_month": year_month, "row_count": int(row_count)})
    return partitions


def export_partition(
    *,
    parquet_root: Path,
    level: str,
    year_month: str,
    row_count: int,
    env_file: Path,
    compose_file: Path,
    overwrite: bool = False,
) -> Dict[str, int | float | str]:
    output_path = partition_output_path(parquet_root, level=level, year_month=year_month)
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return {
            "level": level,
            "year_month": year_month,
            "row_count": row_count,
            "bytes_written": output_path.stat().st_size,
            "seconds": 0.0,
            "output_path": str(output_path),
            "skipped_existing": 1,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f".parquet.tmp.{os.getpid()}.{time.time_ns()}")
    if tmp_path.exists():
        tmp_path.unlink()

    started_at = time.monotonic()
    command = _compose_clickhouse_command(env_file, compose_file, export_query(level=level, year_month=year_month))
    with tmp_path.open("wb") as output_file:
        result = subprocess.run(command, stdout=output_file, stderr=subprocess.PIPE)
    elapsed = max(time.monotonic() - started_at, 0.001)
    if result.returncode != 0:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        detail = result.stderr.decode("utf-8", errors="replace").strip() or f"clickhouse-client exited with {result.returncode}"
        raise RuntimeError(detail)
    bytes_written = tmp_path.stat().st_size
    if row_count > 0 and bytes_written == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("ClickHouse wrote an empty parquet file for a non-empty partition")
    os.replace(tmp_path, output_path)
    return {
        "level": level,
        "year_month": year_month,
        "row_count": row_count,
        "bytes_written": bytes_written,
        "seconds": elapsed,
        "output_path": str(output_path),
        "skipped_existing": 0,
    }


def export_partitions(
    *,
    parquet_root: Path,
    status_db: Path,
    level: str,
    years: Optional[Set[str]],
    env_file: Path,
    compose_file: Path,
    workers: int,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, int]:
    partitions = list_partitions(level=level, years=years, env_file=env_file, compose_file=compose_file)
    status_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(status_db, timeout=60)
    try:
        ensure_status_schema(conn)
        pending: List[Dict[str, int | str]] = []
        skipped = 0
        for partition in partitions:
            year_month = str(partition["year_month"])
            row_count = int(partition["row_count"])
            if not overwrite and is_completed(conn, level=level, year_month=year_month, expected_rows=row_count):
                skipped += 1
                continue
            pending.append(partition)
        print(f"partitions={len(partitions)} pending={len(pending)} skipped={skipped}", flush=True)
        if dry_run:
            for partition in pending:
                out = partition_output_path(parquet_root, level=level, year_month=str(partition["year_month"]))
                print(f"[dry-run] {partition['year_month']} rows={partition['row_count']} -> {out}", flush=True)
            return {"partitions": len(partitions), "exported": 0, "skipped": skipped, "rows": 0, "bytes": 0}

        stats = {"partitions": len(partitions), "exported": 0, "skipped": skipped, "rows": 0, "bytes": 0}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_map = {
                executor.submit(
                    export_partition,
                    parquet_root=parquet_root,
                    level=level,
                    year_month=str(partition["year_month"]),
                    row_count=int(partition["row_count"]),
                    env_file=env_file,
                    compose_file=compose_file,
                    overwrite=overwrite,
                ): partition
                for partition in pending
            }
            for future in as_completed(future_map):
                partition = future_map[future]
                year_month = str(partition["year_month"])
                try:
                    result = future.result()
                    mark_completed(
                        conn,
                        level=level,
                        year_month=year_month,
                        row_count=int(result["row_count"]),
                        bytes_written=int(result["bytes_written"]),
                        output_path=str(result["output_path"]),
                        seconds=float(result["seconds"]),
                    )
                    stats["exported"] += 1
                    stats["rows"] += int(result["row_count"])
                    stats["bytes"] += int(result["bytes_written"])
                    rate = int(int(result["row_count"]) / max(float(result["seconds"]), 0.001))
                    print(
                        f"exported {year_month} rows={result['row_count']} "
                        f"bytes={result['bytes_written']} seconds={float(result['seconds']):.1f} rows_per_sec={rate}",
                        flush=True,
                    )
                except Exception as exc:
                    mark_failed(conn, level=level, year_month=year_month, error=str(exc))
                    raise
        return stats
    finally:
        conn.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export market.bars partitions to parquet files")
    parser.add_argument("--parquet-root", default=str(DEFAULT_PARQUET_ROOT), help="Root directory for parquet output")
    parser.add_argument("--status-db", default=str(DEFAULT_STATUS_DB), help="SQLite export status DB")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    parser.add_argument("--level", default="1m", help="Bar level to export")
    parser.add_argument("--years", default=None, help="Year filter, e.g. 2024 or 2020-2024")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent monthly exports")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing completed parquet files")
    parser.add_argument("--dry-run", action="store_true", help="Print planned exports without writing files")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stats = export_partitions(
        parquet_root=Path(args.parquet_root).expanduser(),
        status_db=Path(args.status_db).expanduser(),
        level=args.level,
        years=parse_years(args.years),
        env_file=Path(args.env_file).expanduser(),
        compose_file=Path(args.compose_file).expanduser(),
        workers=args.workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"summary: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
