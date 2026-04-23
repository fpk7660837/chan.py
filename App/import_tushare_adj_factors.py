from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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


DEFAULT_STATUS_DB = DEFAULT_MARKET_DATA_ROOT / "manifests" / "tushare_adj_factor_import_status.sqlite3"
DEFAULT_API_URL = "http://api.tushare.pro"
INSERT_COLUMNS = ["symbol", "trade_date", "adj_factor", "source"]


class RequestPacer:
    def __init__(self, max_requests_per_minute: int) -> None:
        self._interval = 0.0 if max_requests_per_minute <= 0 else 60.0 / float(max_requests_per_minute)
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + self._interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def normalize_trade_date(value: str) -> str:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text
    raise ValueError(f"unsupported trade_date format: {value!r}")


def parse_tushare_response(payload: Dict[str, object]) -> List[Dict[str, object]]:
    code = int(payload.get("code", -1))
    if code != 0:
        raise RuntimeError(str(payload.get("msg") or f"Tushare returned code {code}"))

    data = payload.get("data") or {}
    if not isinstance(data, dict):
        raise RuntimeError("Tushare response data is not an object")
    fields = data.get("fields") or []
    items = data.get("items") or []
    if not isinstance(fields, list) or not isinstance(items, list):
        raise RuntimeError("Tushare response data.fields/items are invalid")

    rows: List[Dict[str, object]] = []
    for item in items:
        if not isinstance(item, list):
            raise RuntimeError("Tushare response item is not a list")
        rows.append(dict(zip((str(field) for field in fields), item)))
    return rows


def normalize_adj_factor_rows(
    rows: Iterable[Dict[str, object]],
    *,
    source: str = "tushare",
) -> List[Tuple[str, str, float, str]]:
    normalized: List[Tuple[str, str, float, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in rows:
        symbol = str(row["ts_code"]).strip().upper()
        trade_date = normalize_trade_date(str(row["trade_date"]))
        adj_factor = float(row["adj_factor"])
        key = (symbol, trade_date)
        if key in seen:
            continue
        seen.add(key)
        normalized.append((symbol, trade_date, adj_factor, source))
    return normalized


def ensure_status_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tushare_adj_factor_imports (
            trade_date TEXT NOT NULL PRIMARY KEY,
            row_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '',
            imported_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def is_completed(conn: sqlite3.Connection, trade_date: str) -> bool:
    row = conn.execute(
        """
        SELECT status FROM tushare_adj_factor_imports
        WHERE trade_date = ?
        """,
        (trade_date,),
    ).fetchone()
    return bool(row and row[0] == "completed")


def mark_completed(conn: sqlite3.Connection, *, trade_date: str, row_count: int) -> None:
    conn.execute(
        """
        INSERT INTO tushare_adj_factor_imports (
            trade_date, row_count, status, error, imported_at
        ) VALUES (?, ?, 'completed', '', ?)
        ON CONFLICT(trade_date) DO UPDATE SET
            row_count = excluded.row_count,
            status = excluded.status,
            error = excluded.error,
            imported_at = excluded.imported_at
        """,
        (trade_date, row_count, datetime.utcnow().isoformat(timespec="seconds") + "Z"),
    )
    conn.commit()


def mark_failed(conn: sqlite3.Connection, *, trade_date: str, error: str) -> None:
    conn.execute(
        """
        INSERT INTO tushare_adj_factor_imports (
            trade_date, row_count, status, error, imported_at
        ) VALUES (?, 0, 'failed', ?, ?)
        ON CONFLICT(trade_date) DO UPDATE SET
            status = excluded.status,
            error = excluded.error,
            imported_at = excluded.imported_at
        """,
        (trade_date, error[:1000], datetime.utcnow().isoformat(timespec="seconds") + "Z"),
    )
    conn.commit()


def clickhouse_insert_command(env_file: Path, compose_file: Path) -> List[str]:
    input_schema = "symbol String, trade_date Date, adj_factor Float64, source String"
    query = (
        f"INSERT INTO market.adj_factors ({', '.join(INSERT_COLUMNS)}) "
        "SELECT i.symbol, i.trade_date, i.adj_factor, i.source "
        f"FROM input({_sql_literal(input_schema)}) AS i "
        "WHERE (i.symbol, i.trade_date) NOT IN ("
        "SELECT symbol, trade_date FROM market.adj_factors"
        ") FORMAT CSV"
    )
    return _compose_clickhouse_command(env_file, compose_file, query)


def insert_adj_factor_rows(
    rows: Sequence[Tuple[str, str, float, str]],
    *,
    env_file: Path,
    compose_file: Path,
) -> int:
    if not rows:
        return 0

    process = subprocess.Popen(
        clickhouse_insert_command(env_file, compose_file),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    writer = csv.writer(process.stdin, lineterminator="\n")
    write_error: Optional[BaseException] = None
    try:
        for row in rows:
            writer.writerow(row)
    except BaseException as exc:
        write_error = exc
    finally:
        try:
            process.stdin.close()
        except BaseException as exc:
            if write_error is None:
                write_error = exc
        process.stdin = None

    stdout, stderr = process.communicate()
    if write_error is not None or process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"clickhouse-client exited with {process.returncode}"
        if write_error is not None:
            detail = f"{write_error}; {detail}"
        raise RuntimeError(detail)
    return len(rows)


def load_tushare_token(token_file: Optional[Path] = None) -> str:
    for key in ("TUSHARE_TOKEN", "TUSHARE_PRO_TOKEN", "TS_TOKEN"):
        token = os.environ.get(key)
        if token and token.strip():
            return token.strip()
    if token_file is not None:
        token = token_file.expanduser().read_text(encoding="utf-8").strip()
        if token:
            return token
    raise RuntimeError("Tushare token not found; set TUSHARE_TOKEN or pass --token-file")


def request_tushare_json(
    *,
    api_url: str,
    token: str,
    trade_date: str,
    timeout: float,
) -> Dict[str, object]:
    payload = {
        "api_name": "adj_factor",
        "token": token,
        "params": {"trade_date": trade_date},
        "fields": "ts_code,trade_date,adj_factor",
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "chan.py-tushare-adj-factor/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Tushare response is not a JSON object")
    return parsed


def fetch_adj_factor(
    *,
    token: str,
    trade_date: str,
    api_url: str = DEFAULT_API_URL,
    timeout: float = 30.0,
    retries: int = 3,
    retry_sleep: float = 2.0,
) -> List[Dict[str, object]]:
    last_error: Optional[BaseException] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            payload = request_tushare_json(api_url=api_url, token=token, trade_date=trade_date, timeout=timeout)
            return parse_tushare_response(payload)
        except (TimeoutError, socket.timeout, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= max(1, retries):
                break
            time.sleep(max(0.0, retry_sleep) * attempt)
    raise RuntimeError(str(last_error) if last_error else "Tushare request failed")


def list_trade_dates_from_bars(
    *,
    level: str,
    years: Optional[Set[str]],
    start_date: Optional[str],
    end_date: Optional[str],
    env_file: Path,
    compose_file: Path,
) -> List[str]:
    filters = [f"level = {_sql_literal(level)}"]
    if years:
        year_list = ", ".join(str(int(year)) for year in sorted(years))
        filters.append(f"toYear(trade_date) IN ({year_list})")
    if start_date:
        filters.append(f"trade_date >= toDate({_sql_literal(normalize_trade_date(start_date))})")
    if end_date:
        filters.append(f"trade_date <= toDate({_sql_literal(normalize_trade_date(end_date))})")

    query = (
        "SELECT replaceAll(toString(trade_date), '-', '') AS d "
        "FROM market.bars "
        f"WHERE {' AND '.join(filters)} "
        "GROUP BY trade_date "
        "ORDER BY trade_date FORMAT TSV"
    )
    output = run_clickhouse_query(query, env_file=env_file, compose_file=compose_file)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _flush_batch(
    *,
    conn: sqlite3.Connection,
    batch_rows: Sequence[Tuple[str, str, float, str]],
    batch_counts: Sequence[Tuple[str, int]],
    env_file: Path,
    compose_file: Path,
) -> int:
    inserted = insert_adj_factor_rows(batch_rows, env_file=env_file, compose_file=compose_file)
    for trade_date, row_count in batch_counts:
        mark_completed(conn, trade_date=trade_date, row_count=row_count)
    return inserted


def fetch_normalized_adj_factor(
    *,
    token: str,
    trade_date: str,
    api_url: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
    source: str,
    pacer: Optional[RequestPacer] = None,
) -> Tuple[str, List[Tuple[str, str, float, str]]]:
    if pacer is not None:
        pacer.wait()
    rows = fetch_adj_factor(
        token=token,
        trade_date=trade_date,
        api_url=api_url,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    return trade_date, normalize_adj_factor_rows(rows, source=source)


def _process_fetched_date(
    *,
    conn: sqlite3.Connection,
    stats: Dict[str, int],
    trade_date: str,
    normalized: Sequence[Tuple[str, str, float, str]],
    completed_count: int,
    total_count: int,
    batch_rows: List[Tuple[str, str, float, str]],
    batch_counts: List[Tuple[str, int]],
    batch_dates: int,
    env_file: Path,
    compose_file: Path,
) -> Tuple[List[Tuple[str, str, float, str]], List[Tuple[str, int]]]:
    batch_rows.extend(normalized)
    batch_counts.append((trade_date, len(normalized)))
    print(f"fetched_date {trade_date} rows={len(normalized)} progress={completed_count}/{total_count}", flush=True)

    if len(batch_counts) >= max(1, batch_dates):
        inserted = _flush_batch(
            conn=conn,
            batch_rows=batch_rows,
            batch_counts=batch_counts,
            env_file=env_file,
            compose_file=compose_file,
        )
        stats["dates_imported"] += len(batch_counts)
        stats["rows_imported"] += sum(row_count for _, row_count in batch_counts)
        print(f"inserted_batch dates={len(batch_counts)} rows={inserted}", flush=True)
        return [], []
    return batch_rows, batch_counts


def import_adj_factors(
    *,
    status_db: Path,
    env_file: Path,
    compose_file: Path,
    token: str,
    api_url: str = DEFAULT_API_URL,
    level: str = "1m",
    years: Optional[Set[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit_dates: Optional[int] = None,
    batch_dates: int = 20,
    sleep_seconds: float = 0.25,
    timeout: float = 30.0,
    retries: int = 3,
    retry_sleep: float = 2.0,
    source: str = "tushare",
    dry_run: bool = False,
    force: bool = False,
    workers: int = 1,
    max_requests_per_minute: int = 180,
) -> Dict[str, int]:
    trade_dates = list_trade_dates_from_bars(
        level=level,
        years=years,
        start_date=start_date,
        end_date=end_date,
        env_file=env_file,
        compose_file=compose_file,
    )
    if limit_dates is not None:
        trade_dates = trade_dates[:limit_dates]

    status_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(status_db, timeout=60)
    try:
        ensure_status_schema(conn)
        pending: List[str] = []
        skipped = 0
        for trade_date in trade_dates:
            if not force and is_completed(conn, trade_date):
                skipped += 1
                continue
            pending.append(trade_date)

        print(f"trade_dates={len(trade_dates)} pending={len(pending)} skipped={skipped}", flush=True)
        if dry_run:
            for trade_date in pending:
                print(f"[dry-run] would fetch adj_factor trade_date={trade_date}", flush=True)
            return {"dates": len(trade_dates), "dates_imported": 0, "dates_skipped": skipped, "rows_imported": 0}

        stats = {"dates": len(trade_dates), "dates_imported": 0, "dates_skipped": skipped, "rows_imported": 0}
        batch_rows: List[Tuple[str, str, float, str]] = []
        batch_counts: List[Tuple[str, int]] = []
        worker_count = max(1, workers)
        pacer = RequestPacer(max_requests_per_minute)

        if worker_count == 1:
            for index, trade_date in enumerate(pending, start=1):
                try:
                    fetched_date, normalized = fetch_normalized_adj_factor(
                        token=token,
                        trade_date=trade_date,
                        api_url=api_url,
                        timeout=timeout,
                        retries=retries,
                        retry_sleep=retry_sleep,
                        source=source,
                        pacer=pacer,
                    )
                    batch_rows, batch_counts = _process_fetched_date(
                        conn=conn,
                        stats=stats,
                        trade_date=fetched_date,
                        normalized=normalized,
                        completed_count=index,
                        total_count=len(pending),
                        batch_rows=batch_rows,
                        batch_counts=batch_counts,
                        batch_dates=batch_dates,
                        env_file=env_file,
                        compose_file=compose_file,
                    )
                except Exception as exc:
                    if batch_counts:
                        inserted = _flush_batch(
                            conn=conn,
                            batch_rows=batch_rows,
                            batch_counts=batch_counts,
                            env_file=env_file,
                            compose_file=compose_file,
                        )
                        stats["dates_imported"] += len(batch_counts)
                        stats["rows_imported"] += sum(row_count for _, row_count in batch_counts)
                        print(f"inserted_batch dates={len(batch_counts)} rows={inserted}", flush=True)
                    mark_failed(conn, trade_date=trade_date, error=str(exc))
                    raise

                if sleep_seconds > 0 and index < len(pending):
                    time.sleep(sleep_seconds)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            future_map: Dict[Future[Tuple[str, List[Tuple[str, str, float, str]]]], str] = {}
            try:
                for trade_date in pending:
                    future = executor.submit(
                        fetch_normalized_adj_factor,
                        token=token,
                        trade_date=trade_date,
                        api_url=api_url,
                        timeout=timeout,
                        retries=retries,
                        retry_sleep=retry_sleep,
                        source=source,
                        pacer=pacer,
                    )
                    future_map[future] = trade_date

                for completed, future in enumerate(as_completed(future_map), start=1):
                    trade_date = future_map[future]
                    try:
                        fetched_date, normalized = future.result()
                        batch_rows, batch_counts = _process_fetched_date(
                            conn=conn,
                            stats=stats,
                            trade_date=fetched_date,
                            normalized=normalized,
                            completed_count=completed,
                            total_count=len(pending),
                            batch_rows=batch_rows,
                            batch_counts=batch_counts,
                            batch_dates=batch_dates,
                            env_file=env_file,
                            compose_file=compose_file,
                        )
                    except Exception as exc:
                        for pending_future in future_map:
                            pending_future.cancel()
                        if batch_counts:
                            inserted = _flush_batch(
                                conn=conn,
                                batch_rows=batch_rows,
                                batch_counts=batch_counts,
                                env_file=env_file,
                                compose_file=compose_file,
                            )
                            stats["dates_imported"] += len(batch_counts)
                            stats["rows_imported"] += sum(row_count for _, row_count in batch_counts)
                            print(f"inserted_batch dates={len(batch_counts)} rows={inserted}", flush=True)
                        mark_failed(conn, trade_date=trade_date, error=str(exc))
                        raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        if batch_counts:
            inserted = _flush_batch(
                conn=conn,
                batch_rows=batch_rows,
                batch_counts=batch_counts,
                env_file=env_file,
                compose_file=compose_file,
            )
            stats["dates_imported"] += len(batch_counts)
            stats["rows_imported"] += sum(row_count for _, row_count in batch_counts)
            print(f"inserted_batch dates={len(batch_counts)} rows={inserted}", flush=True)

        return stats
    finally:
        conn.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Tushare adj_factor data into ClickHouse market.adj_factors")
    parser.add_argument("--status-db", default=str(DEFAULT_STATUS_DB), help="SQLite status DB for resumable imports")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    parser.add_argument("--token-file", default=None, help="File containing a Tushare token; env vars are checked first")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Tushare JSON API URL")
    parser.add_argument("--level", default="1m", help="Bar level used to discover local trade dates")
    parser.add_argument("--years", default=None, help="Year filter, e.g. 2024 or 2020-2024")
    parser.add_argument("--start-date", default=None, help="Inclusive start date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Inclusive end date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--limit-dates", type=int, default=None, help="Limit number of trade dates")
    parser.add_argument("--batch-dates", type=int, default=20, help="Insert after fetching this many trade dates")
    parser.add_argument("--sleep-seconds", type=float, default=0.25, help="Sleep between Tushare requests")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=3, help="Transport retry count")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Base sleep seconds between transport retries")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Tushare requests; keep low to avoid rate limits")
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=180,
        help="Global Tushare request start limit; 0 disables pacing",
    )
    parser.add_argument("--source", default="tushare", help="Value written to market.adj_factors.source")
    parser.add_argument("--force", action="store_true", help="Ignore local completed status and fetch again")
    parser.add_argument("--dry-run", action="store_true", help="Print work without fetching or inserting")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    token = load_tushare_token(Path(args.token_file).expanduser() if args.token_file else None)
    stats = import_adj_factors(
        status_db=Path(args.status_db).expanduser(),
        env_file=Path(args.env_file).expanduser(),
        compose_file=Path(args.compose_file).expanduser(),
        token=token,
        api_url=args.api_url,
        level=args.level,
        years=parse_years(args.years),
        start_date=args.start_date,
        end_date=args.end_date,
        limit_dates=args.limit_dates,
        batch_dates=args.batch_dates,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        source=args.source,
        dry_run=args.dry_run,
        force=args.force,
        workers=args.workers,
        max_requests_per_minute=args.max_requests_per_minute,
    )
    print(f"summary: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
