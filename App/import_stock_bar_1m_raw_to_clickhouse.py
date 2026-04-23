from __future__ import annotations

import argparse
import csv
import sqlite3
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App import inspect_stock_bar_archives as archive_inspector


DEFAULT_SOURCE_DIR = archive_inspector.DEFAULT_SOURCE_DIR
DEFAULT_MARKET_DATA_ROOT = archive_inspector.DEFAULT_MARKET_DATA_ROOT
DEFAULT_STATUS_DB = DEFAULT_MARKET_DATA_ROOT / "manifests" / "stock_bar_1m_raw_import_status.sqlite3"
DEFAULT_ENV_FILE = ROOT / "infra" / "clickhouse" / ".env"
DEFAULT_COMPOSE_FILE = ROOT / "infra" / "clickhouse" / "docker-compose.yml"
PROJECT_NAME = "chan-clickhouse"

INSERT_COLUMNS = ["symbol", "level", "trade_time", "open", "high", "low", "close", "volume", "amount", "source"]


def select_raw_1m_archives(manifest_rows: Iterable[Dict[str, str]], years: Optional[Set[str]] = None) -> List[Dict[str, str]]:
    selected = [
        row
        for row in manifest_rows
        if row.get("status") == "ok"
        and row.get("level") == "1m"
        and row.get("adjust") == "raw"
        and (years is None or row.get("year_start") in years)
    ]
    return sorted(selected, key=lambda row: (row.get("year_start", ""), row.get("archive_path", "")))


def parse_years(value: Optional[str]) -> Optional[Set[str]]:
    if not value:
        return None
    years: Set[str] = set()
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            start, end = int(start_text), int(end_text)
            years.update(str(year) for year in range(start, end + 1))
        else:
            years.add(str(int(text)))
    return years


def zip_csv_members(archive_path: Path) -> List[str]:
    with zipfile.ZipFile(archive_path) as archive:
        return sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))


def member_identity(archive_path: str, member: str) -> Dict[str, str]:
    archive_name = Path(archive_path).name
    year_start, _ = archive_inspector._infer_years(archive_name)
    level = archive_inspector._infer_level("", archive_name)
    code = Path(member).stem.split("_", 1)[0]
    return {"symbol": archive_inspector.normalize_symbol(code), "level": level, "year": year_start}


def _bar_from_csv_row(row: Dict[str, str], source: str) -> Dict[str, str]:
    return {
        "symbol": archive_inspector.normalize_symbol(str(row["代码"]).strip()),
        "level": "1m",
        "trade_time": str(row["时间"]).strip(),
        "open": str(row["开盘价"]).strip(),
        "high": str(row["最高价"]).strip(),
        "low": str(row["最低价"]).strip(),
        "close": str(row["收盘价"]).strip(),
        "volume": str(row["成交量"]).strip(),
        "amount": str(row["成交额"]).strip(),
        "source": source,
    }


def iter_member_bars(archive_path: Path, member: str, source: str = "stock_bar_zip") -> Iterable[Dict[str, str]]:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as raw_fp:
            text_fp = TextIOWrapper(raw_fp, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_fp)
            for row in reader:
                if not row:
                    continue
                yield _bar_from_csv_row(row, source)


def iter_archive_bars(
    archive_path: Path,
    *,
    source: str = "stock_bar_zip",
    member_limit: Optional[int] = None,
) -> Iterable[Dict[str, str]]:
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
        if member_limit is not None:
            members = members[:member_limit]
        for member in members:
            with archive.open(member) as raw_fp:
                text_fp = TextIOWrapper(raw_fp, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text_fp)
                for row in reader:
                    if not row:
                        continue
                    yield _bar_from_csv_row(row, source)


def ensure_status_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_bar_1m_raw_imports (
            archive_path TEXT NOT NULL,
            member TEXT NOT NULL,
            year TEXT NOT NULL,
            level TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '',
            imported_at TEXT NOT NULL,
            PRIMARY KEY (archive_path, member)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_bar_1m_raw_archive_imports (
            archive_path TEXT NOT NULL PRIMARY KEY,
            year TEXT NOT NULL,
            level TEXT NOT NULL,
            member_count INTEGER NOT NULL DEFAULT 0,
            row_count INTEGER NOT NULL DEFAULT 0,
            seconds REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '',
            imported_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def is_completed(conn: sqlite3.Connection, archive_path: str, member: str) -> bool:
    row = conn.execute(
        """
        SELECT status FROM stock_bar_1m_raw_imports
        WHERE archive_path = ? AND member = ?
        """,
        (archive_path, member),
    ).fetchone()
    return bool(row and row[0] == "completed")


def mark_completed(
    conn: sqlite3.Connection,
    *,
    archive_path: str,
    member: str,
    row_count: int,
    year: str,
    level: str,
) -> None:
    conn.execute(
        """
        INSERT INTO stock_bar_1m_raw_imports (
            archive_path, member, year, level, row_count, status, error, imported_at
        ) VALUES (?, ?, ?, ?, ?, 'completed', '', ?)
        ON CONFLICT(archive_path, member) DO UPDATE SET
            year = excluded.year,
            level = excluded.level,
            row_count = excluded.row_count,
            status = excluded.status,
            error = excluded.error,
            imported_at = excluded.imported_at
        """,
        (archive_path, member, year, level, row_count, datetime.utcnow().isoformat(timespec="seconds") + "Z"),
    )
    conn.commit()


def mark_failed(conn: sqlite3.Connection, *, archive_path: str, member: str, year: str, level: str, error: str) -> None:
    conn.execute(
        """
        INSERT INTO stock_bar_1m_raw_imports (
            archive_path, member, year, level, row_count, status, error, imported_at
        ) VALUES (?, ?, ?, ?, 0, 'failed', ?, ?)
        ON CONFLICT(archive_path, member) DO UPDATE SET
            year = excluded.year,
            level = excluded.level,
            status = excluded.status,
            error = excluded.error,
            imported_at = excluded.imported_at
        """,
        (archive_path, member, year, level, error[:1000], datetime.utcnow().isoformat(timespec="seconds") + "Z"),
    )
    conn.commit()


def is_archive_completed(conn: sqlite3.Connection, archive_path: str) -> bool:
    row = conn.execute(
        """
        SELECT status FROM stock_bar_1m_raw_archive_imports
        WHERE archive_path = ?
        """,
        (archive_path,),
    ).fetchone()
    return bool(row and row[0] == "completed")


def mark_archive_completed(
    conn: sqlite3.Connection,
    *,
    archive_path: str,
    row_count: int,
    member_count: int,
    year: str,
    level: str,
    seconds: float,
) -> None:
    conn.execute(
        """
        INSERT INTO stock_bar_1m_raw_archive_imports (
            archive_path, year, level, member_count, row_count, seconds, status, error, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'completed', '', ?)
        ON CONFLICT(archive_path) DO UPDATE SET
            year = excluded.year,
            level = excluded.level,
            member_count = excluded.member_count,
            row_count = excluded.row_count,
            seconds = excluded.seconds,
            status = excluded.status,
            error = excluded.error,
            imported_at = excluded.imported_at
        """,
        (
            archive_path,
            year,
            level,
            member_count,
            row_count,
            seconds,
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
        ),
    )
    conn.commit()


def mark_archive_failed(conn: sqlite3.Connection, *, archive_path: str, year: str, level: str, error: str) -> None:
    conn.execute(
        """
        INSERT INTO stock_bar_1m_raw_archive_imports (
            archive_path, year, level, member_count, row_count, seconds, status, error, imported_at
        ) VALUES (?, ?, ?, 0, 0, 0, 'failed', ?, ?)
        ON CONFLICT(archive_path) DO UPDATE SET
            year = excluded.year,
            level = excluded.level,
            status = excluded.status,
            error = excluded.error,
            imported_at = excluded.imported_at
        """,
        (archive_path, year, level, error[:1000], datetime.utcnow().isoformat(timespec="seconds") + "Z"),
    )
    conn.commit()


def reset_status_tables(status_db: Path) -> None:
    status_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(status_db, timeout=60)
    try:
        ensure_status_schema(conn)
        conn.execute("DELETE FROM stock_bar_1m_raw_imports")
        conn.execute("DELETE FROM stock_bar_1m_raw_archive_imports")
        conn.commit()
    finally:
        conn.close()


def load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _sql_literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def _compose_clickhouse_command(env_file: Path, compose_file: Path, query: str) -> List[str]:
    env = load_env_file(env_file) if env_file.exists() else {}
    user = env.get("CLICKHOUSE_USER", "chan")
    password = env.get("CLICKHOUSE_PASSWORD", "")
    return [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
        "-p",
        PROJECT_NAME,
        "exec",
        "-T",
        "clickhouse",
        "clickhouse-client",
        "--user",
        user,
        "--password",
        password,
        "--query",
        query,
    ]


def _member_time_bounds(year: str) -> tuple[str, str]:
    start = f"{int(year):04d}-01-01 00:00:00"
    end = f"{int(year) + 1:04d}-01-01 00:00:00"
    return start, end


def clickhouse_insert_command(env_file: Path, compose_file: Path, *, symbol: str, level: str, year: str) -> List[str]:
    start, end = _member_time_bounds(year)
    input_schema = (
        "symbol String, "
        "level String, "
        "trade_time DateTime64(3, 'Asia/Shanghai'), "
        "open Float64, "
        "high Float64, "
        "low Float64, "
        "close Float64, "
        "volume Float64, "
        "amount Float64, "
        "source String"
    )
    query = (
        f"INSERT INTO market.bars ({', '.join(INSERT_COLUMNS)}) "
        "SELECT "
        "i.symbol, i.level, i.trade_time, i.open, i.high, i.low, i.close, i.volume, i.amount, i.source "
        f"FROM input({_sql_literal(input_schema)}) AS i "
        "WHERE (i.symbol, i.level, i.trade_time) NOT IN ("
        "SELECT symbol, level, trade_time FROM market.bars WHERE "
        f"symbol = {_sql_literal(symbol)} "
        f"AND level = {_sql_literal(level)} "
        f"AND trade_time >= toDateTime64({_sql_literal(start)}, 3, 'Asia/Shanghai') "
        f"AND trade_time < toDateTime64({_sql_literal(end)}, 3, 'Asia/Shanghai')"
        ") FORMAT CSV"
    )
    return _compose_clickhouse_command(env_file, compose_file, query)


def clickhouse_bulk_insert_command(env_file: Path, compose_file: Path) -> List[str]:
    query = f"INSERT INTO market.bars ({', '.join(INSERT_COLUMNS)}) FORMAT CSV"
    return _compose_clickhouse_command(env_file, compose_file, query)


def run_clickhouse_query(query: str, *, env_file: Path, compose_file: Path) -> str:
    result = subprocess.run(
        _compose_clickhouse_command(env_file, compose_file, query),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"clickhouse-client exited with {result.returncode}"
        raise RuntimeError(detail)
    return result.stdout.strip()


def truncate_market_bars(*, env_file: Path, compose_file: Path) -> None:
    run_clickhouse_query("TRUNCATE TABLE market.bars", env_file=env_file, compose_file=compose_file)


def import_member_to_clickhouse(
    archive_path: Path,
    member: str,
    *,
    env_file: Path,
    compose_file: Path,
    source: str,
) -> int:
    identity = member_identity(str(archive_path), member)
    process = subprocess.Popen(
        clickhouse_insert_command(
            env_file,
            compose_file,
            symbol=identity["symbol"],
            level=identity["level"],
            year=identity["year"],
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    writer = csv.writer(process.stdin, lineterminator="\n")
    row_count = 0
    try:
        for bar in iter_member_bars(archive_path, member, source=source):
            writer.writerow([bar[column] for column in INSERT_COLUMNS])
            row_count += 1
    finally:
        process.stdin.close()
        process.stdin = None

    stdout, stderr = process.communicate()
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"clickhouse-client exited with {process.returncode}"
        raise RuntimeError(detail)
    return row_count


def import_archive_to_clickhouse(
    archive_path: Path,
    *,
    env_file: Path,
    compose_file: Path,
    source: str,
    member_limit: Optional[int] = None,
    progress_members: int = 100,
) -> Dict[str, float]:
    members = zip_csv_members(archive_path)
    if member_limit is not None:
        members = members[:member_limit]

    process = subprocess.Popen(
        clickhouse_bulk_insert_command(env_file, compose_file),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    writer = csv.writer(process.stdin, lineterminator="\n")
    row_count = 0
    started_at = time.monotonic()
    write_error: Optional[BaseException] = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member_index, member in enumerate(members, start=1):
                with archive.open(member) as raw_fp:
                    text_fp = TextIOWrapper(raw_fp, encoding="utf-8-sig", newline="")
                    reader = csv.DictReader(text_fp)
                    for row in reader:
                        if not row:
                            continue
                        bar = _bar_from_csv_row(row, source)
                        writer.writerow([bar[column] for column in INSERT_COLUMNS])
                        row_count += 1
                if progress_members > 0 and member_index % progress_members == 0:
                    elapsed = max(time.monotonic() - started_at, 0.001)
                    print(
                        f"progress_archive {archive_path.name} members={member_index}/{len(members)} "
                        f"rows={row_count} rows_per_sec={row_count / elapsed:.0f}",
                        flush=True,
                    )
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
    elapsed = max(time.monotonic() - started_at, 0.001)
    if write_error is not None or process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"clickhouse-client exited with {process.returncode}"
        if write_error is not None:
            detail = f"{write_error}; {detail}"
        raise RuntimeError(detail)
    return {"members": float(len(members)), "rows": float(row_count), "seconds": elapsed}


def import_archives(
    source_dir: Path,
    status_db: Path,
    *,
    years: Optional[Set[str]] = None,
    archive_limit: Optional[int] = None,
    member_limit: Optional[int] = None,
    dry_run: bool = False,
    env_file: Path = DEFAULT_ENV_FILE,
    compose_file: Path = DEFAULT_COMPOSE_FILE,
    source: str = "stock_bar_zip",
    mode: str = "archive",
    truncate_bars_first: bool = False,
    progress_members: int = 100,
) -> Dict[str, int]:
    if mode not in {"archive", "member"}:
        raise ValueError(f"unsupported import mode: {mode}")
    if truncate_bars_first and not dry_run:
        truncate_market_bars(env_file=env_file, compose_file=compose_file)
        reset_status_tables(status_db)
        print("truncated market.bars and reset import status", flush=True)

    manifest_rows = archive_inspector.build_manifest(source_dir)
    archives = select_raw_1m_archives(manifest_rows, years=years)
    if archive_limit is not None:
        archives = archives[:archive_limit]

    status_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(status_db, timeout=60)
    ensure_status_schema(conn)
    stats = {
        "archives": len(archives),
        "archives_imported": 0,
        "archives_skipped": 0,
        "members_seen": 0,
        "members_imported": 0,
        "members_skipped": 0,
        "rows_imported": 0,
    }
    try:
        for archive_row in archives:
            archive_path = Path(archive_row["archive_path"])
            members = zip_csv_members(archive_path)
            if member_limit is not None:
                members = members[:member_limit]
            if mode == "archive":
                stats["members_seen"] += len(members)
                use_archive_status = member_limit is None
                if use_archive_status and is_archive_completed(conn, str(archive_path)):
                    stats["archives_skipped"] += 1
                    stats["members_skipped"] += len(members)
                    continue
                if dry_run:
                    print(f"[dry-run] would import archive {archive_path.name} members={len(members)}")
                    continue
                try:
                    result = import_archive_to_clickhouse(
                        archive_path,
                        env_file=env_file,
                        compose_file=compose_file,
                        source=source,
                        member_limit=member_limit,
                        progress_members=progress_members,
                    )
                    row_count = int(result["rows"])
                    member_count = int(result["members"])
                    elapsed = float(result["seconds"])
                    if use_archive_status:
                        mark_archive_completed(
                            conn,
                            archive_path=str(archive_path),
                            row_count=row_count,
                            member_count=member_count,
                            year=archive_row["year_start"],
                            level=archive_row["level"],
                            seconds=elapsed,
                        )
                    stats["archives_imported"] += 1
                    stats["members_imported"] += member_count
                    stats["rows_imported"] += row_count
                    print(
                        f"imported_archive {archive_path.name} members={member_count} rows={row_count} "
                        f"seconds={elapsed:.1f} rows_per_sec={row_count / max(elapsed, 0.001):.0f}",
                        flush=True,
                    )
                except Exception as exc:
                    if use_archive_status:
                        mark_archive_failed(
                            conn,
                            archive_path=str(archive_path),
                            year=archive_row["year_start"],
                            level=archive_row["level"],
                            error=str(exc),
                        )
                    raise
                continue

            for member in members:
                stats["members_seen"] += 1
                if is_completed(conn, str(archive_path), member):
                    stats["members_skipped"] += 1
                    continue
                if dry_run:
                    print(f"[dry-run] would import {archive_path.name}:{member}")
                    continue
                try:
                    row_count = import_member_to_clickhouse(
                        archive_path,
                        member,
                        env_file=env_file,
                        compose_file=compose_file,
                        source=source,
                    )
                    mark_completed(
                        conn,
                        archive_path=str(archive_path),
                        member=member,
                        row_count=row_count,
                        year=archive_row["year_start"],
                        level=archive_row["level"],
                    )
                    stats["members_imported"] += 1
                    stats["rows_imported"] += row_count
                    print(f"imported {archive_path.name}:{member} rows={row_count}")
                except Exception as exc:
                    mark_failed(
                        conn,
                        archive_path=str(archive_path),
                        member=member,
                        year=archive_row["year_start"],
                        level=archive_row["level"],
                        error=str(exc),
                    )
                    raise
    finally:
        conn.close()
    return stats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import raw 1m stock bar zip CSV files into ClickHouse market.bars")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="A-share stock bar archive directory")
    parser.add_argument("--status-db", default=str(DEFAULT_STATUS_DB), help="SQLite status DB for resumable imports")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    parser.add_argument("--years", default=None, help="Year filter, e.g. 2024 or 2020-2024")
    parser.add_argument("--archive-limit", type=int, default=None, help="Limit number of yearly archives")
    parser.add_argument("--member-limit", type=int, default=None, help="Limit CSV members per archive")
    parser.add_argument("--mode", choices=("archive", "member"), default="archive", help="archive is fastest; member is resumable/idempotent")
    parser.add_argument("--truncate-bars", action="store_true", help="TRUNCATE market.bars and reset import status before importing")
    parser.add_argument("--progress-members", type=int, default=100, help="Print archive-mode progress every N members; 0 disables")
    parser.add_argument("--dry-run", action="store_true", help="Print work without inserting into ClickHouse")
    parser.add_argument("--source", default="stock_bar_zip", help="Value written to market.bars.source")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stats = import_archives(
        Path(args.source_dir).expanduser(),
        Path(args.status_db).expanduser(),
        years=parse_years(args.years),
        archive_limit=args.archive_limit,
        member_limit=args.member_limit,
        dry_run=args.dry_run,
        env_file=Path(args.env_file).expanduser(),
        compose_file=Path(args.compose_file).expanduser(),
        source=args.source,
        mode=args.mode,
        truncate_bars_first=args.truncate_bars,
        progress_members=args.progress_members,
    )
    print(f"summary: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
