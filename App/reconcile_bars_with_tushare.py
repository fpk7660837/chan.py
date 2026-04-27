from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.import_stock_bar_1m_raw_to_clickhouse import (  # noqa: E402
    DEFAULT_COMPOSE_FILE,
    DEFAULT_ENV_FILE,
    _sql_literal,
    run_clickhouse_query,
)
from App.import_tushare_adj_factors import DEFAULT_API_URL, load_tushare_token  # noqa: E402


MINUTE_FIELDS = "ts_code,trade_time,open,close,high,low,vol,amount"


def year_month_bounds(year_month: str) -> Tuple[str, str, str, str]:
    if len(year_month) != 6 or not year_month.isdigit():
        raise ValueError(f"invalid year_month {year_month!r}; expected YYYYMM")
    year = int(year_month[:4])
    month = int(year_month[4:])
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year:04d}-{month:02d}-01"
    end_date = f"{year:04d}-{month:02d}-{last_day:02d}"
    return start_date, end_date, f"{start_date} 09:00:00", f"{end_date} 15:30:00"


def parse_tushare_minute_response(payload: Dict[str, object]) -> List[Dict[str, object]]:
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


def normalize_tushare_minute_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, float | str]]:
    normalized: List[Dict[str, float | str]] = []
    for row in rows:
        normalized.append(
            {
                "symbol": str(row["ts_code"]).strip().upper(),
                "trade_time": str(row["trade_time"]).strip(),
                "open": float(row["open"]),
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row["vol"]),
                "amount": float(row["amount"]),
            }
        )
    normalized.sort(key=lambda row: str(row["trade_time"]))
    return normalized


def request_tushare_minute_json(
    *,
    api_url: str,
    token: str,
    symbol: str,
    start_time: str,
    end_time: str,
    timeout: float,
    retries: int = 4,
    retry_backoff: float = 1.0,
) -> Dict[str, object]:
    payload = {
        "api_name": "stk_mins",
        "token": token,
        "params": {
            "ts_code": symbol,
            "freq": "1min",
            "start_date": start_time,
            "end_date": end_time,
        },
        "fields": MINUTE_FIELDS,
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "chan.py-tushare-minute-reconcile/1.0"},
        method="POST",
    )
    attempts = max(0, retries) + 1
    for attempt_index in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8")
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise RuntimeError("Tushare response is not a JSON object")
            return parsed
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError):
            if attempt_index >= attempts - 1:
                raise
            if isinstance(sys.exc_info()[1], urllib.error.HTTPError):
                error = sys.exc_info()[1]
                if isinstance(error, urllib.error.HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                    raise
            time.sleep(retry_backoff * (2**attempt_index))
    raise RuntimeError("unreachable")


def fetch_tushare_minute_rows(
    *,
    symbol: str,
    trade_date: str,
    token: str,
    api_url: str,
    timeout: float,
    retries: int = 4,
    retry_backoff: float = 1.0,
) -> List[Dict[str, float | str]]:
    payload = request_tushare_minute_json(
        api_url=api_url,
        token=token,
        symbol=symbol,
        start_time=f"{trade_date} 09:00:00",
        end_time=f"{trade_date} 15:30:00",
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
    )
    return normalize_tushare_minute_rows(parse_tushare_minute_response(payload))


def fetch_tushare_minute_range_rows(
    *,
    symbol: str,
    start_time: str,
    end_time: str,
    token: str,
    api_url: str,
    timeout: float,
    retries: int = 4,
    retry_backoff: float = 1.0,
) -> List[Dict[str, float | str]]:
    payload = request_tushare_minute_json(
        api_url=api_url,
        token=token,
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
    )
    return normalize_tushare_minute_rows(parse_tushare_minute_response(payload))


def fetch_clickhouse_minute_rows(
    *,
    symbol: str,
    trade_date: str,
    env_file: Path,
    compose_file: Path,
) -> List[Dict[str, float | str]]:
    query = f"""
SELECT
    toString(trade_time) AS trade_time,
    open,
    close,
    high,
    low,
    volume,
    amount
FROM market.bars
WHERE level = '1m'
  AND symbol = {_sql_literal(symbol)}
  AND trade_date = {_sql_literal(trade_date)}
ORDER BY trade_time
FORMAT JSONEachRow
""".strip()
    output = run_clickhouse_query(query, env_file=env_file, compose_file=compose_file)
    rows: List[Dict[str, float | str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "trade_time": str(row["trade_time"])[:19],
                "open": float(row["open"]),
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row["volume"]),
                "amount": float(row["amount"]),
            }
        )
    rows.sort(key=lambda row: str(row["trade_time"]))
    return rows


def fetch_clickhouse_minute_range_rows(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    env_file: Path,
    compose_file: Path,
) -> List[Dict[str, float | str]]:
    query = f"""
SELECT
    toString(trade_time) AS trade_time,
    open,
    close,
    high,
    low,
    volume,
    amount
FROM market.bars
WHERE level = '1m'
  AND symbol = {_sql_literal(symbol)}
  AND trade_date >= {_sql_literal(start_date)}
  AND trade_date <= {_sql_literal(end_date)}
ORDER BY trade_time
FORMAT JSONEachRow
""".strip()
    output = run_clickhouse_query(query, env_file=env_file, compose_file=compose_file)
    rows: List[Dict[str, float | str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "trade_time": str(row["trade_time"])[:19],
                "open": float(row["open"]),
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row["volume"]),
                "amount": float(row["amount"]),
            }
        )
    rows.sort(key=lambda row: str(row["trade_time"]))
    return rows


def summarize_day(rows: Sequence[Dict[str, float | str]], *, volume_multiplier: float = 1.0) -> Dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    return {
        "rows": len(rows),
        "open": float(rows[0]["open"]),
        "close": float(rows[-1]["close"]),
        "high": max(float(row["high"]) for row in rows),
        "low": min(float(row["low"]) for row in rows),
        "volume": sum(float(row["volume"]) for row in rows) * volume_multiplier,
        "amount": sum(float(row["amount"]) for row in rows),
    }


def analyze_minute_alignment(
    clickhouse_rows: Sequence[Dict[str, float | str]],
    tushare_rows: Sequence[Dict[str, float | str]],
    *,
    price_tolerance: float = 0.009,
) -> Dict[str, int]:
    ts_by_time = {str(row["trade_time"]): row for row in tushare_rows}
    common_times = [str(row["trade_time"]) for row in clickhouse_rows if str(row["trade_time"]) in ts_by_time]
    same_timestamp_open_matches = 0
    previous_close_open_matches = 0
    same_timestamp_close_matches = 0
    same_timestamp_high_matches = 0
    same_timestamp_low_matches = 0

    for index, trade_time in enumerate(common_times):
        clickhouse_row = clickhouse_rows[index]
        tushare_row = ts_by_time[trade_time]
        if abs(float(clickhouse_row["open"]) - float(tushare_row["open"])) <= price_tolerance:
            same_timestamp_open_matches += 1
        if abs(float(clickhouse_row["close"]) - float(tushare_row["close"])) <= price_tolerance:
            same_timestamp_close_matches += 1
        if abs(float(clickhouse_row["high"]) - float(tushare_row["high"])) <= price_tolerance:
            same_timestamp_high_matches += 1
        if abs(float(clickhouse_row["low"]) - float(tushare_row["low"])) <= price_tolerance:
            same_timestamp_low_matches += 1
        if index > 0:
            previous_tushare_row = ts_by_time[common_times[index - 1]]
            if abs(float(clickhouse_row["open"]) - float(previous_tushare_row["close"])) <= price_tolerance:
                previous_close_open_matches += 1

    return {
        "common_rows": len(common_times),
        "same_timestamp_open_matches": same_timestamp_open_matches,
        "previous_close_open_matches": previous_close_open_matches,
        "same_timestamp_close_matches": same_timestamp_close_matches,
        "same_timestamp_high_matches": same_timestamp_high_matches,
        "same_timestamp_low_matches": same_timestamp_low_matches,
    }


def group_rows_by_trade_date(rows: Sequence[Dict[str, float | str]]) -> Dict[str, List[Dict[str, float | str]]]:
    grouped: Dict[str, List[Dict[str, float | str]]] = {}
    for row in sorted(rows, key=lambda item: str(item["trade_time"])):
        trade_date = str(row["trade_time"])[:10]
        grouped.setdefault(trade_date, []).append(dict(row))
    return grouped


def is_zero_volume_placeholder(
    rows: Sequence[Dict[str, float | str]],
    *,
    price_tolerance: float = 1e-9,
) -> bool:
    if not rows:
        return False
    first_open = float(rows[0]["open"])
    for row in rows:
        if abs(float(row["volume"])) > price_tolerance:
            return False
        if abs(float(row["amount"])) > price_tolerance:
            return False
        if abs(float(row["open"]) - first_open) > price_tolerance:
            return False
        if abs(float(row["close"]) - first_open) > price_tolerance:
            return False
        if abs(float(row["high"]) - first_open) > price_tolerance:
            return False
        if abs(float(row["low"]) - first_open) > price_tolerance:
            return False
    return True


def reconcile_day(
    *,
    clickhouse_rows: Sequence[Dict[str, float | str]],
    tushare_rows: Sequence[Dict[str, float | str]],
) -> Dict[str, object]:
    clickhouse_day = summarize_day(clickhouse_rows, volume_multiplier=100.0)
    tushare_day = summarize_day(tushare_rows, volume_multiplier=1.0)
    return {
        "clickhouse_day": {
            **clickhouse_day,
            "volume_lots": clickhouse_day["volume"] / 100.0,
        },
        "tushare_day": tushare_day,
        "diff": {
            "open": clickhouse_day["open"] - tushare_day["open"],
            "close": clickhouse_day["close"] - tushare_day["close"],
            "high": clickhouse_day["high"] - tushare_day["high"],
            "low": clickhouse_day["low"] - tushare_day["low"],
            "volume_shares": clickhouse_day["volume"] - tushare_day["volume"],
            "amount": clickhouse_day["amount"] - tushare_day["amount"],
        },
        "minute_alignment": analyze_minute_alignment(clickhouse_rows, tushare_rows),
    }


def build_day_result(
    *,
    symbol: str,
    trade_date: str,
    clickhouse_rows: Sequence[Dict[str, float | str]],
    tushare_rows: Sequence[Dict[str, float | str]],
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "symbol": symbol,
        "trade_date": trade_date,
        "clickhouse_rows": len(clickhouse_rows),
        "tushare_rows": len(tushare_rows),
    }
    if clickhouse_rows and tushare_rows:
        result["status"] = "ok"
        result.update(reconcile_day(clickhouse_rows=clickhouse_rows, tushare_rows=tushare_rows))
        return result
    if clickhouse_rows:
        if is_zero_volume_placeholder(clickhouse_rows):
            result["status"] = "clickhouse_zero_volume_placeholder"
            return result
        result["status"] = "missing_in_tushare"
        return result
    if tushare_rows:
        if is_zero_volume_placeholder(tushare_rows):
            result["status"] = "tushare_zero_volume_placeholder"
            return result
        result["status"] = "missing_in_clickhouse"
        return result
    result["status"] = "missing_both"
    return result


def summarize_results(
    results: Sequence[Dict[str, object]],
    *,
    price_tolerance: float = 0.01,
) -> Dict[str, float | int]:
    ok_count = 0
    missing_in_tushare_count = 0
    missing_in_clickhouse_count = 0
    missing_both_count = 0
    tushare_zero_volume_placeholder_count = 0
    clickhouse_zero_volume_placeholder_count = 0
    row_match_count = 0
    day_open_match_count = 0
    day_close_match_count = 0
    day_high_match_le_0_01_count = 0
    day_low_match_le_0_01_count = 0
    max_abs_volume_shares_diff = 0.0
    max_abs_amount_diff = 0.0
    max_abs_amount_diff_bps = 0.0
    same_ts_open_rates: List[float] = []
    prev_close_open_rates: List[float] = []

    for result in results:
        status = str(result.get("status") or "")
        if status == "ok":
            ok_count += 1
        elif status == "missing_in_tushare":
            missing_in_tushare_count += 1
            continue
        elif status == "missing_in_clickhouse":
            missing_in_clickhouse_count += 1
            continue
        elif status == "missing_both":
            missing_both_count += 1
            continue
        elif status == "tushare_zero_volume_placeholder":
            tushare_zero_volume_placeholder_count += 1
            continue
        elif status == "clickhouse_zero_volume_placeholder":
            clickhouse_zero_volume_placeholder_count += 1
            continue
        else:
            continue

        clickhouse_rows = int(result.get("clickhouse_rows") or 0)
        tushare_rows = int(result.get("tushare_rows") or 0)
        if clickhouse_rows == tushare_rows:
            row_match_count += 1

        diff = result.get("diff") or {}
        if not isinstance(diff, dict):
            diff = {}
        open_diff = abs(float(diff.get("open") or 0.0))
        close_diff = abs(float(diff.get("close") or 0.0))
        high_diff = abs(float(diff.get("high") or 0.0))
        low_diff = abs(float(diff.get("low") or 0.0))
        volume_diff = abs(float(diff.get("volume_shares") or 0.0))
        amount_diff = abs(float(diff.get("amount") or 0.0))

        if open_diff <= 1e-9:
            day_open_match_count += 1
        if close_diff <= 1e-9:
            day_close_match_count += 1
        if high_diff <= price_tolerance:
            day_high_match_le_0_01_count += 1
        if low_diff <= price_tolerance:
            day_low_match_le_0_01_count += 1

        max_abs_volume_shares_diff = max(max_abs_volume_shares_diff, volume_diff)
        max_abs_amount_diff = max(max_abs_amount_diff, amount_diff)

        tushare_day = result.get("tushare_day") or {}
        if isinstance(tushare_day, dict):
            tushare_amount = abs(float(tushare_day.get("amount") or 0.0))
            if tushare_amount > 0:
                max_abs_amount_diff_bps = max(max_abs_amount_diff_bps, amount_diff / tushare_amount * 10000.0)

        minute_alignment = result.get("minute_alignment") or {}
        if isinstance(minute_alignment, dict):
            common_rows = int(minute_alignment.get("common_rows") or 0)
            if common_rows > 0:
                same_ts_open_rates.append(float(minute_alignment.get("same_timestamp_open_matches") or 0.0) / common_rows)
            if common_rows > 1:
                prev_close_open_rates.append(float(minute_alignment.get("previous_close_open_matches") or 0.0) / (common_rows - 1))

    return {
        "symbol_day_count": len(results),
        "ok_count": ok_count,
        "missing_in_tushare_count": missing_in_tushare_count,
        "missing_in_clickhouse_count": missing_in_clickhouse_count,
        "missing_both_count": missing_both_count,
        "tushare_zero_volume_placeholder_count": tushare_zero_volume_placeholder_count,
        "clickhouse_zero_volume_placeholder_count": clickhouse_zero_volume_placeholder_count,
        "row_match_count": row_match_count,
        "day_open_match_count": day_open_match_count,
        "day_close_match_count": day_close_match_count,
        "day_high_match_le_0_01_count": day_high_match_le_0_01_count,
        "day_low_match_le_0_01_count": day_low_match_le_0_01_count,
        "max_abs_volume_shares_diff": max_abs_volume_shares_diff,
        "max_abs_amount_diff": max_abs_amount_diff,
        "max_abs_amount_diff_bps": max_abs_amount_diff_bps,
        "avg_same_ts_open_match_rate": sum(same_ts_open_rates) / len(same_ts_open_rates) if same_ts_open_rates else 0.0,
        "avg_prev_close_open_match_rate": sum(prev_close_open_rates) / len(prev_close_open_rates) if prev_close_open_rates else 0.0,
    }


def sample_symbol_days_query(*, year_month: str, sample_limit: int) -> str:
    limit = max(1, sample_limit)
    return f"""
SELECT symbol, toString(trade_date) AS trade_date
FROM market.bars
WHERE level = '1m'
  AND toYYYYMM(trade_time) = {int(year_month)}
GROUP BY symbol, trade_date
ORDER BY cityHash64(symbol, toString(trade_date))
LIMIT {limit}
FORMAT JSONEachRow
""".strip()


def sample_symbol_days(*, year_month: str, sample_limit: int, env_file: Path, compose_file: Path) -> List[Dict[str, str]]:
    output = run_clickhouse_query(
        sample_symbol_days_query(year_month=year_month, sample_limit=sample_limit),
        env_file=env_file,
        compose_file=compose_file,
    )
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def month_symbols_query(*, year_month: str) -> str:
    return f"""
SELECT symbol
FROM market.bars
WHERE level = '1m'
  AND toYYYYMM(trade_time) = {int(year_month)}
GROUP BY symbol
ORDER BY symbol
FORMAT TSV
""".strip()


def list_symbols_for_month(*, year_month: str, env_file: Path, compose_file: Path) -> List[str]:
    output = run_clickhouse_query(
        month_symbols_query(year_month=year_month),
        env_file=env_file,
        compose_file=compose_file,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def compare_symbol_day(
    *,
    symbol: str,
    trade_date: str,
    token: str,
    api_url: str,
    timeout: float,
    env_file: Path,
    compose_file: Path,
    retries: int = 4,
    retry_backoff: float = 1.0,
) -> Dict[str, object]:
    clickhouse_rows = fetch_clickhouse_minute_rows(
        symbol=symbol,
        trade_date=trade_date,
        env_file=env_file,
        compose_file=compose_file,
    )
    tushare_rows = fetch_tushare_minute_rows(
        symbol=symbol,
        trade_date=trade_date,
        token=token,
        api_url=api_url,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
    )
    return build_day_result(
        symbol=symbol,
        trade_date=trade_date,
        clickhouse_rows=clickhouse_rows,
        tushare_rows=tushare_rows,
    )


def compare_symbol_month(
    *,
    symbol: str,
    year_month: str,
    token: str,
    api_url: str,
    timeout: float,
    env_file: Path,
    compose_file: Path,
    retries: int = 4,
    retry_backoff: float = 1.0,
) -> List[Dict[str, object]]:
    start_date, end_date, start_time, end_time = year_month_bounds(year_month)
    clickhouse_rows = fetch_clickhouse_minute_range_rows(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        env_file=env_file,
        compose_file=compose_file,
    )
    tushare_rows = fetch_tushare_minute_range_rows(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        token=token,
        api_url=api_url,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
    )
    clickhouse_by_date = group_rows_by_trade_date(clickhouse_rows)
    tushare_by_date = group_rows_by_trade_date(tushare_rows)
    results: List[Dict[str, object]] = []
    for trade_date in sorted(set(clickhouse_by_date) | set(tushare_by_date)):
        results.append(
            build_day_result(
                symbol=symbol,
                trade_date=trade_date,
                clickhouse_rows=clickhouse_by_date.get(trade_date, []),
                tushare_rows=tushare_by_date.get(trade_date, []),
            )
        )
    return results


def load_existing_symbol_records(report_path: Path) -> List[Dict[str, object]]:
    if not report_path.exists():
        return []
    records: List[Dict[str, object]] = []
    with report_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def append_json_line(report_path: Path, payload: Dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile raw 1m ClickHouse bars against Tushare stk_mins")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="ClickHouse docker compose env file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE_FILE), help="ClickHouse docker compose file")
    parser.add_argument("--token-file", default=None, help="File containing a Tushare token; env vars are checked first")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Tushare JSON API URL")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=4, help="Retry count for transient Tushare HTTP failures")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Base backoff seconds between retries")
    parser.add_argument("--symbol", default=None, help="Single symbol to reconcile, e.g. 000001.SZ")
    parser.add_argument("--trade-date", default=None, help="Single trade date in YYYY-MM-DD format")
    parser.add_argument("--year-month", default=None, help="Sample symbol-days from a month, e.g. 202506")
    parser.add_argument("--sample-limit", type=int, default=12, help="Sample size when using --year-month")
    parser.add_argument("--full-month", default=None, help="Reconcile every symbol in a month, e.g. 202506")
    parser.add_argument("--max-symbols", type=int, default=None, help="Optional cap when using --full-month")
    parser.add_argument("--report-path", default=None, help="Optional JSONL output path for --full-month records")
    parser.add_argument("--resume", action="store_true", help="Resume a prior --full-month run from --report-path")
    parser.add_argument("--progress-every", type=int, default=50, help="Progress interval for --full-month")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    env_file = Path(args.env_file).expanduser()
    compose_file = Path(args.compose_file).expanduser()
    token = load_tushare_token(Path(args.token_file).expanduser() if args.token_file else None)

    results: List[Dict[str, object]]
    if args.symbol and args.trade_date:
        results = [
            compare_symbol_day(
                symbol=args.symbol,
                trade_date=args.trade_date,
                token=token,
                api_url=args.api_url,
                timeout=args.timeout,
                env_file=env_file,
                compose_file=compose_file,
                retries=args.retries,
                retry_backoff=args.retry_backoff,
            )
        ]
    elif args.year_month:
        results = []
        for item in sample_symbol_days(
            year_month=args.year_month,
            sample_limit=args.sample_limit,
            env_file=env_file,
            compose_file=compose_file,
        ):
            results.append(
                compare_symbol_day(
                    symbol=str(item["symbol"]),
                    trade_date=str(item["trade_date"]),
                    token=token,
                    api_url=args.api_url,
                    timeout=args.timeout,
                    env_file=env_file,
                    compose_file=compose_file,
                    retries=args.retries,
                    retry_backoff=args.retry_backoff,
                )
            )
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2), flush=True)
        return 0

    if args.full_month:
        report_path = Path(args.report_path).expanduser() if args.report_path else None
        existing_records: List[Dict[str, object]] = []
        if report_path and report_path.exists() and not args.resume:
            raise SystemExit(f"report path already exists: {report_path}; pass --resume or use a new path")
        if report_path and args.resume:
            existing_records = load_existing_symbol_records(report_path)
        processed_symbols = {str(record.get("symbol")) for record in existing_records if record.get("symbol")}
        symbols = list_symbols_for_month(year_month=args.full_month, env_file=env_file, compose_file=compose_file)
        if args.max_symbols is not None:
            symbols = symbols[: max(0, args.max_symbols)]
        pending_symbols = [symbol for symbol in symbols if symbol not in processed_symbols]
        records = list(existing_records)
        for index, symbol in enumerate(pending_symbols, start=len(processed_symbols) + 1):
            symbol_results = compare_symbol_month(
                symbol=symbol,
                year_month=args.full_month,
                token=token,
                api_url=args.api_url,
                timeout=args.timeout,
                env_file=env_file,
                compose_file=compose_file,
                retries=args.retries,
                retry_backoff=args.retry_backoff,
            )
            record = {
                "symbol": symbol,
                "year_month": args.full_month,
                "summary": summarize_results(symbol_results),
                "results": symbol_results,
            }
            records.append(record)
            if report_path:
                append_json_line(report_path, record)
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(symbols)):
                print(
                    f"[{args.full_month}] processed {index}/{len(symbols)} symbols ({symbol})",
                    file=sys.stderr,
                    flush=True,
                )
        all_results = [result for record in records for result in record.get("results", []) if isinstance(result, dict)]
        print(
            json.dumps(
                {
                    "mode": "full_month",
                    "year_month": args.full_month,
                    "symbol_count": len(symbols),
                    "processed_symbol_count": len(records),
                    "summary": summarize_results(all_results),
                    "report_path": str(report_path) if report_path else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    raise SystemExit("pass either --symbol with --trade-date, or --year-month, or --full-month")


if __name__ == "__main__":
    raise SystemExit(main())
