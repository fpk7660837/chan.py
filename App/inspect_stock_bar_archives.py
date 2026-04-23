from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_SOURCE_DIR = Path("/Users/kevinfu/Downloads/stock_bar/A股_分时数据_沪深")
DEFAULT_MARKET_DATA_ROOT = Path("/Users/kevinfu/market-data")

MANIFEST_FIELDS = [
    "archive_path",
    "group_dir",
    "level",
    "adjust",
    "year_start",
    "year_end",
    "zip_size_bytes",
    "entry_count",
    "uncompressed_bytes",
    "compressed_bytes",
    "first_member",
    "status",
    "error",
]

DETECTION_FIELDS = [
    "level",
    "year",
    "member",
    "symbol",
    "name",
    "trade_time",
    "raw_open",
    "raw_close",
    "qfq_open",
    "qfq_close",
    "raw_volume",
    "qfq_volume",
    "same_volume",
    "close_ratio_qfq_to_raw",
    "detected",
]


def normalize_symbol(code: str) -> str:
    value = str(code).strip()
    lower = value.lower()
    if re.fullmatch(r"sh\d{6}", lower):
        return f"{lower[2:]}.SH"
    if re.fullmatch(r"sz\d{6}", lower):
        return f"{lower[2:]}.SZ"
    if re.fullmatch(r"bj\d{6}", lower):
        return f"{lower[2:]}.BJ"
    if re.fullmatch(r"\d{6}\.(sh|sz|bj)", lower):
        return value.upper()
    return value


def _infer_level(group_name: str, archive_name: str = "") -> str:
    text = f"{group_name} {archive_name}"
    match = re.search(r"(\d+)\s*分钟", text)
    if match:
        return f"{match.group(1)}m"
    match = re.search(r"(\d+)\s*min", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)}m"
    return "unknown"


def _infer_adjust(group_name: str) -> str:
    if "前复权" in group_name:
        return "qfq"
    if "后复权" in group_name:
        return "hfq"
    return "raw"


def _infer_years(archive_name: str) -> Tuple[str, str]:
    years = re.findall(r"(?:^|_)(\d{4})(?=_|\.|$)", archive_name)
    if not years:
        return "", ""
    return years[0], years[1] if len(years) > 1 else years[0]


def _archive_stats(path: Path) -> Dict[str, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            return {
                "entry_count": str(len(infos)),
                "uncompressed_bytes": str(sum(info.file_size for info in infos)),
                "compressed_bytes": str(sum(info.compress_size for info in infos)),
                "first_member": infos[0].filename if infos else "",
                "status": "ok",
                "error": "",
            }
    except Exception as exc:
        return {
            "entry_count": "0",
            "uncompressed_bytes": "0",
            "compressed_bytes": "0",
            "first_member": "",
            "status": "error",
            "error": str(exc),
        }


def build_manifest(source_dir: Path) -> List[Dict[str, str]]:
    source_dir = Path(source_dir).expanduser().resolve()
    rows: List[Dict[str, str]] = []
    for archive_path in sorted(source_dir.rglob("*.zip")):
        group_dir = archive_path.parent.name
        year_start, year_end = _infer_years(archive_path.name)
        stats = _archive_stats(archive_path)
        rows.append(
            {
                "archive_path": str(archive_path),
                "group_dir": group_dir,
                "level": _infer_level(group_dir, archive_path.name),
                "adjust": _infer_adjust(group_dir),
                "year_start": year_start,
                "year_end": year_end,
                "zip_size_bytes": str(archive_path.stat().st_size),
                **stats,
            }
        )
    return rows


def _index_archives(source_dir: Path) -> Dict[Tuple[str, str, str, str], Path]:
    index: Dict[Tuple[str, str, str, str], Path] = {}
    for row in build_manifest(source_dir):
        if row["status"] != "ok":
            continue
        key = (row["level"], row["adjust"], row["year_start"], row["year_end"])
        index[key] = Path(row["archive_path"])
    return index


def _zip_members(path: Path) -> List[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))


def _first_data_row(path: Path, member: str) -> Optional[Dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as raw_fp:
            text_fp = TextIOWrapper(raw_fp, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_fp)
            for row in reader:
                return {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}
    return None


def _float_text(value: str) -> str:
    number = float(value)
    return str(number).rstrip("0").rstrip(".") if "." in str(number) else str(number)


def _safe_float(value: str) -> float:
    return float(str(value).replace(",", "").strip())


def _compare_rows(level: str, year: str, member: str, raw_row: Dict[str, str], qfq_row: Dict[str, str]) -> Dict[str, str]:
    raw_close = _safe_float(raw_row["收盘价"])
    qfq_close = _safe_float(qfq_row["收盘价"])
    raw_volume = _safe_float(raw_row["成交量"])
    qfq_volume = _safe_float(qfq_row["成交量"])
    same_volume = abs(raw_volume - qfq_volume) < 0.000001
    same_price = abs(raw_close - qfq_close) < 0.000001
    detected = "same_price_pair" if same_price else "raw_and_qfq_pair"
    ratio = qfq_close / raw_close if raw_close else 0.0

    return {
        "level": level,
        "year": year,
        "member": member,
        "symbol": normalize_symbol(raw_row.get("代码", "")),
        "name": raw_row.get("名称", ""),
        "trade_time": raw_row.get("时间", ""),
        "raw_open": _float_text(raw_row["开盘价"]),
        "raw_close": _float_text(raw_row["收盘价"]),
        "qfq_open": _float_text(qfq_row["开盘价"]),
        "qfq_close": _float_text(qfq_row["收盘价"]),
        "raw_volume": _float_text(raw_row["成交量"]),
        "qfq_volume": _float_text(qfq_row["成交量"]),
        "same_volume": "true" if same_volume else "false",
        "close_ratio_qfq_to_raw": f"{ratio:.8f}",
        "detected": detected,
    }


def detect_adjustment_samples(source_dir: Path, sample_limit: int = 20) -> List[Dict[str, str]]:
    source_dir = Path(source_dir).expanduser().resolve()
    index = _index_archives(source_dir)
    samples: List[Dict[str, str]] = []

    for level, adjust, year_start, year_end in sorted(index):
        if adjust != "qfq" or year_start != year_end:
            continue
        raw_path = index.get((level, "raw", year_start, year_end))
        qfq_path = index[(level, adjust, year_start, year_end)]
        if raw_path is None:
            continue

        common_members = sorted(set(_zip_members(raw_path)) & set(_zip_members(qfq_path)))
        for member in common_members:
            if len(samples) >= sample_limit:
                return samples
            raw_row = _first_data_row(raw_path, member)
            qfq_row = _first_data_row(qfq_path, member)
            if not raw_row or not qfq_row:
                continue
            samples.append(_compare_rows(level, year_start, member, raw_row, qfq_row))

    return samples


def _write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_summary(path: Path, manifest_rows: List[Dict[str, str]], detection_rows: List[Dict[str, str]]) -> None:
    by_adjust = Counter(row["adjust"] for row in manifest_rows)
    by_level = Counter(row["level"] for row in manifest_rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "archive_count": len(manifest_rows),
        "archives_by_adjust": dict(sorted(by_adjust.items())),
        "archives_by_level": dict(sorted(by_level.items())),
        "sample_count": len(detection_rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_reports(source_dir: Path, market_data_root: Path, sample_limit: int = 20) -> Tuple[Path, Path]:
    source_dir = Path(source_dir).expanduser().resolve()
    market_data_root = Path(market_data_root).expanduser().resolve()
    manifest_rows = build_manifest(source_dir)
    detection_rows = detect_adjustment_samples(source_dir, sample_limit=sample_limit)

    manifest_path = market_data_root / "manifests" / "stock_bar_source_files.csv"
    detection_path = market_data_root / "reports" / "adjustment_detection" / "stock_bar_adjustment_samples.csv"
    summary_path = market_data_root / "reports" / "adjustment_detection" / "stock_bar_summary.json"
    _write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
    _write_csv(detection_path, detection_rows, DETECTION_FIELDS)
    _write_summary(summary_path, manifest_rows, detection_rows)
    return manifest_path, detection_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local A-share stock bar zip archives without extracting them")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Downloaded stock bar archive directory")
    parser.add_argument("--market-data-root", default=str(DEFAULT_MARKET_DATA_ROOT), help="Output root for manifests/reports")
    parser.add_argument("--sample-limit", type=int, default=20, help="Maximum raw/qfq sample comparisons to write")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    source_dir = Path(args.source_dir).expanduser()
    if not source_dir.exists():
        raise FileNotFoundError(f"source directory not found: {source_dir}")

    manifest_path, detection_path = write_reports(
        source_dir,
        Path(args.market_data_root).expanduser(),
        sample_limit=args.sample_limit,
    )
    print(f"manifest: {manifest_path}")
    print(f"adjustment_detection: {detection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
