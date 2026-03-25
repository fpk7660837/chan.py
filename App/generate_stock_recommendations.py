"""
按指定日期输出股票推荐名单。

示例:
    python3.11 App/generate_stock_recommendations.py --as-of 2024-12-31
    python3.11 App/generate_stock_recommendations.py --as-of 2024-12-31 --codes 600519,000333
    python3.11 App/generate_stock_recommendations.py --as-of 2024-12-31 --codes-file ./codes.txt --output ./outputs/reco.json
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    ak = None
    pd = None

from Chan import CChan
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Config.MLConfig import MLConfig
from ML.FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
from ML.Prediction.Predictor import Predictor
from ML.Utils.ModelIO import ModelIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按日期输出股票推荐名单")
    parser.add_argument("--as-of", required=True, help="推荐日期，格式 YYYY-MM-DD")
    parser.add_argument("--model-version", default=None, help="模型版本号，默认加载最新版本")
    parser.add_argument("--top-k", type=int, default=None, help="输出前 K 只股票")
    parser.add_argument("--min-score", type=float, default=None, help="最低分数阈值")
    parser.add_argument("--signal-lookback-bars", type=int, default=None, help="信号有效回看窗口")
    parser.add_argument("--history-days", type=int, default=900, help="回看多少天历史K线用于构建信号")
    parser.add_argument("--stale-days", type=int, default=20, help="最后一根K线距离 as-of 超过多少天就跳过")
    parser.add_argument("--limit", type=int, default=None, help="仅扫描前 N 只股票，便于试跑")
    parser.add_argument("--codes", default=None, help="逗号分隔的股票代码列表")
    parser.add_argument("--codes-file", default=None, help="股票代码文件，支持 txt/csv/json")
    parser.add_argument("--output", default=None, help="输出文件路径，支持 csv/json")
    return parser.parse_args()


def load_model(version: Optional[str], model_dir: Optional[Path] = None):
    model_io = ModelIO(str(model_dir or "./models"))
    model = model_io.load(version=version)
    metadata = model_io.load_metadata(version=version)
    return model, metadata


def resolve_runtime_config(metadata: Dict, args: argparse.Namespace) -> Dict[str, object]:
    config = MLConfig()
    saved_config = metadata.get("config", {}) if metadata else {}

    if "feature_config" in saved_config:
        config.feature_config.update(saved_config["feature_config"])
    if "prediction_config" in saved_config:
        config.prediction_config.update(saved_config["prediction_config"])

    top_k = args.top_k if args.top_k is not None else int(config.prediction_config.get("top_k", 10))
    min_score = args.min_score if args.min_score is not None else float(config.prediction_config.get("score_threshold", 0.7))
    signal_lookback_bars = (
        args.signal_lookback_bars
        if args.signal_lookback_bars is not None
        else int(config.portfolio_backtest_config.get("signal_lookback_bars", 20))
    )

    return {
        "feature_config": config.feature_config,
        "top_k": top_k,
        "min_score": min_score,
        "signal_lookback_bars": signal_lookback_bars,
    }


def load_universe(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.codes:
        return [(normalize_code(code), "") for code in args.codes.split(",") if code.strip()]
    if args.codes_file:
        return load_codes_from_file(Path(args.codes_file))
    return get_tradable_stocks(limit=args.limit)


def normalize_code(code: str) -> str:
    return code.strip().replace(".SH", "").replace(".SZ", "").replace("sh", "").replace("sz", "")


def load_codes_from_file(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Codes file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, str):
                    result.append((normalize_code(item), ""))
                elif isinstance(item, dict):
                    result.append((normalize_code(str(item.get("code", ""))), str(item.get("name", ""))))
            return [item for item in result if item[0]]

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                code_key = "code" if "code" in reader.fieldnames else reader.fieldnames[0]
                name_key = "name" if "name" in reader.fieldnames else None
                rows = []
                for row in reader:
                    code = normalize_code(str(row.get(code_key, "")))
                    if not code:
                        continue
                    rows.append((code, str(row.get(name_key, "")) if name_key else ""))
                return rows

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "," in raw:
            code, name = raw.split(",", 1)
            rows.append((normalize_code(code), name.strip()))
        else:
            rows.append((normalize_code(raw), ""))
    return rows


def get_tradable_stocks(limit: Optional[int] = None) -> List[Tuple[str, str]]:
    if ak is None or pd is None:
        raise RuntimeError("akshare and pandas are required when codes/codes-file are not provided")

    df = ak.stock_zh_a_spot_em()
    df = df[~df["名称"].str.contains("ST", case=False, na=False)]
    df = df[~df["代码"].str.startswith("688")]
    df = df[~df["代码"].str.startswith("8")]
    df = df[~df["代码"].str.startswith("43")]
    df = df[~df["代码"].str.startswith("200")]
    df = df[~df["代码"].str.startswith("900")]
    df = df[~df["代码"].str.startswith("920")]
    df = df[df["成交量"] > 0]
    df = df[df["最新价"] > 0]

    if limit is not None:
        df = df.head(limit)

    return [(str(row["代码"]), str(row["名称"])) for _, row in df.iterrows()]


def load_chan_pool(
    universe: Sequence[Tuple[str, str]],
    as_of: datetime,
    history_days: int,
    stale_days: int,
) -> Tuple[List[CChan], Dict[str, str], List[Tuple[str, str]]]:
    begin_time = (as_of - timedelta(days=history_days)).strftime("%Y-%m-%d")
    end_time = as_of.strftime("%Y-%m-%d")

    chan_list: List[CChan] = []
    code_name_map: Dict[str, str] = {}
    skipped: List[Tuple[str, str]] = []

    for idx, (code, name) in enumerate(universe, 1):
        print(f"[{idx}/{len(universe)}] loading {code} {name}".rstrip())
        try:
            chan = CChan(
                code=code,
                begin_time=begin_time,
                end_time=end_time,
                data_src=DATA_SRC.AKSHARE,
                lv_list=[KL_TYPE.K_DAY],
                autype=AUTYPE.QFQ,
            )
            bars = list(chan[0].klu_iter())
            if not bars:
                skipped.append((code, "no_data"))
                continue

            last_bar = bars[-1]
            last_date = datetime(last_bar.time.year, last_bar.time.month, last_bar.time.day)
            if (as_of - last_date).days > stale_days:
                skipped.append((code, f"stale_last_bar:{last_bar.time}"))
                continue

            chan_list.append(chan)
            code_name_map[code] = name
        except Exception as exc:
            skipped.append((code, str(exc)))

    return chan_list, code_name_map, skipped


def build_output_rows(
    ranked: Sequence[Dict],
    code_name_map: Dict[str, str],
    as_of: datetime,
    model_version: Optional[str],
) -> List[Dict[str, object]]:
    rows = []
    for rank, item in enumerate(ranked, 1):
        bsp = item["bsp"]
        rows.append({
            "rank": rank,
            "as_of": as_of.strftime("%Y-%m-%d"),
            "code": item["code"],
            "name": code_name_map.get(item["code"], ""),
            "score": round(float(item["score"]), 6),
            "signal_time": bsp.klu.time.to_str(),
            "signal_type": bsp.type2str(),
            "signal_price": round(float(bsp.klu.close), 4),
            "signal_idx": int(bsp.klu.idx),
            "model_version": model_version or "latest",
        })
    return rows


def write_output(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    fieldnames = list(rows[0].keys()) if rows else [
        "rank",
        "as_of",
        "code",
        "name",
        "score",
        "signal_time",
        "signal_type",
        "signal_price",
        "signal_idx",
        "model_version",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[Dict[str, object]], skipped: Sequence[Tuple[str, str]], output_path: Path) -> None:
    print("\nRecommendations")
    print("-" * 80)
    if not rows:
        print("No recommendations generated.")
    else:
        for row in rows:
            print(
                f"{row['rank']:>2}. {row['code']} {row['name']} | "
                f"score={row['score']:.4f} | time={row['signal_time']} | type={row['signal_type']}"
            )
    print("-" * 80)
    print(f"written to: {output_path}")
    print(f"recommended: {len(rows)}")
    print(f"skipped: {len(skipped)}")


def default_output_path(as_of: datetime) -> Path:
    return ROOT / "outputs" / f"recommendations_{as_of.strftime('%Y-%m-%d')}.csv"


def main() -> None:
    args = parse_args()
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d")

    model, metadata = load_model(args.model_version)
    runtime = resolve_runtime_config(metadata, args)
    predictor = Predictor(model, BSPFeatureExtractor(runtime["feature_config"]))

    universe = load_universe(args)
    if not universe:
        raise RuntimeError("Universe is empty")

    chan_list, code_name_map, skipped = load_chan_pool(
        universe=universe,
        as_of=as_of,
        history_days=args.history_days,
        stale_days=args.stale_days,
    )

    ranked = predictor.rank_stock_pool(
        chan_list,
        top_k=int(runtime["top_k"]),
        direction="buy",
        score_threshold=float(runtime["min_score"]),
        signal_lookback_bars=int(runtime["signal_lookback_bars"]),
    )
    rows = build_output_rows(ranked, code_name_map, as_of, metadata.get("version") if metadata else None)

    output_path = Path(args.output) if args.output else default_output_path(as_of)
    write_output(rows, output_path)
    print_summary(rows, skipped, output_path)


if __name__ == "__main__":
    main()
