"""
缠论买点有效性验证 CLI

输出：
1. 标准化信号事件表
2. 前瞻收益/回撤统计
3. 随机/动量基准对照
4. 纯信号回测结果
5. 单信号复盘图和 Markdown 报告
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Research import (
    BaselineSignalGenerator,
    ChanSignalBacktest,
    SignalEvaluator,
    SignalLabeler,
    SignalSnapshotCollector,
)
from Research.SignalReport import build_joint_setup_payload, build_research_report, write_csv, write_json
from Research.Visualization import SignalVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Research Chan buy/sell signal effectiveness.")
    parser.add_argument("--codes", required=True, help="Comma separated stock codes, e.g. 600519,000333")
    parser.add_argument("--begin", required=True, help="Begin date, e.g. 2023-01-01")
    parser.add_argument("--end", required=True, help="End date, e.g. 2024-12-31")
    parser.add_argument("--level", default="day", choices=["day", "week", "60m", "30m", "15m", "5m"])
    parser.add_argument("--parent-level", choices=["week", "day"], default=None)
    parser.add_argument("--child-level", choices=["60m", "30m", "15m", "5m"], default=None)
    parser.add_argument("--child-confirm-main-bars", type=int, default=10)
    parser.add_argument("--data-src", default="akshare", choices=["akshare", "baostock", "csv"])
    parser.add_argument("--direction", default="buy", choices=["buy", "sell", "all"])
    parser.add_argument("--holding-period", type=int, default=20)
    parser.add_argument("--stop-loss", type=float, default=0.05)
    parser.add_argument("--include-seg-bs", action="store_true")
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--output-dir", default="./outputs/chan_signal_research")
    return parser.parse_args()


def get_lv(level: str):
    mapping = {
        "day": KL_TYPE.K_DAY,
        "week": KL_TYPE.K_WEEK,
        "60m": KL_TYPE.K_60M,
        "30m": KL_TYPE.K_30M,
        "15m": KL_TYPE.K_15M,
        "5m": KL_TYPE.K_5M,
    }
    return mapping[level]


def get_data_src(data_src: str):
    mapping = {
        "akshare": DATA_SRC.AKSHARE,
        "baostock": DATA_SRC.BAO_STOCK,
        "csv": DATA_SRC.CSV,
    }
    return mapping[data_src]


def build_lv_list(args):
    levels = []
    if args.parent_level:
        levels.append(get_lv(args.parent_level))
    levels.append(get_lv(args.level))
    if args.child_level:
        levels.append(get_lv(args.child_level))
    return levels


def get_signal_lv_idx(args) -> int:
    return 1 if args.parent_level else 0


def load_chan_list(args) -> List[CChan]:
    config = CChanConfig({
        "bi_strict": True,
        "trigger_step": False,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": "1,1p,2,2s,3a,3b",
        "print_warning": True,
        "zs_algo": "normal",
        "cal_rsi": True,
        "cal_kdj": True,
    })
    chan_list: List[CChan] = []
    for raw_code in args.codes.split(","):
        code = raw_code.strip()
        if not code:
            continue
        print(f"Loading {code} ...")
        try:
            chan = CChan(
                code=code,
                begin_time=args.begin,
                end_time=args.end,
                data_src=get_data_src(args.data_src),
                lv_list=build_lv_list(args),
                config=config,
                autype=AUTYPE.QFQ,
            )
        except Exception as exc:
            print(f"[WARN] Skip {code}: {exc}")
            continue
        chan_list.append(chan)
    return chan_list


def filter_records(records: List[Dict], source: str) -> List[Dict]:
    return [record for record in records if record.get("source") == source]


def filter_snapshots(snapshots, source: str):
    return [snapshot for snapshot in snapshots if snapshot.source == source]


def split_counts_by_direction(snapshots):
    buy_counts: Dict[str, int] = {}
    sell_counts: Dict[str, int] = {}
    for snapshot in snapshots:
        target = buy_counts if snapshot.direction == "buy" else sell_counts
        target[snapshot.code] = target.get(snapshot.code, 0) + 1
    return buy_counts, sell_counts


def calc_percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    left = math.floor(pos)
    right = math.ceil(pos)
    if left == right:
        return ordered[left]
    weight = pos - left
    return ordered[left] * (1 - weight) + ordered[right] * weight


def assign_environment_groups(records: List[Dict]) -> None:
    vol_values = [
        float(record["env_avg_range_10"])
        for record in records
        if record.get("env_avg_range_10") not in (None, "")
    ]
    low_cut = calc_percentile(vol_values, 1 / 3)
    high_cut = calc_percentile(vol_values, 2 / 3)

    for record in records:
        trend_value = record.get("env_trend_return_20")
        if trend_value in (None, ""):
            record["env_trend_regime"] = "unknown"
        else:
            trend_value = float(trend_value)
            if trend_value >= 0.08:
                record["env_trend_regime"] = "bullish"
            elif trend_value <= -0.08:
                record["env_trend_regime"] = "bearish"
            else:
                record["env_trend_regime"] = "ranging"

        vol_value = record.get("env_avg_range_10")
        if vol_value in (None, "") or low_cut is None or high_cut is None:
            record["env_vol_regime"] = "unknown"
        else:
            vol_value = float(vol_value)
            if vol_value <= low_cut:
                record["env_vol_regime"] = "low_vol"
            elif vol_value >= high_cut:
                record["env_vol_regime"] = "high_vol"
            else:
                record["env_vol_regime"] = "mid_vol"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chan_list = load_chan_list(args)
    if not chan_list:
        raise RuntimeError("No chan data loaded.")

    collector = SignalSnapshotCollector()
    labeler = SignalLabeler()
    evaluator = SignalEvaluator()
    baseline_generator = BaselineSignalGenerator(seed=42)
    visualizer = SignalVisualizer()
    signal_lv_idx = get_signal_lv_idx(args)
    backtest = ChanSignalBacktest({
        "holding_period": args.holding_period,
        "stop_loss_pct": args.stop_loss,
    })

    print("Collecting signal snapshots ...")
    chan_snapshots = collector.collect(
        chan_list,
        direction=args.direction,
        include_seg_bs=args.include_seg_bs,
        signal_lv_idx=signal_lv_idx,
        child_confirm_main_bars=args.child_confirm_main_bars,
    )
    if not chan_snapshots:
        raise RuntimeError("No chan snapshots collected for the selected settings.")

    if args.direction == "all":
        buy_counts, sell_counts = split_counts_by_direction(chan_snapshots)
        random_snapshots = baseline_generator.generate_random_signals(
            chan_list,
            target_counts=buy_counts,
            direction="buy",
            signal_lv_idx=signal_lv_idx,
            min_future_bars=args.holding_period,
        )
        random_snapshots.extend(
            baseline_generator.generate_random_signals(
                chan_list,
                target_counts=sell_counts,
                direction="sell",
                signal_lv_idx=signal_lv_idx,
                min_future_bars=args.holding_period,
            )
        )
        momentum_snapshots = baseline_generator.generate_momentum_signals(
            chan_list,
            target_counts=buy_counts,
            direction="buy",
            signal_lv_idx=signal_lv_idx,
            lookback=max(args.holding_period, 20),
            min_future_bars=args.holding_period,
        )
        momentum_snapshots.extend(
            baseline_generator.generate_momentum_signals(
                chan_list,
                target_counts=sell_counts,
                direction="sell",
                signal_lv_idx=signal_lv_idx,
                lookback=max(args.holding_period, 20),
                min_future_bars=args.holding_period,
            )
        )
    else:
        target_counts = baseline_generator.count_by_code(chan_snapshots)
        random_snapshots = baseline_generator.generate_random_signals(
            chan_list,
            target_counts=target_counts,
            direction=args.direction,
            signal_lv_idx=signal_lv_idx,
            min_future_bars=args.holding_period,
        )
        momentum_snapshots = baseline_generator.generate_momentum_signals(
            chan_list,
            target_counts=target_counts,
            direction=args.direction,
            signal_lv_idx=signal_lv_idx,
            lookback=max(args.holding_period, 20),
            min_future_bars=args.holding_period,
        )
    all_snapshots = chan_snapshots + random_snapshots + momentum_snapshots

    print("Evaluating forward returns ...")
    records = labeler.label_snapshots(
        all_snapshots,
        holding_periods=(5, 10, args.holding_period),
        stop_loss_pct=args.stop_loss,
    )
    assign_environment_groups(records)

    snapshot_rows = [snapshot.to_dict() for snapshot in all_snapshots]
    write_csv(output_dir / "signal_snapshots.csv", snapshot_rows)
    write_csv(output_dir / "signal_evaluations.csv", records)

    overall_summary = [
        evaluator.summarize_overall(records, holding_period=args.holding_period, group_name="ALL")
    ]
    source_summary = evaluator.summarize_by(records, group_field="source", holding_period=args.holding_period)
    bsp_summary = evaluator.summarize_by(filter_records(records, "chan_bsp"), group_field="bsp_type", holding_period=args.holding_period)
    joint_type_assessments = evaluator.summarize_joint_type_assessments(
        filter_records(records, "chan_bsp"),
        holding_period=args.holding_period,
    )
    joint_setup_payload = build_joint_setup_payload(joint_type_assessments)

    print("Running event backtests ...")
    backtest_results = {
        "chan_bsp": backtest.run(filter_snapshots(all_snapshots, "chan_bsp")),
        "baseline_random": backtest.run(filter_snapshots(all_snapshots, "baseline_random")),
        "baseline_momentum": backtest.run(filter_snapshots(all_snapshots, "baseline_momentum")),
    }
    write_json(output_dir / "backtest_summary.json", backtest_results)

    print("Rendering case images ...")
    case_images = []
    chan_records = [record for record in records if record.get("source") == "chan_bsp" and record.get(f"h{args.holding_period}_available") == 1]
    chan_records.sort(key=lambda item: item.get(f"h{args.holding_period}_return", 0.0), reverse=True)
    top_cases = chan_records[:args.case_count]
    bottom_cases = list(reversed(chan_records[-args.case_count:])) if chan_records else []
    for title_prefix, selected in [("top", top_cases), ("bottom", bottom_cases)]:
        for idx, record in enumerate(selected, start=1):
            snapshot = next(
                (
                    item for item in chan_snapshots
                    if item.code == record["code"] and item.signal_idx == record["signal_idx"]
                ),
                None,
            )
            if snapshot is None:
                continue
            file_name = f"{title_prefix}_{idx}_{snapshot.code}_{snapshot.signal_time.toDateStr('-')}_{snapshot.bsp_type or snapshot.signal_kind}.png"
            output_path = output_dir / "cases" / file_name
            visualizer.render_signal(snapshot, output_path=output_path, evaluation=record, x_range=max(120, args.holding_period * 8))
            case_images.append({
                "title": f"{title_prefix.upper()} #{idx} {snapshot.code} {snapshot.signal_time_str} {snapshot.bsp_type}",
                "path": str(output_path),
            })

    config_payload = {
        "codes": [code.strip() for code in args.codes.split(",") if code.strip()],
        "begin": args.begin,
        "end": args.end,
        "level": args.level,
        "parent_level": args.parent_level,
        "child_level": args.child_level,
        "child_confirm_main_bars": args.child_confirm_main_bars,
        "data_src": args.data_src,
        "direction": args.direction,
        "holding_period": args.holding_period,
        "stop_loss": args.stop_loss,
        "include_seg_bs": args.include_seg_bs,
    }
    write_json(output_dir / "stage_conclusions.json", {
        "config": config_payload,
        "research_focus": "Evaluate multilevel joint setups: parent context + main-level BSP type + child interval-set path.",
        "overall_summary": overall_summary,
        "source_summary": source_summary,
        "bsp_type_summary": bsp_summary,
        "joint_setup_assessments": joint_setup_payload,
    })
    build_research_report(
        output_path=output_dir / "research_report.md",
        config=config_payload,
        overall_summaries=overall_summary,
        source_summaries=source_summary,
        bsp_type_summaries=bsp_summary,
        joint_type_assessments=joint_type_assessments,
        backtest_metrics=backtest_results,
        case_images=case_images,
    )

    print(f"Research outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
