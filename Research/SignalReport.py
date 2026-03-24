from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "-"


def markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No data_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float) and any(token in col for token in ("rate", "return", "mfe", "mae", "drawdown", "share")):
                values.append(format_pct(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


PARENT_SEG_LABELS = {
    "down_seg": "上级别下降段",
    "up_seg": "上级别上升段",
    "no_seg": "上级别无线段",
}

PARENT_POSITION_LABELS = {
    "below_zs": "位于中枢下方",
    "inside_zs": "位于中枢内部",
    "above_zs": "位于中枢上方",
    "no_zs": "当前段无中枢",
    "no_seg": "无线段结构",
    "no_parent_klu": "无上级别K线映射",
}

PARENT_BIAS_LABELS = {
    "aligned": "最近上级别信号同向",
    "counter": "最近上级别信号反向",
    "none": "上级别无最近信号",
}

CHILD_SIDE_LABELS = {
    "same": "次级别先同向",
    "opposite": "次级别先反向",
}

CHILD_FAMILY_LABELS = {
    "1_like": "1类信号",
    "2_like": "2类信号",
    "3_like": "3类信号",
    "2_3_like": "2/3类复合信号",
    "1_2_like": "1/2类复合信号",
    "1_3_like": "1/3类复合信号",
    "mixed": "混合信号",
    "unknown": "未知信号",
}

CHILD_SPEED_LABELS = {
    "fast": "快速出现",
    "delayed": "延后出现",
    "unknown": "出现时点未知",
}

CHILD_OUTCOME_LABELS = {
    "clean": "后续无反向干扰",
    "mixed": "期间出现反向干扰",
    "then_same": "之后转为同向确认",
    "only": "窗口内未转为同向确认",
}


def _format_parent_context(parent_context: str) -> str:
    if not parent_context:
        return "无上级别背景"
    parts = parent_context.split("|")
    if len(parts) != 3:
        return parent_context
    seg, position, bias = parts
    return " / ".join([
        PARENT_SEG_LABELS.get(seg, seg),
        PARENT_POSITION_LABELS.get(position, position),
        PARENT_BIAS_LABELS.get(bias, bias),
    ])


def _format_child_interval_pattern(pattern: str) -> str:
    if not pattern or pattern == "no_signal":
        return "次级别窗口内未出现可执行信号"
    parts = pattern.split("|")
    if len(parts) != 4:
        return pattern
    side, family, speed, outcome = parts
    return " / ".join([
        CHILD_SIDE_LABELS.get(side, side),
        CHILD_FAMILY_LABELS.get(family, family),
        CHILD_SPEED_LABELS.get(speed, speed),
        CHILD_OUTCOME_LABELS.get(outcome, outcome),
    ])


def format_joint_setup_label(parent_context: str, child_interval_pattern: str) -> str:
    return f"{_format_parent_context(parent_context)} + {_format_child_interval_pattern(child_interval_pattern)}"


def _prepare_joint_setup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared = []
    for row in rows:
        item = dict(row)
        item["setup"] = format_joint_setup_label(
            str(row.get("parent_context", "")),
            str(row.get("child_interval_pattern", "")),
        )
        prepared.append(item)
    return prepared


def _pick_joint_rows(
    rows: List[Dict[str, Any]],
    *,
    min_count: int,
    limit: int,
    reverse: bool,
) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if int(row.get("count", 0)) >= min_count]
    candidates.sort(
        key=lambda row: (
            float(row.get("avg_return", 0.0)),
            float(row.get("win_rate", 0.0)),
            -float(row.get("stop_loss_rate", 0.0)),
        ),
        reverse=reverse,
    )
    return candidates[:limit]


def build_joint_setup_notes(assessment: Dict[str, Any]) -> List[str]:
    core = assessment.get("core_summary", {})
    core_avg_return = float(core.get("avg_return", 0.0))
    core_win_rate = float(core.get("win_rate", 0.0))
    core_stop_loss = float(core.get("stop_loss_rate", 0.0))
    notes = [
        (
            f"基础表现：count={core.get('count', 0)}，"
            f"win_rate={format_pct(core.get('win_rate', 0.0))}，"
            f"avg_return={format_pct(core.get('avg_return', 0.0))}，"
            f"stop_loss_rate={format_pct(core.get('stop_loss_rate', 0.0))}。"
        )
    ]

    joint_rows = assessment.get("joint_setup_summary", [])
    repeated_rows = [row for row in joint_rows if int(row.get("count", 0)) >= 3]
    if not repeated_rows:
        notes.append("该类型可重复的联立 setup 样本还不够，暂时只能看总体有效性，不能下执行细分结论。")
        return notes

    coverage_rows = sorted(repeated_rows, key=lambda row: int(row.get("count", 0)), reverse=True)[:2]
    notes.append(
        "主要 setup 分布：" + "；".join(
            f"{format_joint_setup_label(str(row.get('parent_context', '')), str(row.get('child_interval_pattern', '')))}"
            f" (count={row.get('count', 0)}, share={format_pct(row.get('share_of_type', 0.0))})"
            for row in coverage_rows
        ) + "。"
    )

    strong_rows = _pick_joint_rows(repeated_rows, min_count=3, limit=2, reverse=True)
    if strong_rows:
        notes.append(
            "重复出现且更优的 setup：" + "；".join(
                f"{format_joint_setup_label(str(row.get('parent_context', '')), str(row.get('child_interval_pattern', '')))}"
                f" (count={row.get('count', 0)}, win_rate={format_pct(row.get('win_rate', 0.0))}, "
                f"avg_return={format_pct(row.get('avg_return', 0.0))}, stop_loss_rate={format_pct(row.get('stop_loss_rate', 0.0))})"
                for row in strong_rows
            ) + "。"
        )

    risky_candidates = [
        row for row in repeated_rows
        if (
            float(row.get("stop_loss_rate", 0.0)) > core_stop_loss + 1e-9
            or float(row.get("win_rate", 0.0)) < core_win_rate - 0.05
        )
    ]
    risky_rows = sorted(
        risky_candidates,
        key=lambda row: (
            float(row.get("stop_loss_rate", 0.0)),
            -float(row.get("win_rate", 0.0)),
            -float(row.get("avg_return", 0.0)),
        ),
        reverse=True,
    )[:2]
    if risky_rows:
        notes.append(
            "需要谨慎或回避的 setup：" + "；".join(
                f"{format_joint_setup_label(str(row.get('parent_context', '')), str(row.get('child_interval_pattern', '')))}"
                f" (count={row.get('count', 0)}, win_rate={format_pct(row.get('win_rate', 0.0))}, "
                f"avg_return={format_pct(row.get('avg_return', 0.0))}, stop_loss_rate={format_pct(row.get('stop_loss_rate', 0.0))})"
                for row in risky_rows
            ) + "。"
        )
    else:
        low_priority_rows = sorted(
            [
                row for row in repeated_rows
                if float(row.get("avg_return", 0.0)) < core_avg_return * 0.6
            ],
            key=lambda row: (
                float(row.get("avg_return", 0.0)),
                float(row.get("win_rate", 0.0)),
            ),
        )[:2]
        if low_priority_rows:
            notes.append(
                "重复出现但收益弹性偏弱的 setup：" + "；".join(
                    f"{format_joint_setup_label(str(row.get('parent_context', '')), str(row.get('child_interval_pattern', '')))}"
                    f" (count={row.get('count', 0)}, win_rate={format_pct(row.get('win_rate', 0.0))}, "
                    f"avg_return={format_pct(row.get('avg_return', 0.0))})"
                    for row in low_priority_rows
                ) + "。"
            )
        else:
            notes.append("重复出现的 setup 暂未看到明显失败组合，差异更多体现在收益空间大小，而不是胜率或止损显著恶化。")
    return notes


def build_joint_setup_payload(type_assessments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = []
    for assessment in type_assessments:
        payload.append({
            "bsp_type": assessment.get("bsp_type"),
            "core_summary": assessment.get("core_summary", {}),
            "notes": build_joint_setup_notes(assessment),
            "joint_setup_summary": _prepare_joint_setup_rows(assessment.get("joint_setup_summary", [])),
        })
    return payload


def build_research_report(
    *,
    output_path: Path,
    config: Dict[str, Any],
    overall_summaries: List[Dict[str, Any]],
    source_summaries: List[Dict[str, Any]],
    bsp_type_summaries: List[Dict[str, Any]],
    joint_type_assessments: List[Dict[str, Any]],
    backtest_metrics: Dict[str, Dict[str, Any]],
    case_images: List[Dict[str, Any]],
):
    lines = [
        "# Chan Signal Research Report",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Overall Summary",
        "",
        markdown_table(overall_summaries, ["group", "count", "eligible_count", "win_rate", "avg_return", "avg_mfe", "avg_mae", "stop_loss_rate", "reverse_signal_rate"]),
        "",
        "## Research Focus",
        "",
        "研究对象是多级别联立 setup：`上级别结构背景 + 本级别买卖点类型 + 次级别区间套路径`。本报告不再把上级别和区间套当成独立因子单独打分。",
        "",
        "## Source Comparison",
        "",
        markdown_table(source_summaries, ["source", "count", "eligible_count", "win_rate", "avg_return", "avg_mfe", "avg_mae", "stop_loss_rate"]),
        "",
        "## Chan Signal Coverage",
        "",
        markdown_table(bsp_type_summaries, ["bsp_type", "count", "eligible_count", "win_rate", "avg_return", "avg_mfe", "avg_mae", "stop_loss_rate"]),
        "",
        "## Joint Setup Assessments",
        "",
        "以下各节按本级别类型展开，只研究该类型内部不同多级别联立 setup 的表现差异。",
        "",
    ]

    for assessment in joint_type_assessments:
        bsp_type = assessment.get("bsp_type")
        joint_rows = _prepare_joint_setup_rows(assessment.get("joint_setup_summary", []))
        lines.extend([
            f"### Type {bsp_type}",
            "",
            "#### Base Validity",
            "",
            markdown_table([assessment.get("core_summary", {})], ["bsp_type", "count", "eligible_count", "win_rate", "avg_return", "median_return", "avg_mfe", "avg_mae", "stop_loss_rate", "reverse_signal_rate"]),
            "",
            "#### Joint Setup Table",
            "",
            markdown_table(joint_rows, ["setup", "count", "share_of_type", "eligible_count", "win_rate", "avg_return", "avg_mfe", "avg_mae", "stop_loss_rate", "reverse_signal_rate"]),
            "",
            "#### Setup Notes",
            "",
        ])
        for note in build_joint_setup_notes(assessment):
            lines.append(f"- {note}")
        lines.append("")

    lines.extend([
        "## Backtest Comparison",
        "",
        "这里只保留固定规则回测的辅助指标，用来确认缠论事件相对基准是否具备边际优势，不用于本级别类型之间的排名。",
        "",
    ])

    backtest_rows = []
    for source, payload in backtest_metrics.items():
        metrics = payload.get("metrics", {})
        backtest_rows.append({
            "source": source,
            "executed_trades": int(metrics.get("executed_trades", 0)),
            "win_rate": metrics.get("win_rate", 0.0),
            "avg_return": metrics.get("avg_return", 0.0),
            "profit_loss_ratio": round(float(metrics.get("profit_loss_ratio", 0.0)), 4),
            "stop_loss_rate": metrics.get("stop_loss_rate", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "avg_holding_bars": round(float(metrics.get("avg_holding_bars", 0.0)), 2),
        })
    lines.extend([
        markdown_table(backtest_rows, ["source", "executed_trades", "win_rate", "avg_return", "profit_loss_ratio", "stop_loss_rate", "max_drawdown", "avg_holding_bars"]),
        "",
        "## Case Images",
        "",
    ])

    if not case_images:
        lines.append("_No case images generated_")
    else:
        for case in case_images:
            lines.append(f"### {case['title']}")
            lines.append("")
            lines.append(f"![{case['title']}]({case['path']})")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
