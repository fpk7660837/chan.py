from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from Research.SignalReport import markdown_table, write_csv


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_metric_value(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def build_leaderboard_rows(manifests: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    successful = [manifest for manifest in manifests if manifest.get("status", "completed") == "completed"]
    ordered = sorted(
        successful,
        key=lambda manifest: (
            _as_float(manifest.get("leaderboard_value")),
            _as_float(manifest.get("top_score")),
            _as_float(manifest.get("avg_score")),
            _as_float(manifest.get("recommendation_count")),
        ),
        reverse=True,
    )

    rows: List[Dict[str, Any]] = []
    for idx, manifest in enumerate(ordered, 1):
        rows.append(
            {
                "rank": idx,
                "workflow": str(manifest.get("workflow", "selection")),
                "experiment": str(manifest.get("experiment", "")),
                "run_id": str(manifest.get("run_id", "")),
                "as_of": str(manifest.get("as_of", "")),
                "metric": str(manifest.get("leaderboard_metric", "")),
                "metric_value": _format_metric_value(manifest.get("leaderboard_value")),
                "top_score": _format_metric_value(manifest.get("top_score")),
                "avg_score": _format_metric_value(manifest.get("avg_score")),
                "recommendations": int(_as_float(manifest.get("recommendation_count"))),
                "model_version": str(manifest.get("model_version", "")),
                "tags": ",".join(manifest.get("tags", [])),
            }
        )
    return rows


def render_leaderboard_markdown(rows: List[Dict[str, Any]]) -> str:
    body = markdown_table(
        rows,
        [
            "rank",
            "workflow",
            "experiment",
            "run_id",
            "as_of",
            "metric",
            "metric_value",
            "top_score",
            "avg_score",
            "recommendations",
            "model_version",
            "tags",
        ],
    )
    return "# AutoResearch Leaderboard\n\n" + body


def write_leaderboard(root_dir: Path, manifests: Iterable[Dict[str, Any]], filename: str = "leaderboard.md") -> Tuple[Path, Path]:
    target_root = Path(root_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    markdown_path = target_root / filename
    csv_path = target_root / (Path(filename).stem + ".csv")
    rows = build_leaderboard_rows(manifests)
    markdown_path.write_text(render_leaderboard_markdown(rows), encoding="utf-8")
    write_csv(csv_path, rows)
    return markdown_path, csv_path


def _format_config_mapping(config: Dict[str, Any]) -> str:
    if not config:
        return "-"
    return ", ".join(f"{key}={config[key]}" for key in sorted(config.keys()))


def build_sweep_leaderboard_rows(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    run_items = list(runs)
    rank_rows = build_leaderboard_rows([item["manifest"] for item in run_items if isinstance(item.get("manifest"), dict)])
    rank_by_run_id = {row["run_id"]: row["rank"] for row in rank_rows}

    ordered = sorted(
        run_items,
        key=lambda item: (
            1 if item.get("status") == "completed" else 0,
            _as_float(item.get("leaderboard_value")),
            _as_float(item.get("top_score")),
            _as_float(item.get("avg_score")),
            str(item.get("variant_id", "")),
        ),
        reverse=True,
    )

    rows: List[Dict[str, Any]] = []
    for item in ordered:
        run_id = str(item.get("run_id", ""))
        rows.append(
            {
                "rank": rank_by_run_id.get(run_id, "-"),
                "status": str(item.get("status", "")),
                "variant": str(item.get("variant_id", "")),
                "experiment": str(item.get("experiment", "")),
                "run_id": run_id,
                "metric": str(item.get("leaderboard_metric", "")),
                "metric_value": _format_metric_value(item.get("leaderboard_value")),
                "top_score": _format_metric_value(item.get("top_score")),
                "avg_score": _format_metric_value(item.get("avg_score")),
                "model_version": str(item.get("model_version", "")),
                "config": _format_config_mapping(item.get("config", {})),
            }
        )
    return rows


def render_sweep_leaderboard_markdown(rows: List[Dict[str, Any]]) -> str:
    body = markdown_table(
        rows,
        [
            "rank",
            "status",
            "variant",
            "experiment",
            "run_id",
            "metric",
            "metric_value",
            "top_score",
            "avg_score",
            "model_version",
            "config",
        ],
    )
    return "# AutoResearch Sweep Leaderboard\n\n" + body


def write_sweep_leaderboard(markdown_path: Path, csv_path: Path, runs: Iterable[Dict[str, Any]]) -> Tuple[Path, Path]:
    rows = build_sweep_leaderboard_rows(runs)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_sweep_leaderboard_markdown(rows), encoding="utf-8")
    write_csv(csv_path, rows)
    return markdown_path, csv_path
