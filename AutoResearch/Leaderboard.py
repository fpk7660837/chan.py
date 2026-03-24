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
