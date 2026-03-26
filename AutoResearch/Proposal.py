from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class SweepProposalResult:
    source_summary_path: Path
    source_spec_path: Path
    output_path: Path
    generated_spec: Dict[str, Any]
    selected_runs: List[Dict[str, Any]]


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return normalized or "training-sweep"


def _dedupe_preserve_order(values: Iterable[Any]) -> List[Any]:
    ordered: List[Any] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _resolve_latest_sweep_summary_path(results_root: Path) -> Path:
    candidates = sorted(
        Path(results_root).glob("sweeps/*/runs/*/summary.json"),
        key=lambda path: (path.parent.name, str(path)),
    )
    if not candidates:
        raise FileNotFoundError(f"No sweep summaries found under {results_root / 'sweeps'}")
    return candidates[-1]


def _extract_grid_by_path(spec_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    sweep_payload = spec_payload.get("sweep", {})
    raw_grid = sweep_payload.get("grid", [])
    return {
        str(item.get("path", "")): item
        for item in raw_grid
        if isinstance(item, dict) and str(item.get("path", "")).strip()
    }


def _is_numeric_grid(grid: Dict[str, Any]) -> bool:
    values = grid.get("values", [])
    return isinstance(values, list) and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)


def _as_decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _infer_refinement_step(grid_values: Sequence[Any], winner_values: Sequence[Any]) -> Decimal:
    numeric_values = sorted({_as_decimal(value) for value in grid_values})
    if len(numeric_values) >= 2:
        deltas = [right - left for left, right in zip(numeric_values, numeric_values[1:]) if right > left]
        if deltas:
            return min(deltas) / Decimal("2")

    winner_decimals = sorted({_as_decimal(value) for value in winner_values})
    if len(winner_decimals) >= 2:
        deltas = [right - left for left, right in zip(winner_decimals, winner_decimals[1:]) if right > left]
        if deltas:
            return min(deltas) / Decimal("2")

    lone_value = winner_decimals[0] if winner_decimals else Decimal("0")
    if lone_value == lone_value.to_integral_value():
        return Decimal("1")

    magnitude = abs(lone_value) or Decimal("0.01")
    return max(magnitude * Decimal("0.25"), Decimal("0.01"))


def _to_number(value: Decimal, *, integer_mode: bool) -> Any:
    if integer_mode:
        return int(value)
    return float(value)


def _refine_numeric_grid(grid_values: Sequence[Any], winner_values: Sequence[Any]) -> List[Any]:
    integer_mode = all(float(value).is_integer() for value in grid_values)
    refinement_step = _infer_refinement_step(grid_values, winner_values)
    if integer_mode and refinement_step < Decimal("1"):
        refinement_step = Decimal("1")

    non_negative_only = all(float(value) >= 0.0 for value in grid_values)
    candidates: List[Decimal] = []
    for raw_winner in _dedupe_preserve_order(winner_values):
        winner = _as_decimal(raw_winner)
        neighborhood = [winner - refinement_step, winner, winner + refinement_step]
        for candidate in neighborhood:
            if non_negative_only and candidate < 0:
                continue
            candidates.append(candidate)

    unique_candidates = sorted(set(candidates))
    return [_to_number(value, integer_mode=integer_mode) for value in unique_candidates]


def _select_top_runs(summary_payload: Dict[str, Any], top_runs: int) -> List[Dict[str, Any]]:
    runs = summary_payload.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Sweep summary is missing runs[]")

    completed_runs = [run for run in runs if isinstance(run, dict) and run.get("status") == "completed"]
    if not completed_runs:
        raise ValueError("Sweep summary has no completed runs to refine")
    return completed_runs[: max(1, int(top_runs))]


def _append_unique_tags(existing_tags: Sequence[Any], *new_tags: str) -> List[str]:
    tags = [str(tag) for tag in existing_tags]
    for tag in new_tags:
        if tag not in tags:
            tags.append(tag)
    return tags


def _materialize_benchmark_references(spec_payload: Dict[str, Any]) -> None:
    raw_benchmarks = spec_payload.get("benchmark_selections")
    if isinstance(raw_benchmarks, list):
        for item in raw_benchmarks:
            if isinstance(item, dict) and item.get("selection") is not None:
                item["reference_path"] = None

    legacy_benchmark = spec_payload.get("benchmark_selection")
    if isinstance(legacy_benchmark, dict) and legacy_benchmark.get("selection") is not None:
        legacy_benchmark["reference_path"] = None


def propose_next_sweep(
    *,
    results_root: Path,
    summary_path: Optional[Path] = None,
    output_dir: Path,
    top_runs: int = 2,
) -> SweepProposalResult:
    resolved_results_root = Path(results_root).resolve()
    resolved_summary_path = (
        Path(summary_path).resolve()
        if summary_path is not None
        else _resolve_latest_sweep_summary_path(resolved_results_root).resolve()
    )
    source_spec_path = resolved_summary_path.with_name("spec.json")
    if not source_spec_path.exists():
        raise FileNotFoundError(f"Sweep spec snapshot not found next to summary: {source_spec_path}")

    source_spec = json.loads(source_spec_path.read_text(encoding="utf-8"))
    if str(source_spec.get("mode", "")) != "training_sweep":
        raise ValueError(f"Sweep proposal requires a training_sweep spec snapshot: {source_spec_path}")
    if not isinstance(source_spec.get("sweep", {}).get("grid"), list) or not source_spec["sweep"]["grid"]:
        raise ValueError(f"Sweep proposal requires a non-empty sweep.grid in {source_spec_path}")

    summary_payload = json.loads(resolved_summary_path.read_text(encoding="utf-8"))
    selected_runs = _select_top_runs(summary_payload, top_runs=top_runs)
    grid_by_path = _extract_grid_by_path(source_spec)

    generated_spec = json.loads(json.dumps(source_spec))
    source_run_id = str(summary_payload.get("run_id") or resolved_summary_path.parent.name)
    generated_name = f"{source_spec['name']}-autoresearch-v1-{source_run_id.lower()}"
    generated_spec["name"] = generated_name
    try:
        summary_label = str(resolved_summary_path.relative_to(resolved_results_root))
    except ValueError:
        summary_label = str(resolved_summary_path)
    generated_spec["description"] = (
        f"Auto-generated v1 refinement from {summary_label} "
        f"using the top {len(selected_runs)} completed run(s)."
    )
    generated_spec["tags"] = _append_unique_tags(
        generated_spec.get("tags", []),
        "autoresearch-generated",
        "autoresearch-v1",
    )
    _materialize_benchmark_references(generated_spec)

    best_run_config = dict(selected_runs[0].get("config", {}))
    for path, grid in grid_by_path.items():
        winner_values = [
            dict(run.get("config", {}))[path]
            for run in selected_runs
            if isinstance(run.get("config"), dict) and path in run["config"]
        ]
        if not winner_values:
            continue

        if _is_numeric_grid(grid):
            refined_values = _refine_numeric_grid(grid["values"], winner_values)
            for item in generated_spec["sweep"]["grid"]:
                if item.get("path") == path:
                    item["values"] = refined_values
                    break
            _set_nested_value(generated_spec, path, best_run_config[path])
            continue

        winning_categories = _dedupe_preserve_order(winner_values)
        for item in generated_spec["sweep"]["grid"]:
            if item.get("path") == path:
                item["values"] = winning_categories
                break
        _set_nested_value(generated_spec, path, best_run_config[path])

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / f"{_slugify(generated_name)}.json"
    output_path.write_text(json.dumps(generated_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    return SweepProposalResult(
        source_summary_path=resolved_summary_path,
        source_spec_path=source_spec_path.resolve(),
        output_path=output_path.resolve(),
        generated_spec=generated_spec,
        selected_runs=selected_runs,
    )


def _set_nested_value(payload: Dict[str, Any], path: str, value: Any) -> None:
    parts = [part.strip() for part in path.split(".") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid proposal path '{path}'")

    cursor: Dict[str, Any] = payload
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set proposal path '{path}' because '{part}' is not an object")
        cursor = next_value
    cursor[parts[-1]] = value
