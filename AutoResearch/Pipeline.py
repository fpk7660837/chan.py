from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .Leaderboard import write_leaderboard
from .Selection import SelectionRunResult, run_selection_experiment
from .Spec import ExperimentSpec, load_experiment_spec
from .Storage import RunPaths, RunStorage


@dataclass(frozen=True)
class PipelineRunResult:
    spec: ExperimentSpec
    run_paths: RunPaths
    manifest: Dict[str, object]
    leaderboard_markdown: Path
    leaderboard_csv: Path


class AutoResearchPipeline:
    def __init__(
        self,
        results_root: Optional[Path] = None,
        selection_runner: Callable[[ExperimentSpec], SelectionRunResult] = run_selection_experiment,
    ):
        self.results_root = Path(results_root) if results_root is not None else None
        self.selection_runner = selection_runner

    def run(self, spec_path: Path) -> PipelineRunResult:
        spec = load_experiment_spec(Path(spec_path))
        root_dir = self.results_root or Path(spec.storage.root_dir)
        storage = RunStorage(root_dir)
        run_paths = storage.create_run(spec.name)
        storage.write_spec_snapshot(run_paths, spec.to_dict())

        status = "completed"
        try:
            selection_result = self.selection_runner(spec)
            recommendations = selection_result.recommendations
            summary = selection_result.summary
        except Exception as exc:
            status = "failed"
            recommendations = []
            summary = {
                "as_of": spec.selection.as_of,
                "model_version": spec.selection.model_version or "latest",
                "recommendation_count": 0,
                "skipped_count": 0,
                "top_score": 0.0,
                "avg_score": 0.0,
                "error": str(exc),
            }

        storage.write_recommendations(run_paths, recommendations)
        storage.write_summary(run_paths, summary)

        manifest = self._build_manifest(spec, run_paths, summary, status=status)
        storage.write_manifest(run_paths, manifest)

        leaderboard_markdown, leaderboard_csv = write_leaderboard(
            root_dir=root_dir,
            manifests=storage.load_manifests(),
            filename=spec.storage.leaderboard_filename,
        )
        return PipelineRunResult(
            spec=spec,
            run_paths=run_paths,
            manifest=manifest,
            leaderboard_markdown=leaderboard_markdown,
            leaderboard_csv=leaderboard_csv,
        )

    def run_many(self, spec_paths: Iterable[Path]) -> List[PipelineRunResult]:
        return [self.run(path) for path in spec_paths]

    @staticmethod
    def _build_manifest(
        spec: ExperimentSpec,
        run_paths: RunPaths,
        summary: Dict[str, object],
        *,
        status: str,
    ) -> Dict[str, object]:
        portfolio_summary = summary.get("portfolio_backtest", {})
        if isinstance(portfolio_summary, dict) and "sharpe_ratio" in portfolio_summary:
            leaderboard_metric = "portfolio_sharpe"
            leaderboard_value = float(portfolio_summary["sharpe_ratio"])
        else:
            leaderboard_metric = "top_score"
            leaderboard_value = float(summary.get("top_score", 0.0) or 0.0)

        return {
            "status": status,
            "experiment": spec.name,
            "description": spec.description,
            "tags": spec.tags,
            "run_id": run_paths.run_dir.name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "as_of": spec.selection.as_of,
            "model_version": summary.get("model_version", spec.selection.model_version or "latest"),
            "leaderboard_metric": leaderboard_metric,
            "leaderboard_value": leaderboard_value,
            "top_score": float(summary.get("top_score", 0.0) or 0.0),
            "avg_score": float(summary.get("avg_score", 0.0) or 0.0),
            "recommendation_count": int(summary.get("recommendation_count", 0) or 0),
            "skipped_count": int(summary.get("skipped_count", 0) or 0),
            "summary_path": str(run_paths.summary_json.relative_to(run_paths.root_dir)),
            "recommendations_csv_path": str(run_paths.recommendations_csv.relative_to(run_paths.root_dir)),
            "spec_snapshot_path": str(run_paths.spec_snapshot_json.relative_to(run_paths.root_dir)),
        }
