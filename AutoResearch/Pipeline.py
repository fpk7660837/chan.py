from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .Leaderboard import write_leaderboard
from .Selection import SelectionRunResult, run_selection_experiment
from .Spec import ExperimentSpec, load_experiment_spec
from .Storage import PublishedModelArtifacts, RunPaths, RunStorage
from .Training import TrainingRunResult, run_training_experiment


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
        training_runner: Callable[[ExperimentSpec, RunPaths], TrainingRunResult] = run_training_experiment,
        publish_model: Optional[bool] = None,
        global_model_dir: Optional[Path] = None,
    ):
        self.results_root = Path(results_root) if results_root is not None else None
        self.selection_runner = selection_runner
        self.training_runner = training_runner
        self.publish_model = publish_model
        self.global_model_dir = Path(global_model_dir) if global_model_dir is not None else None

    def run(self, spec_path: Path) -> PipelineRunResult:
        spec = load_experiment_spec(Path(spec_path))
        root_dir = self.results_root or Path(spec.storage.root_dir)
        storage = RunStorage(root_dir)
        run_paths = storage.create_run(spec.name)
        storage.write_spec_snapshot(run_paths, spec.to_dict())

        recommendations: List[Dict[str, object]] = []
        summary: Dict[str, object]
        training_result: Optional[TrainingRunResult] = None
        published_model_artifacts: Optional[PublishedModelArtifacts] = None
        status = "completed"

        try:
            if spec.mode == "training":
                training_result = self.training_runner(spec, run_paths)
                summary = dict(training_result.summary)
                if self._should_publish_model(spec):
                    published_model_artifacts = storage.publish_model_artifacts(
                        training_result.model_path,
                        training_result.metadata_path,
                        self.global_model_dir or Path(spec.storage.publish_model.target_dir),
                    )
                    summary["published_model_path"] = str(published_model_artifacts.model_path)
                    summary["published_metadata_path"] = (
                        str(published_model_artifacts.metadata_path)
                        if published_model_artifacts.metadata_path is not None
                        else None
                    )
            else:
                selection_result = self.selection_runner(spec)
                recommendations = selection_result.recommendations
                summary = selection_result.summary
        except Exception as exc:
            status = "failed"
            summary = self._build_failure_summary(spec, run_paths, exc)

        if spec.mode == "selection":
            storage.write_recommendations(run_paths, recommendations)
        storage.write_summary(run_paths, summary)

        manifest = self._build_manifest(
            spec,
            run_paths,
            summary,
            status=status,
            training_result=training_result,
            published_model_artifacts=published_model_artifacts,
        )
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

    def _should_publish_model(self, spec: ExperimentSpec) -> bool:
        if spec.mode != "training":
            return False
        if self.publish_model is not None:
            return self.publish_model
        return spec.storage.publish_model.enabled

    @staticmethod
    def _relative_to_root(run_paths: RunPaths, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        return str(Path(path).relative_to(run_paths.root_dir))

    @staticmethod
    def _build_failure_summary(
        spec: ExperimentSpec,
        run_paths: RunPaths,
        exc: Exception,
    ) -> Dict[str, object]:
        if spec.mode == "training":
            model_version = (
                spec.training.model_version
                if spec.training is not None and spec.training.model_version is not None
                else run_paths.run_dir.name
            )
            return {
                "model_version": model_version,
                "model_type": spec.training.model_type if spec.training is not None else "",
                "begin_time": spec.training.begin_time if spec.training is not None else "",
                "end_time": spec.training.end_time if spec.training is not None else "",
                "loaded_chan_count": 0,
                "skipped_count": 0,
                "error": str(exc),
            }

        selection = spec.selection
        return {
            "as_of": selection.as_of if selection is not None else "",
            "model_version": selection.model_version if selection is not None and selection.model_version is not None else "latest",
            "recommendation_count": 0,
            "skipped_count": 0,
            "top_score": 0.0,
            "avg_score": 0.0,
            "error": str(exc),
        }

    @classmethod
    def _build_manifest(
        cls,
        spec: ExperimentSpec,
        run_paths: RunPaths,
        summary: Dict[str, object],
        *,
        status: str,
        training_result: Optional[TrainingRunResult] = None,
        published_model_artifacts: Optional[PublishedModelArtifacts] = None,
    ) -> Dict[str, object]:
        if spec.mode == "training":
            return cls._build_training_manifest(
                spec,
                run_paths,
                summary,
                status=status,
                training_result=training_result,
                published_model_artifacts=published_model_artifacts,
            )
        return cls._build_selection_manifest(spec, run_paths, summary, status=status)

    @staticmethod
    def _build_selection_manifest(
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

        selection = spec.selection
        return {
            "status": status,
            "workflow": "selection",
            "experiment": spec.name,
            "description": spec.description,
            "tags": spec.tags,
            "run_id": run_paths.run_dir.name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "as_of": selection.as_of if selection is not None else "",
            "model_version": summary.get(
                "model_version",
                selection.model_version if selection is not None and selection.model_version is not None else "latest",
            ),
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

    @classmethod
    def _build_training_manifest(
        cls,
        spec: ExperimentSpec,
        run_paths: RunPaths,
        summary: Dict[str, object],
        *,
        status: str,
        training_result: Optional[TrainingRunResult],
        published_model_artifacts: Optional[PublishedModelArtifacts],
    ) -> Dict[str, object]:
        training = spec.training
        leaderboard_value = summary.get("leaderboard_value")
        return {
            "status": status,
            "workflow": "training",
            "experiment": spec.name,
            "description": spec.description,
            "tags": spec.tags,
            "run_id": run_paths.run_dir.name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "as_of": "",
            "model_version": summary.get(
                "model_version",
                training.model_version if training is not None and training.model_version is not None else run_paths.run_dir.name,
            ),
            "leaderboard_metric": str(summary.get("leaderboard_metric", "trained_model")),
            "leaderboard_value": float(leaderboard_value) if leaderboard_value is not None else None,
            "top_score": 0.0,
            "avg_score": 0.0,
            "recommendation_count": 0,
            "skipped_count": int(summary.get("skipped_count", 0) or 0),
            "training_begin": training.begin_time if training is not None else "",
            "training_end": training.end_time if training is not None else "",
            "summary_path": str(run_paths.summary_json.relative_to(run_paths.root_dir)),
            "spec_snapshot_path": str(run_paths.spec_snapshot_json.relative_to(run_paths.root_dir)),
            "model_artifact_path": cls._relative_to_root(run_paths, training_result.model_path if training_result else None),
            "metadata_artifact_path": cls._relative_to_root(
                run_paths,
                training_result.metadata_path if training_result else None,
            ),
            "published_model_path": (
                str(published_model_artifacts.model_path) if published_model_artifacts is not None else None
            ),
            "published_metadata_path": (
                str(published_model_artifacts.metadata_path)
                if published_model_artifacts is not None and published_model_artifacts.metadata_path is not None
                else None
            ),
        }
