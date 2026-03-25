from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .Leaderboard import write_leaderboard
from .Selection import SelectionRunResult, run_selection_experiment
from .Spec import BenchmarkSelectionSpec, BenchmarkSuiteScoringSpec, ExperimentSpec, load_experiment_spec
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
        selection_runner: Callable[[ExperimentSpec, Optional[Path]], SelectionRunResult] = run_selection_experiment,
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
                status = self._run_training_benchmark(
                    spec,
                    summary,
                    training_result,
                    published_model_artifacts=published_model_artifacts,
                    current_status=status,
                )
            else:
                selection_result = self.selection_runner(spec, None)
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

    def _run_training_benchmark(
        self,
        spec: ExperimentSpec,
        summary: Dict[str, object],
        training_result: TrainingRunResult,
        *,
        published_model_artifacts: Optional[PublishedModelArtifacts],
        current_status: str,
    ) -> str:
        if not spec.benchmark_selections:
            return current_status

        model_dir = (
            published_model_artifacts.model_path.parent
            if published_model_artifacts is not None
            else training_result.model_path.parent
        )

        benchmark_results: List[Dict[str, object]] = []
        benchmark_errors: List[str] = []
        for benchmark in spec.benchmark_selections:
            benchmark_spec = self._build_benchmark_spec(spec, benchmark, training_result.model_version)
            benchmark_entry: Dict[str, object] = {
                "name": benchmark.name,
                "reference_path": benchmark.reference_path,
                "weight": benchmark.weight,
            }
            try:
                benchmark_result = self.selection_runner(benchmark_spec, model_dir)
            except Exception as exc:
                benchmark_entry["error"] = str(exc)
                benchmark_errors.append(str(exc))
                benchmark_results.append(benchmark_entry)
                continue

            leaderboard_metric, leaderboard_value = self._resolve_selection_leaderboard(benchmark_result.summary)
            benchmark_entry["leaderboard_metric"] = leaderboard_metric
            benchmark_entry["leaderboard_value"] = leaderboard_value
            benchmark_entry["summary"] = benchmark_result.summary
            benchmark_results.append(benchmark_entry)

        summary["downstream_benchmark_results"] = benchmark_results
        aggregate = self._build_downstream_benchmark_aggregate(benchmark_results, spec.benchmark_suite_scoring)
        if aggregate is not None:
            summary["downstream_benchmark_aggregate"] = aggregate

        if len(benchmark_results) == 1:
            only_result = benchmark_results[0]
            if "summary" in only_result:
                summary["downstream_benchmark"] = only_result["summary"]
            if "error" in only_result:
                summary["downstream_benchmark_error"] = only_result["error"]

        if benchmark_errors:
            summary["downstream_benchmark_error"] = benchmark_errors[0]
            summary["downstream_benchmark_errors"] = benchmark_errors
            return "failed"

        return current_status

    @staticmethod
    def _build_benchmark_spec(
        spec: ExperimentSpec,
        benchmark: BenchmarkSelectionSpec,
        model_version: str,
    ) -> ExperimentSpec:
        return ExperimentSpec(
            name=benchmark.name,
            mode="selection",
            selection=replace(benchmark.selection, model_version=model_version),
            training=None,
            benchmark_selections=[],
            description=spec.description,
            tags=spec.tags,
            portfolio_backtest=benchmark.portfolio_backtest,
            storage=spec.storage,
        )

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
        leaderboard_metric, leaderboard_value = AutoResearchPipeline._resolve_selection_leaderboard(summary)

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

    @staticmethod
    def _resolve_selection_leaderboard(summary: Dict[str, object]) -> tuple[str, float]:
        portfolio_summary = summary.get("portfolio_backtest", {})
        if isinstance(portfolio_summary, dict) and "sharpe_ratio" in portfolio_summary:
            return "portfolio_sharpe", float(portfolio_summary["sharpe_ratio"])
        return "top_score", float(summary.get("top_score", 0.0) or 0.0)

    @staticmethod
    def _summarize_benchmark_as_of(as_of_values: List[str]) -> str:
        unique_values = sorted({value for value in as_of_values if value})
        if not unique_values:
            return ""
        if len(unique_values) == 1:
            return unique_values[0]
        return f"{unique_values[0]}..{unique_values[-1]}"

    @staticmethod
    def _weighted_average(values: List[float], weights: List[float]) -> float:
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return 0.0
        return sum(value * weight for value, weight in zip(values, weights)) / total_weight

    @classmethod
    def _weighted_stddev(cls, values: List[float], weights: List[float]) -> float:
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return 0.0
        weighted_mean = cls._weighted_average(values, weights)
        variance = sum(weight * ((value - weighted_mean) ** 2) for value, weight in zip(values, weights)) / total_weight
        return math.sqrt(variance)

    @classmethod
    def _build_downstream_benchmark_aggregate(
        cls,
        benchmark_results: List[Dict[str, object]],
        suite_scoring: BenchmarkSuiteScoringSpec,
    ) -> Optional[Dict[str, object]]:
        successful_results = [
            result
            for result in benchmark_results
            if isinstance(result.get("summary"), dict)
        ]
        if not successful_results:
            return None

        component_metrics: List[Dict[str, object]] = []
        metric_values: List[float] = []
        successful_weights: List[float] = []
        top_scores: List[float] = []
        avg_scores: List[float] = []
        recommendation_counts: List[int] = []
        skipped_counts: List[int] = []
        as_of_values: List[str] = []
        total_weight = 0.0
        successful_weight = 0.0

        for result in benchmark_results:
            total_weight += float(result.get("weight", 1.0) or 1.0)

        for result in successful_results:
            benchmark_summary = result["summary"]
            if not isinstance(benchmark_summary, dict):
                continue
            weight = float(result.get("weight", 1.0) or 1.0)
            leaderboard_metric = str(result.get("leaderboard_metric", "") or "")
            leaderboard_value = float(result.get("leaderboard_value", 0.0) or 0.0)
            as_of = str(benchmark_summary.get("as_of", "") or "")
            metric_values.append(leaderboard_value)
            successful_weights.append(weight)
            successful_weight += weight
            top_scores.append(float(benchmark_summary.get("top_score", 0.0) or 0.0))
            avg_scores.append(float(benchmark_summary.get("avg_score", 0.0) or 0.0))
            recommendation_counts.append(int(benchmark_summary.get("recommendation_count", 0) or 0))
            skipped_counts.append(int(benchmark_summary.get("skipped_count", 0) or 0))
            if as_of:
                as_of_values.append(as_of)
            component_metrics.append(
                {
                    "name": str(result.get("name", "")),
                    "as_of": as_of,
                    "metric": leaderboard_metric,
                    "value": leaderboard_value,
                    "weight": weight,
                }
            )

        weighted_score_mean = cls._weighted_average(metric_values, successful_weights)
        score_dispersion = cls._weighted_stddev(metric_values, successful_weights)
        failed_weight = max(total_weight - successful_weight, 0.0)
        failure_weight_ratio = failed_weight / total_weight if total_weight > 0.0 else 0.0
        dispersion_penalty_value = suite_scoring.dispersion_penalty * score_dispersion
        failure_penalty_value = suite_scoring.failure_penalty * failure_weight_ratio
        leaderboard_metric = "benchmark_suite_score_v2"
        leaderboard_value = weighted_score_mean - dispersion_penalty_value - failure_penalty_value

        return {
            "benchmark_count": len(benchmark_results),
            "successful_benchmark_count": len(successful_results),
            "failed_benchmark_count": len(benchmark_results) - len(successful_results),
            "total_weight": total_weight,
            "successful_weight": successful_weight,
            "failed_weight": failed_weight,
            "failure_weight_ratio": failure_weight_ratio,
            "as_of": cls._summarize_benchmark_as_of(as_of_values),
            "as_of_values": sorted({value for value in as_of_values if value}),
            "leaderboard_metric": leaderboard_metric,
            "leaderboard_value": leaderboard_value,
            "weighted_score_mean": weighted_score_mean,
            "score_dispersion": score_dispersion,
            "dispersion_penalty_factor": suite_scoring.dispersion_penalty,
            "dispersion_penalty_value": dispersion_penalty_value,
            "failure_penalty_factor": suite_scoring.failure_penalty,
            "failure_penalty_value": failure_penalty_value,
            "top_score": cls._weighted_average(top_scores, successful_weights),
            "avg_score": cls._weighted_average(avg_scores, successful_weights),
            "recommendation_count": sum(recommendation_counts),
            "skipped_count": sum(skipped_counts),
            "component_metrics": component_metrics,
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
        benchmark_summary = summary.get("downstream_benchmark", {})
        benchmark_results = summary.get("downstream_benchmark_results", [])
        benchmark_aggregate = summary.get("downstream_benchmark_aggregate", {})
        has_benchmark_summary = isinstance(benchmark_summary, dict) and bool(benchmark_summary)
        has_benchmark_results = isinstance(benchmark_results, list) and len(benchmark_results) > 0
        has_benchmark_aggregate = isinstance(benchmark_aggregate, dict) and bool(benchmark_aggregate)

        if has_benchmark_results and has_benchmark_aggregate and len(benchmark_results) > 1:
            leaderboard_metric = str(benchmark_aggregate.get("leaderboard_metric", "avg_top_score"))
            leaderboard_value = float(benchmark_aggregate.get("leaderboard_value", 0.0) or 0.0)
            as_of = str(benchmark_aggregate.get("as_of", "") or "")
            top_score = float(benchmark_aggregate.get("top_score", 0.0) or 0.0)
            avg_score = float(benchmark_aggregate.get("avg_score", 0.0) or 0.0)
            recommendation_count = int(benchmark_aggregate.get("recommendation_count", 0) or 0)
            skipped_count = int(benchmark_aggregate.get("skipped_count", 0) or 0)
        elif has_benchmark_summary:
            leaderboard_metric, leaderboard_value = cls._resolve_selection_leaderboard(benchmark_summary)
            as_of = str(benchmark_summary.get("as_of", "") or "")
            top_score = float(benchmark_summary.get("top_score", 0.0) or 0.0)
            avg_score = float(benchmark_summary.get("avg_score", 0.0) or 0.0)
            recommendation_count = int(benchmark_summary.get("recommendation_count", 0) or 0)
            skipped_count = int(benchmark_summary.get("skipped_count", 0) or 0)
        else:
            leaderboard_metric = str(summary.get("leaderboard_metric", "trained_model"))
            leaderboard_value = summary.get("leaderboard_value")
            as_of = ""
            top_score = 0.0
            avg_score = 0.0
            recommendation_count = 0
            skipped_count = int(summary.get("skipped_count", 0) or 0)
        return {
            "status": status,
            "workflow": "training",
            "experiment": spec.name,
            "description": spec.description,
            "tags": spec.tags,
            "run_id": run_paths.run_dir.name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "as_of": as_of,
            "model_version": summary.get(
                "model_version",
                training.model_version if training is not None and training.model_version is not None else run_paths.run_dir.name,
            ),
            "leaderboard_metric": leaderboard_metric,
            "leaderboard_value": float(leaderboard_value) if leaderboard_value is not None else None,
            "top_score": top_score,
            "avg_score": avg_score,
            "recommendation_count": recommendation_count,
            "skipped_count": skipped_count,
            "training_begin": training.begin_time if training is not None else "",
            "training_end": training.end_time if training is not None else "",
            "summary_path": str(run_paths.summary_json.relative_to(run_paths.root_dir)),
            "spec_snapshot_path": str(run_paths.spec_snapshot_json.relative_to(run_paths.root_dir)),
            "training_skipped_count": int(summary.get("skipped_count", 0) or 0),
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
            "downstream_benchmark_summary": benchmark_summary if has_benchmark_summary else None,
            "downstream_benchmark_results": benchmark_results if has_benchmark_results else None,
            "downstream_benchmark_aggregate": benchmark_aggregate if has_benchmark_aggregate else None,
            "downstream_benchmark_error": summary.get("downstream_benchmark_error"),
        }
