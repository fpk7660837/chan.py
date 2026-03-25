import json
import tempfile
import unittest
from pathlib import Path

from AutoResearch.Leaderboard import build_leaderboard_rows, render_leaderboard_markdown
from AutoResearch.Pipeline import AutoResearchPipeline
from AutoResearch.Selection import SelectionRunResult
from AutoResearch.Spec import load_experiment_spec
from AutoResearch.Storage import RunStorage
from AutoResearch.Training import TrainingRunResult


class AutoResearchSpecTests(unittest.TestCase):
    def test_load_experiment_spec_applies_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path = Path(tmp_dir) / "baseline.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-daily-selection",
                        "selection": {
                            "as_of": "2024-12-31",
                            "codes": ["600519", "000333"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(spec_path)

            self.assertEqual(spec.name, "baseline-daily-selection")
            self.assertEqual(spec.selection.as_of, "2024-12-31")
            self.assertEqual(spec.selection.codes, ["600519", "000333"])
            self.assertEqual(spec.selection.top_k, 10)
            self.assertEqual(spec.storage.root_dir, "AutoResearch/results")
            self.assertEqual(spec.storage.leaderboard_filename, "leaderboard.md")

    def test_load_training_experiment_spec_defaults_to_run_local_model_storage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path = Path(tmp_dir) / "training.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                            "model_type": "randomforest",
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(spec_path)

            self.assertEqual(spec.mode, "training")
            self.assertIsNone(spec.selection)
            self.assertEqual(spec.training.begin_time, "2020-01-01")
            self.assertEqual(spec.training.end_time, "2022-12-31")
            self.assertEqual(spec.training.codes, ["600519", "000333"])
            self.assertEqual(spec.training.model_type, "randomforest")
            self.assertFalse(spec.storage.publish_model.enabled)
            self.assertEqual(spec.storage.publish_model.target_dir, "./models")

    def test_load_training_experiment_spec_supports_inline_benchmark_selection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path = Path(tmp_dir) / "training.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "benchmark-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                        },
                        "benchmark_selection": {
                            "as_of": "2024-12-31",
                            "codes": ["600519", "000333"],
                            "top_k": 2,
                        },
                        "portfolio_backtest": {
                            "enabled": True,
                            "top_k": 3,
                            "score_threshold": 0.55,
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(spec_path)

            self.assertIsNotNone(spec.benchmark_selection)
            self.assertEqual(len(spec.benchmark_selections), 1)
            self.assertEqual(spec.benchmark_selection.selection.as_of, "2024-12-31")
            self.assertEqual(spec.benchmark_selection.selection.top_k, 2)
            self.assertEqual(spec.benchmark_selection.portfolio_backtest.top_k, 3)
            self.assertIsNone(spec.benchmark_selection.reference_path)

    def test_load_training_experiment_spec_supports_benchmark_selection_reference(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_spec_path = tmp_path / "baseline_daily_selection.json"
            benchmark_spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-daily-selection",
                        "selection": {
                            "as_of": "2025-01-15",
                            "codes": ["600519", "000333"],
                            "top_k": 4,
                        },
                        "portfolio_backtest": {
                            "enabled": True,
                            "top_k": 2,
                            "score_threshold": 0.61,
                        },
                    }
                ),
                encoding="utf-8",
            )

            training_spec_path = tmp_path / "training.json"
            training_spec_path.write_text(
                json.dumps(
                    {
                        "name": "benchmark-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                        },
                        "benchmark_selection": "./baseline_daily_selection.json",
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(training_spec_path)

            self.assertIsNotNone(spec.benchmark_selection)
            self.assertEqual(len(spec.benchmark_selections), 1)
            self.assertEqual(spec.benchmark_selection.reference_path, "./baseline_daily_selection.json")
            self.assertEqual(spec.benchmark_selection.name, "baseline-daily-selection")
            self.assertEqual(spec.benchmark_selection.selection.as_of, "2025-01-15")
            self.assertEqual(spec.benchmark_selection.selection.top_k, 4)
            self.assertEqual(spec.benchmark_selection.portfolio_backtest.top_k, 2)

    def test_load_training_experiment_spec_supports_multiple_benchmark_selections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_spec_path = tmp_path / "baseline_daily_selection.json"
            benchmark_spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-daily-selection",
                        "selection": {
                            "as_of": "2025-01-15",
                            "codes": ["600519", "000333"],
                            "top_k": 4,
                        },
                        "portfolio_backtest": {
                            "enabled": True,
                            "top_k": 2,
                            "score_threshold": 0.61,
                        },
                    }
                ),
                encoding="utf-8",
            )

            training_spec_path = tmp_path / "training.json"
            training_spec_path.write_text(
                json.dumps(
                    {
                        "name": "benchmark-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                        },
                        "benchmark_selections": [
                            "./baseline_daily_selection.json",
                            {
                                "name": "high-confidence-check",
                                "selection": {
                                    "as_of": "2025-02-01",
                                    "codes": ["600519", "000333"],
                                    "top_k": 2,
                                    "min_score": 0.7,
                                },
                            },
                        ],
                        "portfolio_backtest": {
                            "enabled": True,
                            "top_k": 3,
                            "score_threshold": 0.55,
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(training_spec_path)

            self.assertEqual(len(spec.benchmark_selections), 2)
            self.assertEqual(spec.benchmark_selection.name, "baseline-daily-selection")
            self.assertEqual(spec.benchmark_selections[0].selection.as_of, "2025-01-15")
            self.assertEqual(spec.benchmark_selections[0].portfolio_backtest.top_k, 2)
            self.assertEqual(spec.benchmark_selections[1].name, "high-confidence-check")
            self.assertEqual(spec.benchmark_selections[1].selection.as_of, "2025-02-01")
            self.assertEqual(spec.benchmark_selections[1].portfolio_backtest.top_k, 3)

    def test_load_training_experiment_spec_supports_suite_scoring_and_weighted_benchmark_references(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_spec_path = tmp_path / "baseline_daily_selection.json"
            benchmark_spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-daily-selection",
                        "selection": {
                            "as_of": "2025-01-15",
                            "codes": ["600519", "000333"],
                            "top_k": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )

            training_spec_path = tmp_path / "training.json"
            training_spec_path.write_text(
                json.dumps(
                    {
                        "name": "benchmark-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                        },
                        "benchmark_suite_scoring": {
                            "dispersion_penalty": 0.5,
                            "failure_penalty": 0.6,
                        },
                        "benchmark_selections": [
                            {
                                "reference_path": "./baseline_daily_selection.json",
                                "weight": 2.0,
                            },
                            {
                                "name": "high-confidence-check",
                                "weight": 0.5,
                                "selection": {
                                    "as_of": "2025-02-01",
                                    "codes": ["600519", "000333"],
                                    "top_k": 2,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(training_spec_path)

            self.assertEqual(spec.benchmark_suite_scoring.dispersion_penalty, 0.5)
            self.assertEqual(spec.benchmark_suite_scoring.failure_penalty, 0.6)
            self.assertEqual(spec.benchmark_selections[0].reference_path, "./baseline_daily_selection.json")
            self.assertEqual(spec.benchmark_selections[0].weight, 2.0)
            self.assertEqual(spec.benchmark_selections[1].weight, 0.5)


class RunStorageTests(unittest.TestCase):
    def test_run_storage_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = RunStorage(Path(tmp_dir) / "results")
            run_paths = storage.create_run("baseline-daily-selection", run_id="20260325T120000Z")

            storage.write_spec_snapshot(run_paths, {"name": "baseline-daily-selection"})
            storage.write_recommendations(
                run_paths,
                [{"rank": 1, "code": "600519", "score": 0.8123}],
            )
            storage.write_summary(run_paths, {"top_score": 0.8123, "recommendation_count": 1})
            storage.write_manifest(
                run_paths,
                {
                    "experiment": "baseline-daily-selection",
                    "run_id": "20260325T120000Z",
                    "leaderboard_metric": "top_score",
                    "leaderboard_value": 0.8123,
                    "top_score": 0.8123,
                    "recommendation_count": 1,
                    "as_of": "2024-12-31",
                },
            )

            self.assertTrue(run_paths.spec_snapshot_json.exists())
            self.assertTrue(run_paths.recommendations_csv.exists())
            self.assertTrue(run_paths.recommendations_json.exists())
            self.assertTrue(run_paths.summary_json.exists())
            self.assertTrue(run_paths.manifest_json.exists())

    def test_run_storage_places_model_artifacts_under_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = RunStorage(Path(tmp_dir) / "results")
            run_paths = storage.create_run("baseline-model-training", run_id="20260325T120000Z")

            self.assertEqual(run_paths.model_artifacts_dir, run_paths.run_dir / "models")
            self.assertEqual(run_paths.model_artifacts_dir.parent, run_paths.run_dir)


class LeaderboardTests(unittest.TestCase):
    def test_leaderboard_rows_sort_by_primary_metric_desc(self):
        rows = build_leaderboard_rows(
            [
                {
                    "experiment": "exp-b",
                    "run_id": "run-2",
                    "leaderboard_metric": "portfolio_sharpe",
                    "leaderboard_value": 0.84,
                    "top_score": 0.79,
                    "avg_score": 0.73,
                    "recommendation_count": 5,
                    "as_of": "2024-12-31",
                    "model_version": "v2",
                    "tags": ["baseline"],
                },
                {
                    "experiment": "exp-a",
                    "run_id": "run-1",
                    "leaderboard_metric": "portfolio_sharpe",
                    "leaderboard_value": 1.12,
                    "top_score": 0.76,
                    "avg_score": 0.7,
                    "recommendation_count": 4,
                    "as_of": "2024-12-31",
                    "model_version": "v1",
                    "tags": ["ablation"],
                },
            ]
        )

        self.assertEqual(rows[0]["experiment"], "exp-a")
        markdown = render_leaderboard_markdown(rows)
        self.assertIn("AutoResearch Leaderboard", markdown)
        self.assertIn("portfolio_sharpe", markdown)

    def test_leaderboard_rows_rank_training_suite_runs_by_suite_score(self):
        rows = build_leaderboard_rows(
            [
                {
                    "workflow": "training",
                    "experiment": "suite-b",
                    "run_id": "run-2",
                    "leaderboard_metric": "benchmark_suite_score_v2",
                    "leaderboard_value": 0.94,
                    "top_score": 0.91,
                    "avg_score": 0.85,
                    "recommendation_count": 4,
                    "model_version": "v2",
                },
                {
                    "workflow": "training",
                    "experiment": "suite-a",
                    "run_id": "run-1",
                    "leaderboard_metric": "benchmark_suite_score_v2",
                    "leaderboard_value": 1.12,
                    "top_score": 0.87,
                    "avg_score": 0.8,
                    "recommendation_count": 3,
                    "model_version": "v1",
                },
            ]
        )

        self.assertEqual(rows[0]["experiment"], "suite-a")
        self.assertEqual(rows[0]["metric"], "benchmark_suite_score_v2")


class PipelineTests(unittest.TestCase):
    def test_pipeline_run_writes_run_artifacts_and_leaderboard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            spec_path = tmp_path / "baseline.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-daily-selection",
                        "tags": ["baseline", "daily"],
                        "selection": {
                            "as_of": "2024-12-31",
                            "codes": ["600519", "000333"],
                            "top_k": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_selection_runner(spec, model_dir=None):
                return SelectionRunResult(
                    recommendations=[
                        {
                            "rank": 1,
                            "as_of": spec.selection.as_of,
                            "code": "600519",
                            "name": "Kweichow Moutai",
                            "score": 0.9123,
                            "signal_time": "2024-12-30",
                            "signal_type": "1",
                            "signal_price": 1788.0,
                            "signal_idx": 123,
                            "model_version": "demo-v1",
                        }
                    ],
                    summary={
                        "top_score": 0.9123,
                        "avg_score": 0.9123,
                        "recommendation_count": 1,
                        "skipped_count": 0,
                        "model_version": "demo-v1",
                    },
                )

            pipeline = AutoResearchPipeline(
                results_root=tmp_path / "results",
                selection_runner=fake_selection_runner,
            )

            result = pipeline.run(spec_path)

            self.assertTrue(result.run_paths.manifest_json.exists())
            self.assertTrue(result.run_paths.summary_json.exists())
            self.assertTrue(result.run_paths.recommendations_csv.exists())
            self.assertTrue(result.leaderboard_markdown.exists())

            manifest = json.loads(result.run_paths.manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["experiment"], "baseline-daily-selection")
            self.assertEqual(manifest["leaderboard_value"], 0.9123)

    def test_training_pipeline_runs_downstream_benchmark_with_run_local_model_context(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            spec_path = tmp_path / "training.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "benchmark-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                            "model_type": "randomforest",
                            "model_version": "demo-v1",
                        },
                        "benchmark_selection": {
                            "as_of": "2025-01-15",
                            "codes": ["600519", "000333"],
                            "top_k": 2,
                        },
                        "portfolio_backtest": {
                            "enabled": True,
                            "top_k": 2,
                            "score_threshold": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_training_runner(spec, run_paths):
                run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_paths.model_artifacts_dir / "model_demo-v1.pkl"
                metadata_path = run_paths.model_artifacts_dir / "metadata_demo-v1.json"
                model_path.write_text("demo-model", encoding="utf-8")
                metadata_path.write_text(json.dumps({"version": "demo-v1"}), encoding="utf-8")
                return TrainingRunResult(
                    summary={
                        "model_version": "demo-v1",
                        "model_type": spec.training.model_type,
                        "loaded_chan_count": 2,
                        "skipped_count": 0,
                    },
                    model_version="demo-v1",
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

            selection_calls = []

            def fake_selection_runner(spec, model_dir=None):
                selection_calls.append(
                    {
                        "as_of": spec.selection.as_of,
                        "model_version": spec.selection.model_version,
                        "model_dir": str(model_dir) if model_dir is not None else None,
                        "portfolio_top_k": spec.portfolio_backtest.top_k,
                    }
                )
                return SelectionRunResult(
                    recommendations=[],
                    summary={
                        "as_of": spec.selection.as_of,
                        "model_version": spec.selection.model_version,
                        "top_score": 0.88,
                        "avg_score": 0.81,
                        "recommendation_count": 2,
                        "skipped_count": 1,
                        "portfolio_backtest": {
                            "sharpe_ratio": 1.34,
                            "total_return": 0.12,
                        },
                    },
                )

            pipeline = AutoResearchPipeline(
                results_root=tmp_path / "results",
                selection_runner=fake_selection_runner,
                training_runner=fake_training_runner,
            )

            result = pipeline.run(spec_path)

            self.assertEqual(len(selection_calls), 1)
            self.assertEqual(selection_calls[0]["as_of"], "2025-01-15")
            self.assertEqual(selection_calls[0]["model_version"], "demo-v1")
            self.assertEqual(selection_calls[0]["model_dir"], str(result.run_paths.model_artifacts_dir))
            self.assertEqual(selection_calls[0]["portfolio_top_k"], 2)

            summary = json.loads(result.run_paths.summary_json.read_text(encoding="utf-8"))
            self.assertIn("downstream_benchmark", summary)
            self.assertEqual(summary["downstream_benchmark"]["top_score"], 0.88)

            manifest = json.loads(result.run_paths.manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["workflow"], "training")
            self.assertEqual(manifest["as_of"], "2025-01-15")
            self.assertEqual(manifest["leaderboard_metric"], "portfolio_sharpe")
            self.assertEqual(manifest["leaderboard_value"], 1.34)
            self.assertEqual(manifest["top_score"], 0.88)
            self.assertEqual(manifest["avg_score"], 0.81)
            self.assertEqual(manifest["recommendation_count"], 2)
            self.assertEqual(manifest["downstream_benchmark_summary"]["portfolio_backtest"]["sharpe_ratio"], 1.34)

    def test_training_pipeline_keeps_model_artifacts_inside_run_directory_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            spec_path = tmp_path / "training.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                            "model_type": "randomforest",
                            "model_version": "demo-v1",
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_training_runner(spec, run_paths):
                run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_paths.model_artifacts_dir / "model_demo-v1.pkl"
                metadata_path = run_paths.model_artifacts_dir / "metadata_demo-v1.json"
                model_path.write_text("demo-model", encoding="utf-8")
                metadata_path.write_text(json.dumps({"version": "demo-v1"}), encoding="utf-8")
                return TrainingRunResult(
                    summary={
                        "model_version": "demo-v1",
                        "model_type": spec.training.model_type,
                        "loaded_chan_count": 2,
                        "skipped_count": 0,
                    },
                    model_version="demo-v1",
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

            pipeline = AutoResearchPipeline(
                results_root=tmp_path / "results",
                training_runner=fake_training_runner,
                global_model_dir=tmp_path / "global-models",
            )

            result = pipeline.run(spec_path)

            self.assertTrue(result.run_paths.summary_json.exists())
            self.assertTrue(result.run_paths.manifest_json.exists())
            self.assertTrue(result.run_paths.model_artifacts_dir.joinpath("model_demo-v1.pkl").exists())
            self.assertFalse((tmp_path / "global-models" / "model_demo-v1.pkl").exists())
            self.assertEqual(result.manifest["workflow"], "training")
            self.assertIsNone(result.manifest.get("published_model_path"))

            manifest = json.loads(result.run_paths.manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_version"], "demo-v1")
            self.assertTrue((tmp_path / "results" / manifest["model_artifact_path"]).exists())
            self.assertEqual(
                (tmp_path / "results" / manifest["model_artifact_path"]).parent,
                result.run_paths.model_artifacts_dir,
            )

    def test_training_pipeline_can_publish_model_artifacts_to_global_models_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            spec_path = tmp_path / "training.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                            "model_type": "randomforest",
                            "model_version": "demo-v1",
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_training_runner(spec, run_paths):
                run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_paths.model_artifacts_dir / "model_demo-v1.pkl"
                metadata_path = run_paths.model_artifacts_dir / "metadata_demo-v1.json"
                model_path.write_text("demo-model", encoding="utf-8")
                metadata_path.write_text(json.dumps({"version": "demo-v1"}), encoding="utf-8")
                return TrainingRunResult(
                    summary={
                        "model_version": "demo-v1",
                        "model_type": spec.training.model_type,
                        "loaded_chan_count": 2,
                        "skipped_count": 0,
                    },
                    model_version="demo-v1",
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

            pipeline = AutoResearchPipeline(
                results_root=tmp_path / "results",
                training_runner=fake_training_runner,
                publish_model=True,
                global_model_dir=tmp_path / "global-models",
            )

            result = pipeline.run(spec_path)

            published_model_path = tmp_path / "global-models" / "model_demo-v1.pkl"
            published_metadata_path = tmp_path / "global-models" / "metadata_demo-v1.json"

            self.assertTrue(result.run_paths.model_artifacts_dir.joinpath("model_demo-v1.pkl").exists())
            self.assertTrue(published_model_path.exists())
            self.assertTrue(published_metadata_path.exists())
            self.assertEqual(result.manifest["published_model_path"], str(published_model_path))
            self.assertEqual(result.manifest["published_metadata_path"], str(published_metadata_path))

    def test_training_pipeline_runs_downstream_benchmark_with_published_model_context(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            spec_path = tmp_path / "training.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "benchmark-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                            "model_type": "randomforest",
                            "model_version": "demo-v1",
                        },
                        "benchmark_selection": {
                            "as_of": "2025-02-01",
                            "codes": ["600519", "000333"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_training_runner(spec, run_paths):
                run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_paths.model_artifacts_dir / "model_demo-v1.pkl"
                metadata_path = run_paths.model_artifacts_dir / "metadata_demo-v1.json"
                model_path.write_text("demo-model", encoding="utf-8")
                metadata_path.write_text(json.dumps({"version": "demo-v1"}), encoding="utf-8")
                return TrainingRunResult(
                    summary={
                        "model_version": "demo-v1",
                        "model_type": spec.training.model_type,
                        "loaded_chan_count": 2,
                        "skipped_count": 0,
                    },
                    model_version="demo-v1",
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

            selection_calls = []

            def fake_selection_runner(spec, model_dir=None):
                selection_calls.append(str(model_dir) if model_dir is not None else None)
                return SelectionRunResult(
                    recommendations=[],
                    summary={
                        "as_of": spec.selection.as_of,
                        "model_version": spec.selection.model_version,
                        "top_score": 0.71,
                        "avg_score": 0.69,
                        "recommendation_count": 1,
                        "skipped_count": 0,
                    },
                )

            pipeline = AutoResearchPipeline(
                results_root=tmp_path / "results",
                selection_runner=fake_selection_runner,
                training_runner=fake_training_runner,
                publish_model=True,
                global_model_dir=tmp_path / "global-models",
            )

            result = pipeline.run(spec_path)

            self.assertEqual(selection_calls, [str(tmp_path / "global-models")])
            self.assertTrue((tmp_path / "global-models" / "model_demo-v1.pkl").exists())
            self.assertEqual(result.manifest["model_version"], "demo-v1")
            self.assertEqual(result.manifest["recommendation_count"], 1)

    def test_training_pipeline_runs_multiple_downstream_benchmarks_and_aggregates_manifest_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_spec_path = tmp_path / "baseline_daily_selection.json"
            benchmark_spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-daily-selection",
                        "selection": {
                            "as_of": "2025-01-15",
                            "codes": ["600519", "000333"],
                            "top_k": 2,
                        },
                        "portfolio_backtest": {
                            "enabled": True,
                            "top_k": 2,
                            "score_threshold": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec_path = tmp_path / "training.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "benchmark-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                            "model_type": "randomforest",
                            "model_version": "demo-v1",
                        },
                        "benchmark_selections": [
                            {
                                "reference_path": "./baseline_daily_selection.json",
                                "weight": 2.0,
                            },
                            {
                                "name": "high-confidence-check",
                                "weight": 1.0,
                                "selection": {
                                    "as_of": "2025-02-01",
                                    "codes": ["600519", "000333"],
                                    "top_k": 1,
                                },
                            },
                        ],
                        "benchmark_suite_scoring": {
                            "dispersion_penalty": 0.5,
                            "failure_penalty": 0.6,
                        },
                        "portfolio_backtest": {
                            "enabled": True,
                            "top_k": 2,
                            "score_threshold": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_training_runner(spec, run_paths):
                run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_paths.model_artifacts_dir / "model_demo-v1.pkl"
                metadata_path = run_paths.model_artifacts_dir / "metadata_demo-v1.json"
                model_path.write_text("demo-model", encoding="utf-8")
                metadata_path.write_text(json.dumps({"version": "demo-v1"}), encoding="utf-8")
                return TrainingRunResult(
                    summary={
                        "model_version": "demo-v1",
                        "model_type": spec.training.model_type,
                        "loaded_chan_count": 2,
                        "skipped_count": 0,
                    },
                    model_version="demo-v1",
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

            selection_calls = []
            benchmark_summaries = {
                "2025-01-15": {
                    "top_score": 0.88,
                    "avg_score": 0.81,
                    "recommendation_count": 2,
                    "skipped_count": 1,
                    "portfolio_backtest": {
                        "sharpe_ratio": 1.34,
                        "total_return": 0.12,
                    },
                },
                "2025-02-01": {
                    "top_score": 0.73,
                    "avg_score": 0.69,
                    "recommendation_count": 1,
                    "skipped_count": 0,
                    "portfolio_backtest": {
                        "sharpe_ratio": 0.91,
                        "total_return": 0.07,
                    },
                },
            }

            def fake_selection_runner(spec, model_dir=None):
                selection_calls.append(
                    {
                        "name": spec.name,
                        "as_of": spec.selection.as_of,
                        "model_version": spec.selection.model_version,
                        "model_dir": str(model_dir) if model_dir is not None else None,
                        "portfolio_top_k": spec.portfolio_backtest.top_k,
                    }
                )
                benchmark_summary = benchmark_summaries[spec.selection.as_of]
                return SelectionRunResult(
                    recommendations=[],
                    summary={
                        "as_of": spec.selection.as_of,
                        "model_version": spec.selection.model_version,
                        **benchmark_summary,
                    },
                )

            pipeline = AutoResearchPipeline(
                results_root=tmp_path / "results",
                selection_runner=fake_selection_runner,
                training_runner=fake_training_runner,
            )

            result = pipeline.run(spec_path)

            self.assertEqual(len(selection_calls), 2)
            self.assertEqual(selection_calls[0]["as_of"], "2025-01-15")
            self.assertEqual(selection_calls[1]["as_of"], "2025-02-01")
            self.assertEqual(selection_calls[0]["model_version"], "demo-v1")
            self.assertEqual(selection_calls[1]["model_dir"], str(result.run_paths.model_artifacts_dir))

            summary = json.loads(result.run_paths.summary_json.read_text(encoding="utf-8"))
            self.assertEqual(len(summary["downstream_benchmark_results"]), 2)
            self.assertEqual(summary["downstream_benchmark_results"][0]["name"], "baseline-daily-selection")
            self.assertEqual(summary["downstream_benchmark_results"][1]["name"], "high-confidence-check")
            self.assertEqual(summary["downstream_benchmark_results"][0]["weight"], 2.0)
            self.assertEqual(summary["downstream_benchmark_results"][1]["weight"], 1.0)
            self.assertEqual(summary["downstream_benchmark_aggregate"]["leaderboard_metric"], "benchmark_suite_score_v2")
            self.assertAlmostEqual(summary["downstream_benchmark_aggregate"]["weighted_score_mean"], 1.1966666666666668)
            self.assertAlmostEqual(summary["downstream_benchmark_aggregate"]["score_dispersion"], 0.20270394394014365)
            self.assertAlmostEqual(summary["downstream_benchmark_aggregate"]["dispersion_penalty_value"], 0.10135197197007182)
            self.assertAlmostEqual(summary["downstream_benchmark_aggregate"]["failure_penalty_value"], 0.0)
            self.assertAlmostEqual(summary["downstream_benchmark_aggregate"]["leaderboard_value"], 1.095314694696595)

            manifest = json.loads(result.run_paths.manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["workflow"], "training")
            self.assertEqual(manifest["as_of"], "2025-01-15..2025-02-01")
            self.assertEqual(manifest["leaderboard_metric"], "benchmark_suite_score_v2")
            self.assertAlmostEqual(manifest["leaderboard_value"], 1.095314694696595)
            self.assertAlmostEqual(manifest["top_score"], 0.83)
            self.assertAlmostEqual(manifest["avg_score"], 0.77)
            self.assertEqual(manifest["recommendation_count"], 3)
            self.assertEqual(len(manifest["downstream_benchmark_results"]), 2)
            self.assertEqual(manifest["downstream_benchmark_summary"], None)

    def test_training_pipeline_continues_running_all_downstream_benchmarks_after_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            spec_path = tmp_path / "training.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "benchmark-model-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                            "model_type": "randomforest",
                            "model_version": "demo-v1",
                        },
                        "benchmark_selections": [
                            {
                                "name": "first-check",
                                "weight": 3.0,
                                "selection": {
                                    "as_of": "2025-01-15",
                                    "codes": ["600519", "000333"],
                                },
                            },
                            {
                                "name": "second-check",
                                "weight": 1.0,
                                "selection": {
                                    "as_of": "2025-02-01",
                                    "codes": ["600519", "000333"],
                                },
                            },
                        ],
                        "benchmark_suite_scoring": {
                            "dispersion_penalty": 0.5,
                            "failure_penalty": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_training_runner(spec, run_paths):
                run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_paths.model_artifacts_dir / "model_demo-v1.pkl"
                metadata_path = run_paths.model_artifacts_dir / "metadata_demo-v1.json"
                model_path.write_text("demo-model", encoding="utf-8")
                metadata_path.write_text(json.dumps({"version": "demo-v1"}), encoding="utf-8")
                return TrainingRunResult(
                    summary={
                        "model_version": "demo-v1",
                        "model_type": spec.training.model_type,
                        "loaded_chan_count": 2,
                        "skipped_count": 0,
                    },
                    model_version="demo-v1",
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

            selection_calls = []

            def fake_selection_runner(spec, model_dir=None):
                selection_calls.append(spec.selection.as_of)
                if spec.selection.as_of == "2025-01-15":
                    raise RuntimeError("benchmark failed")
                return SelectionRunResult(
                    recommendations=[],
                    summary={
                        "as_of": spec.selection.as_of,
                        "model_version": spec.selection.model_version,
                        "top_score": 0.73,
                        "avg_score": 0.69,
                        "recommendation_count": 1,
                        "skipped_count": 0,
                    },
                )

            pipeline = AutoResearchPipeline(
                results_root=tmp_path / "results",
                selection_runner=fake_selection_runner,
                training_runner=fake_training_runner,
            )

            result = pipeline.run(spec_path)

            self.assertEqual(selection_calls, ["2025-01-15", "2025-02-01"])

            summary = json.loads(result.run_paths.summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["downstream_benchmark_results"][0]["error"], "benchmark failed")
            self.assertEqual(summary["downstream_benchmark_results"][1]["summary"]["top_score"], 0.73)
            self.assertEqual(summary["downstream_benchmark_aggregate"]["successful_benchmark_count"], 1)
            self.assertAlmostEqual(summary["downstream_benchmark_aggregate"]["failure_weight_ratio"], 0.75)
            self.assertAlmostEqual(summary["downstream_benchmark_aggregate"]["failure_penalty_value"], 0.45)
            self.assertEqual(summary["downstream_benchmark_aggregate"]["leaderboard_metric"], "benchmark_suite_score_v2")
            self.assertAlmostEqual(summary["downstream_benchmark_aggregate"]["leaderboard_value"], 0.28)

            manifest = json.loads(result.run_paths.manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["downstream_benchmark_error"], "benchmark failed")
            self.assertEqual(manifest["leaderboard_metric"], "benchmark_suite_score_v2")
            self.assertAlmostEqual(manifest["leaderboard_value"], 0.28)


if __name__ == "__main__":
    unittest.main()
