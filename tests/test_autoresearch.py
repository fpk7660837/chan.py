import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from App import run_autoresearch_pipeline
from AutoResearch.Leaderboard import build_leaderboard_rows, render_leaderboard_markdown
from AutoResearch.Pipeline import AutoResearchPipeline, SweepIterationResult
from AutoResearch.Selection import SelectionRunResult
from AutoResearch.Spec import expand_sweep_spec, load_experiment_spec, load_pipeline_spec
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

    def test_load_hs300_daily_selection_spec(self):
        spec_path = Path("experiments/autoresearch/hs300_daily_selection.json")
        spec = load_experiment_spec(spec_path)

        self.assertEqual(spec.name, "hs300-daily-selection")
        self.assertEqual(spec.selection.as_of, "2026-04-10")
        self.assertEqual(spec.selection.universe, "hs300")
        self.assertEqual(spec.selection.codes, [])

    def test_load_experiment_spec_parses_named_universe_and_keeps_backward_compatible_codes_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path = Path(tmp_dir) / "selection.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "named-universe-selection",
                        "selection": {
                            "as_of": "2024-12-31",
                            "universe": "  HS300  ",
                            "codes": ["600519", "000333"],
                            "codes_file": "./codes.csv",
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(spec_path)

            self.assertEqual(spec.selection.universe, "hs300")
            self.assertEqual(spec.selection.codes, ["600519", "000333"])
            self.assertEqual(spec.selection.codes_file, "./codes.csv")

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

    def test_load_training_experiment_spec_preserves_multilevel_training_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path = Path(tmp_dir) / "training.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "multilevel-buy-training",
                        "mode": "training",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "universe": "hs300",
                            "training_config": {
                                "task_name": "buy_entry",
                                "decision_level": "30m",
                                "context_levels": ["day", "5m"],
                                "execution_level": "5m",
                                "exit_warning_confirmation_horizon_30m": 8,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec = load_experiment_spec(spec_path)

            self.assertEqual(spec.training.training_config["task_name"], "buy_entry")
            self.assertEqual(spec.training.training_config["decision_level"], "30m")
            self.assertEqual(spec.training.training_config["context_levels"], ["day", "5m"])
            self.assertEqual(spec.training.training_config["execution_level"], "5m")
            self.assertEqual(spec.training.training_config["exit_warning_confirmation_horizon_30m"], 8)

    def test_ml_config_defaults_include_multilevel_training_defaults(self):
        from Config.MLConfig import MLConfig

        config = MLConfig()

        self.assertEqual(config.feature_config["level_list"], ["day", "30m", "5m"])
        self.assertEqual(config.training_config["task_name"], "buy_entry")
        self.assertEqual(config.training_config["decision_level"], "30m")
        self.assertEqual(config.training_config["context_levels"], ["day", "5m"])
        self.assertEqual(config.training_config["execution_level"], "5m")
        self.assertEqual(config.training_config["exit_warning_confirmation_horizon_30m"], 8)

    def test_runtime_args_passes_universe_and_keeps_backward_compatible_codes_fields(self):
        from AutoResearch.Selection import _make_runtime_args
        from AutoResearch.Spec import ExperimentSpec, SelectionSpec

        spec = ExperimentSpec(
            name="runtime-args-selection",
            selection=SelectionSpec(
                as_of="2024-12-31",
                universe="hs300",
                codes=["600519", "000333"],
                codes_file="./codes.csv",
            ),
        )

        runtime_args = _make_runtime_args(spec)

        self.assertEqual(runtime_args.universe, "hs300")
        self.assertEqual(runtime_args.codes, "600519,000333")
        self.assertEqual(runtime_args.codes_file, "./codes.csv")

    def test_run_selection_experiment_supports_hs300_universe(self):
        from AutoResearch.Spec import ExperimentSpec, SelectionSpec
        from AutoResearch.Selection import run_selection_experiment

        spec = ExperimentSpec(
            name="hs300-daily-selection",
            selection=SelectionSpec(
                as_of="2024-12-31",
                universe="hs300",
                codes=[],
            ),
        )

        fake_model = object()
        fake_metadata = {
            "version": "demo-v1",
            "feature_config": {},
            "config": {},
        }
        fake_runtime = {
            "feature_config": {},
            "top_k": 10,
            "min_score": 0.6,
            "signal_lookback_bars": 20,
        }
        fake_universe = ["600519", "000333", "600036"]
        fake_chan_list = [SimpleNamespace(code="600519"), SimpleNamespace(code="000333")]
        fake_code_name_map = {"600519": "Kweichow Moutai", "000333": "Midea"}
        fake_rows = [
            {
                "rank": 1,
                "code": "600519",
                "name": "Kweichow Moutai",
                "score": 0.91,
                "signal_time": "2024-12-30",
                "signal_type": "1",
                "signal_price": 1788.0,
                "signal_idx": 123,
                "model_version": "demo-v1",
            }
        ]

        with mock.patch("App.generate_stock_recommendations.load_model", return_value=(fake_model, fake_metadata)) as load_model, \
            mock.patch("App.generate_stock_recommendations.resolve_runtime_config", return_value=fake_runtime) as resolve_runtime_config, \
            mock.patch("App.generate_stock_recommendations.load_universe", return_value=fake_universe) as load_universe, \
            mock.patch("App.generate_stock_recommendations.load_chan_pool", return_value=(fake_chan_list, fake_code_name_map, [])) as load_chan_pool, \
            mock.patch("App.generate_stock_recommendations.build_output_rows", return_value=fake_rows) as build_output_rows, \
            mock.patch("ML.FeatureEngine.BSPFeatureExtractor.BSPFeatureExtractor") as bsp_feature_extractor, \
            mock.patch("ML.Prediction.Predictor.Predictor") as predictor_cls:
            predictor = mock.Mock()
            predictor.rank_stock_pool.return_value = fake_rows
            predictor_cls.return_value = predictor
            bsp_feature_extractor.return_value = mock.Mock()

            result = run_selection_experiment(spec)

        self.assertEqual(load_model.call_count, 1)
        self.assertEqual(resolve_runtime_config.call_count, 1)
        self.assertEqual(load_universe.call_count, 1)
        self.assertEqual(load_universe.call_args.args[0].universe, "hs300")
        self.assertIsNone(load_universe.call_args.args[0].codes)
        self.assertIsNone(load_universe.call_args.args[0].codes_file)
        self.assertEqual(load_chan_pool.call_count, 1)
        self.assertEqual(load_chan_pool.call_args.kwargs["universe_name"], "hs300")
        self.assertEqual(result.summary["universe_size"], len(fake_universe))
        self.assertEqual(result.summary["recommendation_count"], len(fake_rows))

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

    def test_load_training_sweep_spec_expands_grid_into_concrete_training_variants(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path = Path(tmp_dir) / "training_sweep.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "baseline-model-training-sweep",
                        "mode": "training_sweep",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                            "model_type": "lightgbm",
                            "label_config": {
                                "threshold_pct": 0.05,
                            },
                        },
                        "benchmark_selection": {
                            "as_of": "2025-01-15",
                            "codes": ["600519", "000333"],
                            "top_k": 2,
                        },
                        "sweep": {
                            "variant_name_template": "{name}-{model_type}-thr{threshold_pct}",
                            "grid": [
                                {
                                    "name": "model_type",
                                    "path": "training.model_type",
                                    "values": ["lightgbm", "randomforest"],
                                },
                                {
                                    "name": "threshold_pct",
                                    "path": "training.label_config.threshold_pct",
                                    "values": [0.03, 0.05],
                                },
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec = load_pipeline_spec(spec_path)
            variants = expand_sweep_spec(spec)

            self.assertEqual(spec.mode, "training_sweep")
            self.assertEqual(len(variants), 4)
            self.assertEqual(variants[0].variant_id, "model_type=lightgbm__threshold_pct=0.03")
            self.assertEqual(variants[0].experiment.name, "baseline-model-training-sweep-lightgbm-thr0.03")
            self.assertEqual(variants[0].experiment.mode, "training")
            self.assertEqual(variants[0].experiment.training.model_type, "lightgbm")
            self.assertEqual(variants[0].experiment.training.label_config["threshold_pct"], 0.03)
            self.assertEqual(variants[-1].experiment.training.model_type, "randomforest")
            self.assertEqual(variants[-1].experiment.training.label_config["threshold_pct"], 0.05)
            self.assertEqual(variants[-1].experiment.benchmark_selection.selection.as_of, "2025-01-15")


class AutoResearchTrainingExpansionTests(unittest.TestCase):
    @staticmethod
    def _build_training_spec() -> object:
        from AutoResearch.Spec import ExperimentSpec, TrainingSpec

        return ExperimentSpec(
            name="auto-expanding-training",
            mode="training",
            training=TrainingSpec(
                begin_time="2020-01-01",
                end_time="2022-12-31",
                codes=["600519", "000333"],
                model_type="lightgbm",
                model_version="demo-v1",
            ),
        )

    @staticmethod
    def _make_trainer(*, train_side_effect=None, train_result=None, test_auc=0.71) -> mock.Mock:
        trainer = mock.Mock()
        trainer.last_split_info = {
            "mode": "walk_forward_last_fold",
            "train_size": 120,
            "test_size": 30,
            "n_splits": 5,
        }
        trainer.last_dataset_profile = {
            "total_samples": 150,
            "train_samples": 120,
            "test_samples": 30,
            "positive_samples": 54,
            "negative_samples": 96,
            "positive_ratio": 0.36,
        }
        trainer.last_classification_metrics = {
            "train": {"auc": 0.91, "f1": 0.82},
            "test": {"auc": test_auc, "f1": 0.58},
            "generalization_gap": {"auc": 0.20, "f1": 0.24},
        }
        trainer.last_overfit_risk = {
            "level": "high",
            "reasons": ["auc gap 0.20 exceeds limit 0.10"],
            "sample_guard": {
                "enforced": True,
                "min_total_samples": 100,
                "min_train_samples": 80,
                "min_test_samples": 20,
                "max_auc_gap": 0.1,
                "max_f1_gap": 0.15,
            },
        }
        if train_side_effect is not None:
            trainer.train.side_effect = train_side_effect
        else:
            trainer.train.return_value = train_result if train_result is not None else object()
        return trainer

    @staticmethod
    def _fake_model_io_factory():
        class FakeModelIO:
            def __init__(self, model_dir):
                self.model_dir = Path(model_dir)
                self.model_dir.mkdir(parents=True, exist_ok=True)

            def save(self, model, version=None, metadata=None):
                model_path = self.model_dir / f"model_{version}.pkl"
                metadata_path = self.model_dir / f"metadata_{version}.json"
                model_path.write_text("demo-model", encoding="utf-8")
                metadata_path.write_text(json.dumps(metadata or {}, ensure_ascii=False, indent=2), encoding="utf-8")
                return str(model_path)

        return FakeModelIO

    def test_run_training_experiment_retries_with_expanded_begin_time_after_sample_guard_failure(self):
        from AutoResearch.Training import run_training_experiment

        spec = self._build_training_spec()
        first_trainer = self._make_trainer(
            train_side_effect=ValueError("Insufficient samples for reliable training: total=23 < 100")
        )
        second_trainer = self._make_trainer(test_auc=0.73)

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_paths = RunStorage(Path(tmp_dir)).create_run(spec.name)

            with mock.patch(
                "AutoResearch.Training._resolve_training_universe",
                return_value=[("600519", ""), ("000333", "")],
            ), mock.patch(
                "AutoResearch.Training._load_training_chan_pool",
                return_value=([object(), object()], []),
            ), mock.patch(
                "ML.Training.Trainer.Trainer",
                side_effect=[first_trainer, second_trainer],
            ), mock.patch(
                "ML.Utils.ModelIO.ModelIO",
                self._fake_model_io_factory(),
            ):
                result = run_training_experiment(spec, run_paths)

        self.assertEqual([attempt["status"] for attempt in result.summary["training_attempts"]], ["failed", "completed"])
        self.assertEqual(result.summary["training_attempts"][0]["begin_time"], "2020-01-01")
        self.assertEqual(result.summary["training_attempts"][1]["begin_time"], "2018-01-01")
        self.assertEqual(result.summary["effective_training_spec"]["begin_time"], "2018-01-01")
        self.assertEqual(result.summary["original_training_spec"]["begin_time"], "2020-01-01")
        self.assertEqual(result.summary["auto_expansion"]["stopped_reason"], "recovered")

    def test_run_training_experiment_can_expand_from_explicit_codes_to_hs300(self):
        from AutoResearch.Training import run_training_experiment

        spec = self._build_training_spec()
        trainers = [
            self._make_trainer(train_side_effect=ValueError("Insufficient samples for reliable training: total=23 < 100")),
            self._make_trainer(train_side_effect=ValueError("Insufficient samples for reliable training: total=31 < 100")),
            self._make_trainer(train_side_effect=ValueError("Insufficient samples for reliable training: total=44 < 100")),
            self._make_trainer(test_auc=0.76),
        ]

        def resolve_universe(training):
            if getattr(training, "universe", None) == "hs300":
                return [("600519", "Kweichow Moutai"), ("000333", "Midea"), ("600036", "招商银行")]
            return [(code, "") for code in training.codes]

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_paths = RunStorage(Path(tmp_dir)).create_run(spec.name)

            with mock.patch(
                "AutoResearch.Training._resolve_training_universe",
                side_effect=resolve_universe,
            ), mock.patch(
                "AutoResearch.Training._load_training_chan_pool",
                return_value=([object(), object(), object()], []),
            ), mock.patch(
                "ML.Training.Trainer.Trainer",
                side_effect=trainers,
            ), mock.patch(
                "ML.Utils.ModelIO.ModelIO",
                self._fake_model_io_factory(),
            ):
                result = run_training_experiment(spec, run_paths)

        self.assertEqual(len(result.summary["training_attempts"]), 4)
        self.assertEqual(result.summary["training_attempts"][-1]["status"], "completed")
        self.assertEqual(result.summary["training_attempts"][-1]["universe_source"], "hs300")
        self.assertEqual(result.summary["effective_training_spec"]["universe"], "hs300")
        self.assertEqual(result.summary["effective_training_spec"]["codes"], [])
        self.assertEqual(result.summary["auto_expansion"]["stopped_reason"], "recovered")

    def test_run_training_experiment_uses_multilevel_loader_for_explicit_multilevel_task(self):
        from AutoResearch.Training import run_training_experiment

        base_spec = self._build_training_spec()
        spec = replace(
            base_spec,
            training=replace(
                base_spec.training,
                training_config={
                    "task_name": "buy_entry",
                    "decision_level": "30m",
                    "context_levels": ["day", "5m"],
                    "execution_level": "5m",
                },
            ),
        )
        fake_contexts = [object(), object()]
        fake_loader = mock.Mock()
        fake_loader.load_training_contexts.return_value = (fake_contexts, [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_paths = RunStorage(Path(tmp_dir)).create_run(spec.name)

            with mock.patch(
                "AutoResearch.Training._resolve_training_universe",
                return_value=[("600519", ""), ("000333", "")],
            ), mock.patch(
                "AutoResearch.Training._load_training_chan_pool",
                side_effect=AssertionError("daily-only loader should not be used"),
            ), mock.patch(
                "ML.Training.MultiLevelDataLoader.MultiLevelDataLoader",
                return_value=fake_loader,
            ), mock.patch(
                "ML.Training.Trainer.Trainer",
                return_value=self._make_trainer(),
            ) as trainer_cls, mock.patch(
                "ML.Utils.ModelIO.ModelIO",
                self._fake_model_io_factory(),
            ):
                result = run_training_experiment(spec, run_paths)

        self.assertEqual(result.summary["loaded_chan_count"], len(fake_contexts))
        self.assertEqual(fake_loader.load_training_contexts.call_count, 1)
        self.assertEqual(
            fake_loader.load_training_contexts.call_args.kwargs["levels"],
            ["day", "30m", "5m"],
        )
        self.assertEqual(result.summary["task_name"], "buy_entry")
        trainer_cls.return_value.train.assert_called_once()
        self.assertIs(trainer_cls.return_value.train.call_args.args[0], fake_contexts)

    def test_training_pipeline_records_all_attempts_when_auto_expansion_exhausts(self):
        spec = self._build_training_spec()
        pipeline = AutoResearchPipeline(results_root=Path(tempfile.mkdtemp()))
        trainers = [
            self._make_trainer(train_side_effect=ValueError("Insufficient samples for reliable training: total=23 < 100")),
            self._make_trainer(train_side_effect=ValueError("Insufficient samples for reliable training: total=31 < 100")),
            self._make_trainer(train_side_effect=ValueError("Insufficient samples for reliable training: total=44 < 100")),
            self._make_trainer(train_side_effect=ValueError("Insufficient samples for reliable training: total=52 < 100")),
            self._make_trainer(train_side_effect=ValueError("Insufficient samples for reliable training: total=60 < 100")),
        ]

        def resolve_universe(training):
            if getattr(training, "universe", None) == "hs300":
                return [("600519", "Kweichow Moutai"), ("000333", "Midea"), ("600036", "招商银行")]
            return [(code, "") for code in training.codes]

        with mock.patch(
            "AutoResearch.Training._resolve_training_universe",
            side_effect=resolve_universe,
        ), mock.patch(
            "AutoResearch.Training._load_training_chan_pool",
            return_value=([object(), object(), object()], []),
        ), mock.patch(
            "ML.Training.Trainer.Trainer",
            side_effect=trainers,
        ):
            result = pipeline.run_spec(spec)

        summary = json.loads(result.run_paths.summary_json.read_text(encoding="utf-8"))

        self.assertEqual(result.manifest["status"], "failed")
        self.assertEqual(summary["auto_expansion"]["stopped_reason"], "attempts_exhausted")
        self.assertEqual(len(summary["training_attempts"]), 5)
        self.assertEqual(summary["training_attempts"][-1]["universe_source"], "hs300")
        self.assertEqual(summary["effective_training_spec"]["universe"], "hs300")
        self.assertEqual(summary["effective_training_spec"]["begin_time"], "2018-01-01")
        self.assertEqual(summary["original_training_spec"]["codes"], ["600519", "000333"])


class AutoResearchProposalTests(unittest.TestCase):
    @staticmethod
    def _write_sweep_run(
        results_root: Path,
        *,
        sweep_slug: str,
        run_id: str,
        spec_payload: dict,
        summary_payload: dict,
    ) -> Path:
        run_dir = results_root / "sweeps" / sweep_slug / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "spec.json").write_text(json.dumps(spec_payload), encoding="utf-8")
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
        return summary_path

    @staticmethod
    def _build_training_sweep_payload(*, name: str, threshold_values: list[float]) -> dict:
        return {
            "name": name,
            "mode": "training_sweep",
            "training": {
                "begin_time": "2020-01-01",
                "end_time": "2022-12-31",
                "codes": ["600519", "000333"],
                "model_type": "lightgbm",
                "label_config": {
                    "threshold_pct": threshold_values[-1],
                },
            },
            "benchmark_selection": {
                "as_of": "2025-01-15",
                "codes": ["600519", "000333"],
                "top_k": 2,
            },
            "sweep": {
                "variant_name_template": "{name}-{model_type}-thr{threshold_pct}",
                "grid": [
                    {
                        "name": "model_type",
                        "path": "training.model_type",
                        "values": ["lightgbm", "randomforest"],
                    },
                    {
                        "name": "threshold_pct",
                        "path": "training.label_config.threshold_pct",
                        "values": threshold_values,
                    },
                ],
            },
        }

    def test_propose_next_sweep_uses_latest_summary_and_refines_numeric_winner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results_root = tmp_path / "results"
            generated_dir = tmp_path / "generated"
            older_spec = self._build_training_sweep_payload(
                name="older-training-sweep",
                threshold_values=[0.03, 0.05],
            )
            latest_spec = self._build_training_sweep_payload(
                name="baseline-model-training-sweep",
                threshold_values=[0.03, 0.05],
            )

            self._write_sweep_run(
                results_root,
                sweep_slug="older-training-sweep",
                run_id="20260325T120000Z",
                spec_payload=older_spec,
                summary_payload={
                    "name": "older-training-sweep",
                    "run_id": "20260325T120000Z",
                    "runs": [
                        {
                            "variant_id": "model_type=lightgbm__threshold_pct=0.05",
                            "status": "completed",
                            "leaderboard_value": 0.91,
                            "config": {
                                "training.model_type": "lightgbm",
                                "training.label_config.threshold_pct": 0.05,
                            },
                        }
                    ],
                },
            )

            latest_summary_path = self._write_sweep_run(
                results_root,
                sweep_slug="baseline-model-training-sweep",
                run_id="20260326T120000Z",
                spec_payload=latest_spec,
                summary_payload={
                    "name": "baseline-model-training-sweep",
                    "run_id": "20260326T120000Z",
                    "runs": [
                        {
                            "variant_id": "model_type=randomforest__threshold_pct=0.03",
                            "status": "completed",
                            "leaderboard_value": 2.21,
                            "config": {
                                "training.model_type": "randomforest",
                                "training.label_config.threshold_pct": 0.03,
                            },
                        },
                        {
                            "variant_id": "model_type=randomforest__threshold_pct=0.05",
                            "status": "completed",
                            "leaderboard_value": 2.18,
                            "config": {
                                "training.model_type": "randomforest",
                                "training.label_config.threshold_pct": 0.05,
                            },
                        },
                    ],
                },
            )

            pipeline = AutoResearchPipeline(results_root=results_root)

            result = pipeline.propose_next_sweep(output_dir=generated_dir, top_runs=1)

            self.assertEqual(result.source_summary_path, latest_summary_path.resolve())
            self.assertTrue(result.output_path.exists())
            self.assertEqual(result.output_path.parent, generated_dir.resolve())

            generated_spec = json.loads(result.output_path.read_text(encoding="utf-8"))
            grid_by_path = {item["path"]: item["values"] for item in generated_spec["sweep"]["grid"]}
            reloaded_spec = load_pipeline_spec(result.output_path)

            self.assertEqual(generated_spec["mode"], "training_sweep")
            self.assertEqual(generated_spec["training"]["model_type"], "randomforest")
            self.assertEqual(generated_spec["training"]["label_config"]["threshold_pct"], 0.03)
            self.assertEqual(grid_by_path["training.model_type"], ["randomforest"])
            self.assertEqual(grid_by_path["training.label_config.threshold_pct"], [0.02, 0.03, 0.04])
            self.assertEqual(reloaded_spec.benchmark_selection.selection.as_of, "2025-01-15")

    def test_propose_next_sweep_carries_multiple_categorical_winners_from_top_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results_root = tmp_path / "results"
            generated_dir = tmp_path / "generated"
            spec_payload = self._build_training_sweep_payload(
                name="baseline-model-training-sweep",
                threshold_values=[0.01, 0.03, 0.05],
            )
            summary_path = self._write_sweep_run(
                results_root,
                sweep_slug="baseline-model-training-sweep",
                run_id="20260326T120000Z",
                spec_payload=spec_payload,
                summary_payload={
                    "name": "baseline-model-training-sweep",
                    "run_id": "20260326T120000Z",
                    "runs": [
                        {
                            "variant_id": "model_type=lightgbm__threshold_pct=0.05",
                            "status": "completed",
                            "leaderboard_value": 1.52,
                            "config": {
                                "training.model_type": "lightgbm",
                                "training.label_config.threshold_pct": 0.05,
                            },
                        },
                        {
                            "variant_id": "model_type=randomforest__threshold_pct=0.03",
                            "status": "completed",
                            "leaderboard_value": 1.47,
                            "config": {
                                "training.model_type": "randomforest",
                                "training.label_config.threshold_pct": 0.03,
                            },
                        },
                        {
                            "variant_id": "model_type=randomforest__threshold_pct=0.01",
                            "status": "failed",
                            "leaderboard_value": None,
                            "config": {
                                "training.model_type": "randomforest",
                                "training.label_config.threshold_pct": 0.01,
                            },
                        },
                    ],
                },
            )

            pipeline = AutoResearchPipeline(results_root=results_root)

            result = pipeline.propose_next_sweep(
                summary_path=summary_path,
                output_dir=generated_dir,
                top_runs=2,
            )

            generated_spec = json.loads(result.output_path.read_text(encoding="utf-8"))
            grid_by_path = {item["path"]: item["values"] for item in generated_spec["sweep"]["grid"]}

            self.assertEqual(grid_by_path["training.model_type"], ["lightgbm", "randomforest"])
            self.assertEqual(grid_by_path["training.label_config.threshold_pct"], [0.02, 0.03, 0.04, 0.05, 0.06])

    def test_propose_next_sweep_makes_snapshot_based_benchmark_references_runnable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results_root = tmp_path / "results"
            generated_dir = tmp_path / "generated"
            spec_payload = {
                "name": "baseline-model-training-sweep",
                "mode": "training_sweep",
                "training": {
                    "begin_time": "2020-01-01",
                    "end_time": "2022-12-31",
                    "codes": ["600519", "000333"],
                    "model_type": "lightgbm",
                    "label_config": {
                        "threshold_pct": 0.05,
                    },
                },
                "benchmark_selections": [
                    {
                        "name": "baseline-daily-selection",
                        "selection": {
                            "as_of": "2025-01-15",
                            "codes": ["600519", "000333"],
                            "top_k": 2,
                        },
                        "portfolio_backtest": {
                            "enabled": True,
                        },
                        "reference_path": "./baseline_daily_selection.json",
                        "weight": 2.0,
                    }
                ],
                "sweep": {
                    "grid": [
                        {
                            "name": "model_type",
                            "path": "training.model_type",
                            "values": ["lightgbm", "randomforest"],
                        },
                        {
                            "name": "threshold_pct",
                            "path": "training.label_config.threshold_pct",
                            "values": [0.03, 0.05],
                        },
                    ],
                },
            }
            summary_path = self._write_sweep_run(
                results_root,
                sweep_slug="baseline-model-training-sweep",
                run_id="20260326T120000Z",
                spec_payload=spec_payload,
                summary_payload={
                    "name": "baseline-model-training-sweep",
                    "run_id": "20260326T120000Z",
                    "runs": [
                        {
                            "variant_id": "model_type=randomforest__threshold_pct=0.03",
                            "status": "completed",
                            "leaderboard_value": 2.21,
                            "config": {
                                "training.model_type": "randomforest",
                                "training.label_config.threshold_pct": 0.03,
                            },
                        }
                    ],
                },
            )

            pipeline = AutoResearchPipeline(results_root=results_root)
            result = pipeline.propose_next_sweep(
                summary_path=summary_path,
                output_dir=generated_dir,
                top_runs=1,
            )

            reloaded_spec = load_pipeline_spec(result.output_path)

            self.assertEqual(reloaded_spec.benchmark_selection.reference_path, None)
            self.assertEqual(reloaded_spec.benchmark_selection.selection.as_of, "2025-01-15")

    def test_iterate_next_sweep_can_generate_and_execute_next_round(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results_root = tmp_path / "results"
            generated_dir = tmp_path / "generated"
            spec_payload = self._build_training_sweep_payload(
                name="baseline-model-training-sweep",
                threshold_values=[0.03, 0.05],
            )
            summary_path = self._write_sweep_run(
                results_root,
                sweep_slug="baseline-model-training-sweep",
                run_id="20260326T120000Z",
                spec_payload=spec_payload,
                summary_payload={
                    "name": "baseline-model-training-sweep",
                    "run_id": "20260326T120000Z",
                    "runs": [
                        {
                            "variant_id": "model_type=randomforest__threshold_pct=0.03",
                            "status": "completed",
                            "leaderboard_value": 2.21,
                            "config": {
                                "training.model_type": "randomforest",
                                "training.label_config.threshold_pct": 0.03,
                            },
                        }
                    ],
                },
            )

            training_calls = []

            def fake_training_runner(spec, run_paths):
                threshold = float(spec.training.label_config["threshold_pct"])
                training_calls.append((spec.training.model_type, threshold))
                run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_paths.model_artifacts_dir / f"model_{spec.name}.pkl"
                metadata_path = run_paths.model_artifacts_dir / f"metadata_{spec.name}.json"
                model_path.write_text("demo-model", encoding="utf-8")
                metadata_path.write_text(json.dumps({"version": spec.name}), encoding="utf-8")
                return TrainingRunResult(
                    summary={
                        "model_version": spec.name,
                        "model_type": spec.training.model_type,
                        "loaded_chan_count": 2,
                        "skipped_count": 0,
                    },
                    model_version=spec.name,
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

            def fake_selection_runner(spec, model_dir=None):
                match = re.search(r"thr([0-9.]+)$", spec.selection.model_version)
                threshold = float(match.group(1)) if match is not None else 0.0
                sharpe_ratio = {
                    0.02: 1.11,
                    0.03: 1.44,
                    0.04: 1.18,
                }[round(threshold, 2)]
                return SelectionRunResult(
                    recommendations=[],
                    summary={
                        "as_of": spec.selection.as_of,
                        "model_version": spec.selection.model_version,
                        "top_score": sharpe_ratio - 0.2,
                        "avg_score": sharpe_ratio - 0.3,
                        "recommendation_count": 2,
                        "skipped_count": 0,
                        "portfolio_backtest": {
                            "sharpe_ratio": sharpe_ratio,
                        },
                    },
                )

            pipeline = AutoResearchPipeline(
                results_root=results_root,
                selection_runner=fake_selection_runner,
                training_runner=fake_training_runner,
            )

            result = pipeline.iterate_next_sweep(
                summary_path=summary_path,
                output_dir=generated_dir,
                top_runs=1,
                execute=True,
            )

            self.assertTrue(result.proposal.output_path.exists())
            self.assertIsNotNone(result.execution_result)
            self.assertEqual(result.proposal.source_summary_path, summary_path.resolve())
            self.assertEqual(
                training_calls,
                [("randomforest", 0.02), ("randomforest", 0.03), ("randomforest", 0.04)],
            )

            generated_spec = json.loads(result.proposal.output_path.read_text(encoding="utf-8"))
            grid_by_path = {item["path"]: item["values"] for item in generated_spec["sweep"]["grid"]}
            self.assertEqual(generated_spec["training"]["model_type"], "randomforest")
            self.assertEqual(grid_by_path["training.label_config.threshold_pct"], [0.02, 0.03, 0.04])

            execution_result = result.execution_result
            self.assertEqual(execution_result.spec.name, generated_spec["name"])
            self.assertEqual(execution_result.summary["variant_count"], 3)
            self.assertEqual(
                execution_result.summary["best_run"]["config"]["training.label_config.threshold_pct"],
                0.03,
            )
            self.assertTrue(execution_result.sweep_paths.summary_json.exists())


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

    def test_training_sweep_pipeline_runs_all_variants_and_writes_sweep_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            spec_path = tmp_path / "training_sweep.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "name": "benchmark-model-training-sweep",
                        "mode": "training_sweep",
                        "training": {
                            "begin_time": "2020-01-01",
                            "end_time": "2022-12-31",
                            "codes": ["600519", "000333"],
                            "model_type": "lightgbm",
                        },
                        "benchmark_selection": {
                            "as_of": "2025-01-15",
                            "codes": ["600519", "000333"],
                            "top_k": 2,
                        },
                        "sweep": {
                            "variant_name_template": "{name}-{model_type}",
                            "grid": [
                                {
                                    "name": "model_type",
                                    "path": "training.model_type",
                                    "values": ["lightgbm", "randomforest"],
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            training_calls = []

            def fake_training_runner(spec, run_paths):
                training_calls.append(spec.training.model_type)
                run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
                model_path = run_paths.model_artifacts_dir / f"model_{spec.training.model_type}.pkl"
                metadata_path = run_paths.model_artifacts_dir / f"metadata_{spec.training.model_type}.json"
                model_path.write_text("demo-model", encoding="utf-8")
                metadata_path.write_text(json.dumps({"version": spec.name}), encoding="utf-8")
                return TrainingRunResult(
                    summary={
                        "model_version": spec.name,
                        "model_type": spec.training.model_type,
                        "loaded_chan_count": 2,
                        "skipped_count": 0,
                    },
                    model_version=spec.name,
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

            def fake_selection_runner(spec, model_dir=None):
                sharpe_ratio = 1.42 if spec.selection.model_version.endswith("lightgbm") else 0.97
                top_score = 0.91 if spec.selection.model_version.endswith("lightgbm") else 0.74
                return SelectionRunResult(
                    recommendations=[],
                    summary={
                        "as_of": spec.selection.as_of,
                        "model_version": spec.selection.model_version,
                        "top_score": top_score,
                        "avg_score": top_score - 0.05,
                        "recommendation_count": 2,
                        "skipped_count": 0,
                        "portfolio_backtest": {
                            "sharpe_ratio": sharpe_ratio,
                        },
                    },
                )

            pipeline = AutoResearchPipeline(
                results_root=tmp_path / "results",
                selection_runner=fake_selection_runner,
                training_runner=fake_training_runner,
            )

            result = pipeline.run(spec_path)

            self.assertEqual(training_calls, ["lightgbm", "randomforest"])
            self.assertEqual(len(result.variant_results), 2)
            self.assertTrue(result.sweep_paths.summary_json.exists())
            self.assertTrue(result.sweep_paths.leaderboard_markdown.exists())

            summary = json.loads(result.sweep_paths.summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["variant_count"], 2)
            self.assertEqual(summary["best_run"]["experiment"], "benchmark-model-training-sweep-lightgbm")
            self.assertEqual(summary["best_run"]["leaderboard_metric"], "portfolio_sharpe")
            self.assertAlmostEqual(summary["best_run"]["leaderboard_value"], 1.42)
            self.assertEqual(summary["runs"][0]["config"]["training.model_type"], "lightgbm")
            self.assertEqual(summary["runs"][0]["status"], "completed")
            self.assertTrue((tmp_path / "results" / summary["runs"][0]["manifest_path"]).exists())

            leaderboard = result.sweep_paths.leaderboard_markdown.read_text(encoding="utf-8")
            self.assertIn("benchmark-model-training-sweep-lightgbm", leaderboard)
            self.assertIn("training.model_type=lightgbm", leaderboard)


class AutoResearchCliTests(unittest.TestCase):
    def test_main_can_generate_and_execute_next_round_sweep(self):
        proposal = SimpleNamespace(
            source_summary_path=Path("/tmp/source-summary.json"),
            source_spec_path=Path("/tmp/source-spec.json"),
            selected_runs=[{"variant_id": "winner"}],
            output_path=Path("/tmp/generated-spec.json"),
        )
        execution_result = SimpleNamespace(
            sweep_paths=SimpleNamespace(run_dir=Path("/tmp/results/sweeps/generated/run-1")),
            summary={
                "variant_count": 3,
                "completed_variant_count": 3,
                "failed_variant_count": 0,
                "best_run": {
                    "experiment": "generated-sweep-randomforest-thr0.03",
                    "leaderboard_metric": "portfolio_sharpe",
                    "leaderboard_value": 1.44,
                },
            },
            leaderboard_markdown=Path("/tmp/results/sweeps/generated/run-1/leaderboard.md"),
        )
        iteration_result = SweepIterationResult(
            proposal=proposal,
            execution_result=execution_result,
        )

        stdout = io.StringIO()
        with mock.patch.object(
            run_autoresearch_pipeline.AutoResearchPipeline,
            "iterate_next_sweep",
            return_value=iteration_result,
        ) as iterate_mock:
            with mock.patch(
                "sys.argv",
                [
                    "run_autoresearch_pipeline.py",
                    "--generate-next-sweep",
                    "--execute-generated-sweep",
                    "--sweep-summary",
                    "/tmp/source-summary.json",
                    "--generated-spec-dir",
                    "/tmp/generated",
                    "--proposal-top-runs",
                    "1",
                    "--results-root",
                    "/tmp/results",
                ],
            ):
                with redirect_stdout(stdout):
                    exit_code = run_autoresearch_pipeline.main()

        self.assertEqual(exit_code, 0)
        iterate_mock.assert_called_once()
        self.assertEqual(iterate_mock.call_args.kwargs["execute"], True)
        self.assertEqual(iterate_mock.call_args.kwargs["top_runs"], 1)
        self.assertEqual(iterate_mock.call_args.kwargs["output_dir"], Path("/tmp/generated").resolve())
        self.assertEqual(iterate_mock.call_args.kwargs["summary_path"], Path("/tmp/source-summary.json").resolve())

        output = stdout.getvalue()
        self.assertIn("[proposal] generated next-round training sweep", output)
        self.assertIn("[sweep] executed generated next-round training sweep", output)
        self.assertIn("best: generated-sweep-randomforest-thr0.03 (portfolio_sharpe=1.44)", output)


if __name__ == "__main__":
    unittest.main()
