import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from AutoResearch.Spec import ExperimentSpec, TrainingSpec
from AutoResearch.Storage import RunStorage
from AutoResearch.Training import run_training_experiment
from ML.Training.Trainer import Trainer


class FakeDiagnosticsModel:
    feature_names = ["f1"]

    def predict(self, X):
        mapping = {
            0: 0,
            1: 1,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
            6: 0,
        }
        return np.array([mapping[int(row[0])] for row in X], dtype=int)

    def predict_proba(self, X):
        mapping = {
            0: 0.01,
            1: 0.99,
            2: 0.02,
            3: 0.98,
            4: 0.80,
            5: 0.85,
            6: 0.20,
        }
        pos = np.array([mapping[int(row[0])] for row in X], dtype=float)
        return np.vstack([1.0 - pos, pos]).T

    def get_top_features(self, top_k=10):
        return [("f1", 1.0)]


class TrainerDiagnosticsTests(unittest.TestCase):
    def test_train_raises_when_sample_guard_detects_too_few_samples(self):
        trainer = Trainer(
            {
                "training_config": {
                    "min_total_samples": 100,
                    "min_train_samples": 50,
                    "min_test_samples": 20,
                    "enforce_min_samples": True,
                }
            }
        )
        X = np.arange(23, dtype=float).reshape(-1, 1)
        y = np.array([idx % 2 for idx in range(23)], dtype=int)

        with mock.patch.object(trainer, "_extract_bsp_from_chan_list", return_value=list(range(23))), mock.patch.object(
            trainer, "_extract_features", return_value=(X, ["f1"])
        ), mock.patch.object(
            trainer.label_builder,
            "build_labels",
            return_value=(y, np.zeros(len(y), dtype=float)),
        ), mock.patch.object(
            trainer, "_train_model"
        ) as train_model_mock:
            with self.assertRaisesRegex(ValueError, "Insufficient samples for reliable training"):
                trainer.train([object()])

        train_model_mock.assert_not_called()

    def test_train_records_classification_metrics_and_overfit_risk(self):
        trainer = Trainer(
            {
                "training_config": {
                    "min_total_samples": 1,
                    "min_train_samples": 1,
                    "min_test_samples": 1,
                    "enforce_min_samples": False,
                    "max_auc_gap": 0.1,
                    "max_f1_gap": 0.1,
                }
            }
        )
        full_X = np.arange(7, dtype=float).reshape(-1, 1)
        full_y = np.array([0, 1, 0, 1, 0, 1, 1], dtype=int)
        X_train = full_X[:4]
        X_test = full_X[4:]
        y_train = full_y[:4]
        y_test = full_y[4:]

        def split_data(_X, _y):
            trainer.last_split_info = {
                "mode": "manual",
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_splits": 1,
            }
            return X_train, X_test, y_train, y_test

        with mock.patch.object(trainer, "_extract_bsp_from_chan_list", return_value=list(range(7))), mock.patch.object(
            trainer, "_extract_features", return_value=(full_X, ["f1"])
        ), mock.patch.object(
            trainer.label_builder,
            "build_labels",
            return_value=(full_y, np.zeros(len(full_y), dtype=float)),
        ), mock.patch.object(
            trainer, "_split_data", side_effect=split_data
        ), mock.patch.object(
            trainer, "_train_model", return_value=FakeDiagnosticsModel()
        ):
            trainer.train([object()])

        self.assertEqual(trainer.last_dataset_profile["total_samples"], 7)
        self.assertEqual(trainer.last_dataset_profile["train_samples"], 4)
        self.assertEqual(trainer.last_dataset_profile["test_samples"], 3)
        self.assertIn("train", trainer.last_classification_metrics)
        self.assertIn("test", trainer.last_classification_metrics)
        self.assertIn("generalization_gap", trainer.last_classification_metrics)
        self.assertAlmostEqual(trainer.last_classification_metrics["train"]["auc"], 1.0)
        self.assertLess(trainer.last_classification_metrics["test"]["auc"], 1.0)
        self.assertGreater(trainer.last_classification_metrics["generalization_gap"]["f1"], 0.4)
        self.assertEqual(trainer.last_overfit_risk["level"], "high")
        self.assertTrue(any("auc gap" in reason for reason in trainer.last_overfit_risk["reasons"]))


class TrainingExperimentSummaryTests(unittest.TestCase):
    def test_run_training_experiment_writes_diagnostics_into_summary(self):
        spec = ExperimentSpec(
            name="diagnostic-training",
            mode="training",
            training=TrainingSpec(
                begin_time="2020-01-01",
                end_time="2022-12-31",
                codes=["600519", "000333", "600036"],
                model_type="lightgbm",
                model_version="demo-v1",
            ),
        )

        fake_trainer = mock.Mock()
        fake_trainer.last_split_info = {
            "mode": "walk_forward_last_fold",
            "train_size": 120,
            "test_size": 30,
            "n_splits": 5,
        }
        fake_trainer.last_dataset_profile = {
            "total_samples": 150,
            "train_samples": 120,
            "test_samples": 30,
            "positive_samples": 54,
            "negative_samples": 96,
            "positive_ratio": 0.36,
        }
        fake_trainer.last_classification_metrics = {
            "train": {"auc": 0.91, "f1": 0.82},
            "test": {"auc": 0.71, "f1": 0.58},
            "generalization_gap": {"auc": 0.20, "f1": 0.24},
        }
        fake_trainer.last_overfit_risk = {
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
        fake_trainer.train.return_value = object()

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

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_paths = RunStorage(tmp_path).create_run(spec.name)

            with mock.patch("AutoResearch.Training._resolve_training_universe", return_value=[("600519", ""), ("000333", "")]), mock.patch(
                "AutoResearch.Training._load_training_chan_pool",
                return_value=([object(), object()], []),
            ), mock.patch(
                "ML.Training.Trainer.Trainer", return_value=fake_trainer
            ), mock.patch(
                "ML.Utils.ModelIO.ModelIO", FakeModelIO
            ):
                result = run_training_experiment(spec, run_paths)

        self.assertEqual(result.summary["dataset_profile"]["total_samples"], 150)
        self.assertEqual(result.summary["classification_metrics"]["test"]["auc"], 0.71)
        self.assertEqual(result.summary["overfit_risk"]["level"], "high")
        self.assertEqual(result.summary["leaderboard_metric"], "test_auc")
        self.assertAlmostEqual(result.summary["leaderboard_value"], 0.71)

    def test_run_training_experiment_preserves_diagnostics_for_effective_auto_expanded_attempt(self):
        spec = ExperimentSpec(
            name="diagnostic-training",
            mode="training",
            training=TrainingSpec(
                begin_time="2020-01-01",
                end_time="2022-12-31",
                codes=["600519", "000333", "600036"],
                model_type="lightgbm",
                model_version="demo-v1",
            ),
        )

        first_trainer = mock.Mock()
        first_trainer.last_split_info = {}
        first_trainer.last_dataset_profile = {}
        first_trainer.last_classification_metrics = {}
        first_trainer.last_overfit_risk = {}
        first_trainer.train.side_effect = ValueError("Insufficient samples for reliable training: total=23 < 100")

        second_trainer = mock.Mock()
        second_trainer.last_split_info = {
            "mode": "walk_forward_last_fold",
            "train_size": 140,
            "test_size": 35,
            "n_splits": 5,
        }
        second_trainer.last_dataset_profile = {
            "total_samples": 175,
            "train_samples": 140,
            "test_samples": 35,
            "positive_samples": 61,
            "negative_samples": 114,
            "positive_ratio": 0.3486,
        }
        second_trainer.last_classification_metrics = {
            "train": {"auc": 0.89, "f1": 0.79},
            "test": {"auc": 0.74, "f1": 0.60},
            "generalization_gap": {"auc": 0.15, "f1": 0.19},
        }
        second_trainer.last_overfit_risk = {
            "level": "medium",
            "reasons": ["auc gap 0.15 exceeds limit 0.10"],
        }
        second_trainer.train.return_value = object()

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

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_paths = RunStorage(tmp_path).create_run(spec.name)

            with mock.patch(
                "AutoResearch.Training._resolve_training_universe",
                return_value=[("600519", ""), ("000333", ""), ("600036", "")],
            ), mock.patch(
                "AutoResearch.Training._load_training_chan_pool",
                return_value=([object(), object(), object()], []),
            ), mock.patch(
                "ML.Training.Trainer.Trainer",
                side_effect=[first_trainer, second_trainer],
            ), mock.patch(
                "ML.Utils.ModelIO.ModelIO",
                FakeModelIO,
            ):
                result = run_training_experiment(spec, run_paths)

        self.assertEqual(result.summary["effective_training_spec"]["begin_time"], "2018-01-01")
        self.assertEqual(result.summary["classification_metrics"]["test"]["auc"], 0.74)
        self.assertEqual(result.summary["overfit_risk"]["level"], "medium")
        self.assertEqual(result.summary["leaderboard_metric"], "test_auc")
        self.assertAlmostEqual(result.summary["leaderboard_value"], 0.74)


class TrainingLocalMarketDataTests(unittest.TestCase):
    def test_load_training_chan_pool_uses_local_sqlite_data_src_when_available(self):
        from AutoResearch import Training as training_mod

        fake_bar = mock.Mock()
        fake_chan = mock.MagicMock()
        fake_chan.__getitem__.return_value = mock.Mock(klu_iter=mock.Mock(return_value=iter([fake_bar])))

        with mock.patch(
            "App.generate_stock_recommendations.resolve_market_data_src",
            return_value="custom:SQLiteDailyBarAPI.CSQLiteDailyBarAPI",
        ), mock.patch(
            "Chan.CChan",
            return_value=fake_chan,
        ) as chan_cls:
            chan_list, skipped = training_mod._load_training_chan_pool(
                [("600519", "贵州茅台")],
                begin_time="2020-01-01",
                end_time="2022-12-31",
                universe_name="hs300",
            )

        self.assertEqual(len(chan_list), 1)
        self.assertEqual(skipped, [])
        self.assertEqual(chan_cls.call_args.kwargs["data_src"], "custom:SQLiteDailyBarAPI.CSQLiteDailyBarAPI")
