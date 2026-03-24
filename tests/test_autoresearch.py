import json
import tempfile
import unittest
from pathlib import Path

from AutoResearch.Leaderboard import build_leaderboard_rows, render_leaderboard_markdown
from AutoResearch.Pipeline import AutoResearchPipeline
from AutoResearch.Selection import SelectionRunResult
from AutoResearch.Spec import load_experiment_spec
from AutoResearch.Storage import RunStorage


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

            def fake_selection_runner(spec):
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


if __name__ == "__main__":
    unittest.main()
