"""
AutoResearch experiment runner.

Examples:
    python3.11 App/run_autoresearch_pipeline.py
    python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_daily_selection.json
    python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_model_training.json
    python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_model_training_sweep.json
    python3.11 App/run_autoresearch_pipeline.py --generate-next-sweep
    python3.11 App/run_autoresearch_pipeline.py --generate-next-sweep --sweep-summary AutoResearch/results/sweeps/.../summary.json
    python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_model_training.json --publish-model
    python3.11 App/run_autoresearch_pipeline.py --results-root ./tmp/autoresearch
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AutoResearch.Pipeline import AutoResearchPipeline, SweepRunResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoResearch stock-selection experiments")
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        help="Path to a JSON experiment spec. Repeat to run multiple specs.",
    )
    parser.add_argument(
        "--spec-dir",
        default=str(ROOT / "experiments" / "autoresearch"),
        help="Directory scanned for *.json specs when --spec is omitted.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Override the storage root declared in experiment specs.",
    )
    parser.add_argument(
        "--publish-model",
        "--promote-model",
        action="store_true",
        dest="publish_model",
        help="For training specs, also copy the chosen model artifact into the shared global model directory.",
    )
    parser.add_argument(
        "--publish-model-dir",
        default=None,
        help="Override the publish/promote target directory used with --publish-model.",
    )
    parser.add_argument(
        "--generate-next-sweep",
        action="store_true",
        help="Generate a new training_sweep spec by refining around the strongest prior sweep runs.",
    )
    parser.add_argument(
        "--sweep-summary",
        default=None,
        help="Optional path to a prior sweep summary.json. Defaults to the latest summary under results_root/sweeps/.",
    )
    parser.add_argument(
        "--generated-spec-dir",
        default=str(ROOT / "experiments" / "autoresearch" / "generated"),
        help="Directory where generated training sweep specs are written.",
    )
    parser.add_argument(
        "--proposal-top-runs",
        type=int,
        default=2,
        help="Number of top completed runs used when generating the next-round sweep proposal.",
    )
    return parser.parse_args()


def resolve_spec_paths(args: argparse.Namespace) -> list[Path]:
    if args.spec:
        return [Path(path).resolve() for path in args.spec]

    spec_dir = Path(args.spec_dir).resolve()
    spec_paths = sorted(spec_dir.glob("*.json"))
    if not spec_paths:
        raise RuntimeError(f"No experiment specs found under {spec_dir}")
    return spec_paths


def main() -> int:
    args = parse_args()
    pipeline = AutoResearchPipeline(
        results_root=Path(args.results_root).resolve() if args.results_root else None,
        publish_model=(args.publish_model or bool(args.publish_model_dir)) or None,
        global_model_dir=Path(args.publish_model_dir).resolve() if args.publish_model_dir else None,
    )

    if args.generate_next_sweep:
        proposal = pipeline.propose_next_sweep(
            summary_path=Path(args.sweep_summary).resolve() if args.sweep_summary else None,
            output_dir=Path(args.generated_spec_dir).resolve(),
            top_runs=args.proposal_top_runs,
        )
        print("[proposal] generated next-round training sweep")
        print(f"  source summary: {proposal.source_summary_path}")
        print(f"  source spec: {proposal.source_spec_path}")
        print(f"  selected runs: {len(proposal.selected_runs)}")
        print(f"  output spec: {proposal.output_path}")
        return 0

    failed_runs = 0
    for spec_path in resolve_spec_paths(args):
        result = pipeline.run(spec_path)
        if isinstance(result, SweepRunResult):
            print(f"[sweep] {spec_path}")
            print(f"  run: {result.sweep_paths.run_dir}")
            print(f"  variants: {result.summary['variant_count']}")
            print(f"  completed: {result.summary['completed_variant_count']}")
            print(f"  failed: {result.summary['failed_variant_count']}")
            best_run = result.summary.get("best_run")
            if isinstance(best_run, dict):
                print(f"  best: {best_run.get('experiment')} ({best_run.get('leaderboard_metric')}={best_run.get('leaderboard_value')})")
            print(f"  leaderboard: {result.leaderboard_markdown}")
            if int(result.summary.get("failed_variant_count", 0) or 0) > 0:
                failed_runs += 1
            continue

        print(f"[{result.manifest['status']}] {spec_path}")
        print(f"  run: {result.run_paths.run_dir}")
        print(f"  recommendations: {result.manifest['recommendation_count']}")
        print(f"  leaderboard: {result.leaderboard_markdown}")
        if result.manifest.get("published_model_path"):
            print(f"  published model: {result.manifest['published_model_path']}")
        if result.manifest["status"] != "completed":
            failed_runs += 1

    return 1 if failed_runs else 0


if __name__ == "__main__":
    raise SystemExit(main())
