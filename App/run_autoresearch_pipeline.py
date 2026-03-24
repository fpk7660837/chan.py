"""
AutoResearch experiment runner.

Examples:
    python3.11 App/run_autoresearch_pipeline.py
    python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_daily_selection.json
    python3.11 App/run_autoresearch_pipeline.py --results-root ./tmp/autoresearch
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AutoResearch.Pipeline import AutoResearchPipeline


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
        results_root=Path(args.results_root).resolve() if args.results_root else None
    )

    failed_runs = 0
    for spec_path in resolve_spec_paths(args):
        result = pipeline.run(spec_path)
        print(f"[{result.manifest['status']}] {spec_path}")
        print(f"  run: {result.run_paths.run_dir}")
        print(f"  recommendations: {result.manifest['recommendation_count']}")
        print(f"  leaderboard: {result.leaderboard_markdown}")
        if result.manifest["status"] != "completed":
            failed_runs += 1

    return 1 if failed_runs else 0


if __name__ == "__main__":
    raise SystemExit(main())
