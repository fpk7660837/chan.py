from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from Research.SignalReport import write_csv, write_json


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return normalized or "experiment"


@dataclass(frozen=True)
class RunPaths:
    root_dir: Path
    experiment_dir: Path
    run_dir: Path
    spec_snapshot_json: Path
    recommendations_csv: Path
    recommendations_json: Path
    summary_json: Path
    manifest_json: Path


class RunStorage:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def create_run(self, experiment_name: str, run_id: str = None) -> RunPaths:
        resolved_run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        experiment_dir = self.root_dir / "experiments" / _slugify(experiment_name)
        run_dir = experiment_dir / "runs" / resolved_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return RunPaths(
            root_dir=self.root_dir,
            experiment_dir=experiment_dir,
            run_dir=run_dir,
            spec_snapshot_json=run_dir / "spec.json",
            recommendations_csv=run_dir / "recommendations.csv",
            recommendations_json=run_dir / "recommendations.json",
            summary_json=run_dir / "summary.json",
            manifest_json=run_dir / "manifest.json",
        )

    def write_spec_snapshot(self, run_paths: RunPaths, spec_payload: Dict[str, Any]) -> None:
        write_json(run_paths.spec_snapshot_json, spec_payload)

    def write_recommendations(self, run_paths: RunPaths, rows: List[Dict[str, Any]]) -> None:
        write_csv(run_paths.recommendations_csv, rows)
        run_paths.recommendations_json.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_summary(self, run_paths: RunPaths, summary: Dict[str, Any]) -> None:
        write_json(run_paths.summary_json, summary)

    def write_manifest(self, run_paths: RunPaths, manifest: Dict[str, Any]) -> None:
        write_json(run_paths.manifest_json, manifest)

    def load_manifests(self) -> List[Dict[str, Any]]:
        manifests: List[Dict[str, Any]] = []
        if not self.root_dir.exists():
            return manifests
        for path in sorted(self.root_dir.glob("experiments/*/runs/*/manifest.json")):
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        return manifests
