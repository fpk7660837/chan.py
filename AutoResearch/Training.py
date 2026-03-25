from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .Spec import ExperimentSpec, TrainingSpec
from .Storage import RunPaths


@dataclass(frozen=True)
class TrainingRunResult:
    summary: Dict[str, Any]
    model_version: str
    model_path: Path
    metadata_path: Optional[Path] = None


def _apply_training_config_overrides(config: Any, training: TrainingSpec) -> None:
    config.feature_config.update(training.feature_config)
    config.label_config.update(training.label_config)
    config.training_config.update(training.training_config)
    config.model_config["model_type"] = training.model_type

    param_key = f"{training.model_type}_params"
    model_params = dict(config.model_config.get(param_key, {}))
    model_params.update(training.model_params)
    config.model_config[param_key] = model_params


def _resolve_training_universe(training: TrainingSpec) -> List[Tuple[str, str]]:
    from App.generate_stock_recommendations import get_tradable_stocks, load_codes_from_file, normalize_code

    if training.codes:
        return [(normalize_code(code), "") for code in training.codes]
    if training.codes_file:
        return load_codes_from_file(Path(training.codes_file))
    return get_tradable_stocks(limit=training.limit)


def _load_training_chan_pool(
    universe: Sequence[Tuple[str, str]],
    *,
    begin_time: str,
    end_time: str,
) -> Tuple[List[Any], List[Tuple[str, str]]]:
    from Chan import CChan
    from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE

    chan_list: List[Any] = []
    skipped: List[Tuple[str, str]] = []

    for idx, (code, name) in enumerate(universe, 1):
        print(f"[{idx}/{len(universe)}] loading training data for {code} {name}".rstrip())
        try:
            chan = CChan(
                code=code,
                begin_time=begin_time,
                end_time=end_time,
                data_src=DATA_SRC.AKSHARE,
                lv_list=[KL_TYPE.K_DAY],
                autype=AUTYPE.QFQ,
            )
            bars = list(chan[0].klu_iter())
            if not bars:
                skipped.append((code, "no_data"))
                continue
            chan_list.append(chan)
        except Exception as exc:
            skipped.append((code, str(exc)))

    return chan_list, skipped


def run_training_experiment(spec: ExperimentSpec, run_paths: RunPaths) -> TrainingRunResult:
    if spec.training is None:
        raise ValueError("Training experiments require spec.training.")

    from Config.MLConfig import MLConfig
    from ML.Training.Trainer import Trainer
    from ML.Utils.ModelIO import ModelIO

    training = spec.training
    config = MLConfig()
    _apply_training_config_overrides(config, training)

    universe = _resolve_training_universe(training)
    if not universe:
        raise RuntimeError("Training universe is empty.")

    chan_list, skipped = _load_training_chan_pool(
        universe,
        begin_time=training.begin_time,
        end_time=training.end_time,
    )
    if not chan_list:
        raise RuntimeError("No training data could be loaded.")

    trainer = Trainer(config.to_dict())
    model = trainer.train(chan_list, model_type=training.model_type)

    run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_version = training.model_version or run_paths.run_dir.name

    metadata = {
        "description": spec.description or f"AutoResearch training run {spec.name}",
        "tags": spec.tags,
        "config": config.to_dict(),
        "training_spec": asdict(training),
        "autoresearch": {
            "experiment": spec.name,
            "mode": spec.mode,
            "run_id": run_paths.run_dir.name,
            "local_model_dir": str(run_paths.model_artifacts_dir),
        },
        "split_info": trainer.last_split_info,
    }

    model_io = ModelIO(str(run_paths.model_artifacts_dir))
    model_path = Path(model_io.save(model, version=model_version, metadata=metadata))
    metadata_path = run_paths.model_artifacts_dir / f"metadata_{model_version}.json"

    summary: Dict[str, Any] = {
        "model_version": model_version,
        "model_type": training.model_type,
        "begin_time": training.begin_time,
        "end_time": training.end_time,
        "universe_size": len(universe),
        "loaded_chan_count": len(chan_list),
        "skipped_count": len(skipped),
        "skipped": [{"code": code, "reason": reason} for code, reason in skipped[:20]],
        "split_info": trainer.last_split_info,
    }
    if trainer.last_split_info:
        summary["leaderboard_metric"] = "train_size"
        summary["leaderboard_value"] = int(trainer.last_split_info.get("train_size", 0) or 0)

    return TrainingRunResult(
        summary=summary,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path if metadata_path.exists() else None,
    )
