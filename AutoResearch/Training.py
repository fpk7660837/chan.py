from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .Spec import ExperimentSpec, TrainingSpec
from .Storage import RunPaths


AUTO_EXPANSION_STRATEGY = "time_then_hs300_phase1"


@dataclass(frozen=True)
class TrainingRunResult:
    summary: Dict[str, Any]
    model_version: str
    model_path: Path
    metadata_path: Optional[Path] = None


class TrainingExperimentError(RuntimeError):
    def __init__(self, message: str, *, summary: Dict[str, Any]):
        super().__init__(message)
        self.summary = summary


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
    from App.generate_stock_recommendations import (
        get_tradable_stocks,
        load_codes_from_file,
        load_named_universe,
        normalize_code,
        normalize_universe_name,
    )

    if training.codes:
        return [(normalize_code(code), "") for code in training.codes]
    if training.codes_file:
        return load_codes_from_file(Path(training.codes_file))
    if training.universe is not None and str(training.universe).strip():
        named_universe = load_named_universe(normalize_universe_name(training.universe))
        if training.limit is not None:
            return named_universe[: training.limit]
        return named_universe
    return get_tradable_stocks(limit=training.limit)


def _load_training_chan_pool(
    universe: Sequence[Tuple[str, str]],
    *,
    begin_time: str,
    end_time: str,
    universe_name: Optional[str] = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
) -> Tuple[List[Any], List[Tuple[str, str]]]:
    from Chan import CChan
    from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
    from App.generate_stock_recommendations import resolve_market_data_src

    chan_list: List[Any] = []
    skipped: List[Tuple[str, str]] = []
    data_src: DATA_SRC | str = resolve_market_data_src(universe_name)

    for idx, (code, name) in enumerate(universe, 1):
        print(f"[{idx}/{len(universe)}] loading training data for {code} {name}".rstrip())
        last_error: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                chan = CChan(
                    code=code,
                    begin_time=begin_time,
                    end_time=end_time,
                    data_src=data_src,
                    lv_list=[KL_TYPE.K_DAY],
                    autype=AUTYPE.QFQ,
                )
                bars = list(chan[0].klu_iter())
                if not bars:
                    last_error = "no_data"
                else:
                    chan_list.append(chan)
                    last_error = None
                    break
            except Exception as exc:
                last_error = str(exc)

            if attempt < max_retries:
                print(f"  retry {attempt}/{max_retries - 1} for {code}: {last_error}")
                time.sleep(retry_delay_seconds * attempt)

        if last_error is not None:
            skipped.append((code, last_error))

    return chan_list, skipped


def _normalize_training_universe_name(universe: Optional[str]) -> Optional[str]:
    if universe is None or not str(universe).strip():
        return None

    from App.generate_stock_recommendations import normalize_universe_name

    return normalize_universe_name(str(universe))


def _resolve_training_universe_source(training: TrainingSpec) -> str:
    if training.codes:
        return "codes"
    if training.codes_file:
        return "codes_file"
    normalized_universe = _normalize_training_universe_name(training.universe)
    if normalized_universe is not None:
        return normalized_universe
    return "fallback"


def _shift_begin_time(begin_time: str, *, years: int) -> str:
    current = datetime.strptime(begin_time, "%Y-%m-%d")
    shifted_year = current.year + years
    day = current.day
    while day > 0:
        try:
            shifted = current.replace(year=shifted_year, day=day)
            return shifted.strftime("%Y-%m-%d")
        except ValueError:
            day -= 1
    raise ValueError(f"Unable to shift begin_time={begin_time} by {years} years")


def _training_attempt_signature(training: TrainingSpec) -> Tuple[Any, ...]:
    return (
        training.begin_time,
        training.end_time,
        training.model_type,
        training.model_version,
        _normalize_training_universe_name(training.universe),
        tuple(training.codes),
        training.codes_file,
        training.limit,
    )


def _build_training_attempts(training: TrainingSpec) -> List[TrainingSpec]:
    attempts: List[TrainingSpec] = []
    seen: set[Tuple[Any, ...]] = set()

    def add_attempt(candidate: TrainingSpec) -> None:
        signature = _training_attempt_signature(candidate)
        if signature in seen:
            return
        seen.add(signature)
        attempts.append(candidate)

    add_attempt(training)
    add_attempt(replace(training, begin_time=_shift_begin_time(training.begin_time, years=-2)))
    add_attempt(replace(training, begin_time=_shift_begin_time(training.begin_time, years=-4)))

    hs300_training = replace(training, universe="hs300", codes=[], codes_file=None, limit=None)
    add_attempt(hs300_training)
    add_attempt(replace(hs300_training, begin_time=_shift_begin_time(training.begin_time, years=-2)))
    return attempts


def _is_explicit_multilevel_task(training: TrainingSpec) -> bool:
    task_name = str(training.training_config.get("task_name", "") or "").strip().lower()
    if task_name not in {"buy_entry", "exit_warning"}:
        return False
    return any(
        key in training.training_config
        for key in ("decision_level", "context_levels", "execution_level")
    )


def _resolve_multilevel_levels(config: Any) -> List[str]:
    requested = set()

    decision_level = str(config.training_config.get("decision_level", "") or "").strip().lower()
    if decision_level:
        requested.add(decision_level)

    for level in config.training_config.get("context_levels", []) or []:
        normalized = str(level or "").strip().lower()
        if normalized:
            requested.add(normalized)

    execution_level = str(config.training_config.get("execution_level", "") or "").strip().lower()
    if execution_level:
        requested.add(execution_level)

    configured_levels = [
        str(level).strip().lower()
        for level in (config.feature_config.get("level_list", []) or [])
        if str(level).strip()
    ]
    ordered = [level for level in configured_levels if level in requested]
    for level in requested:
        if level not in ordered:
            ordered.append(level)
    return ordered


def _is_insufficient_samples_error(exc: Exception) -> bool:
    return "Insufficient samples for reliable training" in str(exc)


def _build_attempt_record(
    training: TrainingSpec,
    *,
    attempt_index: int,
    universe: Sequence[Tuple[str, str]],
    chan_list: Sequence[Any],
    skipped: Sequence[Tuple[str, str]],
    status: str,
    sample_guard_passed: Optional[bool],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "attempt_index": attempt_index,
        "begin_time": training.begin_time,
        "end_time": training.end_time,
        "universe_source": _resolve_training_universe_source(training),
        "codes_count": len(training.codes),
        "codes_file": training.codes_file,
        "limit": training.limit,
        "auto_expansion_applied": attempt_index > 1,
        "universe_size": len(universe),
        "loaded_chan_count": len(chan_list),
        "skipped_count": len(skipped),
        "sample_guard_passed": sample_guard_passed,
        "status": status,
    }
    normalized_universe = _normalize_training_universe_name(training.universe)
    if normalized_universe is not None:
        record["universe"] = normalized_universe
    if error is not None:
        record["error"] = error
    return record


def _build_auto_expansion_summary(
    attempts: Sequence[Dict[str, Any]],
    *,
    stopped_reason: str,
) -> Dict[str, Any]:
    return {
        "triggered": any(bool(attempt.get("auto_expansion_applied")) for attempt in attempts),
        "strategy": AUTO_EXPANSION_STRATEGY,
        "final_attempt_index": len(attempts),
        "stopped_reason": stopped_reason,
    }


def _build_training_failure_summary(
    original_training: TrainingSpec,
    *,
    run_paths: RunPaths,
    attempts: Sequence[Dict[str, Any]],
    last_attempt_training: Optional[TrainingSpec],
    last_skipped: Sequence[Tuple[str, str]],
    error: str,
    stopped_reason: str,
) -> Dict[str, Any]:
    last_attempt = attempts[-1] if attempts else {}
    return {
        "model_version": original_training.model_version or run_paths.run_dir.name,
        "model_type": original_training.model_type,
        "begin_time": original_training.begin_time,
        "end_time": original_training.end_time,
        "universe_size": int(last_attempt.get("universe_size", 0) or 0),
        "loaded_chan_count": int(last_attempt.get("loaded_chan_count", 0) or 0),
        "skipped_count": int(last_attempt.get("skipped_count", 0) or 0),
        "skipped": [{"code": code, "reason": reason} for code, reason in list(last_skipped)[:20]],
        "error": error,
        "training_attempts": list(attempts),
        "original_training_spec": asdict(original_training),
        "effective_training_spec": asdict(last_attempt_training) if last_attempt_training is not None else {},
        "auto_expansion": _build_auto_expansion_summary(attempts, stopped_reason=stopped_reason),
    }


def run_training_experiment(spec: ExperimentSpec, run_paths: RunPaths) -> TrainingRunResult:
    if spec.training is None:
        raise ValueError("Training experiments require spec.training.")

    from Config.MLConfig import MLConfig
    from ML.Training.MultiLevelDataLoader import MultiLevelDataLoader
    from ML.Training.Trainer import Trainer
    from ML.Utils.ModelIO import ModelIO

    training = spec.training
    attempts = _build_training_attempts(training)
    attempt_records: List[Dict[str, Any]] = []

    model = None
    trainer = None
    config = None
    effective_training: Optional[TrainingSpec] = None
    effective_universe: Sequence[Tuple[str, str]] = []
    effective_chan_list: Sequence[Any] = []
    effective_skipped: Sequence[Tuple[str, str]] = []

    for attempt_index, attempt_training in enumerate(attempts, 1):
        attempt_universe: Sequence[Tuple[str, str]] = []
        attempt_chan_list: Sequence[Any] = []
        attempt_skipped: Sequence[Tuple[str, str]] = []
        try:
            config = MLConfig()
            _apply_training_config_overrides(config, attempt_training)

            attempt_universe = _resolve_training_universe(attempt_training)
            if not attempt_universe:
                raise RuntimeError("Training universe is empty.")

            if _is_explicit_multilevel_task(attempt_training):
                data_loader = MultiLevelDataLoader()
                attempt_chan_list, attempt_skipped = data_loader.load_training_contexts(
                    attempt_universe,
                    begin_time=attempt_training.begin_time,
                    end_time=attempt_training.end_time,
                    levels=_resolve_multilevel_levels(config),
                    universe_name=attempt_training.universe,
                )
            else:
                attempt_chan_list, attempt_skipped = _load_training_chan_pool(
                    attempt_universe,
                    begin_time=attempt_training.begin_time,
                    end_time=attempt_training.end_time,
                    universe_name=attempt_training.universe,
                )
            if not attempt_chan_list:
                raise RuntimeError("No training data could be loaded.")

            trainer = Trainer(config.to_dict())
            model = trainer.train(attempt_chan_list, model_type=attempt_training.model_type)
        except Exception as exc:
            sample_guard_failed = _is_insufficient_samples_error(exc)
            attempt_records.append(
                _build_attempt_record(
                    attempt_training,
                    attempt_index=attempt_index,
                    universe=attempt_universe,
                    chan_list=attempt_chan_list,
                    skipped=attempt_skipped,
                    status="failed",
                    sample_guard_passed=False if sample_guard_failed else None,
                    error=str(exc),
                )
            )
            if sample_guard_failed and attempt_index < len(attempts):
                continue

            raise TrainingExperimentError(
                str(exc),
                summary=_build_training_failure_summary(
                    training,
                    run_paths=run_paths,
                    attempts=attempt_records,
                    last_attempt_training=attempt_training,
                    last_skipped=attempt_skipped,
                    error=str(exc),
                    stopped_reason="attempts_exhausted" if sample_guard_failed else "non_recoverable_error",
                ),
            ) from exc

        attempt_records.append(
            _build_attempt_record(
                attempt_training,
                attempt_index=attempt_index,
                universe=attempt_universe,
                chan_list=attempt_chan_list,
                skipped=attempt_skipped,
                status="completed",
                sample_guard_passed=True,
            )
        )
        effective_training = attempt_training
        effective_universe = attempt_universe
        effective_chan_list = attempt_chan_list
        effective_skipped = attempt_skipped
        break

    if effective_training is None or model is None or trainer is None or config is None:
        raise TrainingExperimentError(
            "Training did not produce a model.",
            summary=_build_training_failure_summary(
                training,
                run_paths=run_paths,
                attempts=attempt_records,
                last_attempt_training=None,
                last_skipped=[],
                error="Training did not produce a model.",
                stopped_reason="attempts_exhausted",
            ),
        )

    run_paths.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_version = effective_training.model_version or run_paths.run_dir.name
    auto_expansion = _build_auto_expansion_summary(
        attempt_records,
        stopped_reason="recovered" if len(attempt_records) > 1 else "original_succeeded",
    )

    metadata = {
        "description": spec.description or f"AutoResearch training run {spec.name}",
        "tags": spec.tags,
        "config": config.to_dict(),
        "training_spec": asdict(effective_training),
        "original_training_spec": asdict(training),
        "effective_training_spec": asdict(effective_training),
        "training_attempts": attempt_records,
        "auto_expansion": auto_expansion,
        "autoresearch": {
            "experiment": spec.name,
            "mode": spec.mode,
            "run_id": run_paths.run_dir.name,
            "local_model_dir": str(run_paths.model_artifacts_dir),
        },
        "split_info": trainer.last_split_info,
        "training_diagnostics": {
            "dataset_profile": trainer.last_dataset_profile,
            "classification_metrics": trainer.last_classification_metrics,
            "overfit_risk": trainer.last_overfit_risk,
        },
    }

    model_io = ModelIO(str(run_paths.model_artifacts_dir))
    model_path = Path(model_io.save(model, version=model_version, metadata=metadata))
    metadata_path = run_paths.model_artifacts_dir / f"metadata_{model_version}.json"

    summary: Dict[str, Any] = {
        "task_name": str(config.training_config.get("task_name", "") or ""),
        "model_version": model_version,
        "model_type": effective_training.model_type,
        "begin_time": effective_training.begin_time,
        "end_time": effective_training.end_time,
        "universe_size": len(effective_universe),
        "loaded_chan_count": len(effective_chan_list),
        "skipped_count": len(effective_skipped),
        "skipped": [{"code": code, "reason": reason} for code, reason in list(effective_skipped)[:20]],
        "original_training_spec": asdict(training),
        "effective_training_spec": asdict(effective_training),
        "training_attempts": attempt_records,
        "auto_expansion": auto_expansion,
        "split_info": trainer.last_split_info,
        "dataset_profile": trainer.last_dataset_profile,
        "classification_metrics": trainer.last_classification_metrics,
        "overfit_risk": trainer.last_overfit_risk,
    }
    test_metrics = trainer.last_classification_metrics.get("test", {})
    if "auc" in test_metrics:
        summary["leaderboard_metric"] = "test_auc"
        summary["leaderboard_value"] = float(test_metrics["auc"])
    elif trainer.last_split_info:
        summary["leaderboard_metric"] = "train_size"
        summary["leaderboard_value"] = int(trainer.last_split_info.get("train_size", 0) or 0)

    return TrainingRunResult(
        summary=summary,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path if metadata_path.exists() else None,
    )
