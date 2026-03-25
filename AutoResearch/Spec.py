from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_codes(raw_codes: Any) -> List[str]:
    if raw_codes is None:
        return []
    if isinstance(raw_codes, str):
        values = raw_codes.split(",")
    else:
        values = raw_codes
    return [str(code).strip() for code in values if str(code).strip()]


@dataclass(frozen=True)
class SelectionSpec:
    as_of: str
    direction: str = "buy"
    model_version: Optional[str] = None
    top_k: int = 10
    min_score: Optional[float] = 0.6
    signal_lookback_bars: int = 20
    history_days: int = 900
    stale_days: int = 20
    codes: List[str] = field(default_factory=list)
    codes_file: Optional[str] = None
    limit: Optional[int] = None


@dataclass(frozen=True)
class PortfolioBacktestSpec:
    enabled: bool = True
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None
    rebalance_bars: Optional[int] = None
    signal_lookback_bars: Optional[int] = None
    holding_period: Optional[int] = None
    min_positions: Optional[int] = None

    def to_overrides(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if key != "enabled" and value is not None}


@dataclass(frozen=True)
class TrainingSpec:
    begin_time: str
    end_time: str
    model_type: str = "lightgbm"
    model_version: Optional[str] = None
    codes: List[str] = field(default_factory=list)
    codes_file: Optional[str] = None
    limit: Optional[int] = None
    feature_config: Dict[str, Any] = field(default_factory=dict)
    label_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublishModelSpec:
    enabled: bool = False
    target_dir: str = "./models"


@dataclass(frozen=True)
class StorageSpec:
    root_dir: str = "AutoResearch/results"
    leaderboard_filename: str = "leaderboard.md"
    publish_model: PublishModelSpec = field(default_factory=PublishModelSpec)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    mode: str = "selection"
    selection: Optional[SelectionSpec] = None
    training: Optional[TrainingSpec] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    portfolio_backtest: PortfolioBacktestSpec = field(default_factory=PortfolioBacktestSpec)
    storage: StorageSpec = field(default_factory=StorageSpec)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_selection_spec(payload: Dict[str, Any], spec_path: Path) -> SelectionSpec:
    if "selection" not in payload or "as_of" not in payload["selection"]:
        raise ValueError(f"Experiment spec must define selection.as_of: {spec_path}")

    selection_payload = dict(payload.get("selection", {}))
    return SelectionSpec(
        as_of=str(selection_payload["as_of"]),
        direction=str(selection_payload.get("direction", "buy")),
        model_version=selection_payload.get("model_version"),
        top_k=int(selection_payload.get("top_k", 10)),
        min_score=selection_payload.get("min_score", 0.6),
        signal_lookback_bars=int(selection_payload.get("signal_lookback_bars", 20)),
        history_days=int(selection_payload.get("history_days", 900)),
        stale_days=int(selection_payload.get("stale_days", 20)),
        codes=_normalize_codes(selection_payload.get("codes")),
        codes_file=selection_payload.get("codes_file"),
        limit=int(selection_payload["limit"]) if selection_payload.get("limit") is not None else None,
    )


def _load_training_spec(payload: Dict[str, Any], spec_path: Path) -> TrainingSpec:
    if "training" not in payload:
        raise ValueError(f"Experiment spec must define training for mode=training: {spec_path}")

    training_payload = dict(payload.get("training", {}))
    if "begin_time" not in training_payload or "end_time" not in training_payload:
        raise ValueError(f"Training experiment spec must define training.begin_time and training.end_time: {spec_path}")

    return TrainingSpec(
        begin_time=str(training_payload["begin_time"]),
        end_time=str(training_payload["end_time"]),
        model_type=str(training_payload.get("model_type", "lightgbm")),
        model_version=training_payload.get("model_version"),
        codes=_normalize_codes(training_payload.get("codes")),
        codes_file=training_payload.get("codes_file"),
        limit=int(training_payload["limit"]) if training_payload.get("limit") is not None else None,
        feature_config=dict(training_payload.get("feature_config", {})),
        label_config=dict(training_payload.get("label_config", {})),
        training_config=dict(training_payload.get("training_config", {})),
        model_params=dict(training_payload.get("model_params", {})),
    )


def load_experiment_spec(path: Path) -> ExperimentSpec:
    spec_path = Path(path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))

    if "name" not in payload:
        raise ValueError(f"Experiment spec is missing required field 'name': {spec_path}")

    mode = str(payload.get("mode", "selection"))
    if mode == "selection":
        selection = _load_selection_spec(payload, spec_path)
        training = None
    elif mode == "training":
        selection = None
        training = _load_training_spec(payload, spec_path)
    else:
        raise ValueError(f"Unsupported experiment mode '{mode}': {spec_path}")

    portfolio_payload = dict(payload.get("portfolio_backtest", {}))
    portfolio_backtest = PortfolioBacktestSpec(
        enabled=bool(portfolio_payload.get("enabled", True)),
        top_k=int(portfolio_payload["top_k"]) if portfolio_payload.get("top_k") is not None else None,
        score_threshold=float(portfolio_payload["score_threshold"]) if portfolio_payload.get("score_threshold") is not None else None,
        rebalance_bars=int(portfolio_payload["rebalance_bars"]) if portfolio_payload.get("rebalance_bars") is not None else None,
        signal_lookback_bars=(
            int(portfolio_payload["signal_lookback_bars"])
            if portfolio_payload.get("signal_lookback_bars") is not None
            else None
        ),
        holding_period=int(portfolio_payload["holding_period"]) if portfolio_payload.get("holding_period") is not None else None,
        min_positions=int(portfolio_payload["min_positions"]) if portfolio_payload.get("min_positions") is not None else None,
    )

    storage_payload = dict(payload.get("storage", {}))
    publish_payload = storage_payload.get("publish_model", {})
    if isinstance(publish_payload, bool):
        publish_model = PublishModelSpec(enabled=publish_payload)
    else:
        publish_model_payload = dict(publish_payload)
        publish_model = PublishModelSpec(
            enabled=bool(publish_model_payload.get("enabled", False)),
            target_dir=str(publish_model_payload.get("target_dir", "./models")),
        )

    storage = StorageSpec(
        root_dir=str(storage_payload.get("root_dir", "AutoResearch/results")),
        leaderboard_filename=str(storage_payload.get("leaderboard_filename", "leaderboard.md")),
        publish_model=publish_model,
    )

    return ExperimentSpec(
        name=str(payload["name"]),
        mode=mode,
        description=str(payload.get("description", "")),
        tags=[str(tag) for tag in payload.get("tags", [])],
        selection=selection,
        training=training,
        portfolio_backtest=portfolio_backtest,
        storage=storage,
    )
