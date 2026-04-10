from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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
    universe: Optional[str] = None
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
class BenchmarkSelectionSpec:
    name: str
    selection: SelectionSpec
    portfolio_backtest: PortfolioBacktestSpec = field(default_factory=PortfolioBacktestSpec)
    reference_path: Optional[str] = None
    weight: float = 1.0


@dataclass(frozen=True)
class BenchmarkSuiteScoringSpec:
    dispersion_penalty: float = 0.15
    failure_penalty: float = 0.25


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
    benchmark_selections: List[BenchmarkSelectionSpec] = field(default_factory=list)
    benchmark_suite_scoring: BenchmarkSuiteScoringSpec = field(default_factory=BenchmarkSuiteScoringSpec)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    portfolio_backtest: PortfolioBacktestSpec = field(default_factory=PortfolioBacktestSpec)
    storage: StorageSpec = field(default_factory=StorageSpec)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def benchmark_selection(self) -> Optional[BenchmarkSelectionSpec]:
        return self.benchmark_selections[0] if self.benchmark_selections else None


@dataclass(frozen=True)
class SweepGridSpec:
    name: str
    path: str
    values: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class SweepSpec:
    grid: List[SweepGridSpec] = field(default_factory=list)
    variant_name_template: Optional[str] = None


@dataclass(frozen=True)
class TrainingSweepSpec:
    name: str
    mode: str = "training_sweep"
    training: Optional[TrainingSpec] = None
    benchmark_selections: List[BenchmarkSelectionSpec] = field(default_factory=list)
    benchmark_suite_scoring: BenchmarkSuiteScoringSpec = field(default_factory=BenchmarkSuiteScoringSpec)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    portfolio_backtest: PortfolioBacktestSpec = field(default_factory=PortfolioBacktestSpec)
    storage: StorageSpec = field(default_factory=StorageSpec)
    sweep: SweepSpec = field(default_factory=SweepSpec)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def benchmark_selection(self) -> Optional[BenchmarkSelectionSpec]:
        return self.benchmark_selections[0] if self.benchmark_selections else None


@dataclass(frozen=True)
class SweepVariantSpec:
    variant_id: str
    experiment: ExperimentSpec
    overrides: Dict[str, Any]


AutoResearchSpec = Union[ExperimentSpec, TrainingSweepSpec]


def _load_selection_spec_from_payload(selection_payload: Dict[str, Any], spec_path: Path) -> SelectionSpec:
    if "as_of" not in selection_payload:
        raise ValueError(f"Experiment spec must define selection.as_of: {spec_path}")

    return SelectionSpec(
        as_of=str(selection_payload["as_of"]),
        direction=str(selection_payload.get("direction", "buy")),
        model_version=selection_payload.get("model_version"),
        universe=str(selection_payload["universe"]).strip().lower() if selection_payload.get("universe") is not None else None,
        top_k=int(selection_payload.get("top_k", 10)),
        min_score=selection_payload.get("min_score", 0.6),
        signal_lookback_bars=int(selection_payload.get("signal_lookback_bars", 20)),
        history_days=int(selection_payload.get("history_days", 900)),
        stale_days=int(selection_payload.get("stale_days", 20)),
        codes=_normalize_codes(selection_payload.get("codes")),
        codes_file=selection_payload.get("codes_file"),
        limit=int(selection_payload["limit"]) if selection_payload.get("limit") is not None else None,
    )


def _load_selection_spec(payload: Dict[str, Any], spec_path: Path) -> SelectionSpec:
    if "selection" not in payload:
        raise ValueError(f"Experiment spec must define selection.as_of: {spec_path}")

    selection_payload = dict(payload.get("selection", {}))
    return _load_selection_spec_from_payload(selection_payload, spec_path)


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


def _load_portfolio_backtest_spec(payload: Dict[str, Any]) -> PortfolioBacktestSpec:
    portfolio_payload = dict(payload)
    return PortfolioBacktestSpec(
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


def _resolve_benchmark_reference_path(reference_path: str, spec_path: Path) -> Path:
    benchmark_spec_path = Path(reference_path)
    if not benchmark_spec_path.is_absolute():
        benchmark_spec_path = (spec_path.parent / benchmark_spec_path).resolve()
    return benchmark_spec_path


def _load_benchmark_suite_scoring_spec(payload: Dict[str, Any]) -> BenchmarkSuiteScoringSpec:
    suite_payload = dict(payload.get("benchmark_suite_scoring", {}))
    return BenchmarkSuiteScoringSpec(
        dispersion_penalty=float(suite_payload.get("dispersion_penalty", 0.15)),
        failure_penalty=float(suite_payload.get("failure_penalty", 0.25)),
    )


def _resolve_benchmark_weight(raw_weight: Any, spec_path: Path) -> float:
    weight = float(raw_weight if raw_weight is not None else 1.0)
    if weight <= 0.0:
        raise ValueError(f"benchmark_selection weight must be > 0: {spec_path}")
    return weight


def _resolve_benchmark_name(
    benchmark_payload: Dict[str, Any],
    *,
    reference_path: Optional[str],
    benchmark_spec_path: Path,
    index: int,
) -> str:
    raw_name = benchmark_payload.get("name")
    if raw_name is not None and str(raw_name).strip():
        return str(raw_name).strip()

    if reference_path:
        referenced_name = benchmark_payload.get("name")
        if referenced_name is not None and str(referenced_name).strip():
            return str(referenced_name).strip()
        return benchmark_spec_path.stem

    return f"benchmark-{index}"


def _load_single_benchmark_selection_spec(
    raw_benchmark: Any,
    spec_path: Path,
    default_portfolio_backtest: PortfolioBacktestSpec,
    index: int,
) -> BenchmarkSelectionSpec:
    benchmark_payload: Dict[str, Any]
    reference_path: Optional[str] = None
    weight: float = 1.0

    if isinstance(raw_benchmark, str):
        reference_path = raw_benchmark
        benchmark_spec_path = _resolve_benchmark_reference_path(raw_benchmark, spec_path)
        benchmark_payload = json.loads(benchmark_spec_path.read_text(encoding="utf-8"))
    elif isinstance(raw_benchmark, dict):
        raw_payload = dict(raw_benchmark)
        weight = _resolve_benchmark_weight(raw_payload.pop("weight", 1.0), spec_path)

        reference_override = raw_payload.pop("reference_path", None)
        if reference_override is not None:
            reference_path = str(reference_override)
            benchmark_spec_path = _resolve_benchmark_reference_path(reference_path, spec_path)
            benchmark_payload = json.loads(benchmark_spec_path.read_text(encoding="utf-8"))
            for key in ("name", "selection", "portfolio_backtest"):
                if key in raw_payload:
                    benchmark_payload[key] = raw_payload[key]
        else:
            benchmark_payload = raw_payload
            benchmark_spec_path = spec_path
    else:
        raise ValueError(f"benchmark_selection must be an object or a relative/absolute JSON path: {spec_path}")

    if not isinstance(raw_benchmark, dict):
        weight = _resolve_benchmark_weight(weight, spec_path)

    name = _resolve_benchmark_name(
        benchmark_payload,
        reference_path=reference_path,
        benchmark_spec_path=benchmark_spec_path,
        index=index,
    )
    benchmark_payload = dict(benchmark_payload)
    benchmark_payload.pop("name", None)

    if "selection" in benchmark_payload:
        selection_payload = dict(benchmark_payload.get("selection", {}))
        portfolio_payload = benchmark_payload.get("portfolio_backtest")
    else:
        selection_payload = benchmark_payload
        portfolio_payload = None

    selection = _load_selection_spec_from_payload(selection_payload, benchmark_spec_path)
    portfolio_backtest = (
        _load_portfolio_backtest_spec(portfolio_payload)
        if isinstance(portfolio_payload, dict)
        else default_portfolio_backtest
    )

    return BenchmarkSelectionSpec(
        name=name,
        selection=selection,
        portfolio_backtest=portfolio_backtest,
        reference_path=reference_path,
        weight=weight,
    )


def _load_benchmark_selection_specs(
    payload: Dict[str, Any],
    spec_path: Path,
    default_portfolio_backtest: PortfolioBacktestSpec,
) -> List[BenchmarkSelectionSpec]:
    has_single = payload.get("benchmark_selection") is not None
    has_multi = payload.get("benchmark_selections") is not None
    if has_single and has_multi:
        raise ValueError(f"Use only one of benchmark_selection or benchmark_selections: {spec_path}")

    raw_benchmarks: Any
    if has_multi:
        raw_benchmarks = payload.get("benchmark_selections")
    else:
        raw_benchmarks = payload.get("benchmark_selection")

    if raw_benchmarks is None:
        return []

    if isinstance(raw_benchmarks, list):
        benchmark_items = raw_benchmarks
    else:
        benchmark_items = [raw_benchmarks]

    return [
        _load_single_benchmark_selection_spec(item, spec_path, default_portfolio_backtest, index + 1)
        for index, item in enumerate(benchmark_items)
    ]


def _build_storage_spec(payload: Dict[str, Any]) -> StorageSpec:
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
    return storage


def _load_sweep_spec(payload: Dict[str, Any], spec_path: Path) -> SweepSpec:
    if "sweep" not in payload:
        raise ValueError(f"Training sweep spec must define sweep.grid: {spec_path}")

    sweep_payload = dict(payload.get("sweep", {}))
    raw_grid = sweep_payload.get("grid")
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError(f"Training sweep spec must define a non-empty sweep.grid: {spec_path}")

    grid: List[SweepGridSpec] = []
    seen_names = set()
    for index, raw_item in enumerate(raw_grid, 1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"sweep.grid entries must be objects: {spec_path}")

        path = str(raw_item.get("path", "")).strip()
        if not path:
            raise ValueError(f"sweep.grid[{index}] is missing required field 'path': {spec_path}")

        raw_values = raw_item.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"sweep.grid[{index}] must define a non-empty 'values' list: {spec_path}")

        name = str(raw_item.get("name") or path.split(".")[-1]).strip()
        if not name:
            raise ValueError(f"sweep.grid[{index}] resolved to an empty name: {spec_path}")
        if name in seen_names:
            raise ValueError(f"sweep.grid names must be unique ('{name}'): {spec_path}")
        seen_names.add(name)

        grid.append(
            SweepGridSpec(
                name=name,
                path=path,
                values=list(raw_values),
            )
        )

    return SweepSpec(
        grid=grid,
        variant_name_template=(
            str(sweep_payload["variant_name_template"])
            if sweep_payload.get("variant_name_template") is not None
            else None
        ),
    )


def _load_training_sweep_spec(payload: Dict[str, Any], spec_path: Path) -> TrainingSweepSpec:
    portfolio_backtest = _load_portfolio_backtest_spec(payload.get("portfolio_backtest", {}))
    training = _load_training_spec(payload, spec_path)
    benchmark_selections = _load_benchmark_selection_specs(payload, spec_path, portfolio_backtest)
    return TrainingSweepSpec(
        name=str(payload["name"]),
        description=str(payload.get("description", "")),
        tags=[str(tag) for tag in payload.get("tags", [])],
        training=training,
        benchmark_selections=benchmark_selections,
        benchmark_suite_scoring=_load_benchmark_suite_scoring_spec(payload),
        portfolio_backtest=portfolio_backtest,
        storage=_build_storage_spec(payload),
        sweep=_load_sweep_spec(payload, spec_path),
    )


def _load_experiment_spec_from_payload(payload: Dict[str, Any], spec_path: Path) -> ExperimentSpec:
    portfolio_backtest = _load_portfolio_backtest_spec(payload.get("portfolio_backtest", {}))
    benchmark_suite_scoring = _load_benchmark_suite_scoring_spec(payload)

    mode = str(payload.get("mode", "selection"))
    if mode == "selection":
        selection = _load_selection_spec(payload, spec_path)
        training = None
        benchmark_selections: List[BenchmarkSelectionSpec] = []
    elif mode == "training":
        selection = None
        training = _load_training_spec(payload, spec_path)
        benchmark_selections = _load_benchmark_selection_specs(payload, spec_path, portfolio_backtest)
    else:
        raise ValueError(f"Unsupported experiment mode '{mode}': {spec_path}")

    return ExperimentSpec(
        name=str(payload["name"]),
        mode=mode,
        description=str(payload.get("description", "")),
        tags=[str(tag) for tag in payload.get("tags", [])],
        selection=selection,
        training=training,
        benchmark_selections=benchmark_selections,
        benchmark_suite_scoring=benchmark_suite_scoring,
        portfolio_backtest=portfolio_backtest,
        storage=_build_storage_spec(payload),
    )


def load_experiment_spec(path: Path) -> ExperimentSpec:
    spec_path = Path(path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if "name" not in payload:
        raise ValueError(f"Experiment spec is missing required field 'name': {spec_path}")
    return _load_experiment_spec_from_payload(payload, spec_path)


def load_pipeline_spec(path: Path) -> AutoResearchSpec:
    spec_path = Path(path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if "name" not in payload:
        raise ValueError(f"Experiment spec is missing required field 'name': {spec_path}")

    mode = str(payload.get("mode", "selection"))
    if mode == "training_sweep":
        return _load_training_sweep_spec(payload, spec_path)
    return _load_experiment_spec_from_payload(payload, spec_path)


def _set_nested_value(payload: Dict[str, Any], path: str, value: Any) -> None:
    parts = [part.strip() for part in path.split(".") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid sweep override path '{path}'")

    cursor: Dict[str, Any] = payload
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set sweep override path '{path}' because '{part}' is not an object")
        cursor = next_value
    cursor[parts[-1]] = value


def _format_sweep_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    return str(value)


def expand_sweep_spec(spec: TrainingSweepSpec, spec_path: Optional[Path] = None) -> List[SweepVariantSpec]:
    if spec.mode != "training_sweep":
        raise ValueError(f"expand_sweep_spec requires mode=training_sweep, got {spec.mode}")

    base_payload = spec.to_dict()
    base_payload["mode"] = "training"
    base_payload.pop("sweep", None)
    resolve_path = Path(spec_path) if spec_path is not None else Path(spec.name)

    expanded: List[SweepVariantSpec] = []
    seen_names = set()
    for index, values in enumerate(product(*(grid.values for grid in spec.sweep.grid)), 1):
        overrides = {
            grid.path: value
            for grid, value in zip(spec.sweep.grid, values)
        }
        variant_context = {
            grid.name: _format_sweep_value(value)
            for grid, value in zip(spec.sweep.grid, values)
        }
        variant_id = "__".join(
            f"{grid.name}={_format_sweep_value(value)}"
            for grid, value in zip(spec.sweep.grid, values)
        )

        concrete_payload = json.loads(json.dumps(base_payload))
        for path, value in overrides.items():
            _set_nested_value(concrete_payload, path, value)

        if spec.sweep.variant_name_template:
            variant_name = spec.sweep.variant_name_template.format(name=spec.name, **variant_context)
        else:
            variant_name = f"{spec.name}-variant-{index:03d}"

        if variant_name in seen_names:
            raise ValueError(f"Sweep variant names must be unique, duplicate '{variant_name}'")
        seen_names.add(variant_name)
        concrete_payload["name"] = variant_name

        expanded.append(
            SweepVariantSpec(
                variant_id=variant_id,
                overrides=overrides,
                experiment=_load_experiment_spec_from_payload(concrete_payload, resolve_path),
            )
        )

    return expanded
