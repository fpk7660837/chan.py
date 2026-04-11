from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .Spec import ExperimentSpec


@dataclass(frozen=True)
class SelectionRunResult:
    recommendations: List[Dict[str, Any]]
    summary: Dict[str, Any]


def _make_runtime_args(spec: ExperimentSpec) -> SimpleNamespace:
    selection = spec.selection
    return SimpleNamespace(
        model_version=selection.model_version,
        universe=selection.universe,
        top_k=selection.top_k,
        min_score=selection.min_score,
        signal_lookback_bars=selection.signal_lookback_bars,
        limit=selection.limit,
        codes=",".join(selection.codes) if selection.codes else None,
        codes_file=selection.codes_file,
    )


def _build_backtest_summary(predictor: Any, metadata: Dict[str, Any], chan_list: List, spec: ExperimentSpec) -> Dict[str, Any]:
    from Config.MLConfig import MLConfig
    from ML.Backtest.CrossSectionBacktest import CrossSectionBacktest

    config = MLConfig()
    saved_config = metadata.get("config", {}) if metadata else {}
    if "portfolio_backtest_config" in saved_config:
        config.portfolio_backtest_config.update(saved_config["portfolio_backtest_config"])

    config.portfolio_backtest_config.update(spec.portfolio_backtest.to_overrides())
    config.portfolio_backtest_config["direction"] = spec.selection.direction

    metrics = CrossSectionBacktest(predictor, config.portfolio_backtest_config).run(chan_list)
    return {
        key: float(metrics[key])
        for key in [
            "total_return",
            "mean_return",
            "std_return",
            "annualized_return",
            "max_drawdown",
            "sharpe_ratio",
            "calmar_ratio",
            "win_rate",
            "profit_loss_ratio",
            "total_trades",
            "periods",
            "avg_positions",
        ]
        if key in metrics
    }


def run_selection_experiment(spec: ExperimentSpec, model_dir: Optional[Path] = None) -> SelectionRunResult:
    from App.generate_stock_recommendations import (
        build_output_rows,
        load_chan_pool,
        load_model,
        load_universe,
        resolve_runtime_config,
    )
    from ML.FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
    from ML.Prediction.Predictor import Predictor

    runtime_args = _make_runtime_args(spec)
    model, metadata = load_model(spec.selection.model_version, model_dir=model_dir)
    runtime = resolve_runtime_config(metadata, runtime_args)
    predictor = Predictor(model, BSPFeatureExtractor(runtime["feature_config"]))

    universe = load_universe(runtime_args)
    if not universe:
        raise RuntimeError("Universe is empty.")

    as_of = datetime.strptime(spec.selection.as_of, "%Y-%m-%d")
    chan_list, code_name_map, skipped = load_chan_pool(
        universe=universe,
        as_of=as_of,
        history_days=spec.selection.history_days,
        stale_days=spec.selection.stale_days,
        universe_name=spec.selection.universe,
    )

    ranked = predictor.rank_stock_pool(
        chan_list,
        top_k=int(runtime["top_k"]),
        direction=spec.selection.direction,
        score_threshold=float(runtime["min_score"]) if runtime["min_score"] is not None else None,
        signal_lookback_bars=int(runtime["signal_lookback_bars"]),
    )
    rows = build_output_rows(ranked, code_name_map, as_of, metadata.get("version") if metadata else None)

    scores = [float(row["score"]) for row in rows]
    summary: Dict[str, Any] = {
        "as_of": spec.selection.as_of,
        "model_version": metadata.get("version", "latest") if metadata else "latest",
        "universe_size": len(universe),
        "loaded_chan_count": len(chan_list),
        "recommendation_count": len(rows),
        "skipped_count": len(skipped),
        "skipped": [{"code": code, "reason": reason} for code, reason in skipped[:20]],
        "top_score": max(scores) if scores else 0.0,
        "avg_score": mean(scores) if scores else 0.0,
        "runtime_config": {
            "top_k": int(runtime["top_k"]),
            "min_score": float(runtime["min_score"]) if runtime["min_score"] is not None else None,
            "signal_lookback_bars": int(runtime["signal_lookback_bars"]),
            "direction": spec.selection.direction,
            "history_days": spec.selection.history_days,
            "stale_days": spec.selection.stale_days,
        },
    }

    if spec.portfolio_backtest.enabled and chan_list:
        try:
            summary["portfolio_backtest"] = _build_backtest_summary(predictor, metadata, chan_list, spec)
        except Exception as exc:
            summary["portfolio_backtest_error"] = str(exc)

    return SelectionRunResult(recommendations=rows, summary=summary)
