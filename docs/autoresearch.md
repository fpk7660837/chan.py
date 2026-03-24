# AutoResearch For Stock Selection

This repo now has a lightweight `AutoResearch/` scaffold for running reproducible stock-selection experiments in the same codebase that already owns signal research and ML ranking.

## Goals

- Keep experiment setup declarative through JSON specs under `experiments/autoresearch/`
- Reuse the existing `ML/` predictor and cross-sectional backtest stack instead of creating another ranking path
- Reuse `Research/SignalReport.py` for artifact writing and leaderboard markdown generation
- Store run artifacts under `AutoResearch/results/` instead of `outputs/`

## Structure

```text
AutoResearch/
├── Spec.py                  # Dataclasses + JSON spec loader
├── Selection.py             # ML-backed stock selection execution
├── Storage.py               # Per-run artifact layout and manifest persistence
├── Leaderboard.py           # Leaderboard synthesis from run manifests
├── Pipeline.py              # End-to-end orchestration
└── results/                 # Default local artifact root (ignored)

App/run_autoresearch_pipeline.py
experiments/autoresearch/baseline_daily_selection.json
tests/test_autoresearch.py
```

## Execution Model

1. Load a spec from `experiments/autoresearch/*.json`
2. Resolve model/runtime settings through the existing recommendation helpers in `App/generate_stock_recommendations.py`
3. Rank the stock pool with `ML.Prediction.Predictor`
4. Optionally run `ML.Backtest.CrossSectionBacktest`
5. Persist:
   - `spec.json`
   - `recommendations.csv`
   - `recommendations.json`
   - `summary.json`
   - `manifest.json`
6. Rebuild a repository-local leaderboard from all successful manifests

## Why This Fits The Current Repo

- `Research/` remains responsible for reporting utilities and experiment-style output formatting.
- `ML/` remains responsible for prediction and backtesting logic.
- `AutoResearch/` is only orchestration, storage, and experiment bookkeeping.

That keeps the new layer thin and makes it a practical place to add future loops such as spec sweeps, model ablations, or signal-research prefilters without moving core logic again.

## Usage

Run every JSON spec in the default directory:

```bash
python3.11 App/run_autoresearch_pipeline.py
```

Run a specific spec:

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/baseline_daily_selection.json
```

Override local artifact storage:

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/baseline_daily_selection.json \
  --results-root ./tmp/autoresearch
```

## Next Extensions

- Add more experiment specs for top-k, threshold, and universe ablations
- Plug in `Research/SignalEvaluator` summaries before ranking to filter weak signal regimes
- Add model-version sweeps once multiple saved models are available
