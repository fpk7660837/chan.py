# AutoResearch For Stock Selection And Training

This repo now has a lightweight `AutoResearch/` scaffold for running reproducible stock-selection and model-training experiments in the same codebase that already owns signal research and ML ranking.

## Goals

- Keep experiment setup declarative through JSON specs under `experiments/autoresearch/`
- Reuse the existing `ML/` predictor and cross-sectional backtest stack instead of creating another ranking path
- Reuse the existing `ML/Training/Trainer.py` and `ML/Utils/ModelIO.py` stack instead of inventing another training path
- Reuse `Research/SignalReport.py` for artifact writing and leaderboard markdown generation
- Store run artifacts under `AutoResearch/results/` instead of `outputs/`

## Structure

```text
AutoResearch/
├── Spec.py                  # Dataclasses + JSON spec loader
├── Selection.py             # ML-backed stock selection execution
├── Training.py              # ML training orchestration + run-local model artifacts
├── Storage.py               # Per-run artifact layout and manifest persistence
├── Leaderboard.py           # Leaderboard synthesis from run manifests
├── Pipeline.py              # End-to-end orchestration
└── results/                 # Default local artifact root (ignored)

App/run_autoresearch_pipeline.py
experiments/autoresearch/baseline_daily_selection.json
tests/test_autoresearch.py
```

## Execution Model

### Selection Mode

1. Load a selection spec from `experiments/autoresearch/*.json`
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

### Training Mode

1. Load a training spec from `experiments/autoresearch/*.json`
2. Build the training universe from `codes`, `codes_file`, or the existing tradable-stock helper
3. Train through `ML.Training.Trainer`
4. Save the model and metadata into `AutoResearch/results/.../runs/<id>/models/`
5. Optionally publish/promote the chosen artifact into the shared global `./models` directory
6. If downstream benchmark selections are configured, run each fixed downstream selection benchmark against the trained model context
7. Persist:
   - `spec.json`
   - `summary.json`
   - `manifest.json`
   - `models/model_<version>.pkl`
   - `models/metadata_<version>.json`

## Why This Fits The Current Repo

- `Research/` remains responsible for reporting utilities and experiment-style output formatting.
- `ML/` remains responsible for prediction and backtesting logic.
- `ML/` remains responsible for training and model serialization logic.
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

Run a training spec and keep the trained model local to the run directory:

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/baseline_model_training.json
```

Override local artifact storage:

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/baseline_daily_selection.json \
  --results-root ./tmp/autoresearch
```

Publish/promote a training artifact into the shared global model directory:

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/baseline_model_training.json \
  --publish-model
```

Override the publish target:

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/baseline_model_training.json \
  --publish-model-dir ./tmp/models
```

## Storage Boundary

- Default behavior: training artifacts stay inside the run directory under `AutoResearch/results/experiments/<experiment>/runs/<run_id>/models/`
- Optional publish/promote behavior: enable `storage.publish_model.enabled` in the spec, or pass `--publish-model`, to also copy the chosen model artifact into the global `./models` directory
- Optional benchmark behavior:
  - preferred: set `benchmark_selections` in a training spec to a list of inline selection configs and/or paths to existing selection spec JSON files
  - backward compatible: `benchmark_selection` still works for a single downstream benchmark
  - each inline benchmark may define an optional `name`; referenced JSON specs default to their spec `name` field, or their filename stem when no name is present
  - each benchmark may define an optional `weight` (default `1.0`)
  - referenced benchmark specs may also be expressed as objects with `reference_path` so weight and future suite-local overrides can live next to the reference
  - a training spec may define `benchmark_suite_scoring` with:
    - `dispersion_penalty` to subtract `dispersion_penalty * weighted_stddev(component_score)`
    - `failure_penalty` to subtract `failure_penalty * failed_weight_ratio`
- When downstream benchmarks are configured, the training run executes every configured benchmark immediately after training:
  - it uses the run-local model artifacts by default
  - it uses the published model directory when publish/promote is enabled
  - `summary.json` stores the full benchmark run list under `downstream_benchmark_results`
  - `summary.json` stores aggregate cross-benchmark metrics under `downstream_benchmark_aggregate`
  - `manifest.json` stores the full benchmark run list under `downstream_benchmark_results`
  - `manifest.json` stores the aggregate benchmark view under `downstream_benchmark_aggregate`
  - legacy single-benchmark fields (`downstream_benchmark` in `summary.json`, `downstream_benchmark_summary` in `manifest.json`) remain populated when only one benchmark is configured
  - top-level manifest leaderboard fields (`as_of`, `leaderboard_metric`, `leaderboard_value`, `top_score`, `avg_score`, `recommendation_count`) reflect the aggregate downstream benchmark view so training runs can be ranked across multiple post-training checks
  - when multiple downstream benchmarks are configured, the manifest uses `benchmark_suite_score_v2` for leaderboard ranking instead of a simple mean
  - the suite score is `weighted_mean(component_score) - dispersion_penalty * weighted_stddev(component_score) - failure_penalty * failed_weight_ratio`
  - `component_score` is each benchmark's resolved leaderboard value (`portfolio_sharpe` when a portfolio backtest produces Sharpe, otherwise `top_score`)
  - `top_score` and `avg_score` in the aggregate view are weight-aware means across successful benchmarks; recommendation and skipped counts remain summed
  - `downstream_benchmark_results` records each benchmark's `weight`, and `downstream_benchmark_aggregate` records the weighted mean, dispersion, penalty factors, and penalty contributions used to produce the suite score
- The run-local artifact is always written first; publishing is a second step, not the primary storage location

## Next Extensions

- Add more experiment specs for top-k, threshold, and universe ablations
- Plug in `Research/SignalEvaluator` summaries before ranking to filter weak signal regimes
- Add model-version sweeps once multiple saved models are available
- Add optional metric normalization if benchmark suites start mixing materially different score scales
