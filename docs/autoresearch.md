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
6. Persist:
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
- The run-local artifact is always written first; publishing is a second step, not the primary storage location

## Next Extensions

- Add more experiment specs for top-k, threshold, and universe ablations
- Plug in `Research/SignalEvaluator` summaries before ranking to filter weak signal regimes
- Add model-version sweeps once multiple saved models are available
- Add post-training evaluation metrics so training runs can rank on validation quality instead of artifact-only bookkeeping
