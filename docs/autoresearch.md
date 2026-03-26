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
├── Proposal.py              # Next-round sweep proposal generation from prior sweep summaries
├── Selection.py             # ML-backed stock selection execution
├── Training.py              # ML training orchestration + run-local model artifacts
├── Storage.py               # Per-run artifact layout and manifest persistence
├── Leaderboard.py           # Leaderboard synthesis from run manifests
├── Pipeline.py              # Single-run + sweep orchestration
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

### Training Sweep Mode

1. Load a `mode: "training_sweep"` spec
2. Expand `sweep.grid` into a concrete cross-product of training experiment variants
3. Materialize each variant as a normal `mode: "training"` `ExperimentSpec`
4. Run each variant through the existing training pipeline, including downstream benchmark selections when configured
5. Persist a sweep run under `AutoResearch/results/sweeps/<sweep>/runs/<run_id>/` with:
   - `spec.json`
   - `variants.json`
   - `summary.json`
   - `leaderboard.md`
   - `leaderboard.csv`
6. Keep the normal per-variant run artifacts and repository-wide leaderboard under `AutoResearch/results/experiments/...`

### Proposal Generation Mode

1. Read a prior `training_sweep` run from a chosen `summary.json`, or default to the latest summary under `AutoResearch/results/sweeps/*/runs/*/summary.json`
2. Load the matching sweep `spec.json` snapshot from the same run directory
3. Select the top completed run(s) from the ranked `runs` list in the summary
4. Build a next-round `training_sweep` spec by:
   - refining numeric grid values into a simple neighborhood around winning values
   - carrying forward winning categorical values such as `training.model_type`
   - updating the base training config to the best run's values for those refined paths
5. Save the generated JSON spec under `experiments/autoresearch/generated/`

## Why This Fits The Current Repo

- `Research/` remains responsible for reporting utilities and experiment-style output formatting.
- `ML/` remains responsible for prediction and backtesting logic.
- `ML/` remains responsible for training and model serialization logic.
- `AutoResearch/` is only orchestration, storage, and experiment bookkeeping.

That keeps the new layer thin and makes it a practical place to add future loops such as spec sweeps, model ablations, or signal-research prefilters without moving core logic again.

## Sweep Spec Format

Training sweeps reuse the normal training spec shape and add a `sweep` block:

```json
{
  "name": "baseline-model-training-sweep",
  "mode": "training_sweep",
  "training": {
    "begin_time": "2020-01-01",
    "end_time": "2022-12-31",
    "codes": ["600519", "000333"],
    "model_type": "lightgbm",
    "label_config": {
      "threshold_pct": 0.05
    }
  },
  "benchmark_selection": {
    "as_of": "2025-01-15",
    "codes": ["600519", "000333"],
    "top_k": 2
  },
  "sweep": {
    "variant_name_template": "{name}-{model_type}-thr{threshold_pct}",
    "grid": [
      {
        "name": "model_type",
        "path": "training.model_type",
        "values": ["lightgbm", "randomforest"]
      },
      {
        "name": "threshold_pct",
        "path": "training.label_config.threshold_pct",
        "values": [0.03, 0.05]
      }
    ]
  }
}
```

Rules:

- `mode` must be `training_sweep`
- the top level still uses the current training + benchmark fields, so every expanded variant is a real training experiment
- `sweep.grid` is a cross-product; each entry defines:
  - `name`: token exposed to `variant_name_template`
  - `path`: dot-path override applied to the concrete experiment payload
  - `values`: list of values used for expansion
- `variant_name_template` is optional; default names are `name-variant-001`, `name-variant-002`, ...
- `variants.json` captures the fully expanded concrete spec plus the config mapping for each variant
- `summary.json` stores the ranked run list and best run selection using the child manifests' existing leaderboard metrics

## Proposal Engine v1

This repo now includes a minimal self-iteration loop for training sweeps.

- Input:
  - a prior sweep `summary.json`, usually from `AutoResearch/results/sweeps/.../runs/<run_id>/summary.json`
  - the paired `spec.json` snapshot written by the sweep run
- Selection:
  - defaults to the latest available sweep summary when no summary path is provided
  - uses the top `N` completed runs from `summary.json` (`N=2` by default)
- Refinement behavior:
  - numeric sweep parameters are narrowed to a local neighborhood around winner values
  - categorical sweep parameters keep the winning values in rank order
  - v1 intentionally only refines paths already present in the original `sweep.grid`
- Output:
  - a new `mode: "training_sweep"` JSON spec under `experiments/autoresearch/generated/`
  - a generated name/tag set so proposed sweeps stay distinct from hand-authored baseline specs

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

Run a training sweep:

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/baseline_model_training_sweep.json
```

Generate a next-round sweep proposal from the latest available sweep summary:

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --generate-next-sweep
```

Generate a next-round sweep proposal from a specific prior summary and use the top 1 run only:

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --generate-next-sweep \
  --sweep-summary AutoResearch/results/sweeps/baseline-model-training-sweep/runs/20260326T033038Z/summary.json \
  --proposal-top-runs 1
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
- Sweep behavior:
  - `mode: "training_sweep"` expands a grid of concrete training variants and runs them one by one through the same training + benchmark code path used by standalone training specs
  - each child run keeps its own `experiments/<experiment>/runs/<run_id>/...` artifacts and contributes to the repository-level leaderboard
  - each sweep run also writes `sweeps/<sweep>/runs/<run_id>/summary.json` and `leaderboard.md` so you can compare only the variants from that batch
  - sweep `summary.json` stores the best run plus the full config/result mapping for every variant
- Proposal behavior:
  - `--generate-next-sweep` reads a prior sweep summary plus its stored `spec.json` snapshot and writes a fresh sweep JSON under `experiments/autoresearch/generated/`
  - when `--sweep-summary` is omitted, AutoResearch uses the latest summary found under `results_root/sweeps/`
  - numeric refinement is intentionally local and simple in v1; it is meant to tighten a search around strong regions, not perform Bayesian optimization or broader experiment planning
- The run-local artifact is always written first; publishing is a second step, not the primary storage location

## Next Extensions

- Add more experiment specs for top-k, threshold, and universe ablations
- Plug in `Research/SignalEvaluator` summaries before ranking to filter weak signal regimes
- Add selection-model sweeps once multiple saved models are available
- Add optional metric normalization if benchmark suites start mixing materially different score scales
