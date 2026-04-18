# Day-30m-5m Multi-Level Training Design

Date: 2026-04-18
Status: Draft approved in chat, written for implementation planning

## Context

The current repo can train and select on a single decision level, but the actual training path is still effectively daily-only:

- `AutoResearch/Training.py` loads `CChan(..., lv_list=[KL_TYPE.K_DAY])`
- `ML/Training/Trainer.py` uses `BSPFeatureExtractor` directly
- `ML/FeatureEngine/MultiLevelExtractor.py` exists but is not connected to the main training flow
- `DataAPI/SQLiteDailyBarAPI.py` and `App/sync_a_share_daily_to_sqlite.py` only cover daily bars

That is sufficient for a single-level prototype, but it does not match the intended trading process:

- `day` sets the background and directional filter
- `30m` is the decision level
- `5m` provides trigger confirmation and early exit warning

The new design must convert that discretionary "区间套" logic into point-in-time features and labels that can be trained, validated, and replayed.

## Goals

1. Store local A-share K-line history with `1m` as the base fact table and derive `5m`, `30m`, and `day`.
2. Train a `30m` buy-entry classifier that uses `day + 30m + 5m` features.
3. Train a `5m` exit-warning classifier for an active `30m` long position.
4. Keep `30m` reverse BSP as the first exit-confirmation rule instead of immediately training a separate confirmation model.
5. Preserve all BSP types (`T1/T1P/T2/T2S/T3A/T3B`) for training and later portfolio composition.
6. Keep the pipeline point-in-time safe and auditable, with no future-structure leakage.

## Non-Goals

This phase does not include:

- a pure `1m` decision model
- a short-selling model
- one separate model per BSP type
- training a model for `30m` exit confirmation
- replacing the existing HS300/daily path before the multi-level path is proven

## Recommended Approach

Use a two-model structure over a shared multi-level event pipeline:

1. a `30m` buy-entry classifier
2. a `5m` exit-warning classifier
3. a rule-based `30m` reverse BSP exit confirmation

This keeps the learned tasks narrow and coherent:

- the buy model answers "is this `30m` buy point worth entering under current `day + 30m + 5m` context?"
- the exit-warning model answers "does this first `5m` reverse warning inside the active position look like a real escalation toward `30m` failure?"

## Data Access Boundary

This implementation does not own K-line persistence. The user will handle that in a separate ClickHouse task.

The training path should therefore depend on an abstract multi-level market-data boundary:

- the storage layer must be able to provide point-in-time `day`, `30m`, and `5m` histories derived from `1m`
- the training path must not care whether those bars come from SQLite, ClickHouse, parquet, or another backend
- all storage-specific sync and aggregation work stays out of scope for this phase

The only requirement for this phase is that the training pipeline can request:

- `day` bars for background structure
- `30m` bars for decision-level structure
- `5m` bars for trigger and warning structure

This separation is intentional. It lets the model pipeline move forward without coupling it to a storage decision that is being handled elsewhere.

## Model Structure

### Buy model

- One model for long-entry scoring
- All buy BSP types remain in one shared training pool
- BSP type is preserved as an explicit feature and evaluation slice
- Decision level is `30m`
- Context levels are `day` and `5m`

This is preferred over one-model-per-type because splitting the sample pool too early will shrink data and increase overfitting risk.

### Exit model

- One model for long-position exit warning
- It is not a short-selling model
- It predicts whether an observed `5m` reverse warning is likely to escalate into a real `30m` exit condition soon

### Exit confirmation

The first implementation keeps confirmation rule-based:

- `30m` reverse BSP confirms the exit
- execution happens on the next tradable `5m` open after confirmation

That keeps the final authority on structure with the `30m` layer while still allowing the `5m` model to provide early warning.

## Event and Label Definitions

## Buy-entry samples

- sample anchor: every `30m` buy BSP
- supported types: `T1`, `T1P`, `T2`, `T2S`, `T3A`, `T3B`
- entry execution price: next `5m` open after the `30m` buy BSP
- exit warning event: first `5m` reverse BSP inside the active position
- exit confirmation: first `30m` reverse BSP after the position is open
- exit execution price: next `5m` open after the `30m` reverse BSP

### Buy label

Positive if:

- the trade reaches the structured exit above
- net PnL after transaction cost and slippage is `> 0`

Negative otherwise.

### Tail handling

If the dataset ends before a `30m` reverse BSP confirms the exit, drop that sample from supervised training. These are right-censored outcomes and should not be force-labeled in phase 1.

## Exit-warning samples

- sample anchor: the first `5m` reverse BSP observed inside an active `30m` long position
- all `5m` reverse BSP events should still be recorded in the raw event table
- only the first warning in each position should enter the first training dataset

### Important correction

An exit-warning label cannot be "did a `30m` reverse BSP happen eventually?" because every completed position eventually exits and that would collapse the task into near-all-positive labels.

Therefore the supervised label must be bounded.

### Exit-warning label

Positive if:

- a `30m` reverse BSP confirms within a short confirmation horizon after the first `5m` reverse BSP

Negative if:

- no `30m` reverse BSP confirms within that horizon

Recommended default:

- `exit_warning_confirmation_horizon_30m = 8`

This means the first `5m` reverse warning is considered correct only if it escalates into a `30m` reverse BSP within the next eight `30m` bars, roughly one trading day. The raw event stream can still preserve the longer eventual outcome for analysis, but the supervised label must stay discriminative.

## Feature Buckets

Features should be organized by level, then merged into one fixed-order vector.

### `day` bucket

- direction of the latest daily BI / SEG
- daily trend alignment flags
- current `30m` decision point relative to daily ZS region
- daily divergence and strength measures
- daily volatility / slope background
- market regime filters that indicate whether new long entries are structurally allowed

The `day` bucket is contextual. It should filter or condition decisions rather than dominate precise timing.

### `30m` bucket

- BSP type and direction
- whether the BSP is segment-level
- BSP type count / combined structure flags
- BI amplitude, MACD area / peak / slope, volume, RSI, K count
- SEG direction, amplitude, slope, BI count
- ZS high / low / width / count / price-to-ZS
- local return, volatility, and trend-strength features
- alignment to the `day` bucket

The `30m` bucket is the main decision layer and should carry the most signal weight.

### `5m` bucket

- whether a same-direction confirming `5m` structure exists around entry
- whether a reverse `5m` BSP has appeared
- time since `30m` entry anchor
- `5m` BI / SEG / ZS intensity around the event
- price position relative to `30m` entry and `30m` ZS
- local micro-volatility and micro-trend state

The `5m` bucket is for timing and warning, not for replacing `30m`.

## Leakage Controls

The multi-level path has a high leakage risk, so the design keeps these rules strict:

1. only use structures that are observable at the event timestamp
2. no future BSP, BI, SEG, or ZS backfill in features
3. derived bars must be materialized from historical `1m` only, with deterministic session boundaries
4. train / validation / test splits must remain chronological
5. position-level grouping must prevent one long position from leaking multiple correlated events across train/test boundaries

## Training and Evaluation

### Training defaults

- model type: LightGBM
- task type: binary classification
- split mode: time-based walk-forward
- sample guardrails remain enabled

### Evaluation requirements

For both models, report:

- train/test classification metrics
- generalization gap
- overfit risk
- metrics sliced by BSP type

For the buy model, also report:

- trade count
- win rate
- average net return
- max drawdown on a replayed position series

For the exit-warning model, also report:

- precision on warning events
- false-warning rate
- lead time from first warning to confirmed `30m` exit

## Runtime Execution Semantics

The intended runtime flow is:

1. build `day`, `30m`, and `5m` structures from the local SQLite store
2. score each eligible `30m` buy BSP with the buy model
3. require `day` environment alignment and optional `5m` confirmation
4. enter on the next `5m` open
5. while holding, monitor the first `5m` reverse BSP
6. score that warning with the exit-warning model
7. if warning risk is high, increase exit readiness
8. execute only when `30m` reverse BSP confirms, at the next `5m` open

## Implementation Direction

The codebase should not be patched by sprinkling special cases into the current daily-only path. The recommended direction is:

- explicit multi-level sample builder
- explicit structure-aware label builders
- explicit task definitions in AutoResearch spec/config
- a storage-agnostic loader boundary that can later be backed by ClickHouse

This keeps the existing daily path stable while the new `day -> 30m -> 5m` path is built and verified in parallel.

## Final Design Decisions

The approved defaults are:

- the external storage system is expected to provide histories that support `5m`, `30m`, and `day`
- one buy model and one exit-warning model
- all BSP types are kept in training
- BSP type is a feature and evaluation slice, not a model split
- `30m` is the decision level
- `day` is the background filter
- `5m` is the trigger / warning level
- buy entry executes at the next `5m` open after the `30m` buy BSP
- exit executes at the next `5m` open after the `30m` reverse BSP
- training drops right-censored buy samples with no confirmed exit
- exit-warning training uses the first `5m` reverse BSP per position
- exit-warning labels use a bounded `30m` confirmation horizon, default `8` bars

These defaults are the baseline for the implementation plan.
