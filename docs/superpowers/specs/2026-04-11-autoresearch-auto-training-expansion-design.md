# AutoResearch Auto-Expansion Training Design

Date: 2026-04-11
Status: Draft approved in chat, written for review

## Context

The repo now has a working AutoResearch pipeline for:

- declarative training specs in `AutoResearch/Spec.py`
- training execution in `AutoResearch/Training.py`
- model scoring and downstream benchmark selections in `AutoResearch/Pipeline.py`
- basic anti-overfitting guardrails in `ML/Training/Trainer.py`

Those guardrails expose a real problem in the current baseline setup: small training specs can fail because the resulting sample count is too low to support meaningful time-series generalization.

That failure is technically correct, but it is not sufficient for the product direction of this repo. The project goal is not a manual workflow where a human sees "too few samples" and hand-edits the next training spec. The goal is AutoResearch: when a training setup is too small, the system should automatically enlarge the training scope and continue researching.

## Goals

This design should make AutoResearch able to:

1. Detect insufficient training samples and treat that condition as recoverable.
2. Automatically retry training with a larger scope instead of immediately failing.
3. Preserve anti-overfitting guardrails rather than weakening or disabling them.
4. Record both the original requested training spec and the effective training spec that finally succeeded.
5. Surface enough attempt history in run summaries for later inspection, ranking, and future automation.

## Non-Goals

This design does not include:

- automatic expansion to full tradable A-shares in phase 1
- a new scheduler or orchestration service
- automatic hyperparameter search as part of the recovery path
- changing ranking logic, labels, or feature engineering
- replacing the current benchmark suite logic

## Approaches Considered

### Option A: Fail and let humans rewrite specs

Pros:

- smallest implementation
- keeps training semantics explicit

Cons:

- breaks the AutoResearch product goal
- turns sample insufficiency into a manual bottleneck

### Option B: Automatic in-training recovery

Pros:

- keeps the current CLI and pipeline contract unchanged
- matches the "automatic research" intent of the repo
- works well with the existing training summary and benchmark pipeline

Cons:

- requires careful bookkeeping so the original spec and effective spec do not get conflated
- introduces retry policy decisions that must be tested

### Option C: Generate a new sweep/spec instead of retrying in place

Pros:

- very flexible
- closer to a general research planner

Cons:

- heavier than the current need
- adds more moving parts before the training recovery loop is proven

### Recommendation

Use Option B.

The system should keep the stricter anti-overfitting guardrails and respond to sample insufficiency by automatically expanding the training scope inside the training workflow itself.

## Design Overview

Training sample insufficiency becomes a recoverable condition.

The training runner will:

1. try the original training spec
2. if the sample guard passes, continue normally
3. if the sample guard fails with an "insufficient samples" error, generate the next larger training attempt
4. retry until one attempt succeeds or the expansion ladder is exhausted
5. if all attempts fail, write a failed summary that includes the full attempt history

This preserves a strict rule:

- the `Trainer` remains responsible for deciding whether a given dataset is large enough
- `AutoResearch/Training.py` becomes responsible for deciding how to expand the training scope after a recoverable failure

## Expansion Policy

The approved automatic expansion strategy is:

### Expansion order

1. expand time range first
2. expand universe second

This keeps the original strategy semantics as intact as possible before broadening the training pool.

### Universe expansion policy

If the original spec uses `codes` or `codes_file`, AutoResearch is allowed to move beyond that explicit pool when the sample guard fails.

Phase-1 universe ladder is:

1. original training pool
2. `hs300`

Phase 1 stops there and does not automatically jump to all tradable A-shares.

### Time expansion policy

The initial retry ladder is:

1. original `begin_time`
2. `begin_time - 2 years`
3. `begin_time - 4 years`

Then, if needed, retry on `hs300` using:

4. `hs300 + original begin_time`
5. `hs300 + begin_time - 2 years`

This keeps the first implementation bounded and predictable while still making AutoResearch materially more autonomous.

## Effective Attempt Semantics

Each training run now has two distinct concepts:

- `original_training_spec`: what the user or experiment author asked for
- `effective_training_spec`: the attempt that actually produced the final model

The original spec is never mutated on disk.

The recovery logic operates on in-memory derived attempts only. That keeps experiment intent auditable and avoids hidden spec drift.

## Attempt Recording

Training summaries should include a `training_attempts` list.

Each attempt entry should record:

- `attempt_index`
- `begin_time`
- `end_time`
- `universe_source`
- `codes_count` when applicable
- `codes_file` when applicable
- `auto_expansion_applied`
- `loaded_chan_count`
- `sample_guard_passed`
- `status`
- `error` when failed

This is required for both success and failure cases.

## Summary Fields

Successful training summaries should additionally include:

- `original_training_spec`
- `effective_training_spec`
- `auto_expansion`
- `training_attempts`
- existing `dataset_profile`
- existing `classification_metrics`
- existing `overfit_risk`

The `auto_expansion` object should include:

- `triggered`
- `strategy`
- `final_attempt_index`
- `stopped_reason`

Failed training summaries should still include:

- the top-level `error`
- the full `training_attempts`
- `original_training_spec`
- the final attempted effective scope if available

That keeps failed runs useful to later automated reasoning.

## Component Changes

### 1. `ML/Training/Trainer.py`

Keep the current responsibilities:

- build samples
- split by time
- enforce minimum sample guardrails
- compute train/test diagnostics
- classify overfit risk

Do not move expansion policy into `Trainer`.

### 2. `AutoResearch/Training.py`

Add an internal attempt generator and recovery loop.

Responsibilities:

- classify recoverable sample-insufficiency failures
- generate the next larger attempt according to the approved ladder
- resolve training universes for original pool vs `hs300`
- record attempt history
- write original/effective spec snapshots into summary and metadata

### 3. Existing HS300 resolver

Reuse the current HS300 universe resolution already added for the recommendation path.

Do not create a separate training-only HS300 resolver unless later refactoring is needed.

### 4. `AutoResearch/Pipeline.py`

No architectural changes are required for the first phase.

It should continue treating the training runner as a black box that returns a completed or failed summary.

## Error Handling

Expected cases:

- original attempt fails due to insufficient samples
- later expanded attempt succeeds
- all attempts fail due to insufficient samples
- an attempt fails due to a non-recoverable error such as dependency or provider failure

Handling rules:

- only sample-insufficiency failures trigger auto-expansion
- non-recoverable failures should stop immediately
- sample-insufficiency failures after the final allowed attempt should produce a failed run with full attempt history
- recovery must not silently widen all the way to the full market in phase 1

## Testing Plan

Minimum test coverage:

1. automatic retry after insufficient samples:
   - original attempt fails
   - expanded time-window attempt succeeds
2. automatic universe expansion:
   - original explicit `codes` attempt fails
   - `hs300` attempt succeeds
3. full failure path:
   - all attempts fail
   - summary includes all attempt entries
4. summary semantics:
   - `original_training_spec` is preserved
   - `effective_training_spec` reflects the successful attempt
   - `auto_expansion` and `training_attempts` are present
5. recovery boundary:
   - non-sample-guard failures do not trigger expansion

## Risks and Tradeoffs

The main tradeoff is experiment purity versus autonomy.

Automatic expansion means the final trained model may come from a meaningfully larger scope than the original request. That is acceptable for AutoResearch only if the system records that difference explicitly and makes it easy to inspect.

The main implementation risk is letting the recovery loop become opaque. The attempt history and explicit original/effective spec fields are therefore mandatory, not optional.
