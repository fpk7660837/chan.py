# HS300 Stock Picker Design

Date: 2026-04-10
Status: Draft approved in chat, written for review

## Context

The repo already has the core pieces needed to rank stocks and output a recommendation list:

- `App/generate_stock_recommendations.py` can load a stock universe, build Chan structures, score buy signals, and write a recommendation file.
- `AutoResearch/Selection.py` wraps the same flow into experiment runs and summaries.
- `AutoResearch/Spec.py` already supports declarative selection specs through JSON.

What is missing is a first-class way to express a named A-share universe such as `HS300` without manually passing a code list every time.

The current user goal is narrower than the full research roadmap:

- market scope: A-shares only
- phase-1 universe: HS300
- output: run the program and get a candidate stock list
- non-goals for v1: ETFs, Hong Kong stocks, US stocks, full-market A-shares, historical index constituent reconstruction

## Goals

Phase-1 should make the system capable of:

1. Selecting from the HS300 universe without manually maintaining a code file.
2. Reusing the existing recommendation and AutoResearch paths instead of creating a parallel pipeline.
3. Producing a ranked candidate list with the current ML predictor and Chan signal scoring stack.
4. Keeping current behavior intact for existing `codes` and `codes_file` users.

## Non-Goals

This design intentionally does not include:

- historical HS300 constituent snapshots by `as_of`
- dynamic universe reconstitution for backtests
- ETF support
- a new model-training workflow
- new ranking logic or feature engineering changes
- productized scheduling or GUI work

## Approaches Considered

### Option A: Static code file only

Ship a checked-in HS300 code file and keep using `codes_file`.

Pros:

- lowest implementation cost
- no schema changes

Cons:

- the file becomes stale
- user experience is poor
- not a durable abstraction for future universes

### Option B: Named universe support

Add a `universe` field to the existing CLI/spec flow and implement `hs300` as the first supported named universe.

Pros:

- minimal but durable abstraction
- reuses existing ranking pipeline
- easy to extend later to `csi500`, `all_a`, or custom families

Cons:

- requires small spec and CLI changes
- introduces a new resolution layer to test

### Option C: Full historical universe engine

Implement time-aware index constituent resolution by `as_of`.

Pros:

- most rigorous for research

Cons:

- too heavy for the current milestone
- significantly expands data and validation requirements

### Recommendation

Use Option B.

It matches the current milestone because it improves usability without changing the scoring architecture, and it leaves room for a more rigorous historical-universe layer later.

## Design Overview

The system will gain a named-universe abstraction that sits above the current code-list loading logic.

Resolution priority will be:

1. `codes`
2. `codes_file`
3. `universe`
4. fallback to current tradable-stock loader only when none of the above are provided

For phase-1, the only named universe will be `hs300`.

The selected universe still flows through the existing pipeline:

1. resolve stock codes
2. load Chan data for each code
3. rank the pool with the current predictor
4. output recommendations
5. optionally run the existing portfolio backtest path in AutoResearch

No new ranking algorithm or model path will be added in this phase.

## Component Changes

### 1. Recommendation CLI

Update `App/generate_stock_recommendations.py` to accept:

- `--universe hs300`

Behavior:

- if `--codes` is passed, use it
- else if `--codes-file` is passed, use it
- else if `--universe hs300` is passed, resolve HS300 members
- else keep current fallback behavior

This preserves backward compatibility.

### 2. Universe Resolver

Introduce a small resolver layer responsible only for returning `(code, name)` pairs.

Initial responsibility:

- resolve `hs300`

Likely implementation shape:

- a helper function in the current recommendation module, or
- a small shared utility module if the code becomes cleaner that way

The resolver should be isolated from ranking logic so future universes can be added without touching the predictor flow.

### 3. AutoResearch Spec

Extend `SelectionSpec` in `AutoResearch/Spec.py` with:

- `universe: Optional[str] = None`

This allows experiment JSON to declare:

```json
{
  "selection": {
    "as_of": "2024-12-31",
    "universe": "hs300",
    "top_k": 10
  }
}
```

`codes` and `codes_file` continue to override it when present.

### 4. AutoResearch Selection Runtime

Update `AutoResearch/Selection.py` so the generated runtime args include `universe`.

The rest of the selection experiment remains unchanged because it already delegates universe loading to the recommendation entry helpers.

### 5. Baseline Experiment Spec

Add a new baseline experiment spec for the first-phase user path, for example:

- `experiments/autoresearch/hs300_daily_selection.json`

This becomes the canonical experiment entry for "run and get HS300 candidate stocks."

## Data and Source Assumptions

Phase-1 assumes the system can fetch the current HS300 constituent list from the same market-data stack already used by the repo.

If the upstream provider is unavailable or returns malformed data:

- fail clearly with an actionable error message
- do not silently fall back to the full tradable A-share universe

That avoids hidden behavior changes and keeps experiment meaning stable.

## Error Handling

Expected error cases:

- unsupported universe name
- universe resolution returns no stocks
- market-data dependency missing
- upstream fetch failure
- individual stock data load failure during Chan construction

Handling rules:

- unsupported universe: raise a direct configuration error
- empty universe: raise a runtime error before ranking begins
- per-stock failures: keep current skip behavior and surface counts/reasons in summary
- upstream universe fetch failures: fail fast rather than silently changing universe semantics

## Testing Plan

The minimum test set for this design is:

1. spec parsing:
   - `SelectionSpec` accepts `universe`
   - defaults remain backward compatible
2. runtime argument wiring:
   - `AutoResearch/Selection.py` passes `universe` through
3. universe resolution:
   - `hs300` resolves to normalized code tuples
   - unsupported names raise errors
4. precedence rules:
   - `codes` beats `codes_file`
   - `codes_file` beats `universe`
5. end-to-end selection flow under mocks:
   - a selection run using `universe=hs300` produces recommendations and summary output

Tests should stay mostly mocked around universe fetching and market data so they remain deterministic.

## Rollout Plan

Phase-1 implementation should be done in this order:

1. extend selection spec and runtime wiring with `universe`
2. implement named-universe resolution for `hs300`
3. add/update tests for parsing, precedence, and mocked selection flow
4. add a baseline HS300 experiment JSON
5. verify the new path through focused tests and one local CLI smoke run if dependencies are available

## Open Follow-Ups

These are explicitly deferred until after phase-1 is working:

- `csi500` and broader A-share universes
- historical constituent snapshots by date
- ETF support
- production scheduling of recurring candidate-list generation
- stronger universe quality filters on top of index membership

## Acceptance Criteria

This design is considered implemented when:

1. The user can run a recommendation command or AutoResearch selection spec with `hs300` as the universe.
2. The program returns a ranked candidate list without requiring a manually maintained code file.
3. Existing `codes` and `codes_file` workflows still behave as before.
4. Tests cover the new parsing and resolution behavior.
