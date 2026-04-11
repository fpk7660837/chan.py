# AutoResearch Auto-Expansion Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AutoResearch automatically retry training with a larger scope when sample guardrails reject the original training attempt, then validate the workflow with a real training run.

**Architecture:** Keep `ML/Training/Trainer.py` responsible for sample guardrails and overfit diagnostics, and add retry/expansion policy inside `AutoResearch/Training.py`. Record every training attempt plus original/effective training specs in the training summary so automatic recovery stays inspectable and auditable.

**Tech Stack:** Python 3.11, dataclasses, unittest/pytest, AutoResearch pipeline, BaoStock/AkShare-backed universe loading.

---

## File Structure

- Modify: `AutoResearch/Training.py`
  Responsibility: classify sample-insufficiency failures, generate expanded training attempts, retry training, and write attempt history plus original/effective training specs to summary and metadata.
- Modify: `AutoResearch/Pipeline.py`
  Responsibility: continue surfacing training summaries cleanly if leaderboard semantics change after automatic recovery.
- Modify: `tests/test_autoresearch.py`
  Responsibility: cover automatic retry success, HS300 fallback expansion, and failure summaries that preserve attempt history.
- Modify: `tests/test_training_diagnostics.py`
  Responsibility: keep the low-level sample guard and diagnostic assertions aligned with the retrying training runner.
- Optionally Create: `experiments/autoresearch/hs300_model_training.json`
  Responsibility: provide a realistic training spec for a real end-to-end validation run after implementation.

### Task 1: Add failing tests for auto-expansion retry behavior

**Files:**
- Modify: `tests/test_autoresearch.py`

- [ ] **Step 1: Write failing tests for time expansion retry success**

```python
def test_run_training_experiment_retries_with_expanded_begin_time_after_sample_guard_failure(self):
    ...
    self.assertEqual(summary["training_attempts"][0]["status"], "failed")
    self.assertEqual(summary["training_attempts"][1]["begin_time"], "2018-01-01")
    self.assertEqual(summary["effective_training_spec"]["begin_time"], "2018-01-01")
```

- [ ] **Step 2: Run the targeted test and verify it fails**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_autoresearch.py -k expanded_begin_time -v`
Expected: FAIL because `run_training_experiment()` does not yet retry.

- [ ] **Step 3: Write failing tests for HS300 expansion after explicit codes fail**

```python
def test_run_training_experiment_can_expand_from_explicit_codes_to_hs300(self):
    ...
    self.assertEqual(summary["training_attempts"][-1]["universe_source"], "hs300")
    self.assertEqual(summary["effective_training_spec"]["universe"], "hs300")
```

- [ ] **Step 4: Run the targeted test and verify it fails**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_autoresearch.py -k expand_from_explicit_codes_to_hs300 -v`
Expected: FAIL because `run_training_experiment()` has no universe expansion policy.

- [ ] **Step 5: Write failing tests for exhausted retry history**

```python
def test_run_training_experiment_records_all_attempts_when_auto_expansion_exhausts(self):
    ...
    self.assertEqual(summary["auto_expansion"]["stopped_reason"], "attempts_exhausted")
    self.assertGreaterEqual(len(summary["training_attempts"]), 2)
```

- [ ] **Step 6: Run the targeted test and verify it fails**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_autoresearch.py -k attempts_exhausted -v`
Expected: FAIL because the current failure summary has no attempt history.

### Task 2: Implement retry policy and attempt accounting

**Files:**
- Modify: `AutoResearch/Training.py`

- [ ] **Step 1: Add a recoverable sample-guard error detector and attempt builder helpers**

```python
def _is_insufficient_samples_error(exc: Exception) -> bool:
    return "Insufficient samples for reliable training" in str(exc)

def _build_training_attempts(training: TrainingSpec) -> list[TrainingSpec]:
    ...
```

- [ ] **Step 2: Run the new targeted tests and keep them red for the missing retry loop**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_autoresearch.py -k "expanded_begin_time or expand_from_explicit_codes_to_hs300 or attempts_exhausted" -v`
Expected: FAIL because helper wiring alone is not enough.

- [ ] **Step 3: Implement the retry loop in `run_training_experiment()`**

```python
for attempt_index, attempt_training in enumerate(_build_training_attempts(training), 1):
    try:
        ...
        model = trainer.train(chan_list, model_type=attempt_training.model_type)
        effective_training = attempt_training
        break
    except Exception as exc:
        ...
```

- [ ] **Step 4: Record `training_attempts`, `original_training_spec`, `effective_training_spec`, and `auto_expansion` in summary and metadata**

```python
summary["training_attempts"] = attempts
summary["original_training_spec"] = asdict(training)
summary["effective_training_spec"] = asdict(effective_training)
summary["auto_expansion"] = {...}
```

- [ ] **Step 5: Re-run the retry-focused tests and verify they pass**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_autoresearch.py -k "expanded_begin_time or expand_from_explicit_codes_to_hs300 or attempts_exhausted" -v`
Expected: PASS

### Task 3: Keep diagnostics and leaderboard semantics coherent

**Files:**
- Modify: `AutoResearch/Training.py`
- Modify: `AutoResearch/Pipeline.py`
- Modify: `tests/test_training_diagnostics.py`

- [ ] **Step 1: Write a failing test that successful auto-expanded training still exposes diagnostics**

```python
def test_run_training_experiment_preserves_diagnostics_for_effective_attempt(self):
    ...
    self.assertIn("classification_metrics", result.summary)
    self.assertEqual(result.summary["leaderboard_metric"], "test_auc")
```

- [ ] **Step 2: Run the targeted test and verify it fails if the retry path drops diagnostics**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_training_diagnostics.py -k preserves_diagnostics -v`
Expected: FAIL if retry summaries do not preserve the trainer diagnostics.

- [ ] **Step 3: Implement the minimal metadata/manifest adjustments needed for retry-produced summaries**

```python
metadata["training_diagnostics"] = ...
summary["leaderboard_metric"] = "test_auc"
```

- [ ] **Step 4: Re-run the targeted diagnostics tests and verify they pass**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_training_diagnostics.py -v`
Expected: PASS

### Task 4: Verify the whole test surface

**Files:**
- Modify: none

- [ ] **Step 1: Run all relevant tests**

Run: `PYTHONPATH=/opt/homebrew/lib/python3.11/site-packages:/Users/fupengkai/Library/Python/3.11/lib/python/site-packages python3.11 -S -m pytest tests/test_autoresearch.py tests/test_generate_stock_recommendations.py tests/test_training_diagnostics.py -v`
Expected: PASS

- [ ] **Step 2: Review the diff for the touched files**

Run: `git diff -- AutoResearch/Training.py AutoResearch/Pipeline.py tests/test_autoresearch.py tests/test_training_diagnostics.py`
Expected: only retry-policy and summary-accounting changes

### Task 5: Validate with a real training run

**Files:**
- Optionally Create: `experiments/autoresearch/hs300_model_training.json`

- [ ] **Step 1: Create or update a realistic training spec that can satisfy the sample guard**

```json
{
  "name": "hs300-model-training",
  "mode": "training",
  "training": {
    "begin_time": "2018-01-01",
    "end_time": "2024-12-31",
    "codes_file": "./hs300_codes.txt",
    "model_type": "lightgbm"
  }
}
```

- [ ] **Step 2: Run a real training command**

Run: `PYTHONPATH=/opt/homebrew/lib/python3.11/site-packages:/Users/fupengkai/Library/Python/3.11/lib/python/site-packages python3.11 -S App/run_autoresearch_pipeline.py --spec experiments/autoresearch/hs300_model_training.json --publish-model`
Expected: completed run with a populated training summary and saved model artifacts

- [ ] **Step 3: Inspect the produced summary for automatic expansion semantics**

Run: `sed -n '1,240p' AutoResearch/results/experiments/hs300-model-training/runs/<run_id>/summary.json`
Expected: includes `training_attempts`, `original_training_spec`, `effective_training_spec`, `dataset_profile`, `classification_metrics`, and `overfit_risk`
