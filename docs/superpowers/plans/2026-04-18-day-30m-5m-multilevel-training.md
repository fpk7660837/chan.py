# Day-30m-5m Multi-Level Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `day -> 30m -> 5m` multi-level training path that supports a `30m` buy-entry model and a `5m` exit-warning model without owning K-line storage in this task.

**Architecture:** Keep the existing daily path stable and add a new explicit training path alongside it. Build position-aware multi-level events from externally provided `day/30m/5m` Chan structures, train two binary LightGBM tasks on those events, and keep the market-data loading boundary abstract so a later ClickHouse task can plug in without rewriting the trainer.

**Tech Stack:** Python 3.11, Chan `CChan`, LightGBM, pytest/unittest, AutoResearch pipeline.

---

## File Structure

- Modify: `AutoResearch/Spec.py`
  Responsibility: add task-level multi-level training spec fields.
- Modify: `Config/MLConfig.py`
  Responsibility: add defaults for decision level, context levels, execution level, event horizons, and task names.
- Create: `ML/Training/MultiLevelSampleBuilder.py`
  Responsibility: build position-aware `30m` entry events and first-`5m` warning events from point-in-time structures.
- Modify: `ML/FeatureEngine/MultiLevelExtractor.py`
  Responsibility: turn `day/30m/5m` context objects into prefixed fixed-order feature vectors.
- Modify: `ML/Training/LabelBuilder.py`
  Responsibility: add structure-aware labels for buy-entry and bounded exit-warning tasks.
- Modify: `ML/Training/Trainer.py`
  Responsibility: route between task types, call the multi-level sample builder, enforce grouped chronological splitting, and preserve diagnostics.
- Modify: `AutoResearch/Training.py`
  Responsibility: load multi-level Chan data through an abstract boundary and invoke the new task-aware trainer path.
- Create: `ML/Training/MultiLevelDataLoader.py`
  Responsibility: define the minimal loader boundary that this task needs from a future ClickHouse-backed provider.
- Modify: `tests/test_autoresearch.py`
  Responsibility: cover new spec parsing and end-to-end training orchestration.
- Modify: `tests/test_training_diagnostics.py`
  Responsibility: cover new grouped datasets and task-specific diagnostics.
- Create: `tests/test_multilevel_sample_builder.py`
  Responsibility: cover entry-event and first-warning-event extraction.
- Create: `tests/test_multilevel_label_builder.py`
  Responsibility: cover buy labels, censored-sample dropping, and bounded exit-warning labels.

### Task 1: Add spec and config coverage for multi-level tasks

**Files:**
- Modify: `AutoResearch/Spec.py`
- Modify: `Config/MLConfig.py`
- Modify: `tests/test_autoresearch.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_training_spec_supports_multilevel_buy_task(self):
    spec = load_experiment_spec(path)
    self.assertEqual(spec.training.training_config["task_name"], "buy_entry")
    self.assertEqual(spec.training.training_config["decision_level"], "30m")
    self.assertEqual(spec.training.training_config["context_levels"], ["day", "5m"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_autoresearch.py -k multilevel_buy_task -v`
Expected: FAIL because the spec schema does not yet preserve these fields coherently.

- [ ] **Step 3: Write minimal implementation**

```python
training_config.update({
    "task_name": "buy_entry",
    "decision_level": "30m",
    "context_levels": ["day", "5m"],
    "execution_level": "5m",
    "exit_warning_confirmation_horizon_30m": 8,
})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_autoresearch.py -k multilevel_buy_task -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add AutoResearch/Spec.py Config/MLConfig.py tests/test_autoresearch.py
git commit -m "feat: add multilevel training spec defaults"
```

### Task 2: Add a storage-agnostic multi-level data-loader boundary

**Files:**
- Create: `ML/Training/MultiLevelDataLoader.py`
- Modify: `AutoResearch/Training.py`
- Modify: `tests/test_autoresearch.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_training_experiment_uses_multilevel_loader_for_buy_entry_task(self):
    ...
    self.assertEqual(load_kwargs["levels"], ["day", "30m", "5m"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_autoresearch.py -k multilevel_loader -v`
Expected: FAIL because the training path only knows the daily loader.

- [ ] **Step 3: Write minimal implementation**

```python
class MultiLevelDataLoader:
    def load_training_contexts(...):
        raise NotImplementedError
```

- [ ] **Step 4: Route multi-level tasks through the new loader boundary**

```python
if task_name in {"buy_entry", "exit_warning"}:
    contexts = loader.load_training_contexts(...)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_autoresearch.py -k multilevel_loader -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ML/Training/MultiLevelDataLoader.py AutoResearch/Training.py tests/test_autoresearch.py
git commit -m "feat: add multilevel data loader boundary"
```

### Task 3: Build multi-level entry and warning events

**Files:**
- Create: `ML/Training/MultiLevelSampleBuilder.py`
- Modify: `ML/FeatureEngine/MultiLevelExtractor.py`
- Create: `tests/test_multilevel_sample_builder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_buy_entry_events_keeps_all_30m_bsp_types():
    events = builder.build_buy_entry_events(contexts)
    assert {event.bsp_type for event in events} >= {"T1", "T2", "T3A"}
```

- [ ] **Step 2: Write the failing test**

```python
def test_build_exit_warning_events_uses_first_5m_reverse_bsp_per_position():
    warnings = builder.build_exit_warning_events(contexts)
    assert len(warnings) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_multilevel_sample_builder.py -v`
Expected: FAIL because the repo has no position-aware multi-level event builder.

- [ ] **Step 4: Write minimal implementation**

```python
event.context = {"day": day_bsp, "30m": bsp_30m, "5m": bsp_5m}
features = extractor.extract_multi_level(event.context)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_multilevel_sample_builder.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ML/Training/MultiLevelSampleBuilder.py ML/FeatureEngine/MultiLevelExtractor.py tests/test_multilevel_sample_builder.py
git commit -m "feat: add multilevel entry and warning events"
```

### Task 4: Add structure-aware labels for buy-entry and exit-warning tasks

**Files:**
- Modify: `ML/Training/LabelBuilder.py`
- Create: `tests/test_multilevel_label_builder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_buy_entry_label_uses_next_5m_open_for_entry_and_exit():
    label, ret = builder.label_buy_entry(event)
    assert label == 1
```

- [ ] **Step 2: Write the failing test**

```python
def test_buy_entry_label_drops_samples_without_confirmed_30m_exit():
    assert builder.label_buy_entry(event) is None
```

- [ ] **Step 3: Write the failing test**

```python
def test_exit_warning_label_requires_30m_confirm_within_horizon():
    label = builder.label_exit_warning(event)
    assert label == 0
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_multilevel_label_builder.py -v`
Expected: FAIL because the current label builder only supports simple forward-return logic.

- [ ] **Step 5: Write minimal implementation**

```python
if task_name == "buy_entry":
    ...
elif task_name == "exit_warning":
    ...
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_multilevel_label_builder.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add ML/Training/LabelBuilder.py tests/test_multilevel_label_builder.py
git commit -m "feat: add multilevel structure-aware labels"
```

### Task 5: Integrate the trainer and AutoResearch pipeline

**Files:**
- Modify: `ML/Training/Trainer.py`
- Modify: `AutoResearch/Training.py`
- Modify: `tests/test_training_diagnostics.py`
- Modify: `tests/test_autoresearch.py`

- [ ] **Step 1: Write the failing test**

```python
def test_multilevel_training_keeps_position_events_in_one_split():
    ...
    assert summary["split_mode"] == "grouped_walk_forward"
```

- [ ] **Step 2: Write the failing test**

```python
def test_run_training_experiment_supports_multilevel_buy_entry_task():
    ...
    self.assertEqual(summary["task_name"], "buy_entry")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_training_diagnostics.py tests/test_autoresearch.py -k "multilevel or grouped_walk_forward" -v`
Expected: FAIL because the trainer still assumes flat BSP samples and one simple label flow.

- [ ] **Step 4: Write minimal implementation**

```python
if task_name in {"buy_entry", "exit_warning"}:
    events = self.sample_builder.build(...)
    X, y = ...
```

- [ ] **Step 5: Preserve diagnostics for both tasks**

```python
summary["dataset_profile"]["task_name"] = task_name
summary["classification_metrics"] = ...
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=... python3.11 -S -m pytest tests/test_training_diagnostics.py tests/test_autoresearch.py -k "multilevel or grouped_walk_forward" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add ML/Training/Trainer.py AutoResearch/Training.py tests/test_training_diagnostics.py tests/test_autoresearch.py
git commit -m "feat: integrate multilevel training pipeline"
```

### Task 6: Run full verification and document the ClickHouse boundary

**Files:**
- Modify: none

- [ ] **Step 1: Run all relevant tests**

Run: `PYTHONPATH=/opt/homebrew/lib/python3.11/site-packages:/Users/fupengkai/Library/Python/3.11/lib/python/site-packages python3.11 -S -m pytest tests/test_autoresearch.py tests/test_training_diagnostics.py tests/test_multilevel_sample_builder.py tests/test_multilevel_label_builder.py -v`
Expected: PASS

- [ ] **Step 2: Review the diff for the full feature set**

Run: `git diff -- AutoResearch/Spec.py Config/MLConfig.py ML/Training/MultiLevelDataLoader.py ML/Training/MultiLevelSampleBuilder.py ML/FeatureEngine/MultiLevelExtractor.py ML/Training/LabelBuilder.py ML/Training/Trainer.py AutoResearch/Training.py tests/test_autoresearch.py tests/test_training_diagnostics.py tests/test_multilevel_sample_builder.py tests/test_multilevel_label_builder.py`
Expected: only multi-level training and loader-boundary changes

- [ ] **Step 3: Commit**

```bash
git add AutoResearch/Spec.py Config/MLConfig.py ML/Training/MultiLevelDataLoader.py ML/Training/MultiLevelSampleBuilder.py ML/FeatureEngine/MultiLevelExtractor.py ML/Training/LabelBuilder.py ML/Training/Trainer.py AutoResearch/Training.py tests/test_autoresearch.py tests/test_training_diagnostics.py tests/test_multilevel_sample_builder.py tests/test_multilevel_label_builder.py
git commit -m "feat: add day-30m-5m multilevel training path"
```
