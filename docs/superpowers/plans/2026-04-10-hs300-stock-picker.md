# HS300 Stock Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class `hs300` named-universe support to the recommendation CLI and AutoResearch selection specs without breaking existing `codes` and `codes_file` workflows.

**Architecture:** Extend the existing selection spec/runtime wiring with a new optional `universe` field and keep precedence centralized in `App/generate_stock_recommendations.py`, which already resolves `codes`, `codes_file`, and the fallback tradable-stock universe. Implement `hs300` via AkShare's documented `index_stock_cons_csindex(symbol="000300")` endpoint, fail loudly when named-universe resolution fails, and keep the ranking and backtest pipeline unchanged.

**Tech Stack:** Python 3.11, argparse, unittest/pytest, AkShare `index_stock_cons_csindex`, AutoResearch selection pipeline.

---

## File Structure

- Modify: `AutoResearch/Spec.py`
  Responsibility: add `SelectionSpec.universe`, parse it from JSON, and preserve backward-compatible defaults.
- Modify: `AutoResearch/Selection.py`
  Responsibility: pass `universe` into the runtime args object consumed by `load_universe()`.
- Modify: `App/generate_stock_recommendations.py`
  Responsibility: add CLI support for `--universe`, implement `hs300` named-universe resolution, enforce precedence, and raise actionable errors for unsupported or empty universes.
- Modify: `tests/test_autoresearch.py`
  Responsibility: cover spec parsing, runtime-arg wiring, and the mocked selection run that exercises the `universe="hs300"` path.
- Create: `tests/test_generate_stock_recommendations.py`
  Responsibility: unit-test parser behavior, precedence, HS300 resolution, and error handling without live market-data dependencies.
- Create: `experiments/autoresearch/hs300_daily_selection.json`
  Responsibility: provide the canonical AutoResearch selection spec for the HS300 candidate-list workflow.

### Task 1: Extend the selection schema and runtime arg wiring

**Files:**
- Modify: `AutoResearch/Spec.py`
- Modify: `AutoResearch/Selection.py`
- Modify: `tests/test_autoresearch.py`

- [ ] **Step 1: Write the failing spec and runtime-wiring tests**

```python
from AutoResearch.Selection import _make_runtime_args
from AutoResearch.Spec import ExperimentSpec, SelectionSpec, load_experiment_spec

def test_load_experiment_spec_accepts_named_universe(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        spec_path = Path(tmp_dir) / "hs300.json"
        spec_path.write_text(
            json.dumps(
                {
                    "name": "hs300-daily-selection",
                    "selection": {
                        "as_of": "2024-12-31",
                        "universe": "hs300",
                    },
                }
            ),
            encoding="utf-8",
        )

        spec = load_experiment_spec(spec_path)

    self.assertEqual(spec.selection.universe, "hs300")
    self.assertEqual(spec.selection.codes, [])
    self.assertIsNone(spec.selection.codes_file)

def test_make_runtime_args_passes_universe_through(self):
    spec = ExperimentSpec(
        name="hs300-daily-selection",
        selection=SelectionSpec(as_of="2024-12-31", universe="hs300"),
    )

    runtime_args = _make_runtime_args(spec)

    self.assertEqual(runtime_args.universe, "hs300")
    self.assertIsNone(runtime_args.codes)
    self.assertIsNone(runtime_args.codes_file)
```

- [ ] **Step 2: Run the targeted tests and verify they fail for the missing field**

Run: `python3 -m pytest tests/test_autoresearch.py -k "named_universe or runtime_args_passes_universe" -v`
Expected: FAIL because `SelectionSpec` does not yet accept `universe` and `_make_runtime_args()` does not expose it.

- [ ] **Step 3: Implement the minimal schema/runtime changes**

```python
# AutoResearch/Spec.py
@dataclass(frozen=True)
class SelectionSpec:
    as_of: str
    direction: str = "buy"
    model_version: Optional[str] = None
    top_k: int = 10
    min_score: Optional[float] = 0.6
    signal_lookback_bars: int = 20
    history_days: int = 900
    stale_days: int = 20
    codes: List[str] = field(default_factory=list)
    codes_file: Optional[str] = None
    universe: Optional[str] = None
    limit: Optional[int] = None

return SelectionSpec(
    as_of=str(selection_payload["as_of"]),
    direction=str(selection_payload.get("direction", "buy")),
    model_version=selection_payload.get("model_version"),
    top_k=int(selection_payload.get("top_k", 10)),
    min_score=selection_payload.get("min_score", 0.6),
    signal_lookback_bars=int(selection_payload.get("signal_lookback_bars", 20)),
    history_days=int(selection_payload.get("history_days", 900)),
    stale_days=int(selection_payload.get("stale_days", 20)),
    codes=_normalize_codes(selection_payload.get("codes")),
    codes_file=selection_payload.get("codes_file"),
    universe=(
        str(selection_payload.get("universe")).strip().lower()
        if selection_payload.get("universe") is not None
        else None
    ),
    limit=int(selection_payload["limit"]) if selection_payload.get("limit") is not None else None,
)

# AutoResearch/Selection.py
return SimpleNamespace(
    model_version=selection.model_version,
    top_k=selection.top_k,
    min_score=selection.min_score,
    signal_lookback_bars=selection.signal_lookback_bars,
    limit=selection.limit,
    codes=",".join(selection.codes) if selection.codes else None,
    codes_file=selection.codes_file,
    universe=selection.universe,
)
```

- [ ] **Step 4: Re-run the targeted tests and verify they pass**

Run: `python3 -m pytest tests/test_autoresearch.py -k "named_universe or runtime_args_passes_universe" -v`
Expected: PASS with both tests green.

- [ ] **Step 5: Commit the schema/runtime slice**

```bash
git add AutoResearch/Spec.py AutoResearch/Selection.py tests/test_autoresearch.py
git commit -m "feat: add named selection universe wiring"
```

### Task 2: Add `hs300` universe resolution and precedence to the recommendation CLI

**Files:**
- Modify: `App/generate_stock_recommendations.py`
- Create: `tests/test_generate_stock_recommendations.py`

- [ ] **Step 1: Write the failing parser and universe-resolution tests**

```python
import unittest
from types import SimpleNamespace
from unittest import mock

from App import generate_stock_recommendations as reco

class FakeFrame:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for index, row in enumerate(self._rows):
            yield index, row

class RecommendationUniverseTests(unittest.TestCase):
    def test_parse_args_accepts_universe_flag(self):
        with mock.patch("sys.argv", ["prog", "--as-of", "2024-12-31", "--universe", "hs300"]):
            args = reco.parse_args()

        self.assertEqual(args.universe, "hs300")

    def test_load_universe_prefers_codes_over_codes_file_and_universe(self):
        args = SimpleNamespace(
            codes="600519,000333",
            codes_file="codes.txt",
            universe="hs300",
            limit=20,
        )

        result = reco.load_universe(args)

        self.assertEqual(result, [("600519", ""), ("000333", "")])

    def test_load_universe_prefers_codes_file_over_named_universe(self):
        args = SimpleNamespace(
            codes=None,
            codes_file="codes.txt",
            universe="hs300",
            limit=20,
        )

        with mock.patch.object(
            reco,
            "load_codes_from_file",
            return_value=[("600036", "招商银行")],
        ) as load_codes_from_file_mock, mock.patch.object(
            reco,
            "load_named_universe",
            return_value=[("600519", "贵州茅台")],
        ) as load_named_universe_mock:
            result = reco.load_universe(args)

        self.assertEqual(result, [("600036", "招商银行")])
        load_codes_from_file_mock.assert_called_once()
        load_named_universe_mock.assert_not_called()

    def test_load_universe_resolves_hs300_from_csindex(self):
        args = SimpleNamespace(codes=None, codes_file=None, universe="hs300", limit=None)
        fake_ak = SimpleNamespace(
            index_stock_cons_csindex=mock.Mock(
                return_value=FakeFrame(
                    [
                        {"成分券代码": "600519", "成分券名称": "贵州茅台"},
                        {"成分券代码": "000333", "成分券名称": "美的集团"},
                    ]
                )
            )
        )

        with mock.patch.object(reco, "ak", fake_ak):
            result = reco.load_universe(args)

        self.assertEqual(result, [("600519", "贵州茅台"), ("000333", "美的集团")])

    def test_load_universe_rejects_unknown_named_universe(self):
        args = SimpleNamespace(codes=None, codes_file=None, universe="unknown", limit=None)

        with self.assertRaisesRegex(ValueError, "Unsupported universe"):
            reco.load_universe(args)

    def test_get_hs300_stocks_raises_when_provider_returns_empty_rows(self):
        fake_ak = SimpleNamespace(index_stock_cons_csindex=mock.Mock(return_value=FakeFrame([])))

        with mock.patch.object(reco, "ak", fake_ak):
            with self.assertRaisesRegex(RuntimeError, "resolved to zero constituents"):
                reco.get_hs300_stocks()
```

- [ ] **Step 2: Run the new unit tests and verify they fail**

Run: `python3 -m pytest tests/test_generate_stock_recommendations.py -v`
Expected: FAIL because the parser has no `--universe` flag and `load_universe()` has no named-universe branch.

- [ ] **Step 3: Implement the minimal named-universe loader**

```python
# App/generate_stock_recommendations.py
parser.add_argument("--universe", default=None, help="命名股票池，目前支持 hs300")

def load_universe(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.codes:
        return [(normalize_code(code), "") for code in args.codes.split(",") if code.strip()]
    if args.codes_file:
        return load_codes_from_file(Path(args.codes_file))
    if getattr(args, "universe", None):
        return load_named_universe(str(args.universe))
    return get_tradable_stocks(limit=args.limit)

def load_named_universe(universe: str) -> List[Tuple[str, str]]:
    normalized = universe.strip().lower()
    if normalized == "hs300":
        return get_hs300_stocks()
    raise ValueError(f"Unsupported universe: {universe}")

def get_hs300_stocks() -> List[Tuple[str, str]]:
    if ak is None:
        raise RuntimeError("akshare is required when using universe=hs300")

    df = ak.index_stock_cons_csindex(symbol="000300")
    rows: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        code = normalize_code(str(row.get("成分券代码", "")))
        name = str(row.get("成分券名称", "")).strip()
        if code:
            rows.append((code, name))

    if not rows:
        raise RuntimeError("Universe hs300 resolved to zero constituents.")

    return rows
```

Implementation notes:
- Keep precedence exactly `codes > codes_file > universe > tradable fallback`.
- Do not silently fall back to tradable A-shares when `universe="hs300"` resolution fails.
- Use AkShare's documented `index_stock_cons_csindex(symbol="000300")` interface for CSI 300 constituents.

- [ ] **Step 4: Re-run the new unit tests and verify they pass**

Run: `python3 -m pytest tests/test_generate_stock_recommendations.py -v`
Expected: PASS with parser, precedence, and named-universe resolution covered.

- [ ] **Step 5: Commit the universe-resolution slice**

```bash
git add App/generate_stock_recommendations.py tests/test_generate_stock_recommendations.py
git commit -m "feat: add hs300 universe resolution"
```

### Task 3: Cover the mocked selection flow and add the canonical HS300 experiment spec

**Files:**
- Modify: `tests/test_autoresearch.py`
- Create: `experiments/autoresearch/hs300_daily_selection.json`

- [ ] **Step 1: Write the failing mocked selection-flow test and baseline spec**

```python
from AutoResearch.Selection import SelectionRunResult, _make_runtime_args, run_selection_experiment
from AutoResearch.Spec import ExperimentSpec, SelectionSpec, expand_sweep_spec, load_experiment_spec, load_pipeline_spec

def test_hs300_daily_selection_spec_declares_named_universe(self):
    spec = load_experiment_spec(Path("experiments/autoresearch/hs300_daily_selection.json"))

    self.assertEqual(spec.name, "hs300-daily-selection")
    self.assertEqual(spec.selection.universe, "hs300")
    self.assertEqual(spec.selection.codes, [])

def test_run_selection_experiment_supports_hs300_universe(self):
    spec = ExperimentSpec(
        name="hs300-daily-selection",
        selection=SelectionSpec(
            as_of="2024-12-31",
            universe="hs300",
            top_k=2,
            min_score=0.6,
        ),
    )

    with mock.patch(
        "App.generate_stock_recommendations.load_model",
        return_value=(object(), {"version": "demo-v1"}),
    ), mock.patch(
        "App.generate_stock_recommendations.resolve_runtime_config",
        return_value={
            "feature_config": {},
            "top_k": 2,
            "min_score": 0.6,
            "signal_lookback_bars": 20,
        },
    ), mock.patch(
        "App.generate_stock_recommendations.load_universe",
        return_value=[("600519", "贵州茅台"), ("000333", "美的集团")],
    ) as load_universe_mock, mock.patch(
        "App.generate_stock_recommendations.load_chan_pool",
        return_value=(["chan-1"], {"600519": "贵州茅台"}, []),
    ), mock.patch(
        "App.generate_stock_recommendations.build_output_rows",
        return_value=[
            {
                "rank": 1,
                "as_of": "2024-12-31",
                "code": "600519",
                "name": "贵州茅台",
                "score": 0.88,
                "signal_time": "2024-12-30",
                "signal_type": "1",
                "signal_price": 1788.0,
                "signal_idx": 123,
                "model_version": "demo-v1",
            }
        ],
    ), mock.patch(
        "ML.FeatureEngine.BSPFeatureExtractor.BSPFeatureExtractor",
        return_value=object(),
    ), mock.patch("ML.Prediction.Predictor.Predictor") as predictor_cls:
        predictor_cls.return_value.rank_stock_pool.return_value = [
            {"code": "600519", "score": 0.88, "bsp": object()}
        ]
        result = run_selection_experiment(spec)

    load_universe_args = load_universe_mock.call_args.args[0]
    self.assertEqual(load_universe_args.universe, "hs300")
    self.assertEqual(result.summary["universe_size"], 2)
    self.assertEqual(result.summary["recommendation_count"], 1)
```

```json
{
  "name": "hs300-daily-selection",
  "description": "AutoResearch stock-selection run over the current HS300 universe using the existing ML predictor.",
  "tags": ["hs300", "daily", "stock-selection"],
  "selection": {
    "as_of": "2024-12-31",
    "direction": "buy",
    "top_k": 10,
    "min_score": 0.6,
    "signal_lookback_bars": 20,
    "history_days": 900,
    "stale_days": 20,
    "universe": "hs300"
  },
  "portfolio_backtest": {
    "enabled": true,
    "top_k": 3,
    "score_threshold": 0.55,
    "rebalance_bars": 5,
    "signal_lookback_bars": 20
  },
  "storage": {
    "root_dir": "AutoResearch/results",
    "leaderboard_filename": "leaderboard.md"
  }
}
```

- [ ] **Step 2: Run the targeted AutoResearch tests and verify they fail**

Run: `python3 -m pytest tests/test_autoresearch.py -k "hs300_daily_selection_spec or supports_hs300_universe" -v`
Expected: FAIL because `experiments/autoresearch/hs300_daily_selection.json` does not exist yet.

- [ ] **Step 3: Implement the minimal integration changes**

```python
# tests/test_autoresearch.py
from AutoResearch.Selection import SelectionRunResult, _make_runtime_args, run_selection_experiment
from AutoResearch.Spec import ExperimentSpec, SelectionSpec, expand_sweep_spec, load_experiment_spec, load_pipeline_spec
```

`experiments/autoresearch/hs300_daily_selection.json`:

```json
{
  "name": "hs300-daily-selection",
  "description": "AutoResearch stock-selection run over the current HS300 universe using the existing ML predictor.",
  "tags": ["hs300", "daily", "stock-selection"],
  "selection": {
    "as_of": "2024-12-31",
    "direction": "buy",
    "top_k": 10,
    "min_score": 0.6,
    "signal_lookback_bars": 20,
    "history_days": 900,
    "stale_days": 20,
    "universe": "hs300"
  },
  "portfolio_backtest": {
    "enabled": true,
    "top_k": 3,
    "score_threshold": 0.55,
    "rebalance_bars": 5,
    "signal_lookback_bars": 20
  },
  "storage": {
    "root_dir": "AutoResearch/results",
    "leaderboard_filename": "leaderboard.md"
  }
}
```

Implementation note:
- Keep the canonical HS300 experiment parallel to `baseline_daily_selection.json`; do not mutate the existing baseline spec.

- [ ] **Step 4: Re-run the targeted AutoResearch tests and verify they pass**

Run: `python3 -m pytest tests/test_autoresearch.py -k "hs300_daily_selection_spec or supports_hs300_universe" -v`
Expected: PASS with the mocked selection flow confirming `universe="hs300"` reaches `load_universe()` and produces summary output.

- [ ] **Step 5: Commit the experiment and integration-test slice**

```bash
git add tests/test_autoresearch.py experiments/autoresearch/hs300_daily_selection.json
git commit -m "test: cover hs300 selection flow"
```

### Task 4: Run full verification and smoke-test the new entrypoint

**Files:**
- Modify: none
- Test: `tests/test_autoresearch.py`
- Test: `tests/test_generate_stock_recommendations.py`

- [ ] **Step 1: Run the focused automated suite**

Run: `python3 -m pytest tests/test_autoresearch.py tests/test_generate_stock_recommendations.py -v`
Expected: PASS with all HS300-related tests green and no regressions in the touched selection flow.

- [ ] **Step 2: Run a CLI smoke test if AkShare is available in the environment**

Run: `python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/hs300_daily_selection.json`
Expected: If `akshare` and model artifacts are available, the run creates a new result directory under `AutoResearch/results/hs300-daily-selection/runs/` and writes recommendation artifacts.

- [ ] **Step 3: If the smoke test cannot run locally, capture the exact blocker instead of guessing**

Expected blockers to record:
- `ModuleNotFoundError: No module named 'akshare'`
- missing model artifacts under `./models`
- upstream market-data fetch failure from the CSI constituent endpoint

- [ ] **Step 4: Inspect git diff and result artifacts before final handoff**

Run: `git status --short`
Expected: only the planned code, tests, and experiment spec are modified or added.

- [ ] **Step 5: Commit the final verification checkpoint**

```bash
git add AutoResearch/Spec.py AutoResearch/Selection.py App/generate_stock_recommendations.py tests/test_autoresearch.py tests/test_generate_stock_recommendations.py experiments/autoresearch/hs300_daily_selection.json
git commit -m "feat: add hs300 autoresearch selection support"
```
