# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `src/chan_py/`, with subpackages such as `core` for Chan element computation, `analysis` for derived metrics, and `visualization` for rendering helpers. Legacy algorithm modules (`Bi/`, `Seg/`, `ZS/`, `KLine/`, `BuySellPoint/`) remain at the repo root for direct imports used by `main.py`. Supporting tooling is grouped in `Script/` (dependency pinning), `DataAPI/` (market data connectors), and `Plot/` for matplotlib drivers. Tests are organized by scope inside `tests/` (`unit/`, `integration/`, `performance/`), while the optional FastAPI + Vue web console lives under `web/`.

## Build, Test, and Development Commands
Create a Python 3.11 environment and install shared dependencies:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r Script/requirements.txt
```
Run the CLI demo to verify data access and plotting:
```bash
python3 main.py
```
Execute the automated test suite (covers unit and integration layers):
```bash
pytest
```
Web contributors should start the dashboard with `cd web && ./start_uv.sh` (uses uv to launch both backend and frontend).

## Coding Style & Naming Conventions
Follow standard Python style: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and module names that stay lowercase with underscores. Prefer explicit type hints in new public APIs, especially within `src/chan_py/core` and `analysis`. Format code with Black (`black .`) and keep it lint-clean with Ruff (`ruff check . --fix`) before opening a PR. Plotting configs and strategy descriptors should be stored as dictionaries named `*_config` to mirror existing patterns.

## Testing Guidelines
Add or update unit tests alongside code in `tests/unit/` and mirror cross-module workflows in `tests/integration/`. Use descriptive test names like `test_seglist_respects_config` to match the current pytest style. When benchmarking new algorithms, gate the code behind a flag and place performance checks in `tests/performance/`. Always run `pytest` locally; include `-k` filters when iterating, but finish with a full run before submission.

## Commit & Pull Request Guidelines
Commits follow Conventional Commits (`feat(web):`, `fix:`, `refactor(core):`), so keep the type + optional scope format and write imperative, present-tense subjects. Squash fixups before pushing. Pull requests should describe the change, list any new commands or configs, and reference related issues. When UI or plotting output changes, attach before/after screenshots. Confirm tests run clean and call out any follow-up work in a checklist.

## Security & Configuration Tips
Do not commit API tokens or account details; `DataAPI/` loaders expect credentials from environment variables or local config files added to `.gitignore`. Review new dependencies for licensing and trading restrictions before adding them to `Script/requirements.txt` or `web/requirements.txt`. When sharing notebooks or scripts, strip cached data to avoid leaking proprietary market feeds.
