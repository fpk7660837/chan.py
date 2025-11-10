# Repository Guidelines

## First Rule
Always response by chinese.

## Project Structure & Module Organization
Core Chan-computation modules live at the repository root (`Chan.py`, `ChanConfig.py`, `Bi/`, `Seg/`, `ZS/`, `KLine/`, `BuySellPoint/`) and are imported directly by `main.py`. Supporting utilities and experiments are gathered under `src/` and validated by suites in `tests/`. The browser console uses a FastAPI backend (`web/backend`) plus a single-page React bundle (`web/frontend/index.html`) that renders draggable sidebars, multi-period charts, and indicator overlays via KLineCharts.

## Build, Test, and Development Commands
Set up Python 3.11 and install dependencies:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r Script/requirements.txt
```
Run `python3 main.py` to confirm Chan data access. Execute the automated suites with `pytest` before every pull request. Launch the web console by running `cd web && ./start_uv.sh`; this spins up FastAPI and serves the in-browser Babel build. Refresh the browser after editing `web/frontend/index.html`.

## Coding Style & Naming Conventions
Backend code follows PEP 8: four-space indentation, `snake_case` identifiers, and PascalCase classes. Prefer explicit type hints on public APIs and keep configuration dictionaries suffixed with `_config`. For the web layer, use functional React components, group inline styles at the top of the script, favour Ant Design primitives, and add terse comments only when overlay math or chart wiring is non-obvious.

## Testing Guidelines
Place new unit tests beside the touched modules in `tests/unit/`, and mirror multi-level workflows in `tests/integration/`. When adjusting ChanConfig parameters or the result trimming logic, craft fixtures that assert counts (Bi, Seg, ZS, Buy/Sell) stay aligned with the returned K-line window. Run `pytest` (optionally narrowed with `-k`) locally, then finish with a full run before submission.

## Commit & Pull Request Guidelines
Follow Conventional Commit prefixes (`feat(web):`, `fix(core):`, etc.) with concise, imperative subjects and squash fixups ahead of pushing. Pull requests should summarise behaviour changes, list new commands or configuration toggles (e.g., K线数量上限), reference related issues, and attach before/after screenshots for UI updates. Confirm `pytest` passes and note any follow-up tasks in a checklist.

## Web Interface Notes
Expose dashboard options through the shared context helpers so selections persist in `localStorage`. Keep sidebar dimensions and K-line limits responsive, and describe tooltip or overlay adjustments in your PR to help reviewers verify hover behaviour easily.
