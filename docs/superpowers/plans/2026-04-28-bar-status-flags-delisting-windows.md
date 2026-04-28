# Bar Status Flags Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate ClickHouse status flag table for daily `1m` bar statuses, then load the reviewed 2025 delisting-related low-price windows without changing `market.bars`.

**Architecture:** Keep anomaly detection and business-state labeling separate. Add a new `market.bar_status_flags` table in the base ClickHouse schema, then add a small idempotent loader script that replays a reviewed set of hard-coded daily windows into that table. Tests should cover schema SQL, row expansion, and idempotent delete/insert query generation.

**Tech Stack:** Python 3, ClickHouse SQL, `unittest`

---

### Task 1: Add failing tests

**Files:**
- Create: `tests/test_flag_delisting_status_windows.py`
- Modify: `tests/test_clickhouse_infra.py`

- [ ] **Step 1: Write the failing test**

Add tests that assert:
- the new module exposes schema and query builders
- the default reviewed windows include the expected statuses
- daily rows expand correctly from date windows
- the infra init SQL includes `market.bar_status_flags`

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_flag_delisting_status_windows tests.test_clickhouse_infra`

Expected: FAIL because the new module and schema do not exist yet.

### Task 2: Implement the schema and loader

**Files:**
- Create: `App/flag_delisting_status_windows.py`
- Modify: `infra/clickhouse/init/001_create_market.sql`

- [ ] **Step 1: Write minimal implementation**

Implement:
- a `market.bar_status_flags` schema
- reviewed default delisting windows for the 11 symbols
- date expansion helpers
- idempotent delete + insert query builders
- a CLI with dry-run default and `--execute`

- [ ] **Step 2: Run targeted tests**

Run: `python3 -m unittest tests.test_flag_delisting_status_windows tests.test_clickhouse_infra`

Expected: PASS

### Task 3: Apply and verify

**Files:**
- Use: `App/flag_delisting_status_windows.py`

- [ ] **Step 1: Dry-run the loader**

Run: `python3 App/flag_delisting_status_windows.py`

Expected: summary of rows and statuses, no writes.

- [ ] **Step 2: Execute the loader**

Run: `python3 App/flag_delisting_status_windows.py --execute`

Expected: flags inserted into `market.bar_status_flags`.

- [ ] **Step 3: Verify inserted data**

Run queries that confirm:
- the table exists
- expected status counts exist for `2025`
- sample symbols (`600804.SH`, `600462.SH`) have the right daily flags

