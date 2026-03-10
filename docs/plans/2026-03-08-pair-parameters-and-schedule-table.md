# Pair Parameter And Schedule Table Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `sim_script/pd_mmw_template_ap_stats.py` so it prints each WiFi/BLE pair's concrete parameters and emits a direct WiFi/BLE schedule table arranged by the minimal scheduling slot index, while keeping the existing office summary and solver behavior intact.

**Architecture:** Keep all formatting logic in `sim_script/pd_mmw_template_ap_stats.py` as pure helper functions so the script remains easy to test without running the full solver. Build the new pair-level table from `env` arrays plus `z_vec`, then derive a schedule table by grouping pair rows on assigned scheduling slot `z` and ordering rows by ascending slot index. Fix the existing script argument/config mismatch first so all downstream output reflects the actual CLI inputs instead of hard-coded values.

**Tech Stack:** Python 3.10, NumPy, subprocess-based smoke tests, pytest-style test files under `sim_script/tests`

---

### Task 1: Fix Script Argument Consistency Before Adding More Output

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:58-95`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

Add a regression test to `sim_script/tests/test_pd_mmw_template_ap_stats_run.py` that proves the script respects both `--cell-size` and `--pair-density` from the command line.

```python
import pathlib
import subprocess
import sys


def test_script_respects_cli_cell_size_and_pair_density():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "1",
            "--pair-density",
            "0.1",
            "--seed",
            "7",
            "--mmw-nit",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "n_office = 1" in proc.stdout
    assert "pair_density_per_m2 = 0.1" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_pd_mmw_template_ap_stats_run import test_script_respects_cli_cell_size_and_pair_density

test_script_respects_cli_cell_size_and_pair_density()
print("PASS")
```

`PY`

Expected: FAIL because the script currently parses `--cell-size` with one default but hard-codes `cell_size=2` inside `env(...)`, so printed `n_office` does not match the CLI input.

**Step 3: Write minimal implementation**

Update `sim_script/pd_mmw_template_ap_stats.py` so `env(...)` uses `cell_size=args.cell_size` instead of the hard-coded `2`. Keep `pair_density_per_m2=args.pair_density` unchanged. Re-check the parser defaults so they match the intended interactive defaults instead of hidden hard-coded behavior.

```python
e = env(
    cell_edge=7.0,
    cell_size=args.cell_size,
    pair_density_per_m2=args.pair_density,
    ...
)
```

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "fix: make ap stats script honor cli parameters"
```

### Task 2: Add Pair-Level Output Helpers And Tests

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

Extend `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py` with a pure-function test that builds pair rows from arrays and checks both WiFi-specific and BLE-specific fields are present.

```python
import numpy as np

from sim_script.pd_mmw_template_ap_stats import build_pair_parameter_rows


def test_build_pair_parameter_rows_contains_wifi_and_ble_fields():
    rows = build_pair_parameter_rows(
        pair_office_id=np.array([0, 0]),
        pair_radio_type=np.array([0, 1]),
        pair_channel=np.array([6, 12]),
        pair_priority=np.array([3.0, 1.0]),
        pair_ble_anchor_slot=np.array([0, 4]),
        pair_ble_ci_slots=np.array([0, 80]),
        pair_ble_ce_slots=np.array([0, 10]),
        pair_ble_ce_feasible=np.array([True, True]),
        z_vec=np.array([2, 5]),
        slot_time=1.25e-4,
        wifi_id=0,
        ble_id=1,
    )

    assert rows[0]["pair_id"] == 0
    assert rows[0]["radio"] == "wifi"
    assert rows[0]["schedule_slot"] == 2
    assert rows[0]["channel"] == 6
    assert rows[0]["ble_ci_slots"] is None

    assert rows[1]["pair_id"] == 1
    assert rows[1]["radio"] == "ble"
    assert rows[1]["schedule_slot"] == 5
    assert rows[1]["ble_ci_slots"] == 80
    assert rows[1]["ble_ce_slots"] == 10
    assert rows[1]["ble_anchor_slot"] == 4
    assert rows[1]["ble_ci_ms"] == 10.0
    assert rows[1]["ble_ce_ms"] == 1.25
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_pd_mmw_template_ap_stats_logic import test_build_pair_parameter_rows_contains_wifi_and_ble_fields

test_build_pair_parameter_rows_contains_wifi_and_ble_fields()
print("PASS")
```

`PY`

Expected: FAIL with `ImportError` or `AttributeError` because `build_pair_parameter_rows` does not exist yet.

**Step 3: Write minimal implementation**

Add a pure helper in `sim_script/pd_mmw_template_ap_stats.py` that returns one dictionary per pair with the exact output columns you want to see. Keep formatting code separate from data-building code.

```python
def build_pair_parameter_rows(
    pair_office_id,
    pair_radio_type,
    pair_channel,
    pair_priority,
    pair_ble_anchor_slot,
    pair_ble_ci_slots,
    pair_ble_ce_slots,
    pair_ble_ce_feasible,
    z_vec,
    slot_time,
    wifi_id,
    ble_id,
):
    rows = []
    for pair_id in range(pair_radio_type.shape[0]):
        is_ble = int(pair_radio_type[pair_id]) == int(ble_id)
        rows.append(
            {
                "pair_id": int(pair_id),
                "office_id": int(pair_office_id[pair_id]),
                "radio": "ble" if is_ble else "wifi",
                "channel": int(pair_channel[pair_id]),
                "priority": float(pair_priority[pair_id]),
                "schedule_slot": int(z_vec[pair_id]),
                "schedule_time_ms": float(z_vec[pair_id] * slot_time * 1e3),
                "ble_anchor_slot": int(pair_ble_anchor_slot[pair_id]) if is_ble else None,
                "ble_ci_slots": int(pair_ble_ci_slots[pair_id]) if is_ble else None,
                "ble_ce_slots": int(pair_ble_ce_slots[pair_id]) if is_ble else None,
                "ble_ci_ms": float(pair_ble_ci_slots[pair_id] * slot_time * 1e3) if is_ble else None,
                "ble_ce_ms": float(pair_ble_ce_slots[pair_id] * slot_time * 1e3) if is_ble else None,
                "ble_ce_feasible": bool(pair_ble_ce_feasible[pair_id]) if is_ble else None,
            }
        )
    return rows
```

Also add a thin wrapper like `compute_pair_parameter_rows(e, z_vec)` that reads directly from `env`.

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add pair parameter row builder for ap stats script"
```

### Task 3: Add A Minimal-Slot Schedule Table Builder And Tests

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

Add a test for a pure schedule-table helper that sorts rows by ascending scheduling slot and groups WiFi/BLE assignments into a direct printable table.

```python
from sim_script.pd_mmw_template_ap_stats import build_schedule_rows


def test_build_schedule_rows_orders_by_min_schedule_slot():
    pair_rows = [
        {"pair_id": 3, "radio": "ble", "schedule_slot": 5, "office_id": 1, "channel": 2},
        {"pair_id": 0, "radio": "wifi", "schedule_slot": 1, "office_id": 0, "channel": 6},
        {"pair_id": 2, "radio": "wifi", "schedule_slot": 5, "office_id": 1, "channel": 1},
    ]

    rows = build_schedule_rows(pair_rows)

    assert rows[0]["schedule_slot"] == 1
    assert rows[0]["pair_ids"] == [0]
    assert rows[0]["wifi_pair_ids"] == [0]
    assert rows[0]["ble_pair_ids"] == []

    assert rows[1]["schedule_slot"] == 5
    assert rows[1]["pair_ids"] == [2, 3]
    assert rows[1]["wifi_pair_ids"] == [2]
    assert rows[1]["ble_pair_ids"] == [3]
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_pd_mmw_template_ap_stats_logic import test_build_schedule_rows_orders_by_min_schedule_slot

test_build_schedule_rows_orders_by_min_schedule_slot()
print("PASS")
```

`PY`

Expected: FAIL because `build_schedule_rows` does not exist yet.

**Step 3: Write minimal implementation**

Implement a helper in `sim_script/pd_mmw_template_ap_stats.py` that:
- groups pair rows by `schedule_slot`
- sorts slots ascending
- sorts pair IDs ascending within each slot
- emits one row per slot
- includes WiFi/BLE-specific pair ID lists and counts

```python
def build_schedule_rows(pair_rows):
    slot_map = {}
    for row in pair_rows:
        slot = int(row["schedule_slot"])
        slot_map.setdefault(slot, []).append(row)

    rows = []
    for slot in sorted(slot_map):
        grouped = sorted(slot_map[slot], key=lambda r: int(r["pair_id"]))
        wifi_ids = [int(r["pair_id"]) for r in grouped if r["radio"] == "wifi"]
        ble_ids = [int(r["pair_id"]) for r in grouped if r["radio"] == "ble"]
        rows.append(
            {
                "schedule_slot": slot,
                "pair_ids": [int(r["pair_id"]) for r in grouped],
                "wifi_pair_ids": wifi_ids,
                "ble_pair_ids": ble_ids,
                "pair_count": len(grouped),
                "wifi_pair_count": len(wifi_ids),
                "ble_pair_count": len(ble_ids),
            }
        )
    return rows
```

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add minimal-slot schedule table builder"
```

### Task 4: Print Pair Parameter Table And Schedule Table In Script Output

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Modify: `sim_script/tests/test_pd_mmw_ble_ci_summary.py` (only if existing assertions become too weak)

**Step 1: Write the failing test**

Extend the smoke test in `sim_script/tests/test_pd_mmw_template_ap_stats_run.py` so it checks for two new sections:
- a pair-parameter table header
- a schedule table header ordered by scheduling slot

```python
def test_script_runs_and_prints_pair_and_schedule_headers():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "2",
            "--pair-density",
            "0.05",
            "--seed",
            "123",
            "--mmw-nit",
            "5",
            "--mmw-eta",
            "0.05",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pair_id,office_id,radio,channel,priority,schedule_slot,schedule_time_ms" in proc.stdout
    assert "schedule_slot,pair_ids,wifi_pair_ids,ble_pair_ids,pair_count,wifi_pair_count,ble_pair_count" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_pd_mmw_template_ap_stats_run import test_script_runs_and_prints_pair_and_schedule_headers

test_script_runs_and_prints_pair_and_schedule_headers()
print("PASS")
```

`PY`

Expected: FAIL because the script currently prints only global summary, BLE summary, and office summary.

**Step 3: Write minimal implementation**

Add two print helpers in `sim_script/pd_mmw_template_ap_stats.py`:
- `print_pair_parameter_rows(rows)`
- `print_schedule_rows(rows)`

Then, after `z_vec` is computed and before office summary output, call:

```python
pair_rows = compute_pair_parameter_rows(e, z_vec)
print_pair_parameter_rows(pair_rows)

schedule_rows = build_schedule_rows(pair_rows)
print_schedule_rows(schedule_rows)
```

Required print contract:
- Pair table must include each pair's concrete parameters, including BLE-only fields as blank or `NA` for WiFi rows.
- Schedule table must be sorted by increasing `schedule_slot`.
- Rows for a given slot must list the pair IDs assigned to that slot, split into WiFi and BLE groups.
- Keep the existing office summary section after the new sections.

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/tests/test_pd_mmw_ble_ci_summary.py
git commit -m "feat: print pair parameter and schedule tables"
```

### Task 5: Perform End-To-End Verification And Clarify Output Semantics

**Files:**
- Modify: `README.md` (only if the script output format is documented there)
- Test: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: Run a full smoke check with explicit small parameters**

Run: `python -u sim_script/pd_mmw_template_ap_stats.py --cell-size 2 --pair-density 0.05 --seed 3 --mmw-nit 5`
Expected:
- Exit code `0`
- Output contains the pair table header
- Output contains the schedule table header
- Schedule table rows are printed in ascending `schedule_slot`
- Existing BLE timing summary and office summary are still present

**Step 2: Review the output for semantic clarity**

Check that the column names make the distinction between these concepts obvious:
- BLE protocol timing slots: `ble_anchor_slot`, `ble_ci_slots`, `ble_ce_slots`
- Scheduler assignment: `schedule_slot`
- Physical time conversions: `schedule_time_ms`, `ble_ci_ms`, `ble_ce_ms`

If any header is still ambiguous, rename it before merging. Good final headers would look like:

```text
pair_id,office_id,radio,channel,priority,schedule_slot,schedule_time_ms,ble_anchor_slot,ble_ci_slots,ble_ci_ms,ble_ce_slots,ble_ce_ms,ble_ce_feasible
schedule_slot,pair_ids,wifi_pair_ids,ble_pair_ids,pair_count,wifi_pair_count,ble_pair_count
```

**Step 3: Update docs only if needed**

If `README.md` or another checked-in doc mentions the old output format, update it with one short section describing the new pair table and schedule table outputs. Otherwise skip the docs edit.

Example doc text:

```markdown
The AP stats script now prints three levels of output:
1. Pair-level parameter table with scheduling slot and BLE timing fields.
2. Schedule table grouped by scheduling slot.
3. Office-level aggregate summary.
```

**Step 4: Commit**

```bash
git add README.md sim_script/pd_mmw_template_ap_stats.py sim_script/tests
git commit -m "docs: clarify pair and schedule output semantics"
```
