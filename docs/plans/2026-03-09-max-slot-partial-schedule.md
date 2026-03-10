# Max Slot Partial Schedule Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a maximum-slot safeguard so the scheduler stops at a configured slot cap, returns the pairs that were successfully scheduled so far, and explicitly reports the unscheduled pairs.

**Architecture:** Keep the existing binary-search + rounding flow, but add a bounded search mode that never exceeds a user-provided maximum slot count. When the cap is reached or macrocycle placement cannot fit all pairs, return a partial schedule result object and thread that through the script output and CSV generation. Preserve the current full-success behavior when the cap is not hit.

**Tech Stack:** Python 3.10, NumPy, SciPy, existing `binary_search_relaxation`, `mmw`, script CSV output helpers, inline test execution in the `sig-sdp` conda environment.

---

### Task 1: Define partial-result behavior at the search layer

**Files:**
- Modify: `sim_src/alg/binary_search_relaxation.py`
- Test: `sim_src/test/test_binary_search_partial_schedule.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_src.alg.binary_search_relaxation import binary_search_relaxation


class _StubAlg:
    def run_with_state(self, bs_iteration, Z, state):
        return True, np.eye(state[0].shape[0])

    def rounding(self, Z, gX, state, user_priority=None, slot_mask=None):
        if Z < 3:
            return np.array([0, 1, 0]), Z, 1
        return np.array([0, 1, 2]), Z, 0


def test_binary_search_returns_partial_result_when_max_slot_cap_blocks_full_schedule():
    state = (np.eye(3), np.eye(3), np.ones(3))
    bs = binary_search_relaxation()
    bs.feasibility_check_alg = _StubAlg()
    bs.max_slot_cap = 2
    z_vec, z_fin, remainder, partial = bs.run(state)
    assert z_fin == 2
    assert remainder == 1
    assert partial["scheduled_pair_ids"] == [0, 1]
    assert partial["unscheduled_pair_ids"] == [2]
```

**Step 2: Run test to verify it fails**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because `binary_search_relaxation.run()` does not return partial metadata yet.

**Step 3: Write minimal implementation**

```python
# binary_search_relaxation.py
self.max_slot_cap = None

# clamp search upper bound by max_slot_cap
right = min(right, self.max_slot_cap) if self.max_slot_cap is not None else right

# when returning, compute scheduled/unscheduled ids from z_vec + remainder
partial = {
    "scheduled_pair_ids": ...,
    "unscheduled_pair_ids": ...,
    "slot_cap_hit": ...,
}
return z_vec, Z, rem, partial
```

**Step 4: Run test to verify it passes**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_src/alg/binary_search_relaxation.py sim_src/test/test_binary_search_partial_schedule.py
git commit -m "feat: return partial schedule metadata when slot cap is hit"
```

### Task 2: Support partial macrocycle placement instead of hard failure

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:59-95`
- Test: `sim_script/tests/test_macrocycle_partial_assignment.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots


class _StubEnv:
    n_pair = 3
    pair_priority = np.array([3.0, 2.0, 1.0])
    RADIO_BLE = 1
    pair_radio_type = np.array([0, 0, 0])
    pair_ble_ce_feasible = np.array([True, True, True])

    def compute_macrocycle_slots(self):
        return 4

    def get_pair_period_slots(self):
        return np.array([4, 4, 4])

    def get_pair_width_slots(self):
        return np.array([2, 2, 2])

    def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
        occ = np.zeros(macrocycle_slots, dtype=bool)
        occ[start_slot:start_slot + 2] = True
        return occ


def test_assign_macrocycle_start_slots_can_return_partial_assignment():
    env = _StubEnv()
    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(env, np.array([0, 1, 2]), allow_partial=True)
    assert macro == 4
    assert starts[0] >= 0
    assert starts[1] >= 0
    assert starts[2] == -1
    assert unscheduled == [2]
```

**Step 2: Run test to verify it fails**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because `assign_macrocycle_start_slots()` currently raises `ValueError` instead of returning partial placement.

**Step 3: Write minimal implementation**

```python
def assign_macrocycle_start_slots(..., allow_partial=False):
    ...
    unscheduled = []
    ...
        if not assigned:
            if allow_partial:
                unscheduled.append(pair_id)
                continue
            raise ValueError(...)
    return assigned_starts, macrocycle_slots, occupancies, unscheduled
```

**Step 4: Run test to verify it passes**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_macrocycle_partial_assignment.py
git commit -m "feat: allow partial macrocycle placement under slot cap"
```

### Task 3: Thread max-slot CLI and partial outputs through the script

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_partial_schedule_run.py`

**Step 1: Write the failing test**

```python
import pathlib
import subprocess
import sys


def test_script_reports_partial_schedule_when_max_slot_cap_is_hit():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--cell-size", "2", "--pair-density", "0.2", "--max-slots", "3", "--mmw-nit", "5", "--seed", "9"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "partial_schedule = True" in proc.stdout
    assert "unscheduled_pair_ids" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because CLI has no `--max-slots` and script does not print partial schedule metadata.

**Step 3: Write minimal implementation**

```python
parser.add_argument("--max-slots", type=int, default=300, ...)
bs.max_slot_cap = args.max_slots
...
result = bs.run(...)
if len(result) == 4:
    z_vec_pref, Z_fin_mmw, remainder, partial = result
else:
    z_vec_pref, Z_fin_mmw, remainder = result
    partial = {...}
...
print("partial_schedule =", bool(partial["unscheduled_pair_ids"]))
print("scheduled_pair_ids =", partial["scheduled_pair_ids"])
print("unscheduled_pair_ids =", partial["unscheduled_pair_ids"])
```

**Step 4: Run test to verify it passes**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_partial_schedule_run.py
git commit -m "feat: expose max slot cap and partial schedule output"
```

### Task 4: Export unscheduled pairs and partial rows to CSV

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_partial_schedule_csv.py`

**Step 1: Write the failing test**

```python
import pathlib
import subprocess
import sys
import tempfile


def test_script_writes_unscheduled_pairs_csv_when_partial_schedule_occurs():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            [sys.executable, str(script), "--cell-size", "2", "--pair-density", "0.2", "--max-slots", "3", "--mmw-nit", "5", "--seed", "9", "--output-dir", tmpdir],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0
        assert (pathlib.Path(tmpdir) / "unscheduled_pairs.csv").exists()
```

**Step 2: Run test to verify it fails**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because no unscheduled CSV exists.

**Step 3: Write minimal implementation**

```python
unscheduled_rows = [row for row in pair_rows if row["pair_id"] in partial["unscheduled_pair_ids"]]
write_rows_to_csv(os.path.join(args.output_dir, "unscheduled_pairs.csv"), [...], unscheduled_rows)
```

Also ensure scheduled-only rows are what feed `wifi_ble_schedule.csv` so the schedule table reflects successful assignments only.

**Step 4: Run test to verify it passes**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_partial_schedule_csv.py
git commit -m "feat: export unscheduled pairs for partial schedules"
```

### Task 5: Regression sweep for full-success and partial-success modes

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: existing MMW and macrocycle tests as needed

**Step 1: Write the failing regression assertions**

```python
def test_full_success_run_does_not_report_partial_schedule():
    ...
    assert "partial_schedule = False" in proc.stdout
    assert "unscheduled_pair_ids = []" in proc.stdout
```

**Step 2: Run targeted regression tests to verify failures**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL until new output contract is implemented everywhere.

**Step 3: Write minimal implementation updates**

- Keep current full-success output unchanged except for the new summary lines.
- Ensure office stats and schedule table use scheduled pairs only.
- Ensure pair parameter CSV still includes all pairs, with unscheduled pairs marked by `schedule_slot = -1` and empty occupancy.

**Step 4: Run regression tests to verify pass**

Run: `conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS for both full-success and partial-success cases.

**Step 5: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "test: cover full and partial schedule output modes"
```

### Task 6: End-to-end verification in `sig-sdp`

**Files:**
- Modify: none unless failures require fixes
- Test: all touched test files and one script smoke run

**Step 1: Run focused verification**

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python - <<'PY'
from sim_src.test.test_binary_search_partial_schedule import test_binary_search_returns_partial_result_when_max_slot_cap_blocks_full_schedule
from sim_script.tests.test_macrocycle_partial_assignment import test_assign_macrocycle_start_slots_can_return_partial_assignment
from sim_script.tests.test_pd_mmw_partial_schedule_run import test_script_reports_partial_schedule_when_max_slot_cap_is_hit
from sim_script.tests.test_pd_mmw_partial_schedule_csv import test_script_writes_unscheduled_pairs_csv_when_partial_schedule_occurs

test_binary_search_returns_partial_result_when_max_slot_cap_blocks_full_schedule()
test_assign_macrocycle_start_slots_can_return_partial_assignment()
test_script_reports_partial_schedule_when_max_slot_cap_is_hit()
test_script_writes_unscheduled_pairs_csv_when_partial_schedule_occurs()
print('partial_schedule: PASS')
PY
```

Expected: `partial_schedule: PASS`

**Step 2: Run full-success smoke**

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --mmw-nit 5 --seed 7 --max-slots 300
```

Expected:
- exit code `0`
- `partial_schedule = False`
- CSV outputs still generated

**Step 3: Commit verification-only fixes if needed**

```bash
git add <touched-files>
git commit -m "fix: stabilize partial schedule verification"
```
