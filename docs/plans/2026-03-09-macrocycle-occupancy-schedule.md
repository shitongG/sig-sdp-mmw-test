# Macrocycle Occupancy Schedule Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the simulator so BLE CE is modeled as true continuous slot occupancy, WiFi also gets explicit periodic occupancy aligned to the same 1.25 ms base grid, and the script emits a realistic per-slot schedule over a macrocycle where all wireless flows are non-overlapping.

**Architecture:** The current code schedules one representative `schedule_slot` per pair and uses BLE CE only as a slot-eligibility mask, which is not sufficient to guarantee collision-free repetition over time. The new implementation should promote both WiFi and BLE into a common periodic occupancy model defined on the 0.125 ms base slot grid, compute a macrocycle from all active periods, and only accept assignments whose repeated occupied intervals do not overlap across that macrocycle. The script output should then be rebuilt from the expanded occupancy schedule instead of the current single-slot `z_vec` interpretation.

**Tech Stack:** Python 3.10, NumPy, SciPy sparse matrices, subprocess-based script smoke tests, test modules under `sim_script/tests`

---

### Task 1: Lock The New Timing Rules In Pure Env Tests

**Files:**
- Modify: `sim_script/tests/test_ble_ci_discrete_candidates.py`
- Modify: `sim_script/tests/test_ble_anchor_and_ce_window_mask.py`
- Create: `sim_script/tests/test_wifi_periodic_timing.py`
- Test: `sim_script/tests/test_wifi_periodic_timing.py`

**Step 1: Write the failing test**

Create `sim_script/tests/test_wifi_periodic_timing.py` to pin the new WiFi timing model and keep the BLE timing model explicit.

```python
import numpy as np
from sim_src.env.env import env


def test_wifi_period_candidates_follow_pow2_rule():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=1,
        radio_prob=(1.0, 0.0),
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
    )
    expected = np.array([16, 32], dtype=int)
    assert np.array_equal(e.wifi_period_quanta_candidates, expected)


def test_wifi_pairs_sample_periods_and_min_tx_duration():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=2,
        radio_prob=(1.0, 0.0),
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
        wifi_min_tx_time_s=5e-3,
    )
    wifi_idx = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    assert wifi_idx.size > 0
    period_quanta = np.rint((e.pair_wifi_period_slots[wifi_idx] * e.slot_time) / 1.25e-3).astype(int)
    assert np.all(np.isin(period_quanta, [16, 32]))
    assert np.all(e.pair_wifi_tx_slots[wifi_idx] >= 40)
```

Add a BLE regression in `sim_script/tests/test_ble_anchor_and_ce_window_mask.py` that proves the occupancy mask marks the whole CE block, not just the start slot.

```python
def test_ble_mask_marks_full_ce_window():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=7,
        radio_prob=(0.0, 1.0),
        ble_ci_exp_min=4,
        ble_ci_exp_max=4,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=2.5e-3,
        ble_phy_rate_bps=2e6,
    )
    Z = 300
    mask = e.build_slot_occupancy_mask(Z)
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    for k in ble_idx:
        if not e.user_ble_ce_feasible[k]:
            assert not mask[k].any()
            continue
        anchor = int(e.user_ble_anchor_slot[k] % e.user_ble_ci_slots[k])
        ce = int(e.user_ble_ce_slots[k])
        assert mask[k, anchor:anchor + ce].all()
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_wifi_periodic_timing import (
    test_wifi_period_candidates_follow_pow2_rule,
    test_wifi_pairs_sample_periods_and_min_tx_duration,
)

test_wifi_period_candidates_follow_pow2_rule()
test_wifi_pairs_sample_periods_and_min_tx_duration()
print("PASS")
```

`PY`

Expected: FAIL because WiFi periodic timing fields do not exist yet, and the occupancy-mask helper is still BLE-only and start-slot oriented.

**Step 3: Write minimal implementation**

Modify `sim_src/env/env.py` to introduce WiFi timing state and candidate ranges:

```python
wifi_min_tx_time_s=5e-3,
wifi_period_exp_min=4,
wifi_period_exp_max=5,
```

Add new fields:

```python
self.wifi_period_quanta_candidates = None
self.pair_wifi_period_slots = None
self.pair_wifi_tx_slots = None
self.pair_wifi_anchor_slot = None
```

Then implement:
- `_config_wifi_timing()` to build candidate periods `1.25ms * 2^n`, `n in {4,5}`
- `_config_wifi_pair_timing()` to randomly assign each WiFi pair one of those periods and a random anchor
- `wifi_min_tx_time_s=5e-3` -> `wifi_tx_slots >= 40` because `5 ms / 0.125 ms = 40`

Do not change BLE CE rules in this task.

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2, plus the BLE occupancy regression:

```bash
python - <<'PY'
from sim_script.tests.test_ble_anchor_and_ce_window_mask import test_ble_mask_marks_full_ce_window
from sim_script.tests.test_wifi_periodic_timing import (
    test_wifi_period_candidates_follow_pow2_rule,
    test_wifi_pairs_sample_periods_and_min_tx_duration,
)

test_wifi_period_candidates_follow_pow2_rule()
test_wifi_pairs_sample_periods_and_min_tx_duration()
test_ble_mask_marks_full_ce_window()
print('PASS')
PY
```

Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_ble_anchor_and_ce_window_mask.py sim_script/tests/test_wifi_periodic_timing.py
git commit -m "feat: add periodic wifi timing model on base slot grid"
```

### Task 2: Replace Start-Slot Eligibility With True Occupancy Expansion

**Files:**
- Modify: `sim_src/env/env.py:300-380`
- Modify: `sim_script/tests/test_ble_anchor_and_ce_window_mask.py`
- Create: `sim_script/tests/test_macrocycle_occupancy.py`
- Test: `sim_script/tests/test_macrocycle_occupancy.py`

**Step 1: Write the failing test**

Create `sim_script/tests/test_macrocycle_occupancy.py` to pin the new occupancy semantics for both radios.

```python
import numpy as np
from sim_src.env.env import env


def test_build_slot_occupancy_mask_marks_full_wifi_and_ble_blocks():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=3,
        radio_prob=(0.5, 0.5),
        wifi_period_exp_min=4,
        wifi_period_exp_max=4,
        ble_ci_exp_min=4,
        ble_ci_exp_max=4,
    )
    Z = 256
    mask = e.build_slot_occupancy_mask(Z)
    for k in range(e.n_pair):
        if e.pair_radio_type[k] == e.RADIO_WIFI:
            assert mask[k].any()
            assert int(mask[k].sum()) >= int(e.pair_wifi_tx_slots[k])
        else:
            if e.pair_ble_ce_feasible[k]:
                assert int(mask[k].sum()) >= int(e.pair_ble_ce_slots[k])
```

Add a second test to prove occupancy repeats periodically across the horizon.

```python
def test_occupancy_mask_repeats_at_period_boundaries():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=4,
        radio_prob=(1.0, 0.0),
        wifi_period_exp_min=4,
        wifi_period_exp_max=4,
    )
    Z = 128
    mask = e.build_slot_occupancy_mask(Z)
    wifi_idx = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    for k in wifi_idx:
        period = int(e.pair_wifi_period_slots[k])
        tx = int(e.pair_wifi_tx_slots[k])
        anchor = int(e.pair_wifi_anchor_slot[k] % period)
        assert mask[k, anchor:anchor + tx].all()
        if anchor + period + tx <= Z:
            assert mask[k, anchor + period:anchor + period + tx].all()
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_macrocycle_occupancy import (
    test_build_slot_occupancy_mask_marks_full_wifi_and_ble_blocks,
    test_occupancy_mask_repeats_at_period_boundaries,
)

test_build_slot_occupancy_mask_marks_full_wifi_and_ble_blocks()
test_occupancy_mask_repeats_at_period_boundaries()
print('PASS')
```

`PY`

Expected: FAIL because the code still uses `build_slot_compatibility_mask()` semantics instead of true occupancy expansion.

**Step 3: Write minimal implementation**

In `sim_src/env/env.py`, replace or supersede the current BLE-only helper with a generalized occupancy mask builder.

```python
def build_slot_occupancy_mask(self, Z):
    mask = np.zeros((self.n_pair, Z), dtype=bool)
    for k in range(self.n_pair):
        if self.pair_radio_type[k] == self.RADIO_WIFI:
            period = int(self.pair_wifi_period_slots[k])
            width = int(self.pair_wifi_tx_slots[k])
            anchor = int(self.pair_wifi_anchor_slot[k] % max(1, period))
            z = anchor
            while z < Z:
                mask[k, z:min(z + width, Z)] = True
                z += period
        else:
            if not self.pair_ble_ce_feasible[k]:
                continue
            period = int(self.pair_ble_ci_slots[k])
            width = int(self.pair_ble_ce_slots[k])
            anchor = int(self.pair_ble_anchor_slot[k] % max(1, period))
            z = anchor
            while z < Z:
                mask[k, z:min(z + width, Z)] = True
                z += period
    return mask
```

Keep `build_slot_compatibility_mask()` as a wrapper if existing callers still use that name, but its meaning must now be “full occupancy availability over the horizon,” not “single-slot eligibility.”

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_macrocycle_occupancy.py sim_script/tests/test_ble_anchor_and_ce_window_mask.py
git commit -m "feat: expand wifi and ble to true periodic occupancy masks"
```

### Task 3: Compute A Macrocycle And Enforce Non-Overlap Across It

**Files:**
- Modify: `sim_src/env/env.py`
- Modify: `sim_src/alg/sdp_solver.py`
- Modify: `sim_src/alg/binary_search_relaxation.py`
- Modify: `sim_script/tests/test_macrocycle_occupancy.py`
- Create: `sim_script/tests/test_macrocycle_scheduler.py`
- Test: `sim_script/tests/test_macrocycle_scheduler.py`

**Step 1: Write the failing test**

Create `sim_script/tests/test_macrocycle_scheduler.py` that proves the final schedule can be expanded over a macrocycle without pair-interval overlap.

```python
import numpy as np
from sim_src.env.env import env
from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.mmw import mmw


def test_macrocycle_schedule_has_no_overlapping_occupancy():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=5,
        radio_prob=(0.6, 0.4),
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
        ble_ci_exp_min=4,
        ble_ci_exp_max=5,
    )
    state = e.generate_S_Q_hmax()
    bs = binary_search_relaxation()
    bs.user_priority = e.pair_priority
    bs.slot_mask_builder = lambda Z, state, ee=e: ee.build_slot_occupancy_mask(Z)
    alg = mmw(nit=5, eta=0.05)
    alg.LOG_GAP = False
    bs.feasibility_check_alg = alg
    z_vec, Z_fin, rem = bs.run(state)
    assert rem == 0

    macro = e.compute_macrocycle_slots()
    occ = e.expand_schedule_occupancy(z_vec, Z_fin, macro)
    assert not occ.any(axis=0).astype(int).sum() < 0
    assert np.all(np.sum(occ, axis=0) <= 1)
```

Add a more targeted helper-level test if needed:

```python
def test_compute_macrocycle_slots_is_lcm_of_active_periods():
    e = env(...)
    macro = e.compute_macrocycle_slots()
    for p in e.get_active_period_slots():
        assert macro % int(p) == 0
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_macrocycle_scheduler import test_macrocycle_schedule_has_no_overlapping_occupancy

test_macrocycle_schedule_has_no_overlapping_occupancy()
print('PASS')
```

`PY`

Expected: FAIL because the solver still reasons in terms of one slot per pair and does not reserve full occupied intervals across the macrocycle.

**Step 3: Write minimal implementation**

Add env helpers:

```python
def get_active_period_slots(self):
    ...

def compute_macrocycle_slots(self):
    return int(np.lcm.reduce(periods))

def expand_schedule_occupancy(self, z_vec, Z, macrocycle_slots):
    ...
```

Then update the scheduling logic so feasibility and rounding no longer treat a pair as occupying only one slot. Minimal acceptable implementation:
- `slot_mask_builder` must expose valid start slots only if the full `[start, start + width)` interval is legal within the periodic occupancy model.
- `rounding_one_attempt()` must reserve the full occupied interval of an assigned pair across the macrocycle, not just the selected start slot.
- Conflict checks must reject an assignment if any occupied slot in the macrocycle overlaps with an already reserved occupied slot.

One possible skeleton inside `rounding_one_attempt()`:

```python
occupied_by_slot = [set() for _ in range(macrocycle_slots)]
...
interval_slots = occupancy_expander(k, z)
if any(occupied_by_slot[t] for t in interval_slots):
    continue
for t in interval_slots:
    occupied_by_slot[t].add(k)
```

Do not attempt a large theoretical rewrite in one step. Prefer an explicit occupancy expansion that is correct first, then optimize later.

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_src/alg/sdp_solver.py sim_src/alg/binary_search_relaxation.py sim_script/tests/test_macrocycle_occupancy.py sim_script/tests/test_macrocycle_scheduler.py
git commit -m "feat: enforce macrocycle non-overlap for periodic occupancies"
```

### Task 4: Rebuild The Pair Table And Schedule Table From True Occupancy

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

Add a logic test that expects pair rows and schedule rows to expose occupancy widths and macrocycle semantics.

```python
from sim_script.pd_mmw_template_ap_stats import build_schedule_rows


def test_build_schedule_rows_emits_true_slot_occupancy_rows():
    pair_rows = [
        {
            "pair_id": 2,
            "radio": "ble",
            "schedule_slot": 9,
            "occupancy_slots": [9, 10, 11],
        },
        {
            "pair_id": 14,
            "radio": "wifi",
            "schedule_slot": 12,
            "occupancy_slots": [12, 13, 14, 15],
        },
    ]
    rows = build_schedule_rows(pair_rows)
    assert rows[0]["schedule_slot"] == 9
    assert rows[0]["pair_ids"] == [2]
    assert rows[1]["schedule_slot"] == 10
    assert rows[1]["pair_ids"] == [2]
    assert rows[3]["schedule_slot"] == 12
    assert rows[3]["wifi_pair_ids"] == [14]
```

Extend the run test to expect new WiFi timing columns and a denser per-slot schedule table.

```python
assert "wifi_period_slots" in proc.stdout
assert "wifi_tx_slots" in proc.stdout
assert "macrocycle_slot" in proc.stdout or "schedule_slot" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_pd_mmw_template_ap_stats_logic import test_build_schedule_rows_emits_true_slot_occupancy_rows

test_build_schedule_rows_emits_true_slot_occupancy_rows()
print('PASS')
```

`PY`

Expected: FAIL because the current schedule table still prints one row per assigned start slot rather than one row per truly occupied slot.

**Step 3: Write minimal implementation**

Update `sim_script/pd_mmw_template_ap_stats.py` so pair rows include new WiFi timing fields and expanded occupancy fields.

Required pair-table additions:
- `wifi_anchor_slot`
- `wifi_period_slots`
- `wifi_period_ms`
- `wifi_tx_slots`
- `wifi_tx_ms`
- `occupancy_start_slot`
- `occupancy_slots` or an equivalent serialized interval field

Then rebuild the schedule table from actual occupied slots over the macrocycle.

Example approach:

```python
def build_schedule_rows(pair_rows):
    slot_map = {}
    for row in pair_rows:
        for slot in row["occupied_slots_in_macrocycle"]:
            slot_map.setdefault(slot, []).append(row)
    ...
```

The resulting CSV/terminal schedule table should be “true per-slot occupancy,” not “one line per pair start slot.”

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2 and the updated run smoke test.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: print macrocycle occupancy schedule with wifi timing fields"
```

### Task 5: Export Macrocycle CSVs And Validate Repeatability

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Modify: `README.md` (only if script output is documented there)

**Step 1: Write the failing test**

Extend the CSV export smoke test so it verifies the exported schedule CSV represents macrocycle occupancy, not just start-slot groups.

```python
def test_script_writes_macrocycle_schedule_csv():
    ...
    schedule_csv = pathlib.Path(tmpdir) / "wifi_ble_schedule.csv"
    text = schedule_csv.read_text(encoding="utf-8")
    assert "schedule_slot" in text
    assert "wifi_pair_ids" in text
    assert "ble_pair_ids" in text
    assert "pair_count" in text
```

Add one stronger assertion that the CSV has repeated occupied rows for multi-slot CE / WiFi TX durations.

```python
assert text.count("[2]") >= 2
```

Use a controlled small scenario if needed so the repetition is deterministic.

**Step 2: Run test to verify it fails**

Run the inline version of the new test.
Expected: FAIL because current CSV exports one row per chosen start slot.

**Step 3: Write minimal implementation**

Update CSV export in `sim_script/pd_mmw_template_ap_stats.py` so:
- `pair_parameters.csv` includes WiFi timing fields and occupancy interval summaries
- `wifi_ble_schedule.csv` exports one row per occupied macrocycle slot
- The script prints the macrocycle length in slots and ms

Suggested extra summary lines:

```python
print("macrocycle_slots =", macrocycle_slots)
print("macrocycle_ms =", macrocycle_slots * e.slot_time * 1e3)
```

**Step 4: Run test to verify it passes**

Run the updated inline smoke test and then:

```bash
python -u sim_script/pd_mmw_template_ap_stats.py --cell-size 2 --pair-density 0.05 --seed 3 --mmw-nit 5 --output-dir /tmp/pd_macrocycle
```

Expected:
- Exit code `0`
- `pair_parameters.csv` exists
- `wifi_ble_schedule.csv` exists
- Schedule CSV reflects true occupied macrocycle slots
- Repeating the same macrocycle is semantically safe because no slot within the macrocycle is multiply occupied

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py README.md
git commit -m "feat: export macrocycle occupancy schedule csv"
```

### Task 6: Final Review Pass On Semantics And Limits

**Files:**
- Modify: `README.md` (if needed)
- Test: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: Run an end-to-end smoke check**

Run: `python -u sim_script/pd_mmw_template_ap_stats.py --cell-size 2 --pair-density 0.05 --seed 3 --mmw-nit 5`

Expected output now includes:
- WiFi timing summary or WiFi timing columns
- BLE timing summary
- `macrocycle_slots` and `macrocycle_ms`
- Pair table with WiFi and BLE periodic timing parameters
- True macrocycle per-slot schedule table
- Office-level summary

**Step 2: Verify semantic clarity**

Read the final headers and ensure they distinguish:
- one pair's chosen start offset
- one pair's occupied interval width
- one macrocycle slot in the exported schedule
- WiFi period vs BLE CI
- WiFi TX duration vs BLE CE duration

If ambiguity remains, rename headers before merge.

Good target headers:

```text
pair_id,office_id,radio,channel,priority,start_slot,start_time_ms,wifi_anchor_slot,wifi_period_slots,wifi_period_ms,wifi_tx_slots,wifi_tx_ms,ble_anchor_slot,ble_ci_slots,ble_ci_ms,ble_ce_slots,ble_ce_ms,ble_ce_feasible,occupied_slots_in_macrocycle
schedule_slot,pair_ids,wifi_pair_ids,ble_pair_ids,pair_count,wifi_pair_count,ble_pair_count
```

**Step 3: Update docs only if needed**

If the script output or timing model is documented anywhere, add a short explanation:

```markdown
The AP stats script now models both WiFi and BLE as periodic occupancies on the base 0.125 ms slot grid.
BLE CE and WiFi TX durations reserve continuous occupied slot intervals.
The exported schedule is a true macrocycle occupancy table and is safe to repeat because no macrocycle slot is multiply occupied.
```

**Step 4: Commit**

```bash
git add README.md sim_script/pd_mmw_template_ap_stats.py sim_src/env/env.py sim_src/alg sim_script/tests
git commit -m "docs: clarify macrocycle occupancy scheduling semantics"
```
