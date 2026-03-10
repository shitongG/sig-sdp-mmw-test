# Interference-Aware Macrocycle And Schedule Plot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `S_gain/h_max` interference-budget checks to macrocycle concurrency decisions and generate a readable schedule plot with time slots on the x-axis and channel/frequency bands on the y-axis.

**Architecture:** Extend the current conflict-aware macrocycle scheduler so each occupied slot tracks both assigned pair ids and accumulated interference usage, mirroring the `rounding()` budget check instead of relying on `Q_conflict` alone. Separately, build a plot-friendly schedule data model from the final pair rows, including channel frequency ranges, and render it to an image file so the result is readable without parsing long CSV rows.

**Tech Stack:** Python 3.10, NumPy, SciPy sparse matrices, existing `env.generate_S_Q_hmax()`, existing macrocycle helpers in `sim_script/pd_mmw_template_ap_stats.py`, Matplotlib for plotting.

---

### Task 1: Expose macrocycle interference-budget inputs from `env`

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_macrocycle_interference_state.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_src.env.env import env


def test_macrocycle_interference_state_matches_pair_dimension():
    e = env(cell_edge=7.0, cell_size=1, pair_density_per_m2=0.05, seed=7, radio_prob=(0.0, 1.0), slot_time=1.25e-3, ble_ce_max_s=7.5e-3, ble_phy_rate_bps=2e6)
    s_gain, q_conflict, h_max = e.get_macrocycle_conflict_state()
    assert s_gain.shape == (e.n_pair, e.n_pair)
    assert q_conflict.shape == (e.n_pair, e.n_pair)
    assert h_max.shape == (e.n_pair,)
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because `get_macrocycle_conflict_state()` does not exist.

**Step 3: Write minimal implementation**

```python
def get_macrocycle_conflict_state(self):
    s_gain, q_conflict, h_max = self.generate_S_Q_hmax(real=False)
    return s_gain.tocsr(), q_conflict.tocsr(), np.asarray(h_max, dtype=float).ravel()
```

Keep this as a thin wrapper so the scheduler can reuse the exact same state shape as `rounding()`.

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_macrocycle_interference_state.py
git commit -m "feat: expose macrocycle interference state"
```

### Task 2: Add interference-budget checks to macrocycle concurrency

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:80-132`
- Test: `sim_script/tests/test_macrocycle_interference_budget.py`

**Step 1: Write the failing test**

```python
import numpy as np
import scipy.sparse

from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots


class _BudgetEnv:
    n_pair = 2
    pair_priority = np.array([2.0, 1.0], dtype=float)
    RADIO_BLE = 1
    pair_radio_type = np.array([0, 0], dtype=int)
    pair_ble_ce_feasible = np.array([True, True], dtype=bool)

    def compute_macrocycle_slots(self):
        return 4

    def get_pair_period_slots(self):
        return np.array([4, 4], dtype=int)

    def get_pair_width_slots(self):
        return np.array([2, 2], dtype=int)

    def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
        occ = np.zeros(macrocycle_slots, dtype=bool)
        occ[0:2] = True
        return occ

    def build_pair_conflict_matrix(self):
        return np.array([[False, False], [False, False]], dtype=bool)

    def get_macrocycle_conflict_state(self):
        s_gain = scipy.sparse.csr_matrix(np.array([[5.0, 4.0], [4.0, 5.0]]))
        q_conflict = scipy.sparse.csr_matrix((2, 2))
        h_max = np.array([1.0, 1.0])
        return s_gain, q_conflict, h_max


def test_macrocycle_scheduler_blocks_overlap_when_h_budget_is_exceeded():
    env = _BudgetEnv()
    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(env, np.array([0, 0]), allow_partial=True)
    assert starts[0] >= 0
    assert starts[1] == -1
    assert unscheduled == [1]
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because current scheduler only checks `Q_conflict` and will allow the overlap.

**Step 3: Write minimal implementation**

```python
s_gain, q_conflict, h_max = e.get_macrocycle_conflict_state()
slot_gain_sum = [np.zeros(e.n_pair, dtype=float) for _ in range(macrocycle_slots)]
slot_pairs = [[] for _ in range(macrocycle_slots)]
...
for slot in occ_slots:
    neighbor_index = np.append(np.array(slot_pairs[slot], dtype=int), pair_id)
    tmp_h = np.asarray(s_gain[pair_id].toarray()).ravel()
    vio = (slot_gain_sum[slot][neighbor_index] + tmp_h[neighbor_index]) > h_max[neighbor_index]
    if np.any(vio):
        violates = True
        break
...
for slot in occ_slots:
    slot_gain_sum[slot] += np.asarray(s_gain[pair_id].toarray()).ravel()
    slot_pairs[slot].append(pair_id)
```

This should mirror the logic already used in `sim_src/alg/sdp_solver.py` for same-slot interference-budget checking.

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_macrocycle_interference_budget.py
git commit -m "feat: enforce S-gain interference budgets in macrocycle scheduler"
```

### Task 3: Keep non-conflicting and budget-safe overlap enabled

**Files:**
- Modify: `sim_script/tests/test_macrocycle_concurrent_assignment.py`
- Modify: `sim_script/pd_mmw_template_ap_stats.py` if needed

**Step 1: Write the failing regression test**

```python
def test_macrocycle_scheduler_allows_overlap_when_conflict_and_h_budget_are_both_safe():
    env = _BudgetSafeEnv()
    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(env, np.array([0, 0]), allow_partial=False)
    assert unscheduled == []
    assert np.any(np.logical_and(occ[0], occ[1]))
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL if the new budget logic becomes too conservative.

**Step 3: Write minimal implementation**

Tighten the acceptance rule so a candidate is rejected only when:
- `Q_conflict` says the pair conflicts with an already assigned pair in that occupied slot, or
- the accumulated `S_gain/h_max` budget would be exceeded in that occupied slot

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/tests/test_macrocycle_concurrent_assignment.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "test: preserve safe overlap under interference budgets"
```

### Task 4: Build a plot-oriented schedule data model

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_schedule_plot_rows.py`

**Step 1: Write the failing test**

```python
from sim_script.pd_mmw_template_ap_stats import build_schedule_plot_rows


def test_schedule_plot_rows_include_slot_channel_and_frequency_band():
    pair_rows = [
        {
            "pair_id": 3,
            "radio": "ble",
            "channel": 19,
            "occupied_slots_in_macrocycle": [5, 6],
        }
    ]
    rows = build_schedule_plot_rows(pair_rows, pair_channel_ranges={3: (2440.0, 2442.0)})
    assert rows[0]["slot"] == 5
    assert rows[0]["pair_id"] == 3
    assert rows[0]["freq_low_mhz"] == 2440.0
    assert rows[0]["freq_high_mhz"] == 2442.0
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because no plot-row builder exists.

**Step 3: Write minimal implementation**

```python
def build_schedule_plot_rows(pair_rows, pair_channel_ranges):
    rows = []
    for row in pair_rows:
        pair_id = int(row["pair_id"])
        low_mhz, high_mhz = pair_channel_ranges[pair_id]
        for slot in row["occupied_slots_in_macrocycle"]:
            rows.append({...})
    return rows
```

Also add a small helper that converts `pair_id -> (freq_low_mhz, freq_high_mhz)` by reusing `env._get_pair_link_range_hz()`.

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_schedule_plot_rows.py
git commit -m "feat: build plot rows for schedule visualization"
```

### Task 5: Render the schedule figure with slot on x-axis and channel band on y-axis

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_schedule_plot_render.py`

**Step 1: Write the failing test**

```python
import pathlib
import tempfile

from sim_script.pd_mmw_template_ap_stats import render_schedule_plot


def test_render_schedule_plot_writes_png_file():
    rows = [
        {
            "pair_id": 3,
            "radio": "ble",
            "slot": 5,
            "freq_low_mhz": 2440.0,
            "freq_high_mhz": 2442.0,
            "channel": 19,
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = pathlib.Path(tmpdir) / "schedule_plot.png"
        render_schedule_plot(rows, out)
        assert out.exists()
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because rendering helper does not exist.

**Step 3: Write minimal implementation**

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def render_schedule_plot(plot_rows, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    for row in plot_rows:
        ax.add_patch(Rectangle((row["slot"], row["freq_low_mhz"]), 1.0, row["freq_high_mhz"] - row["freq_low_mhz"], ...))
    ax.set_xlabel("Slot")
    ax.set_ylabel("Frequency (MHz)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
```

Use different colors for WiFi/BLE and a small alpha so overlapping safe concurrency is visible.

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_schedule_plot_render.py
git commit -m "feat: render slot-by-frequency schedule plot"
```

### Task 6: Integrate plot export into the main script and CSV outputs

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Modify: `sim_script/tests/test_pd_mmw_partial_schedule_csv.py` if needed

**Step 1: Write the failing test**

```python
def test_script_exports_schedule_plot_file():
    ...
    assert (pathlib.Path(tmpdir) / "wifi_ble_schedule.png").exists()
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because the script does not yet create the plot.

**Step 3: Write minimal implementation**

- After computing `scheduled_pair_rows`, build `pair_channel_ranges`
- Convert to plot rows
- Save a PNG such as `wifi_ble_schedule.png` inside `args.output_dir`
- Print the output path alongside the CSV paths

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
 git commit -m "feat: export readable schedule plot"
```

### Task 7: Focused verification in `sig-sdp`

**Files:**
- Modify: none unless fixes are needed
- Test: all touched tests plus one script smoke run

**Step 1: Run focused tests**

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python - <<'PY'
from sim_script.tests.test_macrocycle_interference_state import test_macrocycle_interference_state_matches_pair_dimension
from sim_script.tests.test_macrocycle_interference_budget import test_macrocycle_scheduler_blocks_overlap_when_h_budget_is_exceeded
from sim_script.tests.test_macrocycle_concurrent_assignment import (
    test_macrocycle_scheduler_allows_overlap_for_non_conflicting_pairs,
    test_macrocycle_scheduler_still_blocks_overlap_for_conflicting_pairs,
)
from sim_script.tests.test_schedule_plot_rows import test_schedule_plot_rows_include_slot_channel_and_frequency_band
from sim_script.tests.test_schedule_plot_render import test_render_schedule_plot_writes_png_file

test_macrocycle_interference_state_matches_pair_dimension()
test_macrocycle_scheduler_blocks_overlap_when_h_budget_is_exceeded()
test_macrocycle_scheduler_allows_overlap_for_non_conflicting_pairs()
test_macrocycle_scheduler_still_blocks_overlap_for_conflicting_pairs()
test_schedule_plot_rows_include_slot_channel_and_frequency_band()
test_render_schedule_plot_writes_png_file()
print('interference_plot: PASS')
PY
```

Expected: `interference_plot: PASS`

**Step 2: Run one script smoke**

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --mmw-nit 5 --seed 7
```

Expected:
- exit code `0`
- `wifi_ble_schedule.csv` still generated
- `wifi_ble_schedule.png` generated
- plot visually shows slot on x-axis, frequency/channel band on y-axis

**Step 3: Commit verification-only fixes if needed**

```bash
git add <touched-files>
git commit -m "fix: stabilize interference-aware plot verification"
```
