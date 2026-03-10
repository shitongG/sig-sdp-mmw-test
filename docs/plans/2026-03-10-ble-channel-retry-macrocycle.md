# BLE Channel Retry Macrocycle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep each BLE pair's existing CI/CE fixed, retry channel selection only for unscheduled BLE pairs, and rerun macrocycle placement to improve schedule success without regenerating timing parameters.

**Architecture:** Leave initial pair generation unchanged so BLE CI/CE/anchor are sampled once. After the first macrocycle pass, identify unscheduled BLE pairs, resample only their `pair_channel`, rebuild conflict/interference state from the new channels, and rerun macrocycle assignment. Repeat for a small bounded number of retries, preserving the best result and never touching the already fixed CI/CE values.

**Tech Stack:** Python 3.10, NumPy, SciPy sparse matrices, existing `env`, macrocycle scheduler in `sim_script/pd_mmw_template_ap_stats.py`, CSV/plot export helpers.

---

### Task 1: Add a targeted BLE channel-resampling helper that preserves CI/CE

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_ble_channel_retry_helper.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_src.env.env import env


def test_resample_ble_channels_updates_only_target_ble_pairs_and_preserves_timing():
    e = env(cell_edge=7.0, cell_size=1, pair_density_per_m2=0.05, seed=7, radio_prob=(0.0, 1.0), slot_time=1.25e-3, ble_ce_max_s=7.5e-3, ble_phy_rate_bps=2e6)
    target = np.array([0], dtype=int)
    old_channel = e.pair_channel.copy()
    old_ci = e.pair_ble_ci_slots.copy()
    old_ce = e.pair_ble_ce_slots.copy()
    e.resample_ble_channels(target)
    assert old_channel[0] != e.pair_channel[0] or e.ble_channel_count == 1
    assert np.array_equal(old_ci, e.pair_ble_ci_slots)
    assert np.array_equal(old_ce, e.pair_ble_ce_slots)
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because `resample_ble_channels()` does not exist.

**Step 3: Write minimal implementation**

```python
def resample_ble_channels(self, pair_ids):
    pair_ids = np.asarray(pair_ids, dtype=int)
    ble_ids = pair_ids[self.pair_radio_type[pair_ids] == self.RADIO_BLE]
    if ble_ids.size == 0:
        return
    self.pair_channel[ble_ids] = self.rand_gen_loc.integers(0, self.ble_channel_count, size=ble_ids.size)
    self.device_radio_channel = self.pair_channel
    self.user_radio_channel = self.pair_channel
```

Do not touch `pair_ble_ci_slots`, `pair_ble_ce_slots`, `pair_ble_anchor_slot`, or feasibility flags.

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_ble_channel_retry_helper.py
git commit -m "feat: resample BLE channels without changing timing"
```

### Task 2: Add a bounded BLE channel-retry loop around macrocycle placement

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_ble_channel_retry_scheduler.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_script.pd_mmw_template_ap_stats import retry_ble_channels_and_assign_macrocycle


class _RetryEnv:
    n_pair = 2
    RADIO_BLE = 1
    pair_radio_type = np.array([1, 1], dtype=int)
    pair_priority = np.array([2.0, 1.0], dtype=float)
    pair_channel = np.array([0, 0], dtype=int)
    pair_ble_ce_feasible = np.array([True, True], dtype=bool)
    retry_count = 0

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
        if self.retry_count == 0:
            return np.array([[False, True], [True, False]], dtype=bool)
        return np.array([[False, False], [False, False]], dtype=bool)

    def get_macrocycle_conflict_state(self):
        import scipy.sparse
        s_gain = scipy.sparse.csr_matrix(np.array([[0.4, 0.2], [0.2, 0.4]], dtype=float))
        q_conflict = scipy.sparse.csr_matrix((2, 2))
        h_max = np.array([1.0, 1.0], dtype=float)
        return s_gain, q_conflict, h_max

    def resample_ble_channels(self, pair_ids):
        self.retry_count += 1
        self.pair_channel[pair_ids] = 1


def test_retry_ble_channels_can_reduce_unscheduled_pairs():
    env = _RetryEnv()
    preferred = np.array([0, 0], dtype=int)
    starts, macro, occ, unscheduled, retries_used = retry_ble_channels_and_assign_macrocycle(env, preferred, max_ble_channel_retries=1)
    assert unscheduled == []
    assert retries_used == 1
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because no retry wrapper exists.

**Step 3: Write minimal implementation**

```python
def retry_ble_channels_and_assign_macrocycle(e, preferred_slots, max_ble_channel_retries=3):
    best = assign_macrocycle_start_slots(e, preferred_slots, allow_partial=True)
    best_unscheduled = best[3]
    retries_used = 0
    for _ in range(max_ble_channel_retries):
        ble_unscheduled = [pid for pid in best_unscheduled if e.pair_radio_type[pid] == e.RADIO_BLE]
        if not ble_unscheduled:
            break
        e.resample_ble_channels(np.array(ble_unscheduled, dtype=int))
        candidate = assign_macrocycle_start_slots(e, preferred_slots, allow_partial=True)
        if len(candidate[3]) < len(best_unscheduled):
            best = candidate
            best_unscheduled = candidate[3]
        retries_used += 1
        if not best_unscheduled:
            break
    return (*best, retries_used)
```

Keep this strictly post-processing: it must not regenerate CI/CE or rerun `MMW`.

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_ble_channel_retry_scheduler.py
git commit -m "feat: retry BLE channels for unscheduled macrocycle pairs"
```

### Task 3: Keep already scheduled pairs fixed while retrying only unscheduled BLE pairs

**Files:**
- Modify: `sim_script/tests/test_ble_channel_retry_scheduler.py`
- Modify: `sim_script/pd_mmw_template_ap_stats.py` if needed

**Step 1: Write the failing regression test**

```python
def test_retry_does_not_change_ci_ce_or_scheduled_non_ble_pairs():
    ...
    assert np.array_equal(old_ci, env.pair_ble_ci_slots)
    assert np.array_equal(old_ce, env.pair_ble_ce_slots)
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL if retry code accidentally resamples timing or touches non-target pairs.

**Step 3: Write minimal implementation**

- Retry only `unscheduled_pair_ids` filtered to BLE
- Do not resample channels for already scheduled pairs
- Do not call `_config_ble_pair_timing()` again

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/tests/test_ble_channel_retry_scheduler.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "test: keep BLE timing fixed during channel retries"
```

### Task 4: Integrate retry loop into the main script and expose CLI control

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_ble_channel_retry.py`

**Step 1: Write the failing test**

```python
import pathlib
import subprocess
import sys


def test_script_accepts_ble_channel_retry_flag():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--cell-size", "1", "--pair-density", "0.05", "--seed", "7", "--mmw-nit", "5", "--ble-channel-retries", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "ble_channel_retries_used =" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because CLI flag/output does not exist.

**Step 3: Write minimal implementation**

- Add `--ble-channel-retries` CLI arg with a small default such as `0` or `3`
- Replace the direct `assign_macrocycle_start_slots(...)` call with `retry_ble_channels_and_assign_macrocycle(...)`
- Print `ble_channel_retries_used = N`

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_ble_channel_retry.py
git commit -m "feat: add BLE channel retry control to script"
```

### Task 5: Preserve output/CSV/plot behavior after channel retries

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Modify: `sim_script/tests/test_schedule_plot_rows.py`
- Modify: `sim_script/pd_mmw_template_ap_stats.py` if needed

**Step 1: Write the failing regression tests**

```python
def test_retry_path_still_exports_csv_and_plot_outputs():
    ...
    assert (pathlib.Path(tmpdir) / "pair_parameters.csv").exists()
    assert (pathlib.Path(tmpdir) / "schedule_plot_rows.csv").exists()
    assert (pathlib.Path(tmpdir) / "wifi_ble_schedule.png").exists()
```

**Step 2: Run regression tests to verify failure**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL if retry integration broke output generation.

**Step 3: Write minimal implementation**

- Ensure retried channels are reflected in `pair_rows`
- Ensure plot rows use the final `pair_channel`
- Ensure unscheduled pair CSV still matches final retry result

**Step 4: Run tests to verify pass**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/tests/test_schedule_plot_rows.py
git commit -m "test: preserve exports after BLE channel retries"
```

### Task 6: Focused verification in `sig-sdp`

**Files:**
- Modify: none unless fixes are needed
- Test: all touched tests and one script smoke run

**Step 1: Run focused tests**

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python - <<'PY'
from sim_script.tests.test_ble_channel_retry_helper import test_resample_ble_channels_updates_only_target_ble_pairs_and_preserves_timing
from sim_script.tests.test_ble_channel_retry_scheduler import test_retry_ble_channels_can_reduce_unscheduled_pairs
from sim_script.tests.test_pd_mmw_ble_channel_retry import test_script_accepts_ble_channel_retry_flag

test_resample_ble_channels_updates_only_target_ble_pairs_and_preserves_timing()
test_retry_ble_channels_can_reduce_unscheduled_pairs()
test_script_accepts_ble_channel_retry_flag()
print('ble_channel_retry: PASS')
PY
```

Expected: `ble_channel_retry: PASS`

**Step 2: Run one real script smoke**

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --mmw-nit 5 --seed 7 --ble-channel-retries 2
```

Expected:
- exit code `0`
- `ble_channel_retries_used = ...`
- final CSV/PNG outputs still generated
- BLE CI/CE values unchanged across retries

**Step 3: Commit verification-only fixes if needed**

```bash
git add <touched-files>
git commit -m "fix: stabilize BLE channel retry verification"
```
