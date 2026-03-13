# WiFi-First BLE Hopping Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change the scheduling pipeline to solve WiFi pairs first, then schedule BLE pairs only on BLE channels left available after WiFi spectrum occupancy, with BLE start-slot capacity bounded by the remaining non-overlapping BLE channels and BLE hopping collision probability reported as `(1 - 1/C)^(N - 1)`.

**Architecture:** Split the current joint WiFi/BLE flow into two sequential subproblems. First solve and place WiFi pairs, then derive per-start-slot BLE channel availability from the WiFi spectrum occupancy and solve BLE pairs under a hard `N <= C` start-slot constraint, while recording `C`, `N`, and the theoretical no-collision probability for later BLE connection events.

**Tech Stack:** Python, NumPy, SciPy, pytest

---

### Task 1: Add env helpers for WiFi-blocked BLE channel availability

**Files:**
- Modify: `sim_src/env/env.py`
- Create: `sim_script/tests/test_wifi_first_ble_channel_availability.py`
- Test: `sim_script/tests/test_wifi_first_ble_channel_availability.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_src.env.env import env


def test_ble_available_channels_exclude_wifi_overlapping_spectrum():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=11,
        radio_prob=(1.0, 0.0),
        wifi_channel_bandwidth_hz=20e6,
        ble_channel_bandwidth_hz=2e6,
    )

    e.pair_radio_type = np.array([e.RADIO_WIFI], dtype=int)
    e.pair_channel = np.array([0], dtype=int)
    e.pair_wifi_period_slots = np.array([8], dtype=int)
    e.pair_wifi_tx_slots = np.array([2], dtype=int)
    e.pair_wifi_anchor_slot = np.array([0], dtype=int)

    channels = e.get_available_ble_channels_for_start_slot(
        wifi_pair_ids=np.array([0], dtype=int),
        wifi_start_slots=np.array([0], dtype=int),
        start_slot=0,
    )

    expected = np.array([idx for idx in range(e.ble_channel_count) if idx >= 8], dtype=int)
    assert np.array_equal(channels, expected)
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_wifi_first_ble_channel_availability.py::test_ble_available_channels_exclude_wifi_overlapping_spectrum -v`
Expected: FAIL with missing method `get_available_ble_channels_for_start_slot`

**Step 3: Write minimal implementation**

```python
def get_available_ble_channels_for_start_slot(self, wifi_pair_ids, wifi_start_slots, start_slot):
    blocked = np.zeros(self.ble_channel_count, dtype=bool)
    for pair_id, start in zip(wifi_pair_ids, wifi_start_slots):
        if not self.expand_pair_occupancy(int(pair_id), int(start), int(start_slot) + 1)[int(start_slot)]:
            continue
        w0, w1 = self._get_wifi_channel_range_hz(self.pair_channel[int(pair_id)])
        for ble_idx in range(self.ble_channel_count):
            b0, b1 = self._get_ble_channel_range_hz(ble_idx)
            if self._is_range_overlap(w0, w1, b0, b1):
                blocked[ble_idx] = True
    return np.where(~blocked)[0]
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_wifi_first_ble_channel_availability.py::test_ble_available_channels_exclude_wifi_overlapping_spectrum -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_wifi_first_ble_channel_availability.py
git commit -m "feat: expose BLE channel availability after WiFi occupancy"
```

### Task 2: Add BLE capacity and collision-probability helpers

**Files:**
- Modify: `sim_src/env/env.py`
- Modify: `sim_script/tests/test_wifi_first_ble_channel_availability.py`
- Test: `sim_script/tests/test_wifi_first_ble_channel_availability.py`

**Step 1: Write the failing test**

```python
def test_ble_start_slot_capacity_matches_available_channel_count():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=12,
        radio_prob=(1.0, 0.0),
    )
    e.pair_radio_type = np.array([e.RADIO_WIFI], dtype=int)
    e.pair_channel = np.array([0], dtype=int)
    e.pair_wifi_period_slots = np.array([8], dtype=int)
    e.pair_wifi_tx_slots = np.array([2], dtype=int)
    e.pair_wifi_anchor_slot = np.array([0], dtype=int)

    c = e.get_ble_start_slot_capacity(
        wifi_pair_ids=np.array([0], dtype=int),
        wifi_start_slots=np.array([0], dtype=int),
        start_slot=0,
    )

    assert c == 29


def test_ble_no_collision_probability_matches_closed_form():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=13)
    assert np.isclose(e.compute_ble_no_collision_probability(c=10, n=3), (1.0 - 0.1) ** 2)
    assert e.compute_ble_no_collision_probability(c=10, n=1) == 1.0
    assert e.compute_ble_no_collision_probability(c=0, n=2) == 0.0
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_wifi_first_ble_channel_availability.py -k \"capacity or probability\" -v`
Expected: FAIL with missing helper methods

**Step 3: Write minimal implementation**

```python
def get_ble_start_slot_capacity(self, wifi_pair_ids, wifi_start_slots, start_slot):
    return int(self.get_available_ble_channels_for_start_slot(wifi_pair_ids, wifi_start_slots, start_slot).size)


def compute_ble_no_collision_probability(self, c, n):
    c = int(c)
    n = int(n)
    if n <= 1:
        return 1.0
    if c <= 0:
        return 0.0
    return float((1.0 - 1.0 / c) ** (n - 1))
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_wifi_first_ble_channel_availability.py -k \"capacity or probability\" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_wifi_first_ble_channel_availability.py
git commit -m "feat: add BLE capacity and no-collision probability helpers"
```

### Task 3: Add a WiFi-first split for feasibility solving

**Files:**
- Modify: `sim_src/alg/binary_search_relaxation.py`
- Create: `sim_script/tests/test_wifi_first_binary_search_split.py`
- Test: `sim_script/tests/test_wifi_first_binary_search_split.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_src.alg.binary_search_relaxation import binary_search_relaxation


def test_wifi_first_solver_keeps_wifi_pairs_in_front_stage():
    bs = binary_search_relaxation()
    pair_radio_type = np.array([0, 0, 1, 1], dtype=int)

    wifi_idx, ble_idx = bs.split_pair_indices_by_radio_type(pair_radio_type, wifi_id=0, ble_id=1)

    assert np.array_equal(wifi_idx, np.array([0, 1], dtype=int))
    assert np.array_equal(ble_idx, np.array([2, 3], dtype=int))
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_wifi_first_binary_search_split.py::test_wifi_first_solver_keeps_wifi_pairs_in_front_stage -v`
Expected: FAIL with missing helper `split_pair_indices_by_radio_type`

**Step 3: Write minimal implementation**

```python
@staticmethod
def split_pair_indices_by_radio_type(pair_radio_type, wifi_id=0, ble_id=1):
    arr = np.asarray(pair_radio_type, dtype=int)
    return np.where(arr == wifi_id)[0], np.where(arr == ble_id)[0]
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_wifi_first_binary_search_split.py::test_wifi_first_solver_keeps_wifi_pairs_in_front_stage -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/alg/binary_search_relaxation.py sim_script/tests/test_wifi_first_binary_search_split.py
git commit -m "feat: add radio split helper for wifi-first solver"
```

### Task 4: Add WiFi-first start-slot assignment constraints for BLE

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Create: `sim_script/tests/test_wifi_first_macrocycle_assignment.py`
- Test: `sim_script/tests/test_wifi_first_macrocycle_assignment.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots


class _WifiFirstEnv:
    n_pair = 4
    RADIO_WIFI = 0
    RADIO_BLE = 1
    pair_radio_type = np.array([0, 0, 1, 1], dtype=int)
    pair_priority = np.array([10.0, 9.0, 2.0, 1.0], dtype=float)

    def compute_macrocycle_slots(self):
        return 8

    def get_pair_period_slots(self):
        return np.array([8, 8, 8, 8], dtype=int)

    def get_pair_width_slots(self):
        return np.array([1, 1, 1, 1], dtype=int)

    def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
        occ = np.zeros(macrocycle_slots, dtype=bool)
        occ[start_slot:start_slot + 1] = True
        return occ

    def build_pair_conflict_matrix(self):
        return np.zeros((4, 4), dtype=bool)

    def get_macrocycle_conflict_state(self):
        import scipy.sparse
        return scipy.sparse.csr_matrix((4, 4)), scipy.sparse.csr_matrix((4, 4)), np.full(4, 1e9)

    def get_ble_start_slot_capacity(self, wifi_pair_ids, wifi_start_slots, start_slot):
        return 1 if start_slot == 0 else 2

    def compute_ble_no_collision_probability(self, c, n):
        if n <= 1:
            return 1.0
        return float((1.0 - 1.0 / c) ** (n - 1))


def test_wifi_first_assignment_limits_ble_pairs_by_remaining_channel_capacity():
    env = _WifiFirstEnv()
    starts, macro, occ, unscheduled, ble_stats = assign_macrocycle_start_slots(
        env,
        preferred_slots=np.zeros(env.n_pair, dtype=int),
        allow_partial=True,
        wifi_first=True,
        return_ble_stats=True,
    )

    scheduled_ble = [idx for idx in [2, 3] if idx not in unscheduled and starts[idx] == 0]
    assert len(scheduled_ble) == 1
    assert ble_stats[0]["effective_ble_channels"] == 1
    assert ble_stats[0]["scheduled_ble_pairs"] == 1
    assert ble_stats[0]["no_collision_probability"] == 1.0
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_wifi_first_macrocycle_assignment.py::test_wifi_first_assignment_limits_ble_pairs_by_remaining_channel_capacity -v`
Expected: FAIL because `assign_macrocycle_start_slots` does not yet support `wifi_first` or `ble_stats`

**Step 3: Write minimal implementation**

```python
# Extend assign_macrocycle_start_slots with:
# - wifi_first=False
# - return_ble_stats=False
# - two-pass ordering: schedule WiFi first, then BLE
# - BLE start-slot acceptance rule: scheduled_ble_count_at_slot < effective_ble_channels
# - optional ble_stats output keyed by start slot
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_wifi_first_macrocycle_assignment.py::test_wifi_first_assignment_limits_ble_pairs_by_remaining_channel_capacity -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_wifi_first_macrocycle_assignment.py
git commit -m "feat: add wifi-first BLE start-slot capacity enforcement"
```

### Task 5: Integrate WiFi-first pipeline into the script entrypoint

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_ble_channel_retry.py`
- Modify: `sim_script/tests/test_pd_mmw_ble_channel_modes.py`
- Test: `sim_script/tests/test_pd_mmw_ble_channel_retry.py`
- Test: `sim_script/tests/test_pd_mmw_ble_channel_modes.py`

**Step 1: Write the failing test**

```python
def test_script_reports_wifi_first_ble_channel_stats(tmp_path):
    # invoke pd_mmw_template_ap_stats.py with a small deterministic configuration
    # and assert stdout/csv contains:
    # effective_ble_channels, scheduled_ble_pairs, no_collision_probability
    ...
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_pd_mmw_ble_channel_retry.py tests/test_pd_mmw_ble_channel_modes.py -v`
Expected: FAIL because new WiFi-first fields and flags are not exposed yet

**Step 3: Write minimal implementation**

```python
# Add script/config flags such as:
# --wifi-first-ble-scheduling
# --report-ble-channel-capacity
#
# Include BLE stats in exported rows/summary when wifi_first mode is enabled.
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_pd_mmw_ble_channel_retry.py tests/test_pd_mmw_ble_channel_modes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_ble_channel_retry.py sim_script/tests/test_pd_mmw_ble_channel_modes.py
git commit -m "feat: expose wifi-first BLE hopping scheduling outputs"
```

### Task 6: Run focused verification

**Files:**
- Test: `sim_script/tests/test_wifi_first_ble_channel_availability.py`
- Test: `sim_script/tests/test_wifi_first_binary_search_split.py`
- Test: `sim_script/tests/test_wifi_first_macrocycle_assignment.py`
- Test: `sim_script/tests/test_macrocycle_real_conflict_integration.py`
- Test: `sim_script/tests/test_ble_per_ce_conflict_checks.py`

**Step 1: Run focused WiFi-first suite**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_wifi_first_ble_channel_availability.py tests/test_wifi_first_binary_search_split.py tests/test_wifi_first_macrocycle_assignment.py tests/test_macrocycle_real_conflict_integration.py tests/test_ble_per_ce_conflict_checks.py -v`
Expected: PASS

**Step 2: Run representative script-facing regression**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_pd_mmw_ble_channel_retry.py tests/test_pd_mmw_ble_channel_modes.py tests/test_macrocycle_scheduler.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add sim_src/env/env.py sim_src/alg/binary_search_relaxation.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_wifi_first_ble_channel_availability.py sim_script/tests/test_wifi_first_binary_search_split.py sim_script/tests/test_wifi_first_macrocycle_assignment.py
git commit -m "feat: implement wifi-first BLE hopping scheduling"
```
