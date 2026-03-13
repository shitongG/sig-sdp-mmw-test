# Binary Search WiFi-First Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the WiFi-first two-stage feasibility solving strategy into `binary_search_relaxation` as a built-in `run(...)` strategy while keeping the macrocycle BLE capacity constraints in the scheduling layer.

**Architecture:** Extend `binary_search_relaxation` with a strategy switch, radio-type metadata, and private helpers for state slicing, per-stage execution, and result merging. Keep the existing `joint` path unchanged, make `wifi_first` produce the same `(z_vec, Z, rem)` public return shape, and expose stage metadata via instance fields so script-level callers no longer need to orchestrate feasibility splitting themselves.

**Tech Stack:** Python, NumPy, SciPy, pytest

---

### Task 1: Add failing tests for the solver strategy API

**Files:**
- Modify: `sim_script/tests/test_wifi_first_binary_search_split.py`
- Modify: `sim_src/alg/binary_search_relaxation.py`
- Test: `sim_script/tests/test_wifi_first_binary_search_split.py`

**Step 1: Write the failing test**

```python
def test_binary_search_strategy_defaults_to_joint():
    bs = binary_search_relaxation()
    assert bs.strategy == "joint"
    assert bs.pair_radio_type is None
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest sim_script/tests/test_wifi_first_binary_search_split.py::test_binary_search_strategy_defaults_to_joint -v`
Expected: FAIL because the new fields do not exist yet

**Step 3: Write minimal implementation**

```python
class binary_search_relaxation(...):
    def __init__(self):
        ...
        self.strategy = "joint"
        self.pair_radio_type = None
        self.wifi_radio_id = 0
        self.ble_radio_id = 1
        self.last_stage_results = None
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest sim_script/tests/test_wifi_first_binary_search_split.py::test_binary_search_strategy_defaults_to_joint -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_wifi_first_binary_search_split.py sim_src/alg/binary_search_relaxation.py
git commit -m "feat: add solver strategy metadata to binary search"
```

### Task 2: Add failing tests for internal WiFi/BLE split helpers

**Files:**
- Modify: `sim_script/tests/test_wifi_first_binary_search_split.py`
- Modify: `sim_src/alg/binary_search_relaxation.py`
- Test: `sim_script/tests/test_wifi_first_binary_search_split.py`

**Step 1: Write the failing test**

```python
def test_binary_search_slices_state_for_pair_ids():
    state = (
        scipy.sparse.csr_matrix(np.array([[1.0, 0.2], [0.3, 2.0]])),
        scipy.sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]])),
        np.array([5.0, 6.0]),
    )
    bs = binary_search_relaxation()

    sliced = bs._slice_state_for_pair_ids(state, np.array([1], dtype=int))

    assert sliced[0].shape == (1, 1)
    assert sliced[1].shape == (1, 1)
    assert np.array_equal(sliced[2], np.array([6.0]))
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest sim_script/tests/test_wifi_first_binary_search_split.py::test_binary_search_slices_state_for_pair_ids -v`
Expected: FAIL because the private helper does not exist yet

**Step 3: Write minimal implementation**

```python
def _slice_state_for_pair_ids(self, state, pair_ids):
    ...
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest sim_script/tests/test_wifi_first_binary_search_split.py::test_binary_search_slices_state_for_pair_ids -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_wifi_first_binary_search_split.py sim_src/alg/binary_search_relaxation.py
git commit -m "feat: add wifi-first state slicing helpers"
```

### Task 3: Add failing tests for WiFi-first `run(...)` behavior

**Files:**
- Modify: `sim_script/tests/test_wifi_first_binary_search_split.py`
- Modify: `sim_src/alg/binary_search_relaxation.py`
- Test: `sim_script/tests/test_wifi_first_binary_search_split.py`

**Step 1: Write the failing test**

```python
def test_run_uses_wifi_first_strategy_when_configured():
    state = (
        scipy.sparse.eye(4, format="csr"),
        scipy.sparse.csr_matrix((4, 4), dtype=float),
        np.ones(4, dtype=float),
    )
    bs = binary_search_relaxation()
    bs.strategy = "wifi_first"
    bs.pair_radio_type = np.array([0, 0, 1, 1], dtype=int)
    bs.feasibility_check_alg = _StageAlg(...)

    z_vec, z_fin, remainder = bs.run(state)

    assert z_vec.shape == (4,)
    assert z_fin >= 0
    assert remainder >= 0
    assert set(bs.last_stage_results.keys()) == {"wifi", "ble"}
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest sim_script/tests/test_wifi_first_binary_search_split.py::test_run_uses_wifi_first_strategy_when_configured -v`
Expected: FAIL because `run(...)` only supports the joint path

**Step 3: Write minimal implementation**

```python
def run(self, state):
    if self.strategy == "wifi_first":
        return self._run_wifi_first(state)
    ...
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest sim_script/tests/test_wifi_first_binary_search_split.py::test_run_uses_wifi_first_strategy_when_configured -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_wifi_first_binary_search_split.py sim_src/alg/binary_search_relaxation.py
git commit -m "feat: support wifi-first strategy in binary search run"
```

### Task 4: Remove script-side feasibility orchestration duplication

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_ble_channel_retry.py`
- Modify: `sim_script/tests/test_pd_mmw_ble_channel_modes.py`
- Test: `sim_script/tests/test_pd_mmw_ble_channel_retry.py`
- Test: `sim_script/tests/test_pd_mmw_ble_channel_modes.py`

**Step 1: Write the failing test**

```python
def test_wifi_first_script_path_uses_solver_strategy_output():
    # invoke CLI with --wifi-first-ble-scheduling
    # assert output still contains wifi_first_ble_scheduling = True
    # and stage summary derived from solver metadata
    ...
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest sim_script/tests/test_pd_mmw_ble_channel_retry.py sim_script/tests/test_pd_mmw_ble_channel_modes.py -v`
Expected: FAIL once script-side helpers are removed or changed before being rewired to solver metadata

**Step 3: Write minimal implementation**

```python
# Delete or stop using solve_wifi_first_feasibility(...) in the script.
# Configure:
#   bs.strategy = "wifi_first"
#   bs.pair_radio_type = e.pair_radio_type
#   bs.wifi_radio_id = e.RADIO_WIFI
#   bs.ble_radio_id = e.RADIO_BLE
# Then read bs.last_stage_results for stage summary printing.
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest sim_script/tests/test_pd_mmw_ble_channel_retry.py sim_script/tests/test_pd_mmw_ble_channel_modes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_ble_channel_retry.py sim_script/tests/test_pd_mmw_ble_channel_modes.py sim_src/alg/binary_search_relaxation.py
git commit -m "refactor: move wifi-first feasibility strategy into binary search solver"
```

### Task 5: Run focused verification

**Files:**
- Test: `sim_script/tests/test_wifi_first_binary_search_split.py`
- Test: `sim_script/tests/test_wifi_first_ble_channel_availability.py`
- Test: `sim_script/tests/test_wifi_first_macrocycle_assignment.py`
- Test: `sim_script/tests/test_pd_mmw_ble_channel_retry.py`
- Test: `sim_script/tests/test_pd_mmw_ble_channel_modes.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Run focused solver-and-scheduling suite**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest sim_script/tests/test_wifi_first_binary_search_split.py sim_script/tests/test_wifi_first_ble_channel_availability.py sim_script/tests/test_wifi_first_macrocycle_assignment.py sim_script/tests/test_pd_mmw_ble_channel_retry.py sim_script/tests/test_pd_mmw_ble_channel_modes.py sim_script/tests/test_bler_parameter_selection.py -v`
Expected: PASS

**Step 2: Commit**

```bash
git add sim_src/alg/binary_search_relaxation.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_wifi_first_binary_search_split.py sim_script/tests/test_wifi_first_ble_channel_availability.py sim_script/tests/test_wifi_first_macrocycle_assignment.py
git commit -m "feat: downshift wifi-first strategy into solver api"
```
