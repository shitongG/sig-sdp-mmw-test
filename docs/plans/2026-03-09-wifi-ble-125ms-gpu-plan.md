# WiFi BLE 1.25ms GPU Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the simulator so the base scheduling slot becomes `1.25 ms`, WiFi occupancy duration is sampled from `{5, 6.25, 7.5, 8.75, 10} ms`, BLE CE is constrained to `1.25 ms` quanta with a default max of `7.5 ms`, and the script can run on a user-selected GPU.

**Architecture:** Keep the timing rules centralized in `sim_src/env/env.py` so both WiFi and BLE are quantized against the same `1.25 ms` base slot. Keep scheduling/output logic in `sim_script/pd_mmw_template_ap_stats.py`, but add explicit runtime device selection that flows through `sim_src/util.py` so GPU use is opt-in and deterministic via `--gpu-id`. Avoid mixing timing-model changes with GPU changes in one step; lock timing semantics first, then add device plumbing, then verify the script end to end.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch, subprocess-based smoke tests, test modules under `sim_script/tests`

---

### Task 1: Lock The New 1.25ms Timing Rules In Tests

**Files:**
- Modify: `sim_script/tests/test_wifi_periodic_timing.py`
- Modify: `sim_script/tests/test_ble_anchor_and_ce_window_mask.py`
- Modify: `sim_script/tests/test_ble_ci_discrete_candidates.py`
- Test: `sim_script/tests/test_wifi_periodic_timing.py`
- Test: `sim_script/tests/test_ble_anchor_and_ce_window_mask.py`

**Step 1: Write the failing test**

Update `sim_script/tests/test_wifi_periodic_timing.py` so it pins the new base slot and WiFi duration candidate set.

```python
import numpy as np
from sim_src.env.env import env


def test_base_slot_time_is_1250us():
    e = env(cell_size=2, sta_density_per_1m2=0.01, seed=1)
    assert e.slot_time == 1.25e-3


def test_wifi_tx_duration_candidates_follow_125ms_grid():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=2,
        radio_prob=(1.0, 0.0),
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
    )
    expected_ms = np.array([5.0, 6.25, 7.5, 8.75, 10.0])
    actual_ms = e.wifi_tx_quanta_candidates * e.slot_time * 1e3
    assert np.allclose(actual_ms, expected_ms)
```

Update BLE tests so CE is pinned to `1.25 ms` quanta and default `ble_ce_max_s=7.5e-3`.

```python
def test_ble_ce_range_is_quantized_to_125ms_slots():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=3,
        radio_prob=(0.0, 1.0),
        ble_ce_max_s=7.5e-3,
        ble_phy_rate_bps=2e6,
    )
    ble_idx = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert ble_idx.size > 0
    assert np.all((e.pair_ble_ce_slots[ble_idx] * e.slot_time) >= 1.25e-3)
    assert np.all((e.pair_ble_ce_slots[ble_idx] * e.slot_time) <= 7.5e-3 + 1e-12)
    assert np.allclose((e.pair_ble_ce_slots[ble_idx] * e.slot_time) / 1.25e-3,
                       np.round((e.pair_ble_ce_slots[ble_idx] * e.slot_time) / 1.25e-3))
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_wifi_periodic_timing import (
    test_base_slot_time_is_1250us,
    test_wifi_tx_duration_candidates_follow_125ms_grid,
)

test_base_slot_time_is_1250us()
test_wifi_tx_duration_candidates_follow_125ms_grid()
print('PASS')
```

`PY`

Expected: FAIL because the code currently still uses `slot_time=1.25e-4` and WiFi duration is fixed rather than sampled from the five-value set.

**Step 3: Write minimal implementation**

Modify `sim_src/env/env.py` so:
- default `slot_time=1.25e-3`
- WiFi timing parameters are renamed or extended to:
  - `wifi_tx_min_s=5e-3`
  - `wifi_tx_max_s=10e-3`
- WiFi TX candidate set is built as integer multiples of `slot_time` over `[5, 6.25, 7.5, 8.75, 10] ms`
- BLE CE sampling uses quantized `1.25 ms` steps and default `ble_ce_max_s=7.5e-3`

Implementation sketch:

```python
slot_time=1.25e-3,
ble_ce_max_s=7.5e-3,
...
self.wifi_tx_quanta_candidates = np.arange(
    int(round(wifi_tx_min_s / slot_time)),
    int(round(wifi_tx_max_s / slot_time)) + 1,
    dtype=int,
)
```

Then sample WiFi TX duration from `wifi_tx_quanta_candidates` instead of fixing it to the minimum.

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2 and the BLE CE test.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_wifi_periodic_timing.py sim_script/tests/test_ble_anchor_and_ce_window_mask.py sim_script/tests/test_ble_ci_discrete_candidates.py
git commit -m "feat: quantize wifi and ble timing to 1.25ms base slots"
```

### Task 2: Update Pair-Level Timing State And Macrocycle Helpers

**Files:**
- Modify: `sim_src/env/env.py`
- Modify: `sim_script/tests/test_macrocycle_occupancy.py`
- Modify: `sim_script/tests/test_macrocycle_scheduler.py`
- Test: `sim_script/tests/test_macrocycle_occupancy.py`
- Test: `sim_script/tests/test_macrocycle_scheduler.py`

**Step 1: Write the failing test**

Update macrocycle tests so they expect the new slot scale and WiFi widths sampled from the five allowed TX durations.

```python
def test_wifi_tx_slots_come_from_allowed_set():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=4,
        radio_prob=(1.0, 0.0),
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
    )
    wifi_idx = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    allowed = {4, 5, 6, 7, 8}
    assert set(e.pair_wifi_tx_slots[wifi_idx]).issubset(allowed)
```

Why `{4,5,6,7,8}`: with `slot_time=1.25 ms`, those are exactly `5, 6.25, 7.5, 8.75, 10 ms`.

**Step 2: Run test to verify it fails**

Run the inline version of the updated macrocycle helper tests.
Expected: FAIL until WiFi TX duration sampling and macrocycle helpers are consistent with the new slot size.

**Step 3: Write minimal implementation**

Modify `sim_src/env/env.py` so these fields and helpers reflect the new slot scale cleanly:
- `pair_wifi_tx_slots`
- `pair_ble_ce_slots`
- `get_pair_width_slots()`
- `compute_macrocycle_slots()`
- `expand_pair_occupancy()`

Key requirements:
- all occupancies are now on the `1.25 ms` base slot grid
- WiFi period still follows `1.25ms * 2^n`, `n in {4,5}`
- BLE CI still follows `1.25ms * 2^n`, with its existing exponent range unless explicitly overridden
- BLE CE is quantized to whole base slots and bounded by `7.5 ms` by default

**Step 4: Run test to verify it passes**

Run the updated macrocycle helper test group.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_macrocycle_occupancy.py sim_script/tests/test_macrocycle_scheduler.py
git commit -m "feat: align macrocycle helpers with 1.25ms timing model"
```

### Task 3: Add Explicit GPU Selection Plumbing

**Files:**
- Modify: `sim_src/util.py`
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Create: `sim_script/tests/test_gpu_selection.py`
- Test: `sim_script/tests/test_gpu_selection.py`

**Step 1: Write the failing test**

Create `sim_script/tests/test_gpu_selection.py` with pure helper tests for GPU selection.

```python
from sim_src.util import resolve_torch_device


def test_resolve_torch_device_defaults_to_cpu_when_disabled():
    device = resolve_torch_device(use_gpu=False, gpu_id=0)
    assert str(device) == 'cpu'
```

Add a second test that verifies explicit GPU ID wiring without requiring an actual GPU by monkeypatching `torch.cuda.is_available` if needed.

```python
def test_resolve_torch_device_uses_selected_gpu_id_when_available(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    device = resolve_torch_device(use_gpu=True, gpu_id=2)
    assert str(device) == 'cuda:2'
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_script.tests.test_gpu_selection import test_resolve_torch_device_defaults_to_cpu_when_disabled

test_resolve_torch_device_defaults_to_cpu_when_disabled()
print('PASS')
```

`PY`

Expected: FAIL because device resolution helper does not exist yet.

**Step 3: Write minimal implementation**

In `sim_src/util.py`, add explicit device helpers instead of relying on import-time global CUDA probing.

```python
def resolve_torch_device(use_gpu: bool, gpu_id: int):
    if not use_gpu:
        return torch.device('cpu')
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError('GPU requested but CUDA is unavailable.')
    return torch.device(f'cuda:{gpu_id}')
```

Also update tensor helpers so they accept an explicit device instead of relying on `USE_CUDA` globals.

In `sim_script/pd_mmw_template_ap_stats.py`, add CLI flags:
- `--use-gpu`
- `--gpu-id`

And print the selected runtime device near the top of the script output.

**Step 4: Run test to verify it passes**

Run the inline command from Step 2 plus the second helper test.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/util.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_gpu_selection.py
git commit -m "feat: add explicit gpu device selection"
```

### Task 4: Make The Script Use The New Timing Defaults And Expose GPU Choice

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Modify: `sim_script/tests/test_pd_mmw_ble_ci_summary.py`

**Step 1: Write the failing test**

Extend the script smoke test so it expects the new timing defaults and device summary.

```python
assert 'wifi_period_quanta_candidates: [16, 32]' in proc.stdout
assert 'runtime_device =' in proc.stdout
assert 'macrocycle_slots =' in proc.stdout
```

Add assertions that the pair CSV now includes WiFi TX fields based on the new slot size.

```python
assert 'wifi_tx_slots' in proc.stdout
assert 'wifi_tx_ms' in proc.stdout
```

**Step 2: Run test to verify it fails**

Run the inline version of the updated script smoke test.
Expected: FAIL until the script wiring uses the new defaults and prints device information.

**Step 3: Write minimal implementation**

Modify `sim_script/pd_mmw_template_ap_stats.py` so the script-level `env(...)` call passes:

```python
slot_time=1.25e-3,
wifi_tx_min_s=5e-3,
wifi_tx_max_s=10e-3,
wifi_period_exp_min=4,
wifi_period_exp_max=5,
ble_ce_max_s=7.5e-3,
ble_phy_rate_bps=2e6,
```

Also add:

```python
runtime_device = resolve_torch_device(args.use_gpu, args.gpu_id)
print('runtime_device =', runtime_device)
```

Do not fake GPU acceleration for SciPy-only code. The requirement here is to make runtime device selection explicit and available for code paths using torch utilities, not to claim that all numeric kernels now run on GPU.

**Step 4: Run test to verify it passes**

Run the updated script smoke test.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/tests/test_pd_mmw_ble_ci_summary.py
git commit -m "feat: use 1.25ms timing defaults and expose gpu choice in script"
```

### Task 5: Verify CSV Output Semantics Under The New Slot Scale

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Test: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: Write the failing test**

Update the pair-row logic test so it expects `schedule_time_ms`, `wifi_tx_ms`, `ble_ce_ms`, and `occupied_slots_in_macrocycle` to be computed from the new `1.25 ms` slot base.

```python
assert rows[0]['wifi_tx_ms'] in {5.0, 6.25, 7.5, 8.75, 10.0}
assert rows[1]['ble_ce_ms'] in {1.25, 2.5, 3.75, 5.0, 6.25, 7.5}
```

**Step 2: Run test to verify it fails**

Run the inline logic test group.
Expected: FAIL until all output helpers are aligned to the new slot size.

**Step 3: Write minimal implementation**

Adjust output helpers in `sim_script/pd_mmw_template_ap_stats.py` if needed so printed and CSV-exported ms values reflect the new `1.25 ms` slot size exactly.

Also ensure the exported `wifi_ble_schedule.csv` remains one row per actually occupied macrocycle slot.

**Step 4: Run test to verify it passes**

Run the updated logic tests and one targeted script run:

```bash
python -u sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --seed 3 --mmw-nit 5 --output-dir /tmp/pd_gpu_125ms
```

Expected:
- exit code `0`
- pair CSV exists
- schedule CSV exists
- printed ms values align with `1.25 ms` base slot

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: align csv outputs with 1.25ms slot scale"
```

### Task 6: Final Runtime Verification And Documentation Pass

**Files:**
- Modify: `README.md` (if timing model or GPU selection is documented)
- Test: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: Run CPU smoke check**

Run:

```bash
python -u sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --seed 3 --mmw-nit 5
```

Expected:
- exit code `0`
- output shows `runtime_device = cpu`
- output shows WiFi and BLE timing summaries under the new `1.25 ms` slot scale

**Step 2: Run GPU smoke check if CUDA is available**

Run:

```bash
python -u sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --seed 3 --mmw-nit 5 --use-gpu --gpu-id 0
```

Expected:
- exit code `0` if CUDA is available
- output shows `runtime_device = cuda:0`
- if CUDA is unavailable, the script should fail fast with a clear error message

**Step 3: Update docs only if needed**

If the timing model or runtime options are documented anywhere, add a concise section like:

```markdown
Base scheduling slot is now 1.25 ms.
WiFi TX duration is sampled from {5, 6.25, 7.5, 8.75, 10} ms.
BLE CE is quantized to 1.25 ms and defaults to a maximum of 7.5 ms.
Use `--use-gpu --gpu-id N` to request CUDA device `N`.
```

**Step 4: Commit**

```bash
git add README.md sim_src/env/env.py sim_src/util.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests
git commit -m "docs: clarify 1.25ms timing model and gpu runtime selection"
```
