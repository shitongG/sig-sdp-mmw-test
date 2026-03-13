# Radio-Specific BLER Parameters Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make BLER evaluation use WiFi/BLE-specific bandwidth and packet length defaults, while supporting per-pair overrides without breaking existing callers.

**Architecture:** Keep the existing `env` scheduling model intact and limit the behavior change to BLER parameter selection. Add optional constructor inputs for WiFi/BLE defaults plus per-pair and per-user overrides, normalize them into internal `pair_*`, `device_*`, and `user_*` arrays, and make `evaluate_bler()` consume the normalized per-user arrays.

**Tech Stack:** Python, NumPy, SciPy, pytest

---

### Task 1: Add regression tests for radio-specific BLER parameter normalization

**Files:**
- Create: `sim_script/tests/test_bler_parameter_selection.py`
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_src.env.env import env


def test_env_exposes_radio_specific_user_packet_and_bandwidth_arrays():
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=1,
        wifi_packet_bit=12000,
        ble_packet_bit=320,
    )

    expected_packet_bits = np.where(
        e.user_radio_type == e.RADIO_WIFI,
        12000.0,
        320.0,
    )
    expected_bandwidths = np.where(
        e.user_radio_type == e.RADIO_WIFI,
        e.wifi_channel_bandwidth_hz,
        e.ble_channel_bandwidth_hz,
    )

    assert np.array_equal(e.user_packet_bits, expected_packet_bits)
    assert np.array_equal(e.user_bandwidth_hz, expected_bandwidths)
    assert np.array_equal(e.pair_packet_bits, e.user_packet_bits)
    assert np.array_equal(e.pair_bandwidth_hz, e.user_bandwidth_hz)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bler_parameter_selection.py::test_env_exposes_radio_specific_user_packet_and_bandwidth_arrays -v`
Expected: FAIL with missing constructor arguments or missing attributes such as `user_packet_bits`

**Step 3: Write minimal implementation**

```python
# Add optional constructor args:
# wifi_packet_bit=None, ble_packet_bit=None,
# pair_packet_bits=None, user_packet_bits=None,
# pair_bandwidth_hz=None, user_bandwidth_hz=None
#
# Normalize them after radio types are sampled:
# self.pair_packet_bits = ...
# self.pair_bandwidth_hz = ...
# self.device_packet_bits = self.pair_packet_bits
# self.user_packet_bits = self.pair_packet_bits
# self.device_bandwidth_hz = self.pair_bandwidth_hz
# self.user_bandwidth_hz = self.pair_bandwidth_hz
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bler_parameter_selection.py::test_env_exposes_radio_specific_user_packet_and_bandwidth_arrays -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_src/env/env.py
git commit -m "feat: add radio-specific bler parameter defaults"
```

### Task 2: Add regression tests for per-pair override precedence in BLER

**Files:**
- Modify: `sim_script/tests/test_bler_parameter_selection.py`
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Write the failing test**

```python
def test_evaluate_bler_uses_per_pair_packet_bits_and_bandwidths():
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=2,
        pair_packet_bits=np.array([100.0, 200.0, 300.0, 400.0]),
        pair_bandwidth_hz=np.array([1e6, 2e6, 3e6, 4e6]),
    )
    z = np.arange(e.n_pair, dtype=int)
    Z = int(e.n_pair)

    sinr = e.evaluate_sinr(z, Z)
    expected = np.array(
        [
            env.polyanskiy_model(sinr[k], e.pair_packet_bits[k], e.pair_bandwidth_hz[k], e.slot_time)
            for k in range(e.n_pair)
        ]
    )

    assert np.allclose(e.evaluate_bler(z, Z), expected)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bler_parameter_selection.py::test_evaluate_bler_uses_per_pair_packet_bits_and_bandwidths -v`
Expected: FAIL because `evaluate_bler()` still uses scalar `self.packet_bit` and `self.bandwidth`

**Step 3: Write minimal implementation**

```python
def evaluate_bler(self, z, Z):
    sinr = self.evaluate_sinr(z, Z)
    return np.array(
        [
            env.polyanskiy_model(sinr[k], self.user_packet_bits[k], self.user_bandwidth_hz[k], self.slot_time)
            for k in range(sinr.size)
        ]
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bler_parameter_selection.py::test_evaluate_bler_uses_per_pair_packet_bits_and_bandwidths -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_src/env/env.py
git commit -m "feat: use per-pair parameters in bler evaluation"
```

### Task 3: Verify compatibility of existing env structure tests

**Files:**
- Modify: `sim_script/tests/test_pair_env_structure.py`
- Test: `sim_script/tests/test_pair_env_structure.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Write the failing test**

```python
def test_env_builds_pair_level_bler_parameter_arrays():
    e = env(cell_edge=7.0, cell_size=2, pair_density_per_m2=0.05, seed=1)
    assert e.pair_packet_bits.shape[0] == e.n_pair
    assert e.pair_bandwidth_hz.shape[0] == e.n_pair
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pair_env_structure.py::test_env_builds_pair_level_bler_parameter_arrays -v`
Expected: FAIL because the arrays do not exist yet or are not wired to pair semantics

**Step 3: Write minimal implementation**

```python
# Extend pair/device/user compatibility mapping so the new arrays are always populated.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pair_env_structure.py::test_env_builds_pair_level_bler_parameter_arrays -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_pair_env_structure.py sim_src/env/env.py
git commit -m "test: cover pair-level bler parameter arrays"
```

### Task 4: Run focused verification

**Files:**
- Test: `sim_script/tests/test_bler_parameter_selection.py`
- Test: `sim_script/tests/test_pair_env_structure.py`

**Step 1: Run focused regression suite**

Run: `pytest tests/test_bler_parameter_selection.py tests/test_pair_env_structure.py -v`
Expected: PASS with all tests green

**Step 2: Run one script-level smoke check if needed**

Run: `pytest tests/test_pd_mmw_template_ap_stats_logic.py -v`
Expected: PASS, proving the env change did not break a representative script-facing test

**Step 3: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_script/tests/test_pair_env_structure.py sim_src/env/env.py
git commit -m "feat: add radio-specific bler parameter handling"
```
