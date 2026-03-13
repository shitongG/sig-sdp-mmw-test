# Bandwidth-Dependent Noise Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fixed receive-noise floor with a bandwidth-dependent thermal-noise model so scheduling constraints and BLER evaluation use `N_dBm = -174 + 10*log10(B_Hz) + NOISEFIGURE`.

**Architecture:** Keep the public `env` API stable and change only the internal noise-power conversion path. Update `bandwidth_txpr_to_noise_dBm()` to compute array-safe bandwidth-dependent noise, then verify that `_compute_txp()`, `_compute_state()`, `_compute_state_real()`, `generate_S_Q_hmax()`, and `evaluate_bler()` all continue to work with scalar and per-link bandwidth arrays.

**Tech Stack:** Python, NumPy, SciPy, pytest

---

### Task 1: Add failing tests for the bandwidth-dependent noise formula

**Files:**
- Modify: `sim_script/tests/test_bler_parameter_selection.py`
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Write the failing test**

```python
def test_bandwidth_txpr_to_noise_dbm_uses_thermal_noise_formula():
    bandwidths = np.array([1e6, 2e6, 5e6], dtype=float)

    expected = -174.0 + 10.0 * np.log10(bandwidths) + env.NOISEFIGURE

    assert np.allclose(env.bandwidth_txpr_to_noise_dBm(bandwidths), expected)
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_bandwidth_txpr_to_noise_dbm_uses_thermal_noise_formula -v`
Expected: FAIL because the method still returns the fixed `NOISE_FLOOR_DBM`

**Step 3: Write minimal implementation**

```python
@classmethod
def bandwidth_txpr_to_noise_dBm(cls, B):
    bandwidth_hz = np.asarray(B, dtype=float)
    if np.any(bandwidth_hz <= 0):
        raise ValueError("bandwidth must be positive.")
    return -174.0 + 10.0 * np.log10(bandwidth_hz) + cls.NOISEFIGURE
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_bandwidth_txpr_to_noise_dbm_uses_thermal_noise_formula -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_src/env/env.py
git commit -m "feat: add bandwidth-dependent noise model"
```

### Task 2: Add failing tests for scalar and per-link constraint integration

**Files:**
- Modify: `sim_script/tests/test_bler_parameter_selection.py`
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Write the failing test**

```python
def test_compute_txp_uses_bandwidth_dependent_noise_per_link():
    n_pair = int((2**2) * 0.05 * (7.0**2))
    packet_bits = np.linspace(500.0, 1300.0, n_pair)
    bandwidths = np.linspace(1e6, 5e6, n_pair)
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=7,
        pair_packet_bits=packet_bits,
        pair_bandwidth_hz=bandwidths,
    )

    dis = np.linalg.norm(e.pair_tx_locs - e.pair_rx_locs, axis=1)
    gain = -env.fre_dis_to_loss_dB(e.fre_Hz, dis)
    expected = env.dec_to_db(e.pair_min_sinr) - (
        gain - env.bandwidth_txpr_to_noise_dBm(e.pair_bandwidth_hz)
    )
    expected = expected + env.dec_to_db(e.txp_offset)

    assert np.allclose(e._compute_txp().ravel(), expected)
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_compute_txp_uses_bandwidth_dependent_noise_per_link -v`
Expected: FAIL while `_compute_txp()` still behaves like the old fixed noise floor

**Step 3: Write minimal implementation**

```python
# Reuse bandwidth_txpr_to_noise_dBm(self.pair_bandwidth_hz) directly in _compute_txp
# without special-casing scalar noise floors.
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_compute_txp_uses_bandwidth_dependent_noise_per_link -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_src/env/env.py
git commit -m "feat: use bandwidth-dependent noise in tx power calculation"
```

### Task 3: Add failing tests for state-matrix and h-max propagation

**Files:**
- Modify: `sim_script/tests/test_bler_parameter_selection.py`
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Write the failing test**

```python
def test_generate_s_q_hmax_changes_with_bandwidth_dependent_noise():
    n_pair = int((2**2) * 0.05 * (7.0**2))
    packet_bits = np.linspace(500.0, 1300.0, n_pair)
    low_bw = np.full(n_pair, 1e6)
    high_bw = np.full(n_pair, 5e6)

    e_low = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=8,
        pair_packet_bits=packet_bits,
        pair_bandwidth_hz=low_bw,
    )
    e_high = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=8,
        pair_packet_bits=packet_bits,
        pair_bandwidth_hz=high_bw,
    )

    _, _, h_low = e_low.generate_S_Q_hmax()
    _, _, h_high = e_high.generate_S_Q_hmax()

    assert not np.allclose(h_low, h_high)
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_generate_s_q_hmax_changes_with_bandwidth_dependent_noise -v`
Expected: FAIL if fixed noise still makes the state path insensitive to bandwidth

**Step 3: Write minimal implementation**

```python
# Ensure _compute_state and _compute_state_real propagate the bandwidth-dependent
# noise vector into received-power normalization without collapsing back to a scalar.
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_generate_s_q_hmax_changes_with_bandwidth_dependent_noise -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_src/env/env.py
git commit -m "feat: propagate bandwidth-dependent noise into scheduling state"
```

### Task 4: Verify compatibility and guardrails

**Files:**
- Modify: `sim_script/tests/test_bler_parameter_selection.py`
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`
- Test: `sim_script/tests/test_pair_env_structure.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

```python
def test_bandwidth_txpr_to_noise_dbm_rejects_non_positive_bandwidth():
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        env.bandwidth_txpr_to_noise_dBm(np.array([0.0, 1e6]))
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_bandwidth_txpr_to_noise_dbm_rejects_non_positive_bandwidth -v`
Expected: FAIL because the old implementation accepts any value

**Step 3: Write minimal implementation**

```python
# Add validation in bandwidth_txpr_to_noise_dBm before computing the noise floor.
```

**Step 4: Run focused regression suite**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py tests/test_pair_env_structure.py tests/test_pd_mmw_template_ap_stats_logic.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_script/tests/test_pair_env_structure.py sim_src/env/env.py
git commit -m "feat: make receiver noise depend on bandwidth"
```
