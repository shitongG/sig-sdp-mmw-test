# Per-Link Scheduling Constraints Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make env scheduling constraints use per-link packet length and bandwidth so WiFi/BLE defaults and pair-level overrides affect both feasibility and BLER evaluation consistently.

**Architecture:** Extend env's normalized per-link PHY arrays with a cached `pair_min_sinr` array. Refactor the constraint path to consume per-link arrays in `_compute_min_sinr()`, `_compute_txp()`, `_compute_state()`, `_compute_state_real()`, and `generate_S_Q_hmax()` while preserving existing return shapes and pair/device/user compatibility fields.

**Tech Stack:** Python, NumPy, SciPy, pytest

---

### Task 1: Add failing tests for per-link min-SINR defaults and overrides

**Files:**
- Modify: `sim_script/tests/test_bler_parameter_selection.py`
- Modify: `sim_script/tests/test_pair_env_structure.py`
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Write the failing test**

```python
def test_env_computes_per_link_min_sinr_from_radio_defaults():
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=3,
        radio_prob=(0.5, 0.5),
        wifi_packet_bit=12000,
        ble_packet_bit=320,
    )

    expected = np.array(
        [
            env.db_to_dec(env.bisection_method(e.pair_packet_bits[k], e.pair_bandwidth_hz[k], e.slot_time, e.max_err))
            for k in range(e.n_pair)
        ]
    )

    assert np.allclose(e.pair_min_sinr, expected)
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_env_computes_per_link_min_sinr_from_radio_defaults -v`
Expected: FAIL because `pair_min_sinr` does not exist or still uses one shared scalar threshold

**Step 3: Write minimal implementation**

```python
def _compute_min_sinr(self):
    self.pair_min_sinr = np.array([...])
    self.user_min_sinr = self.pair_min_sinr
    self.device_min_sinr = self.pair_min_sinr
    return self.pair_min_sinr
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_env_computes_per_link_min_sinr_from_radio_defaults -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_src/env/env.py
git commit -m "feat: compute per-link min sinr thresholds"
```

### Task 2: Add failing tests for per-link h_max behavior

**Files:**
- Modify: `sim_script/tests/test_bler_parameter_selection.py`
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Write the failing test**

```python
def test_generate_s_q_hmax_uses_per_link_min_sinr():
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=4,
        pair_packet_bits=np.linspace(200.0, 200.0 * n_pair, n_pair),
        pair_bandwidth_hz=np.linspace(1e6, float(n_pair) * 1e6, n_pair),
    )

    s_gain, _, h_max = e.generate_S_Q_hmax()
    expected = s_gain.diagonal() / e.pair_min_sinr - 1.0

    assert np.allclose(h_max, expected)
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_generate_s_q_hmax_uses_per_link_min_sinr -v`
Expected: FAIL because `generate_S_Q_hmax()` still divides by one scalar threshold

**Step 3: Write minimal implementation**

```python
h_max = S_gain.diagonal() / self._compute_min_sinr() - 1.0
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py::test_generate_s_q_hmax_uses_per_link_min_sinr -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_src/env/env.py
git commit -m "feat: apply per-link thresholds to h-max"
```

### Task 3: Refactor per-link transmit power and state normalization

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_bler_parameter_selection.py`

**Step 1: Write the failing test**

```python
def test_compute_txp_and_state_follow_per_link_bandwidth():
    e = env(...)
    txp = e._compute_txp()
    rxpr = e._compute_state_real().toarray()
    assert txp.shape == (e.n_pair, 1)
    assert rxpr.shape == (e.n_pair, e.n_pair)
```

**Step 2: Run test to verify it fails**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py -k per_link_bandwidth -v`
Expected: FAIL or expose scalar-only behavior

**Step 3: Write minimal implementation**

```python
# Use pair_bandwidth_hz in _compute_txp, _compute_state, and _compute_state_real
# with column-wise broadcast for transmitter-specific noise normalization.
```

**Step 4: Run test to verify it passes**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py -k per_link_bandwidth -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_bler_parameter_selection.py
git commit -m "feat: use per-link bandwidth in scheduling state"
```

### Task 4: Run focused verification

**Files:**
- Test: `sim_script/tests/test_bler_parameter_selection.py`
- Test: `sim_script/tests/test_pair_env_structure.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Run focused regression suite**

Run: `/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest tests/test_bler_parameter_selection.py tests/test_pair_env_structure.py tests/test_pd_mmw_template_ap_stats_logic.py -v`
Expected: PASS

**Step 2: Commit**

```bash
git add sim_script/tests/test_bler_parameter_selection.py sim_script/tests/test_pair_env_structure.py sim_src/env/env.py
git commit -m "feat: align scheduling constraints with per-link radio parameters"
```
