# BLE 2Mbps And Full Pow2 CI Range Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the BLE configuration so the script uses `ble_phy_rate_bps = 2e6` and the BLE CI candidate range follows `CI = 2^n * 1.25ms` for `n in {3,4,5,6,7,8,9,10,11}`, while preserving the existing CE feasibility and sampling constraints.

**Architecture:** The core BLE timing logic already supports power-of-two CI candidates controlled by `ble_ci_exp_min`, `ble_ci_exp_max`, `ble_ci_min_s`, and `ble_ci_max_s`. The implementation should avoid rewriting the timing model; instead, update the script-level configuration that currently narrows CI to `7.5ms..15ms`, add regression tests that pin the new intended behavior, and verify CE constraints still derive from `max(ble_ce_min_s, payload_bits / phy_rate)` and remain capped by `min(ble_ce_max_s, CI)`.

**Tech Stack:** Python 3.10, NumPy, SciPy, pytest

---

### Task 1: Lock The New BLE Defaults In Tests

**Files:**
- Modify: `sim_script/tests/test_ble_ci_discrete_candidates.py`
- Modify: `sim_script/tests/test_pd_mmw_ble_ci_summary.py`
- Test: `sim_script/tests/test_ble_ci_discrete_candidates.py`
- Test: `sim_script/tests/test_pd_mmw_ble_ci_summary.py`

**Step 1: Write the failing test**

Update `sim_script/tests/test_ble_ci_discrete_candidates.py` to pin the full power-of-two candidate list and add a second test that pins the script-facing configuration.

```python
import numpy as np
from sim_src.env.env import env


def test_ble_ci_candidates_follow_pow2_rule():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.005,
        seed=1,
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
    )
    expected = np.array([2**n for n in range(3, 12)], dtype=int)
    assert np.array_equal(e.ble_ci_quanta_candidates, expected)


def test_script_ble_config_exposes_full_pow2_ci_range():
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.1,
        seed=1,
        radio_prob=(0.6, 0.4),
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=2.5e-3,
        ble_payload_bits=800,
        ble_phy_rate_bps=2e6,
        ble_ci_exp_min=3,
        ble_ci_exp_max=11,
    )
    expected = np.array([2**n for n in range(3, 12)], dtype=int)
    assert np.array_equal(e.ble_ci_quanta_candidates, expected)
    assert e.ble_phy_rate_bps == 2e6
```

Update `sim_script/tests/test_pd_mmw_ble_ci_summary.py` so the smoke test checks the printed candidate list is no longer the single-value `10ms` case.

```python
import pathlib
import subprocess
import sys


def test_script_prints_ble_ci_discrete_summary():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "2",
            "--seed",
            "3",
            "--mmw-nit",
            "20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ble_ci_quanta_candidates" in proc.stdout
    assert "[8, 16, 32, 64, 128, 256, 512, 1024, 2048]" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_ci_discrete_candidates.py sim_script/tests/test_pd_mmw_ble_ci_summary.py -v`
Expected: FAIL because the script currently passes `ble_ci_max_s=15e-3` and `ble_phy_rate_bps=1e6`, so the smoke test will still see only `[8]` and the script-facing config assertion will not match `2e6`.

**Step 3: Commit**

Do not commit yet. This task intentionally leaves the branch red until the production code is changed in Task 2.

### Task 2: Update Script-Level BLE Configuration Without Changing CE Rules

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:80-91`
- Test: `sim_script/tests/test_ble_ci_discrete_candidates.py`
- Test: `sim_script/tests/test_pd_mmw_ble_ci_summary.py`

**Step 1: Write the minimal implementation**

Update the `env(...)` construction in `sim_script/pd_mmw_template_ap_stats.py` so it requests the full power-of-two CI range and a 2 Mbps BLE PHY rate.

```python
e = env(
    cell_edge=7.0,
    cell_size=args.cell_size,
    pair_density_per_m2=args.pair_density,
    seed=int(time.time()) if args.seed is None else args.seed,
    radio_prob=(0.6, 0.4),
    ble_ci_min_s=7.5e-3,
    ble_ci_max_s=4.0,
    ble_ce_min_s=1.25e-3,
    ble_ce_max_s=2.5e-3,
    ble_payload_bits=800,
    ble_phy_rate_bps=2e6,
    ble_ci_exp_min=3,
    ble_ci_exp_max=11,
)
```

Implementation notes:
- Keep `ble_ce_min_s` and `ble_ce_max_s` unchanged.
- Do not rewrite `_config_ble_timing()` or `_config_ble_pair_timing()` unless the new tests expose a real bug.
- Preserve the existing CE rules: `ce_required = max(ble_ce_min_s, ble_payload_bits / ble_phy_rate_bps)`, feasibility requires `ce_required <= ble_ce_max_s`, and sampled CE must still satisfy `CE <= min(ble_ce_max_s, CI)`.

**Step 2: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_ci_discrete_candidates.py sim_script/tests/test_pd_mmw_ble_ci_summary.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_ble_ci_discrete_candidates.py sim_script/tests/test_pd_mmw_ble_ci_summary.py
git commit -m "feat: use 2mbps ble phy with full pow2 ci range"
```

### Task 3: Add A CE Regression Test To Prove Constraints Were Preserved

**Files:**
- Modify: `sim_script/tests/test_ble_anchor_and_ce_window_mask.py`
- Test: `sim_script/tests/test_ble_anchor_and_ce_window_mask.py`

**Step 1: Write the failing test**

Append a regression test that checks CE still respects the original rules after the PHY rate change.

```python
import numpy as np
from sim_src.env.env import env


def test_ble_ce_constraints_preserved_with_2mbps_phy():
    e = env(
        cell_size=3,
        sta_density_per_1m2=0.02,
        seed=5,
        radio_prob=(0.0, 1.0),
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=2.5e-3,
        ble_payload_bits=800,
        ble_phy_rate_bps=2e6,
        ble_ci_exp_min=3,
        ble_ci_exp_max=11,
    )
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    assert ble_idx.size > 0
    assert e.ble_ce_required_s == 1.25e-3

    for k in ble_idx:
        ci_s = e.user_ble_ci_slots[k] * e.slot_time
        if not e.user_ble_ce_feasible[k]:
            assert e.user_ble_ce_slots[k] == 0
            continue
        ce_s = e.user_ble_ce_slots[k] * e.slot_time
        assert ce_s >= e.ble_ce_required_s
        assert ce_s >= e.ble_ce_min_s
        assert ce_s <= e.ble_ce_max_s + e.slot_time
        assert ce_s <= ci_s + e.slot_time
```

**Step 2: Run test to verify it fails or guards the intended behavior**

Run: `pytest sim_script/tests/test_ble_anchor_and_ce_window_mask.py::test_ble_ce_constraints_preserved_with_2mbps_phy -v`
Expected: If the implementation accidentally changed CE behavior, FAIL. If it already passes immediately, keep it as the regression that proves the refactor did not loosen CE constraints.

**Step 3: Run the related BLE test set**

Run: `pytest sim_script/tests/test_ble_ci_discrete_candidates.py sim_script/tests/test_ble_ci_sampling_pow2.py sim_script/tests/test_ble_anchor_and_ce_window_mask.py sim_script/tests/test_pd_mmw_ble_ci_summary.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add sim_script/tests/test_ble_anchor_and_ce_window_mask.py
git commit -m "test: cover ble ce constraints under 2mbps phy"
```

### Task 4: Run An End-To-End Smoke Check And Document The Result

**Files:**
- Modify: `README.md` (only if it already documents the old BLE PHY or the old CI script behavior)
- Test: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: Run the script smoke test with explicit small parameters**

Run: `python sim_script/pd_mmw_template_ap_stats.py --cell-size 2 --pair-density 0.05 --seed 3 --mmw-nit 20`
Expected:
- Exit code `0`
- Printed `ble_ci_quanta_candidates` includes `[8, 16, 32, 64, 128, 256, 512, 1024, 2048]`
- BLE timing summary still reports CE slot statistics in the same shape as before

**Step 2: Update docs only if needed**

If `README.md` or another checked-in doc mentions `1 Mbps` BLE PHY or the narrowed `7.5ms..15ms` script CI range, update that exact text with the new script defaults. If no docs mention it, skip this step.

Example patch content if needed:

```markdown
- BLE PHY rate in `sim_script/pd_mmw_template_ap_stats.py` is configured as `2 Mbps`.
- BLE CI candidates follow `CI = 2^n * 1.25 ms` for `n = 3..11`.
- CE constraints remain `CE >= max(ble_ce_min_s, payload_bits / phy_rate)` and `CE <= min(ble_ce_max_s, CI)`.
```

**Step 3: Commit**

```bash
git add README.md sim_script/pd_mmw_template_ap_stats.py sim_script/tests
git commit -m "docs: align ble timing defaults with script behavior"
```
