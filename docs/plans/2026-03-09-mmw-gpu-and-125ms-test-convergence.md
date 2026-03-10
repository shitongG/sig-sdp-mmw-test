# MMW GPU And 1.25ms Test Convergence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the MMW core matrix math onto GPU-backed tensors and update the full relevant test surface so all timing, schedule, and CSV assertions converge on the new 1.25 ms slot semantics.

**Architecture:** Separate the work into two tracks that meet in the script smoke tests: first, make MMW device-aware and port its hottest dense linear algebra paths to torch on an explicit device, while preserving CPU fallback and keeping sparse/problem-setup code coherent. Second, sweep the test suite to remove remaining assumptions tied to the old 0.125 ms slot scale, old WiFi duration model, and old BLE CE cap, then reconnect those tests to the new macrocycle occupancy outputs.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch, subprocess-based smoke tests, test modules under `sim_script/tests` and `sim_src/test`

---

### Task 1: Inventory And Pin The CPU Baseline For MMW

**Files:**
- Modify: `sim_src/test/test_mmw_gpu_baseline.py`
- Modify: `sim_src/alg/mmw.py`
- Test: `sim_src/test/test_mmw_gpu_baseline.py`

**Step 1: Write the failing test**

Create `sim_src/test/test_mmw_gpu_baseline.py` with a deterministic CPU baseline test for a small state.

```python
import numpy as np
from sim_src.alg.mmw import mmw
from sim_src.env.env import env


def test_mmw_cpu_baseline_returns_stable_shape_and_finite_values():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=3,
        slot_time=1.25e-3,
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
        ble_ce_max_s=7.5e-3,
        ble_phy_rate_bps=2e6,
    )
    state = e.generate_S_Q_hmax()
    alg = mmw(nit=3, eta=0.05)
    ok, x_half = alg.run_with_state(0, 4, state)
    assert ok is True
    assert x_half.shape[0] == e.n_pair
    assert np.isfinite(x_half).all()
```

**Step 2: Run test to verify it fails or exposes unstable assumptions**

Run: `python - <<'PY'`

```python
from sim_src.test.test_mmw_gpu_baseline import test_mmw_cpu_baseline_returns_stable_shape_and_finite_values

test_mmw_cpu_baseline_returns_stable_shape_and_finite_values()
print('PASS')
```

`PY`

Expected: PASS or fail in a way that reveals the baseline assumptions needed before GPU migration. Keep the test either way, because it becomes the CPU parity anchor.

**Step 3: Write minimal implementation**

If the baseline test exposes edge-case failures in `mmw.py` for the new 1.25 ms timing model, patch only the minimal bug needed for a stable CPU baseline.

**Step 4: Run test to verify it passes**

Run the same inline command from Step 2.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/test/test_mmw_gpu_baseline.py sim_src/alg/mmw.py
git commit -m "test: pin mmw cpu baseline before gpu migration"
```

### Task 2: Add Explicit Device-Aware MMW Helpers

**Files:**
- Modify: `sim_src/alg/mmw.py`
- Modify: `sim_src/util.py`
- Create: `sim_src/test/test_mmw_device_helpers.py`
- Test: `sim_src/test/test_mmw_device_helpers.py`

**Step 1: Write the failing test**

Create `sim_src/test/test_mmw_device_helpers.py` for helper-level device conversion and CPU/GPU fallback behavior.

```python
import numpy as np
from sim_src.alg.mmw import mmw


def test_mmw_tensor_conversion_returns_cpu_tensor_when_gpu_disabled():
    alg = mmw(nit=1)
    arr = np.eye(3, dtype=float)
    tensor = alg._to_torch(arr, device='cpu')
    assert tuple(tensor.shape) == (3, 3)
    assert str(tensor.device) == 'cpu'
```

Add a second helper test for selected CUDA IDs if torch CUDA is available.

```python
def test_mmw_selects_requested_cuda_device_when_available():
    import torch
    if not torch.cuda.is_available():
        return
    alg = mmw(nit=1, device=torch.device('cuda:0'))
    assert str(alg.device) == 'cuda:0'
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'`

```python
from sim_src.test.test_mmw_device_helpers import test_mmw_tensor_conversion_returns_cpu_tensor_when_gpu_disabled

test_mmw_tensor_conversion_returns_cpu_tensor_when_gpu_disabled()
print('PASS')
```

`PY`

Expected: FAIL because `mmw` is not device-aware yet.

**Step 3: Write minimal implementation**

Refactor `sim_src/alg/mmw.py` so the class accepts an optional explicit device and exposes internal tensor helpers.

```python
class mmw(...):
    def __init__(self, nit=100, rank_radio=2, alpha=1., eta=0.1, log_gap=False, device=None):
        ...
        self.device = device

    def _to_torch(self, arr, device=None, dtype=torch.float64):
        ...
```

Do not migrate the whole algorithm yet. This task is just about making device boundaries explicit and testable.

**Step 4: Run test to verify it passes**

Run the inline helper test group.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/alg/mmw.py sim_src/util.py sim_src/test/test_mmw_device_helpers.py
git commit -m "feat: add explicit device-aware helpers to mmw"
```

### Task 3: Port The Dense MMW Hot Path To Torch

**Files:**
- Modify: `sim_src/alg/mmw.py`
- Modify: `sim_src/test/test_mmw_gpu_baseline.py`
- Create: `sim_src/test/test_mmw_gpu_parity.py`
- Test: `sim_src/test/test_mmw_gpu_parity.py`

**Step 1: Write the failing test**

Create `sim_src/test/test_mmw_gpu_parity.py` to compare CPU and GPU outputs on a tiny deterministic scenario.

```python
import numpy as np
from sim_src.alg.mmw import mmw
from sim_src.env.env import env


def test_mmw_gpu_matches_cpu_on_small_problem():
    import torch
    if not torch.cuda.is_available():
        return
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=4,
        slot_time=1.25e-3,
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
        ble_ce_max_s=7.5e-3,
        ble_phy_rate_bps=2e6,
    )
    state = e.generate_S_Q_hmax()
    cpu_alg = mmw(nit=2, eta=0.05, device=torch.device('cpu'))
    gpu_alg = mmw(nit=2, eta=0.05, device=torch.device('cuda:0'))
    _, cpu_x = cpu_alg.run_with_state(0, 4, state)
    _, gpu_x = gpu_alg.run_with_state(0, 4, state)
    assert cpu_x.shape == gpu_x.shape
    assert np.allclose(cpu_x, gpu_x, atol=1e-5, rtol=1e-4)
```

**Step 2: Run test to verify it fails**

Run the inline parity test.
Expected: FAIL until the dense MMW math uses torch tensors on the selected device.

**Step 3: Write minimal implementation**

Port the densest linear algebra inside `sim_src/alg/mmw.py` to torch while keeping sparse/problem indexing in NumPy/SciPy.

Start with these hot-path operations:
- random sketch generation in `expm_half_randsk`
- row normalization
- dense Gram-style products for `X_half`
- accumulation buffers where GPU provides actual value

Pragmatic implementation guidance:
- convert sparse-derived arrays to dense tensors only where necessary
- keep device transfers coarse-grained, not inside tight inner scalar loops
- preserve CPU fallback when `device='cpu'`
- keep final returned `X_half` as NumPy for downstream compatibility until later tasks explicitly change that API

**Step 4: Run test to verify it passes**

Run the parity test and the CPU baseline test.
Expected: PASS on systems with CUDA; graceful skip/no-op on CPU-only systems.

**Step 5: Commit**

```bash
git add sim_src/alg/mmw.py sim_src/test/test_mmw_gpu_baseline.py sim_src/test/test_mmw_gpu_parity.py
git commit -m "feat: port core mmw dense math to torch device path"
```

### Task 4: Thread Device Selection Into The Script And Solver Construction

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_gpu_selection.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

Extend `sim_script/tests/test_gpu_selection.py` to assert that the script-level solver is built with the resolved device.

```python
def test_script_uses_resolved_device_for_mmw_construction(monkeypatch):
    ...
```

Keep it helper-level if direct subprocess inspection is easier than monkeypatching. The key requirement is that `mmw(device=runtime_device)` is now used instead of ignoring the selected device.

**Step 2: Run test to verify it fails**

Run the inline GPU selection test group.
Expected: FAIL until the script passes `runtime_device` into `mmw(...)`.

**Step 3: Write minimal implementation**

Modify `sim_script/pd_mmw_template_ap_stats.py`:

```python
runtime_device = resolve_torch_device(args.use_gpu, args.gpu_id)
alg = mmw(nit=args.mmw_nit, eta=args.mmw_eta, device=runtime_device)
```

Also make sure the printed device matches the actual solver device.

**Step 4: Run test to verify it passes**

Run the updated helper/smoke test.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_gpu_selection.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: thread selected runtime device into mmw solver"
```

### Task 5: Converge The Legacy Test Surface To 1.25ms Semantics

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_ble_ci_summary.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Modify: `sim_script/tests/test_macrocycle_occupancy.py`
- Modify: `sim_script/tests/test_macrocycle_scheduler.py`
- Modify: any remaining test under `sim_script/tests` or `sim_src/test` that still encodes `0.125 ms`, old WiFi fixed 5 ms occupancy, or old BLE CE max `2.5 ms`

**Step 1: Write the failing test**

Search for stale assumptions first:

Run: `rg -n "1\.25e-4|0\.125|2\.5e-3|wifi_min_tx_time_s|5e-3|schedule_time_ms" sim_script/tests sim_src/test`

Then update the affected tests to assert the new semantics. Example changes:
- `schedule_time_ms` should now advance in `1.25 ms` increments
- BLE CE max default assumptions should use `7.5e-3`
- WiFi TX duration assertions should allow `{5, 6.25, 7.5, 8.75, 10} ms`

**Step 2: Run test to verify it fails**

Run a targeted inline sweep of the modified tests.
Expected: FAIL until all stale assumptions are updated.

**Step 3: Write minimal implementation**

Adjust only the tests and, where necessary, small helper formatting code so all outputs and expected values reflect the new base slot.

Do not change production behavior here unless a genuine mismatch is exposed.

**Step 4: Run test to verify it passes**

Run the updated targeted sweep.
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests sim_src/test
git commit -m "test: converge timing and schedule assertions to 1.25ms model"
```

### Task 6: End-To-End CPU/GPU Verification And Residual Risk Review

**Files:**
- Modify: `README.md` (if runtime/GPU behavior is documented)
- Test: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: Run CPU smoke check**

Run:

```bash
python -u sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --seed 7 --mmw-nit 5 --output-dir /tmp/pd_mmw_cpu
```

Expected:
- exit code `0`
- `runtime_device = cpu`
- CSV outputs exist
- no old CUDA warning in `stderr`

**Step 2: Run GPU smoke check if CUDA is available**

Run:

```bash
python -u sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --seed 7 --mmw-nit 5 --use-gpu --gpu-id 0 --output-dir /tmp/pd_mmw_gpu
```

Expected:
- exit code `0` on CUDA-capable systems
- `runtime_device = cuda:0`
- CPU/GPU outputs have matching shape and compatible magnitude

If CUDA is unavailable, validate the fast-fail path instead.

**Step 3: Review residual technical debt**

Before closing the task, explicitly note any remaining non-GPU paths still living in SciPy/NumPy, for example:
- sparse eigensolvers
- `expm_multiply`
- sparse setup and indexing logic

This is required so the user knows whether “GPU support” is partial acceleration or end-to-end GPU execution.

**Step 4: Update docs only if needed**

If `README.md` or another doc references the old slot scale or vague GPU behavior, update it with concise, exact language.

Example:

```markdown
Base scheduling slot is 1.25 ms.
WiFi TX duration is sampled from {5, 6.25, 7.5, 8.75, 10} ms.
BLE CE is quantized to 1.25 ms with a default maximum of 7.5 ms.
Use `--use-gpu --gpu-id N` to request CUDA device `N`.
Current GPU acceleration covers the torch-backed dense MMW path; sparse SciPy steps remain CPU-bound unless otherwise noted.
```

**Step 5: Commit**

```bash
git add README.md sim_src/alg/mmw.py sim_src/util.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests sim_src/test
git commit -m "docs: clarify gpu-backed mmw path and 1.25ms timing model"
```
