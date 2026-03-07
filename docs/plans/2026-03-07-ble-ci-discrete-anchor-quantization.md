# BLE CI 离散化与 Anchor 量化 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 BLE 的 CI 采样改为离散集合 `CI = 2^n * 1.25ms (n=3..11)`，并保持 `anchor + m*CI` 展开 CE 窗口，同时保证 anchor 量化到仿真最小时隙（`slot_time=0.125ms=125us`）的整数倍。

**Architecture:** 仅改 `env.py` 的 BLE 时序参数生成与 mask 构建逻辑，不改 MMW 主体优化与 rounding 数学模型。通过新增可单测的“CI 候选生成/采样”函数和 anchor 量化断言，确保规则可验证、行为可复现。最后在实验脚本中打印新增摘要字段，便于核对是否满足离散 CI 规则。

**Tech Stack:** Python 3, NumPy, SciPy, 项目内 `sim_src/env/env.py` 与 `sim_script/*.py`

---

### Task 1: CI 离散候选集合与参数校验

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_ble_ci_discrete_candidates.py`

**Step 1: Write the failing test**

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
    # 期望候选: 2^n * 1.25ms, n=3..11
    expected = np.array([2**n for n in range(3, 12)], dtype=int)
    assert np.array_equal(e.ble_ci_quanta_candidates, expected)
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_ci_discrete_candidates.py -v`
Expected: FAIL（`ble_ci_quanta_candidates` 尚未实现）

**Step 3: Write minimal implementation**

```python
# env.__init__
self.ble_ci_exp_min = 3
self.ble_ci_exp_max = 11
self.ble_ci_quanta_candidates = None

# _config_ble_timing
pow2_quanta = np.array([2**n for n in range(self.ble_ci_exp_min, self.ble_ci_exp_max + 1)], dtype=int)
# 与 ble_ci_min_s/ble_ci_max_s 取交集
q_min = int(math.ceil(ci_min / ci_base))
q_max = int(math.floor(ci_max / ci_base))
pow2_quanta = pow2_quanta[(pow2_quanta >= q_min) & (pow2_quanta <= q_max)]
if pow2_quanta.size == 0:
    raise ValueError("No valid BLE CI candidates under 2^n*1.25ms rule.")
self.ble_ci_quanta_candidates = pow2_quanta
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_ci_discrete_candidates.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_ble_ci_discrete_candidates.py
git commit -m "feat: enforce BLE CI discrete candidates with 2^n*1.25ms rule"
```

### Task 2: BLE 用户 CI 采样改为离散候选抽样

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_ble_ci_sampling_pow2.py`

**Step 1: Write the failing test**

```python
import numpy as np
from sim_src.env.env import env


def test_ble_user_ci_is_from_discrete_candidates():
    e = env(cell_size=3, sta_density_per_1m2=0.01, seed=2, radio_prob=(0.0, 1.0))
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    ci_quanta = np.rint((e.user_ble_ci_slots[ble_idx] * e.slot_time) / 1.25e-3).astype(int)
    assert np.all(np.isin(ci_quanta, e.ble_ci_quanta_candidates))
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_ci_sampling_pow2.py -v`
Expected: FAIL（当前采样是连续区间整数而非候选集合）

**Step 3: Write minimal implementation**

```python
# _config_ble_user_timing
ci_quanta = int(self.rand_gen_loc.choice(self.ble_ci_quanta_candidates))
ci_s = ci_quanta * ci_base
ci_slots = int(round(ci_s / self.slot_time))
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_ci_sampling_pow2.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_ble_ci_sampling_pow2.py
git commit -m "feat: sample BLE CI from discrete pow2 candidate set"
```

### Task 3: Anchor 量化到最小时隙并保持 CE 窗口展开

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_ble_anchor_and_ce_window_mask.py`

**Step 1: Write the failing test**

```python
import numpy as np
from sim_src.env.env import env


def test_ble_anchor_quantized_and_window_periodic():
    e = env(cell_size=2, sta_density_per_1m2=0.01, seed=7, radio_prob=(0.0, 1.0))
    Z = 300
    mask = e.build_slot_compatibility_mask(Z)
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    for k in ble_idx:
        if not e.user_ble_ce_feasible[k]:
            assert not mask[k].any()
            continue
        ci = int(e.user_ble_ci_slots[k])
        ce = int(e.user_ble_ce_slots[k])
        anchor = int(e.user_ble_anchor_slot[k])
        assert 0 <= anchor < ci
        # anchor 本身就是 slot 索引，天然满足 slot_time 量化
        assert abs((anchor * e.slot_time) / e.slot_time - anchor) < 1e-12
        assert mask[k, anchor:min(anchor+ce, Z)].all()
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_anchor_and_ce_window_mask.py -v`
Expected: FAIL（若 anchor/窗口逻辑未显式约束或有边界问题）

**Step 3: Write minimal implementation**

```python
# _config_ble_user_timing
self.user_ble_anchor_slot[k] = int(self.rand_gen_loc.integers(low=0, high=self.user_ble_ci_slots[k]))

# build_slot_compatibility_mask
anchor = int(self.user_ble_anchor_slot[k] % ci_slots)
z = anchor
while z < Z:
    end_z = min(z + ce_slots, Z)
    mask[k, z:end_z] = True
    z += ci_slots
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_anchor_and_ce_window_mask.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_ble_anchor_and_ce_window_mask.py
git commit -m "test: enforce BLE anchor slot quantization and periodic CE window mask"
```

### Task 4: 输出新增 CI 离散规则摘要并做实验验证

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Create: `sim_script/output/pd_mmw_template_ap_stats-ci-pow2-<timestamp>.txt`

**Step 1: Write the failing test**

```python
import subprocess
import sys


def test_script_prints_ble_ci_discrete_summary():
    proc = subprocess.run(
        [sys.executable, "sim_script/pd_mmw_template_ap_stats.py", "--cell-size", "2", "--sta-density", "0.005", "--seed", "3", "--mmw-nit", "20"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "ble_ci_quanta_candidates" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_ble_ci_summary.py -v`
Expected: FAIL（脚本尚未打印该摘要）

**Step 3: Write minimal implementation**

```python
print("ble_ci_quanta_candidates:", e.ble_ci_quanta_candidates.tolist())
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pd_mmw_ble_ci_summary.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_ble_ci_summary.py
git commit -m "feat: print BLE discrete CI candidate summary in experiment script"
```

### Task 5: 实验运行与输出核对

**Files:**
- Create: `sim_script/output/pd_mmw_template_ap_stats-ci-pow2-<timestamp>.txt`

**Step 1: Run experiment**

Run:

```bash
C:\Users\18236\.conda\envs\sig-sdp\python.exe -u sim_script\pd_mmw_template_ap_stats.py --cell-size 4 --sta-density 0.01 --seed 20260307 --mmw-nit 80 > sim_script\output\pd_mmw_template_ap_stats-ci-pow2-<timestamp>.txt
```

Expected: 脚本成功运行，输出 `MMW result`、`BLE timing summary`、`ble_ci_quanta_candidates`、AP 统计表头。

**Step 2: Verify key lines**

Run:

```bash
rg "MMW result|BLE timing summary|ble_ci_quanta_candidates|ap_id,wifi_user_count" sim_script/output/pd_mmw_template_ap_stats-ci-pow2-<timestamp>.txt
```

Expected: 四类关键行均存在。

**Step 3: Commit**

```bash
git add sim_script/output/pd_mmw_template_ap_stats-ci-pow2-<timestamp>.txt
git commit -m "chore: add experiment output for BLE discrete CI and anchor-CE window behavior"
```
