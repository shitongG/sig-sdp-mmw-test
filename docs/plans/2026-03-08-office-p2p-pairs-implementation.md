# Office P2P Pair Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将当前仿真从“用户级 AP 互斥调度”改造成“办公室场景下的 WiFi/BLE 通信对调度”，支持同办公室内不同非重叠信道 pair 并发，并保留业务优先级。

**Architecture:** 主要改造 `sim_src/env/env.py` 的数据模型和状态生成，使其直接生成 pair 级任务对象，并输出 pair 级 `S_gain/Q_conflict/h_max`。优化器主体 `binary_search_relaxation + mmw + rounding` 只复用其 K 维调度能力，优先级从 `user_priority` 平移为 `pair_priority`。实验脚本和统计输出同步改到 pair 语义。

**Tech Stack:** Python 3, NumPy, SciPy, 项目内 `sim_src/env`, `sim_src/alg`, `sim_script`

---

### Task 1: 建立 pair 级环境数据结构

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pair_env_structure.py`

**Step 1: Write the failing test**

```python
import numpy as np
from sim_src.env.env import env


def test_env_builds_pair_level_arrays():
    e = env(cell_edge=7.0, cell_size=2, sta_density_per_1m2=0.01, seed=1)
    assert hasattr(e, "n_pair")
    assert e.pair_radio_type.shape[0] == e.n_pair
    assert e.pair_priority.shape[0] == e.n_pair
    assert e.pair_room_id.shape[0] == e.n_pair
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pair_env_structure.py -v`
Expected: FAIL because `n_pair/pair_*` arrays do not exist yet.

**Step 3: Write minimal implementation**

```python
# in env.__init__
self.n_pair = 0
self.pair_radio_type = None
self.pair_priority = None
self.pair_room_id = None
self.pair_channel = None
self.pair_tx_locs = None
self.pair_rx_locs = None

# create pair-level generation path
self._config_pairs()
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pair_env_structure.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pair_env_structure.py
git commit -m "feat: add pair-level environment structure"
```

### Task 2: 办公室内生成 WiFi/BLE 通信对并保留业务优先级

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pair_generation_rules.py`

**Step 1: Write the failing test**

```python
import numpy as np
from sim_src.env.env import env


def test_wifi_pairs_are_pair_objects_and_priority_is_pair_level():
    e = env(cell_edge=7.0, cell_size=3, sta_density_per_1m2=0.02, seed=2)
    wifi_idx = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    ble_idx = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert e.pair_priority.shape[0] == e.n_pair
    assert np.all(e.pair_channel[wifi_idx] >= 0)
    assert np.all(e.pair_channel[ble_idx] >= 0)
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pair_generation_rules.py -v`
Expected: FAIL because pair generation is not wired yet.

**Step 3: Write minimal implementation**

```python
def _config_pairs(self):
    # each room generates wifi pairs and ble pairs directly
    # assign pair_room_id, pair_radio_type, pair_channel
    # assign pair_priority using prio_prob/prio_value
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pair_generation_rules.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pair_generation_rules.py
git commit -m "feat: generate office-level wifi and ble pair tasks with pair priorities"
```

### Task 3: BLE pair 时序迁移到 pair 层

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pair_ble_timing.py`

**Step 1: Write the failing test**

```python
import numpy as np
from sim_src.env.env import env


def test_ble_pair_timing_arrays_exist_and_are_valid():
    e = env(cell_edge=7.0, cell_size=2, sta_density_per_1m2=0.02, seed=3, radio_prob=(0.0, 1.0))
    ble_idx = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert np.all(e.pair_ble_ci_slots[ble_idx] > 0)
    assert np.all(e.pair_ble_ce_slots[ble_idx] <= e.pair_ble_ci_slots[ble_idx])
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pair_ble_timing.py -v`
Expected: FAIL because timing is still user-level.

**Step 3: Write minimal implementation**

```python
# replace user_ble_* arrays with pair_ble_* arrays
# move BLE timing generation to pair-level objects
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pair_ble_timing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pair_ble_timing.py
git commit -m "feat: move BLE CI CE anchor timing to pair-level tasks"
```

### Task 4: 去掉同 AP 全互斥，改成显式频谱冲突图

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_same_office_non_overlapping_pairs_can_share_slot.py`

**Step 1: Write the failing test**

```python
import numpy as np
from sim_src.env.env import env


def test_same_office_non_overlapping_pairs_not_forced_mutually_exclusive():
    e = env(cell_edge=7.0, cell_size=1, sta_density_per_1m2=0.05, seed=4)
    _, q_conflict, _ = e.generate_S_Q_hmax()
    q = q_conflict.toarray()
    # expect at least one pair of tasks in same office not forced to conflict
    assert np.any(q == 0)
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_same_office_non_overlapping_pairs_can_share_slot.py -v`
Expected: FAIL while same-office all-mutual exclusion still exists.

**Step 3: Write minimal implementation**

```python
# remove asso_indicator-based same-AP blanket exclusion
# Q_conflict should come from spectrum overlap rules only
Q_conflict = self._build_radio_interference_constraints(K)
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_same_office_non_overlapping_pairs_can_share_slot.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_same_office_non_overlapping_pairs_can_share_slot.py
git commit -m "feat: remove blanket same-office exclusion and use spectrum conflict graph"
```

### Task 5: 将 `S_gain/Q_conflict/h_max` 改成 pair 级状态

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pair_state_generation.py`

**Step 1: Write the failing test**

```python
from sim_src.env.env import env


def test_generate_state_uses_pair_count():
    e = env(cell_edge=7.0, cell_size=2, sta_density_per_1m2=0.02, seed=5)
    s_gain, q_conflict, h_max = e.generate_S_Q_hmax()
    assert s_gain.shape[0] == e.n_pair
    assert q_conflict.shape[0] == e.n_pair
    assert h_max.shape[0] == e.n_pair
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pair_state_generation.py -v`
Expected: FAIL because current state still uses user count.

**Step 3: Write minimal implementation**

```python
# compute link budget from pair_tx_locs to AP anchors / pair_rx_locs as selected model
# build pair-level state matrices
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pair_state_generation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pair_state_generation.py
git commit -m "feat: generate pair-level scheduling state matrices"
```

### Task 6: 把 BLE 可用窗口 mask 改成 pair 级

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pair_slot_mask.py`

**Step 1: Write the failing test**

```python
import numpy as np
from sim_src.env.env import env


def test_pair_slot_mask_shape_matches_pair_count():
    e = env(cell_edge=7.0, cell_size=2, sta_density_per_1m2=0.02, seed=6)
    mask = e.build_slot_compatibility_mask(100)
    assert mask.shape == (e.n_pair, 100)
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pair_slot_mask.py -v`
Expected: FAIL because mask still uses user arrays.

**Step 3: Write minimal implementation**

```python
# iterate over BLE pairs, not BLE users
# use pair_ble_anchor_slot / pair_ble_ci_slots / pair_ble_ce_slots
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pair_slot_mask.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pair_slot_mask.py
git commit -m "feat: build pair-level BLE slot compatibility mask"
```

### Task 7: 评估函数与加权 BLER 迁移到 pair 级

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pair_evaluation.py`

**Step 1: Write the failing test**

```python
import numpy as np
from sim_src.env.env import env


def test_weighted_bler_uses_pair_priority():
    e = env(cell_edge=7.0, cell_size=2, sta_density_per_1m2=0.02, seed=7)
    z = np.zeros(e.n_pair, dtype=int)
    val = e.evaluate_weighted_bler(z, 1)
    assert isinstance(val, float)
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pair_evaluation.py -v`
Expected: FAIL because evaluation still assumes user-level arrays.

**Step 3: Write minimal implementation**

```python
# evaluate_sinr / evaluate_bler / evaluate_weighted_bler
# all use pair-level arrays and pair_priority
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pair_evaluation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pair_evaluation.py
git commit -m "feat: evaluate pair-level BLER with pair priority weighting"
```

### Task 8: 实验脚本与统计输出切换到 pair 语义

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pair_script_output.py`

**Step 1: Write the failing test**

```python
import subprocess
import sys


def test_script_prints_pair_statistics():
    proc = subprocess.run(
        [sys.executable, "sim_script/pd_mmw_template_ap_stats.py", "--cell-size", "2", "--sta-density", "0.01", "--seed", "8", "--mmw-nit", "20"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "wifi_pair_count" in proc.stdout
    assert "ble_pair_count" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pair_script_output.py -v`
Expected: FAIL because script still prints user-level wording.

**Step 3: Write minimal implementation**

```python
# rename stats output to pair semantics
# print n_pair, n_wifi_pair, n_ble_pair
# print per-office pair counts
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pair_script_output.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pair_script_output.py
git commit -m "feat: report office-level wifi and ble pair statistics"
```

### Task 9: 运行办公室 pair 调度实验并保存结果

**Files:**
- Create: `sim_script/output/pd_mmw_template_office_pairs-<timestamp>.txt`

**Step 1: Run experiment**

Run:

```bash
C:\Users\18236\.conda\envs\sig-sdp\python.exe -u sim_script\pd_mmw_template_ap_stats.py --cell-size 4 --sta-density 0.02 --seed 20260308 --mmw-nit 80 > sim_script\output\pd_mmw_template_office_pairs-<timestamp>.txt
```

Expected: 输出 pair 级统计、BLE timing summary、MMW result。

**Step 2: Verify output**

Run:

```bash
rg "n_pair|wifi_pair_count|ble_pair_count|MMW result|BLE timing summary" sim_script/output/pd_mmw_template_office_pairs-<timestamp>.txt
```

Expected: 关键行全部存在。

**Step 3: Commit**

```bash
git add sim_script/output/pd_mmw_template_office_pairs-<timestamp>.txt
git commit -m "chore: add office pair scheduling experiment output"
```
