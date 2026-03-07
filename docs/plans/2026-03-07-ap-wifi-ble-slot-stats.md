# AP WiFi/BLE 用户与时隙统计输出 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `pd_mmw_template` 实验流程中新增“按 AP 统计 WiFi/BLE 用户数量与各自占用时隙数”的输出，并通过新脚本完成一次可复现实验运行。

**Architecture:** 在 `sim_script/pd_mmw_template_ap_stats.py` 中集中实现统计与打印逻辑，复用现有 `env + binary_search_relaxation + mmw` 主流程，不改动 MMW 主体数学。统计来源使用当前环境下用户关联 AP（按接收功率最大关联）与 `z_vec` 时隙分配结果，最终输出结构化表格（CSV 风格行）。

**Tech Stack:** Python 3, NumPy, 项目内 `sim_src` 算法模块（`binary_search_relaxation`, `mmw`, `env`）

---

### Task 1: 修复并锁定新脚本语法与输出格式

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_smoke.py`（新建）

**Step 1: Write the failing test**

```python
import pathlib
import subprocess
import sys


def test_ap_stats_script_help_or_import_smoke():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    assert script.exists()
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_smoke.py -v`
Expected: FAIL（当前版本 `print_ap_stats` 的 f-string 拼接可能触发语法错误）

**Step 3: Write minimal implementation**

```python
def print_ap_stats(rows):
    print("=== Per-AP WiFi/BLE User & Slot Statistics ===")
    print("ap_id,wifi_user_count,ble_user_count,wifi_slots_used,ble_slots_used")
    for r in rows:
        print(
            f"{r['ap_id']},{r['wifi_user_count']},{r['ble_user_count']},"
            f"{r['wifi_slots_used']},{r['ble_slots_used']}"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_smoke.py
git commit -m "fix: repair ap stats script syntax and add smoke compile test"
```

### Task 2: 增强按 AP 统计输出（中文标注 + 指标完整性）

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`（新建）

**Step 1: Write the failing test**

```python
import numpy as np

from sim_script.pd_mmw_template_ap_stats import _aggregate_ap_stats_from_arrays


def test_aggregate_ap_stats_from_arrays():
    # 4 个用户，2 个 AP
    asso = np.array([0, 0, 1, 1], dtype=int)
    radio = np.array([0, 1, 0, 1], dtype=int)  # 0=WiFi, 1=BLE
    z_vec = np.array([1, 2, 1, 3], dtype=int)
    rows = _aggregate_ap_stats_from_arrays(asso, radio, z_vec, n_ap=2, wifi_id=0, ble_id=1)
    assert rows[0]["wifi_user_count"] == 1
    assert rows[0]["ble_user_count"] == 1
    assert rows[0]["wifi_slots_used"] == 1
    assert rows[0]["ble_slots_used"] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -v`
Expected: FAIL（函数未实现或行为不符）

**Step 3: Write minimal implementation**

```python
def _aggregate_ap_stats_from_arrays(asso, radio, z_vec, n_ap, wifi_id, ble_id):
    rows = []
    for ap in range(n_ap):
        idx = np.where(asso == ap)[0]
        wifi_idx = idx[radio[idx] == wifi_id]
        ble_idx = idx[radio[idx] == ble_id]
        rows.append(
            {
                "ap_id": int(ap),
                "wifi_user_count": int(wifi_idx.size),
                "ble_user_count": int(ble_idx.size),
                "wifi_slots_used": int(np.unique(z_vec[wifi_idx]).size) if wifi_idx.size else 0,
                "ble_slots_used": int(np.unique(z_vec[ble_idx]).size) if ble_idx.size else 0,
            }
        )
    return rows
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add deterministic per-ap wifi/ble user-slot aggregation"
```

### Task 3: 跑实验并保存结果日志

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`（可选：增加 `--seed` 参数便于复现）
- Create: `sim_script/output/pd_mmw_template_ap_stats-<timestamp>.txt`（运行日志）

**Step 1: Write the failing test**

```python
import subprocess
import sys


def test_script_runs_and_prints_ap_table_header():
    proc = subprocess.run(
        [sys.executable, "sim_script/pd_mmw_template_ap_stats.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "ap_id,wifi_user_count,ble_user_count,wifi_slots_used,ble_slots_used" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -v`
Expected: FAIL（在脚本尚未稳定前可能返回码非 0 或缺失表头）

**Step 3: Write minimal implementation**

```python
# 1) 脚本主流程最后调用 print_ap_stats(ap_rows)
# 2) 增加固定 seed 参数（可选）以获得可复现输出
# 3) 保证 stdout 含统计表头
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "test: validate ap stats experiment script execution and output header"
```

### Task 4: 实验执行与结果回传

**Files:**
- Create: `sim_script/output/pd_mmw_template_ap_stats-<timestamp>.txt`

**Step 1: Run experiment command**

Run:

```bash
C:\Users\18236\.conda\envs\sig-sdp\python.exe sim_script\pd_mmw_template_ap_stats.py > sim_script\output\pd_mmw_template_ap_stats-<timestamp>.txt
```

Expected: 成功输出 `MMW result`、`BLE timing summary`、`Per-AP WiFi/BLE User & Slot Statistics` 表格。

**Step 2: Verify output content**

Run:

```bash
rg "MMW result|BLE timing summary|ap_id,wifi_user_count" sim_script/output/pd_mmw_template_ap_stats-<timestamp>.txt
```

Expected: 三类关键行均能检索到。

**Step 3: Commit**

```bash
git add sim_script/output/pd_mmw_template_ap_stats-<timestamp>.txt
git commit -m "chore: add ap-level wifi/ble slot statistics experiment output"
```
