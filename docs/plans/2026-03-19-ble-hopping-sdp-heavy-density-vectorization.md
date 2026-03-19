# BLE Hopping SDP Heavy-Density Vectorization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `ble_macrocycle_hopping_sdp.py` 的 CVXPY 目标函数改成真正的向量化版本，并在高密度 BLE 调度配置下验证它仍能正确运行且较少出现 Python 侧编译瓶颈。

**Architecture:** 保持现有 lifted SDP 建模和 rounding 逻辑不变，只替换 `build_sdp_relaxation()` 中目标函数的构造方式，避免 Python 双层循环逐项拼接子表达式。主脚本 `sim_script/pd_mmw_template_ap_stats.py` 的接口、JSON 配置格式和 `macrocycle_hopping_sdp` 后端保持兼容，重点验证高密度随机 BLE 配置和混合 WiFi/BLE 配置在改动后都能跑通。

**Tech Stack:** Python 3.10, NumPy, CVXPY, pytest, unittest, conda (`sig-sdp`)

---

### Task 1: 固定当前慢点和目标语义

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Reference: `ble_macrocycle_hopping_sdp.py:415-463`

**Step 1: 写一个失败测试，锁定目标函数构造不再逐项累加**

在 `tests/test_ble_macrocycle_hopping_sdp.py` 增加一个测试，检查 `build_sdp_relaxation()` 的实现不再包含 Python 侧 `objective_expr += ...` 风格的双层累加。不要做字符串脆弱断言，优先通过构造一个小型 `Omega`，确认目标表达式来自矩阵级 `multiply/sum` 而不是大量标量项。

示例思路：

```python
def test_build_sdp_relaxation_uses_vectorized_objective():
    pair_ids = [0, 1]
    A_k = {0: [0, 1], 1: [2, 3]}
    omega = np.array(
        [
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0, 4.0],
            [1.0, 3.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, 0.0],
        ]
    )
    problem, Y = build_sdp_relaxation(pair_ids=pair_ids, A_k=A_k, Omega=omega)
    assert problem.objective.args
```

**Step 2: 运行测试确认当前版本不能满足该约束或需要补充断言**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k vectorized
```

Expected:
- 当前测试失败，或测试过于宽松需要收紧，直到能稳定区分“向量化”与“逐项累加”。

**Step 3: 再写一个等价性测试，确保向量化目标和旧数学目标一致**

在同一文件新增一个小矩阵测试，显式比较：

`sum_{i<j} Omega[i,j] * Y[i,j]`

和你打算采用的向量化表达式在数值上等价。

示例：

```python
def test_vectorized_upper_triangle_objective_matches_reference():
    omega = np.array(...)
    y_value = np.array(...)
    reference = ...
    vectorized = ...
    assert np.isclose(reference, vectorized)
```

**Step 4: 运行这两个测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "vectorized or upper_triangle"
```

Expected:
- 至少一个测试先失败，建立改动护栏。

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "test: lock vectorized ble sdp objective"
```

### Task 2: 把 SDP 目标改成真正的向量化实现

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py:415-463`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: 用上三角掩码替换 Python 双层循环目标**

在 `build_sdp_relaxation()` 中：
- 保留现有 `Y`、PSD 约束、每个 pair 的 one-hot 约束、同 pair 非对角归零约束
- 只替换目标函数部分
- 用 `np.triu(..., k=1)` 或等价掩码抽取 `Omega` 上三角
- 用 `cp.sum(cp.multiply(masked_omega, Y))` 构建矩阵级目标

实现形态应类似：

```python
upper_omega = np.triu(Omega, k=1)
objective_expr = cp.sum(cp.multiply(upper_omega, Y))
problem = cp.Problem(cp.Minimize(objective_expr), constraints)
```

注意：
- 不要把下三角重复计入
- 不要改变数学目标
- 不要顺手改动 rounding、碰撞矩阵或接口

**Step 2: 运行 Task 1 的测试，确认现在通过**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "vectorized or upper_triangle"
```

Expected:
- PASS

**Step 3: 运行 BLE-only 全量测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected:
- 全部通过

**Step 4: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: vectorize ble sdp objective"
```

### Task 3: 验证主脚本接口未被破坏

**Files:**
- Reference: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 跑与 BLE backend 接口直接相关的轻量测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "ble_timing_mode or sample_ble_pair_timing or manual_pair"
```

Expected:
- PASS

**Step 2: 跑 BLE backend 的轻量 run 测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "test_script_accepts_ble_schedule_backend_cli_override or test_script_runs_from_macrocycle_hopping_backend_json_config"
```

Expected:
- PASS

**Step 3: 若测试暴露超时或接口漂移，做最小修复**

允许修改：
- `sim_script/pd_mmw_template_ap_stats.py`
- `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

禁止：
- 为了“加快测试”而绕过 `macrocycle_hopping_sdp` 后端本身
- 修改业务语义，只能做兼容性修复

**Step 4: 重新运行上述测试**

Run 同 Step 1、Step 2

Expected:
- PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "fix: keep ble backend integration stable"
```

### Task 4: 在高密度 BLE 配置下做真实性能验证

**Files:**
- Reference: `sim_script/pd_mmw_template_ap_stats_config.json`
- Optional Modify: `README.md`

**Step 1: 直接运行用户指定的高密度默认配置**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
env PYTHONPATH=$PWD python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json
```

Expected:
- 能进入 `macrocycle_hopping_sdp`
- 不再出现“too many subexpressions”告警，或至少确认该告警已经消失
- 程序最终产出 `sim_script/output` 下的 CSV/PNG

**Step 2: 记录关键运行信息**

把以下信息整理到最终说明中：
- `n_pair`
- `n_wifi_pair`
- `n_ble_pair`
- `macrocycle_slots`
- 是否还出现 CVXPY 子表达式告警
- 调度结果输出目录

**Step 3: 再运行一个受控的大规模混合实例**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
env PYTHONPATH=$PWD python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
```

Expected:
- PASS
- 输出写入 `sim_script/macrocycle_output_9wifi_16ble`

**Step 4: 如仍然慢，先诊断，再最小优化**

允许的优化方向：
- 在 `README.md` 明确解释“向量化减少编译时间，但不改变 SDP 规模”
- 增加调试打印，输出候选状态数 `|A|` 或 BLE pair 数

禁止：
- 把高密度配置偷偷改轻
- 把 backend 从 `macrocycle_hopping_sdp` 改回 `legacy`

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: clarify heavy-density ble sdp runtime behavior"
```

### Task 5: 更新 README 的性能说明

**Files:**
- Modify: `README.md`

**Step 1: 补充“为什么高密度 BLE 仍可能慢”**

在 README 的性能或运行说明章节明确写出：
- 本次向量化只减少 CVXPY 在 Python 侧构造目标函数的编译时间
- 真正决定求解时间的是候选状态总数和 SDP 矩阵大小
- `pair_density=0.7` + `macrocycle_hopping_sdp` 属于重配置

**Step 2: 给出推荐实验入口**

README 中同时列出：
- 高密度随机 BLE：`sim_script/pd_mmw_template_ap_stats_config.json`
- 受控混合实例：`sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
- BLE-only demo：`ble_macrocycle_hopping_sdp.py`

**Step 3: 运行一个最小 smoke 检查 README 中的命令**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python ble_macrocycle_hopping_sdp.py
```

Expected:
- PASS

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: explain heavy-density ble sdp performance"
```

### Task 6: 最终验证

**Files:**
- Reference only

**Step 1: 运行语法检查**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python -m py_compile ble_macrocycle_hopping_sdp.py sim_script/pd_mmw_template_ap_stats.py sim_src/env/env.py
```

Expected:
- PASS

**Step 2: 运行最终回归组合**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "ble_timing_mode or sample_ble_pair_timing or manual_pair"
env PYTHONPATH=$PWD python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
```

Expected:
- 全部通过

**Step 3: 整理结果**

最终说明必须包含：
- 目标函数的具体改法
- 告警是否消失
- 高密度默认配置的实际表现
- 若仍慢，剩余瓶颈是什么

**Step 4: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py README.md
git commit -m "chore: verify vectorized ble sdp on heavy density workloads"
```
