# BLE Hopping Vectorization And Large-Scale Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 的 CVXPY 目标函数改为向量化构造，进一步完善 [README.md](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/README.md)，并完成一个 9 对 WiFi + 16 对 BLE 且 BLE 使用 `ble_macrocycle_hopping_sdp` backend 的大规模实验验证。

**Architecture:** 这次改动分三层。第一层只优化 `build_sdp_relaxation()` 的目标表达式构造方式，不改变求解问题本身；第二层补一个稳定可复现的大规模 mixed 配置，让主脚本通过 `ble_schedule_backend=macrocycle_hopping_sdp` 接入 BLE-only 后端；第三层把新的运行入口、性能取舍、实验配置和输出说明补进 README，避免后续把 smoke 配置与大规模配置混用。

**Tech Stack:** Python, NumPy, CVXPY, JSON config, pytest, matplotlib

---

### Task 1: 为 BLE SDP 目标向量化写失败测试

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/tests/test_ble_macrocycle_hopping_sdp.py`
- Inspect: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

新增一个专门约束“不要再逐项 Python 累加目标”的测试。不要靠字符串匹配实现细节，测试应验证向量化结果与原有标量构造的数学等价性。

示例：

```python
def test_build_sdp_relaxation_vectorized_objective_matches_upper_triangle_weights():
    Omega = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 2.5],
            [0.0, 2.5, 0.0],
        ]
    )
    problem, Y = MODULE.build_sdp_relaxation(pair_ids=[0, 1], A_k={0: [0], 1: [1, 2]}, Omega=Omega)
    # 关键：目标应只包含上三角非零项，并且不会靠 Python 循环逐项累加
    assert problem.objective is not None
```

再补一个数值测试，直接给定 `Y_value`，比较：

```python
expected = 1.0 * Y01 + 2.5 * Y12
actual = MODULE.evaluate_vectorized_objective_value(Omega, Y_value)
assert actual == expected
```

如果当前代码里没有辅助函数，就先让测试失败。

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py::test_build_sdp_relaxation_vectorized_objective_matches_upper_triangle_weights -v
pytest tests/test_ble_macrocycle_hopping_sdp.py::test_evaluate_vectorized_objective_value_matches_manual_sum -v
```

Expected: FAIL，因为还没有向量化辅助逻辑。

**Step 3: Write minimal implementation**

在 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 中先补最小辅助函数：

```python
def upper_triangle_weight_matrix(Omega: np.ndarray) -> np.ndarray:
    ...


def evaluate_vectorized_objective_value(Omega: np.ndarray, Y_value: np.ndarray) -> float:
    ...
```

实现要求：
- 只保留严格上三角项
- 与原来的 `sum_{i<j} Omega[i,j] * Y[i,j]` 数学等价

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py::test_build_sdp_relaxation_vectorized_objective_matches_upper_triangle_weights -v
pytest tests/test_ble_macrocycle_hopping_sdp.py::test_evaluate_vectorized_objective_value_matches_manual_sum -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "test: lock vectorized ble sdp objective semantics"
```

### Task 2: 将 `build_sdp_relaxation()` 改成向量化目标

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

补一个回归测试，确保新的目标构造不会再触发“目标由太多子表达式组成”的反模式。测试不要依赖 warning 文案本身，而是检查构造方式是否走矩阵化表达。

示例：

```python
def test_build_sdp_relaxation_uses_matrix_expression_for_objective():
    Omega = np.zeros((4, 4), dtype=float)
    Omega[0, 1] = Omega[1, 0] = 1.0
    Omega[1, 2] = Omega[2, 1] = 2.0
    problem, Y = MODULE.build_sdp_relaxation(...)
    assert problem.objective.expr.shape == ()
```

如果需要，可以额外测 `cp.sum(cp.multiply(...))` 路径返回的数值结果。

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected: FAIL 或至少新增测试未通过。

**Step 3: Write minimal implementation**

把 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 里的：

```python
objective_expr = 0
for i in range(A):
    for j in range(i + 1, A):
        if Omega[i, j] != 0:
            objective_expr += Omega[i, j] * Y[i, j]
```

改成矩阵表达式，例如：

```python
weights = np.triu(Omega, k=1)
objective_expr = cp.sum(cp.multiply(weights, Y))
```

要求：
- 只统计上三角一次
- 不改变约束
- 保持 `hard_collision_threshold` 行为不变

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: vectorize ble sdp objective construction"
```

### Task 3: 为 mixed 主脚本补一个可复现的大规模实验配置

**Files:**
- Create: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

新增一个只检查配置格式与最基本运行入口的 smoke test：

```python
def test_large_scale_macrocycle_hopping_config_exists():
    assert LARGE_SCALE_MACRO_BACKEND_CONFIG_PATH.exists()
```

以及一个最小 JSON 结构测试：

```python
def test_large_scale_macrocycle_hopping_config_has_25_pairs():
    payload = json.loads(LARGE_SCALE_MACRO_BACKEND_CONFIG_PATH.read_text())
    assert payload["pair_generation_mode"] == "manual"
    assert len(payload["pair_parameters"]) == 25
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py::test_large_scale_macrocycle_hopping_config_exists -v
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py::test_large_scale_macrocycle_hopping_config_has_25_pairs -v
```

Expected: FAIL，因为配置文件还不存在。

**Step 3: Write minimal implementation**

创建一份手工配置：

- `pair_generation_mode = "manual"`
- 共 `25` 对：
  - `9` 对 `wifi`
  - `16` 对 `ble`
- `ble_schedule_backend = "macrocycle_hopping_sdp"`
- `ble_channel_mode = "per_ce"`
- `cell_size`、`office_id`、`release_time_slot`、`deadline_slot`、`start_time_slot` 自洽
- WiFi 信道只用 `0/5/10`
- BLE 信道只用 `0..36`

第一版目标不是“全都能调度”，而是“稳定可运行、规模可复现”。

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py::test_large_scale_macrocycle_hopping_config_exists -v
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py::test_large_scale_macrocycle_hopping_config_has_25_pairs -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: add 9wifi-16ble macrocycle hopping config"
```

### Task 4: 用主脚本跑 9 WiFi + 16 BLE backend 实验，并导出结果

**Files:**
- Use: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py`
- Use: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
- Output: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/macrocycle_output_9wifi_16ble/`

**Step 1: Add a bounded smoke command**

先不要直接跑无限制长任务，先用可控命令：

```bash
timeout 300s python sim_script/pd_mmw_template_ap_stats.py \
  --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
```

**Step 2: Record expected outputs**

至少检查：

```bash
ls sim_script/macrocycle_output_9wifi_16ble
```

应包含：
- `pair_parameters.csv`
- `schedule_plot_rows.csv`
- `wifi_ble_schedule_overview.png`

**Step 3: Capture key result summary**

从 stdout 或 CSV 里记录：
- `n_wifi_pair`
- `n_ble_pair`
- `scheduled_pair_ids`
- `unscheduled_pair_ids`
- `macrocycle_slots`
- backend 是否打印为 `macrocycle_hopping_sdp`

**Step 4: If timeout occurs, reduce experiment weight**

如果 300 秒超时，不要继续硬跑。按顺序减负：
- 缩小 `max_slots`
- 缩小 BLE pattern 候选集
- 缩小 deadline/window 范围

然后重跑，直到拿到一份真实可运行结果。

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
git commit -m "test: validate 9wifi-16ble macrocycle hopping run"
```

### Task 5: 为 README 补充向量化说明和大规模实验入口

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/README.md`

**Step 1: Write the failing doc test**

如果项目没有文档测试框架，就用人工检查清单；至少保证 README 明确包含：

- `--ble-schedule-backend macrocycle_hopping_sdp`
- `pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
- “目标函数已向量化，主要减少 CVXPY 编译开销，不保证 SDP 求解本身线性提速”

**Step 2: Run doc grep checks to verify failure**

Run:

```bash
rg -n "9wifi_16ble|vectorized|编译开销|ble-schedule-backend" README.md
```

Expected: 还缺至少一部分内容。

**Step 3: Write minimal implementation**

在 [README.md](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/README.md) 中新增三段：

1. `macrocycle_hopping_sdp` backend 的用途与限制  
2. 9 WiFi + 16 BLE 实验配置的运行命令  
3. 向量化只优化 CVXPY 目标构造，不等于整个 SDP 会同等幅度加速

并把原来的 “empty smoke config” 和 “large-scale config” 区分清楚。

**Step 4: Run grep checks to verify pass**

Run:

```bash
rg -n "9wifi_16ble|编译开销|ble-schedule-backend" README.md
```

Expected: PASS，有清晰命中。

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document vectorized ble backend and large-scale run"
```

### Task 6: 运行最终验证

**Files:**
- Use: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Use: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/tests/test_ble_macrocycle_hopping_sdp.py`
- Use: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Use: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Run focused solver tests**

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected: PASS

**Step 2: Run focused main-script tests**

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "macrocycle_hopping or ble_schedule_backend"
```

Expected: PASS

**Step 3: Run the large-scale experiment once**

```bash
timeout 300s python sim_script/pd_mmw_template_ap_stats.py \
  --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
```

Expected:
- command exits 0
- 产生 CSV/PNG
- stdout 明确显示 `n_wifi_pair = 9`、`n_ble_pair = 16`

**Step 4: Record performance note**

在最终说明里记录：
- 向量化前后是否还会看到 `Objective contains too many subexpressions`
- 大规模实验是否仍主要耗在 solver 而非编译

**Step 5: Final commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json sim_script/tests/test_pd_mmw_template_ap_stats_run.py README.md
git commit -m "feat: vectorize ble sdp objective and validate large-scale mixed run"
```
