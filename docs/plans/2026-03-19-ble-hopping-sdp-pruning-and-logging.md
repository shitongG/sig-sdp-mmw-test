# BLE Hopping SDP Pruning And Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在高密度 BLE 场景下为 `macrocycle_hopping_sdp` 增加候选状态剪枝和诊断日志，降低候选状态总数 `|A|`，并让运行时能直接看到每对 BLE 的可行 offset 数与总候选状态数。

**Architecture:** 保持现有 BLE-only SDP 建模、碰撞矩阵、rounding 和主脚本接口不变，只在候选状态生成前增加轻量剪枝策略，并在主脚本接入路径上打印可解释的规模诊断信息。剪枝应优先选择“不改变现有配置格式、对主调度语义侵入最小”的方式，例如限制每对 BLE 的 offset 候选上限、按时间窗或步长做可复现子采样，并保留原有默认行为可配置关闭。

**Tech Stack:** Python 3.10, NumPy, CVXPY, pytest, unittest, conda (`sig-sdp`)

---

### Task 1: 为候选状态剪枝定义测试护栏

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Reference: `ble_macrocycle_hopping_sdp.py:67-156`

**Step 1: 写一个失败测试，要求候选 offset 支持上限剪枝**

在 `tests/test_ble_macrocycle_hopping_sdp.py` 中新增一个测试，针对单个 `PairConfig` 产生较宽时间窗时，调用未来要新增的剪枝接口，验证：
- 原始 `compute_feasible_offsets(cfg)` 能返回完整 offset 列表
- 剪枝后 offset 数量被限制在指定上限内
- 剪枝结果仍保持有序、可复现、且首尾保留

示例测试：

```python
def test_prune_feasible_offsets_limits_count_and_keeps_order():
    cfg = MODULE.PairConfig(
        pair_id=0,
        release_time=0,
        deadline=30,
        connect_interval=2,
        event_duration=1,
        num_events=4,
    )
    offsets = MODULE.compute_feasible_offsets(cfg)
    pruned = MODULE.prune_feasible_offsets(offsets, max_offsets=5)
    assert len(offsets) > 5
    assert len(pruned) == 5
    assert pruned == sorted(pruned)
    assert pruned[0] == offsets[0]
    assert pruned[-1] == offsets[-1]
```

**Step 2: 运行测试确认当前失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k prune_feasible_offsets
```

Expected:
- FAIL，提示 `prune_feasible_offsets` 尚不存在或行为不满足要求

**Step 3: 写一个失败测试，要求 `build_candidate_states()` 能接受剪枝参数**

新增测试，验证在传入 `max_offsets_per_pair=3` 时，同一 pair 的候选状态数确实下降，并且 `A_k`、`states` 仍然一致。

示例：

```python
def test_build_candidate_states_respects_max_offsets_per_pair():
    pair_configs, _, pattern_dict, _, _ = MODULE.build_demo_instance()
    states_full, _, A_k_full = MODULE.build_candidate_states(pair_configs, pattern_dict)
    states_pruned, _, A_k_pruned = MODULE.build_candidate_states(
        pair_configs, pattern_dict, max_offsets_per_pair=2
    )
    assert len(states_pruned) < len(states_full)
    for pair_id in A_k_pruned:
        assert len(A_k_pruned[pair_id]) <= len(pattern_dict[pair_id]) * 2
```

**Step 4: 运行测试确认失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "prune_feasible_offsets or max_offsets_per_pair"
```

Expected:
- FAIL

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "test: define ble state pruning behavior"
```

### Task 2: 实现可复现的 offset 剪枝

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py:67-156`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: 新增最小剪枝函数**

在 `ble_macrocycle_hopping_sdp.py` 中新增：

```python
def prune_feasible_offsets(offsets: List[int], max_offsets: Optional[int]) -> List[int]:
    ...
```

要求：
- `max_offsets is None` 或 `<= 0` 时返回原列表
- 若 `len(offsets) <= max_offsets`，返回原列表
- 若需要剪枝，采用可复现子采样：
  - 保留第一个和最后一个 offset
  - 中间点按等间距索引采样
- 返回严格递增的整数列表

**Step 2: 让 `build_candidate_states()` 支持 `max_offsets_per_pair`**

修改函数签名：

```python
def build_candidate_states(
    pair_configs,
    pattern_dict,
    max_offsets_per_pair: Optional[int] = None,
):
```

在每个 pair 上：
- 先算完整 `feasible_offsets`
- 再调用 `prune_feasible_offsets(...)`
- 再枚举 `(offset, pattern)`

不要改动未传该参数时的旧行为。

**Step 3: 运行 Task 1 的测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "prune_feasible_offsets or max_offsets_per_pair"
```

Expected:
- PASS

**Step 4: 跑 BLE-only 全量测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: add reproducible ble offset pruning"
```

### Task 3: 把剪枝参数接入主脚本和配置

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:22-145`
- Modify: `sim_script/pd_mmw_template_ap_stats.py:160-238`
- Modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: 写一个失败测试，要求主配置支持剪枝参数**

在 `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py` 增加测试，验证默认配置合并后包含：
- `ble_max_offsets_per_pair`
- `ble_log_candidate_summary`

并且它们能通过 `resolve_runtime_config()` 正确解析。

示例：

```python
def test_resolve_runtime_config_accepts_ble_pruning_keys():
    config = MODULE.merge_config_with_defaults(
        {"ble_max_offsets_per_pair": 4, "ble_log_candidate_summary": True}
    )
    assert config["ble_max_offsets_per_pair"] == 4
    assert config["ble_log_candidate_summary"] is True
```

**Step 2: 运行该测试确认失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k pruning_keys
```

Expected:
- FAIL

**Step 3: 在主脚本默认配置和 JSON 配置中增加参数**

为主脚本增加：
- `ble_max_offsets_per_pair`
- `ble_log_candidate_summary`

要求：
- CLI 可选覆盖
- 默认值保守，例如：
  - `ble_max_offsets_per_pair = None` 或较宽上限
  - `ble_log_candidate_summary = True`
- 高密度默认配置里显式设置合理剪枝值，例如 `8` 或 `12`

**Step 4: 在 `build_ble_hopping_inputs_from_env()` / `solve_ble_hopping_for_env()` 接入剪枝参数**

调用 `build_candidate_states(...)` 时传入 `max_offsets_per_pair=config[...]`。

**Step 5: 跑配置解析相关测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k pruning_keys
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_config.json sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: wire ble offset pruning into runtime config"
```

### Task 4: 增加候选状态规模日志

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py:131-156`
- Modify: `sim_script/pd_mmw_template_ap_stats.py:160-238`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 写一个失败测试，要求输出每对 BLE 的 offset 数和总状态数**

在 `tests/test_ble_macrocycle_hopping_sdp.py` 增加一个测试，调用一个新的摘要函数或打印函数，检查输出里包含：
- `|A|` 或 `state_count`
- 每个 pair 的 `offset_count`
- 每个 pair 的 `pattern_count`

建议新增纯函数，避免直接依赖 `print()`：

```python
def summarize_candidate_space(...):
    return {...}
```

示例测试：

```python
def test_summarize_candidate_space_reports_offsets_and_state_count():
    pair_configs, _, pattern_dict, _, _ = MODULE.build_demo_instance()
    states, _, A_k = MODULE.build_candidate_states(pair_configs, pattern_dict, max_offsets_per_pair=2)
    summary = MODULE.summarize_candidate_space(pair_configs, pattern_dict, A_k, max_offsets_per_pair=2)
    assert summary["state_count"] == len(states)
    assert summary["pairs"][0]["offset_count"] == 2
```

**Step 2: 运行测试确认失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k summarize_candidate_space
```

Expected:
- FAIL

**Step 3: 实现摘要函数和日志打印**

在 `ble_macrocycle_hopping_sdp.py` 中新增：
- `summarize_candidate_space(...)`
- 可选 `print_candidate_space_summary(...)`

输出至少包含：
- `pair_id`
- `offset_count`
- `pattern_count`
- `state_count`
- `max_offsets_per_pair`
- 全局 `total_pairs`
- 全局 `state_count`

在 `sim_script/pd_mmw_template_ap_stats.py` 的 `solve_ble_hopping_for_env()` 中：
- 当 `ble_log_candidate_summary` 为真时，打印该摘要

**Step 4: 跑 BLE-only 测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k summarize_candidate_space
```

Expected:
- PASS

**Step 5: 写一个轻量 run 测试，要求主脚本打印候选空间摘要**

在 `sim_script/tests/test_pd_mmw_template_ap_stats_run.py` 中新增针对小配置的测试，断言 stdout 中出现：
- `BLE candidate summary`
- `state_count`
- `offset_count`

不要直接跑高密度默认配置，避免测试超时；优先使用小型 `macrocycle_hopping_config.json`。

**Step 6: 运行该测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k candidate_summary
```

Expected:
- PASS

**Step 7: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py sim_script/pd_mmw_template_ap_stats.py tests/test_ble_macrocycle_hopping_sdp.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: log ble candidate state summary"
```

### Task 5: 在高密度配置上验证剪枝效果

**Files:**
- Reference: `sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `README.md`

**Step 1: 跑高密度默认配置**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
env PYTHONPATH=$PWD python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json
```

Expected:
- stdout 直接打印 `BLE candidate summary`
- 可以看到每个 pair 的 `offset_count`
- 可以看到全局 `state_count`
- 与改动前相比，总状态数下降

**Step 2: 记录性能和规模**

把以下信息记到最终说明：
- `n_pair`
- `n_ble_pair`
- `state_count`
- 单个 pair 的最大 `offset_count`
- 运行是否仍超时
- 如果仍超时，卡在“编译”还是“求解”

**Step 3: 再跑 9+16 受控配置**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
env PYTHONPATH=$PWD python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
```

Expected:
- PASS
- 候选空间摘要存在
- 受控 mixed 配置仍兼容

**Step 4: 如果高密度配置仍慢，做最小诊断补充**

允许修改：
- `README.md`
- 额外打印运行阶段开始/结束标记

禁止：
- 把高密度默认配置偷偷改轻
- 取消 `macrocycle_hopping_sdp`

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: explain ble pruning and candidate summary logs"
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

**Step 2: 运行最终回归**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "ble_timing_mode or sample_ble_pair_timing or manual_pair or pruning_keys"
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k candidate_summary
env PYTHONPATH=$PWD python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
```

Expected:
- 全部通过

**Step 3: 整理结论**

最终说明必须包含：
- 采用了什么剪枝策略
- 默认高密度配置的 `state_count` 是否下降
- 哪些 pair 的 `offset_count` 最大
- 若仍慢，剩余瓶颈是什么

**Step 4: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py sim_script/pd_mmw_template_ap_stats.py README.md tests/test_ble_macrocycle_hopping_sdp.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "chore: verify ble pruning on heavy-density workloads"
```
