# BLE Hopping SDP Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 的 BLE-only 宏周期 hopping 调度接入现有 WiFi/BLE 主调度链路，同时保留当前 `single` / `per_ce` 原调度方案不变。

**Architecture:** 现有主调度在 [sim_script/pd_mmw_template_ap_stats.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py) 中负责实例生成、MMW 求解、宏周期起始时隙分配、CSV/PNG 导出；现有 BLE hopping SDP 原型在 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 中独立完成 BLE 事件级 hopping 轨迹求解。集成方案应当把后者抽成可复用求解器/适配器，由主脚本在新的配置模式下选择性调用，并把求解结果落回 `env.pair_ble_ce_channels` / schedule plot 输出路径。默认配置仍走原方案，确保兼容现有实验与测试。

**Tech Stack:** Python, NumPy, CVXPY, matplotlib, unittest/pytest, JSON config

---

### Task 1: 明确接入模式并锁定兼容边界

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/docs/plans/2026-03-19-ble-hopping-sdp-integration.md`
- Inspect: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Inspect: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py`
- Inspect: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/env/env.py`

**Step 1: 记录目标模式**

在计划旁注中明确只新增一个可选模式，例如：

```text
ble_schedule_backend = "legacy" | "macrocycle_hopping_sdp"
```

约束：
- `legacy` 保持当前行为
- `macrocycle_hopping_sdp` 仅重写 BLE 的 hopping/事件信道分配，不重写 WiFi 逻辑
- 只在 BLE 存在且 `cvxpy` 可用时启用 SDP 路径

**Step 2: 记录接入点**

锁定三个接入点：
- `resolve_runtime_config()`：读取新配置
- `if config["pair_generation_mode"] == "manual": ...` 后、MMW 之前：准备 BLE hopping 输入
- `build_schedule_plot_rows()` / `build_ble_ce_event_rows()`：复用结果导出图

**Step 3: Commit**

```bash
git add docs/plans/2026-03-19-ble-hopping-sdp-integration.md
git commit -m "docs: define ble hopping sdp integration boundary"
```

### Task 2: 为主脚本增加失败测试，证明默认行为不变

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing tests**

补三类测试。

```python
def test_merge_config_defaults_to_legacy_ble_backend():
    merged = merge_config_with_defaults({})
    assert merged["ble_schedule_backend"] == "legacy"


def test_manual_and_random_configs_accept_macrocycle_hopping_backend():
    merged = merge_config_with_defaults(
        {
            "pair_generation_mode": "manual",
            "pair_parameters": [...],
            "ble_schedule_backend": "macrocycle_hopping_sdp",
        }
    )
    assert merged["ble_schedule_backend"] == "macrocycle_hopping_sdp"


def test_legacy_backend_does_not_require_cvxpy(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
        ...
    )
    assert proc.returncode == 0
```

**Step 2: Run tests to verify failure**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py::test_merge_config_defaults_to_legacy_ble_backend -v
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py::test_legacy_backend_does_not_require_cvxpy -v
```

Expected: FAIL because `ble_schedule_backend` does not exist yet.

**Step 3: Write minimal implementation**

仅先在 [sim_script/pd_mmw_template_ap_stats.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py) 中扩 `DEFAULT_CONFIG` 与 `merge_config_with_defaults()` 的枚举校验：

```python
DEFAULT_CONFIG = {
    ...
    "ble_schedule_backend": "legacy",
}

if merged["ble_schedule_backend"] not in {"legacy", "macrocycle_hopping_sdp"}:
    raise ValueError(...)
```

CLI 暂时先不接。

**Step 4: Run tests to verify pass**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py::test_merge_config_defaults_to_legacy_ble_backend -v
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py::test_legacy_backend_does_not_require_cvxpy -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "test: lock legacy ble backend default"
```

### Task 3: 把 BLE-only SDP 原型抽成主流程可调用的适配器

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

新增一个从主流程视角使用的适配器测试，例如：

```python
def test_solver_can_return_per_pair_selected_state_and_event_blocks():
    result = MODULE.solve_ble_hopping_schedule(
        pair_configs=[...],
        pattern_dict={...},
        num_channels=37,
    )
    assert set(result["selected"].keys()) == {0, 1}
    assert result["blocks"]
```

以及一个把 SDP 解落为每对 CE 信道序列的测试：

```python
def test_selected_schedule_converts_to_ble_ce_channel_map():
    ce_map = MODULE.selected_schedule_to_ce_channels(...)
    assert ce_map[0].tolist() == [1, 6, 11]
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v
```

Expected: FAIL because helper APIs do not exist.

**Step 3: Write minimal implementation**

在 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 中新增纯函数：

```python
def solve_ble_hopping_schedule(...):
    ...
    return {
        "selected": selected,
        "blocks": blocks,
        "overlap_blocks": overlap_blocks,
        "objective_value": float(problem.value),
    }


def selected_schedule_to_ce_channels(
    selected: Dict[int, CandidateState],
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
) -> Dict[int, np.ndarray]:
    ...
```

要求：
- `main()` 继续可运行
- demo 逻辑不删除，只复用新 helper
- 保留原 PNG 输出能力

**Step 4: Run tests to verify pass**

Run:

```bash
python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: expose ble hopping sdp solver helpers"
```

### Task 4: 为现有 `env` 数据结构补一层“外部指定 CE hopping 轨迹”的承接能力

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/env/env.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_per_ce_channel_generation.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_per_ce_occupancy_expansion.py`

**Step 1: Write the failing test**

补一个“外部覆盖 CE hopping 轨迹而非随机生成”的测试：

```python
def test_expand_pair_event_instances_uses_injected_ce_channels():
    e = env(..., ble_channel_mode="per_ce")
    pair_id = ...
    e.pair_ble_ce_channels[pair_id] = np.array([3, 9], dtype=int)
    instances = e.expand_pair_event_instances(pair_id, macrocycle_slots=...)
    assert [inst["channel"] for inst in instances[:2]] == [3, 9]
```

再补一个“不要被 `_assign_ble_ce_channels()` 覆盖”的测试：

```python
def test_manual_ce_channel_map_can_be_preserved():
    ...
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_ble_per_ce_channel_generation.py sim_script/tests/test_ble_per_ce_occupancy_expansion.py -v
```

Expected: FAIL if当前初始化/重采样路径覆盖了外部注入。

**Step 3: Write minimal implementation**

在 [sim_src/env/env.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/env/env.py) 中新增最小 helper：

```python
def set_ble_ce_channel_map(self, channel_map: dict[int, np.ndarray]) -> None:
    ...
```

要求：
- 只接受 BLE pair
- 长度与 event count 对齐
- 不修改 legacy `single` 模式

**Step 4: Run test to verify pass**

Run:

```bash
pytest sim_script/tests/test_ble_per_ce_channel_generation.py sim_script/tests/test_ble_per_ce_occupancy_expansion.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_ble_per_ce_channel_generation.py sim_script/tests/test_ble_per_ce_occupancy_expansion.py
git commit -m "feat: support injected ble ce hopping map"
```

### Task 5: 在主脚本中接入 SDP backend，但只影响 BLE hopping 分配

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

新增一个逻辑层单测，mock 掉 SDP helper，验证主脚本会在配置开启时调用它：

```python
def test_macrocycle_hopping_backend_populates_env_ce_channels(monkeypatch):
    e = env(..., ble_channel_mode="per_ce")
    called = {}

    def fake_solver(...):
        called["ok"] = True
        return {"ce_channel_map": {ble_pair_id: np.array([1, 5], dtype=int)}}

    monkeypatch.setattr(module, "solve_ble_hopping_for_env", fake_solver)
    apply_ble_schedule_backend(e, config={"ble_schedule_backend": "macrocycle_hopping_sdp"})
    assert called["ok"] is True
    assert e.pair_ble_ce_channels[ble_pair_id].tolist() == [1, 5]
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py::test_macrocycle_hopping_backend_populates_env_ce_channels -v
```

Expected: FAIL because backend helper/apply function does not exist.

**Step 3: Write minimal implementation**

在 [sim_script/pd_mmw_template_ap_stats.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py) 中新增：

```python
def solve_ble_hopping_for_env(e: env, schedule_start_slots=None, ...):
    ...


def apply_ble_schedule_backend(e: env, config: dict) -> None:
    if config["ble_schedule_backend"] == "legacy":
        return
    ...
    e.set_ble_ce_channel_map(result["ce_channel_map"])
```

实现原则：
- `legacy` 立即返回
- `macrocycle_hopping_sdp` 自动把 BLE 视为 `per_ce` 事件信道模式
- 只修改 BLE pair 的 `pair_ble_ce_channels`
- 不改 WiFi 相关输入和 MMW 求解器接口

**Step 4: Run test to verify pass**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py::test_macrocycle_hopping_backend_populates_env_ce_channels -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add ble hopping backend integration hook"
```

### Task 6: 把 `PairConfig/HoppingPattern` 输入从 `env` 派生出来

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

补一个从 `env` 构造 BLE-only SDP 输入的测试：

```python
def test_build_ble_hopping_inputs_from_env():
    e = env(..., ble_channel_mode="per_ce")
    pair_configs, pattern_dict, num_channels = build_ble_hopping_inputs_from_env(e)
    assert all(cfg.pair_id in pattern_dict for cfg in pair_configs)
    assert num_channels == 37
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py::test_build_ble_hopping_inputs_from_env -v
```

Expected: FAIL because builder does not exist.

**Step 3: Write minimal implementation**

实现一个最小、显式的转换器：

```python
def build_ble_hopping_inputs_from_env(e: env):
    pair_configs = [...]
    pattern_dict = {
        pair_id: [
            HoppingPattern(pattern_id=0, start_channel=int(e.pair_channel[pair_id]), hop_increment=5),
            ...
        ]
    }
    return pair_configs, pattern_dict, int(e.ble_channel_count)
```

注意：
- 第一版先用确定性的候选 pattern 集，不追求完整 BLE spec
- `release_time` / `deadline` 由 pair 的 release/deadline slot 派生
- `connect_interval` / `event_duration` / `num_events` 对齐 `CI/CE/macrocycle`

**Step 4: Run test to verify pass**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py::test_build_ble_hopping_inputs_from_env -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: derive ble hopping sdp inputs from env"
```

### Task 7: 让配置文件和 CLI 能选择新 backend

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

```python
def test_script_runs_with_macrocycle_hopping_backend_config(tmp_path):
    config = {..., "ble_schedule_backend": "macrocycle_hopping_sdp"}
    ...
    assert proc.returncode == 0
    assert "ble_schedule_backend = macrocycle_hopping_sdp" in proc.stdout
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py::test_script_runs_with_macrocycle_hopping_backend_config -v
```

Expected: FAIL because CLI/config do not expose the backend yet.

**Step 3: Write minimal implementation**

在主脚本中添加：

```python
parser.add_argument(
    "--ble-schedule-backend",
    choices=["legacy", "macrocycle_hopping_sdp"],
    ...
)
```

并在两个 JSON 样例中新增中文说明：

```json
"_comment_ble_schedule_backend": "BLE 调度后端：legacy 保持原方案，macrocycle_hopping_sdp 使用 BLE-only hopping SDP 生成 CE 跳频轨迹",
"ble_schedule_backend": "legacy"
```

**Step 4: Run test to verify pass**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py::test_script_runs_with_macrocycle_hopping_backend_config -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: expose ble schedule backend in config"
```

### Task 8: 保证导出的事件图同时兼容原方案和 SDP 方案

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_per_ce_export_and_plot.py`

**Step 1: Write the failing test**

新增一个“通过 backend 注入的 CE channel map 也能输出 event-level rows”的测试：

```python
def test_build_schedule_plot_rows_emits_backend_generated_ble_events():
    e = env(..., ble_channel_mode="per_ce")
    e.set_ble_ce_channel_map({pair_id: np.array([2, 20], dtype=int)})
    ...
    rows = build_schedule_plot_rows(pair_rows, {}, e=e)
    assert any(row["channel"] == 2 for row in rows if row["radio"] == "ble")
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_ble_per_ce_export_and_plot.py -v
```

Expected: FAIL if plot/export path still assumes only legacy `per_ce` generation path.

**Step 3: Write minimal implementation**

最小实现应当只复用已有 `expand_pair_event_instances()`；如 Task 4/5 正确完成，通常这里只需要补一两个 guard：

```python
if e is not None and row["radio"] == "ble" and e.ble_channel_mode == "per_ce":
    ...
```

并确保 `macrocycle_slots`、`schedule_slot` 为空时不崩。

**Step 4: Run test to verify pass**

Run:

```bash
pytest sim_script/tests/test_ble_per_ce_export_and_plot.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_ble_per_ce_export_and_plot.py
git commit -m "test: cover schedule plot export for sdp ble hopping"
```

### Task 9: 增加端到端 smoke test，证明“新后端可选且旧后端不回归”

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing tests**

补两个 smoke test：

```python
def test_manual_config_runs_with_legacy_backend():
    ...


def test_manual_config_runs_with_macrocycle_hopping_backend():
    ...
    assert (output_dir / "schedule_plot_rows.csv").exists()
    assert "ble_adv_idle" in (output_dir / "schedule_plot_rows.csv").read_text()
```

**Step 2: Run test to verify failure**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -v
```

Expected: 至少新 backend 的 smoke test FAIL。

**Step 3: Write minimal implementation**

修复真实集成问题，优先处理：
- `cvxpy` 缺失时报错信息
- 没有 BLE pair 时跳过 SDP
- 生成的 `ce_channel_map` 长度与 event count 对齐
- legacy/backend 两条路径都能写出相同格式 CSV

**Step 4: Run test to verify pass**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/pd_mmw_template_ap_stats.py ble_macrocycle_hopping_sdp.py sim_src/env/env.py
git commit -m "feat: integrate macrocycle ble hopping backend end to end"
```

### Task 10: 运行完整验证并更新示例说明

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp_config.json`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`

**Step 1: Run focused tests**

```bash
python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v
pytest sim_script/tests/test_ble_per_ce_channel_generation.py -v
pytest sim_script/tests/test_ble_per_ce_occupancy_expansion.py -v
pytest sim_script/tests/test_ble_per_ce_export_and_plot.py -v
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -v
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -v
```

Expected: PASS

**Step 2: Run end-to-end examples**

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python ble_macrocycle_hopping_sdp.py
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json
```

Expected:
- 原 BLE-only demo 正常出图
- legacy backend 正常出图
- `macrocycle_hopping_sdp` backend 正常出图（在样例配置中需要至少再给一份开启该 backend 的 config，或手动临时覆盖）

**Step 3: Update example configs**

在 JSON 里保留默认：

```json
"ble_schedule_backend": "legacy"
```

如果需要演示，再额外增加一个样例配置：

```json
"ble_schedule_backend": "macrocycle_hopping_sdp",
"ble_channel_mode": "per_ce"
```

**Step 4: Final commit**

```bash
git add ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp_config.json sim_src/env/env.py sim_script/pd_mmw_template_ap_stats.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json sim_script/tests
git commit -m "feat: integrate ble macrocycle hopping sdp into main scheduler"
```
