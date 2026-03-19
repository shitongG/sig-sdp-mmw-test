# Manual Pair JSON Input Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 扩展 `sim_script/pd_mmw_template_ap_stats.py` 的 JSON 配置，使其既支持手工输入逐 pair 参数和时间窗口，也在未提供逐 pair 参数时继续沿用现有随机生成功能。

**Architecture:** 在现有 `--config` 入口上增加第二层配置能力：`pair_generation_mode` 用来区分 `random` 和 `manual`。`manual` 模式下从 JSON 读取 `pair_parameters`，并把这些字段注入 `env` 的 pair 级数组；`random` 模式下完全保留当前 `env(...)` 的随机生成逻辑。为避免破坏现有代码，优先新增“配置解析 + 手工参数覆盖”函数，再把 `env` 输出表扩成包含 `release_time` / `deadline_slot` 字段。

**Tech Stack:** Python 3、`argparse`、`json`、`pathlib`、`numpy`、现有 `env` 类、`pytest` / `subprocess`

---

### Task 1: 定义手工 pair JSON 结构与默认样例

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Create: `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 在运行测试里写一个默认手工 JSON 文件存在性的失败测试**

```python
def test_manual_pair_json_config_exists():
    config = REPO_ROOT / "sim_script" / "pd_mmw_template_ap_stats_manual_pairs_config.json"
    assert config.exists()
```

**Step 2: 新建手工 pair 样例 JSON，字段明确到逐 pair**

```json
{
  "_comment_pair_generation_mode": "pair 生成模式：manual 表示直接读取 pair_parameters，random 表示沿用随机生成",
  "pair_generation_mode": "manual",
  "_comment_pair_parameters": "逐 pair 输入参数；radio 为 wifi 或 ble",
  "pair_parameters": [
    {
      "pair_id": 0,
      "office_id": 0,
      "radio": "ble",
      "channel": 8,
      "priority": 1.0,
      "release_time_slot": 0,
      "deadline_slot": 63,
      "start_time_slot": 0,
      "ble_anchor_slot": 12,
      "ble_ci_slots": 64,
      "ble_ce_slots": 5
    },
    {
      "pair_id": 1,
      "office_id": 0,
      "radio": "wifi",
      "channel": 0,
      "priority": 1.0,
      "release_time_slot": 0,
      "deadline_slot": 63,
      "start_time_slot": 0,
      "wifi_anchor_slot": 6,
      "wifi_period_slots": 32,
      "wifi_tx_slots": 5
    }
  ]
}
```

**Step 3: 运行测试确认当前先失败**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py::test_manual_pair_json_config_exists -q`
Expected: FAIL，因为样例文件尚未创建

**Step 4: 创建/更新样例 JSON 文件**

Run: `python -m json.tool sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
Expected: JSON 格式合法

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "docs: add manual pair json config schema"
```

### Task 2: 为手工 pair 参数解析写失败测试

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: 写手工 pair 配置可被加载的失败测试**

```python
def test_load_json_config_keeps_pair_generation_mode_and_pair_parameters(tmp_path: Path):
    config_path = tmp_path / "manual.json"
    config_path.write_text(
        json.dumps(
            {
                "pair_generation_mode": "manual",
                "pair_parameters": [
                    {
                        "pair_id": 0,
                        "office_id": 0,
                        "radio": "ble",
                        "channel": 8,
                        "priority": 1.0,
                        "release_time_slot": 0,
                        "deadline_slot": 63,
                        "start_time_slot": 0,
                        "ble_anchor_slot": 12,
                        "ble_ci_slots": 64,
                        "ble_ce_slots": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_json_config(config_path)
    assert loaded["pair_generation_mode"] == "manual"
    assert loaded["pair_parameters"][0]["deadline_slot"] == 63
```

**Step 2: 写配置校验失败测试**

```python
def test_merge_config_with_defaults_rejects_manual_mode_without_pair_parameters():
    with pytest.raises(ValueError, match="pair_parameters"):
        merge_config_with_defaults({"pair_generation_mode": "manual"})


def test_merge_config_with_defaults_rejects_unknown_radio_value():
    with pytest.raises(ValueError, match="radio"):
        merge_config_with_defaults(
            {
                "pair_generation_mode": "manual",
                "pair_parameters": [
                    {"pair_id": 0, "radio": "zigbee"}
                ],
            }
        )
```

**Step 3: 写逐 pair 字段补全/默认行为失败测试**

```python
def test_merge_config_with_defaults_sets_random_mode_when_pair_parameters_absent():
    merged = merge_config_with_defaults({})
    assert merged["pair_generation_mode"] == "random"
    assert merged["pair_parameters"] is None
```

**Step 4: 运行测试确认当前失败**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q`
Expected: FAIL，因为 `pair_generation_mode` / `pair_parameters` 还未接入

**Step 5: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "test: cover manual pair json config parsing"
```

### Task 3: 扩展配置解析层支持手工 pair 模式

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: 扩展 `DEFAULT_CONFIG`**

```python
DEFAULT_CONFIG = {
    ...
    "pair_generation_mode": "random",
    "pair_parameters": None,
}
```

**Step 2: 在 `merge_config_with_defaults` 里增加手工模式校验**

```python
def _validate_pair_parameters(pair_parameters):
    for row in pair_parameters:
        if row["radio"] not in {"wifi", "ble"}:
            raise ValueError("pair_parameters radio must be 'wifi' or 'ble'.")
        if "release_time_slot" not in row or "deadline_slot" not in row:
            raise ValueError("pair_parameters must include release_time_slot and deadline_slot.")
```

```python
if merged["pair_generation_mode"] not in {"random", "manual"}:
    raise ValueError("pair_generation_mode must be 'random' or 'manual'.")
if merged["pair_generation_mode"] == "manual":
    if not merged["pair_parameters"]:
        raise ValueError("pair_parameters must be provided when pair_generation_mode='manual'.")
    _validate_pair_parameters(merged["pair_parameters"])
else:
    merged["pair_parameters"] = None
```

**Step 3: 运行逻辑测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q`
Expected: 新增配置解析测试通过

**Step 4: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add manual pair mode to ap stats config"
```

### Task 4: 在 env 层补齐 release/deadline 字段容器

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: 在 `env.__init__` 中增加 pair 级 release/deadline 数组**

```python
self.pair_release_time_slot = None
self.pair_deadline_slot = None
```

**Step 2: 在 pair 数组初始化处补零数组**

```python
self.pair_release_time_slot = np.zeros(self.n_pair, dtype=int)
self.pair_deadline_slot = np.zeros(self.n_pair, dtype=int)
```

**Step 3: 保持 device/user 视图同步**

```python
self.device_release_time_slot = self.pair_release_time_slot
self.device_deadline_slot = self.pair_deadline_slot
self.user_release_time_slot = self.pair_release_time_slot
self.user_deadline_slot = self.pair_deadline_slot
```

**Step 4: 在随机生成逻辑里给 release/deadline 一个稳定默认值**

```python
self.pair_release_time_slot[k] = int(self.pair_start_time_slot[k])
period_slots = int(self.get_pair_period_slots()[k]) if ... else ...
self.pair_deadline_slot[k] = int(self.pair_release_time_slot[k] + max(period_slots - 1, 0))
```

**Step 5: 运行相关逻辑测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q`
Expected: env 新字段不破坏现有逻辑

**Step 6: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add release and deadline arrays to env"
```

### Task 5: 实现“手工 pair 参数覆盖 env”的最小注入函数

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: 写失败测试，验证手工 pair 会覆盖 env 数组**

```python
def test_apply_manual_pair_parameters_overrides_env_arrays():
    class DummyEnv:
        RADIO_WIFI = 0
        RADIO_BLE = 1
        n_pair = 2
        pair_office_id = np.zeros(2, dtype=int)
        pair_radio_type = np.zeros(2, dtype=int)
        pair_channel = np.zeros(2, dtype=int)
        pair_priority = np.zeros(2, dtype=float)
        pair_release_time_slot = np.zeros(2, dtype=int)
        pair_deadline_slot = np.zeros(2, dtype=int)
        pair_start_time_slot = np.zeros(2, dtype=int)
        pair_wifi_anchor_slot = np.zeros(2, dtype=int)
        pair_wifi_period_slots = np.zeros(2, dtype=int)
        pair_wifi_tx_slots = np.zeros(2, dtype=int)
        pair_ble_anchor_slot = np.zeros(2, dtype=int)
        pair_ble_ci_slots = np.zeros(2, dtype=int)
        pair_ble_ce_slots = np.zeros(2, dtype=int)
        pair_ble_ce_feasible = np.ones(2, dtype=bool)
    e = DummyEnv()
    apply_manual_pair_parameters(e, [...])
    assert e.pair_deadline_slot[0] == 63
    assert e.pair_radio_type[1] == e.RADIO_WIFI
```

**Step 2: 实现最小覆盖函数**

```python
def apply_manual_pair_parameters(e: env, pair_parameters):
    if len(pair_parameters) != int(e.n_pair):
        raise ValueError("manual pair count must match env.n_pair")
    for row in pair_parameters:
        pair_id = int(row["pair_id"])
        e.pair_office_id[pair_id] = int(row["office_id"])
        e.pair_radio_type[pair_id] = e.RADIO_BLE if row["radio"] == "ble" else e.RADIO_WIFI
        e.pair_channel[pair_id] = int(row["channel"])
        e.pair_priority[pair_id] = float(row["priority"])
        e.pair_release_time_slot[pair_id] = int(row["release_time_slot"])
        e.pair_deadline_slot[pair_id] = int(row["deadline_slot"])
        e.pair_start_time_slot[pair_id] = int(row["start_time_slot"])
        ...
```

**Step 3: 对 WiFi / BLE 分支分别填充对应字段**

```python
if row["radio"] == "wifi":
    e.pair_wifi_anchor_slot[pair_id] = int(row["wifi_anchor_slot"])
    e.pair_wifi_period_slots[pair_id] = int(row["wifi_period_slots"])
    e.pair_wifi_tx_slots[pair_id] = int(row["wifi_tx_slots"])
else:
    e.pair_ble_anchor_slot[pair_id] = int(row["ble_anchor_slot"])
    e.pair_ble_ci_slots[pair_id] = int(row["ble_ci_slots"])
    e.pair_ble_ce_slots[pair_id] = int(row["ble_ce_slots"])
    e.pair_ble_ce_feasible[pair_id] = bool(row.get("ble_ce_feasible", True))
```

**Step 4: 运行逻辑测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q`
Expected: 手工覆盖测试通过

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: apply manual pair parameters onto env"
```

### Task 6: 把手工 pair 模式接入主流程与输出表

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 在 `build_pair_parameter_rows` 中新增 release/deadline 输出列**

```python
"release_time_slot": int(pair_release_time_slot[pair_id]),
"deadline_slot": int(pair_deadline_slot[pair_id]),
```

**Step 2: 在 `compute_pair_parameter_rows` 里传入 env 新字段**

```python
pair_release_time_slot=e.pair_release_time_slot,
pair_deadline_slot=e.pair_deadline_slot,
```

**Step 3: 在 `__main__` 中按模式切换**

```python
e = env(...)
if config["pair_generation_mode"] == "manual":
    apply_manual_pair_parameters(e, config["pair_parameters"])
```

**Step 4: 写一个真实运行测试**

```python
def test_script_runs_from_manual_pair_json_config():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(MANUAL_CONFIG_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "release_time_slot" in proc.stdout
    assert "deadline_slot" in proc.stdout
```

**Step 5: 运行运行测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: 手工 JSON 样例与原随机模式都通过

**Step 6: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: support manual pair json input in ap stats script"
```

### Task 7: 保持随机模式兼容并验证两种模式

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Test: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: 写一个随机模式回归测试**

```python
def test_script_random_mode_still_runs_without_pair_parameters(tmp_path):
    config_path = tmp_path / "random.json"
    config_path.write_text(json.dumps({"cell_size": 1, "pair_density": 0.05, "seed": 123}), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
```

**Step 2: 运行完整 AP stats 测试集**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_smoke.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: 全部通过

**Step 3: 分别运行两种配置**

Run: `python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json`
Expected: 随机模式成功

Run: `python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
Expected: 手工 pair 模式成功

**Step 4: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "test: verify random and manual pair config modes"
```

### Task 8: 完成前代码审查与验收

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_src/env/env.py`
- Modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Create: `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 用 `requesting-code-review` 做完成前审查**

Review scope:
- `manual` / `random` 双模式是否都可用
- `release_time_slot` / `deadline_slot` 是否真的进入输入与输出
- 手工 pair 模式是否破坏现有随机生成路径
- JSON 注释字段与相对 `output_dir` 行为是否仍稳定

**Step 2: 运行最终验收命令**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_smoke.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: 全部通过

Run: `python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json`
Expected: 成功生成随机实例输出

Run: `python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
Expected: 成功生成手工 pair 实例输出

**Step 3: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_src/env/env.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: support manual pair inputs with release and deadline in json"
```
