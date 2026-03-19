# BLE 50-Pair Config Sync And AP Stats JSON Input Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把已验证可运行的 50 对 BLE SDP 配置同步回主工作区，并让 `sim_script/pd_mmw_template_ap_stats.py` 兼容读取类似的 JSON 配置文件输入。

**Architecture:** 先把 worktree 中已经验证通过的 `ble_macrocycle_hopping_sdp_config.json` 同步到主工作区，避免主工作区和实验 worktree 配置漂移。然后在 `pd_mmw_template_ap_stats.py` 上新增一层“配置解析”适配器：保留现有 CLI 作为默认/覆盖入口，同时支持从 JSON 读取结构化参数，并忽略 `_comment_*` 中文注释字段。实现时优先增加解析与验证函数，再把 `__main__` 中环境构建和输出路径逻辑改为统一消费解析后的配置对象。

**Tech Stack:** Python 3、`argparse`、`json`、`pathlib`、现有 `pytest`/`subprocess` 测试、项目内 `env`/`mmw`/`binary_search_relaxation` 逻辑

---

### Task 1: 同步 50 对 BLE JSON 配置到主工作区

**Files:**
- Modify: `.worktrees/ble-hopping-config-interface/ble_macrocycle_hopping_sdp_config.json`
- Create: `ble_macrocycle_hopping_sdp_config.json`
- Test: `ble_macrocycle_hopping_sdp.py`

**Step 1: 检查 worktree 配置是否仍然是 50 对版本**

Run: `python - <<'PY'\nimport json\nfrom pathlib import Path\np = Path('.worktrees/ble-hopping-config-interface/ble_macrocycle_hopping_sdp_config.json')\ndata = json.loads(p.read_text(encoding='utf-8'))\nprint(len(data['pair_configs']))\nprint(min(x['pair_id'] for x in data['pair_configs']), max(x['pair_id'] for x in data['pair_configs']))\nPY`
Expected: 输出 `50`，以及 `0 49`

**Step 2: 把配置文件复制到主工作区根目录**

Run: `cp .worktrees/ble-hopping-config-interface/ble_macrocycle_hopping_sdp_config.json ble_macrocycle_hopping_sdp_config.json`
Expected: 主工作区根目录出现新的 JSON 配置文件

**Step 3: 用主工作区脚本读取主工作区 JSON 配置运行一次**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python ble_macrocycle_hopping_sdp.py --config ble_macrocycle_hopping_sdp_config.json`
Expected: 成功输出候选状态统计和调度结果，并生成 `ble_macrocycle_hopping_sdp_schedule.png`

**Step 4: 记录同步结果**

Run: `git status --short`
Expected: 主工作区出现 `ble_macrocycle_hopping_sdp_config.json`，且没有误改无关文件

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp_config.json
git commit -m "feat: sync 50-pair BLE SDP config"
```

### Task 2: 为 AP Stats 脚本写 JSON 配置解析测试

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Test: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: 在逻辑测试里新增“剥离注释字段”和“读取 JSON 配置”的失败测试**

```python
from pathlib import Path

from sim_script.pd_mmw_template_ap_stats import load_json_config, strip_comment_keys


def test_strip_comment_keys_removes_comment_entries():
    payload = {
        "_comment_root": "说明",
        "seed": 7,
        "nested": {"_comment_x": "忽略", "value": 1},
    }
    assert strip_comment_keys(payload) == {"seed": 7, "nested": {"value": 1}}


def test_load_json_config_reads_core_fields(tmp_path: Path):
    config = tmp_path / "ap_stats_config.json"
    config.write_text(
        json.dumps(
            {
                "_comment_cell_size": "办公室网格边长",
                "cell_size": 1,
                "pair_density": 0.05,
                "seed": 123,
                "mmw_nit": 5,
                "output_dir": "out",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_json_config(config)
    assert loaded["cell_size"] == 1
    assert loaded["pair_density"] == 0.05
    assert loaded["output_dir"] == str(tmp_path / "out")
```

**Step 2: 在运行测试里新增“脚本支持 --config”的失败测试**

```python
def test_script_runs_from_json_config(tmp_path):
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    config = tmp_path / "ap_stats_config.json"
    config.write_text(
        json.dumps(
            {
                "cell_size": 1,
                "pair_density": 0.05,
                "seed": 123,
                "mmw_nit": 5,
                "mmw_eta": 0.05,
                "output_dir": "json_out",
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pair_density_per_m2 = 0.05" in proc.stdout
    assert (tmp_path / "json_out" / "pair_parameters.csv").exists()
```

**Step 3: 运行新测试，确认当前实现先失败**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: 因 `load_json_config` / `--config` 尚未实现而失败

**Step 4: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "test: cover json config input for ap stats script"
```

### Task 3: 实现 AP Stats JSON 配置解析与验证

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: 添加 JSON 和路径工具导入**

```python
import json
from pathlib import Path
```

**Step 2: 实现注释剥离与配置读取函数**

```python
def strip_comment_keys(payload):
    if isinstance(payload, dict):
        return {
            key: strip_comment_keys(value)
            for key, value in payload.items()
            if not str(key).startswith("_comment")
        }
    if isinstance(payload, list):
        return [strip_comment_keys(item) for item in payload]
    return payload


def load_json_config(config_path: Path):
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = strip_comment_keys(raw)
    output_dir = config.get("output_dir", "sim_script/output")
    config["output_dir"] = str((Path(config_path).parent / output_dir).resolve()) if not Path(output_dir).is_absolute() else output_dir
    return config
```

**Step 3: 定义默认参数表和校验逻辑**

```python
DEFAULT_CONFIG = {
    "cell_size": 2,
    "pair_density": 0.05,
    "seed": None,
    "mmw_nit": 200,
    "mmw_eta": 0.05,
    "use_gpu": False,
    "gpu_id": 0,
    "max_slots": 300,
    "ble_channel_retries": 0,
    "ble_channel_mode": "single",
    "output_dir": "sim_script/output",
    "wifi_first_ble_scheduling": False,
}


def merge_config_with_defaults(config: dict) -> dict:
    merged = DEFAULT_CONFIG.copy()
    merged.update(config)
    if int(merged["max_slots"]) < 2:
        raise ValueError("max_slots must be at least 2.")
    if merged["ble_channel_mode"] not in {"single", "per_ce"}:
        raise ValueError("ble_channel_mode must be 'single' or 'per_ce'.")
    return merged
```

**Step 4: 运行逻辑测试，确认解析函数通过**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q`
Expected: 新增 JSON 解析测试通过

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add json config loader for ap stats script"
```

### Task 4: 把 CLI 与 JSON 配置汇总到统一入口

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:943-1040`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 扩展 `parse_args()` 支持 `--config`**

```python
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="JSON 配置文件路径；支持带 _comment_* 中文注释字段。",
)
```

**Step 2: 把 argparse 结果转换成最终配置对象**

```python
def resolve_runtime_config(args):
    cli_values = vars(args).copy()
    config_path = cli_values.pop("config", None)
    if config_path is None:
        return merge_config_with_defaults(cli_values)

    file_config = merge_config_with_defaults(load_json_config(Path(config_path)))
    for key, value in cli_values.items():
        if value != DEFAULT_CONFIG[key]:
            file_config[key] = value
    return file_config
```

**Step 3: 修改 `__main__` 使用 `config` 字典，不再直接读 `args.xxx`**

```python
args = parse_args()
config = resolve_runtime_config(args)
runtime_device = resolve_torch_device(config["use_gpu"], config["gpu_id"])

e = env(
    cell_size=config["cell_size"],
    pair_density_per_m2=config["pair_density"],
    seed=int(time.time()) if config["seed"] is None else config["seed"],
    ble_channel_mode=config["ble_channel_mode"],
    ...
)
```

**Step 4: 运行 `--config` 相关测试，确认脚本可从 JSON 启动**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: 新增 `--config` 运行测试通过，旧 CLI 测试仍然通过

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: allow ap stats script to run from json config"
```

### Task 5: 提供默认 AP Stats JSON 配置样例

**Files:**
- Create: `sim_script/pd_mmw_template_ap_stats_config.json`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 新建带中文注释字段的默认 JSON 样例**

```json
{
  "_comment_cell_size": "办公室网格边长（办公室数量 = cell_size^2）",
  "cell_size": 1,
  "_comment_pair_density": "通信对密度（每平方米）",
  "pair_density": 0.05,
  "_comment_seed": "随机种子；null 表示运行时取当前时间",
  "seed": 123,
  "_comment_mmw_nit": "MMW 迭代次数",
  "mmw_nit": 5,
  "_comment_mmw_eta": "MMW 步长 eta",
  "mmw_eta": 0.05,
  "_comment_output_dir": "输出目录；相对路径相对配置文件所在目录解析",
  "output_dir": "output"
}
```

**Step 2: 补一个“默认样例配置可加载”的测试**

```python
def test_default_ap_stats_json_config_exists_and_runs():
    config = pathlib.Path("sim_script/pd_mmw_template_ap_stats_config.json")
    assert config.exists()
```

**Step 3: 运行样例配置启动脚本**

Run: `python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json`
Expected: 脚本成功运行，输出目录落在 `sim_script/output` 或配置指定目录

**Step 4: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats_config.json sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "docs: add json config example for ap stats script"
```

### Task 6: 全量验证并准备请求代码审查

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Create: `sim_script/pd_mmw_template_ap_stats_config.json`
- Create: `ble_macrocycle_hopping_sdp_config.json`

**Step 1: 运行 AP Stats 相关测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_smoke.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: 全部通过

**Step 2: 运行 BLE SDP 主工作区配置验证**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python ble_macrocycle_hopping_sdp.py --config ble_macrocycle_hopping_sdp_config.json`
Expected: 50 对 BLE 调度成功，PNG 正常生成

**Step 3: 运行 AP Stats JSON 样例验证**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json`
Expected: 成功输出统计表并生成 CSV / PNG

**Step 4: 使用 `requesting-code-review` 做一次完成前检查**

Review scope:
- JSON 配置兼容是否保持旧 CLI 行为
- 相对 `output_dir` 解析是否稳定
- `_comment_*` 注释字段是否完全忽略
- 50 对 BLE 配置同步是否没有引入主工作区漂移

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp_config.json sim_script/pd_mmw_template_ap_stats.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: add json config input for ap stats script"
```
