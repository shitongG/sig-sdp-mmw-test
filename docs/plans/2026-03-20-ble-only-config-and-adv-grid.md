# BLE-Only Config And Advertising-Grid Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 `ble_macrocycle_hopping_sdp.py` 支持通过 `ble_macrocycle_hopping_sdp_config.json` 导入 50 对 BLE pair 配置进行 standalone 调度，并把其输出图片与主脚本的绘图语义对齐，显式标出未占用的 BLE 广播信道。

**Architecture:** 保留 `ble_macrocycle_hopping_sdp.py` 现有 solver / rounding / event-block 主流程，只新增一层 JSON 配置解析与 demo fallback。绘图层在现有 `render_event_grid(...)` 上补充 `BLE adv idle` 灰带和图例，避免重写整套绘图逻辑；同时通过测试保证 standalone 配置入口能读取 50 对实例并成功生成图。

**Tech Stack:** Python 3.10, NumPy, matplotlib, CVXPY, json, pytest, unittest

---

### Task 1: 为 standalone JSON 配置入口写失败测试

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Reference: `ble_macrocycle_hopping_sdp.py:739-885`
- Reference: `ble_macrocycle_hopping_sdp_config.json`

**Step 1: 写一个失败测试，要求脚本能加载 JSON 配置**

在 `tests/test_ble_macrocycle_hopping_sdp.py` 新增测试，验证存在一个类似：

```python
def load_config_from_json(path: pathlib.Path):
    ...
```

或等价函数，能够读取：
- `num_channels`
- `pair_configs`
- `pattern_dict`
- `hard_collision_threshold`
- `plot_title`
- `output_path`

最小断言：

```python
def test_load_config_from_json_reads_pair_configs_and_patterns():
    config = MODULE.load_config_from_json(MODULE_PATH.parent / "ble_macrocycle_hopping_sdp_config.json")
    assert config["num_channels"] == 37
    assert len(config["pair_configs"]) == 50
    assert len(config["pattern_dict"]) == 50
```

**Step 2: 运行测试确认当前失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k load_config_from_json
```

Expected:
- FAIL，因为当前脚本没有 JSON loader

**Step 3: 写一个失败测试，要求 `main()` 支持配置路径入口**

新增测试，不一定直接调用 CLI，但至少要求：
- 可以从配置对象构造完整求解输入
- 50 对 `pair_configs` 会真的进入 `build_candidate_states()`

建议为可测试性新增 `run_instance(...)` 或 `solve_from_config(...)` 之类的纯函数，而不是直接在测试里跑 CLI。

示例：

```python
def test_run_instance_supports_large_config_without_demo_fallback():
    config = MODULE.load_config_from_json(...)
    result = MODULE.run_instance_from_config(config, solve=False)
    assert result["pair_count"] == 50
```

这里 `solve=False` 可以是你为了测试而设计的轻量入口，只要求把配置解析和候选状态构建连通。

**Step 4: 运行测试确认失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "large_config or demo_fallback"
```

Expected:
- FAIL

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "test: define ble-only standalone config loading"
```

### Task 2: 实现 JSON 配置解析与 standalone 入口

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py:1-120`
- Modify: `ble_macrocycle_hopping_sdp.py:739-885`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: 新增配置解析函数**

在 `ble_macrocycle_hopping_sdp.py` 中新增：

```python
def strip_comment_keys(payload): ...
def load_config_from_json(config_path: Path) -> Dict[str, Any]: ...
```

要求：
- 忽略 `_comment_*`
- 支持相对 `output_path` 相对配置文件目录解析
- 把 `pair_configs` 转成 `PairConfig`
- 把 `pattern_dict` 转成 `HoppingPattern`

**Step 2: 新增可测试的实例运行函数**

新增一个类似：

```python
def run_instance(
    pair_configs,
    cfg_dict,
    pattern_dict,
    pair_weight,
    num_channels,
    hard_collision_threshold=None,
    output_path=None,
    plot_title="BLE Event Grid",
):
    ...
```

要求：
- 封装现有 `main()` 里的核心流程
- 返回 `selected`、`blocks`、`overlap_blocks`、`objective_value` 等结果
- 不改变现有求解逻辑

这样测试可以直接调用 `run_instance(...)`，不必依赖 CLI。

**Step 3: 新增 CLI 参数并保留 demo fallback**

为 `main()` 增加：
- `--config`

逻辑要求：
- 若传 `--config`，读取 JSON 并运行配置实例
- 若不传，继续回退到 `build_demo_instance()`

不要移除现有 demo 行为。

**Step 4: 运行 Task 1 的测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "load_config_from_json or large_config or demo_fallback"
```

Expected:
- PASS

**Step 5: 跑 BLE-only 全量测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: add standalone ble-only json config support"
```

### Task 3: 为 BLE 广播信道灰带写失败测试

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Reference: `ble_macrocycle_hopping_sdp.py:344-416`

**Step 1: 写一个失败测试，要求绘图函数支持 BLE adv idle 灰带**

建议先新增可测试的纯函数，例如：

```python
def build_ble_advertising_idle_blocks(slot_end: int) -> List[EventBlock]:
    ...
```

测试要求：
- 返回 3 条广播信道灰带
- 频率分别是 `2402/2426/2480`
- 时间范围覆盖整张图的 slot 轴

示例：

```python
def test_build_ble_advertising_idle_blocks_returns_three_reserved_bands():
    adv_blocks = MODULE.build_ble_advertising_idle_blocks(slot_end=20)
    assert [b.frequency_mhz for b in adv_blocks] == [2402.0, 2426.0, 2480.0]
    assert all(b.start_slot == 0 for b in adv_blocks)
    assert all(b.end_slot == 20 for b in adv_blocks)
```

**Step 2: 运行测试确认失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k advertising_idle_blocks
```

Expected:
- FAIL

**Step 3: 写一个失败测试，要求图例里出现 `BLE adv idle`**

通过 mock 或 stdout / side-effect 方式，确认 `render_event_grid(...)` 的图例包含 `BLE adv idle`。

如果直接测图例太脆，可以退一步测试：
- 渲染前合成的 adv block 被传进了绘图
- 或者在新 helper 中明确返回 `adv_blocks`

**Step 4: 运行测试确认失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "BLE adv idle or advertising_idle"
```

Expected:
- FAIL

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "test: define ble-only advertising-band overlay"
```

### Task 4: 让 BLE-only 图与主脚本语义对齐

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py:344-416`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: 新增广播信道灰带辅助函数**

实现：

```python
def build_ble_advertising_idle_blocks(slot_end: int) -> List[EventBlock]:
    ...
```

要求：
- 返回 3 个灰带 block
- 频率是 `BLE_ADVERTISING_CENTER_FREQ_MHZ`
- 覆盖 `0..slot_end`
- `pair_id/event_index/offset/pattern_id` 可用 `-1` 占位

**Step 2: 修改 `render_event_grid(...)`**

要求：
- 在普通 BLE block 和 overlap block 之外，额外绘制 `BLE adv idle` 灰带
- 图例至少包含：
  - `BLE`
  - `BLE overlap`
  - `BLE adv idle`
- 灰带应画在整张图的全时间范围上

**Step 3: 保持事件块不占广播信道**

不要改变 event block 的频率映射，它们已经在 data channel 上；这里只是把“广播信道保留不用”以视觉方式画出来。

**Step 4: 运行 Task 3 的测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "advertising_idle_blocks or BLE adv idle or advertising_idle"
```

Expected:
- PASS

**Step 5: 跑 BLE-only 全量测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: add ble advertising idle bands to standalone plot"
```

### Task 5: 实际跑 50 对 BLE 配置

**Files:**
- Reference: `ble_macrocycle_hopping_sdp_config.json`
- Optional Modify: `README.md`

**Step 1: 用 JSON 配置直接运行 standalone 脚本**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python ble_macrocycle_hopping_sdp.py --config ble_macrocycle_hopping_sdp_config.json
```

Expected:
- PASS
- stdout 能显示候选空间摘要和求解结果
- 使用的是 50 对 BLE pair，而不是 4 对 demo
- 图片写入 `ble_macrocycle_hopping_sdp_schedule.png`

**Step 2: 记录关键结果**

把以下信息整理到最终说明：
- `pair_count`
- `state_count`
- 是否使用了 JSON 配置
- 输出图片路径
- 图例是否包含 `BLE adv idle`

**Step 3: 若 50 对配置太慢，先诊断再最小修正**

允许：
- 在 stdout 增加更清楚的摘要打印
- 利用现有 `max_offsets_per_pair` 或配置字段控制状态数

禁止：
- 把 50 对配置偷偷改回 4 对
- 跳过求解，仅做伪输出

**Step 4: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp_config.json
git commit -m "test: validate standalone ble-only run with 50-pair config"
```

### Task 6: 最终验证

**Files:**
- Reference only

**Step 1: 运行语法检查**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python -m py_compile ble_macrocycle_hopping_sdp.py
```

Expected:
- PASS

**Step 2: 运行 BLE-only 全量测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected:
- PASS

**Step 3: 跑 demo fallback 与 JSON 配置两条入口**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python ble_macrocycle_hopping_sdp.py
python ble_macrocycle_hopping_sdp.py --config ble_macrocycle_hopping_sdp_config.json
```

Expected:
- 两者都 PASS
- 第一条仍是 4 对 demo
- 第二条是 50 对配置

**Step 4: 整理结论**

最终说明必须包含：
- standalone 脚本现在如何读取 JSON
- 图片如何标出 `BLE adv idle`
- 50 对 BLE 配置是否已经直接可运行
- 如仍有性能限制，瓶颈在哪

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py README.md
git commit -m "chore: verify standalone ble-only config and advertising-grid support"
```
