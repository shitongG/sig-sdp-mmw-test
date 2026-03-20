# BLE-Only SDP Paper Doc And Channel Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `ble_macrocycle_hopping_sdp.py` 的 BLE-only 宏周期 hopping SDP 建模原理、约束方程和符号系统按论文格式整理到 `README.md`，并让该脚本的信道映射和生成图片与 `sim_script/pd_mmw_template_ap_stats.py` 保持一致，确保不会跳到 BLE 广播信道。

**Architecture:** 文档层面，保留现有 README 结构，在 BLE-only SDP 章节中新增“问题定义、符号、目标函数、约束方程、rounding、频谱假设”完整小节，使用论文式编号和数学表达。实现层面，`ble_macrocycle_hopping_sdp.py` 只调整 BLE 数据信道的中心频率映射、hopping 取值空间和绘图标签，不改主求解接口，使 standalone BLE-only 脚本与主仿真环境的 37 个 BLE data channel 语义对齐。

**Tech Stack:** Python 3.10, NumPy, CVXPY, matplotlib, pytest, unittest, Markdown

---

### Task 1: 为 BLE-only 频率映射和广播信道排除写测试

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Reference: `ble_macrocycle_hopping_sdp.py:110-190`

**Step 1: 写一个失败测试，锁定 BLE data channel 的两段式频率映射**

在 `tests/test_ble_macrocycle_hopping_sdp.py` 中新增测试，验证：
- data channel `0 -> 2404 MHz`
- data channel `10 -> 2424 MHz`
- data channel `11 -> 2428 MHz`
- data channel `36 -> 2478 MHz`

示例：

```python
def test_ble_data_channel_frequency_mapping_matches_two_segment_model():
    assert MODULE.ble_channel_to_frequency_mhz(0) == 2404.0
    assert MODULE.ble_channel_to_frequency_mhz(10) == 2424.0
    assert MODULE.ble_channel_to_frequency_mhz(11) == 2428.0
    assert MODULE.ble_channel_to_frequency_mhz(36) == 2478.0
```

**Step 2: 运行测试确认当前失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k two_segment_model
```

Expected:
- FAIL，因为当前 standalone 脚本仍按连续 `2402 + 2*k` 方式映射

**Step 3: 写一个失败测试，验证 hopping 不会落到 BLE 广播信道频点**

测试思路：
- 生成一组 event block
- 检查所有 `frequency_mhz` 不属于 `{2402.0, 2426.0, 2480.0}`

示例：

```python
def test_event_blocks_never_use_ble_advertising_frequencies():
    pair_configs, cfg_dict, pattern_dict, _, num_channels = MODULE.build_demo_instance()
    states, _, A_k = MODULE.build_candidate_states(pair_configs, pattern_dict)
    selected = {k: states[idxs[0]] for k, idxs in A_k.items()}
    blocks = MODULE.build_event_blocks(selected, cfg_dict, pattern_dict, num_channels)
    assert {block.frequency_mhz for block in blocks}.isdisjoint({2402.0, 2426.0, 2480.0})
```

**Step 4: 运行测试确认失败**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k advertising_frequencies
```

Expected:
- FAIL

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "test: lock ble-only data-channel frequency model"
```

### Task 2: 对齐 standalone BLE-only 脚本的信道模型

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py:110-190`
- Modify: `ble_macrocycle_hopping_sdp.py:648-760`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: 将 BLE 频率映射改成与主仿真一致的两段式 data channel 映射**

在 `ble_macrocycle_hopping_sdp.py` 中明确：
- BLE data channel `0..10 -> 2404 + 2*k MHz`
- BLE data channel `11..36 -> 2428 + 2*(k-11) MHz`
- 广播信道频点 `{2402, 2426, 2480}` 只作为保留常量，不进入 data channel 映射

建议新增：

```python
BLE_ADVERTISING_CENTER_FREQ_MHZ = (2402.0, 2426.0, 2480.0)

def ble_data_channel_to_frequency_mhz(channel: int) -> float:
    ...
```

然后让现有调用统一走这个函数。

**Step 2: 检查 hopping 规则只在 37 个 data channel 上循环**

确认 `channel_of_event(...)` 的 `num_channels` 仍为 `37`，并且 pattern 的 `start_channel`、模运算空间、demo 数据都只落在 `0..36`。

如果需要，补一个小的合法性校验：

```python
if not (0 <= pattern.start_channel < num_channels):
    raise ValueError(...)
```

**Step 3: 调整绘图频率标签**

确保事件块图纵轴使用新的 data-channel 频率值，而不是旧的连续 BLE 频点。不要在 BLE-only 图里把广播信道当成可占用资源块。

**Step 4: 运行 Task 1 的测试**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "two_segment_model or advertising_frequencies"
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
git commit -m "feat: align ble-only hopping to data-channel model"
```

### Task 3: 为 README 的论文式建模说明写文档护栏

**Files:**
- Modify: `README.md`

**Step 1: 写出 README 目标结构**

在实现前先在本地梳理要加入 README 的 6 个小节，避免写散：
- 问题定义
- 符号表
- 候选状态定义
- 目标函数
- 约束方程
- rounding 与频谱假设

不用先改文件，但要明确这 6 节都要落到 README。

**Step 2: 检查 README 当前 BLE-only 章节位置**

确认文档插入点在：
- 现有 BLE-only 宏周期 hopping SDP 原型章节附近
- 不破坏已有“运行方式”和“性能说明”结构

Run:

```bash
rg -n "BLE-only 宏周期 hopping SDP 原型|macrocycle_hopping_sdp|调度策略算法" README.md
```

Expected:
- 能找到插入位置

**Step 3: Commit**

这一步不提交，只是文档结构设计。

### Task 4: 把 BLE-only SDP 原理按论文格式写入 README

**Files:**
- Modify: `README.md`
- Reference: `ble_macrocycle_hopping_sdp.py`

**Step 1: 新增问题定义和符号系统**

README 中要新增一段论文式描述，至少定义：
- `\mathcal{K}`：BLE pair 集合
- `k \in \mathcal{K}`：第 `k` 个 BLE pair
- `r_k`：release time
- `D_k`：deadline
- `\Delta_k`：connect interval
- `d_k`：CE duration
- `M_k`：宏周期内 CE 数
- `s \in \mathcal{S}_k`：可行 offset
- `\ell \in \mathcal{L}_k`：候选 hopping pattern
- `a = (k, s, \ell)`：candidate state
- `\Omega_{ab}`：两个 candidate state 的碰撞代价
- `Y`：lifted SDP 松弛变量

文风要求：
- 像论文中的“Problem Formulation”
- 句子短，定义明确
- 先定义时间约束，再定义状态，再定义碰撞

**Step 2: 写出可行 offset 集和事件时间表达式**

README 中明确写：

```text
\mathcal{S}_k = \{ s \mid r_k \le s \le D_k - (M_k - 1)\Delta_k - d_k + 1 \}
```

以及：

```text
t_{k,m}(s) = s + m \Delta_k
I_{k,m}(s) = [t_{k,m}(s), t_{k,m}(s) + d_k - 1]
```

**Step 3: 写出 hopping 信道表达式**

对当前程序的简化 hopping 规则明确写：

```text
c_{k,m}^{(\ell)} = (c_{k,0}^{(\ell)} + h_{k}^{(\ell)} m) \bmod 37
```

并明确说明：
- 这里的 `37` 是 BLE data channel 数量
- 广播信道 `2402/2426/2480 MHz` 不属于该 hopping 空间

**Step 4: 写出碰撞代价与目标函数**

README 中用论文风格说明：
- 两事件重叠长度
- 同信道指示函数
- `\Omega_{ab}` 的定义
- SDP/离散目标最小化总碰撞代价

示例形式：

```text
\Omega_{ab} = \sum_{m} \sum_{n} w_{k,j} \cdot \mathbf{1}\{ c_{k,m}^{(\ell)} = c_{j,n}^{(\ell')} \} \cdot | I_{k,m}(s) \cap I_{j,n}(s') |
```

以及 lifted 目标：

```text
\min \sum_{a < b} \Omega_{ab} Y_{ab}
```

**Step 5: 写出约束方程**

README 中按论文格式列出：
- 每个 pair 只能选一个 candidate state
- 同一 pair 的不同 state 互斥
- `Y \succeq 0`
- 可选硬碰撞阈值约束

至少写出：

```text
\sum_{a \in \mathcal{A}_k} Y_{aa} = 1, \quad \forall k
Y_{ab} = 0, \quad \forall a \ne b, a,b \in \mathcal{A}_k
Y \succeq 0
```

**Step 6: 写 rounding 和实现假设**

README 中说明：
- SDP 解是松弛解
- 最终实现用 `diag(Y)` 做简单 rounding
- 当前脚本里的 hopping 是实验性简化模型
- 频谱语义已对齐主仿真：只使用 37 个 BLE data channel，不碰广播信道

**Step 7: 人工通读 README，检查文档结构**

Run:

```bash
rg -n "Problem Formulation|\\mathcal|Omega|rounding|广播信道|37 个 BLE data channel" README.md
```

Expected:
- 新增小节和符号都能被检索到

**Step 8: Commit**

```bash
git add README.md
git commit -m "docs: add paper-style ble-only sdp formulation"
```

### Task 5: 让 BLE-only 图片风格与主脚本输出保持一致

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Reference: `sim_script/plot_schedule_from_csv.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: 对比主脚本的图形语义**

阅读：
- `sim_script/plot_schedule_from_csv.py`
- `ble_macrocycle_hopping_sdp.py` 的 `render_event_grid(...)`

确认需要对齐的内容：
- 纵轴是实际频率 MHz，而不是错误的广播频点映射
- BLE 块只出现于 data channel 频率
- 不出现跳到广播信道的矩形
- 图例和标签风格尽量接近主脚本

**Step 2: 写一个失败测试，验证渲染前后的 `EventBlock.frequency_mhz` 集合不包含广播信道**

如果 Task 1 已覆盖数据块频率集合，可以复用；若不够，补一个渲染前后 smoke 测试，确保输出图来自 data-channel blocks。

**Step 3: 最小调整绘图实现**

只做必要修改：
- 频率换算正确
- 标签显示和主脚本语义一致
- 若需要，可在标题或图例中说明 `BLE data channels only`

不要在这一步引入额外绘图库或重构整套画图代码。

**Step 4: 跑 BLE-only demo**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python ble_macrocycle_hopping_sdp.py
```

Expected:
- PASS
- 生成新的 `ble_macrocycle_hopping_sdp_schedule.png`
- 图中事件不落到 BLE 广播信道频点

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: align ble-only schedule plot with main simulator"
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

**Step 3: 运行 standalone demo**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python ble_macrocycle_hopping_sdp.py
```

Expected:
- PASS
- 输出图片更新

**Step 4: 整理结论**

最终说明必须包含：
- README 新增了哪些论文式小节
- 使用了哪些符号和方程
- BLE-only 脚本如何保证只在 37 个 data channel 上 hopping
- 图片语义与主脚本对齐到了哪里

**Step 5: Commit**

```bash
git add README.md ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "chore: verify ble-only formulation docs and channel alignment"
```
