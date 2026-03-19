# SIG-SDP-MMW

本仓库包含两部分内容：

1. 原始 SIG-SDP/MMW 论文代码
2. 在此基础上扩展出的 WiFi/BLE 混合调度、宏周期导出、事件级绘图和 BLE hopping 实验代码

论文链接：
- [SIG-SDP: Sparse Interference Graph-Aided Semidefinite Programming for Large-Scale Wireless Time-Sensitive Networking](https://arxiv.org/pdf/2501.11307)

![system](im-mmw.png)

## 1. 环境

推荐直接使用现有 conda 环境：

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
cd /data/home/Jie_Wan/mycode/sig-sdp-mmw-test
```

如果你要重新安装依赖：

```bash
pip install -r requirements.txt
```

说明：
- `cvxpy` 用于 BLE-only 宏周期 hopping SDP 路径
- 当前主实验脚本默认仍可在 `legacy` BLE backend 下运行

## 2. 主要入口

### 2.1 混合 WiFi/BLE 主脚本

主入口：
- `sim_script/pd_mmw_template_ap_stats.py`

这个脚本负责：
- 随机或手工生成用户对参数
- 运行 MMW / binary search 可行性流程
- 进行宏周期起始时隙分配
- 导出 CSV
- 生成 WiFi/BLE 时频调度图

### 2.2 BLE-only 宏周期 hopping SDP 原型

原型脚本：
- `ble_macrocycle_hopping_sdp.py`

这个脚本负责：
- 枚举 BLE candidate state `(offset, pattern)`
- 计算候选状态间碰撞代价矩阵
- 解 SDP 松弛并 rounding
- 输出 BLE 事件级时频块和调度图

现在主脚本已经可以通过配置把这个 BLE-only 后端接入现有调度链路，同时保留原 `legacy` 方案。

## 3. 调度策略算法

### 3.1 主流程

主脚本 [pd_mmw_template_ap_stats.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py) 的调度流程可以概括成 5 步：

1. 生成或读取用户对参数  
   支持 `random` 和 `manual` 两种模式。`manual` 模式下可以直接在 JSON 中给出每个 pair 的无线制式、时间窗口、信道和业务参数。

2. 构造环境与干扰关系  
   [env.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/env/env.py) 负责：
   - 生成 WiFi/BLE pair
   - 设置每个 pair 的时序参数
   - 计算链路损耗、最小 SINR、时频占用
   - 维护 WiFi 固定 1/6/11 与 BLE data channel / advertising channel 的频谱语义

3. 可行性求解  
   主干求解来自原始 SIG-SDP/MMW 流程：
   - [mmw.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/alg/mmw.py)
   - [binary_search_relaxation.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/alg/binary_search_relaxation.py)

   这里先决定哪些 pair 能进入宏周期调度，以及对应的连续松弛结果。

4. 宏周期调度  
   对可行 pair 做起始时隙分配，得到 `schedule_slot`，再导出 `pair_parameters.csv`、`wifi_ble_schedule.csv` 和绘图中间表。

5. BLE 信道调度  
   这里有两条后端：
   - `legacy`
   - `macrocycle_hopping_sdp`

### 3.2 `legacy` BLE 后端

`legacy` 是当前主脚本里保留的原有 BLE 调度方式。

- 如果 `ble_channel_mode = single`，每个 BLE pair 使用一个固定数据信道。
- 如果 `ble_channel_mode = per_ce`，每个 BLE 连接事件单独分配信道，最终写到 `pair_ble_ce_channels`。
- 该模式更贴近原始主脚本的事件展开和绘图链路，速度更稳定，适合大批量随机实验。

### 3.3 `macrocycle_hopping_sdp` BLE 后端

当 `ble_schedule_backend = macrocycle_hopping_sdp` 时，主脚本会调用 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 的 BLE-only 宏周期跳频调度器。

建模步骤是：

1. 对每个 BLE pair 构造候选状态  
   每个状态是 `(pair_id, offset, pattern_id)`，表示这个 pair 选择某个宏周期起始偏移和某个 hopping pattern。

2. 预计算碰撞矩阵 `Omega`  
   如果两个候选状态在宏周期内出现同信道、同时间重叠，就把重叠长度计入代价。

3. 建立 SDP 松弛  
   目标是最小化所有被同时选中的候选状态之间的总碰撞代价，同时保证每个 pair 只选一个候选状态。

4. rounding 回离散解  
   SDP 求得的是松弛矩阵，再通过 rounding 选出每个 pair 的最终 `(offset, pattern)`。

5. 回写主脚本  
   选中状态会被转换成 `pair_ble_ce_channels`，然后继续复用主脚本已有的 CSV 导出和绘图逻辑。

这个后端更适合：
- 研究 BLE-only hopping 规则
- 对比不同 BLE pattern 的碰撞代价
- 做 small/medium scale 的宏周期跳频实验

这个后端不适合：
- 不加约束直接跑很宽的 BLE 时间窗口
- 候选 offset 和 pattern 过多的超大实例

### 3.4 BLE `ble_timing_mode`

在 `manual` JSON 模式下，BLE pair 现在支持两种 timing 输入：

- `ble_timing_mode = manual`
  你手工给出 `ble_ci_slots` 和 `ble_ce_slots`

- `ble_timing_mode = auto`
  脚本按 `seed` 自动生成 BLE 的 `ci/ce`

`auto` 模式的特点：
- 复用环境里的 BLE timing 采样逻辑
- 结果可复现
- 不需要在 JSON 里手填 `ble_ci_slots` 和 `ble_ce_slots`
- 如果不传 JSON，整个脚本仍然保持旧的随机用户对生成和调度方式

## 4. 常用运行方式

### 4.1 随机生成 WiFi/BLE 用户对

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --cell-size 1 \
  --pair-density 0.05 \
  --seed 123 \
  --mmw-nit 5 \
  --output-dir sim_script/output
```

### 4.2 读取手工 JSON 配置

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --config sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json
```

### 4.3 使用 BLE `per_ce` 事件级信道模式

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --cell-size 1 \
  --pair-density 0.05 \
  --seed 123 \
  --mmw-nit 5 \
  --ble-channel-mode per_ce \
  --output-dir sim_script/output
```

### 4.4 使用 BLE-only 宏周期 hopping SDP 后端

快速 smoke 配置：

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_empty_config.json
```

说明：
- `pd_mmw_template_ap_stats_macrocycle_hopping_empty_config.json` 现在是真正的空实例，`pair_density = 0.0`
- 用途是快速验证 `macrocycle_hopping_sdp` 后端入口，不用于性能测试

显式 CLI 覆盖：

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --config sim_script/pd_mmw_template_ap_stats_config.json \
  --ble-schedule-backend macrocycle_hopping_sdp
```

### 4.5 运行 9 WiFi + 16 BLE 的 auto BLE timing 示例

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
```

这份配置的特点：
- `9` 对 WiFi + `16` 对 BLE
- BLE 使用 `macrocycle_hopping_sdp`
- BLE 的 `ble_ci_slots` / `ble_ce_slots` 由 `seed` 自动生成
- BLE 的 `release/deadline` 已收紧到单一可行 offset，避免 SDP 状态空间爆炸

### 4.6 单独运行 BLE-only SDP 原型

```bash
python ble_macrocycle_hopping_sdp.py
```

## 5. 关键配置文件

### 4.1 默认随机配置

- `sim_script/pd_mmw_template_ap_stats_config.json`

用途：
- 随机生成用户对
- 默认 BLE backend 为 `legacy`

### 4.2 手工用户对配置

- `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`

用途：
- 直接手工指定 `pair_parameters`
- 当前文件已经预置了 50 对用户对，方便继续做实验修改

### 4.3 BLE hopping backend smoke 配置

- `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_empty_config.json`

用途：
- 不生成任何用户对
- 只快速验证 `macrocycle_hopping_sdp` 后端入口是否正常

### 4.4 BLE hopping backend 小规模示例

- `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_config.json`

用途：
- 提供一个很小的 hand-crafted 示例
- 适合调试配置格式

### 4.5 BLE hopping backend 大规模混合示例

- `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`

用途：
- 提供 `9` 对 WiFi + `16` 对 BLE 的 mixed 实验
- BLE 使用 `macrocycle_hopping_sdp` 后端
- BLE 的 `ble_ci_slots` 和 `ble_ce_slots` 由 `seed` 自动生成，不再在 JSON 里手填
- 适合作为大规模通信下的回归与性能基线

## 6. 可以改哪些参数

### 5.1 随机模式

在 `pd_mmw_template_ap_stats_config.json` 或 CLI 里常改：

- `cell_size`
- `pair_density`
- `seed`
- `mmw_nit`
- `mmw_eta`
- `max_slots`
- `ble_channel_mode`
- `ble_schedule_backend`
- `ble_channel_retries`
- `wifi_first_ble_scheduling`
- `output_dir`

### 5.2 manual 模式

在 `pair_parameters` 里可逐对修改：

- `pair_id`
- `office_id`
- `radio`
- `channel`
- `priority`
- `release_time_slot`
- `deadline_slot`
- `start_time_slot`
- `wifi_anchor_slot`
- `wifi_period_slots`
- `wifi_tx_slots`
- `ble_anchor_slot`
- `ble_timing_mode`
- `ble_ci_slots`
- `ble_ce_slots`
- `ble_ce_channels`

规则：
- WiFi 信道只能填 `0/5/10`
- BLE 数据信道只能填 `0..36`
- `pair_id` 必须从 `0` 连续编号
- 如果 BLE 使用 `macrocycle_hopping_sdp`，`release_time_slot` 和 `deadline_slot` 会直接影响可行 offset 数量；窗口过宽会显著拖慢 SDP
- `ble_timing_mode = auto` 时，manual JSON 里的 BLE pair 会按 `seed` 自动生成 `ble_ci_slots` 和 `ble_ce_slots`，并复用环境中的 BLE 时序采样逻辑
- `ble_timing_mode = auto` 时不需要手写 `ble_ci_slots` 和 `ble_ce_slots`
- 完全不传 JSON 时，脚本仍保持旧的随机生成模式

## 7. 当前信道建模

### 6.1 WiFi

当前场景中，WiFi 用户对只允许占用固定 3 个 20 MHz 信道：

- `channel 0` 对应中心频率 `2412 MHz`
- `channel 5` 对应中心频率 `2437 MHz`
- `channel 10` 对应中心频率 `2462 MHz`

### 6.2 BLE

BLE 数据链路使用 37 个 data channel，每个信道带宽 2 MHz。

中心频率映射为两段：

- `0..10 -> 2404 + 2*k MHz`
- `11..36 -> 2428 + 2*(k-11) MHz`

BLE 广播信道保留为：

- `2402 MHz`
- `2426 MHz`
- `2480 MHz`

在调度图里，这 3 条广播信道会以 `BLE adv idle` 灰带显示。

## 8. 输出结果

主脚本常见输出在 `output_dir` 下：

- `pair_parameters.csv`
  - 每个 pair 的参数和调度结果
- `wifi_ble_schedule.csv`
  - 时隙级汇总
- `unscheduled_pairs.csv`
  - 未调度用户对
- `schedule_plot_rows.csv`
  - 时频绘图中间表
- `ble_ce_channel_events.csv`
  - `per_ce` 模式下 BLE 事件级信道表
- `wifi_ble_schedule.png`
  - 兼容旧命名的调度图
- `wifi_ble_schedule_overview.png`
  - 整体时频图
- `wifi_ble_schedule_window_*.png`
  - 分窗口时频图

BLE-only SDP 原型会输出：

- `ble_macrocycle_hopping_sdp_schedule.png`

## 9. 性能说明

`ble_macrocycle_hopping_sdp.py` 的 SDP 目标已经改成向量化形式，能减少 CVXPY 在 Python 侧构造和编译目标函数的开销。

这会改善：
- `Objective contains too many subexpressions` 这类警告对应的编译时间
- 小到中等规模实例的启动时间

这不会从根本上改变：
- SDP 的矩阵维度
- SCS 求解器本身的收敛时间

当前最影响 `macrocycle_hopping_sdp` 速度的因素是 BLE 候选状态总数，也就是：
- 每对 BLE 的可行 `offset` 数量
- 每对 BLE 的 hopping pattern 数量
- BLE pair 总数

如果你要控制运行时间，优先收紧：
- `release_time_slot`
- `deadline_slot`
- `ble_ci_slots`
- `ble_ce_slots`

尤其是在 `manual` 模式下，建议先把每对 BLE 的时间窗口收紧到只保留少量可行 offset。

## 10. 大规模 mixed 实验

运行 `9` 对 WiFi + `16` 对 BLE，且 BLE 使用 `macrocycle_hopping_sdp`：

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
```

这份配置做了两点控制：
- WiFi 固定在 `1/6/11` 三个信道
- BLE 的 `release/deadline` 窗口被收紧，避免候选 offset 爆炸

## 11. 代码结构

- `sim_src/env/env.py`
  - 环境建模、WiFi/BLE 时序参数、冲突关系、事件展开
- `sim_src/alg/mmw.py`
  - MMW 可行性求解
- `sim_src/alg/binary_search_relaxation.py`
  - binary search 松弛求解
- `sim_script/pd_mmw_template_ap_stats.py`
  - 当前主实验脚本
- `sim_script/plot_schedule_from_csv.py`
  - 从 CSV 生成 overview/window 调度图
- `ble_macrocycle_hopping_sdp.py`
  - BLE-only 宏周期 hopping SDP 原型
- `sim_script/tests`
  - 混合调度与导出测试
- `tests/test_ble_macrocycle_hopping_sdp.py`
  - BLE-only SDP 原型测试

## 12. 常用测试

建议至少跑这些：

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

只验证新 backend 的入口时，可跑：

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "legacy_ble_backend or ble_schedule_backend_cli_override or macrocycle_hopping_backend_json_config"
```

## 13. 引用

```bibtex
@article{gu2025sig,
  title={SIG-SDP: Sparse Interference Graph-Aided Semidefinite Programming for Large-Scale Wireless Time-Sensitive Networking},
  author={Gu, Zhouyou and Park, Jihong and Vucetic, Branka and Choi, Jinho},
  journal={arXiv preprint arXiv:2501.11307},
  year={2025}
}
```
