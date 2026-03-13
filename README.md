# SIG-SDP-MMW

中文：本仓库提供论文 "[SIG-SDP: Sparse Interference Graph-Aided Semidefinite Programming for Large-Scale Wireless Time-Sensitive Networking](https://arxiv.org/pdf/2501.11307)" 的代码实现，并在原始 SIG-SDP/MMW 框架上扩展了混合 WiFi/BLE 场景下的建模、可行性求解、宏周期调度、导出与可视化能力。  
English: This repository contains the codebase for the paper "[SIG-SDP: Sparse Interference Graph-Aided Semidefinite Programming for Large-Scale Wireless Time-Sensitive Networking](https://arxiv.org/pdf/2501.11307)" and extends the original SIG-SDP/MMW framework with mixed WiFi/BLE modeling, feasibility solving, macrocycle scheduling, export, and visualization.

![system](im-mmw.png)

## Overview / 项目概览

中文：当前仓库同时包含两部分内容。第一部分是论文原始的 SIG-SDP/MMW 相关实现；第二部分是在此基础上扩展出的混合 WiFi/BLE 调度工作流，支持不同制式默认参数、逐链路参数覆盖、BLE 跳频模式以及 WiFi-first 调度策略。  
English: The current repository contains two layers. The first layer is the original SIG-SDP/MMW implementation used in the paper. The second layer extends it to a mixed WiFi/BLE scheduling workflow with radio-specific defaults, per-link parameter overrides, BLE hopping modes, and a WiFi-first scheduling strategy.

中文：若目标是复现论文中的原始结果，建议优先查看 [sim_script/journal_version](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/journal_version)。若目标是运行当前混合 WiFi/BLE 实验、导出调度结果并分析 BLE 跳频行为，则建议使用 [pd_mmw_template_ap_stats.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py)。  
English: If your goal is to reproduce the original paper results, start with [sim_script/journal_version](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/journal_version). If your goal is to run the current mixed WiFi/BLE workflow, export schedules, and analyze BLE hopping behavior, use [pd_mmw_template_ap_stats.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py).

## Installation / 安装说明

中文：建议创建独立 Python 环境后安装依赖。  
English: It is recommended to install the dependencies in a dedicated Python environment.

```bash
pip3 install -r requirements.txt
```

中文：`cvxpy` 需要可用的 SCS 求解器后端。  
English: `cvxpy` requires a working SCS solver backend.

Reference: https://www.cvxpy.org/tutorial/solvers/index.html

## Quick Start / 快速开始

中文：当前混合 WiFi/BLE 工作流的主入口脚本是 [pd_mmw_template_ap_stats.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py)。  
English: The main entry point for the current mixed WiFi/BLE workflow is [pd_mmw_template_ap_stats.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py).

基础运行示例 / Basic example:

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --cell-size 1 \
  --pair-density 0.05 \
  --seed 123 \
  --mmw-nit 5 \
  --output-dir sim_script/output
```

中文：该脚本会输出网络规模、WiFi/BLE 对数、可行性求解结果、宏周期调度结果、BLE 时序摘要，并生成 CSV 与图像文件。  
English: The script prints the network scale, WiFi/BLE pair counts, feasibility results, macrocycle scheduling results, and BLE timing summary, and generates CSV and image outputs.

BLE `per_ce` 跳频示例 / BLE `per_ce` hopping example:

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --cell-size 1 \
  --pair-density 0.05 \
  --seed 123 \
  --mmw-nit 5 \
  --ble-channel-mode per_ce \
  --output-dir sim_script/output
```

WiFi-first BLE 调度示例 / WiFi-first BLE scheduling example:

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --cell-size 1 \
  --pair-density 0.05 \
  --seed 123 \
  --mmw-nit 5 \
  --wifi-first-ble-scheduling \
  --ble-channel-retries 1 \
  --output-dir sim_script/output
```

## Problem Setup / 问题设置

中文：系统在办公室网格上生成通信对，每个 pair 被分配为 WiFi 或 BLE，并在给定时隙长度下构造可行性求解与宏周期调度问题。WiFi 业务主要由发送时长与周期约束刻画；BLE 业务主要由连接间隔（CI）与连接事件（CE）持续时间刻画。  
English: The system generates communication pairs on an office grid, assigns each pair to WiFi or BLE, and constructs feasibility and macrocycle scheduling problems under a fixed slot duration. WiFi traffic is mainly characterized by transmission duration and period constraints, while BLE traffic is characterized by connection interval (CI) and connection event (CE) duration.

中文：从建模视角看，当前代码既保留了论文中的干扰图与半定规划近似求解主线，也加入了更细粒度的 BLE 时序和频谱占用语义。  
English: From a modeling perspective, the code preserves the interference-graph and semidefinite-programming-based feasibility pipeline from the paper, while extending it with finer-grained BLE timing and spectrum-occupancy semantics.

## Modeling Assumptions / 建模假设

### Radio-Specific Defaults / 按制式区分的默认参数

中文：当前环境对 WiFi 与 BLE 使用不同默认带宽。默认情况下，WiFi 信道带宽为 `20e6` Hz，BLE 信道带宽为 `2e6` Hz。  
English: The environment uses different default bandwidths for WiFi and BLE. By default, WiFi uses `20e6` Hz channels and BLE uses `2e6` Hz channels.

中文：包长同样支持按制式区分。环境支持遗留全局参数 `packet_bit`，也支持 `wifi_packet_bit` 和 `ble_packet_bit` 两个制式默认值。  
English: Packet size is also radio-specific. The environment supports the legacy global `packet_bit` parameter as well as radio-specific defaults through `wifi_packet_bit` and `ble_packet_bit`.

### Per-Link Overrides / 逐链路参数覆盖

中文：当前实现支持通过 `pair_packet_bits` 与 `pair_bandwidth_hz` 对单条链路的包长和带宽进行覆盖。内部会将这些参数统一整理为 `pair_*`、`user_*` 和 `device_*` 数组，以便在 BLER、最小 SINR 以及状态矩阵计算中复用。  
English: The current implementation supports per-link overrides through `pair_packet_bits` and `pair_bandwidth_hz`. Internally, these values are normalized into `pair_*`, `user_*`, and `device_*` arrays so that BLER, minimum-SINR, and state-matrix calculations can all use the same per-link parameters.

### BLE Timing Model / BLE 时序模型

中文：BLE 侧使用连接间隔 CI、连接事件 CE、负载大小与 PHY 速率共同决定可调度性。代码会导出 `pair_ble_ci_slots`、`pair_ble_ce_slots` 与 `pair_ble_ce_feasible`。若某 BLE pair 在 CE 时长约束下不可行，则不会被放入宏周期调度。  
English: On the BLE side, schedulability is determined jointly by CI, CE duration, payload size, and PHY rate. The code exposes `pair_ble_ci_slots`, `pair_ble_ce_slots`, and `pair_ble_ce_feasible`. A BLE pair that is infeasible under its CE-duration constraint is excluded from macrocycle scheduling.

### BLE Channel Modes / BLE 信道模式

中文：BLE 支持两种信道模式。`single` 表示每个 BLE pair 固定在单一 BLE 信道上；`per_ce` 表示每个连接事件可独立分配信道。在 `per_ce` 模式下，导出和绘图均基于事件级信道占用。  
English: BLE supports two channel modes. `single` keeps each BLE pair on a fixed BLE channel, while `per_ce` assigns channels independently per connection event. In `per_ce` mode, both export and plotting operate on event-level channel occupancy.

## Physical-Layer Reliability Model / 物理层可靠性模型

### Per-Link BLER / 逐链路 BLER

中文：`evaluate_bler()` 先计算每个用户的 SINR，再将每条链路自己的包长、带宽与时隙长度代入 Polyanskiy 有限码长模型。因此，当前 BLER 不是统一 PHY 近似，而是按链路参数逐项计算。  
English: `evaluate_bler()` first computes the SINR of each user and then applies the Polyanskiy finite-blocklength model using that link's own packet size, bandwidth, and slot duration. The current BLER model is therefore computed per link rather than under a single shared PHY approximation.

### Per-Link Scheduling Constraints / 逐链路调度约束

中文：调度约束已与 BLER 语义保持一致。最小 SINR 门限不再使用统一标量，而是通过每条链路自己的包长、带宽、时隙长度和目标误块率计算得到 `pair_min_sinr`。因此，不同 WiFi/BLE 链路可具有不同的干扰容忍度。  
English: The scheduling constraints are aligned with the BLER semantics. The minimum-SINR threshold is no longer a shared scalar; instead, `pair_min_sinr` is computed from each link's own packet size, bandwidth, slot duration, and target block error rate. Different WiFi/BLE links may therefore have different interference tolerances.

### Bandwidth-Dependent Noise / 与带宽相关的噪声模型

中文：噪声模型已经改为标准热噪声近似形式  
English: The noise model has been updated to the standard thermal-noise approximation

```text
N_dBm = -174 + 10*log10(B_Hz) + NOISEFIGURE
```

中文：其中 `NOISEFIGURE = 13 dB`。这一改动不仅影响 BLER 评估，也会通过发射功率与状态矩阵路径影响调度可行性。  
English: where `NOISEFIGURE = 13 dB`. This change affects not only BLER evaluation, but also scheduling feasibility through the transmit-power and state-matrix paths.

## Solver Pipeline / 求解流程

中文：当前求解流程由环境构造、MMW 可行性检查、binary search 松弛搜索以及宏周期起始时隙分配组成。主干逻辑分别位于 [env.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/env/env.py)、[mmw.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/alg/mmw.py) 与 [binary_search_relaxation.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/alg/binary_search_relaxation.py)。  
English: The current solver pipeline consists of environment construction, MMW-based feasibility checking, binary-search relaxation, and macrocycle start-slot assignment. The main logic resides in [env.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/env/env.py), [mmw.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/alg/mmw.py), and [binary_search_relaxation.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/alg/binary_search_relaxation.py).

### Joint Strategy / 联合求解策略

中文：默认情况下，`binary_search_relaxation` 以 `joint` 策略运行，即将所有 WiFi/BLE pair 放入同一可行性求解流程。  
English: By default, `binary_search_relaxation` runs in `joint` mode, where all WiFi/BLE pairs are handled in a single feasibility-solving pipeline.

### WiFi-First Strategy / WiFi 优先策略

中文：当启用 `--wifi-first-ble-scheduling` 时，solver 侧的 `binary_search_relaxation` 将切换到 `wifi_first` 策略。该策略先对 WiFi pair 求解，再对 BLE pair 求解，并在 solver 内部合并结果。  
English: When `--wifi-first-ble-scheduling` is enabled, `binary_search_relaxation` switches to the `wifi_first` strategy. Under this strategy, WiFi pairs are solved first, BLE pairs are solved second, and the results are merged inside the solver API.

中文：在宏周期起始时隙分配阶段，代码进一步采用 WiFi-first 规则：先放置 WiFi pair，再根据 WiFi 已占用频谱计算 BLE 可用物理信道数 `C`，并施加起始时隙容量约束 `N <= C`。当前 repair/refill 比较逻辑也会优先保护 WiFi admission，使 `wifi_first` 不只是遍历顺序优先，而是以尽量保住更多 WiFi pair 为目标。  
English: During macrocycle start-slot assignment, the code further applies a WiFi-first rule: WiFi pairs are placed first, the remaining BLE physical channel count `C` is computed after WiFi spectrum occupancy, and the start-slot capacity constraint `N <= C` is enforced. The repair/refill comparison logic now also protects WiFi admission, so `wifi_first` is no longer just an iteration order; it explicitly prefers keeping more WiFi pairs scheduled.

### BLE Hopping Collision Metric / BLE 跳频碰撞概率指标

中文：在 WiFi-first 模式下，代码会额外记录理论不碰撞概率  
English: Under WiFi-first mode, the code additionally records the theoretical non-collision probability

```text
P_no_collision = (1 - 1/C)^(N - 1)
```

中文：其中 `C` 为该起始时隙的有效 BLE 可选信道数，`N` 为该起始时隙被调度的 BLE pair 数。当前这一量作为统计指标导出，而不是作为全宏周期的严格硬约束。  
English: where `C` is the effective number of BLE channels available at that start slot and `N` is the number of BLE pairs scheduled at that start slot. At present, this quantity is exported as a statistic rather than enforced as a hard constraint over the full macrocycle.

## Main CLI Arguments / 主要命令行参数

中文：下面列出当前主脚本中最常用的参数。  
English: The following options are the most commonly used arguments of the main script.

- `--cell-size`: office grid edge length / 办公室网格边长
- `--pair-density`: pair density per square meter / 每平方米通信对密度
- `--seed`: random seed / 随机种子
- `--mmw-nit`: MMW iteration count / MMW 迭代次数
- `--mmw-eta`: MMW step size / MMW 步长
- `--max-slots`: slot cap of the feasibility solver / 可行性求解的最大时隙上限
- `--output-dir`: output directory for CSVs and plots / CSV 与图像输出目录
- `--use-gpu --gpu-id`: CUDA device selection / GPU 设备选择
- `--ble-channel-retries`: BLE channel resampling retries for unscheduled pairs / 未调度 BLE pair 的重选信道重试次数
- `--ble-channel-mode {single,per_ce}`: BLE channel mode / BLE 信道模式
- `--wifi-first-ble-scheduling`: enable WiFi-first solver and start-slot scheduling / 启用 WiFi-first 求解与起始时隙调度

## Outputs / 输出结果

中文：主脚本会生成以下主要输出。  
English: The main script generates the following primary outputs.

- `pair_parameters.csv`: pair-level parameters and schedule results / 每个 pair 的参数与调度结果
- `wifi_ble_schedule.csv`: slot-level scheduled pair summary / 时隙级调度汇总
- `unscheduled_pairs.csv`: unscheduled pairs / 未调度 pair 列表
- `schedule_plot_rows.csv`: plotting rows for the time-frequency schedule / 时频调度绘图中间结果
- `ble_ce_channel_events.csv`: BLE per-event channel occupancy in `per_ce` mode / `per_ce` 模式下 BLE 事件级信道占用
- `wifi_ble_schedule.png`: legacy-compatible schedule plot filename / 保留兼容性的调度图文件名
- `wifi_ble_schedule_overview.png`: full schedule overview / 全局调度总览图
- `wifi_ble_schedule_window_*.png`: windowed schedule plots / 分窗口调度图

中文：当启用 `--wifi-first-ble-scheduling` 时，`pair_parameters.csv` 还会额外包含 `effective_ble_channels`、`scheduled_ble_pairs` 和 `no_collision_probability` 三列。  
English: When `--wifi-first-ble-scheduling` is enabled, `pair_parameters.csv` additionally includes `effective_ble_channels`, `scheduled_ble_pairs`, and `no_collision_probability`.

中文：如果导出的 BLE 事件在同一 slot 上发生频谱重叠，当前总览图和窗口图会用单独颜色高亮重叠区域，便于快速识别 BLE-BLE collision segment。  
English: If exported BLE events overlap in frequency within the same slot, the overview and windowed plots now highlight the overlap region with a separate color so BLE-BLE collision segments can be identified quickly.

中文：宏周期导出与绘图采用循环回卷语义。若某个 WiFi/BLE 传输跨越宏周期边界，其占用会同时出现在末尾 slot 和回卷后的起始 slot 中，而不是在边界处被截断。  
English: Macrocycle export and plotting use cyclic wraparound semantics. If a WiFi/BLE transmission crosses the macrocycle boundary, its occupancy appears both at the tail of the cycle and at the wrapped head, rather than being truncated at the boundary.

## Code Structure / 代码结构

- [sim_src/env/env.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/env/env.py): environment generation, PHY abstraction, timing, conflict checks / 环境生成、物理层抽象、时序与冲突检查
- [sim_src/alg/binary_search_relaxation.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/alg/binary_search_relaxation.py): binary-search solver with `joint` and `wifi_first` strategies / 具有 `joint` 与 `wifi_first` 两种策略的二分求解器
- [sim_src/alg/mmw.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/alg/mmw.py): MMW feasibility routine / MMW 可行性检查
- [sim_script/pd_mmw_template_ap_stats.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py): main experiment script, export, and plotting / 主实验脚本、导出与绘图入口
- [sim_script/journal_version](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/journal_version): scripts closer to the original paper workflow / 更接近论文原始流程的脚本

## Focused Regression / 回归验证

中文：以下测试覆盖了当前扩展工作流中的关键行为，包括 BLE `per_ce`、逐链路 BLER、带宽相关噪声和 WiFi-first 调度。  
English: The following tests cover the key behaviors of the current extended workflow, including BLE `per_ce`, per-link BLER, bandwidth-dependent noise, and WiFi-first scheduling.

```bash
python -m pytest \
  sim_script/tests/test_wifi_first_binary_search_split.py \
  sim_script/tests/test_wifi_first_ble_channel_availability.py \
  sim_script/tests/test_wifi_first_macrocycle_assignment.py \
  sim_script/tests/test_pd_mmw_ble_channel_retry.py \
  sim_script/tests/test_pd_mmw_ble_channel_modes.py \
  sim_script/tests/test_bler_parameter_selection.py -v
```

## Citation / 引用

```bibtex
@article{gu2025sig,
  title={SIG-SDP: Sparse Interference Graph-Aided Semidefinite Programming for Large-Scale Wireless Time-Sensitive Networking},
  author={Gu, Zhouyou and Park, Jihong and Vucetic, Branka and Choi, Jinho},
  journal={arXiv preprint arXiv:2501.11307},
  year={2025}
}
```
