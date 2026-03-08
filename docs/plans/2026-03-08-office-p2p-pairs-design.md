# Office P2P Pair Scheduling Design

**Topic:** 将当前基于“单用户/单设备”的调度环境重构为“办公室场景下的 WiFi/BLE 通信对调度”。

## Goal

把现有仿真环境改成办公室确定性通信场景：

- 网格中的每个 cell 表示一个 `7m x 7m` 办公室
- AP 保留为办公室锚点/中央协调节点，只负责收集该办公室内的任务信息
- 调度对象从“单个用户”改成“通信对（pair）”
- WiFi pair 和 BLE pair 在同一办公室内只要信道不冲突，就允许同一时隙并行
- 保留业务优先级，并把优先级作用在 pair 层，而不是设备层

## Current Problem

现有模型默认：

- AP 是接入点，`Q_asso` 会把同一 AP 下的所有用户都做成互斥
- 调度对象是单个用户/设备，而不是 P2P 通信对
- WiFi/BLE 的频谱冲突虽然已经建模，但会被“同 AP 全互斥”覆盖

这与目标场景不符。办公室场景下，AP 不应该再强制同办公室所有链路互斥；真正决定同隙并发的是：

- 是否同一办公室
- 是否频谱重叠
- 是否满足 BLE 的 CI/CE 可用窗口
- 是否满足物理层干扰约束

## Recommended Architecture

采用“通信对为一等公民”的模型：

1. 环境直接生成 `pair`，而不是先生成设备再配对
2. 每个 pair 绑定到一个办公室/AP 锚点
3. pair 类型为 `WiFi` 或 `BLE`
4. 每个 pair 有自己的业务优先级、信道、链路位置和 QoS 参数
5. 所有优化器仍然以 “K 个调度对象” 运行，其中 `K = n_pair`

这样可以最大程度复用现有 `binary_search_relaxation + mmw + rounding` 主体，只需要重构 `env.py` 的状态生成。

## Pair Model

### WiFi Pair

- 必须成对出现
- 每个 WiFi pair 使用 13 个 WiFi 信道中的一个
- pair 的两个端点都位于同一办公室内
- 可以与同办公室内其他不重叠信道的 pair 同时占用一个时隙

### BLE Pair

- 进入调度的 BLE 任务对象也是 pair
- 每个 BLE pair 使用 37 个 BLE 信道中的一个
- BLE pair 拥有独立的 `CI/CE/anchor`
- 在 `anchor + m*CI` 展开的 CE 窗口内才允许发送

## AP / Office Semantics

AP 的新语义：

- 是办公室网格锚点
- 用来标识“任务属于哪个办公室”
- 不再自动产生“同 AP 下所有 pair 互斥”的约束

也就是说，当前 `Q_asso` 中“同 AP 全互斥”的部分应被移除，冲突约束改成显式的“频谱冲突 + 必要的业务规则”。

## Concurrency Rules

### Same Office

同一办公室内：

- `WiFi-WiFi`：如果两个 WiFi pair 的频谱重叠，则冲突；不重叠则可同隙并行
- `BLE-BLE`：如果两个 BLE pair 的频谱重叠，则冲突；不重叠则可同隙并行
- `WiFi-BLE`：如果频谱重叠，则冲突；不重叠则可同隙并行

### Different Offices

不同办公室之间：

- 仍按频谱重叠与链路干扰决定是否能同隙并行
- `S_gain` 继续只保留频谱重叠链路的干扰项

## Priority Model

保留业务优先级，但优先级归属从“用户”改成“pair”。

- 默认不设“WiFi 高于 BLE”或“BLE 高于 WiFi”的固定制式优先级
- 每个 pair 由 `prio_prob/prio_value` 获得一个业务优先级
- rounding 时仍按 `priority` 优先，再按 `||gX_k||` 打破并列
- 评估加权 BLER 继续按 pair 级优先级计算

## Data Model Changes

`env.py` 需要从用户级数组切到 pair 级数组，至少包括：

- `n_pair`
- `pair_radio_type`
- `pair_priority`
- `pair_room_id`
- `pair_channel`
- `pair_tx_locs`
- `pair_rx_locs`
- `pair_ble_anchor_slot`
- `pair_ble_ci_slots`
- `pair_ble_ce_slots`
- `pair_ble_ce_feasible`

现有 `user_*` 命名应逐步替换或兼容映射到 `pair_*`。

## State Generation Changes

`generate_S_Q_hmax()` 需要改为 pair 级：

- `S_gain`: pair 对 pair 的干扰矩阵
- `Q_conflict`: 由频谱重叠规则直接构建，不再包含“同 AP 全互斥”
- `h_max`: 每个 pair 的最大可容忍干扰

优化器接口本身不需要知道对象是用户还是通信对，只需要状态矩阵维度正确。

## Evaluation Changes

`evaluate_sinr/evaluate_bler/evaluate_weighted_bler` 需要改为基于 pair：

- 每个 pair 只有一个激活时隙标签 `z_vec[k]`
- BLE pair 仍受 CE 窗口掩码限制
- 同一时隙内所有被调度 pair 参与 SINR 干扰计算

## Script / Output Changes

实验脚本需要同步改成 pair 语义，重点输出：

- `n_pair`
- `n_wifi_pair`
- `n_ble_pair`
- 每个办公室内 WiFi pair 数、BLE pair 数
- 各办公室在最终分配结果中的时隙占用数
- BLE 的 `CI/CE` 摘要

## Risks

1. 现有代码大量变量仍以 `user` 命名，重构时容易出现语义混淆
2. 如果试图同时保留“设备层”和“pair层”，复杂度会显著上升
3. 评估函数与统计脚本必须同步改，否则输出会继续沿用旧语义

## Scope Control

本次只做以下范围：

- 办公室大小改为 `7m x 7m`
- 调度对象改为 pair
- 去掉同 AP 全互斥
- 保留并迁移业务优先级到 pair 层
- 保留 BLE 的 CI/CE/anchor 机制

本次不做：

- 多 hop
- 设备层详细 MAC 行为
- 每个 CE 内多次子事件展开
- 动态配对或跨办公室配对
