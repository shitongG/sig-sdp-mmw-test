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

## 2.3 BLE-only GA standalone backend

`ble_macrocycle_hopping_sdp.py` 现在还支持一个独立的遗传算法后端 `ga`。它保留了与 SDP 路径相同的 candidate-state 建模方式，但把“每个 BLE pair 选择哪个 `(offset, pattern)` 组合”改成了进化搜索，因此更适合高密度实例下的快速近似求解。

可直接运行的命令：

```bash
python ble_macrocycle_hopping_sdp.py --solver ga
python ble_macrocycle_hopping_sdp.py --config ble_macrocycle_hopping_sdp_config.json
```

如果你想显式控制求解器，也可以在 JSON 里设置：

```json
{
  "solver": "ga"
}
```

#### 2.3.1 编码方式

standalone GA 不会直接在全局状态索引上做搜索，而是先把候选状态按 BLE pair 分组。代码中的 `PairCandidateGroup` 表示：
- 一个 `pair_id`
- 该 pair 的局部候选状态列表

因此，染色体的定义是：
- 染色体长度等于 BLE pair 数量
- 第 `k` 个基因只在第 `k` 个 pair 的局部候选状态集合中取值
- 一个候选状态就是一个 `(offset, pattern)` 组合

这种编码有两个好处：
- 天然满足“每个 BLE pair 恰好选一个状态”的离散约束，不需要额外 repair
- 染色体空间按 pair 局部化后，更适合在高密度实例下快速采样和变异

#### 2.3.2 适应度定义

GA 的适应度是一个越小越好的代价函数：
- `fitness = BLE-BLE collision cost + external interference cost`

其中：
- `BLE-BLE collision cost` 复用 standalone 求解器里现成的碰撞统计函数
- `external interference cost` 用来刻画候选 BLE 轨迹与外部 WiFi 干扰块的重叠

也就是说，GA 优化的不是抽象分数，而是和 SDP 路径尽量一致的物理冲突代价。这样可以保证两种 backend 虽然求解方法不同，但目标语义是对齐的。

#### 2.3.3 进化算子

当前实现采用的是一组很克制、但足够稳定的基本算子：
- `population initialization`：每个基因都只在本 pair 的局部候选中随机采样
- `tournament selection`：从当前种群中抽取少量个体，选择适应度更好的染色体作为父代
- `single-point crossover`：对两个父代做单点交叉
- `mutation`：仅在当前 pair 的局部候选集合内重新采样，不会越界到别的 pair
- `elitism`：每代保留 `elite_count` 个最优染色体，避免当前最好解退化

实现里还会记录 `fitness_history`，所以后续如果你想画收敛曲线，数据已经在 GA 结果对象里了。

#### 2.3.4 为什么在高密度场景下比 SDP 更快

GA 更快的根本原因不是单次碰撞计算更简单，而是它避免了：
- 构造和求解 lifted SDP 松弛矩阵
- 在大候选状态空间上引入 `A x A` 规模的半正定变量
- `cvxpy + solver` 在高密度配置下的编译和收敛开销

因此它的定位是：
- `SDP`：更强的优化基线，适合 small/medium scale 和研究性对比
- `GA`：高密度 BLE 下更可运行的近似搜索器，允许牺牲全局最优性换取速度和规模能力

适合调参的字段有：
- `ga_population_size`：种群规模，越大越稳但越慢
- `ga_generations`：进化代数，越大越容易收敛但越慢
- `ga_mutation_rate`：变异概率，过低容易早熟，过高会破坏收敛
- `ga_crossover_rate`：交叉概率
- `ga_elite_count`：每代保留的精英个体数
- `ga_seed`：固定随机种子，便于复现实验

这个 backend 的定位是“standalone 快速近似求解器”，不是 SDP 的严格替代；它的价值在于高密度实例下能更快给出可用调度。

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
   这里有三条后端：
   - `legacy`
   - `macrocycle_hopping_sdp`
   - `macrocycle_hopping_ga`

### 3.2 `legacy` BLE 后端

`legacy` 是当前主脚本里保留的原有 BLE 调度方式。

- 如果 `ble_channel_mode = single`，每个 BLE pair 使用一个固定数据信道。
- 如果 `ble_channel_mode = per_ce`，每个 BLE 连接事件单独分配信道，最终写到 `pair_ble_ce_channels`。
- 该模式更贴近原始主脚本的事件展开和绘图链路，速度更稳定，适合大批量随机实验。

### 3.3 `macrocycle_hopping_sdp` BLE 后端

当 `ble_schedule_backend = macrocycle_hopping_sdp` 时，主脚本会调用 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 的 BLE-only 宏周期跳频调度器。

当 `wifi_first_ble_scheduling = true` 时，主脚本会先按 WiFi-first 规则得到已调度 WiFi pair 的 `(slot, freq range)` 占用，再把这些占用展开成外部干扰块 `ExternalInterferenceBlock`，送入 BLE-only hopping SDP。这样 BLE 候选 state 会优先选取未被已调度 WiFi 占用的时频资源块；如果某个 BLE pair 至少存在一个零 WiFi 重叠的 candidate state，那么所有与 WiFi 重叠的 candidate state 会被直接禁止。

下面把它按论文里的 Problem Formulation 风格整理。

#### 3.3.1 问题定义

设 BLE pair 集合为 $\mathcal{K}$，其中 $k \in \mathcal{K}$ 表示第 $k$ 个 BLE pair。  
对每个 pair，已知：

- $r_k$：release time
- $D_k$：deadline
- $\Delta_k$：connect interval
- $d_k$：单个 connection event, CE 的持续时长
- $M_k$：宏周期内 CE 数量
- $\mathcal{L}_k$：候选 hopping pattern 集

在当前实现中，一个候选 pattern $\ell \in \mathcal{L}_k$ 由：
- $c_{k,0}^{(\ell)}$：起始 data channel
- $h_k^{(\ell)}$：hop increment

共同描述。

我们的目标是：  
对每个 BLE pair 选择一个宏周期偏移 $s$ 和一个 hopping pattern $\ell$，使所有 pair 在宏周期内的总时频碰撞代价最小。

#### 3.3.2 符号与可行 offset 集

对每个 pair $k$，其可行 offset 集定义为：

```math
\mathcal{S}_k = \{ s \mid r_k \le s \le D_k - (M_k - 1)\Delta_k - d_k + 1 \}
```

该式保证最后一个 CE 仍能在 `deadline` 前结束。

给定 offset $s \in \mathcal{S}_k$，第 $m$ 个 CE 的开始时间为：

```math
t_{k,m}(s) = s + m \Delta_k, \quad m = 0,1,\dots,M_k-1
```

对应的时间占用区间为：

```math
I_{k,m}(s) = [t_{k,m}(s),\; t_{k,m}(s) + d_k - 1]
```

#### 3.3.3 Hopping 轨迹与 data channel 语义

当前程序采用一个简化但可运行的 hopping 规则：

```math
c_{k,m}^{(\ell)} = (c_{k,0}^{(\ell)} + h_k^{(\ell)} m) \bmod 37
```

这里的 `37` 对应 BLE 的 `37` 个 data channel，而不是包含广播信道的全集。

频率映射与主仿真环境保持一致：

- `0..10 -> 2404 + 2k MHz`
- `11..36 -> 2428 + 2(k-11) MHz`

因此：
- `2402 MHz`
- `2426 MHz`
- `2480 MHz`

这三个 BLE 广播信道不进入 `macrocycle_hopping_sdp` 的 hopping 空间，standalone BLE-only 图里也不会把它们画成可被 data CE 占用的资源块。

#### 3.3.4 Candidate state 定义

对每个 pair $k$，定义 candidate state：

```math
a = (k, s, \ell), \quad s \in \mathcal{S}_k,\; \ell \in \mathcal{L}_k
```

记第 $k$ 个 pair 的候选状态全集为：

```math
\mathcal{A}_k = \{ (k, s, \ell) \mid s \in \mathcal{S}_k,\; \ell \in \mathcal{L}_k \}
```

整个系统的候选状态总集合为：

```math
\mathcal{A} = \bigcup_{k \in \mathcal{K}} \mathcal{A}_k
```

程序里的 `CandidateState(pair_id, offset, pattern_id)` 就是该定义的离散实现。

#### 3.3.5 碰撞代价矩阵 $\Omega$

对任意两个 candidate state

```math
a = (k, s, \ell), \quad b = (j, s', \ell')
```

定义 CE 级碰撞代价为：

```math
\Omega_{ab}
= \sum_{m=0}^{M_k-1} \sum_{n=0}^{M_j-1}
  w_{k,j}\;
  \mathbf{1}\{ c_{k,m}^{(\ell)} = c_{j,n}^{(\ell')} \}\;
  | I_{k,m}(s) \cap I_{j,n}(s') |
```

其中：
- `w_{k,j}` 是 pair 间权重
- $\mathbf{1}\{\cdot\}$ 是同信道指示函数
- $| I_{k,m}(s) \cap I_{j,n}(s') |$ 是两个闭区间的重叠长度

代码中的 $\Omega$ 就是对所有 candidate state 两两预计算得到的碰撞矩阵。

若 WiFi 已先完成调度，还会额外构造外部干扰块集合 $\mathcal{W}$。对任一 candidate state $a$，定义其 WiFi 外部干扰代价：

```math
\Gamma_a = \sum_{b \in \mathcal{W}} \mathbf{1}\{ F(a) \cap F(b) \neq \varnothing \}\; | T(a) \cap T(b) |
```

其中 $T(a)$ 和 $F(a)$ 分别表示 candidate state $a$ 在宏周期展开后的时间占用与频谱占用。当前实现里，$\Gamma_a$ 会以对角代价的形式并入 SDP 目标；并且只要某个 BLE pair 仍存在 $\Gamma_a = 0$ 的候选状态，就会把同一 pair 内所有 $\Gamma_a > 0$ 的候选状态直接禁止。

#### 3.3.6 离散优化与 lifted SDP 松弛

如果直接做离散选择，可以写成：

```math
\min \sum_{a < b} \Omega_{ab} y_a y_b
```

其中 $y_a \in \{0,1\}$ 表示是否选择 candidate state $a$。

约束为每个 pair 必须且只能选一个状态：

```math
\sum_{a \in \mathcal{A}_k} y_a = 1, \quad \forall k \in \mathcal{K}
```

程序中采用的是 lifted SDP 松弛，令 $Y$ 为对称矩阵变量，并最小化：

```math
\min \sum_{a < b} \Omega_{ab} Y_{ab} + \sum_a \Gamma_a Y_{aa}
```

满足：

```math
\sum_{a \in \mathcal{A}_k} Y_{aa} = 1, \quad \forall k \in \mathcal{K}
```

```math
Y_{ab} = 0, \quad \forall a \ne b,\; a,b \in \mathcal{A}_k
```

```math
Y \succeq 0
```

若设置硬碰撞阈值 $\eta$，还可加：

```math
\Omega_{ab} > \eta \Rightarrow Y_{ab} = 0
```

这正是 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 里 `build_sdp_relaxation()` 的数学含义。

#### 3.3.7 Rounding 与实现假设

SDP 解得到的是松弛矩阵 $Y$，当前实现使用 $\mathrm{diag}(Y)$ 做简单 rounding：

- 对每个 pair $k$
- 在 $`\mathcal{A}_k`$ 中选择 $`Y_{aa}`$ 最大的 candidate state $`a`$

它不是最强的 rounding 方法，但足够适合当前的 prototype / experimental workflow。

实现层面还需要注意两点：

1. 当前 BLE-only 脚本中的 hopping 规则是实验性简化模型，不是 BLE Core Spec 的完整 channel selection algorithm。  
2. 其频谱语义已经和主仿真脚本对齐：只在 37 个 BLE data channel 上 hopping，不触碰 BLE 广播信道。

对应的程序步骤可以概括成：

1. 对每个 BLE pair 构造 candidate state `(pair_id, offset, pattern_id)`
2. 预计算碰撞矩阵 $\Omega$
3. 解 lifted SDP 松弛
4. 用 $\mathrm{diag}(Y)$ 做 rounding
5. 把选中状态展开成事件级时频块，并输出图像

这个后端更适合：
- 研究 BLE-only hopping 规则
- 对比不同 BLE pattern 的碰撞代价
- 做 small/medium scale 的宏周期跳频实验

这个后端不适合：
- 不加约束直接跑很宽的 BLE 时间窗口
- 候选 offset 和 pattern 过多的超大实例

### 3.4 `macrocycle_hopping_ga` BLE 后端

当 `ble_schedule_backend = macrocycle_hopping_ga` 时，主脚本会复用与 `macrocycle_hopping_sdp` 相同的 BLE 输入建模：
- 从 `env` 中抽取每个 BLE pair 的 `release_time / deadline / ci / ce`
- 构造相同的 candidate state `(pair_id, offset, pattern_id)`
- 在同样的 `37` 个 BLE data channel 语义下展开事件级资源块

差别只在求解器本身：
- `macrocycle_hopping_sdp` 调用 SDP 松弛 + rounding
- `macrocycle_hopping_ga` 调用 [ble_macrocycle_hopping_ga.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_ga.py) 中的遗传算法搜索器

#### 3.4.1 主脚本里的 GA 设计

主脚本接入 GA 后端时，仍然保持了现有调度链路的边界不变：
1. 先由主脚本/环境生成 BLE pair 的时间窗口和时序参数
2. 调用 GA backend 选出每个 BLE pair 的 `(offset, pattern)`
3. 把结果写回 `pair_ble_ce_channels`
4. 复用现有 CSV、冲突判定和绘图链路输出最终调度结果

这意味着 `macrocycle_hopping_ga` 是一个真正的“可替换 BLE backend”，而不是旁路实验脚本。

#### 3.4.2 与 WiFi-first 的关系

当 `wifi_first_ble_scheduling = true` 时，主脚本会先把已调度 WiFi 的 `(slot, freq range)` 展开成 `ExternalInterferenceBlock`，然后把这些干扰块传给 GA backend。这样 GA 的 fitness 会优先惩罚与已占用 WiFi 时频块重叠的 BLE 候选状态。

因此，GA 并不是盲目搜索 BLE hopping，而是在已有 WiFi 占用背景下做近似避让。

#### 3.4.3 适用场景

`macrocycle_hopping_ga` 更适合：
- 高密度 BLE 场景
- 候选 offset 和 pattern 数量较多，SDP 很慢的场景
- 需要快速做参数扫描、重复试验和大规模近似比较的场景

它不适合：
- 把“最优性证明”作为第一优先级的实验
- 需要直接把 SDP 目标值当作理论对照的分析

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

如果要直接读取 standalone BLE-only JSON 配置，例如运行 `50` 对 BLE pair：

```bash
python ble_macrocycle_hopping_sdp.py \
  --config ble_macrocycle_hopping_sdp_config.json
```

说明：
- 不带 `--config` 时，使用脚本内置的 `4` 对 demo
- 带 `--config` 时，会读取 `ble_macrocycle_hopping_sdp_config.json` 里的 `pair_configs`、`pattern_dict`、`pair_weight`、`output_path`
- standalone 图也会额外标出 `BLE adv idle`，表示 `2402 / 2426 / 2480 MHz` 广播信道未被 data hopping 占用

## 5. 关键配置文件

### 4.1 默认随机配置

- `sim_script/pd_mmw_template_ap_stats_config.json`

用途：
- 随机生成用户对
- 当前文件默认 BLE backend 为 `macrocycle_hopping_sdp`
- 这是一个高密度随机实验入口，不是轻量 smoke 配置

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

### 4.6 BLE-only standalone 50 对配置

- `ble_macrocycle_hopping_sdp_config.json`

用途：
- 直接给 `ble_macrocycle_hopping_sdp.py` 读取
- 当前文件预置 `50` 对 BLE pair
- 输出 BLE-only 宏周期 hopping 的事件级块表和调度图

## 6. 可以改哪些参数

### 6.1 随机模式

在 `pd_mmw_template_ap_stats_config.json` 或 CLI 里常改：

- `cell_size`
- `pair_density`
- `seed`
- `mmw_nit`
- `mmw_eta`
- `max_slots`
- `ble_channel_mode`
- `ble_schedule_backend`
- `ble_max_offsets_per_pair`
- `ble_log_candidate_summary`
- `ble_channel_retries`
- `wifi_first_ble_scheduling`
- `output_dir`

### 6.2 manual 模式

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
- `ble_max_offsets_per_pair` 可以对每个 BLE pair 的可行 offset 做确定性剪枝，直接减小 `|A|`
- `ble_log_candidate_summary = true` 时，BLE backend 会在求解前打印每个 pair 的 `offset_count`、`pattern_count` 和全局 `state_count`

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

同时，主脚本的 BLE backend 现在支持候选状态剪枝：
- `ble_max_offsets_per_pair` 控制每个 pair 最多保留多少个可行 offset
- `ble_log_candidate_summary` 用来输出候选空间摘要，帮助定位是哪几个 pair 把 `|A|` 撑大了

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

当前代码已经提供两个直接可用的手段：
- `ble_max_offsets_per_pair`
  - 对每个 BLE pair 的可行 offset 做可复现剪枝
  - 剪枝规则是保留首尾，并对中间 offset 做等间距采样
- `ble_log_candidate_summary`
  - 在求解前打印候选空间摘要
  - 能直接看到 `pair_count`、全局 `state_count`，以及每对的 `offset_count/pattern_count/state_count`

因此要明确：
- 向量化主要解决的是“CVXPY 建模/编译阶段太慢”
- 它不会把一个本来就很大的 SDP 问题变成小问题
- 如果你使用 [sim_script/pd_mmw_template_ap_stats_config.json](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats_config.json) 这种高密度随机配置，即使没有子表达式告警，求解本身仍可能很慢
- 现在这份高密度配置已经默认开启 `ble_max_offsets_per_pair` 和 candidate summary 日志，但如果 `state_count` 仍然很大，瓶颈会继续落在 SDP 求解器本身

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

如果你要直接测试高密度随机 BLE：

```bash
python sim_script/pd_mmw_template_ap_stats.py \
  --config sim_script/pd_mmw_template_ap_stats_config.json
```

需要注意：
- 这份配置会随机生成较多 BLE pair
- 默认就启用 `macrocycle_hopping_sdp`
- 默认也启用 `ble_log_candidate_summary`
- 默认会把 `ble_max_offsets_per_pair` 限制到较小上限，目前是 `2`
- 现在通常不会再出现 `too many subexpressions` 告警
- 但仍可能因为候选状态总数大而运行较久，甚至超过几分钟

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
