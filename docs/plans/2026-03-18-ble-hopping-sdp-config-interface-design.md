# BLE Hopping SDP Config Interface Design

## Goal

把 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py) 从“写死在 `build_demo_instance()` 里的示例参数”改成“通过 JSON 配置文件驱动运行”的脚本，并把所有可调参数用中文清晰标明。

## Why

当前脚本虽然能运行，但主要参数散落在 `build_demo_instance()` 里：
- `num_channels`
- `pair_configs`
- `pattern_dict`
- `pair_weight`
- 输出图路径和标题

这种写法适合快速原型，不适合重复做实验。每次修改参数都要改 Python 源码，且参数语义不够集中，不利于保存、比较、复现多组实验。

## Chosen Approach

采用“JSON 配置文件 + `--config` 命令行参数”的单一入口方案。

运行方式：

```bash
python ble_macrocycle_hopping_sdp.py
python ble_macrocycle_hopping_sdp.py --config ble_macrocycle_hopping_sdp_config.json
```

默认行为：
- 不传 `--config` 时，脚本读取一个仓库内置的默认 JSON 配置文件
- 传入 `--config` 时，读取用户指定 JSON

不做命令行逐项覆盖，保持第一版接口简单稳定。

## Configuration Shape

建议的 JSON 顶层字段如下：

```json
{
  "num_channels": 37,
  "hard_collision_threshold": null,
  "plot_title": "BLE Event Grid",
  "output_path": "ble_macrocycle_hopping_sdp_schedule.png",
  "pair_configs": [
    {
      "pair_id": 0,
      "release_time": 1,
      "deadline": 18,
      "connect_interval": 4,
      "event_duration": 1,
      "num_events": 4
    }
  ],
  "pattern_dict": {
    "0": [
      {
        "pattern_id": 0,
        "start_channel": 1,
        "hop_increment": 5
      }
    ]
  },
  "pair_weight": {
    "0-1": 1.0,
    "0-2": 1.2
  }
}
```

## Parameter Meanings

配置文件内需要在字段名旁边给出中文说明，至少覆盖以下参数：

- `num_channels`
  中文：物理信道总数

- `hard_collision_threshold`
  中文：硬碰撞阈值；若某对候选状态碰撞代价超过该阈值，则在 SDP 中直接禁止

- `plot_title`
  中文：输出时频调度图标题

- `output_path`
  中文：输出图片路径

- `pair_configs[*].pair_id`
  中文：BLE 链路编号

- `pair_configs[*].release_time`
  中文：最早允许开始时间

- `pair_configs[*].deadline`
  中文：最晚完成时间

- `pair_configs[*].connect_interval`
  中文：连接间隔

- `pair_configs[*].event_duration`
  中文：单个连接事件持续时间

- `pair_configs[*].num_events`
  中文：宏周期内连接事件数

- `pattern_dict[k][*].pattern_id`
  中文：候选跳频 pattern 编号

- `pattern_dict[k][*].start_channel`
  中文：跳频起始信道

- `pattern_dict[k][*].hop_increment`
  中文：跳频步长

- `pair_weight["i-j"]`
  中文：链路 `i` 与链路 `j` 之间碰撞代价权重

## File Changes

- 修改 [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py)
  - 增加 `argparse`
  - 增加 JSON 读取与解析逻辑
  - 增加配置校验与对象转换
  - 调整 `main()` 从配置对象读取参数

- 新增 `ble_macrocycle_hopping_sdp_config.json`
  - 仓库内置默认配置

- 新增或扩展测试 [tests/test_ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/tests/test_ble_macrocycle_hopping_sdp.py)
  - 覆盖 JSON 配置加载
  - 覆盖 `pair_weight` 键解析
  - 覆盖默认配置入口

## Error Handling

第一版只处理最关键的配置错误：
- JSON 文件不存在
- JSON 格式非法
- `pair_id` 重复
- `pattern_dict` 缺少某个 pair
- `pair_weight` 键格式非法，如不是 `"i-j"`
- 必填字段缺失

出错时直接抛出清晰的 `ValueError` 或 `FileNotFoundError`，不做复杂恢复逻辑。

## Testing Strategy

测试重点放在配置层，不重复验证已有 SDP 细节：
- 能从 JSON 构造 `PairConfig` / `HoppingPattern`
- 能正确解析 `pair_weight`
- 默认配置文件能跑通现有主流程
- 保持现有事件块展开、overlap、绘图 smoke test 继续通过

## Result

改完之后，用户只需要改 JSON 文件，不再需要改 Python 源码就能调整实验参数。脚本仍保持单文件入口，但从“写死 demo”升级成“可复现实验配置驱动”的接口。
