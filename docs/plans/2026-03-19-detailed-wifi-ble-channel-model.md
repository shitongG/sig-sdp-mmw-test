# Detailed WiFi BLE Channel Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 `sim_src/env/env.py` 中的 WiFi/BLE 信道建模改成贴合当前应用场景的精细版本，并在最终调度结果图中明确标出未被占用的 BLE 广播信道。

**Architecture:** 先把频谱模型收紧到场景约束：WiFi 只允许固定 1/6/11 三个 20 MHz 信道，BLE 数据链路只允许 37 个数据信道，并用两段式中心频率映射显式排除 2402/2426/2480 MHz 三个广播信道。然后把这些频率映射统一接到冲突判定、BLE 可用信道容量计算和绘图逻辑里；最后扩展绘图层，在时频图中叠加“未被占用的 BLE 广播信道”标识。

**Tech Stack:** Python 3、`numpy`、`scipy`、`matplotlib`、现有 `env` 频谱/调度逻辑、`pytest`

---

### Task 1: 为新信道模型写失败测试

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Test: `sim_src/env/env.py`

**Step 1: 写 BLE 两段式数据信道中心频率的失败测试**

```python
def test_ble_data_channel_centers_follow_two_segment_mapping():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=1)
    assert e.get_ble_data_channel_center_mhz(0) == 2404.0
    assert e.get_ble_data_channel_center_mhz(10) == 2424.0
    assert e.get_ble_data_channel_center_mhz(11) == 2428.0
    assert e.get_ble_data_channel_center_mhz(36) == 2478.0
```

**Step 2: 写 BLE 广播信道常量的失败测试**

```python
def test_ble_advertising_channel_centers_are_reserved():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=1)
    assert e.ble_advertising_center_freq_mhz == [2402.0, 2426.0, 2480.0]
```

**Step 3: 写 WiFi pair 只能落到 1/6/11 的失败测试**

```python
def test_wifi_pairs_only_use_fixed_1_6_11_channels():
    e = env(cell_size=1, pair_density_per_m2=0.2, seed=3)
    wifi_ids = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    assert set(e.pair_channel[wifi_ids].tolist()).issubset({0, 5, 10})
```

**Step 4: 写 BLE 重采样仍然只在数据信道集合内的失败测试**

```python
def test_resample_ble_channels_stays_within_data_channel_indices():
    e = env(cell_size=1, pair_density_per_m2=0.2, seed=3)
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    e.resample_ble_channels(ble_ids)
    assert np.all((e.pair_channel[ble_ids] >= 0) & (e.pair_channel[ble_ids] <= 36))
```

**Step 5: 写运行测试，断言调度图会包含 BLE 广播信道标识**

```python
def test_schedule_plot_rows_include_idle_ble_advertising_channels(tmp_path):
    ...
    assert "BLE adv idle" in schedule_plot_rows_csv_text
```

**Step 6: 运行测试确认失败**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: FAIL，因为新频率映射/广播信道标识尚未实现

**Step 7: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "test: cover detailed wifi ble channel model"
```

### Task 2: 在 env 中显式建模 WiFi 固定信道和 BLE 广播/数据信道

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: 增加场景常量和频率表**

```python
self.wifi_fixed_channel_indices = np.array([0, 5, 10], dtype=int)
self.wifi_fixed_center_freq_mhz = np.array([2412.0, 2437.0, 2462.0], dtype=float)
self.ble_advertising_center_freq_mhz = [2402.0, 2426.0, 2480.0]
self.ble_data_channel_indices = np.arange(37, dtype=int)
```

**Step 2: 添加 BLE 数据信道中心频率查询函数**

```python
def get_ble_data_channel_center_mhz(self, ble_idx):
    ble_idx = int(ble_idx)
    if ble_idx < 0 or ble_idx >= 37:
        raise ValueError("BLE data channel index must be in [0, 36].")
    if ble_idx <= 10:
        return 2404.0 + 2.0 * ble_idx
    return 2428.0 + 2.0 * (ble_idx - 11)
```

**Step 3: 添加 WiFi 固定信道中心频率查询函数**

```python
def get_wifi_channel_center_mhz(self, wifi_idx):
    wifi_idx = int(wifi_idx)
    if wifi_idx not in {0, 5, 10}:
        raise ValueError("WiFi channel index must be one of {0, 5, 10}.")
    return 2412.0 + 5.0 * wifi_idx
```

**Step 4: 运行逻辑测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q`
Expected: 中心频率相关测试开始通过

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add explicit wifi and ble channel frequency model"
```

### Task 3: 把 pair 信道分配和重采样收紧到场景允许集合

**Files:**
- Modify: `sim_src/env/env.py:225-233`
- Modify: `sim_src/env/env.py:296-307`
- Modify: `sim_src/env/env.py:864-889`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: 修改 AP WiFi 固定信道分配**

```python
self.office_wifi_channel = self.wifi_fixed_channel_indices[office_ids % self.wifi_fixed_channel_indices.size]
```

**Step 2: 修改 WiFi pair 信道采样，只从 `{0,5,10}` 采样**

```python
self.pair_channel[wifi_mask] = self.rand_gen_loc.choice(
    self.wifi_fixed_channel_indices,
    size=int(np.sum(wifi_mask)),
)
```

**Step 3: 保持 BLE 数据链路采样只在 `0..36` 数据信道索引范围内**

```python
self.pair_channel[ble_mask] = self.rand_gen_loc.choice(
    self.ble_data_channel_indices,
    size=int(np.sum(ble_mask)),
)
```

**Step 4: 修改 `resample_ble_channels()` / `_assign_ble_ce_channels()` 也只从数据信道集合采样**

```python
new_channels = self.rand_gen_loc.choice(self.ble_data_channel_indices, size=prev.size)
```

**Step 5: 运行逻辑测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q`
Expected: WiFi 只落 1/6/11，BLE 重采样测试通过

**Step 6: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: restrict wifi and ble links to scenario channel sets"
```

### Task 4: 用新频率映射替换冲突判定和频带区间函数

**Files:**
- Modify: `sim_src/env/env.py:618-636`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: 改写 `_get_wifi_channel_range_hz()`**

```python
center = self.get_wifi_channel_center_mhz(wifi_idx) * 1e6
half_bw = 10e6
return center - half_bw, center + half_bw
```

**Step 2: 改写 `_get_ble_channel_range_hz()`**

```python
center = self.get_ble_data_channel_center_mhz(ble_idx) * 1e6
half_bw = 1e6
return center - half_bw, center + half_bw
```

**Step 3: 写一个 WiFi 1/6/11 不覆盖 BLE 广播信道语义的测试**

```python
def test_wifi_fixed_channels_do_not_cover_ble_advertising_centers():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=1)
    for wifi_idx in [0, 5, 10]:
        low, high = e._get_wifi_channel_range_hz(wifi_idx)
        for adv in e.ble_advertising_center_freq_mhz:
            assert not (low <= adv * 1e6 < high)
```

**Step 4: 运行逻辑测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q`
Expected: 新区间判定测试通过

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: align interference ranges with detailed channel model"
```

### Task 5: 在绘图数据中加入未占用 BLE 广播信道行

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 新增构造 BLE 广播信道绘图行的函数**

```python
def build_ble_advertising_plot_rows(macrocycle_slots, e: env):
    rows = []
    for center_mhz in e.ble_advertising_center_freq_mhz:
        rows.append(
            {
                "pair_id": -2,
                "radio": "ble_adv_idle",
                "channel": -1,
                "slot": 0,
                "slot_width": int(macrocycle_slots),
                "freq_low_mhz": center_mhz - 1.0,
                "freq_high_mhz": center_mhz + 1.0,
                "label": f"BLE adv idle {center_mhz:.0f} MHz",
            }
        )
    return rows
```

**Step 2: 把广播信道行并入 `schedule_plot_rows`**

```python
schedule_plot_rows = build_schedule_plot_rows(...)
schedule_plot_rows.extend(build_ble_advertising_plot_rows(macrocycle_slots, e))
```

**Step 3: 扩展 `render_schedule_plot()`，支持 `slot_width` 和新颜色**

```python
colors = {"wifi": "#0B6E4F", "ble": "#C84C09", "ble_adv_idle": "#D9D9D9"}
width = row.get("slot_width", 1.0)
```

**Step 4: 运行测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: 广播信道标识测试通过

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: show idle ble advertising channels in schedule plot"
```

### Task 6: 把广播信道标识写入 CSV 并验证最终图

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:1320-1355`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 确保 `schedule_plot_rows.csv` 包含新行的 `radio=ble_adv_idle`**

```python
write_rows_to_csv(
    ...,
    ["pair_id", "radio", "channel", "slot", "slot_width", "freq_low_mhz", "freq_high_mhz", "label"],
    schedule_plot_rows,
)
```

**Step 2: 写运行测试，断言 CSV 中包含广播信道标记**

```python
def test_schedule_plot_rows_csv_contains_idle_ble_advertising_channels(tmp_path):
    ...
    text = (tmp_path / "json_out" / "schedule_plot_rows.csv").read_text(encoding="utf-8")
    assert "ble_adv_idle" in text
    assert "2402" in text
    assert "2426" in text
    assert "2480" in text
```

**Step 3: 运行运行测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: CSV 与绘图相关测试通过

**Step 4: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "test: verify idle ble advertising channels are exported"
```

### Task 7: 全量验证并请求代码审查

**Files:**
- Modify: `sim_src/env/env.py`
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: 运行完整 AP stats 测试**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_smoke.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: 全部通过

**Step 2: 运行默认 JSON 配置生成结果图**

Run: `python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json`
Expected: 生成 CSV / PNG，且 `schedule_plot_rows.csv` 中有 `ble_adv_idle`

**Step 3: 使用 `requesting-code-review` 做完成前审查**

Review scope:
- BLE 两段式数据信道映射是否正确
- WiFi 是否严格限制到 1/6/11 三个信道
- 广播信道是否没有被数据链路占用
- 最终调度图和 CSV 是否包含未占用 BLE 广播信道标识

**Step 4: Commit**

```bash
git add sim_src/env/env.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: refine wifi ble channel model and highlight ble advertising channels"
```
