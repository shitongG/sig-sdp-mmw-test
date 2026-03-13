# BLE Multi-Band Per-CE Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve the current BLE single-channel assignment path, and add an alternate implementation where the same BLE pair can transmit on different frequency bands across different connection events under the same CI.

**Architecture:** Keep the current `pair_channel`-based BLE path untouched as the default behavior. Add a second BLE timing/channel model that expands each BLE pair into per-CE transmission instances over the macrocycle, each CE carrying its own BLE channel choice while sharing the same pair-level CI, CE width, anchor, and priority. Thread this alternate representation through macrocycle occupancy, conflict checking, retry logic, CSV export, and plotting without breaking existing single-channel outputs.

**Tech Stack:** Python, NumPy, SciPy sparse matrices, pytest, existing `env` timing/channel model, current macrocycle scheduler and plotting pipeline.

---

### Task 1: Freeze current BLE single-channel behavior with regression tests

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Create: `sim_script/tests/test_ble_single_channel_regression.py`
- Test: `sim_script/tests/test_ble_single_channel_regression.py`

**Step 1: Write the failing test**

```python
def test_ble_default_mode_keeps_single_pair_channel():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=1, radio_prob=(0.0, 1.0))
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert ble_ids.size > 0
    assert e.pair_channel.shape[0] == e.n_pair
    assert not hasattr(e, "pair_ble_ce_channels") or e.pair_ble_ce_channels is None
```

**Step 2: Run test to verify it fails or is missing coverage**

Run: `pytest sim_script/tests/test_ble_single_channel_regression.py -v`
Expected: FAIL because the test file does not exist yet.

**Step 3: Write minimal implementation**

```python
# Add regression test only in this task. No production code change.
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_single_channel_regression.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_ble_single_channel_regression.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "test: lock BLE single-channel default behavior"
```

### Task 2: Add explicit BLE channel mode configuration and per-CE channel storage

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_ble_channel_mode_config.py`
- Create: `sim_script/tests/test_ble_channel_mode_config.py`

**Step 1: Write the failing test**

```python
def test_ble_multi_ce_mode_initializes_per_ce_channel_storage():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=2,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
    )
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert ble_ids.size > 0
    assert e.ble_channel_mode == "per_ce"
    assert isinstance(e.pair_ble_ce_channels, dict)
```
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_channel_mode_config.py::test_ble_multi_ce_mode_initializes_per_ce_channel_storage -v`
Expected: FAIL with unexpected keyword argument or missing attribute.

**Step 3: Write minimal implementation**

```python
class env:
    def __init__(..., ble_channel_mode="single", ...):
        if ble_channel_mode not in {"single", "per_ce"}:
            raise ValueError("ble_channel_mode must be 'single' or 'per_ce'.")
        self.ble_channel_mode = ble_channel_mode
        self.pair_ble_ce_channels = None
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_channel_mode_config.py::test_ble_multi_ce_mode_initializes_per_ce_channel_storage -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_ble_channel_mode_config.py
git commit -m "feat: add BLE channel mode configuration"
```

### Task 3: Generate per-CE BLE channel assignments while keeping pair-level CI/CE fixed

**Files:**
- Modify: `sim_src/env/env.py:375-425`
- Test: `sim_script/tests/test_ble_per_ce_channel_generation.py`
- Create: `sim_script/tests/test_ble_per_ce_channel_generation.py`

**Step 1: Write the failing test**

```python
def test_ble_per_ce_mode_assigns_channels_per_connection_event():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=3,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
    )
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    pair_id = int(ble_ids[0])
    ce_channels = e.pair_ble_ce_channels[pair_id]
    assert len(ce_channels) > 0
    assert all(0 <= ch < e.ble_channel_count for ch in ce_channels)
    assert e.pair_ble_ci_slots[pair_id] > 0
    assert e.pair_ble_ce_slots[pair_id] > 0
```
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_per_ce_channel_generation.py::test_ble_per_ce_mode_assigns_channels_per_connection_event -v`
Expected: FAIL because `pair_ble_ce_channels` is empty or missing.

**Step 3: Write minimal implementation**

```python
def _assign_ble_ce_channels(self, pair_id, macrocycle_slots):
    event_count = max(1, macrocycle_slots // self.pair_ble_ci_slots[pair_id])
    self.pair_ble_ce_channels[pair_id] = self.rand_gen_loc.integers(
        low=0,
        high=self.ble_channel_count,
        size=event_count,
        dtype=int,
    )
```

Add a call after CI/CE/anchor are fixed, without changing `pair_channel` for `ble_channel_mode="single"`.

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_per_ce_channel_generation.py::test_ble_per_ce_mode_assigns_channels_per_connection_event -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_ble_per_ce_channel_generation.py
git commit -m "feat: generate BLE channels per CE"
```

### Task 4: Expand macrocycle occupancy into CE instances with channel-aware frequency ranges

**Files:**
- Modify: `sim_src/env/env.py:427-519`
- Modify: `sim_script/pd_mmw_template_ap_stats.py:537-558`
- Test: `sim_script/tests/test_ble_per_ce_occupancy_expansion.py`
- Create: `sim_script/tests/test_ble_per_ce_occupancy_expansion.py`

**Step 1: Write the failing test**

```python
def test_ble_per_ce_mode_exposes_distinct_channel_ranges_across_events():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=4,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
    )
    pair_id = int(np.where(e.pair_radio_type == e.RADIO_BLE)[0][0])
    instances = e.expand_pair_event_instances(pair_id, macrocycle_slots=e.compute_macrocycle_slots())
    assert len(instances) > 0
    assert all("channel" in inst for inst in instances)
    assert all("slot_range" in inst for inst in instances)
```
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_per_ce_occupancy_expansion.py::test_ble_per_ce_mode_exposes_distinct_channel_ranges_across_events -v`
Expected: FAIL because `expand_pair_event_instances` does not exist.

**Step 3: Write minimal implementation**

```python
def expand_pair_event_instances(self, pair_id, macrocycle_slots):
    # Return a list of per-event dicts with start slot, end slot, channel, and frequency range.
```

Use existing `pair_channel` for WiFi and BLE single-channel mode, and `pair_ble_ce_channels` for BLE per-CE mode.

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_per_ce_occupancy_expansion.py::test_ble_per_ce_mode_exposes_distinct_channel_ranges_across_events -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_ble_per_ce_occupancy_expansion.py
git commit -m "feat: expand per-CE BLE occupancy instances"
```

### Task 5: Make macrocycle conflict checks channel-aware per CE event

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:82-161`
- Modify: `sim_src/env/env.py:541-617`
- Test: `sim_script/tests/test_ble_per_ce_conflict_checks.py`
- Create: `sim_script/tests/test_ble_per_ce_conflict_checks.py`

**Step 1: Write the failing test**

```python
def test_ble_per_ce_mode_allows_same_pair_to_use_different_channels_in_different_events():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=5,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
    )
    pair_id = int(np.where(e.pair_radio_type == e.RADIO_BLE)[0][0])
    e.pair_ble_ce_channels[pair_id] = np.array([1, 10, 20, 30], dtype=int)
    starts, macrocycle_slots, occ, unscheduled = assign_macrocycle_start_slots(
        e,
        preferred_slots=np.zeros(e.n_pair, dtype=int),
        allow_partial=True,
    )
    assert pair_id not in unscheduled
```
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_per_ce_conflict_checks.py::test_ble_per_ce_mode_allows_same_pair_to_use_different_channels_in_different_events -v`
Expected: FAIL because conflict logic still uses only `pair_channel`.

**Step 3: Write minimal implementation**

```python
# Refactor conflict evaluation to consume per-event channel/frequency data
# instead of only pair-level channel for BLE per-CE mode.
```

Preserve current pair-level conflict path when `ble_channel_mode == "single"`.

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_per_ce_conflict_checks.py::test_ble_per_ce_mode_allows_same_pair_to_use_different_channels_in_different_events -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_ble_per_ce_conflict_checks.py
git commit -m "feat: add per-CE BLE conflict evaluation"
```

### Task 6: Extend BLE retry logic to resample per-CE channels in alternate mode

**Files:**
- Modify: `sim_src/env/env.py:635-648`
- Modify: `sim_script/pd_mmw_template_ap_stats.py:358-390`
- Test: `sim_script/tests/test_ble_per_ce_channel_retry.py`
- Create: `sim_script/tests/test_ble_per_ce_channel_retry.py`

**Step 1: Write the failing test**

```python
def test_ble_retry_resamples_per_ce_channels_without_changing_ci_ce():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=6,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
    )
    pair_id = int(np.where(e.pair_radio_type == e.RADIO_BLE)[0][0])
    ci_before = int(e.pair_ble_ci_slots[pair_id])
    ce_before = int(e.pair_ble_ce_slots[pair_id])
    channels_before = e.pair_ble_ce_channels[pair_id].copy()
    e.resample_ble_channels(np.array([pair_id], dtype=int))
    assert int(e.pair_ble_ci_slots[pair_id]) == ci_before
    assert int(e.pair_ble_ce_slots[pair_id]) == ce_before
    assert not np.array_equal(e.pair_ble_ce_channels[pair_id], channels_before)
```
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_per_ce_channel_retry.py::test_ble_retry_resamples_per_ce_channels_without_changing_ci_ce -v`
Expected: FAIL because retry only rewrites pair-level `pair_channel`.

**Step 3: Write minimal implementation**

```python
def resample_ble_channels(self, pair_ids):
    if self.ble_channel_mode == "per_ce":
        # rewrite per-event BLE channels only
        return
    # existing single-channel logic unchanged
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_per_ce_channel_retry.py::test_ble_retry_resamples_per_ce_channels_without_changing_ci_ce -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_ble_per_ce_channel_retry.py
git commit -m "feat: retry BLE channels per CE"
```

### Task 7: Export and plot the alternate BLE per-CE band usage without breaking current outputs

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:420-558`
- Modify: `sim_script/pd_mmw_template_ap_stats.py:629-906`
- Test: `sim_script/tests/test_ble_per_ce_export_and_plot.py`
- Create: `sim_script/tests/test_ble_per_ce_export_and_plot.py`

**Step 1: Write the failing test**

```python
def test_ble_per_ce_mode_exports_event_level_channel_rows(tmp_path):
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=7,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
    )
    rows = build_schedule_plot_rows(
        build_pair_parameter_rows(
            pair_office_id=e.pair_office_id,
            pair_radio_type=e.pair_radio_type,
            pair_channel=e.pair_channel,
            pair_priority=e.pair_priority,
            schedule_start_slots=np.zeros(e.n_pair, dtype=int),
            occupied_slots=e.expand_pair_occupancy(np.zeros(e.n_pair, dtype=int), e.compute_macrocycle_slots()),
            slot_time=e.slot_time,
            pair_start_time_slot=e.pair_start_time_slot,
            pair_wifi_anchor_slot=e.pair_wifi_anchor_slot,
            pair_wifi_period_slots=e.pair_wifi_period_slots,
            pair_wifi_tx_slots=e.pair_wifi_tx_slots,
            pair_ble_anchor_slot=e.pair_ble_anchor_slot,
            pair_ble_ci_slots=e.pair_ble_ci_slots,
            pair_ble_ce_slots=e.pair_ble_ce_slots,
            pair_ble_ce_feasible=e.pair_ble_ce_feasible,
            macrocycle_slots=e.compute_macrocycle_slots(),
        ),
        get_pair_channel_ranges_mhz(e, np.arange(e.n_pair)),
    )
    assert len(rows) > 0
```
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_ble_per_ce_export_and_plot.py::test_ble_per_ce_mode_exports_event_level_channel_rows -v`
Expected: FAIL because exported rows only understand one channel per pair.

**Step 3: Write minimal implementation**

```python
# Extend exported rows with optional event-level BLE channel data.
# Keep existing CSV columns; add a second event-level CSV only when ble_channel_mode == "per_ce".
```

Add a new export such as `ble_ce_channel_events.csv` and ensure the schedule plot renders per-event channel rectangles.

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_ble_per_ce_export_and_plot.py::test_ble_per_ce_mode_exports_event_level_channel_rows -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_ble_per_ce_export_and_plot.py
git commit -m "feat: export and plot BLE per-CE band usage"
```

### Task 8: Add CLI plumbing and end-to-end verification for both BLE channel modes

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:699-760`
- Test: `sim_script/tests/test_pd_mmw_ble_channel_modes.py`
- Create: `sim_script/tests/test_pd_mmw_ble_channel_modes.py`
- Check: `docs/plans/2026-03-11-ble-multi-band-per-ce.md`

**Step 1: Write the failing test**

```python
def test_cli_accepts_ble_channel_mode_per_ce(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "sim_script/pd_mmw_template_ap_stats.py",
            "--cell-size", "1",
            "--pair-density", "0.05",
            "--ble-channel-mode", "per_ce",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert (tmp_path / "wifi_ble_schedule.png").exists()
```
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_ble_channel_modes.py::test_cli_accepts_ble_channel_mode_per_ce -v`
Expected: FAIL because CLI does not expose the mode.

**Step 3: Write minimal implementation**

```python
parser.add_argument(
    "--ble-channel-mode",
    choices=["single", "per_ce"],
    default="single",
    help="BLE 信道模式：每 pair 单信道或每 CE 单独信道。",
)
```

Thread the argument into `env(...)` and keep current behavior as the default.

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pd_mmw_ble_channel_modes.py::test_cli_accepts_ble_channel_mode_per_ce -v`
Expected: PASS

**Step 5: Run focused regression suite**

Run:
```bash
pytest \
  sim_script/tests/test_ble_single_channel_regression.py \
  sim_script/tests/test_ble_channel_mode_config.py \
  sim_script/tests/test_ble_per_ce_channel_generation.py \
  sim_script/tests/test_ble_per_ce_occupancy_expansion.py \
  sim_script/tests/test_ble_per_ce_conflict_checks.py \
  sim_script/tests/test_ble_per_ce_channel_retry.py \
  sim_script/tests/test_ble_per_ce_export_and_plot.py \
  sim_script/tests/test_pd_mmw_ble_channel_modes.py -q
```
Expected: PASS

**Step 6: Run script smoke test for both modes**

Run:
```bash
python sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --output-dir /tmp/ble_single_smoke
python sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --ble-channel-mode per_ce --output-dir /tmp/ble_per_ce_smoke
```
Expected:
- Both commands exit `0`
- Both output directories contain `pair_parameters.csv`, `wifi_ble_schedule.csv`, `wifi_ble_schedule.png`
- `per_ce` mode additionally contains `ble_ce_channel_events.csv`

**Step 7: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_ble_channel_modes.py docs/plans/2026-03-11-ble-multi-band-per-ce.md
git commit -m "feat: add BLE per-CE multi-band scheduling mode"
```
