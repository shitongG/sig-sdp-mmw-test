# WiFi-First-Aware BLE Hopping SDP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When `wifi_first_ble_scheduling = true`, expand already scheduled WiFi pairs into external time-frequency interference blocks and make `ble_macrocycle_hopping_sdp.py` choose BLE hopping sequences that preferentially avoid WiFi-occupied resources during macrocycle expansion.

**Architecture:** Split the feature into two layers. First, extract scheduled WiFi occupancy into a normalized external-interference representation `(slot, freq_range)` at the main-script layer. Second, extend the BLE-only SDP solver so each BLE candidate state is penalized or forbidden when its CE-level blocks overlap those WiFi interference blocks in both time and frequency. Keep the existing BLE-BLE collision matrix and rounding flow intact, but add a WiFi-aware external cost term to candidate evaluation. This avoids rewriting the full scheduler while making the BLE backend aware of WiFi-first decisions.

**Tech Stack:** Python, NumPy, cvxpy, existing `ble_macrocycle_hopping_sdp.py`, existing `sim_script/pd_mmw_template_ap_stats.py`, existing `env.py`, `pytest`.

---

### Task 1: Freeze the WiFi-first requirement in tests

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

```python
def test_wifi_interference_block_overlapping_ble_event_has_positive_cost():
    block = ExternalInterferenceBlock(
        start_slot=4,
        end_slot=5,
        freq_low_mhz=2402.0,
        freq_high_mhz=2422.0,
        source_type="wifi",
        source_pair_id=1,
    )
    assert external_interference_cost_for_state(...) > 0.0
```

```python
def test_build_wifi_interference_blocks_from_schedule_expands_slot_and_frequency_ranges():
    blocks = build_wifi_interference_blocks_from_schedule(...)
    assert len(blocks) > 0
    assert all(block.source_type == "wifi" for block in blocks)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "external_interference"`
Expected: FAIL because no external interference model exists yet.

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "wifi_interference_blocks"`
Expected: FAIL because WiFi interference extraction does not exist yet.

**Step 3: Write minimal implementation**

Add placeholders:

```python
@dataclass(frozen=True)
class ExternalInterferenceBlock:
    start_slot: int
    end_slot: int
    freq_low_mhz: float
    freq_high_mhz: float
    source_type: str
    source_pair_id: int


def external_interference_cost_for_state(...):
    raise NotImplementedError


def build_wifi_interference_blocks_from_schedule(...):
    raise NotImplementedError
```

**Step 4: Run test to verify it fails**

Run the same `pytest` commands.
Expected: FAIL with `NotImplementedError` rather than import errors.

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py ble_macrocycle_hopping_sdp.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "test: define WiFi-first BLE interference interfaces"
```

### Task 2: Add an external interference cost model to BLE candidate evaluation

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py:267-340`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
def test_external_interference_cost_zero_when_only_time_but_not_frequency_overlaps():
    assert external_interference_cost_for_state(...) == 0.0
```

```python
def test_external_interference_cost_scales_with_time_overlap():
    assert external_interference_cost_for_state(...) > 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "external_interference_cost"`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
def frequency_overlap_mhz(low0, high0, low1, high1):
    return max(0.0, min(high0, high1) - max(low0, low1))


def external_interference_cost_for_state(
    state,
    cfg_dict,
    pattern_dict,
    num_channels,
    interference_blocks,
):
    blocks = build_event_blocks({state.pair_id: state}, cfg_dict, pattern_dict, num_channels)
    total = 0.0
    for block in blocks:
        for ext in interference_blocks:
            time_overlap = max(0, min(block.end_slot, ext.end_slot) - max(block.start_slot, ext.start_slot) + 1)
            freq_overlap = frequency_overlap_mhz(
                block.frequency_mhz - 1.0,
                block.frequency_mhz + 1.0,
                ext.freq_low_mhz,
                ext.freq_high_mhz,
            )
            total += float(time_overlap) * float(freq_overlap > 0.0)
    return total
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "external_interference_cost"`
Expected: PASS.

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: add external interference cost model for BLE candidates"
```

### Task 3: Fold WiFi interference cost into the BLE-only objective

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py:671-760`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
def test_rounding_prefers_state_with_lower_wifi_interference_when_ble_ble_cost_equal():
    result = solve_ble_hopping_schedule(..., external_interference_blocks=[...])
    assert result["selected"][0].pattern_id == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "lower_wifi_interference"`
Expected: FAIL because the solver objective only depends on `Omega`.

**Step 3: Write minimal implementation**

Extend the solver interface:

```python
def solve_ble_hopping_schedule(..., external_interference_blocks=None):
    ...
```

Build a per-state cost vector:

```python
candidate_external_cost = np.array([
    external_interference_cost_for_state(
        state,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=num_channels,
        interference_blocks=external_interference_blocks or [],
    )
    for state in states
], dtype=float)
```

Add it to the objective:

```python
objective_expr = cp.sum(cp.multiply(np.triu(Omega, k=1), Y))
objective_expr += cp.sum(cp.multiply(candidate_external_cost, cp.diag(Y)))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "lower_wifi_interference or external_interference"`
Expected: PASS.

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: add WiFi-aware external cost to BLE SDP objective"
```

### Task 4: Build WiFi interference blocks from already scheduled WiFi pairs

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:1113-1215`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

```python
def test_build_wifi_interference_blocks_from_schedule_uses_wifi_slot_and_frequency_ranges():
    blocks = build_wifi_interference_blocks_from_schedule(...)
    assert blocks[0].start_slot == 0
    assert blocks[0].freq_low_mhz == 2402.0
    assert blocks[0].freq_high_mhz == 2422.0
```

```python
def test_build_wifi_interference_blocks_ignores_unscheduled_or_non_wifi_pairs():
    assert build_wifi_interference_blocks_from_schedule(...) == []
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "wifi_interference_blocks"`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
def build_wifi_interference_blocks_from_schedule(e, pair_rows):
    blocks = []
    for row in pair_rows:
        if row["radio"] != "wifi" or int(row["schedule_slot"]) < 0:
            continue
        pair_id = int(row["pair_id"])
        low_mhz, high_mhz = e._get_pair_link_range_hz(pair_id)
        for slot in row["occupied_slots_in_macrocycle"]:
            blocks.append(
                ExternalInterferenceBlock(
                    start_slot=int(slot),
                    end_slot=int(slot),
                    freq_low_mhz=float(low_mhz / 1e6),
                    freq_high_mhz=float(high_mhz / 1e6),
                    source_type="wifi",
                    source_pair_id=pair_id,
                )
            )
    return blocks
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "wifi_interference_blocks"`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: extract scheduled WiFi occupancy into interference blocks"
```

### Task 5: Reorder the WiFi-first pipeline so BLE hopping can see scheduled WiFi occupancy

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:1500-1620`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

```python
def test_wifi_first_pipeline_solves_wifi_then_builds_wifi_interference_then_runs_ble_backend(monkeypatch):
    call_order = []
    ...
    assert call_order == [
        "solve_wifi_stage",
        "build_wifi_interference_blocks",
        "apply_ble_schedule_backend",
        "retry_ble_channels_and_assign_macrocycle",
    ]
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "wifi_first_pipeline"`
Expected: FAIL because `apply_ble_schedule_backend(e, config)` currently runs before the WiFi-first stage schedule is known.

**Step 3: Write minimal implementation**

Refactor the main flow:

```python
if not config["wifi_first_ble_scheduling"]:
    apply_ble_schedule_backend(e, config)
else:
    # 1. run WiFi-first MMW stage
    # 2. derive provisional scheduled WiFi rows
    # 3. build WiFi interference blocks
    # 4. rerun BLE backend with external_interference_blocks
    pass
```

Add optional plumbing:

```python
def solve_ble_hopping_for_env(e, config=None, external_interference_blocks=None):
    ...
```

```python
def apply_ble_schedule_backend(e, config, external_interference_blocks=None):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "wifi_first_pipeline or apply_ble_schedule_backend"`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: feed scheduled WiFi occupancy into BLE hopping backend"
```

### Task 6: Make the BLE backend prefer unoccupied WiFi-free blocks during WiFi-first runs

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `ble_macrocycle_hopping_sdp.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

```python
def test_wifi_first_ble_backend_avoids_wifi_occupied_blocks(tmp_path):
    ...
    assert not ble_rows_overlap_wifi_rows(...)
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "avoids_wifi_occupied_blocks"`
Expected: FAIL because BLE currently ignores WiFi occupancy during hopping selection.

**Step 3: Write minimal implementation**

Wire together:

```python
wifi_interference_blocks = build_wifi_interference_blocks_from_schedule(...)
result = solve_ble_hopping_for_env(
    e=e,
    config=config,
    external_interference_blocks=wifi_interference_blocks,
)
```

and ensure `solve_ble_hopping_schedule(...)` consumes the external blocks.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "avoids_wifi_occupied_blocks or macrocycle_hopping"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: make BLE hopping avoid WiFi-occupied time-frequency blocks"
```

### Task 7: Expose diagnostics in logs and outputs

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `README.md`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

```python
def test_wifi_first_ble_diagnostics_summary_contains_external_interference_stats():
    ...
```

Expected fields:
- `wifi_block_count`
- `candidate_external_cost_min`
- `candidate_external_cost_avg`
- `candidate_external_cost_max`

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "external_interference_stats"`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
print("BLE WiFi-aware interference summary:", {
    "wifi_block_count": ...,
    "candidate_external_cost_min": ...,
    "candidate_external_cost_avg": ...,
    "candidate_external_cost_max": ...,
})
```

Update `README.md` to explain that in WiFi-first mode the BLE hopping backend now avoids WiFi-occupied time-frequency blocks through an external interference term.

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "external_interference_stats"`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py README.md sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "docs: add WiFi-first aware BLE hopping diagnostics"
```

### Task 8: Final regression pass

**Files:**
- No code changes expected

**Step 1: Run BLE-only tests**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q`
Expected: PASS.

**Step 2: Run main-script logic tests**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q`
Expected: PASS.

**Step 3: Run runtime tests**

Run: `pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q`
Expected: PASS.

**Step 4: Run one WiFi-first end-to-end execution**

Run: `python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json`
Expected: command succeeds, logs mention WiFi interference block count, BLE backend runs with WiFi-aware external costs, and resulting `schedule_plot_rows.csv` shows fewer BLE events overlapping WiFi-occupied blocks than before.

**Step 5: Final commit**

```bash
git add ble_macrocycle_hopping_sdp.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py tests/test_ble_macrocycle_hopping_sdp.py README.md
git commit -m "feat: add WiFi-first aware BLE hopping backend"
```
