# Cross-Radio Task-Driven Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the current WiFi/BLE scheduler from “directly provide `wifi_tx_slots` / `ble_ci_slots` / `ble_ce_slots`” to a task-driven model where users mainly provide `packet_bytes`, `release_time_slot`, `deadline_slot`, and optionally a radio choice, while the system can derive timing parameters and, when requested, choose WiFi or BLE as the transmission medium.

**Architecture:** Build this in two layers. First, introduce a packet-size-driven timing derivation layer that converts task payload size into feasible WiFi and BLE timing candidates while preserving the existing downstream schedule/plot/export pipeline. Second, add a radio-selection layer that can generate both WiFi and BLE candidates for the same task, score them, and choose one medium before populating the existing env arrays. This keeps the macrocycle scheduler, BLE hopping backend, CSV exports, and plotting code reusable rather than rewriting the whole pipeline.

**Tech Stack:** Python, NumPy, existing `sim_src/env/env.py`, existing `sim_script/pd_mmw_template_ap_stats.py`, existing BLE hopping backend `ble_macrocycle_hopping_sdp.py`, `pytest`, JSON config files, CSV/plot pipeline.

---

### Task 1: Add a new task-oriented config vocabulary without breaking the old slot-based interface

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:1-180`
- Modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing schema tests**

Add tests that define the new user-facing fields:

```python
def test_manual_task_accepts_packet_bytes_and_radio_mode_fixed():
    config = merge_config_with_defaults({
        "pair_generation_mode": "manual",
        "pair_parameters": [{
            "pair_id": 0,
            "office_id": 0,
            "radio_mode": "fixed",
            "radio": "ble",
            "packet_bytes": 247,
            "release_time_slot": 0,
            "deadline_slot": 31,
            "start_time_slot": 0,
        }],
    })
    assert config["pair_parameters"][0]["radio_mode"] == "fixed"
```

```python
def test_manual_task_accepts_packet_bytes_and_radio_mode_auto():
    config = merge_config_with_defaults({
        "pair_generation_mode": "manual",
        "pair_parameters": [{
            "pair_id": 0,
            "office_id": 0,
            "radio_mode": "auto",
            "packet_bytes": 1200,
            "release_time_slot": 0,
            "deadline_slot": 63,
            "start_time_slot": 0,
        }],
    })
    assert config["pair_parameters"][0]["radio_mode"] == "auto"
```

Also add failure tests:
- missing `packet_bytes`
- missing `release_time_slot`
- missing `deadline_slot`
- `radio_mode="fixed"` but missing `radio`
- invalid `packet_bytes` for fixed BLE and fixed WiFi

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "radio_mode or packet_bytes"
```

Expected: FAIL because the schema still assumes direct slot fields and fixed `radio`.

**Step 3: Implement the minimal schema evolution**

In `_validate_pair_parameters(...)` and `merge_config_with_defaults(...)`, add support for:

```python
row["radio_mode"] in {"fixed", "auto"}
row["packet_bytes"]
row.get("radio")
row.get("timing_mode", "derive")
```

Validation rules:
- `radio_mode="fixed"` requires `radio in {"wifi", "ble"}`
- `radio_mode="auto"` must not require `radio`
- keep old `radio`-only rows valid by normalizing them to `radio_mode="fixed"`
- keep old direct-slot fields valid for backward compatibility

**Step 4: Update sample JSON comments**

In both JSON files, add comments such as:

```json
"_comment_radio_mode": "fixed=固定介质; auto=系统在 WiFi/BLE 中选择一种",
"_comment_packet_bytes": "WiFi 任务建议 512~1500B; BLE 单个 PDU 0~247B; 更大任务可拆成多个 BLE CE 或改选 WiFi"
```

**Step 5: Run tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "radio_mode or packet_bytes"
```

Expected: PASS.

**Step 6: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add task-oriented config schema for cross-radio scheduling"
```

### Task 2: Extract packet-size-driven timing derivation helpers for both radios

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing helper tests**

Add tests for explicit formulas:

```python
def test_derive_wifi_tx_slots_from_packet_bytes_is_monotonic():
    small = derive_wifi_tx_slots_from_packet_bytes(512, slot_time=1.25e-3, phy_rate_bps=20e6)
    large = derive_wifi_tx_slots_from_packet_bytes(1500, slot_time=1.25e-3, phy_rate_bps=20e6)
    assert small >= 1
    assert large >= small
```

```python
def test_derive_ble_ce_slots_from_packet_bytes_maps_247b_to_one_slot():
    assert derive_ble_ce_slots_from_packet_bytes(0, slot_time=1.25e-3) == 1
    assert derive_ble_ce_slots_from_packet_bytes(247, slot_time=1.25e-3) == 1
    assert derive_ble_ce_slots_from_packet_bytes(248, slot_time=1.25e-3) == 2
```

```python
def test_derive_ble_ci_slots_selects_smallest_feasible_discrete_candidate():
    ci = derive_ble_ci_slots_from_packet_bytes(
        packet_bytes=247,
        slot_time=1.25e-3,
        ble_ci_quanta_candidates=np.array([8, 16, 32, 64], dtype=int),
        ce_slots=1,
    )
    assert ci == 8
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "derive_wifi_tx_slots or derive_ble_ce_slots or derive_ble_ci_slots"
```

Expected: FAIL because the helpers are absent or incomplete.

**Step 3: Implement the helpers**

Add these functions to `sim_script/pd_mmw_template_ap_stats.py`:

```python
def derive_wifi_tx_slots_from_packet_bytes(packet_bytes, slot_time, phy_rate_bps):
    bits = max(0, int(packet_bytes)) * 8
    airtime_s = bits / float(phy_rate_bps)
    return max(1, int(math.ceil(airtime_s / float(slot_time))))


def derive_ble_ce_slots_from_packet_bytes(packet_bytes, slot_time, ble_payload_limit_bytes=247):
    payload_bytes = max(0, int(packet_bytes))
    pdu_count = max(1, int(math.ceil(payload_bytes / float(ble_payload_limit_bytes))))
    return pdu_count


def derive_ble_ci_slots_from_packet_bytes(packet_bytes, slot_time, ble_ci_quanta_candidates, ce_slots):
    candidates = np.asarray(ble_ci_quanta_candidates, dtype=int)
    feasible = candidates[candidates >= int(ce_slots)]
    if feasible.size == 0:
        return int(candidates.max())
    return int(feasible.min())
```

**Step 4: Run tests**

Run the same `pytest` command.

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add packet-size timing derivation helpers"
```

### Task 3: Introduce a task-to-candidate conversion layer

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write failing tests for candidate generation**

Add tests that define a normalized candidate interface:

```python
def test_build_task_candidates_for_fixed_wifi_returns_only_wifi_candidate():
    row = {
        "pair_id": 0,
        "radio_mode": "fixed",
        "radio": "wifi",
        "packet_bytes": 1200,
        "release_time_slot": 0,
        "deadline_slot": 31,
        "start_time_slot": 0,
        "office_id": 0,
    }
    candidates = build_task_candidates(row=row, slot_time=1.25e-3, ble_ci_quanta_candidates=np.array([8, 16, 32, 64]))
    assert {c["radio"] for c in candidates} == {"wifi"}
```

```python
def test_build_task_candidates_for_auto_returns_wifi_and_ble():
    row = {
        "pair_id": 0,
        "radio_mode": "auto",
        "packet_bytes": 200,
        "release_time_slot": 0,
        "deadline_slot": 31,
        "start_time_slot": 0,
        "office_id": 0,
    }
    candidates = build_task_candidates(...)
    assert {c["radio"] for c in candidates} == {"wifi", "ble"}
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "build_task_candidates"
```

Expected: FAIL because no candidate layer exists yet.

**Step 3: Implement a minimal candidate builder**

Add a helper like:

```python
def build_task_candidates(row, slot_time, ble_ci_quanta_candidates, wifi_phy_rate_bps=20e6):
    ...
    return [
        {
            "radio": "wifi",
            "packet_bytes": ...,
            "wifi_tx_slots": ...,
            "wifi_period_slots": ...,
            "wifi_anchor_slot": ...,
        },
        {
            "radio": "ble",
            "packet_bytes": ...,
            "ble_ce_slots": ...,
            "ble_ci_slots": ...,
            "ble_anchor_slot": ...,
        },
    ]
```

For this task, keep it simple:
- WiFi candidate uses current `wifi_period_slots` if explicitly supplied, otherwise derive a minimal feasible period from `tx_slots`
- BLE candidate uses derived `ce_slots` and derived smallest feasible `ci_slots`
- anchor defaults to `start_time_slot` unless explicitly provided

**Step 4: Run tests**

Run the same `pytest` command.

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add normalized task candidate builder for wifi and ble"
```

### Task 4: Implement a first-pass radio-selection policy before touching the main scheduler

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write failing tests for medium choice**

Add tests that define a minimal selection rule:

```python
def test_select_radio_candidate_prefers_ble_when_ble_fits_and_payload_is_small():
    chosen = select_radio_candidate([
        {"radio": "wifi", "packet_bytes": 200, "wifi_tx_slots": 1},
        {"radio": "ble", "packet_bytes": 200, "ble_ce_slots": 1, "ble_ci_slots": 8},
    ])
    assert chosen["radio"] == "ble"
```

```python
def test_select_radio_candidate_prefers_wifi_when_ble_requires_more_than_one_ce():
    chosen = select_radio_candidate([
        {"radio": "wifi", "packet_bytes": 1200, "wifi_tx_slots": 1},
        {"radio": "ble", "packet_bytes": 1200, "ble_ce_slots": 5, "ble_ci_slots": 8},
    ])
    assert chosen["radio"] == "wifi"
```

**Step 2: Run test to verify failure**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "select_radio_candidate"
```

Expected: FAIL because no selection policy exists yet.

**Step 3: Implement a minimal deterministic policy**

For the first version, use a simple rule:
- If `radio_mode="fixed"`, return that candidate
- If `radio_mode="auto"`:
  - prefer BLE when `packet_bytes <= 247` and BLE candidate is feasible within the deadline window
  - otherwise choose WiFi

Code shape:

```python
def select_radio_candidate(candidates, radio_mode, packet_bytes, release_time_slot, deadline_slot):
    if radio_mode == "fixed":
        return candidates[0]
    ble = next((c for c in candidates if c["radio"] == "ble"), None)
    wifi = next((c for c in candidates if c["radio"] == "wifi"), None)
    if ble is not None and int(packet_bytes) <= 247:
        return ble
    return wifi if wifi is not None else ble
```

This task is intentionally heuristic. Do not try to do global cross-task optimization yet.

**Step 4: Run tests**

Run the same `pytest` command.

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add first-pass radio selection heuristic"
```

### Task 5: Route manual JSON tasks through the candidate-and-selection layer

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:300-430`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write failing integration tests around `apply_manual_pair_parameters(...)`**

Add tests like:

```python
def test_apply_manual_pair_parameters_auto_radio_can_choose_ble():
    row = {
        "pair_id": 0,
        "office_id": 0,
        "radio_mode": "auto",
        "packet_bytes": 200,
        "release_time_slot": 0,
        "deadline_slot": 31,
        "start_time_slot": 0,
        "channel": 8,
    }
    ...
    assert e.pair_radio_type[0] == e.RADIO_BLE
```

```python
def test_apply_manual_pair_parameters_auto_radio_can_choose_wifi():
    row = {
        "pair_id": 0,
        "office_id": 0,
        "radio_mode": "auto",
        "packet_bytes": 1200,
        "release_time_slot": 0,
        "deadline_slot": 31,
        "start_time_slot": 0,
        "channel": 0,
    }
    ...
    assert e.pair_radio_type[0] == e.RADIO_WIFI
```

**Step 2: Run test to verify failure**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "auto_radio_can_choose"
```

Expected: FAIL because `apply_manual_pair_parameters(...)` still assumes one fixed radio path per row.

**Step 3: Implement the minimal integration**

Inside `apply_manual_pair_parameters(...)`:
- normalize each row
- build candidates using `build_task_candidates(...)`
- choose one candidate using `select_radio_candidate(...)`
- populate existing arrays:
  - `pair_radio_type`
  - `pair_wifi_*`
  - `pair_ble_*`
  - `pair_packet_bits`
  - `pair_bandwidth_hz`

Backward compatibility:
- if the row still supplies old explicit slot fields, continue honoring them
- if the row uses old `radio` without `radio_mode`, treat as `radio_mode="fixed"`

**Step 4: Run tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "apply_manual_pair_parameters"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: route manual tasks through cross-radio candidate selection"
```

### Task 6: Change random generation from “sample slots directly” to “sample packet size then derive timing”

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write failing tests for random-mode packet sampling**

Add tests:

```python
def test_random_env_wifi_packet_bytes_are_within_512_to_1500():
    e = env(cell_size=1, pair_density_per_m2=0.1, seed=1)
    wifi_ids = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    assert np.all(e.pair_packet_bytes[wifi_ids] >= 512)
    assert np.all(e.pair_packet_bytes[wifi_ids] <= 1500)
```

```python
def test_random_env_ble_packet_bytes_are_within_0_to_247():
    e = env(cell_size=1, pair_density_per_m2=0.1, seed=1)
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert np.all(e.pair_packet_bytes[ble_ids] >= 0)
    assert np.all(e.pair_packet_bytes[ble_ids] <= 247)
```

```python
def test_random_env_derived_slots_match_sampled_packet_bytes():
    e = env(cell_size=1, pair_density_per_m2=0.1, seed=1)
    ...
```

**Step 2: Run test to verify failure**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "random_env_wifi_packet_bytes or random_env_ble_packet_bytes or derived_slots_match_sampled_packet_bytes"
```

Expected: FAIL because the env still samples direct timing fields first.

**Step 3: Refactor random timing generation**

In `sim_src/env/env.py`:
- add `_sample_pair_packet_bytes()`
- sample:
  - WiFi bytes in `[512, 1500]`
  - BLE bytes in `[0, 247]`
- derive WiFi `tx_slots`
- derive BLE `ce_slots`
- derive BLE `ci_slots`
- keep release/deadline/anchor generation logic intact

Do not attempt auto cross-radio selection in random mode yet; keep the existing random radio split. This task only changes how timing is derived once a radio type is assigned.

**Step 4: Run tests**

Run the same `pytest` command plus:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "auto_ble_timing"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: derive random wifi and ble timing from sampled packet sizes"
```

### Task 7: Surface packet size and timing source in CSV outputs

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:990-1380`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write failing output-row tests**

Add tests asserting `compute_pair_parameter_rows(...)` or emitted CSV rows include:

```python
assert rows[0]["packet_bytes"] == 247
assert rows[0]["packet_bits"] == 1976
assert rows[0]["timing_source"] in {"manual_slots", "manual_packet_bytes", "auto_packet_bytes"}
assert rows[0]["radio_mode"] in {"fixed", "auto"}
assert rows[0]["selected_radio"] in {"wifi", "ble"}
```

**Step 2: Run test to verify failure**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "timing_source or selected_radio or packet_bits"
```

Expected: FAIL because the current output schema does not expose these fields.

**Step 3: Implement the row schema update**

Extend row building and CSV columns to include:
- `packet_bytes`
- `packet_bits`
- `radio_mode`
- `selected_radio`
- `timing_source`

If the task used the old slot-based fields, mark:
- `timing_source = "manual_slots"`

If it used payload-driven derivation:
- `timing_source = "manual_packet_bytes"` or `auto_packet_bytes`

**Step 4: Add a runtime smoke test**

In `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`, execute the script with a task-driven manual config and assert `pair_parameters.csv` contains the new columns.

**Step 5: Run tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "timing_source or selected_radio or packet_bytes"
```

Expected: PASS.

**Step 6: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: expose task-driven radio selection in outputs"
```

### Task 8: Add a second-phase plan hook for global cross-task medium optimization

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-03-20-packet-size-driven-radio-timing.md`

**Step 1: Document the boundary of phase 1**

Update README to state clearly:
- phase 1 does per-task candidate derivation + heuristic radio choice
- it does not yet solve a global mixed WiFi/BLE assignment optimization

**Step 2: Add a short “next step” section**

Describe the future phase 2:
- each task generates both WiFi and BLE candidate states
- a global optimizer chooses one state per task
- conflict/collision objective extends the current BLE-only and macrocycle scheduling logic

**Step 3: Update the earlier packet-size plan**

At the top or bottom of `docs/plans/2026-03-20-packet-size-driven-radio-timing.md`, add a note:

```markdown
Superseded in scope by `docs/plans/2026-03-20-cross-radio-task-driven-scheduling.md` for the cross-radio task-selection architecture.
```

**Step 4: Verify docs**

Run:

```bash
rg -n "radio_mode|selected_radio|phase 2|cross-radio" README.md docs/plans/2026-03-20-packet-size-driven-radio-timing.md
```

Expected: matching lines are present.

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-03-20-packet-size-driven-radio-timing.md
git commit -m "docs: define phase-1 and phase-2 cross-radio scheduling scope"
```

### Task 9: Final regression pass

**Files:**
- No code changes expected

**Step 1: Run focused logic tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q
```

Expected: PASS.

**Step 2: Run runtime tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q
```

Expected: PASS.

**Step 3: Run one random-mode end-to-end execution**

Run:

```bash
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json
```

Expected:
- command succeeds
- random mode samples packet size first
- derived timing fields appear in `pair_parameters.csv`

**Step 4: Run one manual-mode auto-radio execution**

Run:

```bash
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json
```

Expected:
- command succeeds
- task rows with `radio_mode="auto"` can choose WiFi or BLE
- derived timing and selected medium appear in outputs

**Step 5: Final commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_src/env/env.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json README.md docs/plans/2026-03-20-packet-size-driven-radio-timing.md
git commit -m "feat: add task-driven cross-radio scheduling inputs"
```
