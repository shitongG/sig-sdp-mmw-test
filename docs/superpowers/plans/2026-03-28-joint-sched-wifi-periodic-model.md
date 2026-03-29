# Joint Scheduler WiFi Periodic Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `joint_sched/` model WiFi tasks as true periodic flows, consistent with the legacy heuristic/mainline experiment semantics, so joint scheduling and legacy scheduling can be compared on the same physical meaning.

**Architecture:** Keep all work isolated inside `.worktrees/joint-wifi-ble/joint_sched/`. Extend the joint task and candidate-state model so WiFi states carry periodic repetition semantics, then update the mainline CSV adapter to reconstruct those semantics from `pair_parameters.csv`, and finally rerun the faithful comparison path. Do not change mainline `sim_script/pd_mmw_template_ap_stats.py`; only make `joint_sched/` match it better.

**Tech Stack:** Python 3.10, NumPy, pandas, pytest, existing `joint_sched/` model/GA/HGA/plot pipeline, mainline `pair_parameters.csv` as input artifact.

---

## File Structure

- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_model.py`
  Add WiFi periodic repetition semantics to `WiFiPairConfig`, candidate expansion, and block generation.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_adapter.py`
  Reconstruct WiFi periodic event count and anchor semantics from mainline `pair_parameters.csv` rows.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_plot.py`
  Ensure exported rows correctly reflect repeated WiFi events when the same WiFi state expands to multiple periodic blocks.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py`
  Keep summary fields stable after the model change and expose the updated faithful comparison outputs.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py`
  Add direct coverage for periodic WiFi block expansion.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_adapter.py`
  Add coverage that the adapter reconstructs WiFi periodic behavior from mainline CSV rows.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_plot.py`
  Verify plotting/export still works with repeated WiFi blocks.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`
  Add a runner-level regression for faithful-mainline WiFi periodic semantics.
- Modify: `.worktrees/joint-wifi-ble/README.md`
  Update the `joint_sched` section so the documented math and semantics say WiFi is periodic, not single-shot.

This stays in one plan because the model change, adapter change, plot/export change, and faithful comparison all depend on the same semantic correction: WiFi candidate states must expand to repeated periodic occupancy blocks.

### Task 1: Extend the joint model so WiFi states are periodic

**Files:**
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_model.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_expand_wifi_candidate_blocks_repeats_periodically_across_macrocycle():
    payload = {
        "macrocycle_slots": 32,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1000,
                "release_slot": 0,
                "deadline_slot": 31,
                "preferred_channel": 0,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 8,
                "repetitions": 3,
                "max_offsets": 1,
            }
        ],
    }
    space = build_joint_candidate_states(payload)
    state = next(space.states[idx] for idx in space.pair_to_state_indices[0] if space.states[idx].medium == "wifi")
    blocks = expand_candidate_blocks(state)
    assert len(blocks) == 3
    assert [(block.slot_start, block.slot_end) for block in blocks] == [(0, 4), (8, 12), (16, 20)]
```

```python
def test_wifi_pair_feasibility_uses_repeated_events_not_single_block():
    left = JointCandidateState(
        state_id=0,
        pair_id=0,
        medium="wifi",
        offset=0,
        channel=0,
        width_slots=4,
        period_slots=8,
        num_events=3,
    )
    right = JointCandidateState(
        state_id=1,
        pair_id=1,
        medium="wifi",
        offset=8,
        channel=0,
        width_slots=4,
        period_slots=8,
        num_events=1,
    )
    assert state_pair_is_feasible(left, right) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py -q -k "periodically_across_macrocycle or repeated_events"`
Expected: FAIL because WiFi expansion currently emits only one block and `JointCandidateState` does not carry WiFi repetition count.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class WiFiPairConfig:
    pair_id: int
    payload_bytes: int
    release_slot: int
    deadline_slot: int
    tx_slots: int
    period_slots: int
    num_events: int
    channel_options: tuple[int, ...]
    max_offsets: int = DEFAULT_MAX_OFFSETS
```

```python
@dataclass(frozen=True)
class JointCandidateState:
    state_id: int
    pair_id: int
    medium: str
    offset: int
    channel: int | None = None
    period_slots: int | None = None
    width_slots: int | None = None
    num_events: int | None = None
    pattern_id: int | None = None
    ci_slots: int | None = None
    ce_slots: int | None = None
```

```python
def expand_wifi_candidate_blocks(state: JointCandidateState) -> list[ResourceBlock]:
    if state.channel is None or state.width_slots is None or state.period_slots is None or state.num_events is None:
        raise ValueError("WiFi state missing channel, width, period, or repetition count")
    center = WIFI_CHANNEL_TO_MHZ[state.channel]
    blocks = []
    for event_index in range(state.num_events):
        start = state.offset + event_index * state.period_slots
        blocks.append(
            ResourceBlock(
                state_id=state.state_id,
                pair_id=state.pair_id,
                medium="wifi",
                event_index=event_index,
                slot_start=start,
                slot_end=start + state.width_slots,
                freq_low_mhz=center - WIFI_BANDWIDTH_MHZ / 2.0,
                freq_high_mhz=center + WIFI_BANDWIDTH_MHZ / 2.0,
                label=f"wifi-{state.pair_id}-ev{event_index}",
            )
        )
    return blocks
```

Use `repetitions` from `JointTaskSpec` to populate `WiFiPairConfig.num_events`, and pass `num_events` into WiFi `JointCandidateState` objects.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py -q -k "periodically_across_macrocycle or repeated_events"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_model.py
git -C .worktrees/joint-wifi-ble commit -m "feat: model wifi candidate states as periodic flows"
```

### Task 2: Reconstruct WiFi periodic semantics from mainline `pair_parameters.csv`

**Files:**
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_adapter.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_adapter.py`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_adapter_recovers_wifi_periodic_repetitions_from_mainline_rows():
    rows = [
        {
            "pair_id": 0,
            "radio": "wifi",
            "channel": 0,
            "release_time_slot": 0,
            "deadline_slot": 31,
            "wifi_anchor_slot": 3,
            "wifi_period_slots": 16,
            "wifi_tx_slots": 5,
            "occupied_slots_in_macrocycle": "[3,4,5,6,7,19,20,21,22,23]",
        }
    ]
    tasks = build_joint_tasks_from_mainline_pair_parameter_rows(rows)
    assert tasks[0].wifi_period_slots == 16
    assert tasks[0].repetitions == 2
```

```python
def test_adapter_uses_wifi_anchor_slot_as_preferred_offset_reference():
    rows = [
        {
            "pair_id": 0,
            "radio": "wifi",
            "channel": 5,
            "release_time_slot": 0,
            "deadline_slot": 63,
            "wifi_anchor_slot": 11,
            "wifi_period_slots": 16,
            "wifi_tx_slots": 4,
            "occupied_slots_in_macrocycle": "[11,12,13,14,27,28,29,30]",
        }
    ]
    tasks = build_joint_tasks_from_mainline_pair_parameter_rows(rows)
    assert tasks[0].release_slot == 0
    assert tasks[0].deadline_slot == 63
    assert tasks[0].preferred_channel == 5
    assert tasks[0].repetitions == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_adapter.py -q -k "wifi_periodic_repetitions or wifi_anchor_slot"`
Expected: FAIL because the adapter currently treats WiFi as effectively single-shot and does not derive a faithful repetition count.

- [ ] **Step 3: Write minimal implementation**

```python
def _count_periodic_events_from_slots(slots: list[int], width_slots: int) -> int:
    if not slots:
        return 0
    event_count = 1
    for left, right in zip(slots, slots[1:]):
        if right != left + 1:
            event_count += 1
    return event_count
```

```python
if radio == "wifi":
    wifi_tx_slots = int(_parse_int(row.get("wifi_tx_slots"), 1) or 1)
    wifi_period_slots = int(_parse_int(row.get("wifi_period_slots"), wifi_tx_slots) or wifi_tx_slots)
    repetitions = _count_periodic_events_from_slots(occupied_slots, wifi_tx_slots)
    tasks.append(
        JointTaskSpec(
            task_id=pair_id,
            radio="wifi",
            payload_bytes=wifi_tx_slots * DEFAULT_WIFI_BYTES_PER_SLOT,
            release_slot=release_slot,
            deadline_slot=deadline_slot,
            preferred_channel=preferred_channel,
            repetitions=max(1, repetitions),
            wifi_tx_slots=wifi_tx_slots,
            wifi_period_slots=wifi_period_slots,
            max_offsets=_suggest_max_offsets(release_slot, deadline_slot, wifi_period_slots, wifi_tx_slots),
        )
    )
```

Use the mainline CSV as the source of truth for WiFi periodicity; do not fall back to `repetitions=1` when `occupied_slots_in_macrocycle` clearly contains multiple events.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_adapter.py -q -k "wifi_periodic_repetitions or wifi_anchor_slot"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_adapter.py joint_sched/tests/test_joint_wifi_ble_adapter.py
git -C .worktrees/joint-wifi-ble commit -m "feat: recover periodic wifi semantics from mainline pair csv"
```

### Task 3: Preserve plotting/export correctness for repeated WiFi events

**Files:**
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_plot.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_plot.py`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_plot.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_plot_payload_includes_multiple_rows_for_periodic_wifi_state(tmp_path):
    result = {
        "selected_states": [
            {
                "state_id": 0,
                "pair_id": 0,
                "medium": "wifi",
                "offset": 3,
                "channel": 0,
                "width_slots": 4,
                "period_slots": 16,
                "num_events": 2,
            }
        ],
        "blocks": [],
        "unscheduled_pair_ids": [],
    }
    payload = build_plot_payload(result, tasks=[{"task_id": 0, "radio": "wifi"}], macrocycle_slots=64)
    wifi_rows = [row for row in payload["schedule_plot_rows"] if row["medium"] == "wifi"]
    assert len(wifi_rows) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_plot.py -q -k periodic_wifi_state`
Expected: FAIL if plot payload/export still assumes a single WiFi block.

- [ ] **Step 3: Write minimal implementation**

Update the plotting/export code so it trusts `blocks` expanded from `expand_candidate_blocks(state)` rather than reconstructing WiFi occupancy from single-shot assumptions.

```python
for block in result["blocks"]:
    schedule_plot_rows.append(
        {
            "pair_id": block["pair_id"],
            "medium": block["medium"],
            "event_index": block["event_index"],
            "slot_start": block["slot_start"],
            "slot_end": block["slot_end"],
            "freq_low_mhz": block["freq_low_mhz"],
            "freq_high_mhz": block["freq_high_mhz"],
            ...,
        }
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_plot.py -q -k periodic_wifi_state`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_plot.py joint_sched/tests/test_joint_wifi_ble_plot.py
git -C .worktrees/joint-wifi-ble commit -m "fix: export periodic wifi events in joint plot artifacts"
```

### Task 4: Add a runner-level faithful regression for periodic WiFi semantics

**Files:**
- Modify: `.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing test**

```python
def test_runner_faithful_mainline_mode_keeps_periodic_wifi_rows(tmp_path):
    summary = run_joint_demo(
        config_path="sim_script/output_ga_wifi_reschedule/pair_parameters.csv",
        solver="ga",
        output_dir=tmp_path / "out",
    )
    rows = pd.read_csv(summary["schedule_csv_path"])
    wifi_rows = rows[rows["medium"] == "wifi"]
    assert wifi_rows.groupby("pair_id").size().max() > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q -k periodic_wifi_rows`
Expected: FAIL because faithful mode currently emits WiFi as single-shot.

- [ ] **Step 3: Write minimal implementation**

No new architecture here; use the model and adapter fixes from Tasks 1-3. If the runner still strips or rewrites rows, remove that logic so the exported schedule CSV preserves repeated WiFi events.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q -k periodic_wifi_rows`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_runner.py
git -C .worktrees/joint-wifi-ble commit -m "test: verify faithful runner preserves periodic wifi semantics"
```

### Task 5: Update README math and semantics to say WiFi is periodic

**Files:**
- Modify: `.worktrees/joint-wifi-ble/README.md`
- Test: manual markdown review

- [ ] **Step 1: Update the problem definition**

Replace any statement that implies WiFi is a single time-frequency block with periodic notation:

```math
\mathcal{B}(a) = \bigcup_{m=0}^{M_k-1} \{ [s + mp_k,\, s + mp_k + w_k) \times [f_c - 10,\, f_c + 10] \}
```

State explicitly that:
- WiFi pair in `joint_sched` is now treated as a periodic flow
- `p_k` is WiFi period
- `w_k` is WiFi transmission width
- `M_k` is the number of WiFi periodic events within the macrocycle or deadline window

- [ ] **Step 2: Update the comparison caveat**

Add a short note that earlier `joint_sched` comparisons were pessimistic because WiFi had been under-modeled as single-shot, and that the new faithful mode removes that semantic mismatch.

- [ ] **Step 3: Commit**

```bash
git -C .worktrees/joint-wifi-ble add README.md
git -C .worktrees/joint-wifi-ble commit -m "docs: describe periodic wifi semantics in joint scheduler"
```

### Task 6: Final regression and faithful rerun

**Files:**
- Modify: `.worktrees/joint-wifi-ble/README.md`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/`

- [ ] **Step 1: Run focused regression**

Run:

```bash
python -m pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py \
  .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_adapter.py \
  .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_plot.py \
  .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the faithful comparison again**

Run:

```bash
python .worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py \
  --config sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --solver ga \
  --output .worktrees/joint-wifi-ble/joint_sched/output_compare_ga_faithful_periodic

python .worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py \
  --config sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --solver hga \
  --output .worktrees/joint-wifi-ble/joint_sched/output_compare_hga_faithful_periodic
```

Expected:
- both commands exit `0`
- both outputs preserve repeated WiFi events in `schedule_plot_rows.csv`
- both outputs remain hard collision-free
- the results are now directly comparable to the legacy periodic-WiFi heuristic semantics

- [ ] **Step 3: Record the new fair-comparison note in README**

Add a short subsection with:
- periodic WiFi now aligned with legacy semantics
- where the faithful outputs are written
- why any remaining gap is now algorithmic rather than semantic

- [ ] **Step 4: Commit**

```bash
git -C .worktrees/joint-wifi-ble add README.md joint_sched/output_compare_ga_faithful_periodic joint_sched/output_compare_hga_faithful_periodic
git -C .worktrees/joint-wifi-ble commit -m "test: rerun faithful joint comparison with periodic wifi model"
```
