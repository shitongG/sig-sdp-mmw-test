# Joint Scheduler Mainline Alignment And Residual Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `joint_sched/` solve the same effective instance as the mainline WiFi/BLE experiment and improve the unified joint HGA so it can exploit residual time-frequency holes without falling back to sequential WiFi-first/BLE-second scheduling.

**Architecture:** Keep all work isolated inside `.worktrees/joint-wifi-ble/joint_sched/`. First replace the simplified random-task generator with a faithful adapter from mainline pair parameters or `env` state. Then expand the unified candidate space and improve the unified HGA with residual-hole-aware seeding, mutation, and local repair that still operate on one joint chromosome over a single mixed candidate-state space.

**Tech Stack:** Python 3.10, NumPy, pandas, pytest, existing `joint_sched/` model/GA/HGA/plot pipeline, mainline `sim_script/pd_mmw_template_ap_stats.py` and `sim_src/env/env.py` as reference only.

---

## File Structure

- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_random.py`
  Replace the current simplified random-task generator with adapters that can ingest real mainline pair parameters and build equivalent joint tasks.
- Create: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_mainline_adapter.py`
  New focused module that converts either a mainline config + generated `env`, or an existing `pair_parameters.csv`, into faithful `JointTaskSpec` payloads.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_model.py`
  Expand candidate generation so WiFi/BLE tasks no longer collapse to `max_offsets=1`, `pattern_count=1`, or overly truncated BLE repetition counts unless explicitly requested.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga_model.py`
  Add residual-spectrum scoring helpers for unified joint local search.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga.py`
  Upgrade HGA from blocker-guided reseeding to residual-hole-aware unified search.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py`
  Add explicit switches for “simplified random” vs “mainline faithful” task generation and write richer summary fields.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_random.py`
  Replace tests that depend on the simplified random abstraction.
- Create: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_mainline_adapter.py`
  New tests for faithful instance conversion.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py`
  Add coverage for wider candidate spaces.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py`
  Add residual-hole scoring tests.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py`
  Add behavioral tests showing the unified HGA uses residual holes while preserving unified scheduling.
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`
  Add runner coverage for faithful-mainline mode.
- Modify: `.worktrees/joint-wifi-ble/README.md`
  Update the `joint_sched` section to document the mainline-faithful adapter and residual-packing HGA.

This stays in one plan because the comparison problem and the heuristic problem are coupled: without faithful instance alignment, “better utilization” is not measurable; without stronger joint heuristics, the aligned model still underperforms the legacy packing behavior.

### Task 1: Add a faithful mainline-to-joint task adapter

**Files:**
- Create: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_mainline_adapter.py`
- Create: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_mainline_adapter.py`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_mainline_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
from joint_sched.joint_wifi_ble_mainline_adapter import joint_tasks_from_pair_parameter_rows


def test_adapter_maps_wifi_and_ble_rows_to_joint_tasks():
    rows = [
        {
            "pair_id": 0,
            "radio": "wifi",
            "channel": 0,
            "release_time_slot": 0,
            "deadline_slot": 31,
            "wifi_tx_slots": 5,
            "wifi_period_slots": 16,
        },
        {
            "pair_id": 1,
            "radio": "ble",
            "channel": 8,
            "release_time_slot": 2,
            "deadline_slot": 40,
            "ble_ce_slots": 2,
            "ble_ci_slots": 16,
            "ble_anchor_slot": 2,
        },
    ]
    tasks = joint_tasks_from_pair_parameter_rows(rows)
    assert len(tasks) == 2
    assert tasks[0].radio == "wifi"
    assert tasks[0].wifi_tx_slots == 5
    assert tasks[1].radio == "ble"
    assert tasks[1].ble_ce_slots == 2
    assert tasks[1].ble_ci_slots_options == (16,)
```

```python
from joint_sched.joint_wifi_ble_mainline_adapter import joint_tasks_from_pair_parameter_rows


def test_adapter_preserves_mainline_repetition_semantics():
    rows = [
        {
            "pair_id": 3,
            "radio": "ble",
            "channel": 10,
            "release_time_slot": 0,
            "deadline_slot": 63,
            "ble_ce_slots": 1,
            "ble_ci_slots": 8,
            "ble_anchor_slot": 0,
        }
    ]
    tasks = joint_tasks_from_pair_parameter_rows(rows, macrocycle_slots=64)
    assert tasks[0].repetitions == 8
    assert tasks[0].ble_num_events == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_mainline_adapter.py -q`
Expected: FAIL because the adapter module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from typing import Iterable, Mapping

from .joint_wifi_ble_model import JointTaskSpec


def _ble_num_events(deadline_slot: int, release_slot: int, ci_slots: int, ce_slots: int, macrocycle_slots: int) -> int:
    del macrocycle_slots
    usable = max(ce_slots, deadline_slot - release_slot + 1)
    return max(1, 1 + max(0, usable - ce_slots) // max(1, ci_slots))


def joint_tasks_from_pair_parameter_rows(rows: Iterable[Mapping[str, object]], macrocycle_slots: int = 64) -> list[JointTaskSpec]:
    tasks: list[JointTaskSpec] = []
    for row in rows:
        radio = str(row["radio"])
        pair_id = int(row["pair_id"])
        release_slot = int(row["release_time_slot"])
        deadline_slot = int(row["deadline_slot"])
        channel = int(row["channel"])
        if radio == "wifi":
            tx_slots = int(row["wifi_tx_slots"])
            period_slots = int(row["wifi_period_slots"])
            payload_bytes = int(row.get("payload_bytes", max(512, tx_slots * 750)))
            tasks.append(
                JointTaskSpec(
                    task_id=pair_id,
                    radio="wifi",
                    payload_bytes=payload_bytes,
                    release_slot=release_slot,
                    deadline_slot=deadline_slot,
                    preferred_channel=channel,
                    repetitions=max(1, 1 + max(0, macrocycle_slots - tx_slots) // max(1, period_slots)),
                    wifi_tx_slots=tx_slots,
                    wifi_period_slots=period_slots,
                    max_offsets=4,
                )
            )
        else:
            ce_slots = int(row["ble_ce_slots"])
            ci_slots = int(row["ble_ci_slots"])
            payload_bytes = int(row.get("payload_bytes", max(1, min(247, ce_slots * 247))))
            num_events = _ble_num_events(deadline_slot, release_slot, ci_slots, ce_slots, macrocycle_slots)
            tasks.append(
                JointTaskSpec(
                    task_id=pair_id,
                    radio="ble",
                    payload_bytes=payload_bytes,
                    release_slot=release_slot,
                    deadline_slot=deadline_slot,
                    preferred_channel=channel,
                    repetitions=num_events,
                    ble_ce_slots=ce_slots,
                    ble_ci_slots_options=(ci_slots,),
                    ble_num_events=num_events,
                    ble_pattern_count=3,
                    max_offsets=4,
                )
            )
    return tasks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_mainline_adapter.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_mainline_adapter.py joint_sched/tests/test_joint_wifi_ble_mainline_adapter.py
git -C .worktrees/joint-wifi-ble commit -m "feat: add faithful mainline joint task adapter"
```

### Task 2: Replace simplified random-task generation with mainline-faithful generation mode

**Files:**
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_random.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_random.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_random.py`

- [ ] **Step 1: Write the failing tests**

```python
from joint_sched.joint_wifi_ble_random import generate_joint_tasks_from_main_style_config


def test_main_style_generation_supports_faithful_mode():
    config = {
        "cell_size": 1,
        "pair_density": 0.1,
        "seed": 7,
        "joint_generation_mode": "faithful_mainline",
    }
    tasks = generate_joint_tasks_from_main_style_config(config)
    assert tasks
    assert any(task.radio == "wifi" for task in tasks)
    assert any(task.radio == "ble" for task in tasks)
    assert max(task.max_offsets for task in tasks) > 1
```

```python
from joint_sched.run_joint_wifi_ble_demo import resolve_joint_runtime_config


def test_runner_marks_faithful_mainline_source_mode(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"cell_size":1,"pair_density":0.1,"seed":7,"joint_generation_mode":"faithful_mainline"}', encoding="utf-8")
    resolved = resolve_joint_runtime_config(cfg)
    assert resolved["_joint_generation_mode"] == "faithful_mainline"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_random.py .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q -k "faithful_mode or source_mode"`
Expected: FAIL because only the simplified generator exists.

- [ ] **Step 3: Write minimal implementation**

```python
def generate_joint_tasks_from_main_style_config(config: Mapping[str, Any]) -> list[JointTaskSpec]:
    mode = str(config.get("joint_generation_mode", "faithful_mainline"))
    if mode == "simplified_random":
        return _generate_simplified_joint_tasks(config)
    return _generate_faithful_joint_tasks(config)
```

```python
def resolve_joint_runtime_config(config_path: str | Path) -> dict[str, Any]:
    config = load_joint_config(config_path)
    if is_native_joint_config(config):
        return dict(config)
    tasks = generate_joint_tasks_from_main_style_config(config)
    return {
        ...,
        "_joint_generation_mode": str(config.get("joint_generation_mode", "faithful_mainline")),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_random.py .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q -k "faithful_mode or source_mode"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_random.py joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_random.py joint_sched/tests/test_joint_wifi_ble_runner.py
git -C .worktrees/joint-wifi-ble commit -m "feat: align joint random generation with mainline semantics"
```

### Task 3: Expand the unified candidate space to stop under-modeling the problem

**Files:**
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_model.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py`

- [ ] **Step 1: Write the failing tests**

```python
from joint_sched.joint_wifi_ble_model import build_joint_candidate_states


def test_joint_candidate_builder_keeps_multiple_wifi_offsets():
    payload = {
        "macrocycle_slots": 64,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1000,
                "release_slot": 0,
                "deadline_slot": 20,
                "preferred_channel": 0,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 16,
                "max_offsets": 4,
            }
        ],
    }
    space = build_joint_candidate_states(payload)
    active = [space.states[idx] for idx in space.pair_to_state_indices[0] if space.states[idx].medium == "wifi"]
    assert len({state.offset for state in active}) > 1
```

```python
from joint_sched.joint_wifi_ble_model import build_joint_candidate_states


def test_joint_candidate_builder_keeps_multiple_ble_patterns_and_offsets():
    payload = {
        "macrocycle_slots": 64,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 40,
                "preferred_channel": 8,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 4,
                "ble_pattern_count": 3,
                "max_offsets": 4,
            }
        ],
    }
    space = build_joint_candidate_states(payload)
    active = [space.states[idx] for idx in space.pair_to_state_indices[0] if space.states[idx].medium == "ble"]
    assert len({state.offset for state in active}) > 1
    assert len({state.pattern_id for state in active}) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py -q -k "multiple_wifi_offsets or multiple_ble_patterns"`
Expected: FAIL on the current simplified generation path or candidate compression assumptions.

- [ ] **Step 3: Write minimal implementation**

```python
def build_wifi_pair_config(task: JointTaskSpec, wifi_channels: Iterable[int]) -> WiFiPairConfig:
    ...
    return WiFiPairConfig(..., max_offsets=max(1, task.max_offsets))


def build_ble_pair_config(task: JointTaskSpec, ble_channels: Iterable[int]) -> BLEPairConfig:
    ...
    return BLEPairConfig(..., pattern_count=max(1, task.ble_pattern_count), max_offsets=max(1, task.max_offsets))
```

Keep the existing hard-feasibility oracle unchanged; only widen the candidate set.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_model.py -q -k "multiple_wifi_offsets or multiple_ble_patterns"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_model.py
git -C .worktrees/joint-wifi-ble commit -m "feat: widen unified candidate space for joint scheduling"
```

### Task 4: Add residual-spectrum scoring helpers for unified HGA

**Files:**
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga_model.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py`

- [ ] **Step 1: Write the failing tests**

```python
from joint_sched.joint_wifi_ble_hga_model import score_ble_state_against_residual_holes
from joint_sched.joint_wifi_ble_model import JointCandidateState


def test_residual_hole_scoring_prefers_ble_state_that_fits_open_gap():
    selected = [
        JointCandidateState(state_id=0, pair_id=0, medium="wifi", offset=0, channel=0, width_slots=4, period_slots=16),
        JointCandidateState(state_id=1, pair_id=1, medium="wifi", offset=8, channel=0, width_slots=4, period_slots=16),
    ]
    good = JointCandidateState(state_id=2, pair_id=2, medium="ble", offset=4, channel=5, pattern_id=0, ci_slots=8, ce_slots=1, num_events=1)
    bad = JointCandidateState(state_id=3, pair_id=2, medium="ble", offset=1, channel=5, pattern_id=0, ci_slots=8, ce_slots=1, num_events=1)
    assert score_ble_state_against_residual_holes(good, selected) > score_ble_state_against_residual_holes(bad, selected)
```

```python
from joint_sched.joint_wifi_ble_hga_model import rank_residual_candidate_swaps


def test_rank_residual_candidate_swaps_returns_best_local_joint_moves_first():
    ranked = rank_residual_candidate_swaps(current_selection, blocker_pair_id=0, replacement_candidates=replacements, unscheduled_ble_candidates=ble_candidates)
    assert ranked
    assert ranked[0]["combined_gain"] >= ranked[-1]["combined_gain"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py -q -k "residual_hole_scoring or candidate_swaps"`
Expected: FAIL because the residual helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def score_ble_state_against_residual_holes(
    ble_state: JointCandidateState,
    selected_states: Iterable[JointCandidateState],
) -> float:
    from .joint_wifi_ble_model import expand_candidate_blocks, blocks_overlap_cost
    ble_blocks = expand_candidate_blocks(ble_state)
    overlap = 0.0
    for selected_state in selected_states:
        for left in ble_blocks:
            for right in expand_candidate_blocks(selected_state):
                overlap += blocks_overlap_cost(left, right)
    occupied_span_penalty = sum(block.slot_end - block.slot_start for block in ble_blocks)
    return -overlap - 0.01 * occupied_span_penalty
```

```python
def rank_residual_candidate_swaps(...):
    ...
    return sorted(candidates, key=lambda item: item["combined_gain"], reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py -q -k "residual_hole_scoring or candidate_swaps"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_hga_model.py joint_sched/tests/test_joint_wifi_ble_hga_model.py
git -C .worktrees/joint-wifi-ble commit -m "feat: add residual spectrum scoring for unified HGA"
```

### Task 5: Upgrade unified HGA to use residual-hole-aware seeds and local search

**Files:**
- Modify: `.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py`

- [ ] **Step 1: Write the failing tests**

```python
from joint_sched.joint_wifi_ble_hga import solve_joint_wifi_ble_hga


def test_hga_reports_residual_seed_usage():
    result = solve_joint_wifi_ble_hga(config)
    assert result["search_mode"] == "unified_joint"
    assert result["residual_seed_count"] >= 1
```

```python
from joint_sched.joint_wifi_ble_ga import solve_joint_wifi_ble_ga
from joint_sched.joint_wifi_ble_hga import solve_joint_wifi_ble_hga


def test_hga_is_never_worse_than_joint_ga_on_benchmark_fixture():
    ga_result = solve_joint_wifi_ble_ga(config)
    hga_result = solve_joint_wifi_ble_hga(config)
    assert hga_result["scheduled_payload_bytes"] >= ga_result["scheduled_payload_bytes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py -q -k "residual_seed_usage or never_worse_than_joint_ga"`
Expected: FAIL because HGA does not expose those metrics or guarantee the new behavior.

- [ ] **Step 3: Write minimal implementation**

```python
def solve_joint_wifi_ble_hga(config: Mapping[str, Any]) -> dict[str, Any]:
    ...
    greedy_seeds = _seeded_chromosomes(cfg)
    residual_seeds = _build_residual_hole_seeds(cfg, best_result)
    candidate_ga["seeded_chromosomes"] = greedy_seeds + residual_seeds
    ...
    return {
        **best_result,
        "solver": "hga",
        "search_mode": "unified_joint",
        "residual_seed_count": len(residual_seeds),
        ...
    }
```

Keep the WiFi payload floor:

```python
if candidate_wifi_payload < wifi_seed_payload:
    continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py -q -k "residual_seed_usage or never_worse_than_joint_ga"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_hga.py joint_sched/tests/test_joint_wifi_ble_hga.py
git -C .worktrees/joint-wifi-ble commit -m "feat: upgrade unified HGA with residual-hole local search"
```

### Task 6: Add faithful-mainline benchmark comparison against legacy output

**Files:**
- Modify: `.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`
- Modify: `.worktrees/joint-wifi-ble/README.md`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing test**

```python
from joint_sched.run_joint_wifi_ble_demo import run_joint_demo


def test_runner_summary_includes_generation_mode_and_residual_metrics(tmp_path):
    summary = run_joint_demo(
        config_path="sim_script/pd_mmw_template_ap_stats_config.json",
        solver="hga",
        output_dir=tmp_path / "hga",
    )
    assert summary["_joint_generation_mode"] == "faithful_mainline"
    assert summary["residual_seed_count"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q -k generation_mode_and_residual_metrics`
Expected: FAIL because summary does not expose all fields yet.

- [ ] **Step 3: Write minimal implementation**

```python
for key in (
    "search_mode",
    "wifi_seed_payload_bytes",
    "final_wifi_payload_bytes",
    "coordination_rounds_used",
    "heuristic_seed_count",
    "candidate_state_count",
    "residual_seed_count",
):
    if key in result:
        summary[key] = result[key]
summary["_joint_generation_mode"] = config.get("_joint_generation_mode", "native_joint")
```

Update README section `11. joint_sched` with:
- why earlier comparisons were unfair
- what `faithful_mainline` means
- why HGA still remains unified joint scheduling

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q -k generation_mode_and_residual_metrics`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C .worktrees/joint-wifi-ble add joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_runner.py README.md
git -C .worktrees/joint-wifi-ble commit -m "docs: clarify faithful joint comparison and residual HGA metrics"
```

### Task 7: Final regression and comparison run

**Files:**
- Modify: `.worktrees/joint-wifi-ble/README.md`
- Test: `.worktrees/joint-wifi-ble/joint_sched/tests/`

- [ ] **Step 1: Run focused regression**

Run:

```bash
PYTHONPATH=. pytest .worktrees/joint-wifi-ble/joint_sched/tests -q
```

Expected: PASS.

- [ ] **Step 2: Run faithful comparison**

Run:

```bash
PYTHONPATH=. python .worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py \
  --config .worktrees/joint-wifi-ble/sim_script/pd_mmw_template_ap_stats_config.json \
  --solver ga \
  --output .worktrees/joint-wifi-ble/joint_sched/output_compare_ga_faithful

PYTHONPATH=. python .worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py \
  --config .worktrees/joint-wifi-ble/sim_script/pd_mmw_template_ap_stats_config.json \
  --solver hga \
  --output .worktrees/joint-wifi-ble/joint_sched/output_compare_hga_faithful
```

Expected:
- both commands exit `0`
- both summaries report `_joint_generation_mode = faithful_mainline`
- `hga` reports `search_mode = unified_joint`
- neither result contains collisions

- [ ] **Step 3: Record comparison notes in README**

Add a short subsection with:
- `GA vs HGA` payload
- `GA vs HGA` selected pair count
- why this comparison is now fairer than the earlier simplified-random comparison

- [ ] **Step 4: Commit**

```bash
git -C .worktrees/joint-wifi-ble add README.md joint_sched/output_compare_ga_faithful joint_sched/output_compare_hga_faithful
git -C .worktrees/joint-wifi-ble commit -m "test: benchmark faithful joint GA and HGA comparison"
```
