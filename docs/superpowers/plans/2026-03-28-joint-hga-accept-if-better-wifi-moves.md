# Joint HGA Accept-If-Better WiFi Moves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a true accept-if-better WiFi local move stage to unified joint HGA so whole WiFi periodic-flow states can be moved, immediately repacked with BLE, and accepted only when they improve the protected joint objective without violating the WiFi floor.

**Architecture:** Keep `joint_sched/` isolated and preserve unified joint scheduling. Extend the current HGA in two layers: first, model-level WiFi-state move ranking helpers that can score whole-WiFi-state alternatives against residual BLE holes; second, a solver-level local move loop that directly evaluates WiFi state replacements plus BLE repack on the current best solution and accepts them only if the protected lexicographic score improves. Update runner summaries and README to document the new stage and its experiment outcomes.

**Tech Stack:** Python 3.10, pytest, existing `joint_sched/` GA/HGA/model helpers, Markdown with GitHub-renderable math.

---

## File Map

- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga_model.py`
  - Add helper functions to enumerate and rank whole-WiFi-state moves for direct local acceptance.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga.py`
  - Add direct accept-if-better WiFi local move stage and track accepted move counts separately from seed generation.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py`
  - Expose and persist new counters in `joint_summary.json` and `experiment_record.md`.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py`
  - Add tests for direct WiFi move ranking helpers.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py`
  - Add tests for accepted WiFi local moves and protected objective improvement.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`
  - Add summary tests for accepted WiFi move counters.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/README.md`
  - Document the new direct local WiFi move stage in paper-style notation and update the faithful experiment record.

### Task 1: Add Model Helpers For Direct WiFi-State Move Ranking

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga_model.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py`

- [ ] **Step 1: Write the failing tests**

```python
from joint_sched.joint_wifi_ble_hga_model import (
    build_wifi_state_move_candidates,
    rank_wifi_state_moves_for_ble_holes,
)
from joint_sched.joint_wifi_ble_model import build_joint_candidate_states


def _wifi_move_test_config() -> dict:
    return {
        "macrocycle_slots": 16,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1500,
                "release_slot": 0,
                "deadline_slot": 15,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 8,
                "repetitions": 2,
                "cyclic_periodic": True,
                "preferred_channel": 0,
                "max_offsets": 4,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 15,
                "preferred_channel": 20,
                "ble_ce_slots": 2,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 2,
                "max_offsets": 4,
            },
        ],
    }


def test_build_wifi_state_move_candidates_returns_whole_state_alternatives_only():
    space = build_joint_candidate_states(_wifi_move_test_config())
    current_state = next(state for state in space.states if state.pair_id == 0 and state.medium == "wifi")
    all_wifi_states = [space.states[idx] for idx in space.pair_to_state_indices[0]]

    moves = build_wifi_state_move_candidates(
        current_state=current_state,
        all_wifi_states=all_wifi_states,
        max_candidates=6,
    )

    assert moves
    assert all(move.medium == "wifi" for move in moves)
    assert all(move.pair_id == current_state.pair_id for move in moves)
    assert all(move.state_id != current_state.state_id for move in moves)


def test_rank_wifi_state_moves_for_ble_holes_prefers_move_that_frees_ble_holes():
    space = build_joint_candidate_states(_wifi_move_test_config())
    current_state = next(state for state in space.states if state.pair_id == 0 and state.medium == "wifi")
    all_wifi_states = [space.states[idx] for idx in space.pair_to_state_indices[0]]
    moves = build_wifi_state_move_candidates(current_state=current_state, all_wifi_states=all_wifi_states, max_candidates=6)
    residual_holes = [
        {"slot_start": 0, "slot_end": 8, "freq_low_mhz": 2450.0, "freq_high_mhz": 2452.0},
        {"slot_start": 8, "slot_end": 16, "freq_low_mhz": 2450.0, "freq_high_mhz": 2452.0},
    ]

    ranked = rank_wifi_state_moves_for_ble_holes(
        current_state=current_state,
        wifi_move_states=moves,
        residual_holes=residual_holes,
    )

    assert ranked
    assert ranked[0]["score"] >= ranked[-1]["score"]
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py -q`
Expected: FAIL because the direct-ranking helper behavior is not yet covered or helper output shape differs.

- [ ] **Step 3: Implement the minimal model helpers**

```python
def build_wifi_state_move_candidates(
    current_state: JointCandidateState,
    all_wifi_states: Iterable[JointCandidateState],
    *,
    selected_states: Iterable[JointCandidateState] | None = None,
    max_candidates: int = 4,
) -> list[JointCandidateState]:
    candidates: list[JointCandidateState] = []
    protected_states = [
        state
        for state in (selected_states or [])
        if not state_is_idle(state) and int(state.pair_id) != int(current_state.pair_id)
    ]
    for candidate in all_wifi_states:
        if candidate.medium != "wifi" or candidate.pair_id != current_state.pair_id:
            continue
        if candidate.state_id == current_state.state_id:
            continue
        if any(not state_pair_is_feasible(candidate, protected_state) for protected_state in protected_states):
            continue
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return candidates


def rank_wifi_state_moves_for_ble_holes(
    *,
    current_state: JointCandidateState,
    wifi_move_states: Iterable[JointCandidateState],
    residual_holes: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    current_overlap = score_wifi_state_against_residual_holes(current_state, residual_holes)
    ranked: list[dict[str, Any]] = []
    for candidate in wifi_move_states:
        candidate_overlap = score_wifi_state_against_residual_holes(candidate, residual_holes)
        ranked.append(
            {
                "state": candidate,
                "score": current_overlap - candidate_overlap,
                "offset": candidate.offset,
                "channel": candidate.channel,
            }
        )
    ranked.sort(key=lambda item: (float(item["score"]), -int(item["state"].state_id)), reverse=True)
    return ranked
```

- [ ] **Step 4: Run the tests to verify pass**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_hga_model.py joint_sched/tests/test_joint_wifi_ble_hga_model.py
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble commit -m "feat: rank direct wifi state moves for joint hga"
```

### Task 2: Add Accept-If-Better WiFi Local Move Stage To HGA

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py`

- [ ] **Step 1: Write the failing tests**

```python
from joint_sched.joint_wifi_ble_hga import solve_joint_wifi_ble_hga


def test_hga_reports_accepted_wifi_local_moves():
    config = _hga_test_config()
    config["hga"]["wifi_move_accept_budget"] = 4

    result = solve_joint_wifi_ble_hga(config)

    assert "accepted_wifi_local_moves" in result
    assert result["accepted_wifi_local_moves"] >= 0


def test_hga_keeps_wifi_floor_while_trying_direct_wifi_moves():
    config = _hga_test_config()
    config["objective"] = {"mode": "lexicographic", "wifi_payload_floor_bytes": 1200}
    config["hga"]["wifi_move_accept_budget"] = 4

    result = solve_joint_wifi_ble_hga(config)

    assert result["final_wifi_payload_bytes"] >= 1200
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py -q`
Expected: FAIL because accepted local move accounting and direct move stage do not exist.

- [ ] **Step 3: Implement direct accept-if-better WiFi move stage**

```python
def _try_accept_better_wifi_moves(
    *,
    current_states: list[JointCandidateState],
    space: JointCandidateSpace,
    payload_by_pair: Mapping[int, int],
    wifi_payload_floor_bytes: int,
    accept_budget: int,
) -> tuple[list[JointCandidateState], int]:
    accepted_moves = 0
    best_states = list(current_states)
    best_metrics = {
        **summarize_radio_payloads(best_states, payload_by_pair),
        **summarize_selected_schedule_metrics({"macrocycle_slots": 64, "wifi_channels": [], "ble_channels": [], "tasks": []}, best_states),
    }
    residual_holes = extract_residual_holes(
        selected_states=best_states,
        macrocycle_slots=max((int(getattr(state, "macrocycle_slots", 0) or 0) for state in best_states), default=64),
        freq_grid_mhz=_residual_freq_grid(),
    )
    for wifi_state in [state for state in best_states if state.medium == "wifi"]:
        all_wifi_states = [space.states[idx] for idx in space.pair_to_state_indices[int(wifi_state.pair_id)]]
        ranked_moves = rank_wifi_state_moves_for_ble_holes(
            current_state=wifi_state,
            wifi_move_states=build_wifi_state_move_candidates(
                current_state=wifi_state,
                all_wifi_states=all_wifi_states,
                selected_states=best_states,
                max_candidates=accept_budget,
            ),
            residual_holes=residual_holes,
        )
        for ranked in ranked_moves:
            replacement = ranked["state"]
            trial_states = [replacement if state.pair_id == wifi_state.pair_id else state for state in best_states]
            trial_states, _, _ = _repair_pack_selected_states(
                selected_states=trial_states,
                space=space,
                payload_by_pair=payload_by_pair,
                wifi_payload_floor_bytes=wifi_payload_floor_bytes,
                insert_budget=accept_budget,
                swap_budget=accept_budget,
            )
            trial_metrics = {
                **summarize_radio_payloads(trial_states, payload_by_pair),
                **summarize_selected_schedule_metrics({"macrocycle_slots": 64, "wifi_channels": [], "ble_channels": [], "tasks": []}, trial_states),
            }
            if compare_joint_candidate_scores(trial_metrics, best_metrics, wifi_payload_floor_bytes) > 0:
                best_states = trial_states
                best_metrics = trial_metrics
                accepted_moves += 1
                break
        if accepted_moves >= accept_budget:
            break
    return best_states, accepted_moves
```

Integrate it in `solve_joint_wifi_ble_hga(...)` after the seed-level GA rounds and before final return, and add summary field `accepted_wifi_local_moves`.

- [ ] **Step 4: Run the tests to verify pass**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_hga.py joint_sched/tests/test_joint_wifi_ble_hga.py
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble commit -m "feat: accept better wifi local moves in joint hga"
```

### Task 3: Surface New Counters In Runner And README

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/README.md`

- [ ] **Step 1: Write the failing runner test**

```python
def test_runner_summary_reports_accepted_wifi_local_moves(tmp_path: Path):
    summary = run_joint_demo(
        config_path=MAINLINE_ROOT / "sim_script/output_ga_wifi_reschedule/pair_parameters.csv",
        solver="hga",
        output_dir=tmp_path / "faithful_hga",
    )

    assert "accepted_wifi_local_moves" in summary
    assert summary["accepted_wifi_local_moves"] >= 0
```

- [ ] **Step 2: Run the runner test to verify failure**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q`
Expected: FAIL because `accepted_wifi_local_moves` is not yet surfaced.

- [ ] **Step 3: Implement runner summary passthrough and README update**

```python
for key in (
    "search_mode",
    "wifi_seed_payload_bytes",
    "final_wifi_payload_bytes",
    "coordination_rounds_used",
    "heuristic_seed_count",
    "candidate_state_count",
    "residual_seed_count",
    "wifi_move_seed_count",
    "wifi_move_repairs_used",
    "accepted_wifi_local_moves",
    "repair_insertions_used",
    "repair_swaps_used",
):
    if key in result:
        summary[key] = result[key]
```

And append to README joint HGA section:

```markdown
After the seed-level GA rounds, HGA now executes a direct accept-if-better WiFi local move stage. It selects a whole WiFi periodic-flow state, enumerates alternative WiFi states for the same task, immediately repacks BLE around the moved WiFi state, and accepts the move only if:

```math
P_{\mathrm{wifi}}(x') \ge P_{\mathrm{wifi}}^{\min}
```

and

```math
\mathrm{score}(x') > \mathrm{score}(x)
```

where the score comparison uses the same protected lexicographic objective as the main joint GA/HGA search.
```

- [ ] **Step 4: Run the runner test to verify pass**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble add joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_runner.py README.md
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble commit -m "docs: record accepted wifi local moves in joint hga"
```

### Task 4: Re-run Faithful GA/HGA Comparison And Verify Outcomes

**Files:**
- Verify only; no new files required.

- [ ] **Step 1: Run focused verification suite**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q`
Expected: PASS.

- [ ] **Step 2: Re-run faithful protected GA/HGA**

Run: `python /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py --solver ga --config /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/output_ga_wifi_reschedule/pair_parameters.csv --output /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments/faithful_ga_wifi_state_moves`
Expected: PASS.

Run: `python /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py --solver hga --config /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/output_ga_wifi_reschedule/pair_parameters.csv --output /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments/faithful_hga_wifi_state_moves`
Expected: PASS.

- [ ] **Step 3: Compare summaries manually**

Run: `python - <<'PY'
import json
from pathlib import Path
for name in ['faithful_ga_wifi_state_moves', 'faithful_hga_wifi_state_moves']:
    path = Path('/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments') / name / 'joint_summary.json'
    data = json.loads(path.read_text())
    print(name, data['selected_pairs'], data['scheduled_payload_bytes'], data['final_wifi_payload_bytes'], data.get('accepted_wifi_local_moves', 0), data.get('wifi_move_seed_count', 0), data.get('wifi_move_repairs_used', 0))
PY`
Expected: HGA summary prints `accepted_wifi_local_moves >= 0`; WiFi floor remains unchanged; if accepted moves stay zero, that should now be explicit in the record.

- [ ] **Step 4: Commit the verified state**

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble add joint_sched README.md
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble commit -m "test: verify accept-if-better wifi local moves"
```

---

## Self-Review

- Spec coverage:
  - Add true accept-if-better WiFi local move stage: covered in Task 2.
  - Preserve unified joint scheduling and WiFi floor: covered in Tasks 2 and 4.
  - Surface new counters in outputs and README: covered in Task 3.
  - Re-run faithful experiments: covered in Task 4.
- Placeholder scan:
  - No `TODO`, `TBD`, or vague references remain.
  - Every code-changing step includes concrete code snippets.
  - Every test/verification step has exact commands and expected outcomes.
- Type consistency:
  - New field name is consistently `accepted_wifi_local_moves` across solver, runner, tests, and README.
  - New helper names are consistent across Tasks 1 and 2.

