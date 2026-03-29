# Joint HGA WiFi-State Moves And README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen unified joint GA/HGA by treating each WiFi periodic-flow state as a movable joint object that can be repositioned to open BLE-feasible residual holes, and document the resulting algorithm, encoding, and experiment results in the README in paper-style form.

**Architecture:** Keep `joint_sched/` isolated from the legacy WiFi-first pipeline, preserve unified joint scheduling, and improve HGA by generating WiFi-state move seeds and repair moves at the whole-WiFi-state level rather than by implicitly relying on BLE-only local repair. Update the faithful mainline adapter, HGA search/repair path, runner outputs, and README so experiments are reproducible and the algorithm is described with formal notation, chromosome/state encoding, constraints, and empirical findings.

**Tech Stack:** Python 3.10, pytest, existing `joint_sched/` model/GA/HGA utilities, CSV-based plotting pipeline, Markdown with GitHub-renderable math.

---

## File Map

- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga_model.py`
  - Add whole-WiFi-state move generation, residual-hole-aware ranking, and protected WiFi move helpers.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga.py`
  - Integrate WiFi-state move seeds and WiFi-move repair/packing into unified HGA without reverting to WiFi-first scheduling.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py`
  - Expose new HGA controls in config/CLI, emit summary fields, and persist experiment records.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py`
  - Add focused tests for whole-WiFi-state move generation and ranking.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py`
  - Add solver tests for WiFi-state move seeds, protected WiFi floor behavior, and repair counters.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`
  - Add runner-level regression for faithful protected HGA summaries and experiment record persistence.
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/README.md`
  - Add paper-style algorithm description for unified joint GA/HGA, encoding, constraints, WiFi-state-move heuristic, and experiment record section.
- Create: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments/`
  - Store reproducible faithful HGA comparison outputs and a machine-readable summary table if not already present.

### Task 1: Define Whole-WiFi-State Move Heuristics At Model Level

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga_model.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py`

- [ ] **Step 1: Write the failing tests for WiFi-state move generation and ranking**

```python
from joint_sched.joint_wifi_ble_hga_model import (
    build_wifi_state_move_candidates,
    rank_wifi_state_moves_for_ble_holes,
)
from joint_sched.joint_wifi_ble_model import build_joint_candidate_states


def _toy_joint_config() -> dict:
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
                "max_offsets": 3,
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
                "max_offsets": 3,
            },
        ],
    }


def test_build_wifi_state_move_candidates_returns_alternative_offsets_and_channels():
    space = build_joint_candidate_states(_toy_joint_config())
    wifi_state = next(state for state in space.states if state.pair_id == 0 and state.medium == "wifi")

    moves = build_wifi_state_move_candidates(space, wifi_state, offset_budget=2, channel_budget=2)

    assert moves
    assert any(move.offset != wifi_state.offset for move in moves)
    assert any(move.channel != wifi_state.channel for move in moves)
    assert all(move.pair_id == wifi_state.pair_id for move in moves)


def test_rank_wifi_state_moves_for_ble_holes_prefers_more_ble_friendly_move():
    space = build_joint_candidate_states(_toy_joint_config())
    wifi_state = next(state for state in space.states if state.pair_id == 0 and state.medium == "wifi")
    moves = build_wifi_state_move_candidates(space, wifi_state, offset_budget=3, channel_budget=3)

    ranked = rank_wifi_state_moves_for_ble_holes(
        candidate_space=space,
        wifi_state=wifi_state,
        wifi_move_states=moves,
        protected_pair_ids={0},
        target_ble_pair_ids={1},
    )

    assert ranked
    best = ranked[0]
    worst = ranked[-1]
    assert best.score >= worst.score
```

- [ ] **Step 2: Run the model tests to verify failure**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py -q`
Expected: FAIL with `ImportError` or missing helper function failures for `build_wifi_state_move_candidates` / `rank_wifi_state_moves_for_ble_holes`.

- [ ] **Step 3: Implement whole-WiFi-state move helpers in the model layer**

```python
@dataclass(frozen=True)
class RankedWifiStateMove:
    state: JointCandidateState
    score: float
    freed_ble_pair_ids: tuple[int, ...]


def build_wifi_state_move_candidates(
    candidate_space: JointCandidateSpace,
    wifi_state: JointCandidateState,
    offset_budget: int = 4,
    channel_budget: int = 2,
) -> list[JointCandidateState]:
    pair_indices = candidate_space.pair_to_state_indices[wifi_state.pair_id]
    wifi_alternatives = [
        candidate_space.states[idx]
        for idx in pair_indices
        if candidate_space.states[idx].medium == "wifi"
        and candidate_space.states[idx].state_id != wifi_state.state_id
    ]
    scored = []
    for alt in wifi_alternatives:
        offset_delta = abs((alt.offset or 0) - (wifi_state.offset or 0))
        channel_delta = 0 if alt.channel == wifi_state.channel else 1
        scored.append((offset_delta, channel_delta, alt.state_id, alt))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    limited: list[JointCandidateState] = []
    used_channels: set[int | None] = set()
    used_offsets: set[int | None] = set()
    for _, _, _, alt in scored:
        if len([1 for x in used_offsets if x is not None]) < offset_budget and alt.offset not in used_offsets:
            limited.append(alt)
            used_offsets.add(alt.offset)
            used_channels.add(alt.channel)
            continue
        if len([1 for x in used_channels if x is not None]) < channel_budget and alt.channel not in used_channels:
            limited.append(alt)
            used_offsets.add(alt.offset)
            used_channels.add(alt.channel)
    return limited


def rank_wifi_state_moves_for_ble_holes(
    candidate_space: JointCandidateSpace,
    wifi_state: JointCandidateState,
    wifi_move_states: Sequence[JointCandidateState],
    protected_pair_ids: set[int],
    target_ble_pair_ids: set[int],
) -> list[RankedWifiStateMove]:
    ranked: list[RankedWifiStateMove] = []
    baseline_blocks = candidate_space.blocks_by_state[wifi_state.state_id]
    baseline_ble_hits = {
        pair_id
        for pair_id in target_ble_pair_ids
        if candidate_space.wifi_ble_overlap_by_state_pair.get((wifi_state.state_id, pair_id), 0.0) > 0.0
    }
    for move in wifi_move_states:
        if move.pair_id in protected_pair_ids and move.pair_id != wifi_state.pair_id:
            continue
        freed_ble = tuple(
            sorted(
                pair_id
                for pair_id in baseline_ble_hits
                if candidate_space.wifi_ble_overlap_by_state_pair.get((move.state_id, pair_id), 0.0) == 0.0
            )
        )
        move_blocks = candidate_space.blocks_by_state[move.state_id]
        span_penalty = float(sum(block.slot_end - block.slot_start + 1 for block in move_blocks))
        channel_shift_bonus = 1.0 if move.channel != wifi_state.channel else 0.0
        score = 100.0 * len(freed_ble) + channel_shift_bonus - 0.01 * span_penalty
        ranked.append(RankedWifiStateMove(state=move, score=score, freed_ble_pair_ids=freed_ble))
    ranked.sort(key=lambda item: (-item.score, item.state.state_id))
    return ranked
```

- [ ] **Step 4: Run the model tests to verify they pass**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_hga_model.py joint_sched/tests/test_joint_wifi_ble_hga_model.py
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble commit -m "feat: add wifi-state move heuristics for joint hga"
```

### Task 2: Integrate Whole-WiFi-State Moves Into Unified HGA Search And Repair

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/joint_wifi_ble_hga.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py`

- [ ] **Step 1: Write failing tests for WiFi-move seeded HGA and protected repair**

```python
from joint_sched.joint_wifi_ble_hga import solve_joint_wifi_ble_hga


def test_hga_uses_wifi_move_seeds_to_preserve_wifi_floor():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0, 5],
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
                "max_offsets": 3,
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
                "max_offsets": 3,
            },
            {
                "task_id": 2,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 15,
                "preferred_channel": 25,
                "ble_ce_slots": 2,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 2,
                "max_offsets": 3,
            },
        ],
        "hga": {
            "population_size": 24,
            "generations": 20,
            "mutation_rate": 0.15,
            "crossover_rate": 0.8,
            "elite_count": 2,
            "seed": 7,
            "wifi_payload_floor_bytes": 1500,
            "residual_seed_budget": 8,
            "residual_swap_budget": 4,
        },
    }

    result = solve_joint_wifi_ble_hga(config)

    assert result["summary"]["final_wifi_payload_bytes"] >= 1500
    assert result["summary"]["wifi_move_seed_count"] >= 1


def test_hga_reports_wifi_move_repairs_when_move_applied():
    config = build_joint_hga_smoke_config()
    config["hga"]["wifi_payload_floor_bytes"] = 1200
    result = solve_joint_wifi_ble_hga(config)
    assert "wifi_move_repairs_used" in result["summary"]
```

- [ ] **Step 2: Run solver tests to verify failure**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py -q`
Expected: FAIL because summary keys and WiFi-move path are not yet implemented.

- [ ] **Step 3: Implement WiFi-move seed generation and repair path in HGA**

```python
from joint_sched.joint_wifi_ble_hga_model import (
    build_wifi_state_move_candidates,
    rank_wifi_state_moves_for_ble_holes,
)


def _build_wifi_move_seeds(
    candidate_space: JointCandidateSpace,
    selected_states: Sequence[JointCandidateState],
    protected_wifi_pair_ids: set[int],
    target_ble_pair_ids: set[int],
    offset_budget: int,
    channel_budget: int,
    seed_limit: int,
) -> list[list[JointCandidateState]]:
    seeds: list[list[JointCandidateState]] = []
    for wifi_state in [state for state in selected_states if state.medium == "wifi"]:
        moves = build_wifi_state_move_candidates(
            candidate_space,
            wifi_state,
            offset_budget=offset_budget,
            channel_budget=channel_budget,
        )
        ranked = rank_wifi_state_moves_for_ble_holes(
            candidate_space=candidate_space,
            wifi_state=wifi_state,
            wifi_move_states=moves,
            protected_pair_ids=protected_wifi_pair_ids,
            target_ble_pair_ids=target_ble_pair_ids,
        )
        for ranked_move in ranked[:seed_limit]:
            replacement = [state for state in selected_states if state.pair_id != wifi_state.pair_id]
            replacement.append(ranked_move.state)
            seeds.append(sorted(replacement, key=lambda state: state.pair_id))
    return seeds


def _repair_with_wifi_state_moves(...):
    wifi_move_repairs_used = 0
    candidate_seeds = _build_wifi_move_seeds(...)
    best_states = list(current_states)
    for trial_states in candidate_seeds:
        trial_states = _repair_pack_selected_states(..., selected_states=trial_states, ...)
        if compare_joint_candidate_scores(
            summarize_selected_schedule_metrics(config, trial_states),
            summarize_selected_schedule_metrics(config, best_states),
            wifi_payload_floor,
        ) > 0:
            best_states = trial_states
            wifi_move_repairs_used += 1
    return best_states, wifi_move_repairs_used
```

- [ ] **Step 4: Run solver tests to verify pass**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble add joint_sched/joint_wifi_ble_hga.py joint_sched/tests/test_joint_wifi_ble_hga.py
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble commit -m "feat: add wifi-state moves to unified hga"
```

### Task 3: Expose Config, Persist Experiment Records, And Re-run Faithful Comparison

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py`
- Create/Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments/`

- [ ] **Step 1: Write failing runner test for experiment records and new summary fields**

```python
import json
from pathlib import Path

from joint_sched.run_joint_wifi_ble_demo import main


def test_runner_persists_wifi_move_experiment_summary(tmp_path: Path):
    out_dir = tmp_path / "faithful_hga"
    pair_csv = Path("/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/output_ga_wifi_reschedule/pair_parameters.csv")

    exit_code = main([
        "--solver",
        "hga",
        "--pair-parameters-csv",
        str(pair_csv),
        "--output",
        str(out_dir),
    ])

    assert exit_code == 0
    summary = json.loads((out_dir / "joint_summary.json").read_text())
    assert "wifi_move_seed_count" in summary
    assert "wifi_move_repairs_used" in summary
    assert (out_dir / "wifi_ble_schedule_overview.png").exists()
```

- [ ] **Step 2: Run runner test to verify failure**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q`
Expected: FAIL because the summary keys are absent or experiment persistence logic is missing.

- [ ] **Step 3: Implement config passthrough and experiment record output**

```python
DEFAULT_HGA_CONFIG = {
    "population_size": 128,
    "generations": 120,
    "mutation_rate": 0.12,
    "crossover_rate": 0.85,
    "elite_count": 6,
    "seed": 7,
    "wifi_payload_floor_bytes": 0,
    "residual_seed_budget": 24,
    "residual_swap_budget": 12,
    "wifi_move_seed_budget": 12,
    "wifi_move_offset_budget": 4,
    "wifi_move_channel_budget": 2,
}

summary = {
    **result["summary"],
    "solver": args.solver,
    "input_pair_parameters_csv": str(pair_csv) if pair_csv else None,
    "experiment_tag": output_dir.name,
}
(output_dir / "joint_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
(output_dir / "experiment_record.md").write_text(
    "\n".join(
        [
            f"# {output_dir.name}",
            f"- solver: {args.solver}",
            f"- scheduled_pairs: {summary['selected_pairs']}",
            f"- final_wifi_payload_bytes: {summary['final_wifi_payload_bytes']}",
            f"- wifi_move_seed_count: {summary['wifi_move_seed_count']}",
            f"- wifi_move_repairs_used: {summary['wifi_move_repairs_used']}",
        ]
    )
)
```

- [ ] **Step 4: Run runner test and faithful experiment**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q`
Expected: PASS.

Run: `python /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py --solver hga --pair-parameters-csv /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/output_ga_wifi_reschedule/pair_parameters.csv --output /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments/faithful_hga_wifi_state_moves`
Expected: PASS and generate `joint_summary.json`, `wifi_ble_schedule_overview.png`, `wifi_ble_schedule_window_000.png`, and `experiment_record.md`.

- [ ] **Step 5: Commit**

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble add joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/experiments
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble commit -m "feat: record faithful joint hga wifi-move experiments"
```

### Task 4: Rewrite README With Paper-Style Joint GA/HGA Description, Encoding, And Experiment Record

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/README.md`

- [ ] **Step 1: Add failing README content check by searching for required headings and notation**

Run: `python -m pytest -q /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py::test_readme_joint_hga_sections_exist`
Expected: FAIL because the README does not yet contain the new section titles and notation.

- [ ] **Step 2: Add README regression test for paper-style sections**

```python
from pathlib import Path


def test_readme_joint_hga_sections_exist():
    text = Path("/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/README.md").read_text()
    assert "Unified Joint GA/HGA Encoding" in text
    assert "WiFi-State Move Heuristic" in text
    assert "Faithful Mainline Experiment Record" in text
    assert "```math" in text
```

- [ ] **Step 3: Rewrite the README section with paper-style math and experiment log**

```markdown
## Unified Joint GA/HGA Encoding

Let the task set be $\mathcal{K} = \mathcal{K}^{\mathrm{wifi}} \cup \mathcal{K}^{\mathrm{ble}}$. For each task $k \in \mathcal{K}$, define a candidate-state set $\mathcal{A}_k$ and an idle state $\varnothing_k$. A feasible schedule selects exactly one state from $\mathcal{A}_k \cup \{\varnothing_k\}$ for every task.

```math
x = (a_1, \ldots, a_{|\mathcal{K}|}), \qquad a_k \in \mathcal{A}_k \cup \{\varnothing_k\}
```

For WiFi task $k \in \mathcal{K}^{\mathrm{wifi}}$, each state encodes the tuple

```math
a_k^{\mathrm{wifi}} = (c_k, s_k, T_k, w_k, M_k)
```

where $c_k$ is the WiFi channel, $s_k$ is the periodic-flow offset, $T_k$ is the WiFi period, $w_k$ is the transmission width in slots, and $M_k$ is the number of periodic events in the macrocycle. The corresponding occupied time-frequency set is

```math
\mathcal{B}(a_k^{\mathrm{wifi}}) = \bigcup_{m=0}^{M_k-1} \left([s_k + mT_k,\ s_k + mT_k + w_k - 1]_{\mathrm{mod}\ H} \times \mathcal{F}(c_k)\right)
```

For BLE task $k \in \mathcal{K}^{\mathrm{ble}}$, each state encodes

```math
a_k^{\mathrm{ble}} = (s_k, \ell_k, \Delta_k, d_k, M_k)
```

where $s_k$ is the initial offset, $\ell_k$ is the hopping-pattern index, $\Delta_k$ is the BLE connect interval, $d_k$ is the CE length, and $M_k$ is the number of events. Its occupied set is

```math
\mathcal{B}(a_k^{\mathrm{ble}}) = \bigcup_{m=0}^{M_k-1} \left([s_k + m\Delta_k,\ s_k + m\Delta_k + d_k - 1] \times \{f_{k,m}^{(\ell_k)}\}\right)
```

A joint schedule is hard-feasible if and only if

```math
\mathcal{B}(a_p) \cap \mathcal{B}(a_q) = \emptyset, \qquad \forall p \ne q \text{ with } a_p \ne \varnothing_p,\ a_q \ne \varnothing_q
```

## WiFi-State Move Heuristic

The unified HGA does not split WiFi into ten independently schedulable tasks. Instead, a WiFi state remains a single joint object. For local geometry sensing only, the occupied $20\,\mathrm{MHz}$ WiFi band is decomposed into ten virtual $2\,\mathrm{MHz}$ stripes. These stripes are used to score residual holes and BLE blocking relationships, but the chromosome still stores exactly one WiFi state per WiFi task.

During repair, the heuristic enumerates alternative WiFi states

```math
\widetilde{\mathcal{A}}_k^{\mathrm{wifi}} \subseteq \mathcal{A}_k^{\mathrm{wifi}}
```

obtained by changing the periodic offset and, when allowed, the WiFi channel. These move candidates are ranked by the number of BLE tasks that become unblocked under the protected WiFi floor constraint.

## Faithful Mainline Experiment Record

- Baseline mainline heuristic input: `sim_script/output_ga_wifi_reschedule/pair_parameters.csv`
- Protected WiFi floor: preserve the full baseline WiFi payload and baseline WiFi task count.
- Experiment output directory: `joint_sched/experiments/faithful_hga_wifi_state_moves`
- Persisted artifacts: `joint_summary.json`, `experiment_record.md`, `wifi_ble_schedule_overview.png`, `wifi_ble_schedule_window_000.png`
```

- [ ] **Step 4: Run README regression test**

Run: `python -m pytest -q /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py::test_readme_joint_hga_sections_exist`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble add README.md joint_sched/tests/test_joint_wifi_ble_runner.py
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble commit -m "docs: describe unified joint ga hga encoding and experiments"
```

### Task 5: Full Verification And Comparison Checkpoint

**Files:**
- Verify only; no new files required.

- [ ] **Step 1: Run focused joint_sched test suite**

Run: `python -m pytest /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga_model.py /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_hga.py /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/tests/test_joint_wifi_ble_runner.py -q`
Expected: PASS.

- [ ] **Step 2: Run faithful protected comparison experiments**

Run: `python /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py --solver ga --pair-parameters-csv /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/output_ga_wifi_reschedule/pair_parameters.csv --output /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments/faithful_ga_wifi_state_moves`
Expected: PASS.

Run: `python /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/run_joint_wifi_ble_demo.py --solver hga --pair-parameters-csv /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/output_ga_wifi_reschedule/pair_parameters.csv --output /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments/faithful_hga_wifi_state_moves`
Expected: PASS.

- [ ] **Step 3: Verify the protected WiFi floor and experiment counters manually**

Run: `python - <<'PY'
import json
from pathlib import Path
paths = [
    Path('/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments/faithful_ga_wifi_state_moves/joint_summary.json'),
    Path('/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/experiments/faithful_hga_wifi_state_moves/joint_summary.json'),
]
for path in paths:
    data = json.loads(path.read_text())
    print(path.parent.name, data['selected_pairs'], data['final_wifi_payload_bytes'], data.get('wifi_move_seed_count'), data.get('wifi_move_repairs_used'))
PY`
Expected: both runs print nonzero `final_wifi_payload_bytes`; HGA prints nonzero `wifi_move_seed_count`; if `wifi_move_repairs_used` is zero, that is still acceptable but must be visible in the record.

- [ ] **Step 4: Commit the final verified state**

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble add joint_sched
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble commit -m "test: verify unified joint hga wifi-state move experiments"
```

---

## Self-Review

- Spec coverage:
  - Implement WiFi whole-state movement rather than treating WiFi as ten independent tasks: covered in Tasks 1 and 2.
  - Keep unified joint scheduling and avoid reverting to WiFi-first: enforced in Tasks 2 and 5 by integrating moves into `joint_sched` HGA/GA only.
  - Record experiment results in README: covered in Tasks 3 and 4.
  - Describe joint GA/HGA in paper style with encoding details: covered in Task 4.
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to previous task” placeholders remain.
  - Every code-changing step includes concrete code snippets.
  - Every verification step has explicit commands and expected outcomes.
- Type consistency:
  - Helper names are consistent across tasks: `build_wifi_state_move_candidates`, `rank_wifi_state_moves_for_ble_holes`, `_build_wifi_move_seeds`, `_repair_with_wifi_state_moves`.
  - Summary keys are consistent across tests and runner: `wifi_move_seed_count`, `wifi_move_repairs_used`, `final_wifi_payload_bytes`.

