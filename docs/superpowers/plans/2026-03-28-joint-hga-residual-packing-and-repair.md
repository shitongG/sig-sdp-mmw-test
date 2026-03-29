# Joint HGA Residual Packing And Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the unified joint WiFi/BLE HGA so it preserves the baseline 5 scheduled WiFi periodic flows while packing more BLE flows into the residual time-frequency holes, approaching the legacy heuristic's 34 BLE pairs without reverting to WiFi-first scheduling.

**Architecture:** Keep `joint_sched/` as the isolated experimental surface. Extend the unified HGA in two directions: (1) stronger residual-hole seed construction that explicitly scores BLE candidates against actual leftover geometry after protected WiFi states are fixed, and (2) a repair/packing phase that starts from a WiFi-floor-feasible chromosome and greedily inserts additional BLE states or swaps local blockers to recover dense packing. Preserve the current faithful-mainline adapter and cyclic periodic WiFi model; do not route through `sim_script/pd_mmw_template_ap_stats.py`.

**Tech Stack:** Python 3.10, pytest, existing `joint_sched/` data model and GA/HGA solvers, CSV-based faithful adapter, matplotlib/CSV renderer already in repo.

---

## File Structure

- Modify: `joint_sched/joint_wifi_ble_hga_model.py`
  - Add residual-hole extraction helpers, BLE fit scoring, and repair candidate ranking utilities.
- Modify: `joint_sched/joint_wifi_ble_hga.py`
  - Integrate stronger residual seeds and a post-seed repair/packing stage into the unified HGA loop.
- Modify: `joint_sched/joint_wifi_ble_ga.py`
  - Reuse comparison and chromosome-evaluation helpers if needed by the repair stage; keep behavior compatible with current GA tests.
- Modify: `joint_sched/joint_wifi_ble_model.py`
  - Expose any small model-level helpers needed to compute residual occupancy summaries or fast feasibility checks for repair.
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
  - Report new HGA diagnostics in `joint_summary.json`.
- Modify: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`
  - Add unit tests for residual-hole scoring and repair candidate generation.
- Modify: `joint_sched/tests/test_joint_wifi_ble_hga.py`
  - Add solver-level tests for protected WiFi floor + extra BLE packing behavior.
- Modify: `joint_sched/tests/test_joint_wifi_ble_runner.py`
  - Add runner regression for new summary fields and faithful protected HGA output.
- Modify: `README.md`
  - Update the `joint_sched` section with the new residual-packing and repair phase in paper-style prose.

### Task 1: Add failing tests for residual-hole packing helpers

**Files:**
- Modify: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_extract_residual_holes_returns_wifi_safe_gaps():
    selected = [
        JointCandidateState(
            state_id=0,
            pair_id=1,
            medium="wifi",
            offset=0,
            channel=0,
            width_slots=5,
            period_slots=16,
            num_events=4,
            cyclic_periodic=True,
            macrocycle_slots=64,
        )
    ]
    holes = extract_residual_holes(
        selected_states=selected,
        macrocycle_slots=64,
        freq_grid_mhz=[2404.0, 2406.0, 2408.0, 2410.0, 2412.0],
    )

    assert holes
    assert all(hole["slot_end"] > hole["slot_start"] for hole in holes)
    assert any(hole["freq_low_mhz"] >= 2428.0 for hole in holes)


def test_rank_ble_insertions_prefers_candidate_that_fills_existing_hole():
    hole = {"slot_start": 20, "slot_end": 24, "freq_low_mhz": 2450.0, "freq_high_mhz": 2452.0}
    good_state = JointCandidateState(
        state_id=10,
        pair_id=10,
        medium="ble",
        offset=20,
        channel=23,
        pattern_id=0,
        ci_slots=16,
        ce_slots=4,
        num_events=1,
        macrocycle_slots=64,
    )
    bad_state = JointCandidateState(
        state_id=11,
        pair_id=11,
        medium="ble",
        offset=3,
        channel=0,
        pattern_id=0,
        ci_slots=16,
        ce_slots=1,
        num_events=1,
        macrocycle_slots=64,
    )

    ranked = rank_ble_insertions_for_holes(
        candidate_states=[good_state, bad_state],
        residual_holes=[hole],
        selected_states=[],
    )

    assert ranked[0].pair_id == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py -q`
Expected: FAIL with missing `extract_residual_holes` / `rank_ble_insertions_for_holes` or incorrect ordering.

- [ ] **Step 3: Write minimal implementation**

```python
def extract_residual_holes(*, selected_states, macrocycle_slots, freq_grid_mhz):
    occupied = build_slot_frequency_occupancy(selected_states, macrocycle_slots=macrocycle_slots)
    holes = []
    for center_mhz in freq_grid_mhz:
        current_start = None
        for slot in range(macrocycle_slots):
            busy = occupied.get((slot, center_mhz), False)
            if not busy and current_start is None:
                current_start = slot
            if busy and current_start is not None:
                holes.append(
                    {
                        "slot_start": current_start,
                        "slot_end": slot,
                        "freq_low_mhz": center_mhz - 1.0,
                        "freq_high_mhz": center_mhz + 1.0,
                    }
                )
                current_start = None
        if current_start is not None:
            holes.append(
                {
                    "slot_start": current_start,
                    "slot_end": macrocycle_slots,
                    "freq_low_mhz": center_mhz - 1.0,
                    "freq_high_mhz": center_mhz + 1.0,
                }
            )
    return [hole for hole in holes if hole["slot_end"] > hole["slot_start"]]


def rank_ble_insertions_for_holes(*, candidate_states, residual_holes, selected_states):
    scored = []
    for state in candidate_states:
        score = max(score_residual_hole_fit(state, hole, selected_states=selected_states) for hole in residual_holes)
        scored.append((score, state))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [state for _, state in scored]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add joint_sched/tests/test_joint_wifi_ble_hga_model.py joint_sched/joint_wifi_ble_hga_model.py
git commit -m "test: cover residual hole packing helpers"
```

### Task 2: Add failing tests for HGA repair/packing stage

**Files:**
- Modify: `joint_sched/tests/test_joint_wifi_ble_hga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga.py`

- [ ] **Step 1: Write the failing test**

```python
def test_joint_hga_repair_packs_additional_ble_without_losing_wifi_floor():
    config = build_small_protected_joint_config()
    config["objective"]["wifi_payload_floor_bytes"] = 3000
    config["hga"]["enable_repair_packing"] = True
    config["hga"]["repair_insert_budget"] = 6
    config["hga"]["repair_swap_budget"] = 4

    baseline = solve_joint_wifi_ble_ga(config)
    improved = solve_joint_wifi_ble_hga(config)

    assert improved["final_wifi_payload_bytes"] >= 3000
    assert improved["scheduled_payload_bytes"] >= improved["final_wifi_payload_bytes"]
    assert improved["selected_pairs"] >= baseline["selected_pairs"]
    assert improved["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga.py::test_joint_hga_repair_packs_additional_ble_without_losing_wifi_floor -q`
Expected: FAIL because repair packing does not exist or does not improve over GA.

- [ ] **Step 3: Write minimal implementation**

```python
def repair_pack_ble_states(
    *,
    selected_states,
    candidate_space,
    wifi_payload_floor_bytes,
    insert_budget,
    swap_budget,
):
    protected_wifi = [state for state in selected_states if state.medium == "wifi"]
    working = list(selected_states)
    holes = extract_residual_holes(
        selected_states=working,
        macrocycle_slots=max(int(state.macrocycle_slots or 64) for state in working) if working else 64,
        freq_grid_mhz=BLE_DATA_GRID_MHZ,
    )
    ble_candidates = [state for state in candidate_space.states if state.medium == "ble"]
    for state in rank_ble_insertions_for_holes(candidate_states=ble_candidates, residual_holes=holes, selected_states=working)[:insert_budget]:
        if state_pair_is_feasible_against_selection(state, working):
            working.append(state)
    return enforce_wifi_floor_and_unique_pairs(
        selected_states=working,
        protected_wifi=protected_wifi,
        wifi_payload_floor_bytes=wifi_payload_floor_bytes,
        swap_budget=swap_budget,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga.py::test_joint_hga_repair_packs_additional_ble_without_losing_wifi_floor -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add joint_sched/tests/test_joint_wifi_ble_hga.py joint_sched/joint_wifi_ble_hga.py joint_sched/joint_wifi_ble_hga_model.py
git commit -m "feat: add protected repair packing to joint hga"
```

### Task 3: Integrate residual-hole seeds into unified HGA runtime

**Files:**
- Modify: `joint_sched/joint_wifi_ble_hga.py`
- Modify: `joint_sched/joint_wifi_ble_hga_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga.py`

- [ ] **Step 1: Write the failing test**

```python
def test_joint_hga_uses_residual_seed_budget_to_add_extra_seeded_chromosomes():
    config = build_small_protected_joint_config()
    config["hga"]["residual_seed_budget"] = 5
    result = solve_joint_wifi_ble_hga(config)

    assert result["residual_seed_count"] >= 1
    assert result["heuristic_seed_count"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga.py::test_joint_hga_uses_residual_seed_budget_to_add_extra_seeded_chromosomes -q`
Expected: FAIL because `residual_seed_count` stays zero or is missing.

- [ ] **Step 3: Write minimal implementation**

```python
residual_seed_specs = build_residual_ble_seed_specs(
    selected_states=protected_wifi_seed,
    candidate_space=space,
    budget=int(hga_cfg.get("residual_seed_budget", 0)),
)
seeded_chromosomes = list(seed_chromosomes)
for spec_list in residual_seed_specs:
    chromosome = _build_seeded_chromosome_from_specs(space, spec_list)
    if chromosome is not None:
        seeded_chromosomes.append(chromosome)

result["residual_seed_count"] = len(residual_seed_specs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga.py::test_joint_hga_uses_residual_seed_budget_to_add_extra_seeded_chromosomes -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_hga.py joint_sched/joint_wifi_ble_hga_model.py joint_sched/tests/test_joint_wifi_ble_hga.py
git commit -m "feat: integrate residual seeds into joint hga runtime"
```

### Task 4: Expose repair diagnostics in runner and summary

**Files:**
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `joint_sched/tests/test_joint_wifi_ble_runner.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing test**

```python
def test_runner_reports_repair_packing_metrics(tmp_path: Path):
    summary = run_joint_demo(
        config_path=Path("joint_sched/joint_wifi_ble_demo_config.json"),
        solver="hga",
        output_dir=tmp_path / "out",
    )

    assert "residual_seed_count" in summary
    assert "repair_insertions_used" in summary
    assert "repair_swaps_used" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_runner.py::test_runner_reports_repair_packing_metrics -q`
Expected: FAIL because the fields are absent.

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
    "repair_insertions_used",
    "repair_swaps_used",
):
    if key in result:
        summary[key] = result[key]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_runner.py::test_runner_reports_repair_packing_metrics -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_runner.py
git commit -m "feat: report joint hga repair metrics"
```

### Task 5: Document the stronger unified HGA in README

**Files:**
- Modify: `README.md`
- Test: none

- [ ] **Step 1: Add the new paper-style subsection**

```markdown
### 11.x Protected Residual-Packing HGA

Let $\mathcal{W}_{\mathrm{base}}$ denote the WiFi baseline states imported from the faithful mainline instance. The protected floor is

```math
P_{\mathrm{wifi}}^{\min} = \sum_{a \in \mathcal{W}_{\mathrm{base}}} p_a.
```

The HGA first constructs a feasible protected seed $x^{(0)}$ such that

```math
\sum_{a \in \mathcal{A}_{\mathrm{wifi}}} p_a x_a^{(0)} \ge P_{\mathrm{wifi}}^{\min}.
```

Residual holes are then extracted from the complement of the occupied time-frequency set induced by $x^{(0)}$, and BLE candidates are ranked by hole-fit score before insertion and local swap repair.
```

- [ ] **Step 2: Save the wording and proofread math delimiters**

Run: `rg -n "Protected Residual-Packing HGA|P_\\{\\\\mathrm\\{wifi\\}\\}\\^\\{\\\\min\\}" README.md`
Expected: matching lines in the README.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe protected residual-packing joint hga"
```

### Task 6: Re-run faithful protected comparison and capture evidence

**Files:**
- Modify: none
- Test: runtime comparison only

- [ ] **Step 1: Run protected faithful GA**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py \
  --config /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --solver ga \
  --output /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/output_compare_ga_faithful_periodic_protected
```

Expected: `status = ok`, `selected_pairs` and `final` summary JSON emitted.

- [ ] **Step 2: Run protected faithful HGA**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py \
  --config /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --solver hga \
  --output /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/output_compare_hga_faithful_periodic_protected
```

Expected: `status = ok`, `final_wifi_payload_bytes >= wifi_payload_floor_bytes`.

- [ ] **Step 3: Compare scheduled counts and payloads**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path

paths = {
    "legacy": Path("sim_script/output_ga_wifi_reschedule/pair_parameters.csv"),
    "ga": Path(".worktrees/joint-wifi-ble/joint_sched/output_compare_ga_faithful_periodic_protected/pair_parameters.csv"),
    "hga": Path(".worktrees/joint-wifi-ble/joint_sched/output_compare_hga_faithful_periodic_protected/pair_parameters.csv"),
}
for name, path in paths.items():
    rows = list(csv.DictReader(path.open()))
    scheduled = [row for row in rows if int(float(row.get("schedule_slot", "-1"))) >= 0]
    wifi = sum(1 for row in scheduled if row["radio"] == "wifi")
    ble = sum(1 for row in scheduled if row["radio"] == "ble")
    print(name, len(scheduled), wifi, ble)
PY
```

Expected: evidence that protected `HGA` preserves all baseline WiFi pairs and improves over protected `GA`.

- [ ] **Step 4: Commit**

```bash
git add .worktrees/joint-wifi-ble/joint_sched/output_compare_ga_faithful_periodic_protected .worktrees/joint-wifi-ble/joint_sched/output_compare_hga_faithful_periodic_protected
git commit -m "chore: refresh protected joint faithful comparison outputs"
```

## Self-Review

- Spec coverage:
  - Stronger residual-hole seed generation: covered by Tasks 1 and 3.
  - Repair/packing phase after WiFi preservation: covered by Task 2.
  - Keep unified joint scheduling, no WiFi-first fallback: preserved in Tasks 2 and 3 by modifying only `joint_sched/`.
  - Diagnostics and comparison evidence: covered by Tasks 4 and 6.
  - README explanation: covered by Task 5.
- Placeholder scan:
  - Removed generic TODO-style language; every task contains concrete files, tests, commands, and code snippets.
- Type consistency:
  - The plan consistently uses `wifi_payload_floor_bytes`, `residual_seed_count`, `repair_insertions_used`, and `repair_swaps_used`.
  - The repair helpers referenced in Task 2 are introduced in Task 1/Task 2 and do not conflict with existing solver names.

Plan complete and saved to `docs/superpowers/plans/2026-03-28-joint-hga-residual-packing-and-repair.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
