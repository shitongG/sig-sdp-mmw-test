# Joint HGA Aggressive Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the unified joint HGA so that, after preserving the baseline WiFi floor, it aggressively inserts BLE candidates into residual spectrum holes and performs multi-BLE local swaps to move the faithful protected result from `5 WiFi + 29 BLE` closer to the legacy heuristic's `5 WiFi + 34 BLE`.

**Architecture:** Keep all changes inside `joint_sched/`. Extend the residual-hole model so repair works from explicit hole capacity instead of only local GA perturbations, then add a packing stage that can replace a low-value conflicting BLE subset with a denser BLE subset while leaving protected WiFi untouched. Preserve unified joint scheduling and the current faithful adapter; do not route through the legacy WiFi-first pipeline.

**Tech Stack:** Python 3.10, pytest, existing `joint_sched` unified GA/HGA solver stack, CSV faithful adapter, JSON runner summaries.

---

## File Structure

- Modify: `joint_sched/joint_wifi_ble_hga_model.py`
  - Add hole-capacity utilities and candidate-subset packing helpers.
- Modify: `joint_sched/joint_wifi_ble_hga.py`
  - Use hole-capacity-driven insertion and multi-BLE subset replacement in the repair stage.
- Modify: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`
  - Add focused tests for residual-hole capacity and dense BLE subset ranking.
- Modify: `joint_sched/tests/test_joint_wifi_ble_hga.py`
  - Add solver and helper tests for aggressive protected packing.
- Modify: `joint_sched/tests/test_joint_wifi_ble_runner.py`
  - Assert new repair counters and faithful comparison behavior remain valid.
- Modify: `README.md`
  - Update the `joint_sched` HGA section with the stronger packing/replacement formulation.

### Task 1: Add failing tests for hole-capacity-driven BLE insertion ranking

**Files:**
- Modify: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_rank_ble_insertions_prefers_candidate_that_uses_larger_fraction_of_hole_capacity():
    hole = {
        "slot_start": 20.0,
        "slot_end": 28.0,
        "freq_low_mhz": 2450.0,
        "freq_high_mhz": 2452.0,
    }
    dense = JointCandidateState(
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
    sparse = JointCandidateState(
        state_id=11,
        pair_id=11,
        medium="ble",
        offset=20,
        channel=23,
        pattern_id=0,
        ci_slots=16,
        ce_slots=1,
        num_events=1,
        macrocycle_slots=64,
    )

    ranked = rank_ble_insertions_for_holes(
        candidates=[dense, sparse],
        residual_holes=[hole],
        selected_states=[],
    )

    assert ranked[0].pair_id == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py::test_rank_ble_insertions_prefers_candidate_that_uses_larger_fraction_of_hole_capacity -q`
Expected: FAIL because current ranking does not explicitly use hole capacity ratio.

- [ ] **Step 3: Write minimal implementation**

```python
def compute_hole_capacity(hole: Mapping[str, Any]) -> float:
    return max(0.0, float(hole["slot_end"]) - float(hole["slot_start"])) * max(
        0.0,
        float(hole["freq_high_mhz"]) - float(hole["freq_low_mhz"]),
    )


def score_candidate_state_against_hole(candidate, hole, selected_states=None):
    overlap_area = ...
    hole_capacity = compute_hole_capacity(hole)
    fill_ratio = 0.0 if hole_capacity <= 0.0 else overlap_area / hole_capacity
    return fill_ratio - 10.0 * wifi_overlap_area
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py::test_rank_ble_insertions_prefers_candidate_that_uses_larger_fraction_of_hole_capacity -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_hga_model.py joint_sched/tests/test_joint_wifi_ble_hga_model.py
git commit -m "test: cover hole-capacity BLE insertion ranking"
```

### Task 2: Add failing tests for multi-BLE subset replacement packing

**Files:**
- Modify: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_rank_ble_subset_replacements_prefers_two_dense_ble_over_one_sparse_blocker():
    current_sparse = JointCandidateState(
        state_id=50,
        pair_id=50,
        medium="ble",
        offset=10,
        channel=20,
        pattern_id=0,
        ci_slots=16,
        ce_slots=4,
        num_events=1,
        macrocycle_slots=64,
    )
    dense_a = JointCandidateState(
        state_id=60,
        pair_id=60,
        medium="ble",
        offset=10,
        channel=23,
        pattern_id=0,
        ci_slots=16,
        ce_slots=2,
        num_events=1,
        macrocycle_slots=64,
    )
    dense_b = JointCandidateState(
        state_id=61,
        pair_id=61,
        medium="ble",
        offset=12,
        channel=24,
        pattern_id=0,
        ci_slots=16,
        ce_slots=2,
        num_events=1,
        macrocycle_slots=64,
    )

    ranked = rank_ble_subset_replacements(
        selected_ble_states=[current_sparse],
        candidate_ble_states=[dense_a, dense_b],
        protected_wifi_states=[],
        subset_size_limit=2,
    )

    assert ranked
    assert {state.pair_id for state in ranked[0]["insert_states"]} == {60, 61}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py::test_rank_ble_subset_replacements_prefers_two_dense_ble_over_one_sparse_blocker -q`
Expected: FAIL because subset replacement ranking is missing.

- [ ] **Step 3: Write minimal implementation**

```python
def rank_ble_subset_replacements(
    *,
    selected_ble_states,
    candidate_ble_states,
    protected_wifi_states,
    subset_size_limit,
):
    ranked = []
    for blocked in selected_ble_states:
        for subset in combinations(candidate_ble_states, r=2):
            if not subset_is_jointly_feasible(subset, protected_wifi_states, selected_ble_states, blocked):
                continue
            gain = subset_payload_density(subset) - state_payload_density(blocked)
            ranked.append(
                {
                    "remove_states": [blocked],
                    "insert_states": list(subset),
                    "gain": gain,
                }
            )
    return sorted(ranked, key=lambda item: item["gain"], reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py::test_rank_ble_subset_replacements_prefers_two_dense_ble_over_one_sparse_blocker -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_hga_model.py joint_sched/tests/test_joint_wifi_ble_hga_model.py
git commit -m "feat: add multi-ble subset replacement ranking"
```

### Task 3: Integrate aggressive insertion and subset replacement into HGA repair

**Files:**
- Modify: `joint_sched/joint_wifi_ble_hga.py`
- Modify: `joint_sched/tests/test_joint_wifi_ble_hga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga.py`

- [ ] **Step 1: Write the failing test**

```python
def test_repair_pack_selected_states_can_use_subset_replacement_to_increase_ble_count():
    config = build_dense_protected_joint_config()
    result = solve_joint_wifi_ble_hga(config)

    assert result["status"] == "ok"
    assert result["repair_insertions_used"] + result["repair_swaps_used"] >= 1
    assert result["final_wifi_payload_bytes"] >= result["wifi_seed_payload_bytes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga.py::test_repair_pack_selected_states_can_use_subset_replacement_to_increase_ble_count -q`
Expected: FAIL because current repair stage reports zero insertions/swaps on the dense protected case.

- [ ] **Step 3: Write minimal implementation**

```python
def _repair_pack_selected_states(...):
    ...
    ranked_insertions = rank_ble_insertions_for_holes(...)
    for candidate in ranked_insertions[:insert_budget]:
        if is_feasible_against_current(candidate, current):
            current.append(candidate)
            insertions_used += 1
            continue

    ranked_replacements = rank_ble_subset_replacements(
        selected_ble_states=[state for state in current if state.medium == "ble"],
        candidate_ble_states=remaining_ble_candidates,
        protected_wifi_states=[state for state in current if state.medium == "wifi"],
        subset_size_limit=max(2, swap_budget),
    )
    for replacement in ranked_replacements:
        trial = [state for state in current if state not in replacement["remove_states"]] + list(replacement["insert_states"])
        if not selection_is_feasible(trial):
            continue
        if _current_wifi_payload(trial, payload_by_pair) < wifi_payload_floor_bytes:
            continue
        if compare_joint_candidate_scores(current_metrics(trial), current_metrics(current), wifi_payload_floor_bytes) > 0:
            current = trial
            swaps_used += len(replacement["remove_states"])
            break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_hga.py::test_repair_pack_selected_states_can_use_subset_replacement_to_increase_ble_count -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_hga.py joint_sched/tests/test_joint_wifi_ble_hga.py
git commit -m "feat: use aggressive packing in joint hga repair"
```

### Task 4: Add runner regression for nonzero repair metrics on faithful protected run

**Files:**
- Modify: `joint_sched/tests/test_joint_wifi_ble_runner.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing test**

```python
def test_faithful_hga_reports_repair_activity_on_protected_mainline_instance(tmp_path: Path):
    summary = run_joint_demo(
        config_path=REPO_ROOT / "sim_script/output_ga_wifi_reschedule/pair_parameters.csv",
        solver="hga",
        output_dir=tmp_path / "faithful_hga",
    )

    assert summary["final_wifi_payload_bytes"] >= summary["wifi_payload_floor_bytes"]
    assert summary["repair_insertions_used"] + summary["repair_swaps_used"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_runner.py::test_faithful_hga_reports_repair_activity_on_protected_mainline_instance -q`
Expected: FAIL because current faithful protected result reports zero repair activity.

- [ ] **Step 3: Write minimal implementation**

```python
# No new runner code should be needed beyond existing summary fields.
# Adjust HGA repair so the faithful protected mainline case actually reports nonzero repair counters.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest joint_sched/tests/test_joint_wifi_ble_runner.py::test_faithful_hga_reports_repair_activity_on_protected_mainline_instance -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/joint_wifi_ble_hga.py
git commit -m "test: require faithful protected hga repair activity"
```

### Task 5: Update README with aggressive packing formulation

**Files:**
- Modify: `README.md`
- Test: none

- [ ] **Step 1: Add the new HGA packing subsection**

```markdown
#### 11.6.5 Aggressive residual packing

After protected WiFi floor enforcement, the repair stage extracts residual holes

```math
\mathcal{H}(x) = \{h_r\}_{r=1}^{R}
```

and first attempts direct BLE insertion by maximizing

```math
\phi(b, h_r) = \frac{\operatorname{OverlapArea}(b, h_r)}{\operatorname{Area}(h_r)} - \lambda \operatorname{ConflictPenalty}(b \mid x).
```

If no direct insertion is feasible, the repair stage evaluates subset replacement moves

```math
x' = x - \mathcal{B}_{\mathrm{out}} + \mathcal{B}_{\mathrm{in}}
```

subject to

```math
\sum_{a \in \mathcal{A}_{\mathrm{wifi}}} p_a x'_a \ge P_{\mathrm{wifi}}^{\min}
```

and zero hard conflicts.
```

- [ ] **Step 2: Save and verify the section exists**

Run: `rg -n "Aggressive residual packing|subset replacement moves" README.md`
Expected: matching lines

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe aggressive protected hga packing"
```

### Task 6: Re-run faithful protected comparison and record the improved result

**Files:**
- Modify: none
- Test: runtime comparison only

- [ ] **Step 1: Run protected faithful HGA**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py \
  --config /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --solver hga \
  --output /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/output_compare_hga_faithful_periodic_protected
```

Expected: `status = ok`, `final_wifi_payload_bytes >= wifi_payload_floor_bytes`, and nonzero repair counters.

- [ ] **Step 2: Compare legacy vs protected HGA**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path

paths = {
    "legacy": Path("sim_script/output_ga_wifi_reschedule/pair_parameters.csv"),
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

Expected: protected HGA keeps `5` WiFi and improves BLE count over the previous protected baseline of `29`.

- [ ] **Step 3: Commit**

```bash
git add .worktrees/joint-wifi-ble/joint_sched/output_compare_hga_faithful_periodic_protected
git commit -m "chore: refresh aggressive protected hga comparison output"
```

## Self-Review

- Spec coverage:
  - Direct residual-hole-capacity BLE insertion: covered by Tasks 1 and 3.
  - Multi-BLE subset replacement of low-value blockers: covered by Tasks 2 and 3.
  - Keep unified joint scheduling and preserve WiFi floor: built into Tasks 3, 4, and 6.
  - Updated documentation and evidence output: covered by Tasks 5 and 6.
- Placeholder scan:
  - No TODO/TBD text remains; every task includes specific files, code, and commands.
- Type consistency:
  - The plan consistently uses `repair_insertions_used`, `repair_swaps_used`, `wifi_payload_floor_bytes`, and `rank_ble_subset_replacements`.
  - The helper names introduced in model tasks align with the solver references in Task 3.

Plan complete and saved to `docs/superpowers/plans/2026-03-28-joint-hga-aggressive-packing.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
