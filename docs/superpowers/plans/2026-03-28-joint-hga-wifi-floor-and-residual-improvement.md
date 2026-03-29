# Joint HGA WiFi Floor And Residual Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the isolated `joint_sched/` unified GA/HGA schedulers so they preserve WiFi service under joint scheduling while increasing BLE fill quality via stronger residual-spectrum heuristics.

**Architecture:** Keep the solver unified: WiFi and BLE remain in one joint candidate-state space and one joint chromosome. Add a hard WiFi-preservation floor and WiFi-aware lexicographic comparison in the solver layer, then strengthen HGA seed generation and local repair so BLE candidates are packed into residual holes without degrading the protected WiFi baseline.

**Tech Stack:** Python 3.10, dataclasses, pandas, pytest, existing `joint_sched/` model/GA/HGA runner stack

---

## File Map

- Modify: `joint_sched/joint_wifi_ble_ga.py`
  - Add WiFi-floor aware comparison helpers and solver-level baseline handling.
- Modify: `joint_sched/joint_wifi_ble_hga.py`
  - Add protected WiFi baseline extraction, residual-seed orchestration, and WiFi-safe candidate acceptance.
- Modify: `joint_sched/joint_wifi_ble_hga_model.py`
  - Improve residual-hole scoring, stripe-capacity matching, and local swap candidate ranking.
- Modify: `joint_sched/joint_wifi_ble_model.py`
  - Expose objective policy fields and any summary metrics needed by the new comparison logic.
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
  - Surface WiFi-floor metrics and comparison diagnostics in `joint_summary.json`.
- Modify: `joint_sched/joint_wifi_ble_demo_config.json`
  - Add documented defaults for the new joint HGA knobs.
- Modify: `README.md`
  - Document the new joint HGA objective, WiFi floor semantics, and recommended experiment flow.
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`
  - Add focused unit tests for WiFi-floor lexicographic comparison.
- Test: `joint_sched/tests/test_joint_wifi_ble_hga.py`
  - Add residual-hole seed and protected-WiFi acceptance tests.
- Test: `joint_sched/tests/test_joint_wifi_ble_runner.py`
  - Add integration tests that assert WiFi is not sacrificed under faithful-mainline mode.

---

### Task 1: Add WiFi-Floor Lexicographic Comparison To Joint GA

**Files:**
- Modify: `joint_sched/joint_wifi_ble_ga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`

- [ ] **Step 1: Write the failing tests**

```python
from joint_sched.joint_wifi_ble_ga import (
    compare_joint_candidate_scores,
    summarize_radio_payloads,
)


def test_compare_joint_candidate_scores_rejects_lower_wifi_payload():
    baseline = {"wifi_payload_bytes": 2000, "scheduled_payload_bytes": 6000, "fill_penalty": 100.0, "selected_pairs": 10}
    candidate = {"wifi_payload_bytes": 1800, "scheduled_payload_bytes": 7000, "fill_penalty": 90.0, "selected_pairs": 11}

    assert compare_joint_candidate_scores(candidate, baseline, wifi_payload_floor=2000) < 0


def test_compare_joint_candidate_scores_prefers_more_total_payload_when_wifi_floor_equal():
    left = {"wifi_payload_bytes": 2000, "scheduled_payload_bytes": 6400, "fill_penalty": 150.0, "selected_pairs": 10}
    right = {"wifi_payload_bytes": 2000, "scheduled_payload_bytes": 6200, "fill_penalty": 80.0, "selected_pairs": 10}

    assert compare_joint_candidate_scores(left, right, wifi_payload_floor=2000) > 0


def test_summarize_radio_payloads_splits_wifi_and_ble_payloads():
    selected_states = [
        {"pair_id": 0, "medium": "wifi"},
        {"pair_id": 1, "medium": "ble"},
    ]
    task_payloads = {
        0: {"radio": "wifi", "payload_bytes": 1500},
        1: {"radio": "ble", "payload_bytes": 247},
    }

    summary = summarize_radio_payloads(selected_states, task_payloads)

    assert summary["wifi_payload_bytes"] == 1500
    assert summary["ble_payload_bytes"] == 247
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: FAIL because the new comparison helpers and WiFi-payload split summary do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def summarize_radio_payloads(selected_states, task_payloads):
    wifi_payload = 0
    ble_payload = 0
    for state in selected_states:
        pair_id = int(state["pair_id"])
        task = task_payloads[pair_id]
        payload = int(task.get("payload_bytes", 0))
        if str(task.get("radio", task.get("medium", ""))) == "wifi":
            wifi_payload += payload
        else:
            ble_payload += payload
    return {
        "wifi_payload_bytes": wifi_payload,
        "ble_payload_bytes": ble_payload,
        "scheduled_payload_bytes": wifi_payload + ble_payload,
    }


def compare_joint_candidate_scores(left, right, wifi_payload_floor):
    left_wifi = int(left.get("wifi_payload_bytes", 0))
    right_wifi = int(right.get("wifi_payload_bytes", 0))
    left_valid = left_wifi >= wifi_payload_floor
    right_valid = right_wifi >= wifi_payload_floor
    if left_valid != right_valid:
        return 1 if left_valid else -1
    left_key = (
        left_wifi,
        int(left.get("scheduled_payload_bytes", 0)),
        int(left.get("selected_pairs", 0)),
        -float(left.get("fill_penalty", 0.0)),
    )
    right_key = (
        right_wifi,
        int(right.get("scheduled_payload_bytes", 0)),
        int(right.get("selected_pairs", 0)),
        -float(right.get("fill_penalty", 0.0)),
    )
    return (left_key > right_key) - (left_key < right_key)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: PASS for the newly added comparison tests.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_ga.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "feat: add wifi floor comparison for joint ga"
```

---

### Task 2: Thread WiFi-Floor Baseline Through The Unified GA Solver

**Files:**
- Modify: `joint_sched/joint_wifi_ble_ga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`

- [ ] **Step 1: Write the failing solver-level tests**

```python
from joint_sched.joint_wifi_ble_ga import solve_joint_wifi_ble_ga


def test_joint_ga_respects_wifi_payload_floor():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "ga": {"population_size": 8, "generations": 4, "seed": 7},
        "objective": {"mode": "lexicographic", "wifi_payload_floor_bytes": 1500},
        "tasks": [
            {"task_id": 0, "radio": "wifi", "payload_bytes": 1500, "release_slot": 0, "deadline_slot": 15, "wifi_tx_slots": 4, "wifi_period_slots": 8, "wifi_num_events": 2, "max_offsets": 2},
            {"task_id": 1, "radio": "ble", "payload_bytes": 247, "release_slot": 0, "deadline_slot": 15, "ble_ce_slots": 1, "ble_ci_slots_options": [8], "ble_num_events": 2, "ble_pattern_count": 1, "max_offsets": 2},
        ],
    }

    result = solve_joint_wifi_ble_ga(config)

    assert result["wifi_payload_bytes"] >= 1500
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: FAIL because the solver does not yet propagate or enforce `wifi_payload_floor_bytes`.

- [ ] **Step 3: Write minimal solver integration**

```python
objective_cfg = resolve_joint_objective_policy(config)
wifi_payload_floor = int(objective_cfg.get("wifi_payload_floor_bytes", 0))

baseline_summary = summarize_radio_payloads(seed_selected_states, task_payloads)
wifi_payload_floor = max(wifi_payload_floor, baseline_summary["wifi_payload_bytes"])

candidate_summary = summarize_radio_payloads(selected_states, task_payloads)
candidate_metrics = {
    **candidate_summary,
    "selected_pairs": len(selected_states),
    "fill_penalty": fill_penalty,
}

if compare_joint_candidate_scores(candidate_metrics, best_metrics, wifi_payload_floor) > 0:
    best_metrics = candidate_metrics
    best_solution = chromosome
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: PASS with solver output reporting `wifi_payload_bytes` and respecting the floor.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_ga.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "feat: enforce wifi payload floor in joint ga"
```

---

### Task 3: Strengthen Residual-Hole Heuristics In The Unified HGA

**Files:**
- Modify: `joint_sched/joint_wifi_ble_hga_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga.py`

- [ ] **Step 1: Write the failing heuristic tests**

```python
from joint_sched.joint_wifi_ble_hga_model import (
    score_residual_hole_fit,
    rank_ble_candidates_for_residual_hole,
)


def test_score_residual_hole_fit_prefers_candidate_that_fills_more_of_hole():
    hole = {"slot_start": 10, "slot_end": 14, "freq_low_mhz": 2430.0, "freq_high_mhz": 2432.0}
    tight_fit = {"occupied_area_mhz_slots": 8.0, "overlap_area_mhz_slots": 8.0}
    loose_fit = {"occupied_area_mhz_slots": 4.0, "overlap_area_mhz_slots": 4.0}

    assert score_residual_hole_fit(tight_fit, hole) > score_residual_hole_fit(loose_fit, hole)


def test_rank_ble_candidates_for_residual_hole_prefers_wifi_safe_dense_candidate():
    hole = {"slot_start": 0, "slot_end": 8, "freq_low_mhz": 2450.0, "freq_high_mhz": 2452.0}
    candidates = [
        {"state_index": 1, "wifi_overlap_area": 0.0, "overlap_area_mhz_slots": 6.0},
        {"state_index": 2, "wifi_overlap_area": 2.0, "overlap_area_mhz_slots": 8.0},
    ]

    ranked = rank_ble_candidates_for_residual_hole(candidates, hole)

    assert ranked[0]["state_index"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_hga.py -q
```

Expected: FAIL because the residual-hole scoring/ranking helpers do not yet encode the stronger WiFi-safe packing policy.

- [ ] **Step 3: Write minimal heuristic implementation**

```python
def score_residual_hole_fit(candidate_metrics, hole):
    hole_area = max(0.0, float(hole["slot_end"] - hole["slot_start"])) * max(
        0.0, float(hole["freq_high_mhz"] - hole["freq_low_mhz"])
    )
    overlap_area = float(candidate_metrics.get("overlap_area_mhz_slots", 0.0))
    wifi_overlap = float(candidate_metrics.get("wifi_overlap_area", 0.0))
    fill_ratio = 0.0 if hole_area <= 0 else overlap_area / hole_area
    return fill_ratio - 10.0 * wifi_overlap


def rank_ble_candidates_for_residual_hole(candidates, hole):
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("wifi_overlap_area", 0.0)) > 0.0,
            -score_residual_hole_fit(item, hole),
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_hga.py -q
```

Expected: PASS for the new residual-hole ordering rules.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_hga_model.py joint_sched/tests/test_joint_wifi_ble_hga.py
git commit -m "feat: improve residual hole ranking for joint hga"
```

---

### Task 4: Make HGA Seeds And Acceptance WiFi-Safe

**Files:**
- Modify: `joint_sched/joint_wifi_ble_hga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga.py`

- [ ] **Step 1: Write the failing HGA tests**

```python
from joint_sched.joint_wifi_ble_hga import solve_joint_wifi_ble_hga


def test_joint_hga_keeps_wifi_payload_at_or_above_seed_floor():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "hga": {"population_size": 10, "generations": 5, "coordination_rounds": 2, "seed": 11},
        "objective": {"mode": "lexicographic", "wifi_payload_floor_bytes": 3000},
        "tasks": [
            {"task_id": 0, "radio": "wifi", "payload_bytes": 1500, "release_slot": 0, "deadline_slot": 15, "wifi_tx_slots": 4, "wifi_period_slots": 8, "wifi_num_events": 2, "max_offsets": 2},
            {"task_id": 1, "radio": "wifi", "payload_bytes": 1500, "release_slot": 1, "deadline_slot": 15, "wifi_tx_slots": 4, "wifi_period_slots": 8, "wifi_num_events": 2, "max_offsets": 2},
            {"task_id": 2, "radio": "ble", "payload_bytes": 247, "release_slot": 0, "deadline_slot": 15, "ble_ce_slots": 1, "ble_ci_slots_options": [8], "ble_num_events": 2, "ble_pattern_count": 2, "max_offsets": 2},
        ],
    }

    result = solve_joint_wifi_ble_hga(config)

    assert result["final_wifi_payload_bytes"] >= 3000
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_hga.py -q
```

Expected: FAIL because HGA currently reports WiFi payload but does not yet use the floor as a hard candidate-acceptance rule.

- [ ] **Step 3: Write minimal HGA integration**

```python
objective_cfg = resolve_joint_objective_policy(config)
wifi_payload_floor = int(objective_cfg.get("wifi_payload_floor_bytes", 0))
wifi_payload_floor = max(wifi_payload_floor, seed_metrics["wifi_payload_bytes"])

candidate_metrics = summarize_radio_payloads(selected_states, task_payloads)
candidate_metrics.update({
    "selected_pairs": len(selected_states),
    "fill_penalty": fill_penalty,
})

if compare_joint_candidate_scores(candidate_metrics, best_metrics, wifi_payload_floor) > 0:
    best_metrics = candidate_metrics
    best_solution = selected_states
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_hga.py -q
```

Expected: PASS with `final_wifi_payload_bytes` never below the protected floor.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_hga.py joint_sched/tests/test_joint_wifi_ble_hga.py
git commit -m "feat: preserve wifi payload floor in joint hga"
```

---

### Task 5: Expose New Knobs In Runner And Config

**Files:**
- Modify: `joint_sched/joint_wifi_ble_model.py`
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `joint_sched/joint_wifi_ble_demo_config.json`
- Test: `joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing runner/config tests**

```python
from joint_sched.run_joint_wifi_ble_demo import resolve_joint_runtime_config


def test_resolve_joint_runtime_config_reads_wifi_floor_and_hga_knobs(tmp_path):
    config_path = tmp_path / "joint.json"
    config_path.write_text(
        """
{
  "objective": {
    "mode": "lexicographic",
    "wifi_payload_floor_bytes": 3000
  },
  "hga": {
    "residual_seed_budget": 6,
    "residual_swap_budget": 8
  },
  "tasks": [],
  "wifi_channels": [0],
  "ble_channels": [0],
  "macrocycle_slots": 8
}
""".strip(),
        encoding="utf-8",
    )

    config = resolve_joint_runtime_config(config_path)

    assert config["objective"]["wifi_payload_floor_bytes"] == 3000
    assert config["hga"]["residual_seed_budget"] == 6
    assert config["hga"]["residual_swap_budget"] == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: FAIL if the new knobs are not normalized or surfaced consistently.

- [ ] **Step 3: Write minimal config/runner support**

```python
objective = dict(config.get("objective", {}))
objective.setdefault("wifi_payload_floor_bytes", 0)

hga_cfg = dict(config.get("hga", {}))
hga_cfg.setdefault("residual_seed_budget", 4)
hga_cfg.setdefault("residual_swap_budget", 6)

summary["wifi_payload_floor_bytes"] = int(objective_cfg.get("wifi_payload_floor_bytes", 0))
summary["residual_seed_budget"] = int(hga_cfg.get("residual_seed_budget", 0))
summary["residual_swap_budget"] = int(hga_cfg.get("residual_swap_budget", 0))
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: PASS, and `joint_summary.json` includes the new knobs.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_model.py joint_sched/run_joint_wifi_ble_demo.py joint_sched/joint_wifi_ble_demo_config.json joint_sched/tests/test_joint_wifi_ble_runner.py
git commit -m "feat: expose wifi floor and hga residual knobs"
```

---

### Task 6: Run Faithful Comparison Regression On Mainline Pair CSV

**Files:**
- Modify: `joint_sched/tests/test_joint_wifi_ble_runner.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing faithful-regression tests**

```python
from pathlib import Path

from joint_sched.run_joint_wifi_ble_demo import run_joint_demo


def test_hga_faithful_mainline_csv_keeps_wifi_floor(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    summary = run_joint_demo(
        config_path=repo_root / "sim_script/output_ga_wifi_reschedule/pair_parameters.csv",
        solver="hga",
        output_dir=tmp_path / "faithful_hga",
    )

    assert summary["_joint_generation_mode"] == "faithful_mainline_csv"
    assert summary["final_wifi_payload_bytes"] >= summary["wifi_seed_payload_bytes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: FAIL until the HGA summary and WiFi floor plumbing are complete.

- [ ] **Step 3: Write minimal regression support**

```python
summary["wifi_seed_payload_bytes"] = result.get("wifi_seed_payload_bytes", 0)
summary["final_wifi_payload_bytes"] = result.get("final_wifi_payload_bytes", 0)
summary["_joint_generation_mode"] = config.get("_joint_generation_mode", "native_joint")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: PASS, proving faithful-mainline HGA does not degrade protected WiFi.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/run_joint_wifi_ble_demo.py
git commit -m "test: lock faithful joint hga wifi floor behavior"
```

---

### Task 7: Update README With The Improved Unified HGA Design

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the documentation delta**

```markdown
### 11.x Protected WiFi Payload Floor

在 faithful-mainline 联合调度场景中，不允许求解器通过牺牲 WiFi 周期流来换取更多 BLE pair。为此，对联合 GA/HGA 额外引入 WiFi payload floor：

```math
P_{\mathrm{wifi}}(x) \ge P_{\mathrm{wifi}}^{\min}
```

其中 $`P_{\mathrm{wifi}}^{\min}`$ 可以来自配置项 `wifi_payload_floor_bytes`，也可以由 HGA 种子解中的 WiFi payload 自动抬高。

联合比较顺序改为：

```math
\text{key}(x) =
\Big(
P_{\mathrm{wifi}}(x),
P_{\mathrm{all}}(x),
N_{\mathrm{scheduled}}(x),
-\Phi_{\mathrm{fill}}(x)
\Big)
```

只有满足 WiFi floor 的解才参与后续比较。
```

- [ ] **Step 2: Save the README changes**

Apply the documentation block in the `joint_sched` section, directly after the lexicographic objective description and before the HGA residual-hole discussion.

- [ ] **Step 3: Verify the README section renders cleanly**

Run:

```bash
rg -n "wifi_payload_floor_bytes|Protected WiFi Payload Floor|P_\\{\\\\mathrm\\{wifi\\}\\}" README.md
```

Expected: the new section is present exactly once in the `joint_sched` chapter.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: describe protected wifi floor in joint hga"
```

---

### Task 8: Final Verification On The Isolated Joint Scheduler

**Files:**
- Modify: `joint_sched/`
- Test: `joint_sched/tests/`

- [ ] **Step 1: Run the focused joint test suite**

Run:

```bash
pytest joint_sched/tests -q
```

Expected: PASS for all joint scheduler tests.

- [ ] **Step 2: Run faithful mainline GA and HGA comparisons**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py \
  --config sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --solver ga \
  --output joint_sched/output_compare_ga_faithful_periodic_protected

python joint_sched/run_joint_wifi_ble_demo.py \
  --config sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --solver hga \
  --output joint_sched/output_compare_hga_faithful_periodic_protected
```

Expected:
- both commands exit `0`
- both generate `pair_parameters.csv`, `schedule_plot_rows.csv`, `wifi_ble_schedule_overview.png`, `joint_summary.json`
- `joint_hga` reports `final_wifi_payload_bytes >= wifi_seed_payload_bytes`

- [ ] **Step 3: Inspect the key summaries**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

for name in [
    "joint_sched/output_compare_ga_faithful_periodic_protected/joint_summary.json",
    "joint_sched/output_compare_hga_faithful_periodic_protected/joint_summary.json",
]:
    data = json.loads(Path(name).read_text(encoding="utf-8"))
    print(name, {
        "selected_pairs": data["selected_pairs"],
        "scheduled_payload_bytes": data["scheduled_payload_bytes"],
        "wifi_seed_payload_bytes": data.get("wifi_seed_payload_bytes"),
        "final_wifi_payload_bytes": data.get("final_wifi_payload_bytes"),
        "fill_penalty": data["fill_penalty"],
    })
PY
```

Expected: printed summaries show the protected WiFi floor and allow side-by-side comparison of GA vs HGA.

- [ ] **Step 4: Commit**

```bash
git add joint_sched README.md
git commit -m "feat: improve joint hga with protected wifi floor"
```

---

## Self-Review

- Spec coverage: The plan covers both requested themes: preserving WiFi instead of sacrificing it, and improving joint HGA residual-spectrum packing. It also keeps the work isolated inside `joint_sched/` and retains unified joint scheduling rather than reverting to a WiFi-first sequential solver.
- Placeholder scan: No `TODO`/`TBD` placeholders remain; every task includes exact files, code snippets, test snippets, commands, and expected outcomes.
- Type consistency: The new names are consistent across tasks: `wifi_payload_floor_bytes`, `summarize_radio_payloads`, `compare_joint_candidate_scores`, `score_residual_hole_fit`, and `residual_seed_budget`/`residual_swap_budget`.

