# Joint Scheduling Spectral Efficiency Objective Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current “schedule as many pairs as possible” objective in `joint_sched/` with a utility model that also values carried traffic and spectrum efficiency, so the solver stops dropping large-resource WiFi pairs too aggressively and better uses dense time-frequency resources.

**Architecture:** Keep the isolated `joint_sched/` experiment separate from the main WiFi-first pipeline, but change both the joint SDP and joint GA to optimize a shared per-state utility function instead of only maximizing scheduled-pair count. Add explicit diagnostics so each run reports scheduled pair count, scheduled payload, occupied slots, and utilization, making it possible to compare “more pairs” versus “more carried traffic” on the same random instance.

**Tech Stack:** Python, `cvxpy`, NumPy, pytest, existing `joint_sched/` plotting/CSV export pipeline.

---

### Task 1: Freeze the Current Reproduction and Define the New Objective Contract

**Files:**
- Modify: `joint_sched/tests/test_joint_wifi_ble_runner.py`
- Create: `joint_sched/tests/test_joint_wifi_ble_objective_metrics.py`
- Reference: `joint_sched/run_joint_wifi_ble_demo.py`
- Reference: `joint_sched/joint_wifi_ble_model.py`

**Step 1: Write the failing tests for objective metrics**

Add a new test file that asserts the solver summary exposes the metrics needed to judge spectrum efficiency:

```python
from joint_sched.run_joint_wifi_ble_demo import run_joint_demo


def test_joint_runner_reports_utility_and_utilization_metrics(tmp_path):
    summary = run_joint_demo(
        config_path="joint_sched/joint_wifi_ble_demo_config.json",
        solver="ga",
        output_dir=tmp_path / "joint_metrics_output",
    )

    assert "scheduled_payload_bytes" in summary
    assert "occupied_slot_count" in summary
    assert "resource_utilization" in summary
    assert "objective_mode" in summary
```

Also extend the existing runner test so it still checks the old artifact paths and now also checks these fields are present.

**Step 2: Run the tests to verify they fail**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/tests/test_joint_wifi_ble_objective_metrics.py -q
```

Expected: FAIL because the runner summary does not yet expose the new metrics.

**Step 3: Add the metrics extraction helpers**

In the joint runner or a small helper section near it, plan to compute:

- `scheduled_pair_count`
- `unscheduled_pair_count`
- `scheduled_payload_bytes`
- `occupied_slot_count`
- `resource_utilization = occupied_slot_count / (macrocycle_slots * active_channel_count)`
- `objective_mode`

Do not change solver behavior yet; just expose the bookkeeping path.

**Step 4: Run the tests to verify they pass**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/tests/test_joint_wifi_ble_objective_metrics.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/tests/test_joint_wifi_ble_objective_metrics.py joint_sched/run_joint_wifi_ble_demo.py
git commit -m "test: expose joint scheduling utility metrics"
```

### Task 2: Add a Shared State Utility Model to the Joint Candidate Space

**Files:**
- Modify: `joint_sched/joint_wifi_ble_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_model.py`

**Step 1: Write the failing tests for per-state utility**

Add tests that verify:

1. A scheduled WiFi state gets positive utility tied to payload.
2. A scheduled BLE state gets positive utility tied to payload.
3. An idle state gets zero utility.
4. A wider WiFi state is penalized more than a narrower one if payload is equal.

Example test skeleton:

```python
from joint_sched.joint_wifi_ble_model import (
    JointCandidateState,
    build_state_utility_vector,
)


def test_idle_state_has_zero_utility():
    states = [
        JointCandidateState(state_id=0, pair_id=0, medium="idle", offset=0),
    ]

    utility = build_state_utility_vector(states, payload_by_pair={0: 1024})

    assert utility.tolist() == [0.0]
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k utility
```

Expected: FAIL because `build_state_utility_vector` does not exist yet.

**Step 3: Implement the shared utility function**

In `joint_wifi_ble_model.py`, add:

- a helper to map `pair_id -> payload_bytes`
- a helper to estimate scheduled-resource cost from the expanded blocks
- `build_state_utility_vector(...)`

Keep it simple and explicit:

```python
utility_a = alpha * payload_bytes_a - beta * occupied_slot_count_a - gamma * occupied_area_a
```

Where:

- `alpha` rewards carrying data
- `beta` penalizes slot footprint
- `gamma` penalizes wide-band occupation
- idle states always get `0.0`

Do not add more terms than needed.

**Step 4: Run the tests to verify they pass**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k utility
```

Expected: PASS

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_model.py
git commit -m "feat: add shared joint-state utility model"
```

### Task 3: Make the Joint SDP Optimize Utility Under Hard No-Collision Constraints

**Files:**
- Modify: `joint_sched/joint_wifi_ble_sdp.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_sdp.py`

**Step 1: Write the failing SDP objective tests**

Add tests covering:

1. If two states conflict, SDP still avoids overlap.
2. Between a large-payload WiFi state and a tiny BLE state, the SDP can prefer the higher-utility option when only one fits.
3. Idle state is selected only when all scheduled options are lower-utility or infeasible.

Example:

```python
def test_joint_sdp_prefers_higher_utility_feasible_state():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "objective": {"alpha_payload": 1.0, "beta_slots": 0.1, "gamma_area": 0.0},
        "tasks": [...],
    }

    result = solve_joint_wifi_ble_sdp(config)

    assert result["selected_pairs"] == 1
    assert result["scheduled_payload_bytes"] == 1500
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_sdp.py -q
```

Expected: FAIL because the SDP objective still uses the older reward shape.

**Step 3: Replace the pair-count reward with the utility vector**

In `joint_wifi_ble_sdp.py`:

- import `build_state_utility_vector(...)`
- replace the old scheduled-mask reward term with:

```python
utility = build_state_utility_vector(space, config)
objective = cp.Minimize(
    cp.sum(cp.multiply(upper, Y)) - cp.sum(cp.multiply(utility, diag_y))
)
```

- preserve all hard forbidden-pair constraints
- keep rounding conflict-free

**Step 4: Run the tests to verify they pass**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_sdp.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_sdp.py joint_sched/tests/test_joint_wifi_ble_sdp.py
git commit -m "feat: optimize joint SDP for traffic utility"
```

### Task 4: Make the Joint GA Optimize the Same Shared Utility

**Files:**
- Modify: `joint_sched/joint_wifi_ble_ga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`

**Step 1: Write the failing GA objective tests**

Add tests mirroring the SDP cases:

1. GA stays collision-free.
2. GA uses the same utility preference ordering as SDP on a tiny deterministic instance.
3. Idle states appear only when a scheduled alternative is infeasible or lower utility.

**Step 2: Run the tests to verify they fail**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: FAIL because the GA still uses the old `scheduled_count - collision_cost` style fitness.

**Step 3: Replace GA fitness with the shared utility**

In `joint_wifi_ble_ga.py`:

- import the same utility builder used by SDP
- precompute `utility[state_id]`
- change:

```python
fitness = scheduled_count * reward - cost
```

to:

```python
fitness = sum(utility[idx] for idx in chromosome) - cost
```

- keep infeasible chromosomes at `INVALID_FITNESS`
- keep repair logic preferring feasible non-idle states before idle states

**Step 4: Run the tests to verify they pass**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_ga.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "feat: optimize joint GA for traffic utility"
```

### Task 5: Add Config Knobs for Utility Weights and Objective Modes

**Files:**
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `joint_sched/joint_wifi_ble_demo_config.json`
- Create: `joint_sched/joint_wifi_ble_main_config_utility.json`
- Test: `joint_sched/tests/test_joint_wifi_ble_runner.py`

**Step 1: Write the failing config tests**

Add tests that verify the runner accepts:

- `objective_mode`
- `alpha_payload`
- `beta_slots`
- `gamma_area`

Example:

```python
def test_runner_accepts_joint_utility_weights(tmp_path):
    summary = run_joint_demo(
        config_path="joint_sched/joint_wifi_ble_demo_config.json",
        solver="ga",
        output_dir=tmp_path / "out",
    )
    assert summary["objective_mode"] in {"pair_count", "utility"}
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: FAIL because the summary/config path does not yet expose the objective-mode fields.

**Step 3: Add minimal config plumbing**

Support a config block like:

```json
"objective": {
  "mode": "utility",
  "alpha_payload": 1.0,
  "beta_slots": 0.15,
  "gamma_area": 0.02
}
```

Use `"utility"` as the new default in `joint_sched/`, but keep the parser tolerant so older configs still run.

**Step 4: Run the tests to verify they pass**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add joint_sched/run_joint_wifi_ble_demo.py joint_sched/joint_wifi_ble_demo_config.json joint_sched/joint_wifi_ble_main_config_utility.json joint_sched/tests/test_joint_wifi_ble_runner.py
git commit -m "feat: add configurable utility weights for joint schedulers"
```

### Task 6: Export Utilization Diagnostics Into the Output Folder

**Files:**
- Modify: `joint_sched/joint_wifi_ble_plot.py`
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_plot.py`

**Step 1: Write the failing artifact test**

Add a test that checks the output folder contains a small machine-readable metrics file, for example `joint_summary.json`, with:

- scheduled pair count
- unscheduled pair count
- scheduled payload bytes
- occupied slot count
- utilization
- solver
- elapsed time

**Step 2: Run the test to verify it fails**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_plot.py -q -k summary
```

Expected: FAIL because the summary artifact does not exist yet.

**Step 3: Write the minimal exporter**

Create a simple JSON summary in the output directory. Keep it flat and explicit:

```json
{
  "solver": "ga",
  "scheduled_pairs": 38,
  "unscheduled_pairs": 11,
  "scheduled_payload_bytes": 12345,
  "occupied_slot_count": 87,
  "resource_utilization": 0.31
}
```

**Step 4: Run the test to verify it passes**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_plot.py -q -k summary
```

Expected: PASS

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_plot.py joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_plot.py
git commit -m "feat: export joint scheduling utilization summary"
```

### Task 7: Reproduce the Main Random Config and Compare Old vs New Behavior

**Files:**
- Modify: `README.md`
- Reference: `joint_sched/output_main_config_sdp/`
- Reference: `joint_sched/output_main_config_ga/`

**Step 1: Run the joint SDP on the main random config**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py \
  --config sim_script/pd_mmw_template_ap_stats_config.json \
  --solver sdp \
  --output joint_sched/output_main_config_sdp
```

Expected: completes and writes the full artifact family.

**Step 2: Run the joint GA on the main random config**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py \
  --config sim_script/pd_mmw_template_ap_stats_config.json \
  --solver ga \
  --output joint_sched/output_main_config_ga
```

Expected: completes and writes the full artifact family.

**Step 3: Verify zero-collision and improved utility metrics**

Run a short verification snippet:

```bash
python - <<'PY'
from joint_sched.run_joint_wifi_ble_demo import resolve_joint_runtime_config
from joint_sched.joint_wifi_ble_model import JointCandidateState, selected_schedule_has_no_conflicts
from joint_sched.joint_wifi_ble_sdp import solve_joint_wifi_ble_sdp
from joint_sched.joint_wifi_ble_ga import solve_joint_wifi_ble_ga

config = resolve_joint_runtime_config("sim_script/pd_mmw_template_ap_stats_config.json")
for name, solver in [("sdp", solve_joint_wifi_ble_sdp), ("ga", solve_joint_wifi_ble_ga)]:
    result = solver(config)
    states = [JointCandidateState(**state) for state in result["selected_states"]]
    assert selected_schedule_has_no_conflicts(states)
    print(name, len(result["selected_states"]), len(result.get("unscheduled_pair_ids", [])))
PY
```

Expected: no assertion failure.

**Step 4: Document the comparison in README**

Add one short subsection that explains:

- why “maximum pair count” was insufficient
- how the new utility objective trades off payload and slot footprint
- how to interpret `scheduled_pairs` versus `scheduled_payload_bytes` versus `resource_utilization`

**Step 5: Commit**

```bash
git add README.md joint_sched/output_main_config_sdp joint_sched/output_main_config_ga
git commit -m "docs: record utility-based joint scheduling comparison"
```

### Task 8: Final Regression Pass

**Files:**
- Reference: `joint_sched/tests/`
- Reference: `README.md`

**Step 1: Run the full joint test suite**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests -q
```

Expected: PASS

**Step 2: Run syntax checks**

Run:

```bash
python -m py_compile \
  joint_sched/joint_wifi_ble_model.py \
  joint_sched/joint_wifi_ble_sdp.py \
  joint_sched/joint_wifi_ble_ga.py \
  joint_sched/joint_wifi_ble_plot.py \
  joint_sched/run_joint_wifi_ble_demo.py
```

Expected: no output

**Step 3: Inspect README rendering risks**

Check for unsupported GitHub math macros and malformed backslashes. Avoid `\operatorname`; prefer GitHub-safe math notation already used elsewhere in the repo.

**Step 4: Confirm the worktree diff is only the intended files**

Run:

```bash
git status --short
```

Expected: only `joint_sched/*`, `README.md`, new plan docs, and generated comparison outputs.

**Step 5: Commit**

```bash
git add README.md joint_sched
git commit -m "test: finalize utility-based joint joint-scheduling objective"
```
