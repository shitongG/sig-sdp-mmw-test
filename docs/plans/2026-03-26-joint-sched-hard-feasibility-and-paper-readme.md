# Joint Scheduler Hard Feasibility And Paper Readme Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the isolated `joint_sched/` experiment produce the same output artifact family as `sim_script/output`, enforce hard no-collision scheduling constraints so overlapping WiFi/WiFi, BLE/BLE, and WiFi/BLE assignments are rejected instead of drawn as valid schedules, and expand the README with a paper-style mathematical description of the joint SDP and GA formulations.

**Architecture:** Keep all joint-scheduling work isolated under `joint_sched/`, but stop treating overlap as a soft plotting artifact. Introduce an explicit feasibility layer over the mixed candidate-state space so collisions become invalid state-pair combinations or invalid chromosomes, not merely high-cost outcomes. Then propagate that feasibility model into both the SDP relaxation and the GA searcher, add full output artifacts mirroring `sim_script/output`, and rewrite the `joint_sched` README section in formal notation with objective/constraint equations.

**Tech Stack:** Python, NumPy, CVXPY, matplotlib, pytest, existing `sim_script/plot_schedule_from_csv.py`

---

### Task 1: Add failing tests for hard no-collision feasibility in the shared joint model

**Files:**
- Modify: `joint_sched/joint_wifi_ble_model.py`
- Modify: `joint_sched/tests/test_joint_wifi_ble_model.py`

**Step 1: Write the failing test**

Add tests that construct overlapping candidate states and assert the shared model can explicitly classify them as infeasible pairs.

```python
def test_joint_state_pair_feasibility_rejects_wifi_ble_overlap():
    left = JointCandidateState(... medium="wifi" ...)
    right = JointCandidateState(... medium="ble" ...)

    assert state_pair_is_feasible(left, right) is False
```

Also add:
- WiFi/WiFi overlap -> infeasible
- BLE/BLE overlap -> infeasible
- disjoint pair -> feasible

**Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k feasibility
```

Expected: FAIL because the model currently only computes overlap cost, not hard feasibility.

**Step 3: Write minimal implementation**

In `joint_sched/joint_wifi_ble_model.py`, implement:
- `blocks_conflict(...)`
- `state_pair_is_feasible(...)`
- `build_joint_forbidden_state_pairs(...)`

The first version should use the existing shared block expansion and define infeasibility as any positive time-frequency overlap.

**Step 4: Run test to verify it passes**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k feasibility
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_model.py
git commit -m "feat: add hard feasibility checks for joint state pairs"
```

### Task 2: Make the joint SDP enforce hard forbidden state-pair constraints

**Files:**
- Modify: `joint_sched/joint_wifi_ble_sdp.py`
- Modify: `joint_sched/tests/test_joint_wifi_ble_sdp.py`

**Step 1: Write the failing test**

Add a test that builds a tiny instance with one feasible and one infeasible mixed assignment and asserts the SDP output never returns the overlapping combination.

```python
def test_joint_sdp_never_selects_overlapping_state_pair():
    result = solve_joint_wifi_ble_sdp(config)
    assert selected_schedule_has_no_conflicts(result)
```

**Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_sdp.py -q
```

Expected: FAIL because the current SDP only minimizes overlap cost and may still return colliding states.

**Step 3: Write minimal implementation**

Update `joint_sched/joint_wifi_ble_sdp.py` to:
- compute forbidden state pairs from the shared model
- add hard constraints such as `Y[i, j] == 0` for infeasible pairs
- add a post-rounding validation pass
- if rounding still yields a conflict, repair or reject the rounded schedule instead of returning it as valid

Do not add new fairness terms yet. Keep the goal strictly “no collisions first”.

**Step 4: Run test to verify it passes**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_sdp.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_sdp.py joint_sched/tests/test_joint_wifi_ble_sdp.py
git commit -m "feat: enforce hard forbidden pairs in joint sdp"
```

### Task 3: Make the joint GA treat collisions as invalid, not just costly

**Files:**
- Modify: `joint_sched/joint_wifi_ble_ga.py`
- Modify: `joint_sched/tests/test_joint_wifi_ble_ga.py`

**Step 1: Write the failing test**

Add a GA test that runs on a tiny mixed instance and asserts the final chromosome is collision-free.

```python
def test_joint_ga_returns_conflict_free_schedule():
    result = solve_joint_wifi_ble_ga(config)
    assert selected_schedule_has_no_conflicts(result)
```

**Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: FAIL because the GA currently optimizes a soft overlap cost and can return colliding states.

**Step 3: Write minimal implementation**

In `joint_sched/joint_wifi_ble_ga.py`:
- precompute forbidden state pairs in the GA context
- redefine chromosome fitness to assign a massive penalty or direct invalidity to any forbidden pair
- add optional chromosome repair during initialization/mutation if needed
- validate the final solution before returning blocks

Keep the existing tournament/crossover/mutation structure.

**Step 4: Run test to verify it passes**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_ga.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "feat: enforce hard feasibility in joint ga"
```

### Task 4: Generate the full main-style artifact set for joint schedules

**Files:**
- Modify: `joint_sched/joint_wifi_ble_plot.py`
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Create: `joint_sched/tests/test_joint_wifi_ble_outputs.py`
- Reference: `sim_script/output/`

**Step 1: Write the failing test**

Add a test that runs the joint demo and asserts the output directory now contains the same file family as `sim_script/output/`, including CSV tables beyond `schedule_plot_rows.csv`.

```python
def test_joint_runner_emits_main_output_family(tmp_path):
    summary = run_joint_demo(...)
    assert (tmp_path / "pair_parameters.csv").exists()
    assert (tmp_path / "wifi_ble_schedule.csv").exists()
    assert (tmp_path / "unscheduled_pairs.csv").exists()
    assert (tmp_path / "schedule_plot_rows.csv").exists()
```

If `ble_channel_mode` semantics are represented in the joint experiment, also emit `ble_ce_channel_events.csv`; otherwise document why it is omitted and reflect that in the test.

**Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_outputs.py -q
```

Expected: FAIL because the joint runner currently only emits plot rows and images.

**Step 3: Write minimal implementation**

In `joint_sched/run_joint_wifi_ble_demo.py` and/or helper files, add export functions that write:
- `pair_parameters.csv`
- `wifi_ble_schedule.csv`
- `unscheduled_pairs.csv`
- `schedule_plot_rows.csv`
- `wifi_ble_schedule_overview.png`
- `wifi_ble_schedule_window_*.png`

The CSV schema should stay as close as practical to the main experiment, but only include fields that the joint experiment truly models. If some main-only fields are not meaningful, emit explicit `NA` or empty cells and document that choice.

**Step 4: Run test to verify it passes**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_outputs.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_plot.py joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_outputs.py
git commit -m "feat: export full main-style joint output artifacts"
```

### Task 5: Rewrite the README joint-scheduler section in paper style

**Files:**
- Modify: `README.md`
- Reference: current section around `## 11. \`joint_sched/\` WiFi-BLE 统一联合调度实验`

**Step 1: Write the failing documentation expectation**

Define the missing documentation elements that must be added:
- formal notation for the mixed pair set, candidate-state sets, and time-frequency blocks
- hard feasibility constraints for no-collision scheduling
- SDP objective and constraints in display math
- GA chromosome, fitness, mutation, crossover, and feasibility handling in paper style

**Step 2: Run documentation review to confirm the current gap**

Run:

```bash
rg -n "joint_sched|联合 SDP|联合 GA" README.md
sed -n '747,930p' README.md
```

Expected: the current section is descriptive but not yet a full paper-style formulation and does not explain the hard feasibility model precisely.

**Step 3: Write minimal implementation**

Rewrite the section so it includes, in GitHub-renderable math:
- symbol table
- joint candidate-state definition
- overlap-free feasibility condition
- SDP problem:

```math
\min \sum_{i<j} \Omega_{ij} Y_{ij}
```

subject to one-state-per-pair and forbidden-pair constraints
- GA problem encoding and fitness / penalty formulation
- explanation of why overlapping assignments are invalid and must not appear in output plots

Use ` ```math ` blocks or GitHub-safe math syntax only.

**Step 4: Run a quick rendering sanity check**

Run:

```bash
sed -n '747,930p' README.md
```

Expected: formulas are readable in source and do not use forbidden GitHub math macros.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: formalize joint sdp and ga scheduling model"
```

### Task 6: Full verification on the high-density main random config

**Files:**
- Output: `joint_sched/output_main_config_sdp/`
- Output: `joint_sched/output_main_config_ga/`

**Step 1: Run the full verification commands**

Run:

```bash
python -m py_compile joint_sched/__init__.py joint_sched/joint_wifi_ble_model.py joint_sched/joint_wifi_ble_sdp.py joint_sched/joint_wifi_ble_ga.py joint_sched/joint_wifi_ble_plot.py joint_sched/joint_wifi_ble_random.py joint_sched/run_joint_wifi_ble_demo.py
env PYTHONPATH=. pytest joint_sched/tests -q
python joint_sched/run_joint_wifi_ble_demo.py --config sim_script/pd_mmw_template_ap_stats_config.json --solver sdp --output joint_sched/output_main_config_sdp
python joint_sched/run_joint_wifi_ble_demo.py --config sim_script/pd_mmw_template_ap_stats_config.json --solver ga --output joint_sched/output_main_config_ga
```

**Step 2: Verify expected results**

Expected:
- all tests pass
- both commands succeed
- both output directories contain the full artifact set
- neither output schedule contains overlapping selected assignments
- the overview/window plots no longer visualize invalid collisions as accepted schedules

**Step 3: Commit**

```bash
git add README.md joint_sched/
git commit -m "feat: harden joint scheduler feasibility and outputs"
```
