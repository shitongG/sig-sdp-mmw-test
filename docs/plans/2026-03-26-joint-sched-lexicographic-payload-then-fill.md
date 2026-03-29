# Joint Scheduling Lexicographic Payload-Then-Fill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change the isolated `joint_sched/` WiFi-BLE joint scheduler so it first maximizes `scheduled_payload_bytes`, then, among payload-tied or near-tied solutions, minimizes fragmentation and idle spectral area.

**Architecture:** Keep the current hard no-collision feasibility model and isolated `joint_sched/` experiment directory. Replace the current single utility objective with a lexicographic-style objective pipeline shared by both SDP and GA: first optimize total scheduled payload, then apply a second-stage tie-break objective that favors denser packing and lower fragmentation. Expose the objective policy through the isolated `joint_sched` config only, and verify behavior with targeted regression tests and metric comparisons on the main-style random config.

**Tech Stack:** Python 3.10, NumPy, CVXPY, pytest, pandas, existing `joint_sched/` model/runner/export pipeline.

---

### Task 1: Define the new objective policy contract in the isolated joint config

**Files:**
- Modify: `joint_sched/joint_wifi_ble_model.py`
- Modify: `joint_sched/joint_wifi_ble_demo_config.json`
- Create: `joint_sched/joint_wifi_ble_main_config_payload_fill.json`
- Test: `joint_sched/tests/test_joint_wifi_ble_model.py`

**Step 1: Write the failing test**

Add tests that assert the isolated config/objective resolver supports a lexicographic policy block such as:

```python
def test_resolve_objective_policy_defaults_to_payload_then_fill():
    objective = resolve_joint_objective_policy({})
    assert objective["mode"] == "lexicographic"
    assert objective["primary"] == "payload"
    assert objective["secondary"] == "fill"
    assert objective["payload_tie_tolerance"] >= 0
```

```python
def test_objective_policy_accepts_fill_penalties():
    objective = resolve_joint_objective_policy({
        "objective": {
            "mode": "lexicographic",
            "primary": "payload",
            "secondary": "fill",
            "payload_tie_tolerance": 64,
            "fragmentation_penalty": 1.0,
            "idle_area_penalty": 0.5,
            "slot_span_penalty": 0.1,
        }
    })
    assert objective["payload_tie_tolerance"] == 64
    assert objective["fragmentation_penalty"] == 1.0
```

**Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k objective_policy`
Expected: FAIL because `resolve_joint_objective_policy` does not exist yet.

**Step 3: Write minimal implementation**

In `joint_sched/joint_wifi_ble_model.py`, add a shared objective-policy resolver that:
- defaults to `mode=lexicographic`
- defaults `primary=payload`
- defaults `secondary=fill`
- carries fill penalties and payload tie tolerance
- preserves backward compatibility for existing configs if `objective` is absent

Also update `joint_sched/joint_wifi_ble_demo_config.json` and create `joint_sched/joint_wifi_ble_main_config_payload_fill.json` with explicit objective fields.

**Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k objective_policy`
Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_model.py joint_sched/joint_wifi_ble_demo_config.json joint_sched/joint_wifi_ble_main_config_payload_fill.json joint_sched/tests/test_joint_wifi_ble_model.py
git commit -m "feat: define lexicographic payload-fill objective policy"
```

### Task 2: Add shared payload/fill metrics at state and schedule level

**Files:**
- Modify: `joint_sched/joint_wifi_ble_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_model.py`

**Step 1: Write the failing test**

Add tests for reusable metrics:

```python
def test_state_fill_metrics_penalize_wider_and_more_fragmented_states():
    metrics = build_state_fill_metrics(states)
    assert metrics[state_b] > metrics[state_a]
```

```python
def test_selected_schedule_metrics_report_payload_and_idle_area():
    metrics = summarize_selected_schedule_metrics(config, states)
    assert metrics["scheduled_payload_bytes"] == 3000
    assert metrics["idle_area_penalty"] >= 0
    assert metrics["fragmentation_penalty"] >= 0
```

**Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k fill_metrics`
Expected: FAIL because the metric helpers do not exist.

**Step 3: Write minimal implementation**

In `joint_sched/joint_wifi_ble_model.py`, add helpers to compute:
- per-state payload
- per-state occupied slot count
- per-state occupied area
- per-state fragmentation proxy
- selected-schedule metrics including:
  - `scheduled_payload_bytes`
  - `occupied_slot_count`
  - `occupied_area_mhz_slots`
  - `fragmentation_penalty`
  - `idle_area_penalty`
  - `slot_span_penalty`

Keep the helpers pure so both SDP and GA can reuse them.

**Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k fill_metrics`
Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_model.py
git commit -m "feat: add shared payload and fill metrics for joint scheduling"
```

### Task 3: Convert joint SDP to a two-stage lexicographic solve

**Files:**
- Modify: `joint_sched/joint_wifi_ble_sdp.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_sdp.py`

**Step 1: Write the failing test**

Add a regression test where two feasible solutions have equal payload but different packing density:

```python
def test_joint_sdp_breaks_payload_tie_with_fill_objective():
    result = solve_joint_wifi_ble_sdp(config)
    assert result["scheduled_payload_bytes"] == expected_payload
    assert result["occupied_slot_count"] == expected_smaller_slot_count
```

Add another test where a lower-fragmentation solution with slightly lower payload is rejected when outside tolerance:

```python
def test_joint_sdp_keeps_higher_payload_when_gap_exceeds_tolerance():
    result = solve_joint_wifi_ble_sdp(config)
    assert result["scheduled_pair_ids"] == {high_payload_pair_ids}
```

**Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_sdp.py -q -k payload_tie`
Expected: FAIL because SDP still uses the old single-stage utility objective.

**Step 3: Write minimal implementation**

Refactor `solve_joint_wifi_ble_sdp` to:
1. solve stage 1: maximize total scheduled payload under hard no-collision constraints
2. record optimal payload `P*`
3. solve stage 2 with added constraint `payload >= P* - tolerance`
4. minimize fill penalties (fragmentation, idle area, slot span, occupied area) within the payload-tied region
5. export both payload and fill metrics in the result dict

Keep hard feasibility unchanged. Preserve idle-state handling.

**Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_sdp.py -q -k payload_tie`
Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_sdp.py joint_sched/tests/test_joint_wifi_ble_sdp.py
git commit -m "feat: add lexicographic payload-then-fill objective to joint SDP"
```

### Task 4: Convert joint GA fitness and survivor selection to payload-first, fill-second

**Files:**
- Modify: `joint_sched/joint_wifi_ble_ga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`

**Step 1: Write the failing test**

Add tests that assert chromosome comparison is lexicographic:

```python
def test_joint_ga_prefers_higher_payload_before_fill():
    assert compare_chromosomes(high_payload, dense_low_payload, context) > 0
```

```python
def test_joint_ga_uses_fill_penalty_when_payload_is_tied():
    assert compare_chromosomes(dense_solution, sparse_solution, context) > 0
```

**Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q -k payload_tie`
Expected: FAIL because GA still uses a scalar utility sum not a payload-first comparison.

**Step 3: Write minimal implementation**

Refactor `joint_wifi_ble_ga.py` so GA ranking uses a tuple-like objective:
- primary: total scheduled payload bytes
- secondary: lower fill penalty / lower idle area / lower fragmentation
- tertiary: lower soft cost if still needed

Implement this in:
- chromosome evaluation
- tournament comparison
- elite retention
- final best-solution selection

Avoid weakening the current repair and hard no-collision guarantees.

**Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q -k payload_tie`
Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_ga.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "feat: add payload-first fill-second objective to joint GA"
```

### Task 5: Extend runner outputs and artifacts with payload/fill diagnostics

**Files:**
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_runner.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_objective_metrics.py`

**Step 1: Write the failing test**

Add tests asserting runner output includes both stage metrics and tie-break metrics:

```python
def test_runner_writes_payload_fill_summary_metrics(tmp_path):
    summary = run_joint_demo(...)
    assert summary["scheduled_payload_bytes"] > 0
    assert summary["occupied_slot_count"] >= 0
    assert summary["fragmentation_penalty"] >= 0
    assert summary["idle_area_penalty"] >= 0
    assert summary["objective_mode"] == "lexicographic"
```

**Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/tests/test_joint_wifi_ble_objective_metrics.py -q`
Expected: FAIL because the runner summary does not yet include the new diagnostics.

**Step 3: Write minimal implementation**

Update the runner to:
- include new summary metrics from solver output
- write them into `joint_summary.json`
- keep old keys for backward compatibility where possible
- keep output folder layout unchanged

**Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/tests/test_joint_wifi_ble_objective_metrics.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/tests/test_joint_wifi_ble_objective_metrics.py
git commit -m "feat: export payload-fill diagnostics from joint runner"
```

### Task 6: Re-run the main-style random config and compare against WiFi-first baseline

**Files:**
- Modify: `joint_sched/run_joint_wifi_ble_demo.py` (only if a comparison helper is needed)
- Create: `joint_sched/output_main_config_sdp_payload_fill/*`
- Create: `joint_sched/output_main_config_ga_payload_fill/*`
- Test: no new unit test; this is verification + artifact generation

**Step 1: Run the joint SDP main-config experiment**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py   --config sim_script/pd_mmw_template_ap_stats_config.json   --solver sdp   --output joint_sched/output_main_config_sdp_payload_fill
```

Expected: completes successfully and writes full artifact family plus `joint_summary.json`.

**Step 2: Run the joint GA main-config experiment**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py   --config sim_script/pd_mmw_template_ap_stats_config.json   --solver ga   --output joint_sched/output_main_config_ga_payload_fill
```

Expected: completes successfully and writes full artifact family plus `joint_summary.json`.

**Step 3: Compare against WiFi-first outputs**

Run a small comparison script that prints:
- scheduled WiFi count
- scheduled BLE count
- scheduled payload bytes
- occupied slot count
- occupied area
- fragmentation penalty
- whether hard conflicts remain zero

Expected: the new objective should improve payload and/or fill behavior relative to the current utility objective, and the comparison should be recorded in the implementation notes.

**Step 4: Commit generated config/helper changes if any**

```bash
git add joint_sched/output_main_config_sdp_payload_fill joint_sched/output_main_config_ga_payload_fill

git commit -m "test: capture main-config payload-fill joint scheduling artifacts"
```

### Task 7: Rewrite the `joint_sched` README section in paper style for the new lexicographic objective

**Files:**
- Modify: `README.md`
- Test: manual render inspection via `sed`/GitHub-safe math blocks

**Step 1: Write the documentation diff**

Replace the current utility-only explanation in Section 11 with:
- motivation: pair-count objective underuses spectrum
- primary objective: maximize scheduled payload bytes
- secondary objective: minimize fragmentation / idle spectral area within a payload tie window
- new SDP stage-1 / stage-2 math
- new GA lexicographic comparison rule
- explanation of `payload_tie_tolerance`
- explanation of `joint_summary.json`

Include formulas like:

```math
P(x) = \sum_{k=1}^{K} b_{k} \cdot \mathbf{1}[x_k 
eq arnothing_k]
```

```math
F(x) = \mu_1 \operatorname{Frag}(x) + \mu_2 \operatorname{Idle}(x) + \mu_3 \operatorname{Span}(x)
```

and the lexicographic interpretation:

```math
x^\star = rg\min F(x)
\quad 	ext{s.t.} \quad P(x) \ge P^\star - arepsilon
```

**Step 2: Inspect the updated section locally**

Run: `sed -n '760,1085p' README.md`
Expected: the `joint_sched` section matches the implemented objective and metrics.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe lexicographic payload-fill joint objective"
```

### Task 8: Final regression and handoff

**Files:**
- Modify: none expected
- Test: `joint_sched/tests/*`

**Step 1: Run syntax checks**

Run:

```bash
python -m py_compile   joint_sched/joint_wifi_ble_model.py   joint_sched/joint_wifi_ble_sdp.py   joint_sched/joint_wifi_ble_ga.py   joint_sched/joint_wifi_ble_plot.py   joint_sched/joint_wifi_ble_random.py   joint_sched/run_joint_wifi_ble_demo.py
```

Expected: no output.

**Step 2: Run all joint tests**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests -q
```

Expected: all PASS.

**Step 3: Run a no-conflict sanity check on the new main-config outputs**

Run a small Python snippet that loads the solver results and asserts `selected_schedule_has_no_conflicts(...) is True` for both `sdp` and `ga` payload-fill runs.

Expected: both remain hard-feasible and collision-free.

**Step 4: Commit**

```bash
git add joint_sched
git add README.md
git commit -m "chore: finalize payload-fill joint scheduling objective"
```
