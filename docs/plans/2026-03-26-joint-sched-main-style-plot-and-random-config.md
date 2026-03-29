# Joint Scheduler Main-Style Plot And Random Config Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the isolated `joint_sched/` experiment render schedule figures in the same visual style as the main `sim_script` pipeline, and add a standalone path that can ingest the same random-instance config style as `sim_script/pd_mmw_template_ap_stats.py` so the joint SDP/GA can be run on an equivalent randomly generated WiFi/BLE population.

**Architecture:** Keep the joint experiment isolated under `joint_sched/`, but stop using its current ad-hoc plot rectangles and minimal demo-only config. Instead, add a translation layer that converts joint solver results into the same row schema consumed by `sim_script/plot_schedule_from_csv.py`, then reuse the main renderer. Separately, add a random-instance generator in `joint_sched/` that reads the same high-level fields from `sim_script/pd_mmw_template_ap_stats_config.json` and builds an equivalent mixed WiFi/BLE task set for the joint solver without mutating the main experiment chain.

**Tech Stack:** Python, NumPy, matplotlib, pytest, existing `sim_script/plot_schedule_from_csv.py`

---

### Task 1: Add a failing compatibility test for main-style plotting

**Files:**
- Modify: `joint_sched/joint_wifi_ble_plot.py`
- Create: `joint_sched/tests/test_joint_wifi_ble_plot_main_style.py`
- Reference: `sim_script/plot_schedule_from_csv.py`

**Step 1: Write the failing test**

Create a test that solves a tiny joint instance, converts the result into plot rows, and asserts the rows include the fields expected by the main renderer:

```python
def test_joint_plot_rows_match_main_renderer_schema():
    result = solve_joint_wifi_ble_sdp(config)
    rows = build_main_style_plot_rows(result)

    assert rows
    assert {"slot_start", "slot_end", "freq_low_mhz", "freq_high_mhz", "medium", "label"} <= set(rows[0])
    assert any(row["medium"] == "ble_adv_idle" for row in rows)
```

**Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_plot_main_style.py -q
```

Expected: FAIL because `build_main_style_plot_rows(...)` does not exist yet.

**Step 3: Write minimal implementation**

In `joint_sched/joint_wifi_ble_plot.py`, add:
- `build_main_style_plot_rows(result)`
- a row adapter that converts joint solver `blocks` into the main CSV-style plotting schema
- BLE advertising idle rows aligned with the main renderer semantics

Do not replace the old plotting entrypoints yet.

**Step 4: Run test to verify it passes**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_plot_main_style.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_plot.py joint_sched/tests/test_joint_wifi_ble_plot_main_style.py
git commit -m "feat: add main-style plot row adapter for joint scheduler"
```

### Task 2: Switch the isolated joint plot path to reuse the main renderer

**Files:**
- Modify: `joint_sched/joint_wifi_ble_plot.py`
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `joint_sched/tests/test_joint_wifi_ble_plot.py`
- Reference: `sim_script/plot_schedule_from_csv.py`

**Step 1: Write the failing test**

Extend `joint_sched/tests/test_joint_wifi_ble_plot.py` so it asserts the standalone renderer now produces the same artifact family as the main line, for example an overview PNG generated via the main renderer:

```python
def test_joint_plot_uses_main_renderer_outputs(tmp_path):
    render_joint_schedule(result, tmp_path)
    assert (tmp_path / "wifi_ble_schedule_overview.png").exists()
```

**Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_plot.py -q
```

Expected: FAIL because the isolated renderer currently writes a single custom PNG.

**Step 3: Write minimal implementation**

Refactor `joint_sched/joint_wifi_ble_plot.py` to:
- reuse `sim_script.plot_schedule_from_csv.render_all_from_csv`
- write `schedule_plot_rows.csv`
- emit `wifi_ble_schedule_overview.png` and window plots in the same style as the main line
- keep BLE advertising idle rows visible

Update `joint_sched/run_joint_wifi_ble_demo.py` to point to the overview image path in its summary.

**Step 4: Run test to verify it passes**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_plot.py joint_sched/tests/test_joint_wifi_ble_plot_main_style.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_plot.py joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_plot.py joint_sched/tests/test_joint_wifi_ble_plot_main_style.py
git commit -m "feat: reuse main plot renderer for joint scheduler"
```

### Task 3: Add a failing test for random joint-task generation from main config style

**Files:**
- Create: `joint_sched/joint_wifi_ble_random.py`
- Create: `joint_sched/tests/test_joint_wifi_ble_random.py`
- Reference: `sim_script/pd_mmw_template_ap_stats.py`
- Reference: `sim_script/pd_mmw_template_ap_stats_config.json`

**Step 1: Write the failing test**

Create a test that feeds a minimal config with the same high-level fields used by the main script and asserts the new generator returns a deterministic mixed population:

```python
def test_random_joint_task_generation_matches_config_shape():
    tasks = generate_joint_tasks_from_main_style_config(config)
    assert any(task.radio == "wifi" for task in tasks)
    assert any(task.radio == "ble" for task in tasks)
    assert len(tasks) > 0
```

Also assert determinism under the same `seed`.

**Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_random.py -q
```

Expected: FAIL because the random-instance generator does not exist.

**Step 3: Write minimal implementation**

Implement `joint_sched/joint_wifi_ble_random.py` with:
- `generate_joint_tasks_from_main_style_config(config)`
- deterministic WiFi/BLE population generation from:
  - `cell_size`
  - `pair_density`
  - `seed`
- radio split consistent with the current main script assumptions
- payload generation in the same spirit as the main experiment
- task windows and medium-specific defaults bounded for the joint experiment

Do not call the main `env` object directly; keep the joint experiment isolated.

**Step 4: Run test to verify it passes**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_random.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_random.py joint_sched/tests/test_joint_wifi_ble_random.py
git commit -m "feat: add random joint task generation from main config style"
```

### Task 4: Add a config resolver for joint demo runs using main-style random config

**Files:**
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `joint_sched/__init__.py`
- Modify: `joint_sched/tests/test_joint_wifi_ble_model.py`
- Create: `joint_sched/tests/test_joint_wifi_ble_runner.py`
- Reference: `joint_sched/joint_wifi_ble_demo_config.json`

**Step 1: Write the failing test**

Add a runner/config test asserting that the standalone runner can accept `sim_script/pd_mmw_template_ap_stats_config.json` and resolve it into a joint mixed-task config instead of rejecting it:

```python
def test_runner_accepts_main_style_random_config(tmp_path):
    summary = run_joint_demo(config_path="sim_script/pd_mmw_template_ap_stats_config.json", solver="ga")
    assert summary["task_count"] > 0
    assert summary["state_count"] > 0
```

**Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: FAIL because the runner only understands the current joint demo JSON schema.

**Step 3: Write minimal implementation**

Refactor `joint_sched/run_joint_wifi_ble_demo.py` to:
- detect whether the input JSON is:
  - native joint config (`tasks`, `wifi_channels`, `ble_channels`)
  - or main-style random config (`cell_size`, `pair_density`, `seed`, etc.)
- translate main-style config into the joint task list via `joint_wifi_ble_random.py`
- keep the resulting joint config fully local to `joint_sched/`
- expose a pure helper like `run_joint_demo(...)` for testing

**Step 4: Run test to verify it passes**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/run_joint_wifi_ble_demo.py joint_sched/__init__.py joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/tests/test_joint_wifi_ble_model.py
git commit -m "feat: accept main-style random config in joint demo runner"
```

### Task 5: Verify a full joint-scheduler run against the main random config

**Files:**
- Modify: `README.md`
- Output: `joint_sched/output_main_config_sdp/`
- Output: `joint_sched/output_main_config_ga/`

**Step 1: Write the failing documentation expectation**

Add README notes describing:
- how the joint scheduler now reuses the main plot style
- how to run the joint scheduler on `sim_script/pd_mmw_template_ap_stats_config.json`
- the fact that the random pair population is generated independently inside `joint_sched/`, not by mutating the main pipeline

**Step 2: Run the standalone commands before implementation to capture the current gap**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py --config sim_script/pd_mmw_template_ap_stats_config.json --solver sdp
python joint_sched/run_joint_wifi_ble_demo.py --config sim_script/pd_mmw_template_ap_stats_config.json --solver ga
```

Expected: at least one command fails or does not emit main-style plotting artifacts yet.

**Step 3: Write minimal implementation**

Update README and, if needed, adjust the runner output directory behavior so each solver writes isolated artifacts such as:
- `joint_sched/output_main_config_sdp/schedule_plot_rows.csv`
- `joint_sched/output_main_config_sdp/wifi_ble_schedule_overview.png`
- `joint_sched/output_main_config_ga/wifi_ble_schedule_overview.png`

**Step 4: Run full verification**

Run:

```bash
python -m py_compile joint_sched/__init__.py joint_sched/joint_wifi_ble_model.py joint_sched/joint_wifi_ble_sdp.py joint_sched/joint_wifi_ble_ga.py joint_sched/joint_wifi_ble_plot.py joint_sched/joint_wifi_ble_random.py joint_sched/run_joint_wifi_ble_demo.py
env PYTHONPATH=. pytest joint_sched/tests -q
python joint_sched/run_joint_wifi_ble_demo.py --config sim_script/pd_mmw_template_ap_stats_config.json --solver sdp --output joint_sched/output_main_config_sdp
python joint_sched/run_joint_wifi_ble_demo.py --config sim_script/pd_mmw_template_ap_stats_config.json --solver ga --output joint_sched/output_main_config_ga
```

Expected:
- all `joint_sched/tests` pass
- both commands succeed
- both output directories contain main-style overview/window plots and `schedule_plot_rows.csv`
- the printed summaries show non-zero `task_count` and `state_count`

**Step 5: Commit**

```bash
git add README.md joint_sched/
git commit -m "feat: run isolated joint scheduler on main random config"
```
