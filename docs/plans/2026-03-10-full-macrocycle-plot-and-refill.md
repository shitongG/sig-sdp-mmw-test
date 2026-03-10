# Full Macrocycle Plot and Pair Refill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Render the schedule plot over the full macrocycle and add real refill stages for unscheduled WiFi and BLE pairs that may reshuffle already scheduled pairs to improve final scheduling success.

**Architecture:** Keep the existing `assign_macrocycle_start_slots(...)` as the core feasibility checker, then layer stronger post-processing on top of it. First, update plotting so the x-axis always spans the full macrocycle. Second, add WiFi-specific refill and stronger BLE refill/retry stages that can generate alternative pair orders and preferred-slot patterns, rerun the full macrocycle assignment, and keep only strictly better solutions.

**Tech Stack:** Python, NumPy, SciPy sparse, matplotlib, pytest.

---

### Task 1: Lock full-macrocycle plot behavior with failing tests

**Files:**
- Modify: `sim_script/tests/test_schedule_plot_render.py`
- Create: `sim_script/tests/test_schedule_plot_axis.py`
- Modify: `sim_script/pd_mmw_template_ap_stats.py:451-494`

**Step 1: Write the failing test for x-axis coverage**

```python
def test_render_schedule_plot_uses_full_macrocycle_xlim():
    rows = [{... slot: 5 ...}]
    render_schedule_plot(rows, out, macrocycle_slots=64)
    fig = plt.gcf()
    xmin, xmax = fig.axes[0].get_xlim()
    assert xmin == 0
    assert xmax == 64
```

**Step 2: Write the failing test for empty tail visibility**

```python
def test_render_schedule_plot_keeps_empty_slots_visible():
    rows = [{... slot: 5 ...}]
    render_schedule_plot(rows, out, macrocycle_slots=32)
    xmin, xmax = fig.axes[0].get_xlim()
    assert xmax - xmin == 32
```

**Step 3: Run tests to verify they fail**

Run:
```bash
pytest sim_script/tests/test_schedule_plot_render.py sim_script/tests/test_schedule_plot_axis.py -q
```

Expected:
- FAIL because `render_schedule_plot` currently derives x-limits from min/max occupied slot only

**Step 4: Implement the minimal change**

Update the renderer signature:

```python
def render_schedule_plot(plot_rows, output_path, macrocycle_slots):
```

Set:

```python
ax.set_xlim(0, macrocycle_slots)
```

Do not infer limits from `plot_rows` anymore.

**Step 5: Run tests to verify they pass**

Run:
```bash
pytest sim_script/tests/test_schedule_plot_render.py sim_script/tests/test_schedule_plot_axis.py -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add sim_script/tests/test_schedule_plot_render.py sim_script/tests/test_schedule_plot_axis.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "feat: render schedule plot over full macrocycle"
```

### Task 2: Add score-based result selection for refill stages

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:164-275`
- Create: `sim_script/tests/test_refill_result_selection.py`

**Step 1: Write the failing comparator test**

```python
def test_refill_prefers_more_scheduled_pairs_then_higher_priority_saved():
    best = (...)
    candidate = (...)
    assert _is_better_refill_result(candidate, best, priorities)
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest sim_script/tests/test_refill_result_selection.py -q
```

Expected:
- FAIL because no reusable comparator exists yet

**Step 3: Implement minimal comparator helper**

Add a helper like:

```python
def _is_better_refill_result(candidate_unscheduled, best_unscheduled, pair_priority):
    ...
```

Ranking rules:
1. fewer unscheduled pairs wins
2. if tied, smaller sum of unscheduled priorities wins
3. if still tied, keep existing result

**Step 4: Run test to verify it passes**

Run:
```bash
pytest sim_script/tests/test_refill_result_selection.py -q
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_refill_result_selection.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "refactor: add refill result comparator"
```

### Task 3: Add WiFi-specific refill stage

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:164-275`
- Create: `sim_script/tests/test_wifi_refill_scheduler.py`
- Reference: `sim_script/tests/test_macrocycle_repair.py`

**Step 1: Write the failing WiFi refill test**

```python
def test_wifi_refill_can_schedule_unscheduled_wifi_pair_by_reshuffling_existing_pairs():
    env = _WifiRefillEnv()
    preferred = np.array([...], dtype=int)
    result = retry_ble_channels_and_assign_macrocycle(env, preferred, max_ble_channel_retries=0)
    assert result[3] == []
```

Use an env stub where:
- initial assignment leaves one WiFi pair unscheduled
- allowing reshuffle and alternate preferred-slot patterns makes it feasible

**Step 2: Run test to verify it fails**

Run:
```bash
pytest sim_script/tests/test_wifi_refill_scheduler.py -q
```

Expected:
- FAIL because no WiFi-specific refill exists yet

**Step 3: Implement `_refill_unscheduled_wifi_pairs(...)`**

Requirements:
- only target current unscheduled WiFi pairs
- allowed to reshuffle already scheduled pairs
- generate multiple candidate orders and staggered preferred-slot patterns
- rerun `assign_macrocycle_start_slots(...)` from scratch
- accept only better solutions using the comparator helper

**Step 4: Insert the WiFi refill stage into the main refill flow**

Update `retry_ble_channels_and_assign_macrocycle(...)` to run:
1. initial assignment
2. generic repair
3. WiFi refill
4. BLE refill/retry

**Step 5: Run tests to verify they pass**

Run:
```bash
pytest sim_script/tests/test_wifi_refill_scheduler.py sim_script/tests/test_macrocycle_repair.py -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add sim_script/tests/test_wifi_refill_scheduler.py sim_script/tests/test_macrocycle_repair.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "feat: add wifi-specific refill stage"
```

### Task 4: Strengthen BLE refill before channel retry

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:240-281`
- Modify: `sim_script/tests/test_ble_channel_retry_scheduler.py`
- Create: `sim_script/tests/test_ble_refill_scheduler.py`

**Step 1: Write the failing BLE refill test**

```python
def test_ble_refill_can_schedule_unscheduled_ble_pair_before_channel_retry():
    env = _BleRefillEnv()
    preferred = np.array([...], dtype=int)
    result = retry_ble_channels_and_assign_macrocycle(env, preferred, max_ble_channel_retries=0)
    assert result[3] == []
```

**Step 2: Update the BLE retry test to assert refill runs before channel resample**

```python
def test_ble_retry_count_stays_zero_when_refill_already_fixes_schedule():
    ...
    assert retries_used == 0
```

**Step 3: Run tests to verify they fail**

Run:
```bash
pytest sim_script/tests/test_ble_refill_scheduler.py sim_script/tests/test_ble_channel_retry_scheduler.py -q
```

Expected:
- FAIL because current logic goes straight from generic repair to BLE channel resample loop

**Step 4: Implement `_refill_unscheduled_ble_pairs(...)`**

Requirements:
- same reshuffle permissions as WiFi refill
- no BLE channel changes inside this helper
- try alternative pair orders and staggered preferred slots first
- only if BLE refill still leaves unscheduled BLE, enter channel retry loop

**Step 5: Re-run full assignment after each BLE channel resample**

Keep current `resample_ble_channels(...)`, but after each resample, rerun:
1. full assignment
2. generic repair
3. WiFi refill
4. BLE refill

**Step 6: Run tests to verify they pass**

Run:
```bash
pytest sim_script/tests/test_ble_refill_scheduler.py sim_script/tests/test_ble_channel_retry_scheduler.py -q
```

Expected:
- PASS

**Step 7: Commit**

```bash
git add sim_script/tests/test_ble_refill_scheduler.py sim_script/tests/test_ble_channel_retry_scheduler.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "feat: add ble refill before channel retry"
```

### Task 5: Wire full-macrocycle plotting into the script output path

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:759-795`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing script-level assertion**

```python
def test_script_plot_covers_macrocycle_range():
    ...
    assert plot metadata or helper evidence shows xlim == (0, macrocycle_slots)
```

If direct image inspection is too brittle, assert through helper-level coverage only and keep script smoke focused on successful PNG export.

**Step 2: Pass `macrocycle_slots` into `render_schedule_plot(...)`**

```python
render_schedule_plot(schedule_plot_rows, schedule_plot_path, macrocycle_slots=macrocycle_slots)
```

**Step 3: Run tests to verify they pass**

Run:
```bash
pytest sim_script/tests/test_schedule_plot_render.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q
```

Expected:
- PASS

**Step 4: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/tests/test_schedule_plot_render.py
 git commit -m "feat: wire full macrocycle plotting into script"
```

### Task 6: End-to-end verification in `sig-sdp`

**Files:**
- Verify: `sim_script/pd_mmw_template_ap_stats.py`
- Verify: `sim_script/tests/test_wifi_refill_scheduler.py`
- Verify: `sim_script/tests/test_ble_refill_scheduler.py`

**Step 1: Run focused test suite**

Run:
```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && \
pytest \
  sim_script/tests/test_schedule_plot_render.py \
  sim_script/tests/test_schedule_plot_axis.py \
  sim_script/tests/test_refill_result_selection.py \
  sim_script/tests/test_wifi_refill_scheduler.py \
  sim_script/tests/test_ble_refill_scheduler.py \
  sim_script/tests/test_ble_channel_retry_scheduler.py \
  sim_script/tests/test_macrocycle_repair.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q
```

Expected:
- PASS

**Step 2: Run a script smoke with dense mixed traffic**

Run:
```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && \
python sim_script/pd_mmw_template_ap_stats.py \
  --cell-size 1 \
  --pair-density 0.5 \
  --seed 7 \
  --mmw-nit 5 \
  --output-dir /tmp/full_macrocycle_refill
```

Expected:
- exit code 0
- `wifi_ble_schedule.png` spans the full macrocycle
- `scheduled_pair_ids` count is not worse than before refill changes

**Step 3: Commit**

```bash
git add docs/plans/2026-03-10-full-macrocycle-plot-and-refill.md
 git commit -m "docs: add full macrocycle plot and refill plan"
```
