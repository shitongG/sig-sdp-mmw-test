# WiFi Guarantee and BLE Overlap Highlighting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strengthen the WiFi-first scheduler so it explicitly maximizes WiFi admission before BLE fill, and highlight BLE-BLE overlap regions in the exported schedule plots with a separate collision color.

**Architecture:** Keep the existing solver split (`binary_search_relaxation(strategy="wifi_first")`) intact, but change the macrocycle assignment/refill pipeline so WiFi is treated as a protected admission class instead of just an earlier iteration order. For plotting, augment the event-span pipeline with a derived BLE-overlap layer so collision geometry is exported once and rendered consistently in both overview and windowed plots.

**Tech Stack:** Python, NumPy, SciPy sparse matrices, matplotlib, pytest

---

### Task 1: Lock the desired WiFi-first behavior in tests

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_wifi_first_macrocycle_assignment.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_wifi_refill_scheduler.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_wifi_first_macrocycle_assignment.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_wifi_refill_scheduler.py`

**Step 1: Write the failing WiFi protection/admission tests**

Add one test that proves `wifi_first=True` prefers a schedule with more admitted WiFi pairs even if the total number of admitted pairs is unchanged.

```python
def test_wifi_first_prefers_more_wifi_pairs_when_candidate_counts_tie(monkeypatch):
    ...
    assert result_unscheduled_wifi_count == 0
    assert result_unscheduled_ble_count == 1
```

Add one test that proves the refill pipeline does not sacrifice an already scheduled WiFi pair in order to admit more BLE pairs under WiFi-first mode.

```python
def test_wifi_first_refill_keeps_wifi_admission_as_primary_objective(monkeypatch):
    ...
    assert scheduled_wifi_ids == {0, 1}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_wifi_first_macrocycle_assignment.py \
  sim_script/tests/test_wifi_refill_scheduler.py -v
```

Expected: FAIL because the current refill comparison only minimizes total unscheduled count and does not encode a WiFi-first admission objective.

**Step 3: Write minimal implementation hooks**

Plan the production entry points the tests will need:

- extend `_is_better_refill_result(...)`
- thread a `wifi_first` flag through `_refill_unscheduled_pairs_by_radio(...)`
- thread the same flag through `_apply_refill_pipeline(...)`
- ensure `retry_ble_channels_and_assign_macrocycle(...)` passes the flag into the refill pipeline

No production code in this task. This task ends with the failing tests in place.

**Step 4: Re-run the tests and keep them failing cleanly**

Run the same command again and confirm the failures are still due to the missing WiFi-admission preference, not test bugs.

**Step 5: Commit**

```bash
git add sim_script/tests/test_wifi_first_macrocycle_assignment.py sim_script/tests/test_wifi_refill_scheduler.py
git commit -m "test: capture wifi-first admission priority"
```

### Task 2: Implement a true WiFi admission guarantee in the macrocycle pipeline

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_wifi_first_macrocycle_assignment.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_wifi_refill_scheduler.py`

**Step 1: Implement the minimal comparison rule**

Update `_is_better_refill_result(...)` so that when `wifi_first=True`, result ranking is lexicographic:

1. fewer unscheduled WiFi pairs
2. lower sum of unscheduled WiFi priority
3. fewer total unscheduled pairs
4. lower total unscheduled priority

Minimal sketch:

```python
def _is_better_refill_result(candidate, best, pair_priority, pair_radio_type=None, wifi_radio_id=0, wifi_first=False):
    ...
    if wifi_first:
        candidate_wifi_unscheduled = [i for i in candidate_unscheduled if pair_radio_type[i] == wifi_radio_id]
        best_wifi_unscheduled = [i for i in best_unscheduled if pair_radio_type[i] == wifi_radio_id]
        ...
```

**Step 2: Propagate the WiFi-first flag through refill**

Modify:

- `_refill_unscheduled_pairs_by_radio(...)`
- `_apply_refill_pipeline(...)`
- `retry_ble_channels_and_assign_macrocycle(...)`

so WiFi-first mode uses the stronger comparison rule during repair and refill passes.

**Step 3: Keep WiFi as the protected class during BLE retries**

Make sure BLE channel resampling retries cannot replace a better WiFi-admission solution with a BLE-heavier solution under `wifi_first=True`.

The decision point is already in:

```python
if _is_better_refill_result(candidate, best, e.pair_priority):
```

Change it to use WiFi-aware comparison metadata:

```python
if _is_better_refill_result(
    candidate,
    best,
    e.pair_priority,
    pair_radio_type=e.pair_radio_type,
    wifi_radio_id=e.RADIO_WIFI,
    wifi_first=wifi_first,
):
```

**Step 4: Run tests to verify they pass**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_wifi_first_macrocycle_assignment.py \
  sim_script/tests/test_wifi_refill_scheduler.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_wifi_first_macrocycle_assignment.py sim_script/tests/test_wifi_refill_scheduler.py
git commit -m "feat: protect wifi admission in wifi-first refill"
```

### Task 3: Add a failing BLE-overlap export test

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_per_ce_export_and_plot.py`
- Create: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_overlap_plot_highlighting.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_overlap_plot_highlighting.py`

**Step 1: Write a failing data-layer test**

Add a test that feeds two overlapping BLE plot spans and expects a derived overlap span with a distinct marker field.

```python
def test_build_schedule_plot_rows_emits_ble_overlap_segments():
    rows = [...]
    overlap_rows = [r for r in rows if r.get("overlay_kind") == "ble_overlap"]
    assert overlap_rows == [
        {
            "radio": "ble_overlap",
            "slot": 10,
            "freq_low_mhz": 2442.0,
            "freq_high_mhz": 2443.0,
            ...
        }
    ]
```

**Step 2: Write a failing render-layer test**

Add a rendering test that checks the plotting function includes a legend entry or color patch for BLE collision overlays.

```python
def test_render_event_grid_plot_draws_ble_overlap_layer():
    spans = [...]
    render_event_grid_plot(spans, out, macrocycle_slots=16)
    assert out.exists()
```

Monkeypatch `matplotlib.axes.Axes.add_patch` or inspect patch colors to verify the collision color is actually used.

**Step 3: Run tests to verify they fail**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_ble_per_ce_export_and_plot.py \
  sim_script/tests/test_ble_overlap_plot_highlighting.py -v
```

Expected: FAIL because no BLE-overlap spans or collision color are currently produced.

**Step 4: Re-run to ensure failures are stable**

Run the same command once more. Confirm failures are still due to missing overlap export/rendering behavior.

**Step 5: Commit**

```bash
git add sim_script/tests/test_ble_per_ce_export_and_plot.py sim_script/tests/test_ble_overlap_plot_highlighting.py
git commit -m "test: capture ble overlap highlighting behavior"
```

### Task 4: Implement BLE overlap export and highlighting

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/plot_schedule_from_csv.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_per_ce_export_and_plot.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_overlap_plot_highlighting.py`

**Step 1: Add a helper to derive BLE overlap spans**

In `pd_mmw_template_ap_stats.py`, add a helper that scans plot rows grouped by slot and computes pairwise BLE-BLE frequency intersections.

Minimal sketch:

```python
def build_ble_overlap_plot_rows(plot_rows):
    overlaps = []
    ...
    if lo < hi:
        overlaps.append(
            {
                "pair_id": -1,
                "radio": "ble_overlap",
                "channel": -1,
                "slot": slot,
                "freq_low_mhz": lo,
                "freq_high_mhz": hi,
                "label": f"BLE overlap {a_pair}/{b_pair}",
            }
        )
    return overlaps
```

**Step 2: Merge overlap rows into the plot export**

Update `build_schedule_plot_rows(...)` to optionally append these overlap rows after the base WiFi/BLE rows are created.

Keep YAGNI:

- only compute overlap from already scheduled rows
- only derive BLE-BLE overlap geometry
- do not change scheduling semantics

**Step 3: Render overlap rows with a distinct color**

In `plot_schedule_from_csv.py`:

- extend `RADIO_COLORS` with a collision color, e.g. `ble_overlap`
- ensure grouped spans preserve `radio == "ble_overlap"`
- include a legend patch labeled `BLE overlap`
- render the overlap layer after base BLE rectangles so the collision geometry is visible

Minimal sketch:

```python
RADIO_COLORS = {
    "wifi": "#0B6E4F",
    "ble": "#C84C09",
    "ble_overlap": "#D7263D",
}
```

**Step 4: Run tests to verify they pass**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_ble_per_ce_export_and_plot.py \
  sim_script/tests/test_ble_overlap_plot_highlighting.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/plot_schedule_from_csv.py sim_script/tests/test_ble_per_ce_export_and_plot.py sim_script/tests/test_ble_overlap_plot_highlighting.py
git commit -m "feat: highlight ble overlap regions in plots"
```

### Task 5: Run focused end-to-end verification

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/README.md`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_wifi_first_macrocycle_assignment.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_wifi_refill_scheduler.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_per_ce_export_and_plot.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_overlap_plot_highlighting.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_ble_channel_modes.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_ble_channel_retry.py`

**Step 1: Update README usage notes**

Add a short note that:

- `wifi_first` now prioritizes maximizing admitted WiFi pairs in macrocycle scheduling
- plots highlight BLE-BLE overlap regions with a distinct color

No large doc rewrite. Keep it to the scheduling/outputs sections.

**Step 2: Run the focused regression suite**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_wifi_first_macrocycle_assignment.py \
  sim_script/tests/test_wifi_refill_scheduler.py \
  sim_script/tests/test_ble_per_ce_export_and_plot.py \
  sim_script/tests/test_ble_overlap_plot_highlighting.py \
  sim_script/tests/test_pd_mmw_ble_channel_modes.py \
  sim_script/tests/test_pd_mmw_ble_channel_retry.py -v
```

Expected: PASS.

**Step 3: Run one script-level smoke test**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 sim_script/pd_mmw_template_ap_stats.py \
  --cell-size 1 \
  --pair-density 0.5 \
  --seed 123 \
  --mmw-nit 5 \
  --wifi-first-ble-scheduling \
  --ble-channel-retries 1 \
  --ble-channel-mode per_ce \
  --output-dir sim_script/output
```

Expected:

- exit code `0`
- `pair_parameters.csv` regenerated
- `schedule_plot_rows.csv` regenerated with overlap rows when collisions exist
- `wifi_ble_schedule_overview.png` and window plots regenerated

**Step 4: Inspect the regenerated outputs**

Manually verify:

- WiFi admission count does not regress relative to the previous WiFi-first policy for the same seed
- overlap color appears only on BLE-BLE collided regions
- non-overlap BLE regions keep the original BLE color

**Step 5: Commit**

```bash
git add README.md sim_script/pd_mmw_template_ap_stats.py sim_script/plot_schedule_from_csv.py sim_script/tests/test_wifi_first_macrocycle_assignment.py sim_script/tests/test_wifi_refill_scheduler.py sim_script/tests/test_ble_per_ce_export_and_plot.py sim_script/tests/test_ble_overlap_plot_highlighting.py
git commit -m "feat: maximize wifi admission and highlight ble collisions"
```
