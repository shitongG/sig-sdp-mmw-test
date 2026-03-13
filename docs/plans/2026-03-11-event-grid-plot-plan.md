# Event Grid Plot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the unreadable per-slot schedule image with a CSV-driven event-grid visualization that preserves CE event boundaries and produces both a full-macrocycle overview and readable windowed detail plots.

**Architecture:** Keep the scheduler and current CSV exports as the source of truth. Add a dedicated plotting script that reads `pair_parameters.csv`, `schedule_plot_rows.csv`, and `ble_ce_channel_events.csv`, converts them into event spans, renders a lightweight overview plot for the full macrocycle, and emits windowed detail plots for dense regions. Leave scheduling behavior unchanged.

**Tech Stack:** Python, csv, pathlib, matplotlib, pytest, existing simulation CSV exports.

---

### Task 1: Lock the current failure mode with a plotting regression test

**Files:**
- Create: `sim_script/tests/test_event_grid_plot_labels.py`
- Test: `sim_script/tests/test_event_grid_plot_labels.py`

**Step 1: Write the failing test**

```python
def test_event_grid_plot_labels_once_per_event():
    event_rows = [
        {"pair_id": 7, "radio": "ble", "channel": 3, "event_index": 0, "slot_start": 10, "slot_end": 14,
         "freq_low_mhz": 2408.0, "freq_high_mhz": 2410.0, "label": "7 B-ch3 ev0"}
    ]
    labels = build_event_text_annotations(event_rows)
    assert len(labels) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_event_grid_plot_labels.py -v`
Expected: FAIL because helper does not exist.

**Step 3: Write minimal implementation**

```python
# Test only in this task.
```

**Step 4: Run test to verify it passes later when helper exists**

Run: `pytest sim_script/tests/test_event_grid_plot_labels.py -v`
Expected: still FAIL until Task 3 implements the helper.

**Step 5: Commit**

```bash
git add sim_script/tests/test_event_grid_plot_labels.py
git commit -m "test: lock event-grid labeling behavior"
```

### Task 2: Add a CSV-driven event-span builder

**Files:**
- Create: `sim_script/plot_schedule_from_csv.py`
- Create: `sim_script/tests/test_event_span_builder.py`
- Test: `sim_script/tests/test_event_span_builder.py`

**Step 1: Write the failing test**

```python
def test_build_event_spans_from_ble_event_csv():
    rows = [{
        "pair_id": "5", "event_index": "2", "channel": "17",
        "slot_start": "20", "slot_end": "24",
        "freq_low_mhz": "2435.0", "freq_high_mhz": "2437.0",
    }]
    spans = build_ble_event_spans(rows)
    assert spans[0]["pair_id"] == 5
    assert spans[0]["slot_start"] == 20
    assert spans[0]["slot_end"] == 24
    assert spans[0]["label"] == "5 B-ch17 ev2"
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_event_span_builder.py::test_build_event_spans_from_ble_event_csv -v`
Expected: FAIL because script/helper does not exist.

**Step 3: Write minimal implementation**

```python
def build_ble_event_spans(rows):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_event_span_builder.py::test_build_event_spans_from_ble_event_csv -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/plot_schedule_from_csv.py sim_script/tests/test_event_span_builder.py
git commit -m "feat: add BLE event-span builder from CSV"
```

### Task 3: Build contiguous WiFi/BLE-single event spans from per-slot CSV rows

**Files:**
- Modify: `sim_script/plot_schedule_from_csv.py`
- Create: `sim_script/tests/test_contiguous_span_grouping.py`
- Test: `sim_script/tests/test_contiguous_span_grouping.py`

**Step 1: Write the failing test**

```python
def test_group_contiguous_slot_rows_into_one_event_span():
    rows = [
        {"pair_id": "1", "radio": "wifi", "channel": "9", "slot": "10", "freq_low_mhz": "2447.0", "freq_high_mhz": "2467.0", "label": "1 W-ch9"},
        {"pair_id": "1", "radio": "wifi", "channel": "9", "slot": "11", "freq_low_mhz": "2447.0", "freq_high_mhz": "2467.0", "label": "1 W-ch9"},
    ]
    spans = group_slot_rows_into_event_spans(rows)
    assert len(spans) == 1
    assert spans[0]["slot_start"] == 10
    assert spans[0]["slot_end"] == 12
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_contiguous_span_grouping.py::test_group_contiguous_slot_rows_into_one_event_span -v`
Expected: FAIL because grouping helper does not exist.

**Step 3: Write minimal implementation**

```python
def group_slot_rows_into_event_spans(rows):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_contiguous_span_grouping.py::test_group_contiguous_slot_rows_into_one_event_span -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/plot_schedule_from_csv.py sim_script/tests/test_contiguous_span_grouping.py
git commit -m "feat: group contiguous schedule rows into event spans"
```

### Task 4: Render overview plots from event spans without per-slot repeated labels

**Files:**
- Modify: `sim_script/plot_schedule_from_csv.py`
- Create: `sim_script/tests/test_event_grid_plot_render.py`
- Test: `sim_script/tests/test_event_grid_plot_render.py`

**Step 1: Write the failing test**

```python
def test_render_overview_plot_writes_png_and_places_one_label_per_event(tmp_path):
    spans = [{
        "pair_id": 7, "radio": "ble", "channel": 3, "event_index": 0,
        "slot_start": 10, "slot_end": 14,
        "freq_low_mhz": 2408.0, "freq_high_mhz": 2410.0,
        "label": "7 B-ch3 ev0",
    }]
    out = tmp_path / "overview.png"
    render_event_grid_plot(spans, out, macrocycle_slots=64)
    assert out.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_event_grid_plot_render.py::test_render_overview_plot_writes_png_and_places_one_label_per_event -v`
Expected: FAIL because renderer does not exist.

**Step 3: Write minimal implementation**

```python
def render_event_grid_plot(...):
    # draw one rectangle per event span, one text annotation per event span
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_event_grid_plot_render.py::test_render_overview_plot_writes_png_and_places_one_label_per_event -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/plot_schedule_from_csv.py sim_script/tests/test_event_grid_plot_render.py
git commit -m "feat: render event-grid overview plot"
```

### Task 5: Add windowed detail plot generation

**Files:**
- Modify: `sim_script/plot_schedule_from_csv.py`
- Create: `sim_script/tests/test_event_grid_windowed_plot.py`
- Test: `sim_script/tests/test_event_grid_windowed_plot.py`

**Step 1: Write the failing test**

```python
def test_render_windowed_plots_splits_macrocycle_into_multiple_pngs(tmp_path):
    spans = [
        {"pair_id": 1, "radio": "wifi", "channel": 9, "slot_start": 0, "slot_end": 20,
         "freq_low_mhz": 2447.0, "freq_high_mhz": 2467.0, "label": "1 W-ch9"},
        {"pair_id": 2, "radio": "ble", "channel": 3, "event_index": 0, "slot_start": 130, "slot_end": 134,
         "freq_low_mhz": 2408.0, "freq_high_mhz": 2410.0, "label": "2 B-ch3 ev0"},
    ]
    outputs = render_windowed_event_grid_plots(spans, tmp_path, macrocycle_slots=256, window_slots=128)
    assert len(outputs) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_event_grid_windowed_plot.py::test_render_windowed_plots_splits_macrocycle_into_multiple_pngs -v`
Expected: FAIL because helper does not exist.

**Step 3: Write minimal implementation**

```python
def render_windowed_event_grid_plots(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_event_grid_windowed_plot.py::test_render_windowed_plots_splits_macrocycle_into_multiple_pngs -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/plot_schedule_from_csv.py sim_script/tests/test_event_grid_windowed_plot.py
git commit -m "feat: add windowed event-grid detail plots"
```

### Task 6: Wire the main simulation script to call the CSV-driven plotter

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/tests/test_pd_mmw_ble_channel_modes.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

```python
def test_per_ce_mode_writes_overview_and_windowed_plot_outputs(tmp_path):
    completed = subprocess.run([...], ...)
    assert (tmp_path / "wifi_ble_schedule_overview.png").exists()
    assert any(p.name.startswith("wifi_ble_schedule_window_") for p in tmp_path.iterdir())
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_pd_mmw_ble_channel_modes.py::test_per_ce_mode_writes_overview_and_windowed_plot_outputs -v`
Expected: FAIL because outputs do not exist.

**Step 3: Write minimal implementation**

```python
# After CSV export, invoke the CSV-driven plotting helpers/script.
# Keep existing summary image name for backward compatibility.
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_pd_mmw_ble_channel_modes.py::test_per_ce_mode_writes_overview_and_windowed_plot_outputs -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_ble_channel_modes.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: wire CSV-driven schedule plotting into main script"
```

### Task 7: Replace the old unreadable plot assertions with event-grid assertions

**Files:**
- Modify: `sim_script/tests/test_schedule_plot_render.py`
- Modify: `sim_script/tests/test_schedule_plot_axis.py`
- Modify: `sim_script/tests/test_schedule_plot_rows.py`

**Step 1: Write the failing test**

```python
def test_event_grid_plot_uses_event_spans_not_per_slot_repeated_labels():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest sim_script/tests/test_schedule_plot_render.py sim_script/tests/test_schedule_plot_axis.py sim_script/tests/test_schedule_plot_rows.py -q`
Expected: FAIL where expectations still reflect the old renderer.

**Step 3: Write minimal implementation**

```python
# Update tests to reflect the CSV-driven event-grid renderer and new output filenames.
```

**Step 4: Run test to verify it passes**

Run: `pytest sim_script/tests/test_schedule_plot_render.py sim_script/tests/test_schedule_plot_axis.py sim_script/tests/test_schedule_plot_rows.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add sim_script/tests/test_schedule_plot_render.py sim_script/tests/test_schedule_plot_axis.py sim_script/tests/test_schedule_plot_rows.py
git commit -m "test: update plotting assertions for event-grid renderer"
```

### Task 8: Run focused end-to-end verification

**Files:**
- Check: `sim_script/plot_schedule_from_csv.py`
- Check: `sim_script/pd_mmw_template_ap_stats.py`
- Check: `docs/plans/2026-03-11-event-grid-plot-design.md`
- Check: `docs/plans/2026-03-11-event-grid-plot-plan.md`

**Step 1: Run focused test suite**

Run:
```bash
pytest \
  sim_script/tests/test_event_grid_plot_labels.py \
  sim_script/tests/test_event_span_builder.py \
  sim_script/tests/test_contiguous_span_grouping.py \
  sim_script/tests/test_event_grid_plot_render.py \
  sim_script/tests/test_event_grid_windowed_plot.py \
  sim_script/tests/test_pd_mmw_ble_channel_modes.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q
```
Expected: PASS

**Step 2: Run script smoke test**

Run:
```bash
python sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.5 --seed 123 --mmw-nit 5 --ble-channel-mode per_ce --output-dir /tmp/event_grid_plot_check
```
Expected:
- exit code `0`
- output directory contains:
  - `pair_parameters.csv`
  - `schedule_plot_rows.csv`
  - `ble_ce_channel_events.csv`
  - `wifi_ble_schedule.png`
  - `wifi_ble_schedule_overview.png`
  - one or more `wifi_ble_schedule_window_*.png`

**Step 3: Inspect output inventory**

Run:
```bash
ls /tmp/event_grid_plot_check
```
Expected: both CSV files and new plot files present.

**Step 4: Commit**

```bash
git add sim_script/plot_schedule_from_csv.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests docs/plans/2026-03-11-event-grid-plot-design.md docs/plans/2026-03-11-event-grid-plot-plan.md
git commit -m "feat: add CSV-driven event-grid schedule plots"
```
