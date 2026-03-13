# Event Grid Plot Design

**Goal:** Replace the current unreadable full-macrocycle schedule image with a CSV-driven visualization pipeline that preserves per-event BLE/WiFi occupancy boundaries without merging different CE events.

## Problem
The current plot renders one rectangle per occupied slot and one text label per rectangle. For large macrocycles, this produces thousands of black edges and text glyphs, making the image unreadable. A naive interval merge would reduce clutter but would incorrectly collapse different CE events across CI boundaries, which is not acceptable for this project.

## Chosen Approach
Use a CSV-driven event-grid plotting pipeline.

- Keep the existing scheduler and CSV exports as the source of truth.
- Add a dedicated plotting script that reads exported CSV files instead of depending on in-memory scheduler state.
- Preserve event boundaries by plotting event-level spans, not merged cross-event intervals.
- Reduce clutter by placing at most one label per event, centered over the event span, instead of one label per occupied slot.

## Outputs
1. Summary full-macrocycle figure
- Covers the full macrocycle `0..macrocycle_slots-1`
- Shows all scheduled events as colored spans on the frequency axis
- Uses sparse labeling rules so the figure remains readable

2. Windowed detail figures
- Splits the macrocycle into fixed slot windows
- Produces multiple zoomed figures with event labels visible
- Lets users inspect dense regions without losing event identity

3. CSV-first workflow
- Main simulation script continues exporting:
  - `pair_parameters.csv`
  - `schedule_plot_rows.csv`
  - `ble_ce_channel_events.csv`
- New plotting script reads these CSVs and writes figures

## Data Model
Use `ble_ce_channel_events.csv` as the primary event-level source for BLE in `per_ce` mode.

Each row represents one CE event with:
- `pair_id`
- `event_index`
- `channel`
- `slot_start`
- `slot_end`
- `freq_low_mhz`
- `freq_high_mhz`

For WiFi and for BLE `single` mode, derive event spans from `schedule_plot_rows.csv` by grouping contiguous rows that share:
- `pair_id`
- `channel`
- `freq_low_mhz`
- `freq_high_mhz`

This preserves separate events while still compressing repeated per-slot rows into one visual span per event.

## Labeling Strategy
- Full-macrocycle plot:
  - No label on very short spans
  - One label per event on spans wide enough to hold text
  - Optional cap on labels per viewport region if density is extreme
- Windowed plots:
  - Label every event that fits
  - Format:
    - WiFi: `pair_id W-chX`
    - BLE single: `pair_id B-chX`
    - BLE per_ce: `pair_id B-chX evN`

## Rendering Strategy
- Draw event spans as rectangles from `slot_start` to `slot_end`
- Use lighter or no edgecolor by default
- Keep radio-color legend
- Keep the full macrocycle x-axis for the overview figure
- Add a second script option for fixed-width windows, for example `--window-slots 128`

## Why This Design
This preserves the semantics that matter:
- different CE events stay distinct
- different channels per CE remain visible
- the plotting logic becomes iteratable without touching scheduling code

It also fixes the current failure mode:
- no more one-label-per-slot repetition
- no more heavy black edge density from thousands of 1-slot boxes

## Scope Boundaries
This design does not change:
- scheduler feasibility logic
- CI/CE timing semantics
- macrocycle assignment behavior

This design only changes:
- how plotting rows are consumed
- how images are rendered
- how much event-level detail is exposed in figures
