import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


RADIO_COLORS = {"wifi": "#0B6E4F", "ble": "#C84C09", "ble_overlap": "#D7263D"}


def build_ble_event_spans(rows):
    spans = []
    for row in rows:
        pair_id = int(row["pair_id"])
        event_index = int(row["event_index"])
        channel = int(row["channel"])
        spans.append(
            {
                "pair_id": pair_id,
                "radio": "ble",
                "channel": channel,
                "event_index": event_index,
                "slot_start": int(row["slot_start"]),
                "slot_end": int(row["slot_end"]),
                "freq_low_mhz": float(row["freq_low_mhz"]),
                "freq_high_mhz": float(row["freq_high_mhz"]),
                "label": f"{pair_id} B-ch{channel} ev{event_index}",
            }
        )
    return spans


def group_slot_rows_into_event_spans(rows):
    parsed = []
    for row in rows:
        parsed.append(
            {
                "pair_id": int(row["pair_id"]),
                "radio": row["radio"],
                "channel": int(row["channel"]),
                "slot": int(row["slot"]),
                "freq_low_mhz": float(row["freq_low_mhz"]),
                "freq_high_mhz": float(row["freq_high_mhz"]),
                "label": row["label"],
            }
        )
    parsed.sort(key=lambda r: (r["pair_id"], r["channel"], r["freq_low_mhz"], r["freq_high_mhz"], r["slot"]))
    spans = []
    current = None
    for row in parsed:
        key = (row["pair_id"], row["radio"], row["channel"], row["freq_low_mhz"], row["freq_high_mhz"], row["label"])
        if current is None:
            current = {**row, "slot_start": row["slot"], "slot_end": row["slot"] + 1, "_key": key}
            continue
        if current["_key"] == key and row["slot"] == current["slot_end"]:
            current["slot_end"] += 1
            continue
        spans.append({k: v for k, v in current.items() if k not in {"slot", "_key"}})
        current = {**row, "slot_start": row["slot"], "slot_end": row["slot"] + 1, "_key": key}
    if current is not None:
        spans.append({k: v for k, v in current.items() if k not in {"slot", "_key"}})
    return spans


def build_event_text_annotations(event_spans, min_span_slots=2):
    annotations = []
    for span in event_spans:
        span_width = int(span["slot_end"]) - int(span["slot_start"])
        if span_width < int(min_span_slots):
            continue
        annotations.append(
            {
                "x": (float(span["slot_start"]) + float(span["slot_end"])) / 2.0,
                "y": (float(span["freq_low_mhz"]) + float(span["freq_high_mhz"])) / 2.0,
                "text": span["label"],
            }
        )
    return annotations


def render_event_grid_plot(event_spans, output_path, macrocycle_slots, slot_window=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 6))
    for span in event_spans:
        if slot_window is not None:
            win0, win1 = slot_window
            if int(span["slot_end"]) <= win0 or int(span["slot_start"]) >= win1:
                continue
            x0 = max(int(span["slot_start"]), win0)
            x1 = min(int(span["slot_end"]), win1)
        else:
            x0 = int(span["slot_start"])
            x1 = int(span["slot_end"])
        rect = Rectangle(
            (x0, float(span["freq_low_mhz"])),
            max(0.8, x1 - x0),
            float(span["freq_high_mhz"]) - float(span["freq_low_mhz"]),
            facecolor=RADIO_COLORS.get(span["radio"], "#4C4C4C"),
            edgecolor="none",
            linewidth=0.0,
            alpha=0.75,
        )
        ax.add_patch(rect)

    for ann in build_event_text_annotations(event_spans):
        if slot_window is not None:
            if not (slot_window[0] <= ann["x"] <= slot_window[1]):
                continue
        ax.text(ann["x"], ann["y"], ann["text"], ha="center", va="center", fontsize=7, color="black", clip_on=True)

    if slot_window is None:
        ax.set_xlim(0, int(macrocycle_slots))
    else:
        ax.set_xlim(int(slot_window[0]), int(slot_window[1]))
    if event_spans:
        ax.set_ylim(min(float(s["freq_low_mhz"]) for s in event_spans), max(float(s["freq_high_mhz"]) for s in event_spans))
    ax.set_xlabel("Slot")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("WiFi/BLE Event Grid")
    ax.grid(True, axis="x", alpha=0.15)
    ax.legend(
        handles=[
            Patch(facecolor=RADIO_COLORS["wifi"], edgecolor="none", alpha=0.75, label="WiFi"),
            Patch(facecolor=RADIO_COLORS["ble"], edgecolor="none", alpha=0.75, label="BLE"),
            Patch(facecolor=RADIO_COLORS["ble_overlap"], edgecolor="none", alpha=0.75, label="BLE overlap"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_windowed_event_grid_plots(event_spans, output_dir, macrocycle_slots, window_slots=128, prefix="wifi_ble_schedule_window"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    n_windows = int(math.ceil(int(macrocycle_slots) / int(window_slots)))
    for idx in range(n_windows):
        start = idx * int(window_slots)
        end = min(int(macrocycle_slots), start + int(window_slots))
        out = output_dir / f"{prefix}_{idx:03d}.png"
        render_event_grid_plot(event_spans, out, macrocycle_slots, slot_window=(start, end))
        outputs.append(out)
    return outputs


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_event_spans_from_csv(output_dir):
    output_dir = Path(output_dir)
    ble_rows = read_csv_rows(output_dir / "ble_ce_channel_events.csv")
    plot_rows = read_csv_rows(output_dir / "schedule_plot_rows.csv")
    spans = []
    if ble_rows:
        spans.extend(build_ble_event_spans(ble_rows))
        ble_pair_ids = {int(r["pair_id"]) for r in ble_rows}
        plot_rows = [r for r in plot_rows if not (r["radio"] == "ble" and int(r["pair_id"]) in ble_pair_ids)]
    if plot_rows:
        spans.extend(group_slot_rows_into_event_spans(plot_rows))
    return spans


def render_all_from_csv(output_dir, macrocycle_slots, window_slots=128):
    output_dir = Path(output_dir)
    spans = build_event_spans_from_csv(output_dir)
    overview = output_dir / "wifi_ble_schedule_overview.png"
    render_event_grid_plot(spans, overview, macrocycle_slots)
    windows = render_windowed_event_grid_plots(spans, output_dir, macrocycle_slots, window_slots=window_slots)
    return overview, windows
