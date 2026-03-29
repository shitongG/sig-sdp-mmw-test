"""Plotting and CSV export helpers for the isolated joint WiFi/BLE experiment."""

from __future__ import annotations

import csv
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from sim_script.plot_schedule_from_csv import render_all_from_csv

from .joint_wifi_ble_model import BLE_ADV_CHANNELS_MHZ, WIFI_CHANNEL_TO_MHZ, ble_data_channel_center_mhz

SLOT_MS = 1.25
PLOT_FIELDS = [
    "pair_id",
    "radio",
    "channel",
    "slot",
    "slot_width",
    "freq_low_mhz",
    "freq_high_mhz",
    "label",
    "event_index",
]
PAIR_FIELDS = [
    "pair_id",
    "office_id",
    "radio",
    "channel",
    "priority",
    "release_time_slot",
    "deadline_slot",
    "schedule_slot",
    "schedule_time_ms",
    "ble_channel_mode",
    "ble_ce_channel_summary",
    "start_time_slot",
    "wifi_anchor_slot",
    "wifi_period_slots",
    "wifi_period_ms",
    "wifi_tx_slots",
    "wifi_tx_ms",
    "ble_anchor_slot",
    "ble_ci_slots",
    "ble_ci_ms",
    "ble_ce_slots",
    "ble_ce_ms",
    "ble_ce_feasible",
    "effective_ble_channels",
    "scheduled_ble_pairs",
    "no_collision_probability",
    "macrocycle_slots",
    "occupied_slots_in_macrocycle",
]
SCHEDULE_FIELDS = [
    "schedule_slot",
    "pair_ids",
    "wifi_pair_ids",
    "ble_pair_ids",
    "pair_count",
    "wifi_pair_count",
    "ble_pair_count",
]
BLE_EVENT_FIELDS = [
    "pair_id",
    "event_index",
    "channel",
    "slot_start",
    "slot_end",
    "freq_low_mhz",
    "freq_high_mhz",
]


def _infer_wifi_channel(freq_center_mhz: float) -> int:
    for channel, center_mhz in WIFI_CHANNEL_TO_MHZ.items():
        if abs(float(center_mhz) - float(freq_center_mhz)) < 1e-6:
            return int(channel)
    raise ValueError(f"Unknown WiFi center frequency: {freq_center_mhz}")


def _infer_ble_channel(freq_center_mhz: float) -> int:
    for channel in range(37):
        if abs(ble_data_channel_center_mhz(channel) - float(freq_center_mhz)) < 1e-6:
            return int(channel)
    raise ValueError(f"Unknown BLE center frequency: {freq_center_mhz}")


def _normalize_task(task: Any) -> dict[str, Any]:
    if is_dataclass(task):
        return asdict(task)
    return dict(task)


def _normalize_tasks(tasks: Iterable[Any]) -> list[dict[str, Any]]:
    return [_normalize_task(task) for task in tasks]


def _selected_state_by_pair(result: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(state["pair_id"]): dict(state) for state in result.get("selected_states", [])}


def _blocks_by_pair(result: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    blocks_by_pair: dict[int, list[dict[str, Any]]] = {}
    for block in result.get("blocks", []):
        blocks_by_pair.setdefault(int(block["pair_id"]), []).append(dict(block))
    for blocks in blocks_by_pair.values():
        blocks.sort(key=lambda item: (int(item["slot_start"]), int(item.get("event_index", 0))))
    return blocks_by_pair


def _occupied_slots(blocks: list[dict[str, Any]]) -> list[int]:
    slots: set[int] = set()
    for block in blocks:
        slots.update(range(int(block["slot_start"]), int(block["slot_end"])))
    return sorted(slots)


def _ble_channel_summary(blocks: list[dict[str, Any]]) -> list[int]:
    channels: list[int] = []
    for block in sorted(blocks, key=lambda item: int(item.get("event_index", 0))):
        center = (float(block["freq_low_mhz"]) + float(block["freq_high_mhz"])) / 2.0
        channels.append(_infer_ble_channel(center))
    return channels


def build_pair_parameter_rows(result: Mapping[str, Any], tasks: Iterable[Any], macrocycle_slots: int) -> list[dict[str, Any]]:
    task_rows = _normalize_tasks(tasks)
    selected_by_pair = _selected_state_by_pair(result)
    blocks_by_pair = _blocks_by_pair(result)
    rows: list[dict[str, Any]] = []
    for task in sorted(task_rows, key=lambda item: int(item["task_id"])):
        pair_id = int(task["task_id"])
        radio = str(task["radio"])
        state = selected_by_pair.get(pair_id)
        blocks = blocks_by_pair.get(pair_id, [])
        occupied_slots = _occupied_slots(blocks)
        schedule_slot = int(state["offset"]) if state is not None else -1
        if radio == "wifi":
            channel = int(state["channel"]) if state is not None and state.get("channel") is not None else int(task.get("preferred_channel", 0) or 0)
            wifi_period_slots = int(state.get("period_slots") or task.get("wifi_period_slots") or task.get("wifi_tx_slots") or 1) if state is not None else int(task.get("wifi_period_slots") or task.get("wifi_tx_slots") or 1)
            wifi_tx_slots = int(state.get("width_slots") or task.get("wifi_tx_slots") or 1) if state is not None else int(task.get("wifi_tx_slots") or 1)
            row = {
                "pair_id": pair_id,
                "office_id": 0,
                "radio": radio,
                "channel": channel,
                "priority": "1.000",
                "release_time_slot": int(task["release_slot"]),
                "deadline_slot": int(task["deadline_slot"]),
                "schedule_slot": schedule_slot,
                "schedule_time_ms": f"{schedule_slot * SLOT_MS:.3f}" if schedule_slot >= 0 else f"{-SLOT_MS:.3f}",
                "ble_channel_mode": "NA",
                "ble_ce_channel_summary": "NA",
                "start_time_slot": int(task["release_slot"]),
                "wifi_anchor_slot": schedule_slot if schedule_slot >= 0 else "NA",
                "wifi_period_slots": wifi_period_slots,
                "wifi_period_ms": f"{wifi_period_slots * SLOT_MS:.3f}",
                "wifi_tx_slots": wifi_tx_slots,
                "wifi_tx_ms": f"{wifi_tx_slots * SLOT_MS:.3f}",
                "ble_anchor_slot": "NA",
                "ble_ci_slots": "NA",
                "ble_ci_ms": "NA",
                "ble_ce_slots": "NA",
                "ble_ce_ms": "NA",
                "ble_ce_feasible": "NA",
                "effective_ble_channels": "NA",
                "scheduled_ble_pairs": "NA",
                "no_collision_probability": "NA",
                "macrocycle_slots": int(macrocycle_slots),
                "occupied_slots_in_macrocycle": str(occupied_slots),
            }
        else:
            channel = int(state["channel"]) if state is not None and state.get("channel") is not None else int(task.get("preferred_channel", 0) or 0)
            ble_ci_slots = int(state.get("ci_slots") or (task.get("ble_ci_slots_options") or [1])[0]) if state is not None else int((task.get("ble_ci_slots_options") or [1])[0])
            ble_ce_slots = int(state.get("ce_slots") or task.get("ble_ce_slots") or 1) if state is not None else int(task.get("ble_ce_slots") or 1)
            row = {
                "pair_id": pair_id,
                "office_id": 0,
                "radio": radio,
                "channel": channel,
                "priority": "1.000",
                "release_time_slot": int(task["release_slot"]),
                "deadline_slot": int(task["deadline_slot"]),
                "schedule_slot": schedule_slot,
                "schedule_time_ms": f"{schedule_slot * SLOT_MS:.3f}" if schedule_slot >= 0 else f"{-SLOT_MS:.3f}",
                "ble_channel_mode": "per_ce",
                "ble_ce_channel_summary": str(_ble_channel_summary(blocks)) if blocks else str([]),
                "start_time_slot": int(task["release_slot"]),
                "wifi_anchor_slot": "NA",
                "wifi_period_slots": "NA",
                "wifi_period_ms": "NA",
                "wifi_tx_slots": "NA",
                "wifi_tx_ms": "NA",
                "ble_anchor_slot": schedule_slot if schedule_slot >= 0 else int(task["release_slot"]),
                "ble_ci_slots": ble_ci_slots,
                "ble_ci_ms": f"{ble_ci_slots * SLOT_MS:.3f}",
                "ble_ce_slots": ble_ce_slots,
                "ble_ce_ms": f"{ble_ce_slots * SLOT_MS:.3f}",
                "ble_ce_feasible": "true",
                "effective_ble_channels": "NA",
                "scheduled_ble_pairs": "NA",
                "no_collision_probability": "NA",
                "macrocycle_slots": int(macrocycle_slots),
                "occupied_slots_in_macrocycle": str(occupied_slots),
            }
        rows.append(row)
    return rows


def build_schedule_rows(result: Mapping[str, Any], tasks: Iterable[Any], macrocycle_slots: int) -> list[dict[str, Any]]:
    task_rows = {int(task["task_id"]): _normalize_task(task) for task in tasks}
    blocks = [dict(block) for block in result.get("blocks", [])]
    rows: list[dict[str, Any]] = []
    for slot in range(int(macrocycle_slots)):
        pair_ids = sorted({int(block["pair_id"]) for block in blocks if int(block["slot_start"]) <= slot < int(block["slot_end"])})
        if not pair_ids:
            continue
        wifi_pair_ids = [pair_id for pair_id in pair_ids if task_rows[pair_id]["radio"] == "wifi"]
        ble_pair_ids = [pair_id for pair_id in pair_ids if task_rows[pair_id]["radio"] == "ble"]
        rows.append(
            {
                "schedule_slot": slot,
                "pair_ids": str(pair_ids),
                "wifi_pair_ids": str(wifi_pair_ids),
                "ble_pair_ids": str(ble_pair_ids),
                "pair_count": len(pair_ids),
                "wifi_pair_count": len(wifi_pair_ids),
                "ble_pair_count": len(ble_pair_ids),
            }
        )
    return rows


def build_ble_ce_event_rows(result: Mapping[str, Any], tasks: Iterable[Any]) -> list[dict[str, Any]]:
    task_rows = {int(task["task_id"]): _normalize_task(task) for task in tasks}
    rows: list[dict[str, Any]] = []
    for block in result.get("blocks", []):
        pair_id = int(block["pair_id"])
        if task_rows[pair_id]["radio"] != "ble":
            continue
        center = (float(block["freq_low_mhz"]) + float(block["freq_high_mhz"])) / 2.0
        rows.append(
            {
                "pair_id": pair_id,
                "event_index": int(block.get("event_index", 0)),
                "channel": _infer_ble_channel(center),
                "slot_start": int(block["slot_start"]),
                "slot_end": int(block["slot_end"]),
                "freq_low_mhz": f"{float(block['freq_low_mhz']):.3f}",
                "freq_high_mhz": f"{float(block['freq_high_mhz']):.3f}",
            }
        )
    return rows


def build_main_style_plot_rows(result: dict[str, Any], macrocycle_slots: int | None = None) -> list[dict[str, Any]]:
    """Translate solver output into the row schema consumed by the main renderer."""

    blocks = list(result.get("blocks", []))
    if macrocycle_slots is None:
        macrocycle_slots = max((int(block["slot_end"]) for block in blocks), default=1)

    rows: list[dict[str, Any]] = []
    for block in blocks:
        medium = str(block["medium"])
        freq_low = float(block["freq_low_mhz"])
        freq_high = float(block["freq_high_mhz"])
        center = (freq_low + freq_high) / 2.0
        if medium == "wifi":
            channel = _infer_wifi_channel(center)
            radio = "wifi"
        elif medium == "ble":
            channel = _infer_ble_channel(center)
            radio = "ble"
        else:
            channel = int(block.get("channel", -1))
            radio = medium
        label = str(block.get("label", f"{radio}-{int(block.get('pair_id', -1))}"))
        event_index = int(block.get("event_index", 0))
        for slot in range(int(block["slot_start"]), int(block["slot_end"])):
            rows.append(
                {
                    "pair_id": int(block["pair_id"]),
                    "radio": radio,
                    "channel": channel,
                    "slot": slot,
                    "slot_width": 1,
                    "freq_low_mhz": freq_low,
                    "freq_high_mhz": freq_high,
                    "label": label,
                    "event_index": event_index,
                }
            )

    for idle_index, center in enumerate(BLE_ADV_CHANNELS_MHZ):
        for slot in range(int(macrocycle_slots)):
            rows.append(
                {
                    "pair_id": -1,
                    "radio": "ble_adv_idle",
                    "channel": int(center),
                    "slot": slot,
                    "slot_width": 1,
                    "freq_low_mhz": float(center) - 1.0,
                    "freq_high_mhz": float(center) + 1.0,
                    "label": f"BLE adv idle {center:.0f} MHz",
                    "event_index": idle_index,
                }
            )

    return rows


def build_plot_payload(result: dict[str, Any], title: str = "joint_wifi_ble_schedule", macrocycle_slots: int | None = None) -> dict[str, Any]:
    return {"title": title, "rows": build_main_style_plot_rows(result, macrocycle_slots=macrocycle_slots)}


def _write_rows(output_path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path


def write_main_style_schedule_csv(output_dir: str | Path, rows: list[dict[str, Any]]) -> Path:
    return _write_rows(Path(output_dir) / "schedule_plot_rows.csv", PLOT_FIELDS, rows)


def render_joint_schedule(result: dict[str, Any], output_dir: str | Path, macrocycle_slots: int, window_slots: int = 128) -> tuple[Path, list[Path]]:
    rows = build_main_style_plot_rows(result, macrocycle_slots=macrocycle_slots)
    write_main_style_schedule_csv(output_dir, rows)
    return render_all_from_csv(output_dir, macrocycle_slots, window_slots=window_slots)


def export_joint_output_artifacts(result: Mapping[str, Any], tasks: Iterable[Any], output_dir: str | Path, macrocycle_slots: int, window_slots: int = 128) -> dict[str, str]:
    output_dir = Path(output_dir)
    task_rows = _normalize_tasks(tasks)
    pair_rows = build_pair_parameter_rows(result, task_rows, macrocycle_slots)
    unscheduled_pair_ids = set(int(pair_id) for pair_id in result.get("unscheduled_pair_ids", []))
    unscheduled_rows = [row for row in pair_rows if int(row["pair_id"]) in unscheduled_pair_ids]
    schedule_rows = build_schedule_rows(result, task_rows, macrocycle_slots)
    ble_event_rows = build_ble_ce_event_rows(result, task_rows)
    plot_rows = build_main_style_plot_rows(dict(result), macrocycle_slots=macrocycle_slots)

    pair_path = _write_rows(output_dir / "pair_parameters.csv", PAIR_FIELDS, pair_rows)
    schedule_path = _write_rows(output_dir / "wifi_ble_schedule.csv", SCHEDULE_FIELDS, schedule_rows)
    unscheduled_path = _write_rows(output_dir / "unscheduled_pairs.csv", PAIR_FIELDS, unscheduled_rows)
    plot_csv_path = _write_rows(output_dir / "schedule_plot_rows.csv", PLOT_FIELDS, plot_rows)
    ble_event_path = _write_rows(output_dir / "ble_ce_channel_events.csv", BLE_EVENT_FIELDS, ble_event_rows)
    overview_path, window_paths = render_all_from_csv(output_dir, macrocycle_slots, window_slots=window_slots)
    legacy_plot_path = output_dir / "wifi_ble_schedule.png"
    shutil.copyfile(overview_path, legacy_plot_path)
    return {
        "pair_parameters_csv": str(pair_path),
        "wifi_ble_schedule_csv": str(schedule_path),
        "unscheduled_pairs_csv": str(unscheduled_path),
        "schedule_plot_rows_csv": str(plot_csv_path),
        "schedule_plot_rows_path": str(plot_csv_path),
        "ble_ce_channel_events_csv": str(ble_event_path),
        "wifi_ble_schedule_png": str(legacy_plot_path),
        "wifi_ble_schedule_overview_png": str(overview_path),
        "overview_path": str(overview_path),
        "window_paths": [str(path) for path in window_paths],
        "window_count": str(len(window_paths)),
    }
