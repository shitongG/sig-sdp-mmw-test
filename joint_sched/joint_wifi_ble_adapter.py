"""Adapters for importing mainline pair parameters into the joint scheduler."""

from __future__ import annotations

import ast
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .joint_wifi_ble_model import (
    DEFAULT_BLE_BYTES_PER_CE_SLOT,
    DEFAULT_WIFI_BYTES_PER_SLOT,
    JointTaskSpec,
)
from .joint_wifi_ble_random import BLE_CHANNELS, WIFI_CHANNELS


def _parse_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.upper() == "NA":
        return default
    return int(float(text))


def _parse_occupied_slots(value: Any) -> list[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return []
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, list):
        return []
    return sorted({int(slot) for slot in parsed})


def _count_contiguous_segments(slots: list[int]) -> int:
    if not slots:
        return 0
    segments = 1
    for left, right in zip(slots, slots[1:]):
        if right != left + 1:
            segments += 1
    return segments


def _infer_wifi_num_events(occupied_slots: list[int], width_slots: int) -> int:
    if not occupied_slots:
        return 0
    if width_slots <= 0:
        return _count_contiguous_segments(occupied_slots)
    return max(1, len(occupied_slots) // width_slots)


def _suggest_max_offsets(
    release_slot: int,
    deadline_slot: int,
    step_slots: int,
    width_slots: int,
    num_events: int = 1,
    limit: int = 4,
    cyclic_periodic: bool = False,
    macrocycle_slots: int | None = None,
) -> int:
    if cyclic_periodic and macrocycle_slots is not None and macrocycle_slots > 0:
        feasible_count = max(0, min(deadline_slot, macrocycle_slots - 1) - release_slot + 1)
        return max(1, min(limit, feasible_count))
    latest_offset = deadline_slot - max(0, num_events - 1) * step_slots - width_slots + 1
    feasible_count = max(0, latest_offset - release_slot + 1)
    return max(1, min(limit, feasible_count))


def load_mainline_pair_parameter_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def build_joint_tasks_from_mainline_pair_parameter_rows(rows: list[Mapping[str, Any]]) -> list[JointTaskSpec]:
    tasks: list[JointTaskSpec] = []
    for row in sorted(rows, key=lambda item: int(_parse_int(item.get("pair_id"), 0) or 0)):
        pair_id = int(_parse_int(row.get("pair_id"), 0) or 0)
        radio = str(row.get("radio", "")).strip().lower()
        release_slot = int(_parse_int(row.get("release_time_slot"), 0) or 0)
        deadline_slot = int(_parse_int(row.get("deadline_slot"), release_slot) or release_slot)
        preferred_channel = _parse_int(row.get("channel"))
        occupied_slots = _parse_occupied_slots(row.get("occupied_slots_in_macrocycle"))

        if radio == "wifi":
            wifi_tx_slots = int(_parse_int(row.get("wifi_tx_slots"), 1) or 1)
            wifi_period_slots = int(_parse_int(row.get("wifi_period_slots"), wifi_tx_slots) or wifi_tx_slots)
            repetitions = max(1, _infer_wifi_num_events(occupied_slots, wifi_tx_slots) or 1)
            schedule_slot = int(_parse_int(row.get("schedule_slot"), -1) or -1)
            wifi_max_offsets = _suggest_max_offsets(
                release_slot,
                deadline_slot,
                wifi_period_slots,
                wifi_tx_slots,
                num_events=repetitions,
                limit=10_000 if schedule_slot >= 0 else 4,
                cyclic_periodic=True,
                macrocycle_slots=int(_parse_int(row.get("macrocycle_slots"), 64) or 64),
            )
            tasks.append(
                JointTaskSpec(
                    task_id=pair_id,
                    radio="wifi",
                    payload_bytes=wifi_tx_slots * DEFAULT_WIFI_BYTES_PER_SLOT,
                    release_slot=release_slot,
                    deadline_slot=deadline_slot,
                    preferred_channel=preferred_channel,
                    repetitions=repetitions,
                    wifi_tx_slots=wifi_tx_slots,
                    wifi_period_slots=wifi_period_slots,
                    max_offsets=wifi_max_offsets,
                    cyclic_periodic=True,
                )
            )
            continue

        if radio == "ble":
            ble_ce_slots = int(_parse_int(row.get("ble_ce_slots"), 1) or 1)
            ble_ci_slots = int(_parse_int(row.get("ble_ci_slots"), max(ble_ce_slots, 8)) or max(ble_ce_slots, 8))
            num_events = _count_contiguous_segments(occupied_slots)
            if num_events <= 0:
                macrocycle_slots = int(_parse_int(row.get("macrocycle_slots"), 64) or 64)
                num_events = max(1, min(3, macrocycle_slots // max(ble_ci_slots, 1)))
            tasks.append(
                JointTaskSpec(
                    task_id=pair_id,
                    radio="ble",
                    payload_bytes=ble_ce_slots * DEFAULT_BLE_BYTES_PER_CE_SLOT,
                    release_slot=release_slot,
                    deadline_slot=deadline_slot,
                    preferred_channel=preferred_channel,
                    repetitions=max(1, num_events),
                    ble_ce_slots=ble_ce_slots,
                    ble_ci_slots_options=(ble_ci_slots,),
                    ble_num_events=max(1, num_events),
                    ble_pattern_count=3,
                    max_offsets=_suggest_max_offsets(release_slot, deadline_slot, ble_ci_slots, ble_ce_slots),
                )
            )
            continue

        raise ValueError(f"Unsupported radio in mainline pair row: {radio!r}")

    return tasks


def load_joint_tasks_from_mainline_pair_parameters_csv(csv_path: str | Path) -> list[JointTaskSpec]:
    return build_joint_tasks_from_mainline_pair_parameter_rows(load_mainline_pair_parameter_rows(csv_path))


def _derive_wifi_payload_floor_bytes(rows: list[Mapping[str, Any]]) -> int:
    floor_bytes = 0
    for task in build_joint_tasks_from_mainline_pair_parameter_rows(rows):
        if task.radio != "wifi":
            continue
        floor_bytes += int(task.payload_bytes)
    return floor_bytes


def _build_faithful_wifi_seed_specs(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("radio", "")).strip().lower() != "wifi":
            continue
        schedule_slot = _parse_int(row.get("schedule_slot"), -1)
        if schedule_slot is None or schedule_slot < 0:
            continue
        occupied_slots = _parse_occupied_slots(row.get("occupied_slots_in_macrocycle"))
        specs.append(
            {
                "pair_id": int(_parse_int(row.get("pair_id"), 0) or 0),
                "medium": "wifi",
                "offset": int(schedule_slot),
                "channel": _parse_int(row.get("channel")),
                "period_slots": int(_parse_int(row.get("wifi_period_slots"), _parse_int(row.get("wifi_tx_slots"), 1) or 1) or 1),
                "width_slots": int(_parse_int(row.get("wifi_tx_slots"), 1) or 1),
                "num_events": max(1, _infer_wifi_num_events(occupied_slots, int(_parse_int(row.get("wifi_tx_slots"), 1) or 1)) or 1),
            }
        )
    return specs


def infer_macrocycle_slots_from_mainline_pair_rows(
    rows: list[Mapping[str, Any]],
    default: int = 64,
) -> int:
    for row in rows:
        macrocycle_slots = _parse_int(row.get("macrocycle_slots"))
        if macrocycle_slots is not None:
            return int(macrocycle_slots)
    return int(default)


def build_joint_runtime_config_from_mainline_pair_parameters_csv(
    csv_path: str | Path,
    base_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(csv_path)
    rows = load_mainline_pair_parameter_rows(path)
    tasks = load_joint_tasks_from_mainline_pair_parameters_csv(path)
    runtime_config = dict(base_config or {})
    objective = dict(runtime_config.get("objective", {})) if isinstance(runtime_config.get("objective", {}), Mapping) else {}
    objective.setdefault("wifi_payload_floor_bytes", _derive_wifi_payload_floor_bytes(rows))
    runtime_config.update(
        {
            "macrocycle_slots": infer_macrocycle_slots_from_mainline_pair_rows(rows, default=int(runtime_config.get("macrocycle_slots", 64))),
            "wifi_channels": list(WIFI_CHANNELS),
            "ble_channels": list(BLE_CHANNELS),
            "tasks": [asdict(task) for task in tasks],
            "pair_generation_mode": "mainline_csv",
            "pair_parameters_csv": str(path),
            "_source_config": str(path),
        }
    )
    runtime_config["objective"] = objective
    ga_cfg = dict(runtime_config.get("ga", {})) if isinstance(runtime_config.get("ga", {}), Mapping) else {}
    ga_cfg.setdefault("seeded_state_specs", _build_faithful_wifi_seed_specs(rows))
    runtime_config["ga"] = ga_cfg
    return runtime_config
