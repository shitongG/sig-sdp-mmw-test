from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from joint_sched.joint_wifi_ble_adapter import (
    build_joint_tasks_from_mainline_pair_parameter_rows,
    build_joint_runtime_config_from_mainline_pair_parameters_csv,
    load_mainline_pair_parameter_rows,
)
from joint_sched.joint_wifi_ble_model import (
    DEFAULT_BLE_BYTES_PER_CE_SLOT,
    DEFAULT_WIFI_BYTES_PER_SLOT,
    build_joint_candidate_states,
    expand_candidate_blocks,
)
from joint_sched.run_joint_wifi_ble_demo import run_joint_demo


MAINLINE_PAIR_PARAMETER_FIELDS = [
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


def _write_mainline_pair_parameters_csv(path: Path) -> None:
    rows = [
        {
            "pair_id": 0,
            "office_id": 0,
            "radio": "wifi",
            "channel": 0,
            "priority": "1.000",
            "release_time_slot": 0,
            "deadline_slot": 63,
            "schedule_slot": 3,
            "schedule_time_ms": "3.750",
            "ble_channel_mode": "per_ce",
            "ble_ce_channel_summary": "NA",
            "start_time_slot": 0,
            "wifi_anchor_slot": 3,
            "wifi_period_slots": 16,
            "wifi_period_ms": "20.000",
            "wifi_tx_slots": 5,
            "wifi_tx_ms": "6.250",
            "ble_anchor_slot": "NA",
            "ble_ci_slots": "NA",
            "ble_ci_ms": "NA",
            "ble_ce_slots": "NA",
            "ble_ce_ms": "NA",
            "ble_ce_feasible": "NA",
            "effective_ble_channels": "NA",
            "scheduled_ble_pairs": "NA",
            "no_collision_probability": "NA",
            "macrocycle_slots": 64,
            "occupied_slots_in_macrocycle": "[3, 4, 5, 6, 7, 19, 20, 21, 22, 23]",
        },
        {
            "pair_id": 1,
            "office_id": 0,
            "radio": "ble",
            "channel": 8,
            "priority": "1.000",
            "release_time_slot": 0,
            "deadline_slot": 31,
            "schedule_slot": 4,
            "schedule_time_ms": "5.000",
            "ble_channel_mode": "per_ce",
            "ble_ce_channel_summary": "[8, 13]",
            "start_time_slot": 0,
            "wifi_anchor_slot": "NA",
            "wifi_period_slots": "NA",
            "wifi_period_ms": "NA",
            "wifi_tx_slots": "NA",
            "wifi_tx_ms": "NA",
            "ble_anchor_slot": 4,
            "ble_ci_slots": 8,
            "ble_ci_ms": "10.000",
            "ble_ce_slots": 2,
            "ble_ce_ms": "2.500",
            "ble_ce_feasible": "true",
            "effective_ble_channels": 17,
            "scheduled_ble_pairs": 1,
            "no_collision_probability": "1.000",
            "macrocycle_slots": 64,
            "occupied_slots_in_macrocycle": "[4, 5, 12, 13]",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAINLINE_PAIR_PARAMETER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_mainline_pair_parameter_csv_rows_are_translated_to_joint_tasks(tmp_path: Path):
    csv_path = tmp_path / "pair_parameters.csv"
    _write_mainline_pair_parameters_csv(csv_path)

    rows = load_mainline_pair_parameter_rows(csv_path)
    tasks = build_joint_tasks_from_mainline_pair_parameter_rows(rows)

    assert [task.task_id for task in tasks] == [0, 1]
    wifi_task = tasks[0]
    ble_task = tasks[1]
    assert wifi_task.radio == "wifi"
    assert wifi_task.payload_bytes == 5 * DEFAULT_WIFI_BYTES_PER_SLOT
    assert wifi_task.wifi_tx_slots == 5
    assert wifi_task.wifi_period_slots == 16
    assert wifi_task.repetitions == 2
    assert wifi_task.max_offsets > 1
    assert ble_task.radio == "ble"
    assert ble_task.payload_bytes == 2 * DEFAULT_BLE_BYTES_PER_CE_SLOT
    assert ble_task.ble_ce_slots == 2
    assert ble_task.ble_ci_slots_options == (8,)
    assert ble_task.ble_num_events == 2
    assert ble_task.ble_pattern_count == 3
    assert ble_task.max_offsets > 1


def test_mainline_wifi_csv_rows_expand_to_periodic_wifi_blocks(tmp_path: Path):
    csv_path = tmp_path / "pair_parameters.csv"
    _write_mainline_pair_parameters_csv(csv_path)

    tasks = build_joint_tasks_from_mainline_pair_parameter_rows(load_mainline_pair_parameter_rows(csv_path))
    runtime_config = {
        "macrocycle_slots": 64,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "tasks": [asdict(task) for task in tasks],
    }
    space = build_joint_candidate_states(runtime_config)
    wifi_state = next(
        space.states[idx]
        for idx in space.pair_to_state_indices[0]
        if space.states[idx].medium == "wifi"
    )

    blocks = expand_candidate_blocks(wifi_state)

    assert wifi_state.num_events == 2
    assert wifi_state.period_slots == 16
    assert wifi_state.width_slots == 5
    assert len(blocks) == 2
    assert [block.slot_end - block.slot_start for block in blocks] == [5, 5]
    assert blocks[1].slot_start - blocks[0].slot_start == 16


def test_runner_prefers_mainline_pair_parameter_csv_over_random_generation(tmp_path: Path):
    csv_path = tmp_path / "pair_parameters.csv"
    _write_mainline_pair_parameters_csv(csv_path)

    config_path = tmp_path / "joint_from_mainline_csv.json"
    config_path.write_text(
        json.dumps(
            {
                "solver": "sdp",
                "objective": {
                    "mode": "lexicographic",
                    "primary": "payload",
                    "secondary": "fill",
                },
                "pair_parameters_csv": "pair_parameters.csv",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = run_joint_demo(config_path=config_path, solver="sdp", output_dir=tmp_path / "joint_out")

    assert summary["task_count"] == 2
    assert summary["state_count"] > 2
    assert summary["objective_mode"] == "lexicographic"
    assert summary["_joint_generation_mode"] == "faithful_mainline_csv"
    assert Path(summary["pair_parameters_csv"]).exists()
    assert Path(summary["joint_summary_json"]).exists()


def test_runner_uses_output_dir_pair_parameters_csv_when_present(tmp_path: Path):
    output_dir = tmp_path / "mainline_output"
    output_dir.mkdir()
    _write_mainline_pair_parameters_csv(output_dir / "pair_parameters.csv")

    config_path = tmp_path / "joint_from_mainline_output_dir.json"
    config_path.write_text(
        json.dumps(
            {
                "solver": "ga",
                "output_dir": "mainline_output",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = run_joint_demo(config_path=config_path, solver="ga", output_dir=tmp_path / "joint_out")

    assert summary["task_count"] == 2
    assert summary["_joint_generation_mode"] == "faithful_mainline_csv"


def test_adapter_derives_wifi_payload_floor_from_scheduled_wifi_rows(tmp_path: Path):
    csv_path = tmp_path / "pair_parameters.csv"
    _write_mainline_pair_parameters_csv(csv_path)

    runtime = build_joint_runtime_config_from_mainline_pair_parameters_csv(csv_path)

    assert runtime["objective"]["wifi_payload_floor_bytes"] == 5 * DEFAULT_WIFI_BYTES_PER_SLOT


def test_adapter_recovers_cyclic_wifi_event_count_from_wrapped_mainline_rows(tmp_path: Path):
    csv_path = tmp_path / "pair_parameters.csv"
    csv_path.write_text(
        "pair_id,office_id,radio,channel,priority,release_time_slot,deadline_slot,schedule_slot,schedule_time_ms,ble_channel_mode,ble_ce_channel_summary,start_time_slot,wifi_anchor_slot,wifi_period_slots,wifi_period_ms,wifi_tx_slots,wifi_tx_ms,ble_anchor_slot,ble_ci_slots,ble_ci_ms,ble_ce_slots,ble_ce_ms,ble_ce_feasible,effective_ble_channels,scheduled_ble_pairs,no_collision_probability,macrocycle_slots,occupied_slots_in_macrocycle\n"
        "1,0,wifi,0,2.000,0,63,13,16.250,per_ce,NA,0,3,16,20.000,5,6.250,NA,NA,NA,NA,NA,NA,NA,NA,NA,64,\"[0,1,13,14,15,16,17,29,30,31,32,33,45,46,47,48,49,61,62,63]\"\n",
        encoding="utf-8",
    )

    runtime = build_joint_runtime_config_from_mainline_pair_parameters_csv(csv_path)
    task = runtime["tasks"][0]
    seed = runtime["ga"]["seeded_state_specs"][0]

    assert task["repetitions"] == 4
    assert task["cyclic_periodic"] is True
    assert seed["num_events"] == 4


def test_adapter_preserves_explicit_wifi_floor_override(tmp_path: Path):
    csv_path = tmp_path / "pair_parameters.csv"
    _write_mainline_pair_parameters_csv(csv_path)

    default_runtime = build_joint_runtime_config_from_mainline_pair_parameters_csv(csv_path)
    override_runtime = build_joint_runtime_config_from_mainline_pair_parameters_csv(
        csv_path,
        base_config={"objective": {"wifi_payload_floor_bytes": 99999}},
    )

    assert default_runtime["objective"]["wifi_payload_floor_bytes"] == 5 * DEFAULT_WIFI_BYTES_PER_SLOT
    assert override_runtime["objective"]["wifi_payload_floor_bytes"] == 99999
