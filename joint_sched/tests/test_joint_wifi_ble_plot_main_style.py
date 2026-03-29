from pathlib import Path

from joint_sched import build_main_style_plot_rows, export_joint_output_artifacts, solve_joint_wifi_ble_sdp


def test_joint_plot_rows_match_main_renderer_schema():
    config = {
        "macrocycle_slots": 32,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1024,
                "release_slot": 0,
                "deadline_slot": 8,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 1,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 20,
                "preferred_channel": 5,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 2,
            },
        ],
    }
    result = solve_joint_wifi_ble_sdp(config)
    rows = build_main_style_plot_rows(result, macrocycle_slots=32)

    assert rows
    assert {"pair_id", "radio", "channel", "slot", "slot_width", "freq_low_mhz", "freq_high_mhz", "label", "event_index"} <= set(rows[0])
    assert any(row["radio"] == "ble_adv_idle" for row in rows)
    assert any(row["radio"] == "wifi" for row in rows)
    assert any(row["radio"] == "ble" for row in rows)


def test_joint_export_writes_main_style_artifacts(tmp_path: Path):
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1024,
                "release_slot": 0,
                "deadline_slot": 4,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 1,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 12,
                "preferred_channel": 5,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
        ],
    }
    result = solve_joint_wifi_ble_sdp(config)
    artifacts = export_joint_output_artifacts(result, tasks=config["tasks"], output_dir=tmp_path, macrocycle_slots=16)

    assert Path(artifacts["schedule_plot_rows_path"]).exists()
    assert Path(artifacts["wifi_ble_schedule_csv"]).exists()
    assert Path(artifacts["pair_parameters_csv"]).exists()
    assert Path(artifacts["unscheduled_pairs_csv"]).exists()
    assert Path(artifacts["ble_ce_channel_events_csv"]).exists()
    assert Path(artifacts["wifi_ble_schedule_png"]).exists()
