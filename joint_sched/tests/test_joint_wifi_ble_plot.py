from pathlib import Path

from joint_sched import build_main_style_plot_rows, render_joint_schedule, solve_joint_wifi_ble_sdp



def test_joint_plot_render_uses_main_renderer_artifacts(tmp_path: Path):
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

    output_dir = tmp_path / "joint_demo_output"
    overview_path, window_paths = render_joint_schedule(result, output_dir, macrocycle_slots=32)

    assert output_dir.joinpath("schedule_plot_rows.csv").exists()
    assert overview_path.exists()
    assert overview_path.name == "wifi_ble_schedule_overview.png"
    assert window_paths
    assert all(path.exists() for path in window_paths)
    assert all(path.name.startswith("wifi_ble_schedule_window_") for path in window_paths)


def test_joint_runner_writes_summary_artifact(tmp_path: Path):
    from joint_sched.run_joint_wifi_ble_demo import run_joint_demo

    summary = run_joint_demo(
        config_path="joint_sched/joint_wifi_ble_demo_config.json",
        solver="ga",
        output_dir=tmp_path / "joint_summary_output",
    )

    assert Path(summary["joint_summary_json"]).exists()


def test_build_main_style_plot_rows_keeps_multiple_wifi_periodic_events():
    result = {
        "selected_states": [],
        "blocks": [
            {
                "state_id": 0,
                "pair_id": 0,
                "medium": "wifi",
                "event_index": 0,
                "slot_start": 3,
                "slot_end": 7,
                "freq_low_mhz": 2402.0,
                "freq_high_mhz": 2422.0,
                "label": "wifi-0-ev0",
            },
            {
                "state_id": 0,
                "pair_id": 0,
                "medium": "wifi",
                "event_index": 1,
                "slot_start": 19,
                "slot_end": 23,
                "freq_low_mhz": 2402.0,
                "freq_high_mhz": 2422.0,
                "label": "wifi-0-ev1",
            },
        ],
        "unscheduled_pair_ids": [],
    }

    rows = build_main_style_plot_rows(result, macrocycle_slots=32)
    wifi_rows = [row for row in rows if row["radio"] == "wifi"]

    assert len({(row["event_index"], row["slot"]) for row in wifi_rows}) == 8
    assert {row["event_index"] for row in wifi_rows} == {0, 1}
