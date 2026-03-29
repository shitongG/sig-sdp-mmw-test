import json
from pathlib import Path

import pandas as pd

from joint_sched.run_joint_wifi_ble_demo import main, run_joint_demo


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
MAINLINE_ROOT = Path(__file__).resolve().parents[2]


def test_runner_accepts_main_style_random_config(tmp_path: Path):
    summary = run_joint_demo(
        config_path="sim_script/pd_mmw_template_ap_stats_config.json",
        solver="ga",
        output_dir=tmp_path / "joint_random_output",
    )

    assert summary["task_count"] > 0
    assert summary["state_count"] > 0
    assert Path(summary["schedule_csv_path"]).exists()
    assert Path(summary["overview_path"]).exists()
    assert Path(summary["pair_parameters_csv"]).exists()
    assert Path(summary["wifi_ble_schedule_csv"]).exists()
    assert Path(summary["unscheduled_pairs_csv"]).exists()
    assert Path(summary["ble_ce_channel_events_csv"]).exists()
    assert Path(summary["wifi_ble_schedule_png"]).exists()
    assert summary["objective_mode"] == "lexicographic"
    assert summary["scheduled_payload_bytes"] >= 0
    assert summary["occupied_slot_count"] >= 0
    assert 0.0 <= summary["resource_utilization"] <= 1.0
    assert summary["fragmentation_penalty"] >= 0
    assert summary["idle_area_penalty"] >= 0
    assert summary["fill_penalty"] >= 0
    assert Path(summary["joint_summary_json"]).exists()


def _write_hga_test_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "joint_hga_test_config.json"
    config_path.write_text(
        json.dumps(
            {
                "macrocycle_slots": 16,
                "wifi_channels": [0, 10],
                "ble_channels": list(range(37)),
                "ga": {
                    "population_size": 10,
                    "generations": 4,
                    "seed": 3,
                },
                "hga": {
                    "population_size": 12,
                    "generations": 6,
                    "coordination_rounds": 3,
                    "seed": 5,
                },
                "tasks": [
                    {
                        "task_id": 0,
                        "radio": "wifi",
                        "payload_bytes": 1200,
                        "release_slot": 0,
                        "deadline_slot": 3,
                        "wifi_tx_slots": 4,
                        "max_offsets": 1,
                    },
                    {
                        "task_id": 1,
                        "radio": "ble",
                        "payload_bytes": 247,
                        "release_slot": 0,
                        "deadline_slot": 0,
                        "preferred_channel": 0,
                        "ble_ce_slots": 1,
                        "ble_ci_slots_options": [8],
                        "ble_num_events": 1,
                        "ble_pattern_count": 1,
                        "max_offsets": 1,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_path


def test_runner_accepts_hga_solver(tmp_path: Path):
    config_path = _write_hga_test_config(tmp_path)

    summary = run_joint_demo(
        config_path=config_path,
        solver="hga",
        output_dir=tmp_path / "joint_hga_output",
    )

    assert summary["solver"] == "hga"
    assert summary["status"] == "ok"
    assert summary["search_mode"] == "unified_joint"
    assert Path(summary["overview_path"]).exists()
    assert summary["residual_seed_budget"] == 4
    assert summary["residual_swap_budget"] == 6


def test_hga_summary_reports_seed_and_final_payloads(tmp_path: Path):
    config_path = _write_hga_test_config(tmp_path)

    summary = run_joint_demo(
        config_path=config_path,
        solver="hga",
        output_dir=tmp_path / "joint_hga_output",
    )

    assert summary["wifi_seed_payload_bytes"] >= 0
    assert summary["final_wifi_payload_bytes"] >= 0
    assert summary["coordination_rounds_used"] >= 0
    assert summary["repair_insertions_used"] >= 0
    assert summary["repair_swaps_used"] >= 0
    assert summary["wifi_move_seed_count"] >= 0
    assert summary["wifi_move_repairs_used"] >= 0
    assert summary["accepted_wifi_local_moves"] >= 0


def test_resolve_joint_runtime_config_reads_wifi_floor_and_hga_knobs(tmp_path: Path):
    config_path = tmp_path / "joint.json"
    config_path.write_text(
        json.dumps(
            {
                "objective": {
                    "mode": "lexicographic",
                    "wifi_payload_floor_bytes": 3000,
                },
                "hga": {
                    "residual_seed_budget": 6,
                    "residual_swap_budget": 8,
                },
                "tasks": [],
                "wifi_channels": [0],
                "ble_channels": [0],
                "macrocycle_slots": 8,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    from joint_sched.run_joint_wifi_ble_demo import resolve_joint_runtime_config

    config = resolve_joint_runtime_config(config_path)

    assert config["objective"]["wifi_payload_floor_bytes"] == 3000
    assert config["hga"]["residual_seed_budget"] == 6
    assert config["hga"]["residual_swap_budget"] == 8


def test_hga_faithful_mainline_csv_keeps_wifi_floor(tmp_path: Path):
    summary = run_joint_demo(
        config_path=MAINLINE_ROOT / "sim_script/output_ga_wifi_reschedule/pair_parameters.csv",
        solver="hga",
        output_dir=tmp_path / "faithful_hga",
    )

    assert summary["_joint_generation_mode"] == "faithful_mainline_csv"
    assert summary["final_wifi_payload_bytes"] >= summary["wifi_seed_payload_bytes"]


def test_faithful_runner_reports_nonzero_derived_wifi_floor(tmp_path: Path):
    summary = run_joint_demo(
        config_path=MAINLINE_ROOT / "sim_script/output_ga_wifi_reschedule/pair_parameters.csv",
        solver="ga",
        output_dir=tmp_path / "faithful_ga",
    )

    assert summary["_joint_generation_mode"] == "faithful_mainline_csv"
    assert summary["wifi_payload_floor_bytes"] > 0


def test_faithful_hga_keeps_final_wifi_payload_above_derived_floor(tmp_path: Path):
    summary = run_joint_demo(
        config_path=MAINLINE_ROOT / "sim_script/output_ga_wifi_reschedule/pair_parameters.csv",
        solver="hga",
        output_dir=tmp_path / "faithful_hga",
    )

    assert summary["wifi_payload_floor_bytes"] > 0
    assert summary["final_wifi_payload_bytes"] >= summary["wifi_payload_floor_bytes"]


def test_runner_persists_wifi_move_experiment_summary(tmp_path: Path):
    out_dir = tmp_path / "faithful_hga"
    pair_csv = MAINLINE_ROOT / "sim_script/output_ga_wifi_reschedule/pair_parameters.csv"

    exit_code = main(
        [
            "--solver",
            "hga",
            "--config",
            str(pair_csv),
            "--output",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((out_dir / "joint_summary.json").read_text())
    assert "wifi_move_seed_count" in summary
    assert "wifi_move_repairs_used" in summary
    assert "accepted_wifi_local_moves" in summary
    assert (out_dir / "experiment_record.md").exists()
    assert (out_dir / "wifi_ble_schedule_overview.png").exists()


def test_readme_joint_hga_sections_exist():
    text = (WORKTREE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Unified Joint GA/HGA Encoding" in text
    assert "whole-WiFi-state move heuristic" in text
    assert "Faithful Mainline Experiment Record" in text
    assert "```math" in text
