from pathlib import Path

from joint_sched.run_joint_wifi_ble_demo import run_joint_demo


def test_joint_runner_reports_payload_fill_metrics(tmp_path):
    summary = run_joint_demo(
        config_path="joint_sched/joint_wifi_ble_demo_config.json",
        solver="ga",
        output_dir=tmp_path / "joint_metrics_output",
    )

    assert summary["objective_mode"] == "lexicographic"
    assert summary["scheduled_payload_bytes"] >= 0
    assert summary["occupied_slot_count"] >= 0
    assert 0.0 <= summary["resource_utilization"] <= 1.0
    assert summary["fragmentation_penalty"] >= 0
    assert summary["idle_area_penalty"] >= 0
    assert summary["fill_penalty"] >= 0
    assert Path(summary["joint_summary_json"]).exists()
