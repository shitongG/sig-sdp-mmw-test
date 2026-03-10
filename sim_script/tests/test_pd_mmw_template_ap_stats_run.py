import pathlib
import subprocess
import sys
import tempfile


def test_script_runs_and_prints_office_table_header():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "1",
            "--pair-density",
            "0.05",
            "--seed",
            "123",
            "--mmw-nit",
            "5",
            "--mmw-eta",
            "0.05",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pair_density_per_m2 = 0.05" in proc.stdout
    assert "runtime_device = cpu" in proc.stdout
    assert "office_id,wifi_pair_count,ble_pair_count,wifi_slots_used,ble_slots_used" in proc.stdout


def test_script_runs_and_prints_pair_and_schedule_headers():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "1",
            "--pair-density",
            "0.05",
            "--seed",
            "123",
            "--mmw-nit",
            "5",
            "--mmw-eta",
            "0.05",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pair_id,office_id,radio,channel,priority,schedule_slot,schedule_time_ms" in proc.stdout
    assert "start_time_slot,wifi_anchor_slot,wifi_period_slots,wifi_period_ms,wifi_tx_slots,wifi_tx_ms" in proc.stdout
    assert "schedule_slot,pair_ids,wifi_pair_ids,ble_pair_ids,pair_count,wifi_pair_count,ble_pair_count" in proc.stdout
    assert "macrocycle_slots =" in proc.stdout
    assert "runtime_device = cpu" in proc.stdout


def test_script_writes_csv_outputs_and_avoids_known_warnings():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--cell-size",
                "1",
                "--pair-density",
                "0.05",
                "--seed",
                "123",
                "--mmw-nit",
                "5",
                "--output-dir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (pathlib.Path(tmpdir) / "pair_parameters.csv").exists()
        assert (pathlib.Path(tmpdir) / "wifi_ble_schedule.csv").exists()
        assert (pathlib.Path(tmpdir) / "wifi_ble_schedule.png").exists()
        assert (pathlib.Path(tmpdir) / "schedule_plot_rows.csv").exists()
        assert "CUDA initialization" not in proc.stderr
        assert "SparseEfficiencyWarning" not in proc.stderr


def test_script_respects_cli_cell_size_and_pair_density():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "1",
            "--pair-density",
            "0.1",
            "--seed",
            "7",
            "--mmw-nit",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "n_office = 1" in proc.stdout
    assert "pair_density_per_m2 = 0.1" in proc.stdout


def test_script_exposes_gpu_flags_in_cpu_mode():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "1",
            "--pair-density",
            "0.05",
            "--seed",
            "7",
            "--mmw-nit",
            "5",
            "--gpu-id",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "runtime_device = cpu" in proc.stdout
