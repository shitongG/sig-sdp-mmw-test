import pathlib
import subprocess
import sys
import tempfile
import csv


def test_cli_accepts_ble_channel_mode_per_ce():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        completed = subprocess.run(
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
                "--ble-channel-mode",
                "per_ce",
                "--output-dir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (pathlib.Path(tmpdir) / "wifi_ble_schedule.png").exists()
        assert (pathlib.Path(tmpdir) / "wifi_ble_schedule_overview.png").exists()
        assert any(p.name.startswith("wifi_ble_schedule_window_") for p in pathlib.Path(tmpdir).iterdir())
        assert (pathlib.Path(tmpdir) / "ble_ce_channel_events.csv").exists()
        assert "ble_channel_mode,ble_ce_channel_summary,start_time_slot" in completed.stdout

        with open(pathlib.Path(tmpdir) / "pair_parameters.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert "ble_channel_mode" in row
            assert "ble_ce_channel_summary" in row

        with open(pathlib.Path(tmpdir) / "schedule_plot_rows.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert any("ev" in row["label"] for row in rows if row["radio"] == "ble")


def test_cli_exports_wifi_first_ble_stats_columns():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        completed = subprocess.run(
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
                "--wifi-first-ble-scheduling",
                "--output-dir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "wifi_first_ble_scheduling = True" in completed.stdout

        with open(pathlib.Path(tmpdir) / "pair_parameters.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert "effective_ble_channels" in row
            assert "scheduled_ble_pairs" in row
            assert "no_collision_probability" in row
