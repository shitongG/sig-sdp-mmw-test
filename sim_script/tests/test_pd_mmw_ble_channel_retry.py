import pathlib
import subprocess
import sys


def test_script_accepts_ble_channel_retry_flag():
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
            "--ble-channel-retries",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ble_channel_retries_used =" in proc.stdout


def test_script_accepts_wifi_first_ble_scheduling_flag():
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
            "--wifi-first-ble-scheduling",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wifi_first_ble_scheduling = True" in proc.stdout
