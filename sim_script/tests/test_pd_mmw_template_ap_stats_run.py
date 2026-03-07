import pathlib
import subprocess
import sys


def test_script_runs_and_prints_ap_table_header():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "2",
            "--sta-density",
            "0.005",
            "--seed",
            "123",
            "--mmw-nit",
            "20",
            "--mmw-eta",
            "0.05",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ap_id,wifi_user_count,ble_user_count,wifi_slots_used,ble_slots_used" in proc.stdout
