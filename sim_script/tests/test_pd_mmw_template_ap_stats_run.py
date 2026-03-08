import pathlib
import subprocess
import sys


def test_script_runs_and_prints_office_table_header():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "2",
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
    assert "pair_density_per_m2 = 0.5" in proc.stdout
    assert "office_id,wifi_pair_count,ble_pair_count,wifi_slots_used,ble_slots_used" in proc.stdout
