import pathlib
import subprocess
import sys


def test_script_prints_ble_ci_discrete_summary():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "2",
            "--seed",
            "3",
            "--mmw-nit",
            "20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ble_ci_quanta_candidates" in proc.stdout
