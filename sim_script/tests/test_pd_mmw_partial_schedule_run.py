import pathlib
import subprocess
import sys


def test_script_reports_partial_schedule_when_max_slot_cap_is_hit():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cell-size",
            "1",
            "--pair-density",
            "0.2",
            "--max-slots",
            "2",
            "--mmw-nit",
            "3",
            "--seed",
            "9",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "partial_schedule = True" in proc.stdout
    assert "unscheduled_pair_ids =" in proc.stdout
