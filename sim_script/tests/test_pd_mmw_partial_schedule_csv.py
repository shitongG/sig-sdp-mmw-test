import pathlib
import subprocess
import sys
import tempfile


def test_script_writes_unscheduled_pairs_csv_when_partial_schedule_occurs():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    with tempfile.TemporaryDirectory() as tmpdir:
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
                "--output-dir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (pathlib.Path(tmpdir) / "unscheduled_pairs.csv").exists()
