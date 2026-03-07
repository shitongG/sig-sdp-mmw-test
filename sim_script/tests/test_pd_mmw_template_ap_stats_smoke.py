import pathlib
import subprocess
import sys


def test_ap_stats_script_compile_smoke():
    script = pathlib.Path("sim_script/pd_mmw_template_ap_stats.py")
    assert script.exists()
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
