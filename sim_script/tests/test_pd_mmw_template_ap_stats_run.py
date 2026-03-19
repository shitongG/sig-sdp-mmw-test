import csv
import pathlib
import json
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "sim_script" / "pd_mmw_template_ap_stats.py"
DEFAULT_CONFIG_PATH = REPO_ROOT / "sim_script" / "pd_mmw_template_ap_stats_config.json"
MANUAL_CONFIG_PATH = REPO_ROOT / "sim_script" / "pd_mmw_template_ap_stats_manual_pairs_config.json"
MACRO_BACKEND_CONFIG_PATH = REPO_ROOT / "sim_script" / "pd_mmw_template_ap_stats_macrocycle_hopping_empty_config.json"
MACRO_BACKEND_9WIFI_16BLE_CONFIG_PATH = (
    REPO_ROOT / "sim_script" / "pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json"
)


def test_script_runs_and_prints_office_table_header():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pair_density_per_m2 = 0.05" in proc.stdout
    assert "runtime_device = cpu" in proc.stdout
    assert "office_id,wifi_pair_count,ble_pair_count,wifi_slots_used,ble_slots_used" in proc.stdout


def test_script_runs_and_prints_pair_and_schedule_headers():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pair_id,office_id,radio,channel,priority,release_time_slot,deadline_slot,schedule_slot,schedule_time_ms" in proc.stdout
    assert "start_time_slot,wifi_anchor_slot,wifi_period_slots,wifi_period_ms,wifi_tx_slots,wifi_tx_ms" in proc.stdout
    assert "schedule_slot,pair_ids,wifi_pair_ids,ble_pair_ids,pair_count,wifi_pair_count,ble_pair_count" in proc.stdout
    assert "macrocycle_slots =" in proc.stdout
    assert "runtime_device = cpu" in proc.stdout


def test_script_writes_csv_outputs_and_avoids_known_warnings():
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
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
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (pathlib.Path(tmpdir) / "pair_parameters.csv").exists()
        assert (pathlib.Path(tmpdir) / "wifi_ble_schedule.csv").exists()
        assert (pathlib.Path(tmpdir) / "wifi_ble_schedule.png").exists()
        assert (pathlib.Path(tmpdir) / "wifi_ble_schedule_overview.png").exists()
        assert any(p.name.startswith("wifi_ble_schedule_window_") for p in pathlib.Path(tmpdir).iterdir())
        assert (pathlib.Path(tmpdir) / "schedule_plot_rows.csv").exists()
        assert "CUDA initialization" not in proc.stderr
        assert "SparseEfficiencyWarning" not in proc.stderr


def test_script_respects_cli_cell_size_and_pair_density():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--cell-size",
            "1",
            "--pair-density",
            "0.1",
            "--seed",
            "7",
            "--mmw-nit",
            "5",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "n_office = 1" in proc.stdout
    assert "pair_density_per_m2 = 0.1" in proc.stdout


def test_script_exposes_gpu_flags_in_cpu_mode():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "runtime_device = cpu" in proc.stdout


def test_script_runs_from_json_config(tmp_path):
    config_path = tmp_path / "ap_stats_config.json"
    config_path.write_text(
        json.dumps(
            {
                "cell_size": 1,
                "pair_density": 0.05,
                "seed": 123,
                "mmw_nit": 5,
                "mmw_eta": 0.05,
                "output_dir": "json_out",
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pair_density_per_m2 = 0.05" in proc.stdout
    assert (tmp_path / "json_out" / "pair_parameters.csv").exists()


def test_default_json_config_exists():
    assert DEFAULT_CONFIG_PATH.exists()


def test_default_json_config_runs_successfully():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = pathlib.Path(tmpdir) / "ap_stats_config.json"
        config_path.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "pair_density_per_m2 = 0.05" in proc.stdout
        assert (pathlib.Path(tmpdir) / "output" / "pair_parameters.csv").exists()


def test_manual_pair_json_config_exists():
    assert MANUAL_CONFIG_PATH.exists()


def test_script_runs_from_manual_pair_json_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = pathlib.Path(tmpdir) / "manual.json"
        config_path.write_text(MANUAL_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "pair_generation_mode = manual" in proc.stdout
        assert "release_time_slot" in proc.stdout
        assert "deadline_slot" in proc.stdout
        assert (pathlib.Path(tmpdir) / "manual_output" / "pair_parameters.csv").exists()


def test_script_runs_from_legacy_ble_backend_json_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = pathlib.Path(tmpdir) / "legacy_ble_backend.json"
        config_path.write_text(
            json.dumps(
                {
                    "cell_size": 1,
                    "pair_density": 0.05,
                    "seed": 123,
                    "mmw_nit": 5,
                    "mmw_eta": 0.05,
                    "ble_schedule_backend": "legacy",
                    "output_dir": "legacy_out",
                }
            ),
            encoding="utf-8",
        )

        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "pair_generation_mode = random" in proc.stdout
        assert "ble_schedule_backend = legacy" in proc.stdout
        assert "legacy_out" in proc.stdout
        assert (pathlib.Path(tmpdir) / "legacy_out" / "pair_parameters.csv").exists()


def test_script_accepts_ble_schedule_backend_cli_override():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = pathlib.Path(tmpdir) / "legacy_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "cell_size": 1,
                    "pair_density": 0.0,
                    "seed": 123,
                    "mmw_nit": 1,
                    "mmw_eta": 0.05,
                    "ble_schedule_backend": "legacy",
                    "output_dir": "cli_override_out",
                }
            ),
            encoding="utf-8",
        )

        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--config",
                str(config_path),
                "--ble-schedule-backend",
                "macrocycle_hopping_sdp",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "ble_schedule_backend = macrocycle_hopping_sdp" in proc.stdout
        assert (pathlib.Path(tmpdir) / "cli_override_out" / "pair_parameters.csv").exists()


def test_macrocycle_hopping_backend_json_config_exists():
    assert MACRO_BACKEND_CONFIG_PATH.exists()


def test_script_runs_from_macrocycle_hopping_backend_json_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = pathlib.Path(tmpdir) / "macrocycle_backend.json"
        config_path.write_text(MACRO_BACKEND_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "ble_schedule_backend = macrocycle_hopping_sdp" in proc.stdout
        assert "pair_generation_mode = random" in proc.stdout
        assert (pathlib.Path(tmpdir) / "macrocycle_output_empty" / "pair_parameters.csv").exists()


def test_macrocycle_hopping_9wifi_16ble_json_config_exists():
    assert MACRO_BACKEND_9WIFI_16BLE_CONFIG_PATH.exists()


def test_macrocycle_hopping_9wifi_16ble_json_config_is_manual_with_25_pairs():
    config = json.loads(MACRO_BACKEND_9WIFI_16BLE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["pair_generation_mode"] == "manual"
    assert config["ble_schedule_backend"] == "macrocycle_hopping_sdp"
    assert config["output_dir"] == "macrocycle_output_9wifi_16ble"
    assert len(config["pair_parameters"]) == 25
    pair_ids = [row["pair_id"] for row in config["pair_parameters"]]
    assert pair_ids == list(range(25))
    wifi_rows = [row for row in config["pair_parameters"] if row["radio"] == "wifi"]
    ble_rows = [row for row in config["pair_parameters"] if row["radio"] == "ble"]
    assert len(wifi_rows) == 9
    assert len(ble_rows) == 16
    assert all(row["channel"] in {0, 5, 10} for row in wifi_rows)
    assert all(0 <= row["channel"] <= 36 for row in ble_rows)
    assert all(row["office_id"] == 0 for row in config["pair_parameters"])
    assert all(row["ble_timing_mode"] == "auto" for row in ble_rows)
    assert all("ble_ci_slots" not in row for row in ble_rows)
    assert all("ble_ce_slots" not in row for row in ble_rows)


def test_script_runs_from_macrocycle_hopping_9wifi_16ble_json_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = pathlib.Path(tmpdir) / "macrocycle_9wifi_16ble.json"
        config_path.write_text(
            MACRO_BACKEND_9WIFI_16BLE_CONFIG_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "pair_generation_mode = manual" in proc.stdout
        assert "ble_schedule_backend = macrocycle_hopping_sdp" in proc.stdout
        assert "n_wifi_pair = 9" in proc.stdout
        assert "n_ble_pair  = 16" in proc.stdout
        csv_path = pathlib.Path(tmpdir) / "macrocycle_output_9wifi_16ble" / "pair_parameters.csv"
        assert csv_path.exists()
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ble_rows = [row for row in rows if row["radio"] == "ble"]
        assert len({row["ble_ci_slots"] for row in ble_rows}) > 1
        assert len({row["ble_ce_slots"] for row in ble_rows}) > 1


def test_schedule_plot_rows_include_idle_ble_advertising_channels():
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
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
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        with open(pathlib.Path(tmpdir) / "schedule_plot_rows.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ble_adv_rows = [row for row in rows if row["radio"] == "ble_adv_idle"]
        assert ble_adv_rows, "expected BLE advertising channel rows in schedule plot export"
        labels = {row["label"] for row in ble_adv_rows}
        assert any("2402" in label for label in labels)
        assert any("2426" in label for label in labels)
        assert any("2480" in label for label in labels)


def test_schedule_plot_rows_csv_contains_idle_ble_advertising_channels(tmp_path):
    config_path = tmp_path / "ap_stats_config.json"
    config_path.write_text(
        json.dumps(
            {
                "cell_size": 1,
                "pair_density": 0.05,
                "seed": 123,
                "mmw_nit": 5,
                "mmw_eta": 0.05,
                "output_dir": "json_out",
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = (tmp_path / "json_out" / "schedule_plot_rows.csv").read_text(encoding="utf-8")
    assert "ble_adv_idle" in text
    assert "2402" in text
    assert "2426" in text
    assert "2480" in text
