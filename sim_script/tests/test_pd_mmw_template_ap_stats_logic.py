import json
from pathlib import Path

import numpy as np
import pytest

from sim_script.pd_mmw_template_ap_stats import (
    _aggregate_office_stats_from_arrays,
    _occupancy_within_time_window,
    apply_manual_pair_parameters,
    apply_ble_schedule_backend,
    build_pair_parameter_rows,
    build_ble_hopping_inputs_from_env,
    build_schedule_rows,
    load_json_config,
    merge_config_with_defaults,
    strip_comment_keys,
    solve_ble_hopping_for_env,
)
from sim_src.env.env import env, sample_ble_pair_timing


def test_aggregate_office_stats_from_arrays():
    office_id = np.array([0, 0, 1, 1], dtype=int)
    radio = np.array([0, 1, 0, 1], dtype=int)  # 0=WiFi, 1=BLE
    z_vec = np.array([1, 2, 1, 3], dtype=int)

    rows = _aggregate_office_stats_from_arrays(
        office_id=office_id,
        radio=radio,
        z_vec=z_vec,
        n_office=2,
        wifi_id=0,
        ble_id=1,
    )

    assert rows[0]["wifi_pair_count"] == 1
    assert rows[0]["ble_pair_count"] == 1
    assert rows[0]["wifi_slots_used"] == 1
    assert rows[0]["ble_slots_used"] == 1

    assert rows[1]["wifi_pair_count"] == 1
    assert rows[1]["ble_pair_count"] == 1
    assert rows[1]["wifi_slots_used"] == 1
    assert rows[1]["ble_slots_used"] == 1


def test_build_pair_parameter_rows_contains_wifi_and_ble_fields():
    rows = build_pair_parameter_rows(
        pair_office_id=np.array([0, 0]),
        pair_radio_type=np.array([0, 1]),
        pair_channel=np.array([6, 12]),
        pair_priority=np.array([3.0, 1.0]),
        pair_release_time_slot=np.array([0, 1]),
        pair_deadline_slot=np.array([15, 16]),
        ble_channel_mode="per_ce",
        pair_ble_ce_channels={1: np.array([3, 7], dtype=int)},
        pair_start_time_slot=np.array([0, 0]),
        pair_wifi_anchor_slot=np.array([4, 0]),
        pair_wifi_period_slots=np.array([16, 0]),
        pair_wifi_tx_slots=np.array([5, 0]),
        pair_ble_anchor_slot=np.array([0, 4]),
        pair_ble_ci_slots=np.array([0, 16]),
        pair_ble_ce_slots=np.array([0, 3]),
        pair_ble_ce_feasible=np.array([True, True]),
        z_vec=np.array([2, 5]),
        occupied_slots=np.array(
            [
                [False, False, True, True, False, False],
                [False, False, False, False, False, True],
            ],
            dtype=bool,
        ),
        macrocycle_slots=6,
        slot_time=1.25e-3,
        wifi_id=0,
        ble_id=1,
    )

    assert rows[0]["pair_id"] == 0
    assert rows[0]["radio"] == "wifi"
    assert rows[0]["schedule_slot"] == 2
    assert rows[0]["channel"] == 6
    assert rows[0]["release_time_slot"] == 0
    assert rows[0]["deadline_slot"] == 15
    assert rows[0]["start_time_slot"] == 0
    assert rows[0]["wifi_period_slots"] == 16
    assert rows[0]["wifi_period_ms"] == 20.0
    assert rows[0]["wifi_tx_slots"] == 5
    assert rows[0]["wifi_tx_ms"] == 6.25
    assert rows[0]["ble_channel_mode"] == "per_ce"
    assert rows[0]["ble_ce_channel_summary"] is None
    assert rows[0]["ble_ci_slots"] is None

    assert rows[1]["pair_id"] == 1
    assert rows[1]["radio"] == "ble"
    assert rows[1]["schedule_slot"] == 5
    assert rows[1]["release_time_slot"] == 1
    assert rows[1]["deadline_slot"] == 16
    assert rows[1]["start_time_slot"] == 0
    assert rows[1]["ble_ci_slots"] == 16
    assert rows[1]["ble_ce_slots"] == 3
    assert rows[1]["ble_anchor_slot"] == 4
    assert rows[1]["ble_ci_ms"] == 20.0
    assert rows[1]["ble_ce_ms"] == 3.75
    assert rows[1]["ble_channel_mode"] == "per_ce"
    assert rows[1]["ble_ce_channel_summary"] == [3, 7]


def test_build_schedule_rows_orders_by_min_schedule_slot():
    pair_rows = [
        {"pair_id": 3, "radio": "ble", "schedule_slot": 5, "office_id": 1, "channel": 2, "occupied_slots_in_macrocycle": [5]},
        {"pair_id": 0, "radio": "wifi", "schedule_slot": 1, "office_id": 0, "channel": 6, "occupied_slots_in_macrocycle": [1]},
        {"pair_id": 2, "radio": "wifi", "schedule_slot": 5, "office_id": 1, "channel": 1, "occupied_slots_in_macrocycle": [5]},
    ]

    rows = build_schedule_rows(pair_rows)

    assert rows[0]["schedule_slot"] == 1
    assert rows[0]["pair_ids"] == [0]
    assert rows[0]["wifi_pair_ids"] == [0]
    assert rows[0]["ble_pair_ids"] == []

    assert rows[1]["schedule_slot"] == 5
    assert rows[1]["pair_ids"] == [2, 3]
    assert rows[1]["wifi_pair_ids"] == [2]
    assert rows[1]["ble_pair_ids"] == [3]


def test_strip_comment_keys_removes_comment_entries_recursively():
    payload = {
        "_comment_root": "说明",
        "seed": 7,
        "nested": {
            "_comment_nested": "忽略",
            "value": 1,
            "items": [{"_comment_item": "忽略", "name": "kept"}],
        },
    }

    assert strip_comment_keys(payload) == {
        "seed": 7,
        "nested": {
            "value": 1,
            "items": [{"name": "kept"}],
        },
    }


def test_load_json_config_reads_fields_and_resolves_relative_output_dir(tmp_path: Path):
    config_path = tmp_path / "ap_stats_config.json"
    config_path.write_text(
        json.dumps(
            {
                "_comment_cell_size": "办公室网格边长",
                "cell_size": 1,
                "pair_density": 0.05,
                "seed": 123,
                "mmw_nit": 5,
                "output_dir": "out",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_json_config(config_path)

    assert loaded["cell_size"] == 1
    assert loaded["pair_density"] == 0.05
    assert loaded["seed"] == 123
    assert loaded["mmw_nit"] == 5
    assert loaded["output_dir"] == str((tmp_path / "out").resolve())


def test_load_json_config_keeps_pair_generation_mode_and_pair_parameters(tmp_path: Path):
    config_path = tmp_path / "manual.json"
    config_path.write_text(
        json.dumps(
            {
                "pair_generation_mode": "manual",
                "pair_parameters": [
                    {
                        "pair_id": 0,
                        "office_id": 0,
                        "radio": "ble",
                        "channel": 8,
                        "priority": 1.0,
                        "release_time_slot": 0,
                        "deadline_slot": 63,
                        "start_time_slot": 0,
                        "ble_anchor_slot": 12,
                        "ble_ci_slots": 64,
                        "ble_ce_slots": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_json_config(config_path)

    assert loaded["pair_generation_mode"] == "manual"
    assert loaded["pair_parameters"][0]["deadline_slot"] == 63


def test_merge_config_with_defaults_rejects_manual_mode_without_pair_parameters():
    with pytest.raises(ValueError, match="pair_parameters"):
        merge_config_with_defaults({"pair_generation_mode": "manual"})


def test_merge_config_with_defaults_accepts_ble_auto_timing_without_explicit_ci_ce():
    merged = merge_config_with_defaults(
        {
            "pair_generation_mode": "manual",
            "pair_parameters": [
                {
                    "pair_id": 0,
                    "office_id": 0,
                    "radio": "ble",
                    "channel": 8,
                    "priority": 1.0,
                    "release_time_slot": 0,
                    "deadline_slot": 31,
                    "start_time_slot": 0,
                    "ble_anchor_slot": 0,
                    "ble_timing_mode": "auto",
                }
            ],
        }
    )

    assert merged["pair_parameters"][0]["ble_timing_mode"] == "auto"


def test_merge_config_with_defaults_rejects_invalid_ble_timing_mode():
    with pytest.raises(ValueError, match="ble_timing_mode"):
        merge_config_with_defaults(
            {
                "pair_generation_mode": "manual",
                "pair_parameters": [
                    {
                        "pair_id": 0,
                        "office_id": 0,
                        "radio": "ble",
                        "channel": 8,
                        "priority": 1.0,
                        "release_time_slot": 0,
                        "deadline_slot": 31,
                        "start_time_slot": 0,
                        "ble_anchor_slot": 0,
                        "ble_timing_mode": "seeded-ish",
                    }
                ],
            }
        )


def test_sample_ble_pair_timing_from_seed_is_deterministic():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    t1 = sample_ble_pair_timing(
        rand_gen=rng1,
        slot_time=1.25e-3,
        ble_ci_quanta_candidates=np.array([8, 16, 32, 64], dtype=int),
        ble_ce_required_s=1.25e-3,
        ble_ce_max_s=5e-3,
        start_time_slot=0,
    )
    t2 = sample_ble_pair_timing(
        rand_gen=rng2,
        slot_time=1.25e-3,
        ble_ci_quanta_candidates=np.array([8, 16, 32, 64], dtype=int),
        ble_ce_required_s=1.25e-3,
        ble_ce_max_s=5e-3,
        start_time_slot=0,
    )

    assert t1 == t2
    assert t1["ci_slots"] in {8, 16, 32, 64}
    assert 1 <= t1["ce_slots"] <= t1["ci_slots"]
    assert t1["anchor_slot"] >= 0


def test_merge_config_with_defaults_rejects_unknown_radio_value():
    with pytest.raises(ValueError, match="radio"):
        merge_config_with_defaults(
            {
                "pair_generation_mode": "manual",
                "pair_parameters": [
                    {
                        "pair_id": 0,
                        "office_id": 0,
                        "radio": "zigbee",
                        "channel": 1,
                        "priority": 1.0,
                        "release_time_slot": 0,
                        "deadline_slot": 10,
                        "start_time_slot": 0,
                    }
                ],
            }
        )


def test_merge_config_with_defaults_rejects_invalid_wifi_channel_for_manual_mode():
    with pytest.raises(ValueError, match="wifi channel"):
        merge_config_with_defaults(
            {
                "cell_size": 1,
                "pair_generation_mode": "manual",
                "pair_parameters": [
                    {
                        "pair_id": 0,
                        "office_id": 0,
                        "radio": "wifi",
                        "channel": 1,
                        "priority": 1.0,
                        "release_time_slot": 0,
                        "deadline_slot": 10,
                        "start_time_slot": 0,
                        "wifi_anchor_slot": 0,
                        "wifi_period_slots": 16,
                        "wifi_tx_slots": 4,
                    }
                ],
            }
        )


def test_merge_config_with_defaults_rejects_invalid_ble_channel_for_manual_mode():
    with pytest.raises(ValueError, match="ble channel"):
        merge_config_with_defaults(
            {
                "cell_size": 1,
                "pair_generation_mode": "manual",
                "pair_parameters": [
                    {
                        "pair_id": 0,
                        "office_id": 0,
                        "radio": "ble",
                        "channel": 37,
                        "priority": 1.0,
                        "release_time_slot": 0,
                        "deadline_slot": 10,
                        "start_time_slot": 0,
                        "ble_anchor_slot": 0,
                        "ble_ci_slots": 16,
                        "ble_ce_slots": 2,
                    }
                ],
            }
        )


def test_merge_config_with_defaults_rejects_non_contiguous_pair_ids():
    with pytest.raises(ValueError, match="contiguous range"):
        merge_config_with_defaults(
            {
                "cell_size": 1,
                "pair_generation_mode": "manual",
                "pair_parameters": [
                    {
                        "pair_id": 1,
                        "office_id": 0,
                        "radio": "ble",
                        "channel": 1,
                        "priority": 1.0,
                        "release_time_slot": 0,
                        "deadline_slot": 10,
                        "start_time_slot": 0,
                        "ble_anchor_slot": 0,
                        "ble_ci_slots": 16,
                        "ble_ce_slots": 2,
                    }
                ],
            }
        )


def test_merge_config_with_defaults_rejects_out_of_range_office_id():
    with pytest.raises(ValueError, match="office_id"):
        merge_config_with_defaults(
            {
                "cell_size": 1,
                "pair_generation_mode": "manual",
                "pair_parameters": [
                    {
                        "pair_id": 0,
                        "office_id": 1,
                        "radio": "ble",
                        "channel": 1,
                        "priority": 1.0,
                        "release_time_slot": 0,
                        "deadline_slot": 10,
                        "start_time_slot": 0,
                        "ble_anchor_slot": 0,
                        "ble_ci_slots": 16,
                        "ble_ce_slots": 2,
                    }
                ],
            }
        )


def test_merge_config_with_defaults_sets_random_mode_when_pair_parameters_absent():
    merged = merge_config_with_defaults({})

    assert merged["pair_generation_mode"] == "random"
    assert merged["pair_parameters"] is None


def test_merge_config_with_defaults_defaults_ble_backend_to_legacy():
    merged = merge_config_with_defaults({})

    assert merged["ble_schedule_backend"] == "legacy"


def test_merge_config_with_defaults_accepts_macrocycle_hopping_ble_backend():
    merged = merge_config_with_defaults(
        {
            "ble_schedule_backend": "macrocycle_hopping_sdp",
        }
    )

    assert merged["ble_schedule_backend"] == "macrocycle_hopping_sdp"


def test_merge_config_with_defaults_accepts_ble_candidate_summary_keys():
    merged = merge_config_with_defaults(
        {
            "ble_max_offsets_per_pair": 4,
            "ble_log_candidate_summary": True,
        }
    )

    assert merged["ble_max_offsets_per_pair"] == 4
    assert merged["ble_log_candidate_summary"] is True


def test_merge_config_with_defaults_rejects_non_positive_ble_max_offsets_per_pair():
    with pytest.raises(ValueError, match="ble_max_offsets_per_pair"):
        merge_config_with_defaults({"ble_max_offsets_per_pair": 0})


def test_build_ble_hopping_inputs_from_env_derives_pair_configs_and_patterns():
    e = env(cell_size=1, pair_density_per_m2=0.2, seed=1, ble_channel_mode="per_ce")
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert ble_ids.size > 0

    pair_configs, cfg_dict, pattern_dict, num_channels = build_ble_hopping_inputs_from_env(e)

    assert num_channels == e.ble_channel_count
    assert len(pair_configs) == ble_ids.size
    assert set(cfg_dict) == set(ble_ids.tolist())
    assert set(pattern_dict) == set(ble_ids.tolist())
    first_cfg = cfg_dict[int(ble_ids[0])]
    assert first_cfg.connect_interval == int(e.pair_ble_ci_slots[int(ble_ids[0])])
    assert first_cfg.event_duration == int(e.pair_ble_ce_slots[int(ble_ids[0])])


def test_apply_ble_schedule_backend_legacy_is_noop():
    class DummyEnv:
        ble_schedule_backend = "legacy"

        def __init__(self):
            self.called = False

    dummy = DummyEnv()

    result = apply_ble_schedule_backend(dummy, {})

    assert result is None
    assert dummy.called is False


def test_apply_ble_schedule_backend_macrocycle_hopping_calls_solver_and_writes_channels(monkeypatch):
    class DummyEnv:
        def __init__(self):
            self.ble_schedule_backend = "macrocycle_hopping_sdp"
            self.ble_channel_mode = "per_ce"
            self.ble_channel_count = 37
            self.RADIO_BLE = 1
            self.pair_radio_type = np.array([1], dtype=int)
            self.pair_channel = np.array([8], dtype=int)
            self.pair_ble_ci_slots = np.array([16], dtype=int)
            self.pair_ble_ce_slots = np.array([2], dtype=int)
            self.pair_ble_anchor_slot = np.array([0], dtype=int)
            self.pair_release_time_slot = np.array([0], dtype=int)
            self.pair_deadline_slot = np.array([63], dtype=int)
            self.pair_ble_ce_feasible = np.array([True], dtype=bool)
            self.pair_ble_ce_channels = {}
            self.compute_macrocycle_slots = lambda: 16
            self.set_ble_ce_channel_map_called_with = None

        def set_ble_ce_channel_map(self, channel_map):
            self.set_ble_ce_channel_map_called_with = channel_map

    dummy = DummyEnv()
    fake_result = {
        "ce_channel_map": {0: np.array([3, 7], dtype=int)},
        "selected": {0: object()},
        "blocks": [],
        "overlap_blocks": [],
        "objective_value": 0.0,
    }
    calls = {}

    def fake_solver(**kwargs):
        calls["kwargs"] = kwargs
        return fake_result

    monkeypatch.setattr("sim_script.pd_mmw_template_ap_stats.solve_ble_hopping_for_env", fake_solver)

    result = apply_ble_schedule_backend(dummy, {"ble_max_offsets_per_pair": 4, "ble_log_candidate_summary": True})

    assert calls["kwargs"]["e"] is dummy
    assert calls["kwargs"]["config"]["ble_max_offsets_per_pair"] == 4
    assert result is fake_result
    assert dummy.set_ble_ce_channel_map_called_with == fake_result["ce_channel_map"]


def test_apply_manual_pair_parameters_overrides_env_arrays():
    class DummyEnv:
        RADIO_WIFI = 0
        RADIO_BLE = 1

        def __init__(self):
            self.n_pair = 2
            self.slot_time = 1.25e-3
            self.ble_channel_mode = "single"
            self._pair_packet_bits_input = None
            self._user_packet_bits_input = None
            self._pair_bandwidth_hz_input = None
            self._user_bandwidth_hz_input = None
            self.wifi_packet_bit = 1000.0
            self.ble_packet_bit = 200.0
            self.wifi_channel_bandwidth_hz = 20e6
            self.ble_channel_bandwidth_hz = 2e6
            self.pair_office_id = np.zeros(2, dtype=int)
            self.pair_radio_type = np.zeros(2, dtype=int)
            self.pair_channel = np.zeros(2, dtype=int)
            self.pair_priority = np.zeros(2, dtype=float)
            self.pair_release_time_slot = np.zeros(2, dtype=int)
            self.pair_deadline_slot = np.zeros(2, dtype=int)
            self.pair_start_time_slot = np.zeros(2, dtype=int)
            self.pair_wifi_anchor_slot = np.zeros(2, dtype=int)
            self.pair_wifi_period_slots = np.zeros(2, dtype=int)
            self.pair_wifi_tx_slots = np.zeros(2, dtype=int)
            self.pair_ble_anchor_slot = np.zeros(2, dtype=int)
            self.pair_ble_ci_slots = np.zeros(2, dtype=int)
            self.pair_ble_ce_slots = np.zeros(2, dtype=int)
            self.pair_ble_ce_required_s = np.zeros(2, dtype=float)
            self.pair_ble_ce_feasible = np.ones(2, dtype=bool)
            self.pair_ble_ce_channels = None
            self.pair_tx_locs = np.zeros((2, 2), dtype=float)
            self.pair_rx_locs = np.zeros((2, 2), dtype=float)
            self.device_dirs = np.zeros((2, 2), dtype=float)

        def _sample_pair_endpoint_in_office(self, office_id):
            return np.array([float(office_id), float(office_id) + 0.5], dtype=float)

        def _resolve_pair_float_array(self, pair_values, user_values, wifi_default, ble_default, attr_name):
            return np.where(self.pair_radio_type == self.RADIO_WIFI, wifi_default, ble_default).astype(float)

        def _compute_min_sinr(self):
            self.pair_min_sinr = np.ones(self.n_pair, dtype=float)
            self.device_min_sinr = self.pair_min_sinr
            self.user_min_sinr = self.pair_min_sinr

    e = DummyEnv()
    apply_manual_pair_parameters(
        e,
        [
            {
                "pair_id": 0,
                "office_id": 0,
                "radio": "ble",
                "channel": 8,
                "priority": 1.0,
                "release_time_slot": 0,
                "deadline_slot": 63,
                "start_time_slot": 0,
                "ble_anchor_slot": 12,
                "ble_ci_slots": 64,
                "ble_ce_slots": 5,
            },
            {
                "pair_id": 1,
                "office_id": 0,
                "radio": "wifi",
                "channel": 0,
                "priority": 2.0,
                "release_time_slot": 1,
                "deadline_slot": 31,
                "start_time_slot": 1,
                "wifi_anchor_slot": 6,
                "wifi_period_slots": 32,
                "wifi_tx_slots": 5,
            },
        ],
    )

    assert e.pair_deadline_slot[0] == 63
    assert e.pair_radio_type[1] == e.RADIO_WIFI
    assert e.pair_wifi_period_slots[1] == 32
    assert e.pair_ble_ci_slots[0] == 64
    assert e.pair_tx_locs[1].tolist() == [0.0, 0.5]
    assert e.pair_packet_bits.tolist() == [200.0, 1000.0]
    assert e.pair_bandwidth_hz.tolist() == [2e6, 20e6]


def test_apply_manual_pair_parameters_auto_ble_timing_is_seeded_and_reproducible():
    class DummyEnv:
        RADIO_WIFI = 0
        RADIO_BLE = 1

        def __init__(self, seed):
            self.n_pair = 1
            self.seed = seed
            self.slot_time = 1.25e-3
            self.ble_channel_mode = "single"
            self.rand_gen_loc = np.random.default_rng(seed)
            self.ble_ci_quanta_candidates = np.array([8, 16, 32, 64], dtype=int)
            self.ble_ce_required_s = 1.25e-3
            self.ble_ce_max_s = 5e-3
            self._pair_packet_bits_input = None
            self._user_packet_bits_input = None
            self._pair_bandwidth_hz_input = None
            self._user_bandwidth_hz_input = None
            self.wifi_packet_bit = 1000.0
            self.ble_packet_bit = 200.0
            self.wifi_channel_bandwidth_hz = 20e6
            self.ble_channel_bandwidth_hz = 2e6
            self.pair_office_id = np.zeros(1, dtype=int)
            self.pair_radio_type = np.zeros(1, dtype=int)
            self.pair_channel = np.zeros(1, dtype=int)
            self.pair_priority = np.zeros(1, dtype=float)
            self.pair_release_time_slot = np.zeros(1, dtype=int)
            self.pair_deadline_slot = np.zeros(1, dtype=int)
            self.pair_start_time_slot = np.zeros(1, dtype=int)
            self.pair_wifi_anchor_slot = np.zeros(1, dtype=int)
            self.pair_wifi_period_slots = np.zeros(1, dtype=int)
            self.pair_wifi_tx_slots = np.zeros(1, dtype=int)
            self.pair_ble_anchor_slot = np.zeros(1, dtype=int)
            self.pair_ble_ci_slots = np.zeros(1, dtype=int)
            self.pair_ble_ce_slots = np.zeros(1, dtype=int)
            self.pair_ble_ce_required_s = np.zeros(1, dtype=float)
            self.pair_ble_ce_feasible = np.ones(1, dtype=bool)
            self.pair_ble_ce_channels = None
            self.pair_tx_locs = np.zeros((1, 2), dtype=float)
            self.pair_rx_locs = np.zeros((1, 2), dtype=float)
            self.device_dirs = np.zeros((1, 2), dtype=float)

        def _sample_pair_endpoint_in_office(self, office_id):
            return np.array([float(office_id), float(office_id) + 0.5], dtype=float)

        def _resolve_pair_float_array(self, pair_values, user_values, wifi_default, ble_default, attr_name):
            return np.where(self.pair_radio_type == self.RADIO_WIFI, wifi_default, ble_default).astype(float)

        def _compute_min_sinr(self):
            self.pair_min_sinr = np.ones(self.n_pair, dtype=float)
            self.device_min_sinr = self.pair_min_sinr
            self.user_min_sinr = self.pair_min_sinr

    row = {
        "pair_id": 0,
        "office_id": 0,
        "radio": "ble",
        "channel": 8,
        "priority": 1.0,
        "release_time_slot": 0,
        "deadline_slot": 31,
        "start_time_slot": 0,
        "ble_anchor_slot": 0,
        "ble_timing_mode": "auto",
    }

    e1 = DummyEnv(seed=123)
    e2 = DummyEnv(seed=123)
    apply_manual_pair_parameters(e1, [row])
    apply_manual_pair_parameters(e2, [row])

    assert e1.pair_ble_ci_slots[0] > 0
    assert e1.pair_ble_ce_slots[0] > 0
    assert e1.pair_ble_ce_slots[0] <= e1.pair_ble_ci_slots[0]
    assert e1.pair_ble_anchor_slot[0] >= e1.pair_start_time_slot[0]
    assert e1.pair_ble_ci_slots.tolist() == e2.pair_ble_ci_slots.tolist()
    assert e1.pair_ble_ce_slots.tolist() == e2.pair_ble_ce_slots.tolist()
    assert e1.pair_ble_anchor_slot.tolist() == e2.pair_ble_anchor_slot.tolist()


def test_occupancy_within_time_window_checks_all_occupied_slots():
    occupancy = np.array([False, True, True, False, True], dtype=bool)

    assert _occupancy_within_time_window(occupancy, release_time_slot=1, deadline_slot=4)
    assert not _occupancy_within_time_window(occupancy, release_time_slot=2, deadline_slot=4)
    assert not _occupancy_within_time_window(occupancy, release_time_slot=1, deadline_slot=3)


def test_ble_data_channel_centers_follow_two_segment_mapping():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=1)
    assert e.get_ble_data_channel_center_mhz(0) == 2404.0
    assert e.get_ble_data_channel_center_mhz(10) == 2424.0
    assert e.get_ble_data_channel_center_mhz(11) == 2428.0
    assert e.get_ble_data_channel_center_mhz(36) == 2478.0


def test_ble_advertising_channel_centers_are_reserved():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=1)
    assert e.ble_advertising_center_freq_mhz == [2402.0, 2426.0, 2480.0]


def test_wifi_pairs_only_use_fixed_1_6_11_channels():
    e = env(cell_size=2, pair_density_per_m2=0.2, seed=1)
    wifi_ids = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    assert set(e.pair_channel[wifi_ids].tolist()).issubset({0, 5, 10})


def test_resample_ble_channels_stays_within_data_channel_indices():
    e = env(cell_size=1, pair_density_per_m2=0.2, seed=1)
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    e.resample_ble_channels(ble_ids)
    assert set(e.pair_channel[ble_ids].tolist()).issubset(set(e.ble_data_channel_indices.tolist()))
    assert all(
        e.get_ble_data_channel_center_mhz(ch) not in e.ble_advertising_center_freq_mhz
        for ch in e.pair_channel[ble_ids]
    )


def test_wifi_fixed_channels_do_not_cover_ble_advertising_centers():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=1)
    for wifi_idx in [0, 5, 10]:
        low, high = e._get_wifi_channel_range_hz(wifi_idx)
        for adv in e.ble_advertising_center_freq_mhz:
            assert not (low < adv * 1e6 < high)
