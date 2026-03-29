import json
from pathlib import Path

import numpy as np
import pytest

from sim_script.pd_mmw_template_ap_stats import (
    _aggregate_office_stats_from_arrays,
    _is_better_schedule_attempt,
    _occupancy_within_time_window,
    apply_manual_pair_parameters,
    apply_ble_schedule_backend,
    build_pair_parameter_rows,
    build_ble_hopping_inputs_from_env,
    build_schedule_rows,
    build_wifi_first_ble_external_interference_blocks,
    build_wifi_interference_blocks_from_schedule,
    build_wifi_local_reshuffle_candidates,
    diagnose_unscheduled_ble_pairs,
    load_json_config,
    merge_config_with_defaults,
    run_iterative_wifi_ble_coordination,
    run_wifi_first_schedule_attempt,
    strip_comment_keys,
    solve_ble_hopping_for_env,
    solve_ble_hopping_ga_for_env,
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


def test_merge_config_with_defaults_accepts_macrocycle_hopping_ga_backend():
    merged = merge_config_with_defaults(
        {
            "ble_schedule_backend": "macrocycle_hopping_ga",
            "ble_ga_population_size": 16,
            "ble_ga_generations": 20,
            "ble_ga_mutation_rate": 0.1,
            "ble_ga_crossover_rate": 0.85,
            "ble_ga_elite_count": 2,
            "ble_ga_seed": 11,
        }
    )

    assert merged["ble_schedule_backend"] == "macrocycle_hopping_ga"
    assert merged["ble_ga_population_size"] == 16
    assert merged["ble_ga_generations"] == 20
    assert merged["ble_ga_mutation_rate"] == 0.1
    assert merged["ble_ga_crossover_rate"] == 0.85
    assert merged["ble_ga_elite_count"] == 2
    assert merged["ble_ga_seed"] == 11


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


def test_build_wifi_interference_blocks_from_schedule_uses_wifi_slot_and_frequency_ranges():
    class DummyEnv:
        RADIO_WIFI = 0
        RADIO_BLE = 1

        def __init__(self):
            self.pair_radio_type = np.array([self.RADIO_WIFI], dtype=int)
            self.pair_channel = np.array([0], dtype=int)
            self._get_pair_link_range_hz = lambda pair_id: (2402.0e6, 2422.0e6)

    blocks = build_wifi_interference_blocks_from_schedule(
        DummyEnv(),
        [
            {
                "pair_id": 0,
                "radio": "wifi",
                "schedule_slot": 0,
                "occupied_slots_in_macrocycle": [0, 1],
            }
        ],
    )

    assert len(blocks) == 2
    assert blocks[0].start_slot == 0
    assert blocks[0].end_slot == 0
    assert blocks[0].freq_low_mhz == 2402.0
    assert blocks[0].freq_high_mhz == 2422.0
    assert blocks[0].source_type == "wifi"
    assert blocks[0].source_pair_id == 0


def test_build_wifi_interference_blocks_ignores_unscheduled_or_non_wifi_pairs():
    class DummyEnv:
        RADIO_WIFI = 0
        RADIO_BLE = 1

        def __init__(self):
            self.pair_radio_type = np.array([self.RADIO_WIFI, self.RADIO_BLE], dtype=int)
            self.pair_channel = np.array([0, 10], dtype=int)
            self._get_pair_link_range_hz = lambda pair_id: (2402.0e6, 2422.0e6) if int(pair_id) == 0 else (2436.0e6, 2438.0e6)

    blocks = build_wifi_interference_blocks_from_schedule(
        DummyEnv(),
        [
            {
                "pair_id": 0,
                "radio": "wifi",
                "schedule_slot": -1,
                "occupied_slots_in_macrocycle": [0, 1],
            },
            {
                "pair_id": 1,
                "radio": "ble",
                "schedule_slot": 0,
                "occupied_slots_in_macrocycle": [0, 1],
            },
        ],
    )

    assert blocks == []


def test_build_wifi_first_ble_external_interference_blocks_runs_wifi_first_assignment(monkeypatch):
    class DummyEnv:
        pass

    dummy = DummyEnv()
    calls = {}

    def fake_assign(e, preferred_slots, allow_partial=False, wifi_first=False, **kwargs):
        calls["assign"] = {
            "e": e,
            "preferred_slots": np.asarray(preferred_slots, dtype=int).copy(),
            "allow_partial": allow_partial,
            "wifi_first": wifi_first,
        }
        return (
            np.array([0, -1], dtype=int),
            8,
            np.array([[True, False, False, False, False, False, False, False], [False] * 8], dtype=bool),
            [1],
        )

    def fake_rows(e, z_vec, occupied_slots, macrocycle_slots, ble_slot_stats=None):
        calls["rows"] = {
            "e": e,
            "z_vec": np.asarray(z_vec, dtype=int).copy(),
            "occupied_slots": np.asarray(occupied_slots, dtype=bool).copy(),
            "macrocycle_slots": macrocycle_slots,
        }
        return [
            {
                "pair_id": 0,
                "radio": "wifi",
                "schedule_slot": 0,
                "occupied_slots_in_macrocycle": [0],
            },
            {
                "pair_id": 1,
                "radio": "ble",
                "schedule_slot": -1,
                "occupied_slots_in_macrocycle": [],
            },
        ]

    expected_blocks = [object()]

    def fake_blocks(e, rows):
        calls["blocks"] = {"e": e, "rows": rows}
        return expected_blocks

    monkeypatch.setattr("sim_script.pd_mmw_template_ap_stats.assign_macrocycle_start_slots", fake_assign)
    monkeypatch.setattr("sim_script.pd_mmw_template_ap_stats.compute_pair_parameter_rows", fake_rows)
    monkeypatch.setattr("sim_script.pd_mmw_template_ap_stats.build_wifi_interference_blocks_from_schedule", fake_blocks)

    preferred_slots = np.array([3, 5], dtype=int)
    result = build_wifi_first_ble_external_interference_blocks(dummy, preferred_slots)

    assert result is expected_blocks
    assert calls["assign"]["e"] is dummy
    assert calls["assign"]["allow_partial"] is True
    assert calls["assign"]["wifi_first"] is True
    assert np.array_equal(calls["assign"]["preferred_slots"], preferred_slots)
    assert calls["rows"]["macrocycle_slots"] == 8
    assert calls["blocks"]["rows"][0]["radio"] == "wifi"




def test_apply_ble_schedule_backend_macrocycle_hopping_ga_calls_solver_and_writes_channels(monkeypatch):
    class DummyEnv:
        def __init__(self):
            self.ble_schedule_backend = "macrocycle_hopping_ga"
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
        "ce_channel_map": {0: np.array([5, 9], dtype=int)},
        "selected": {0: object()},
        "blocks": [],
        "overlap_blocks": [],
        "objective_value": 0.0,
        "ga_result": object(),
    }
    calls = {}

    def fake_solver(**kwargs):
        calls["kwargs"] = kwargs
        return fake_result

    monkeypatch.setattr("sim_script.pd_mmw_template_ap_stats.solve_ble_hopping_ga_for_env", fake_solver)

    external_blocks = [object()]
    result = apply_ble_schedule_backend(
        dummy,
        {
            "ble_schedule_backend": "macrocycle_hopping_ga",
            "ble_ga_population_size": 16,
            "ble_ga_generations": 20,
            "ble_ga_mutation_rate": 0.1,
            "ble_ga_crossover_rate": 0.85,
            "ble_ga_elite_count": 2,
            "ble_ga_seed": 11,
        },
        external_interference_blocks=external_blocks,
    )

    assert calls["kwargs"]["e"] is dummy
    assert calls["kwargs"]["config"]["ble_schedule_backend"] == "macrocycle_hopping_ga"
    assert calls["kwargs"]["config"]["ble_ga_population_size"] == 16
    assert calls["kwargs"]["external_interference_blocks"] is external_blocks
    assert result is fake_result
    assert dummy.set_ble_ce_channel_map_called_with == fake_result["ce_channel_map"]


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

    external_blocks = [object()]
    result = apply_ble_schedule_backend(
        dummy,
        {"ble_max_offsets_per_pair": 4, "ble_log_candidate_summary": True},
        external_interference_blocks=external_blocks,
    )

    assert calls["kwargs"]["e"] is dummy
    assert calls["kwargs"]["config"]["ble_max_offsets_per_pair"] == 4
    assert calls["kwargs"]["external_interference_blocks"] is external_blocks
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


def test_merge_config_with_defaults_accepts_iterative_coordination_keys():
    merged = merge_config_with_defaults(
        {
            "wifi_ble_coordination_mode": "iterative",
            "wifi_ble_coordination_rounds": 2,
            "wifi_ble_coordination_top_k_wifi_pairs": 3,
            "wifi_ble_coordination_candidate_start_limit": 4,
        }
    )

    assert merged["wifi_ble_coordination_mode"] == "iterative"
    assert merged["wifi_ble_coordination_rounds"] == 2
    assert merged["wifi_ble_coordination_top_k_wifi_pairs"] == 3
    assert merged["wifi_ble_coordination_candidate_start_limit"] == 4


def test_merge_config_with_defaults_rejects_invalid_iterative_coordination_mode():
    with pytest.raises(ValueError, match="wifi_ble_coordination_mode"):
        merge_config_with_defaults({"wifi_ble_coordination_mode": "loop"})


def test_merge_config_with_defaults_rejects_negative_iterative_coordination_limit():
    with pytest.raises(ValueError, match="wifi_ble_coordination_rounds"):
        merge_config_with_defaults({"wifi_ble_coordination_rounds": -1})


def test_run_wifi_first_schedule_attempt_returns_structured_result(monkeypatch):
    class DummyEnv:
        def __init__(self):
            self.n_pair = 2
            self.RADIO_WIFI = 0
            self.RADIO_BLE = 1
            self.pair_radio_type = np.array([0, 1], dtype=int)
            self.pair_priority = np.array([2.0, 1.0], dtype=float)

    dummy = DummyEnv()

    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.build_wifi_first_ble_external_interference_blocks",
        lambda e, preferred_slots: [{"start_slot": 0}],
    )
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.apply_ble_schedule_backend",
        lambda e, config, external_interference_blocks=None: {"ce_channel_map": {}},
    )
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.retry_ble_channels_and_assign_macrocycle",
        lambda e, preferred_slots, max_ble_channel_retries=0, wifi_first=False, return_ble_stats=False: (
            np.array([0, 1], dtype=int),
            8,
            np.array([[True] + [False] * 7, [False, True] + [False] * 6], dtype=bool),
            [],
            0,
            {1: {"effective_ble_channels": 2, "scheduled_ble_pairs": 1, "no_collision_probability": 1.0}},
        ),
    )
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.compute_pair_parameter_rows",
        lambda e, starts, occupancy, macrocycle_slots, ble_slot_stats=None: [
            {"pair_id": 0, "radio": "wifi", "schedule_slot": 0, "occupied_slots_in_macrocycle": [0], "macrocycle_slots": 8},
            {"pair_id": 1, "radio": "ble", "schedule_slot": 1, "occupied_slots_in_macrocycle": [1], "macrocycle_slots": 8},
        ],
    )
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.build_schedule_rows",
        lambda rows: [{"schedule_slot": 0, "pair_ids": [0]}, {"schedule_slot": 1, "pair_ids": [1]}],
    )
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.get_pair_channel_ranges_mhz",
        lambda e, pair_ids: {0: (2412.0, 2432.0), 1: (2440.0, 2442.0)},
    )
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.build_schedule_plot_rows",
        lambda pair_rows, pair_channel_ranges, e=None: [],
    )

    attempt = run_wifi_first_schedule_attempt(
        dummy,
        {"ble_schedule_backend": "macrocycle_hopping_sdp", "ble_channel_retries": 0},
        np.array([0, 1], dtype=int),
    )

    assert attempt.total_scheduled_count == 2
    assert attempt.ble_scheduled_count == 1
    assert attempt.wifi_scheduled_count == 1
    assert len(attempt.wifi_interference_blocks) == 1


def test_diagnose_unscheduled_ble_pairs_reports_wifi_capacity_blocked():
    class DummyEnv:
        RADIO_WIFI = 0
        RADIO_BLE = 1

        def __init__(self):
            self.pair_radio_type = np.array([0, 1], dtype=int)
            self.pair_ble_ce_feasible = np.array([True, True], dtype=bool)
            self.pair_release_time_slot = np.array([0, 0], dtype=int)
            self.pair_deadline_slot = np.array([7, 7], dtype=int)
            self._period_slots = np.array([4, 4], dtype=int)
            self._conflict = np.zeros((2, 2), dtype=bool)

        def get_pair_period_slots(self):
            return self._period_slots

        def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
            occ = np.zeros(macrocycle_slots, dtype=bool)
            occ[int(start_slot)] = True
            return occ

        def get_ble_start_slot_capacity(self, wifi_pair_ids, wifi_start_slots, start_slot):
            return 0

        def build_pair_conflict_matrix(self):
            return self._conflict

        def is_slot_channel_conflict(self, *args, **kwargs):
            return False

    dummy = DummyEnv()
    attempt = type("Attempt", (), {
        "env_obj": dummy,
        "scheduled_pair_ids": [0],
        "unscheduled_pair_ids": [1],
        "preferred_slots": np.array([0, 0], dtype=int),
        "schedule_start_slots": np.array([0, -1], dtype=int),
        "macrocycle_slots": 8,
        "occupancy": np.array([[True] + [False] * 7, [False] * 8], dtype=bool),
    })()

    result = diagnose_unscheduled_ble_pairs(attempt)

    assert result["ble_pair_diagnostics"][0]["reason"] == "wifi_capacity_blocked"


def test_build_wifi_local_reshuffle_candidates_changes_only_blocking_wifi_pairs():
    class DummyEnv:
        RADIO_WIFI = 0
        RADIO_BLE = 1

        def __init__(self):
            self.pair_radio_type = np.array([0, 0, 1], dtype=int)
            self.pair_release_time_slot = np.zeros(3, dtype=int)
            self.pair_deadline_slot = np.full(3, 7, dtype=int)
            self._period_slots = np.array([4, 4, 4], dtype=int)
            self._conflict = np.zeros((3, 3), dtype=bool)

        def get_pair_period_slots(self):
            return self._period_slots

        def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
            occ = np.zeros(macrocycle_slots, dtype=bool)
            occ[int(start_slot)] = True
            return occ

        def build_pair_conflict_matrix(self):
            return self._conflict

    dummy = DummyEnv()
    attempt = type("Attempt", (), {
        "env_obj": dummy,
        "scheduled_pair_ids": [0, 1],
        "preferred_slots": np.array([0, 1, 2], dtype=int),
        "schedule_start_slots": np.array([0, 1, -1], dtype=int),
        "macrocycle_slots": 8,
        "occupancy": np.array([[True] + [False] * 7, [False, True] + [False] * 6, [False] * 8], dtype=bool),
    })()
    diagnostics = {"blocking_wifi_pair_counts": {1: 2}}

    candidates = build_wifi_local_reshuffle_candidates(
        attempt,
        diagnostics,
        {
            "wifi_ble_coordination_top_k_wifi_pairs": 1,
            "wifi_ble_coordination_candidate_start_limit": 2,
        },
    )

    assert candidates
    assert all(candidate["pair_id"] == 1 for candidate in candidates)
    assert all(int(candidate["preferred_slots"][0]) == 0 for candidate in candidates)


def test_run_iterative_wifi_ble_coordination_keeps_better_attempt(monkeypatch):
    class DummyEnv:
        pass

    baseline = type("Attempt", (), {
        "total_scheduled_count": 2,
        "wifi_scheduled_count": 1,
        "ble_scheduled_count": 1,
        "overlap_row_count": 0,
    })()
    improved = type("Attempt", (), {
        "total_scheduled_count": 3,
        "wifi_scheduled_count": 1,
        "ble_scheduled_count": 2,
        "overlap_row_count": 0,
    })()

    calls = {"count": 0}

    def fake_run(e, config, preferred_slots, coordination_round=0, candidate_index=0):
        calls["count"] += 1
        return baseline if calls["count"] == 1 else improved

    monkeypatch.setattr("sim_script.pd_mmw_template_ap_stats.run_wifi_first_schedule_attempt", fake_run)
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.diagnose_unscheduled_ble_pairs",
        lambda attempt: {"blocking_wifi_pair_counts": {0: 1}, "ble_pair_diagnostics": []},
    )
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.build_wifi_local_reshuffle_candidates",
        lambda attempt, diagnostics, config: [{"pair_id": 0, "preferred_slots": np.array([1, 2], dtype=int), "new_start_slot": 1}],
    )

    best, summary = run_iterative_wifi_ble_coordination(
        DummyEnv(),
        {"wifi_ble_coordination_mode": "iterative", "wifi_ble_coordination_rounds": 1},
        np.array([0, 0], dtype=int),
    )

    assert best is improved
    assert summary["improved"] is True
    assert summary["tested_candidates"] == 1


def test_is_better_schedule_attempt_rejects_candidate_that_drops_wifi():
    baseline = type("Attempt", (), {
        "total_scheduled_count": 36,
        "wifi_scheduled_count": 5,
        "ble_scheduled_count": 31,
        "overlap_row_count": 0,
    })()
    candidate = type("Attempt", (), {
        "total_scheduled_count": 38,
        "wifi_scheduled_count": 4,
        "ble_scheduled_count": 34,
        "overlap_row_count": 0,
    })()

    assert not _is_better_schedule_attempt(candidate, baseline, baseline_wifi_count=5)


def test_is_better_schedule_attempt_prefers_higher_total_when_wifi_floor_is_kept():
    baseline = type("Attempt", (), {
        "total_scheduled_count": 36,
        "wifi_scheduled_count": 5,
        "ble_scheduled_count": 31,
        "overlap_row_count": 0,
    })()
    candidate = type("Attempt", (), {
        "total_scheduled_count": 37,
        "wifi_scheduled_count": 5,
        "ble_scheduled_count": 32,
        "overlap_row_count": 0,
    })()

    assert _is_better_schedule_attempt(candidate, baseline, baseline_wifi_count=5)


def test_run_iterative_wifi_ble_coordination_retains_baseline_when_candidate_drops_wifi(monkeypatch):
    class DummyEnv:
        pass

    baseline = type("Attempt", (), {
        "total_scheduled_count": 36,
        "wifi_scheduled_count": 5,
        "ble_scheduled_count": 31,
        "overlap_row_count": 0,
    })()
    dropped_wifi = type("Attempt", (), {
        "total_scheduled_count": 38,
        "wifi_scheduled_count": 4,
        "ble_scheduled_count": 34,
        "overlap_row_count": 0,
    })()

    calls = {"count": 0}

    def fake_run(e, config, preferred_slots, coordination_round=0, candidate_index=0):
        calls["count"] += 1
        return baseline if calls["count"] == 1 else dropped_wifi

    monkeypatch.setattr("sim_script.pd_mmw_template_ap_stats.run_wifi_first_schedule_attempt", fake_run)
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.diagnose_unscheduled_ble_pairs",
        lambda attempt: {"blocking_wifi_pair_counts": {0: 1}, "ble_pair_diagnostics": []},
    )
    monkeypatch.setattr(
        "sim_script.pd_mmw_template_ap_stats.build_wifi_local_reshuffle_candidates",
        lambda attempt, diagnostics, config: [{"pair_id": 0, "preferred_slots": np.array([1, 2], dtype=int), "new_start_slot": 1}],
    )

    best, summary = run_iterative_wifi_ble_coordination(
        DummyEnv(),
        {"wifi_ble_coordination_mode": "iterative", "wifi_ble_coordination_rounds": 1},
        np.array([0, 0], dtype=int),
    )

    assert best is baseline
    assert summary["improved"] is False
    assert summary["tested_candidates"] == 1
