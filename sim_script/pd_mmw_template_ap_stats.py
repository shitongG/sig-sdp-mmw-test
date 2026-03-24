import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from sim_script.plot_schedule_from_csv import render_all_from_csv
from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.mmw import mmw
from sim_src.env.env import env, sample_ble_pair_timing
from sim_src.util import resolve_torch_device


DEFAULT_CONFIG = {
    "cell_size": 2,
    "pair_density": 0.05,
    "seed": None,
    "mmw_nit": 200,
    "mmw_eta": 0.05,
    "use_gpu": False,
    "gpu_id": 0,
    "max_slots": 300,
    "ble_channel_retries": 0,
    "ble_channel_mode": "single",
    "ble_schedule_backend": "legacy",
    "ble_max_offsets_per_pair": None,
    "ble_log_candidate_summary": False,
    "ble_ga_population_size": 32,
    "ble_ga_generations": 40,
    "ble_ga_mutation_rate": 0.05,
    "ble_ga_crossover_rate": 0.8,
    "ble_ga_elite_count": 1,
    "ble_ga_seed": 7,
    "output_dir": "sim_script/output",
    "wifi_first_ble_scheduling": False,
    "pair_generation_mode": "random",
    "pair_parameters": None,
}


def _load_local_module(filename: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_ble_hopping_module():
    return _load_local_module("ble_macrocycle_hopping_sdp.py", "ble_macrocycle_hopping_sdp")


def _load_local_ble_hopping_ga_module():
    _load_local_ble_hopping_module()
    return _load_local_module("ble_macrocycle_hopping_ga.py", "ble_macrocycle_hopping_ga")


def strip_comment_keys(payload):
    if isinstance(payload, dict):
        return {
            key: strip_comment_keys(value)
            for key, value in payload.items()
            if not str(key).startswith("_comment")
        }
    if isinstance(payload, list):
        return [strip_comment_keys(item) for item in payload]
    return payload


def load_json_config(config_path: Path):
    config_path = Path(config_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = strip_comment_keys(raw)
    output_dir = config.get("output_dir", DEFAULT_CONFIG["output_dir"])
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = (config_path.parent / output_dir_path).resolve()
    config["output_dir"] = str(output_dir_path)
    return config


def _validate_pair_parameters(pair_parameters, cell_size: int):
    required_common = {
        "pair_id",
        "office_id",
        "radio",
        "channel",
        "priority",
        "release_time_slot",
        "deadline_slot",
        "start_time_slot",
    }
    required_wifi = {"wifi_anchor_slot", "wifi_period_slots", "wifi_tx_slots"}
    required_ble = {"ble_anchor_slot", "ble_ci_slots", "ble_ce_slots"}
    seen_pair_ids = set()
    expected_pair_ids = set(range(len(pair_parameters)))
    n_office = int(cell_size) ** 2

    for row in pair_parameters:
        missing_common = sorted(required_common - set(row))
        if missing_common:
            raise ValueError(f"pair_parameters missing required fields: {missing_common}")
        pair_id = int(row["pair_id"])
        if pair_id in seen_pair_ids:
            raise ValueError("pair_parameters pair_id must be unique.")
        seen_pair_ids.add(pair_id)
        office_id = int(row["office_id"])
        if office_id < 0 or office_id >= n_office:
            raise ValueError("pair_parameters office_id must be within [0, cell_size^2).")
        radio = str(row["radio"])
        if radio not in {"wifi", "ble"}:
            raise ValueError("pair_parameters radio must be 'wifi' or 'ble'.")
        channel = int(row["channel"])
        if radio == "wifi" and channel not in {0, 5, 10}:
            raise ValueError("pair_parameters wifi channel must be one of {0, 5, 10}.")
        if radio == "ble" and not (0 <= channel <= 36):
            raise ValueError("pair_parameters ble channel must be within [0, 36].")
        if int(row["deadline_slot"]) < int(row["release_time_slot"]):
            raise ValueError("pair_parameters deadline_slot must be >= release_time_slot.")
        if int(row["start_time_slot"]) < int(row["release_time_slot"]):
            raise ValueError("pair_parameters start_time_slot must be >= release_time_slot.")
        if radio == "wifi":
            missing_radio = sorted(required_wifi - set(row))
            if missing_radio:
                raise ValueError(f"pair_parameters missing {radio} fields: {missing_radio}")
        else:
            ble_timing_mode = str(row.get("ble_timing_mode", "manual"))
            if ble_timing_mode not in {"manual", "auto"}:
                raise ValueError("pair_parameters ble_timing_mode must be 'manual' or 'auto'.")
            missing_radio = sorted(required_ble - set(row))
            if ble_timing_mode == "auto":
                missing_radio = [field for field in missing_radio if field not in {"ble_ci_slots", "ble_ce_slots"}]
            if missing_radio:
                raise ValueError(f"pair_parameters missing {radio} fields: {missing_radio}")
    if seen_pair_ids != expected_pair_ids:
        raise ValueError("pair_parameters pair_id must form a contiguous range starting at 0.")


def merge_config_with_defaults(config: dict):
    merged = DEFAULT_CONFIG.copy()
    for key, value in config.items():
        if key not in DEFAULT_CONFIG:
            raise ValueError(f"Unknown config key: {key}")
        merged[key] = value
    if int(merged["max_slots"]) < 2:
        raise ValueError("max_slots must be at least 2.")
    if merged["ble_channel_mode"] not in {"single", "per_ce"}:
        raise ValueError("ble_channel_mode must be 'single' or 'per_ce'.")
    if merged["ble_schedule_backend"] not in {"legacy", "macrocycle_hopping_sdp", "macrocycle_hopping_ga"}:
        raise ValueError("ble_schedule_backend must be 'legacy', 'macrocycle_hopping_sdp', or 'macrocycle_hopping_ga'.")
    ble_max_offsets_per_pair = merged["ble_max_offsets_per_pair"]
    if ble_max_offsets_per_pair is not None and int(ble_max_offsets_per_pair) < 1:
        raise ValueError("ble_max_offsets_per_pair must be None or a positive integer.")
    merged["ble_max_offsets_per_pair"] = (
        None if ble_max_offsets_per_pair is None else int(ble_max_offsets_per_pair)
    )
    merged["ble_log_candidate_summary"] = bool(merged["ble_log_candidate_summary"])
    merged["ble_ga_population_size"] = int(merged["ble_ga_population_size"])
    merged["ble_ga_generations"] = int(merged["ble_ga_generations"])
    merged["ble_ga_mutation_rate"] = float(merged["ble_ga_mutation_rate"])
    merged["ble_ga_crossover_rate"] = float(merged["ble_ga_crossover_rate"])
    merged["ble_ga_elite_count"] = int(merged["ble_ga_elite_count"])
    merged["ble_ga_seed"] = int(merged["ble_ga_seed"]) if merged["ble_ga_seed"] is not None else None
    if merged["pair_generation_mode"] not in {"random", "manual"}:
        raise ValueError("pair_generation_mode must be 'random' or 'manual'.")
    if merged["pair_generation_mode"] == "manual":
        if not merged["pair_parameters"]:
            raise ValueError("pair_parameters must be provided when pair_generation_mode='manual'.")
        _validate_pair_parameters(merged["pair_parameters"], int(merged["cell_size"]))
    else:
        merged["pair_parameters"] = None
    return merged


def build_ble_hopping_inputs_from_env(e: env):
    ble_hopping = _load_local_ble_hopping_module()
    HoppingPattern = ble_hopping.HoppingPattern
    PairConfig = ble_hopping.PairConfig

    num_channels = int(getattr(e, "ble_channel_count", 37))
    if num_channels <= 0:
        raise ValueError("BLE channel count must be positive.")

    macrocycle_slots = max(int(e.compute_macrocycle_slots()), 1)
    pair_configs = []
    cfg_dict = {}
    pattern_dict = {}

    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    for pair_id in ble_ids:
        pair_id = int(pair_id)
        ci_slots = int(e.pair_ble_ci_slots[pair_id])
        ce_slots = int(e.pair_ble_ce_slots[pair_id])
        event_count = max(1, macrocycle_slots // max(ci_slots, 1))
        cfg = PairConfig(
            pair_id=pair_id,
            release_time=int(e.pair_release_time_slot[pair_id]),
            deadline=int(e.pair_deadline_slot[pair_id]),
            connect_interval=ci_slots,
            event_duration=ce_slots,
            num_events=event_count,
        )
        pair_configs.append(cfg)
        cfg_dict[pair_id] = cfg

        base_channel = int(e.pair_channel[pair_id]) % num_channels
        pattern_dict[pair_id] = [
            HoppingPattern(pattern_id=0, start_channel=base_channel, hop_increment=1 if num_channels > 1 else 0),
            HoppingPattern(
                pattern_id=1,
                start_channel=(base_channel + pair_id + 3) % num_channels,
                hop_increment=5 if num_channels > 1 else 0,
            ),
            HoppingPattern(
                pattern_id=2,
                start_channel=(base_channel + 2 * pair_id + 7) % num_channels,
                hop_increment=9 if num_channels > 2 else 0,
            ),
        ]

    return pair_configs, cfg_dict, pattern_dict, num_channels


def solve_ble_hopping_for_env(
    e: env,
    config: dict | None = None,
    external_interference_blocks=None,
):
    ble_hopping = _load_local_ble_hopping_module()
    build_candidate_states = ble_hopping.build_candidate_states
    print_candidate_summary = ble_hopping.print_candidate_summary
    solve_ble_hopping_schedule = ble_hopping.solve_ble_hopping_schedule

    pair_configs, cfg_dict, pattern_dict, num_channels = build_ble_hopping_inputs_from_env(e)
    if not pair_configs:
        return {
            "problem": None,
            "Y": None,
            "selected": {},
            "blocks": [],
            "overlap_blocks": [],
            "ce_channel_map": {},
            "objective_value": 0.0,
        }

    pair_ids = [cfg.pair_id for cfg in pair_configs]
    max_offsets_per_pair = None if config is None else config.get("ble_max_offsets_per_pair")
    if config is not None and config.get("ble_log_candidate_summary", False):
        print_candidate_summary(
            pair_configs=pair_configs,
            pattern_dict=pattern_dict,
            max_offsets_per_pair=max_offsets_per_pair,
        )
    states, _, A_k = build_candidate_states(
        pair_configs,
        pattern_dict,
        max_offsets_per_pair=max_offsets_per_pair,
    )
    return solve_ble_hopping_schedule(
        pair_configs=pair_configs,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        pair_ids=pair_ids,
        A_k=A_k,
        states=states,
        num_channels=num_channels,
        pair_weight=None,
        hard_collision_threshold=None,
        external_interference_blocks=external_interference_blocks,
    )


def solve_ble_hopping_ga_for_env(
    e: env,
    config: dict | None = None,
    external_interference_blocks=None,
):
    ble_hopping = _load_local_ble_hopping_module()
    ble_hopping_ga = _load_local_ble_hopping_ga_module()
    build_candidate_states = ble_hopping.build_candidate_states
    print_candidate_summary = ble_hopping.print_candidate_summary
    solve_ble_hopping_schedule_ga = ble_hopping_ga.solve_ble_hopping_schedule_ga

    pair_configs, cfg_dict, pattern_dict, num_channels = build_ble_hopping_inputs_from_env(e)
    if not pair_configs:
        return {
            "problem": None,
            "Y": None,
            "selected": {},
            "blocks": [],
            "overlap_blocks": [],
            "ce_channel_map": {},
            "objective_value": 0.0,
            "ga_result": None,
        }

    pair_ids = [cfg.pair_id for cfg in pair_configs]
    max_offsets_per_pair = None if config is None else config.get("ble_max_offsets_per_pair")
    if config is not None and config.get("ble_log_candidate_summary", False):
        print_candidate_summary(
            pair_configs=pair_configs,
            pattern_dict=pattern_dict,
            max_offsets_per_pair=max_offsets_per_pair,
        )
    states, _, A_k = build_candidate_states(
        pair_configs,
        pattern_dict,
        max_offsets_per_pair=max_offsets_per_pair,
    )
    ga_result = solve_ble_hopping_schedule_ga(
        candidate_states=states,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=num_channels,
        pair_ids=pair_ids,
        pair_weight=None,
        external_interference_blocks=external_interference_blocks,
        population_size=32 if config is None else int(config.get("ble_ga_population_size", 32)),
        generations=40 if config is None else int(config.get("ble_ga_generations", 40)),
        mutation_rate=0.05 if config is None else float(config.get("ble_ga_mutation_rate", 0.05)),
        crossover_rate=0.8 if config is None else float(config.get("ble_ga_crossover_rate", 0.8)),
        elite_count=1 if config is None else int(config.get("ble_ga_elite_count", 1)),
        seed=None if config is None else config.get("ble_ga_seed"),
    )
    return {
        "problem": None,
        "Y": None,
        "selected": ga_result.selected,
        "blocks": ga_result.blocks,
        "overlap_blocks": ga_result.overlap_blocks,
        "ce_channel_map": ga_result.ce_channel_map,
        "objective_value": float(ga_result.best_fitness),
        "ga_result": ga_result,
        "A_k": A_k,
        "states": states,
    }


def build_wifi_interference_blocks_from_schedule(e: env, scheduled_pair_rows):
    ble_hopping = _load_local_ble_hopping_module()
    ExternalInterferenceBlock = ble_hopping.ExternalInterferenceBlock

    blocks = []
    for row in scheduled_pair_rows:
        if row.get("radio") != "wifi":
            continue
        if int(row.get("schedule_slot", -1)) < 0:
            continue
        pair_id = int(row["pair_id"])
        low_hz, high_hz = e._get_pair_link_range_hz(pair_id)
        for slot in row.get("occupied_slots_in_macrocycle", []):
            blocks.append(
                ExternalInterferenceBlock(
                    start_slot=int(slot),
                    end_slot=int(slot),
                    freq_low_mhz=float(low_hz / 1e6),
                    freq_high_mhz=float(high_hz / 1e6),
                    source_type="wifi",
                    source_pair_id=pair_id,
                )
            )
    return blocks


def apply_ble_schedule_backend(e: env, config: dict, external_interference_blocks=None):
    backend = str(config.get("ble_schedule_backend", getattr(e, "ble_schedule_backend", "legacy")))
    if backend == "legacy":
        return None
    if backend not in {"macrocycle_hopping_sdp", "macrocycle_hopping_ga"}:
        raise ValueError("ble_schedule_backend must be 'legacy', 'macrocycle_hopping_sdp', or 'macrocycle_hopping_ga'.")

    if getattr(e, "ble_channel_mode", "single") != "per_ce":
        e.ble_channel_mode = "per_ce"
        if getattr(e, "pair_ble_ce_channels", None) is None:
            e.pair_ble_ce_channels = {}
        if getattr(e, "_manual_ble_ce_channel_pairs", None) is None:
            e._manual_ble_ce_channel_pairs = set()

    solver = solve_ble_hopping_for_env if backend == "macrocycle_hopping_sdp" else solve_ble_hopping_ga_for_env
    result = solver(
        e=e,
        config=config,
        external_interference_blocks=external_interference_blocks,
    )
    if result.get("ce_channel_map") and hasattr(e, "set_ble_ce_channel_map"):
        e.set_ble_ce_channel_map(result["ce_channel_map"])
    return result


def build_wifi_first_ble_external_interference_blocks(e: env, preferred_slots: np.ndarray):
    starts, macrocycle_slots, occupancy, _ = assign_macrocycle_start_slots(
        e,
        preferred_slots,
        allow_partial=True,
        wifi_first=True,
    )
    pair_rows = compute_pair_parameter_rows(
        e,
        starts,
        occupancy,
        macrocycle_slots,
    )
    return build_wifi_interference_blocks_from_schedule(e, pair_rows)


def resolve_runtime_config(args):
    cli_values = vars(args).copy()
    config_path = cli_values.pop("config", None)
    cli_overrides = {key: value for key, value in cli_values.items() if value is not None}
    if config_path is None:
        return merge_config_with_defaults(cli_overrides)
    file_config = merge_config_with_defaults(load_json_config(Path(config_path)))
    file_config.update(cli_overrides)
    return merge_config_with_defaults(file_config)


def compute_pair_density_for_manual_pairs(cell_size: int, pair_count: int, cell_edge: float = 7.0):
    office_area_m2 = float(cell_edge) ** 2
    n_office = int(cell_size) ** 2
    if n_office <= 0:
        raise ValueError("cell_size must be positive.")
    base_density = float(pair_count) / float(n_office * office_area_m2)
    return math.nextafter(base_density, float("inf"))


def apply_manual_pair_parameters(e: env, pair_parameters):
    if len(pair_parameters) != int(e.n_pair):
        raise ValueError("manual pair count must match env.n_pair")

    ble_timing_seed = 0 if getattr(e, "seed", None) is None else int(getattr(e, "seed"))
    ble_timing_rng = np.random.default_rng(ble_timing_seed)

    e.pair_radio_type = np.zeros(e.n_pair, dtype=int)
    e.pair_office_id = np.zeros(e.n_pair, dtype=int)
    e.pair_channel = np.zeros(e.n_pair, dtype=int)
    e.pair_priority = np.zeros(e.n_pair, dtype=float)
    e.pair_start_time_slot = np.zeros(e.n_pair, dtype=int)
    e.pair_release_time_slot = np.zeros(e.n_pair, dtype=int)
    e.pair_deadline_slot = np.zeros(e.n_pair, dtype=int)
    e.pair_wifi_anchor_slot = np.zeros(e.n_pair, dtype=int)
    e.pair_wifi_period_slots = np.zeros(e.n_pair, dtype=int)
    e.pair_wifi_tx_slots = np.zeros(e.n_pair, dtype=int)
    e.pair_ble_anchor_slot = np.zeros(e.n_pair, dtype=int)
    e.pair_ble_ci_slots = np.zeros(e.n_pair, dtype=int)
    e.pair_ble_ce_slots = np.zeros(e.n_pair, dtype=int)
    e.pair_ble_ce_required_s = np.zeros(e.n_pair, dtype=float)
    e.pair_ble_ce_feasible = np.ones(e.n_pair, dtype=bool)
    e.pair_ble_ce_channels = {} if e.ble_channel_mode == "per_ce" else None

    for row in pair_parameters:
        pair_id = int(row["pair_id"])
        radio = str(row["radio"])
        e.pair_office_id[pair_id] = int(row["office_id"])
        if hasattr(e, "_sample_pair_endpoint_in_office"):
            e.pair_tx_locs[pair_id] = e._sample_pair_endpoint_in_office(e.pair_office_id[pair_id])
            e.pair_rx_locs[pair_id] = e._sample_pair_endpoint_in_office(e.pair_office_id[pair_id])
        e.pair_radio_type[pair_id] = e.RADIO_BLE if radio == "ble" else e.RADIO_WIFI
        e.pair_channel[pair_id] = int(row["channel"])
        e.pair_priority[pair_id] = float(row["priority"])
        e.pair_release_time_slot[pair_id] = int(row["release_time_slot"])
        e.pair_deadline_slot[pair_id] = int(row["deadline_slot"])
        e.pair_start_time_slot[pair_id] = int(row["start_time_slot"])
        if radio == "wifi":
            e.pair_wifi_anchor_slot[pair_id] = int(row["wifi_anchor_slot"])
            e.pair_wifi_period_slots[pair_id] = int(row["wifi_period_slots"])
            e.pair_wifi_tx_slots[pair_id] = int(row["wifi_tx_slots"])
        else:
            ble_timing_mode = str(row.get("ble_timing_mode", "manual"))
            if ble_timing_mode == "auto":
                timing = sample_ble_pair_timing(
                    rand_gen=ble_timing_rng,
                    slot_time=e.slot_time,
                    ble_ci_quanta_candidates=e.ble_ci_quanta_candidates,
                    ble_ce_required_s=e.ble_ce_required_s,
                    ble_ce_max_s=e.ble_ce_max_s,
                    start_time_slot=int(row["start_time_slot"]),
                )
                e.pair_ble_anchor_slot[pair_id] = int(timing["anchor_slot"])
                e.pair_ble_ci_slots[pair_id] = int(timing["ci_slots"])
                e.pair_ble_ce_slots[pair_id] = int(timing["ce_slots"])
                e.pair_ble_ce_feasible[pair_id] = bool(timing["feasible"])
                e.pair_ble_ce_required_s[pair_id] = float(timing["ce_required_s"])
            else:
                e.pair_ble_anchor_slot[pair_id] = int(row["ble_anchor_slot"])
                e.pair_ble_ci_slots[pair_id] = int(row["ble_ci_slots"])
                e.pair_ble_ce_slots[pair_id] = int(row["ble_ce_slots"])
                e.pair_ble_ce_feasible[pair_id] = bool(row.get("ble_ce_feasible", True))
                e.pair_ble_ce_required_s[pair_id] = float(e.pair_ble_ce_slots[pair_id] * e.slot_time)
            if e.ble_channel_mode == "per_ce":
                ci_slots = max(1, int(e.pair_ble_ci_slots[pair_id]))
                macrocycle_slots = max(ci_slots, int(e.compute_macrocycle_slots()))
                event_count = max(1, macrocycle_slots // ci_slots)
                if "ble_ce_channels" in row:
                    ch = np.asarray(row["ble_ce_channels"], dtype=int)
                    if ch.size != event_count:
                        if ch.size == 0:
                            ch = np.zeros(event_count, dtype=int)
                        else:
                            ch = np.resize(ch, event_count)
                    e.pair_ble_ce_channels[pair_id] = ch
                else:
                    e._assign_ble_ce_channels(pair_id)

    if hasattr(e, "_resolve_pair_float_array"):
        e.pair_packet_bits = e._resolve_pair_float_array(
            pair_values=e._pair_packet_bits_input,
            user_values=e._user_packet_bits_input,
            wifi_default=e.wifi_packet_bit,
            ble_default=e.ble_packet_bit,
            attr_name="packet bits",
        )
        e.pair_bandwidth_hz = e._resolve_pair_float_array(
            pair_values=e._pair_bandwidth_hz_input,
            user_values=e._user_bandwidth_hz_input,
            wifi_default=e.wifi_channel_bandwidth_hz,
            ble_default=e.ble_channel_bandwidth_hz,
            attr_name="bandwidth",
        )

    e.device_priority = e.pair_priority
    e.user_priority = e.pair_priority
    e.device_radio_type = e.pair_radio_type
    e.user_radio_type = e.pair_radio_type
    e.device_radio_channel = e.pair_channel
    e.user_radio_channel = e.pair_channel
    e.device_start_time_slot = e.pair_start_time_slot
    e.user_start_time_slot = e.pair_start_time_slot
    e.device_release_time_slot = e.pair_release_time_slot
    e.user_release_time_slot = e.pair_release_time_slot
    e.device_deadline_slot = e.pair_deadline_slot
    e.user_deadline_slot = e.pair_deadline_slot
    e.device_wifi_anchor_slot = e.pair_wifi_anchor_slot
    e.user_wifi_anchor_slot = e.pair_wifi_anchor_slot
    e.device_wifi_period_slots = e.pair_wifi_period_slots
    e.user_wifi_period_slots = e.pair_wifi_period_slots
    e.device_wifi_tx_slots = e.pair_wifi_tx_slots
    e.user_wifi_tx_slots = e.pair_wifi_tx_slots
    e.device_ble_anchor_slot = e.pair_ble_anchor_slot
    e.user_ble_anchor_slot = e.pair_ble_anchor_slot
    e.device_ble_ci_slots = e.pair_ble_ci_slots
    e.user_ble_ci_slots = e.pair_ble_ci_slots
    e.device_ble_ce_slots = e.pair_ble_ce_slots
    e.user_ble_ce_slots = e.pair_ble_ce_slots
    e.device_ble_ce_required_s = e.pair_ble_ce_required_s
    e.user_ble_ce_required_s = e.pair_ble_ce_required_s
    e.device_ble_ce_feasible = e.pair_ble_ce_feasible
    e.user_ble_ce_feasible = e.pair_ble_ce_feasible
    if hasattr(e, "pair_tx_locs"):
        e.device_locs = e.pair_tx_locs
        e.sta_locs = e.device_locs
    if hasattr(e, "device_dirs"):
        e.sta_dirs = e.device_dirs
    if hasattr(e, "pair_packet_bits"):
        e.device_packet_bits = e.pair_packet_bits
        e.user_packet_bits = e.pair_packet_bits
    if hasattr(e, "pair_bandwidth_hz"):
        e.device_bandwidth_hz = e.pair_bandwidth_hz
        e.user_bandwidth_hz = e.pair_bandwidth_hz
    if hasattr(e, "_compute_min_sinr"):
        e._compute_min_sinr()


def _aggregate_office_stats_from_arrays(
    office_id: np.ndarray,
    radio: np.ndarray,
    z_vec: np.ndarray,
    n_office: int,
    wifi_id: int,
    ble_id: int,
):
    """Aggregate pair counts and slot usage per office."""
    rows = []
    for office in range(n_office):
        office_pairs = np.where(office_id == office)[0]
        wifi_pairs = office_pairs[radio[office_pairs] == wifi_id]
        ble_pairs = office_pairs[radio[office_pairs] == ble_id]
        rows.append(
            {
                "office_id": int(office),
                "wifi_pair_count": int(wifi_pairs.size),
                "ble_pair_count": int(ble_pairs.size),
                "wifi_slots_used": int(np.unique(z_vec[wifi_pairs]).size) if wifi_pairs.size else 0,
                "ble_slots_used": int(np.unique(z_vec[ble_pairs]).size) if ble_pairs.size else 0,
            }
        )
    return rows


def compute_office_pair_slot_stats(e: env, z_vec: np.ndarray):
    return _aggregate_office_stats_from_arrays(
        office_id=e.pair_office_id,
        radio=e.pair_radio_type,
        z_vec=z_vec,
        n_office=e.n_office,
        wifi_id=e.RADIO_WIFI,
        ble_id=e.RADIO_BLE,
    )


def compute_office_pair_slot_stats_for_pair_ids(e: env, z_vec: np.ndarray, pair_ids):
    pair_ids = np.asarray(pair_ids, dtype=int)
    if pair_ids.size == 0:
        return _aggregate_office_stats_from_arrays(
            office_id=np.array([], dtype=int),
            radio=np.array([], dtype=int),
            z_vec=np.array([], dtype=int),
            n_office=e.n_office,
            wifi_id=e.RADIO_WIFI,
            ble_id=e.RADIO_BLE,
        )
    return _aggregate_office_stats_from_arrays(
        office_id=e.pair_office_id[pair_ids],
        radio=e.pair_radio_type[pair_ids],
        z_vec=np.asarray(z_vec, dtype=int)[pair_ids],
        n_office=e.n_office,
        wifi_id=e.RADIO_WIFI,
        ble_id=e.RADIO_BLE,
    )


def _sorted_candidate_starts(period_slots: int, preferred_slot: int):
    period_slots = int(period_slots)
    preferred_slot = int(preferred_slot % period_slots)
    candidates = list(range(period_slots))
    candidates.sort(key=lambda s: (min((s - preferred_slot) % period_slots, (preferred_slot - s) % period_slots), s))
    return candidates


def _occupancy_within_time_window(occupancy: np.ndarray, release_time_slot: int, deadline_slot: int):
    occupied_slots = np.where(np.asarray(occupancy, dtype=bool))[0]
    if occupied_slots.size == 0:
        return True
    return bool(
        np.all(occupied_slots >= int(release_time_slot))
        and np.all(occupied_slots <= int(deadline_slot))
    )


def assign_macrocycle_start_slots(
    e: env,
    preferred_slots: np.ndarray,
    allow_partial: bool = False,
    pair_order: np.ndarray | None = None,
    wifi_first: bool = False,
    return_ble_stats: bool = False,
):
    preferred_slots = np.asarray(preferred_slots, dtype=int).ravel()
    macrocycle_slots = e.compute_macrocycle_slots()
    if macrocycle_slots <= 0:
        result = (np.zeros(e.n_pair, dtype=int), 0, np.zeros((e.n_pair, 0), dtype=bool), [])
        if return_ble_stats:
            return result + ({},)
        return result

    assigned_starts = np.full(e.n_pair, -1, dtype=int)
    occupancies = np.zeros((e.n_pair, macrocycle_slots), dtype=bool)
    conflict = e.build_pair_conflict_matrix()
    slot_asn = [[] for _ in range(macrocycle_slots)]
    s_gain, _, h_max = e.get_macrocycle_conflict_state()
    s_gain = s_gain.copy().tocsr()
    s_gain.setdiag(0.0)
    s_gain.eliminate_zeros()
    slot_gain_sum = [np.zeros(e.n_pair, dtype=float) for _ in range(macrocycle_slots)]
    unscheduled = []

    period_slots = e.get_pair_period_slots()
    width_slots = e.get_pair_width_slots()
    if pair_order is None:
        order = np.lexsort((preferred_slots, -e.pair_priority, -width_slots))
        if wifi_first:
            wifi_pairs = order[e.pair_radio_type[order] == e.RADIO_WIFI]
            ble_pairs = order[e.pair_radio_type[order] == e.RADIO_BLE]
            order = np.concatenate((wifi_pairs, ble_pairs))
    else:
        order = np.asarray(pair_order, dtype=int).ravel()
    for pair_id in order:
        pair_id = int(pair_id)
        if e.pair_radio_type[pair_id] == e.RADIO_BLE and not e.pair_ble_ce_feasible[pair_id]:
            if allow_partial:
                unscheduled.append(pair_id)
                continue
            raise ValueError(f"BLE pair {pair_id} is infeasible under CE constraints.")

        period = int(period_slots[pair_id])
        if period <= 0:
            if allow_partial:
                unscheduled.append(pair_id)
                continue
            raise ValueError(f"Pair {pair_id} has invalid period {period}.")

        assigned = False
        for start_slot in _sorted_candidate_starts(period, preferred_slots[pair_id]):
            occ = e.expand_pair_occupancy(pair_id, start_slot, macrocycle_slots)
            occ_slots = np.where(occ)[0]
            if not _occupancy_within_time_window(
                occ,
                e.pair_release_time_slot[pair_id],
                e.pair_deadline_slot[pair_id],
            ):
                continue
            tmp_h = np.asarray(s_gain[pair_id].toarray()).ravel()
            violates = False
            if wifi_first and e.pair_radio_type[pair_id] == e.RADIO_BLE:
                assigned_wifi = np.where(np.logical_and(assigned_starts >= 0, e.pair_radio_type == e.RADIO_WIFI))[0]
                slot_capacity = e.get_ble_start_slot_capacity(
                    wifi_pair_ids=assigned_wifi,
                    wifi_start_slots=assigned_starts[assigned_wifi] if assigned_wifi.size else np.array([], dtype=int),
                    start_slot=int(start_slot),
                )
                scheduled_ble_at_start = np.where(
                    np.logical_and(
                        assigned_starts == int(start_slot),
                        e.pair_radio_type == e.RADIO_BLE,
                    )
                )[0]
                if scheduled_ble_at_start.size >= int(slot_capacity):
                    continue
            for slot in occ_slots:
                for other_pair in slot_asn[int(slot)]:
                    slot_conflict = bool(conflict[pair_id, other_pair])
                    if (
                        slot_conflict
                        and (
                            e.ble_channel_mode != "per_ce"
                            or e.pair_radio_type[pair_id] == e.RADIO_WIFI
                            or e.pair_radio_type[other_pair] == e.RADIO_WIFI
                        )
                    ):
                        violates = True
                        break
                    if slot_conflict and e.ble_channel_mode == "per_ce":
                        if e.is_slot_channel_conflict(
                            pair_id,
                            start_slot,
                            other_pair,
                            assigned_starts[other_pair],
                            int(slot),
                        ):
                            violates = True
                            break
                    elif e.ble_channel_mode == "per_ce" and e.is_slot_channel_conflict(
                        pair_id,
                        start_slot,
                        other_pair,
                        assigned_starts[other_pair],
                        int(slot),
                    ):
                        violates = True
                        break
                if violates:
                    break
                neighbor_index = np.append(np.array(slot_asn[int(slot)], dtype=int), pair_id).astype(int)
                vio_h = (slot_gain_sum[int(slot)][neighbor_index] + tmp_h[neighbor_index]) > h_max[neighbor_index]
                if np.any(vio_h):
                    violates = True
                    break
                if violates:
                    break
            if violates:
                continue
            assigned_starts[pair_id] = int(start_slot)
            occupancies[pair_id] = occ
            for slot in occ_slots:
                slot_asn[int(slot)].append(pair_id)
                slot_gain_sum[int(slot)] += tmp_h
            assigned = True
            break

        if not assigned:
            unscheduled.append(pair_id)
            if allow_partial:
                continue
            raise ValueError(f"Unable to assign non-overlapping macrocycle start slot for pair {pair_id}.")

    result = (assigned_starts, macrocycle_slots, occupancies, unscheduled)
    if not return_ble_stats:
        return result
    return result + (_build_wifi_first_ble_stats(e, assigned_starts) if wifi_first else {},)


def _build_wifi_first_ble_stats(e: env, assigned_starts: np.ndarray):
    assigned_starts = np.asarray(assigned_starts, dtype=int).ravel()
    ble_stats = {}
    assigned_wifi = np.where(np.logical_and(assigned_starts >= 0, e.pair_radio_type == e.RADIO_WIFI))[0]
    assigned_ble = np.where(np.logical_and(assigned_starts >= 0, e.pair_radio_type == e.RADIO_BLE))[0]
    for start_slot in np.unique(assigned_starts[assigned_ble]) if assigned_ble.size else np.array([], dtype=int):
        slot = int(start_slot)
        scheduled_ble = np.where(
            np.logical_and(
                assigned_starts == slot,
                e.pair_radio_type == e.RADIO_BLE,
            )
        )[0]
        capacity = int(
            e.get_ble_start_slot_capacity(
                wifi_pair_ids=assigned_wifi,
                wifi_start_slots=assigned_starts[assigned_wifi] if assigned_wifi.size else np.array([], dtype=int),
                start_slot=slot,
            )
        )
        ble_stats[slot] = {
            "effective_ble_channels": capacity,
            "scheduled_ble_pairs": int(scheduled_ble.size),
            "no_collision_probability": float(
                e.compute_ble_no_collision_probability(capacity, int(scheduled_ble.size))
            ),
        }
    return ble_stats


def _repair_macrocycle_assignment_by_reordering(
    e: env,
    preferred_slots: np.ndarray,
    best_result,
    max_repair_passes: int = 2,
):
    best_starts, best_macrocycle_slots, best_occupancies, best_unscheduled = best_result
    best_unscheduled = list(best_unscheduled)
    if not best_unscheduled:
        return best_starts, best_macrocycle_slots, best_occupancies, best_unscheduled

    for _ in range(int(max_repair_passes)):
        if not best_unscheduled:
            break

        period_slots = e.get_pair_period_slots()
        unscheduled_set = set(best_unscheduled)
        scheduled = [pair_id for pair_id in range(e.n_pair) if pair_id not in unscheduled_set]
        candidate_orders = [
            np.array(best_unscheduled + scheduled, dtype=int),
            np.array(
                best_unscheduled
                + sorted(
                    scheduled,
                    key=lambda pair_id: (
                        -float(e.pair_priority[pair_id]),
                        -int(e.get_pair_width_slots()[pair_id]),
                        int(preferred_slots[pair_id]),
                    ),
                ),
                dtype=int,
            ),
            np.array(
                best_unscheduled
                + sorted(
                    scheduled,
                    key=lambda pair_id: (
                        -int(e.get_pair_width_slots()[pair_id]),
                        -float(e.pair_priority[pair_id]),
                        int(preferred_slots[pair_id]),
                    ),
                ),
                dtype=int,
            ),
        ]

        improved = False
        for candidate_order in candidate_orders:
            staggered_preferred = np.asarray(preferred_slots, dtype=int).copy()
            for rank, pair_id in enumerate(candidate_order):
                period = int(period_slots[int(pair_id)])
                if period > 0:
                    staggered_preferred[int(pair_id)] = rank % period
            for candidate_preferred in (np.asarray(preferred_slots, dtype=int), staggered_preferred):
                candidate = assign_macrocycle_start_slots(
                    e,
                    candidate_preferred,
                    allow_partial=True,
                    pair_order=candidate_order,
                )
                candidate_unscheduled = list(candidate[3])
                if len(candidate_unscheduled) < len(best_unscheduled):
                    best_starts = candidate[0].copy()
                    best_macrocycle_slots = int(candidate[1])
                    best_occupancies = candidate[2].copy()
                    best_unscheduled = candidate_unscheduled
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_starts, best_macrocycle_slots, best_occupancies, best_unscheduled


def _is_better_refill_result(
    candidate,
    best,
    pair_priority: np.ndarray,
    pair_radio_type: np.ndarray | None = None,
    wifi_radio_id: int = 0,
    wifi_first: bool = False,
):
    candidate_unscheduled = list(candidate[3])
    best_unscheduled = list(best[3])
    if wifi_first:
        if pair_radio_type is None:
            raise ValueError("pair_radio_type must be provided when wifi_first=True.")
        pair_radio_type = np.asarray(pair_radio_type, dtype=int)
        candidate_wifi_unscheduled = [
            pair_id for pair_id in candidate_unscheduled if int(pair_radio_type[pair_id]) == int(wifi_radio_id)
        ]
        best_wifi_unscheduled = [
            pair_id for pair_id in best_unscheduled if int(pair_radio_type[pair_id]) == int(wifi_radio_id)
        ]
        if len(candidate_wifi_unscheduled) != len(best_wifi_unscheduled):
            return len(candidate_wifi_unscheduled) < len(best_wifi_unscheduled)
        candidate_wifi_priority = (
            float(np.sum(np.asarray(pair_priority)[candidate_wifi_unscheduled])) if candidate_wifi_unscheduled else 0.0
        )
        best_wifi_priority = float(np.sum(np.asarray(pair_priority)[best_wifi_unscheduled])) if best_wifi_unscheduled else 0.0
        if candidate_wifi_priority != best_wifi_priority:
            return candidate_wifi_priority < best_wifi_priority
    if len(candidate_unscheduled) != len(best_unscheduled):
        return len(candidate_unscheduled) < len(best_unscheduled)
    candidate_priority = float(np.sum(np.asarray(pair_priority)[candidate_unscheduled])) if candidate_unscheduled else 0.0
    best_priority = float(np.sum(np.asarray(pair_priority)[best_unscheduled])) if best_unscheduled else 0.0
    return candidate_priority < best_priority


def _refill_unscheduled_pairs_by_radio(
    e: env,
    preferred_slots: np.ndarray,
    best_result,
    target_radio_type: int,
    max_refill_passes: int = 2,
    wifi_first: bool = False,
):
    best = (
        best_result[0].copy(),
        int(best_result[1]),
        best_result[2].copy(),
        list(best_result[3]),
    )
    for _ in range(int(max_refill_passes)):
        best_unscheduled = list(best[3])
        target_unscheduled = [
            pair_id for pair_id in best_unscheduled if int(e.pair_radio_type[pair_id]) == int(target_radio_type)
        ]
        if not target_unscheduled:
            break

        other_unscheduled = [pair_id for pair_id in best_unscheduled if pair_id not in set(target_unscheduled)]
        scheduled = [pair_id for pair_id in range(e.n_pair) if pair_id not in set(best_unscheduled)]
        period_slots = e.get_pair_period_slots()
        candidate_orders = [
            np.array(target_unscheduled + other_unscheduled + scheduled, dtype=int),
            np.array(
                target_unscheduled
                + sorted(
                    other_unscheduled + scheduled,
                    key=lambda pair_id: (
                        -float(e.pair_priority[pair_id]),
                        -int(e.get_pair_width_slots()[pair_id]),
                        int(preferred_slots[pair_id]),
                    ),
                ),
                dtype=int,
            ),
            np.array(
                target_unscheduled
                + sorted(
                    other_unscheduled + scheduled,
                    key=lambda pair_id: (
                        -int(e.get_pair_width_slots()[pair_id]),
                        -float(e.pair_priority[pair_id]),
                        int(preferred_slots[pair_id]),
                    ),
                ),
                dtype=int,
            ),
        ]

        improved = False
        for candidate_order in candidate_orders:
            staggered_preferred = np.asarray(preferred_slots, dtype=int).copy()
            for rank, pair_id in enumerate(candidate_order):
                period = int(period_slots[int(pair_id)])
                if period > 0:
                    staggered_preferred[int(pair_id)] = rank % period
            for candidate_preferred in (np.asarray(preferred_slots, dtype=int), staggered_preferred):
                candidate = assign_macrocycle_start_slots(
                    e,
                    candidate_preferred,
                    allow_partial=True,
                    pair_order=candidate_order,
                )
                if _is_better_refill_result(
                    candidate,
                    best,
                    e.pair_priority,
                    pair_radio_type=e.pair_radio_type,
                    wifi_radio_id=e.RADIO_WIFI,
                    wifi_first=wifi_first,
                ):
                    best = (
                        candidate[0].copy(),
                        int(candidate[1]),
                        candidate[2].copy(),
                        list(candidate[3]),
                    )
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _apply_refill_pipeline(e: env, preferred_slots: np.ndarray, initial_result, wifi_first: bool = False):
    best = _repair_macrocycle_assignment_by_reordering(
        e,
        preferred_slots,
        (
            initial_result[0].copy(),
            int(initial_result[1]),
            initial_result[2].copy(),
            list(initial_result[3]),
        ),
    )
    best = _refill_unscheduled_pairs_by_radio(
        e,
        preferred_slots,
        best,
        target_radio_type=e.RADIO_WIFI,
        wifi_first=wifi_first,
    )
    best = _refill_unscheduled_pairs_by_radio(
        e,
        preferred_slots,
        best,
        target_radio_type=e.RADIO_BLE,
        wifi_first=wifi_first,
    )
    return best


def retry_ble_channels_and_assign_macrocycle(
    e: env,
    preferred_slots: np.ndarray,
    max_ble_channel_retries: int = 0,
    wifi_first: bool = False,
    return_ble_stats: bool = False,
):
    starts, macrocycle_slots, occupancies, unscheduled = assign_macrocycle_start_slots(
        e,
        preferred_slots,
        allow_partial=True,
        wifi_first=wifi_first,
    )
    best = _apply_refill_pipeline(
        e,
        preferred_slots,
        (starts.copy(), int(macrocycle_slots), occupancies.copy(), list(unscheduled)),
        wifi_first=wifi_first,
    )
    best_unscheduled = list(best[3])
    retries_used = 0

    for _ in range(int(max_ble_channel_retries)):
        ble_unscheduled = np.array(
            [pair_id for pair_id in best_unscheduled if e.pair_radio_type[pair_id] == e.RADIO_BLE],
            dtype=int,
        )
        if ble_unscheduled.size == 0:
            break

        e.resample_ble_channels(ble_unscheduled)
        retries_used += 1
        candidate = assign_macrocycle_start_slots(
            e,
            preferred_slots,
            allow_partial=True,
            wifi_first=wifi_first,
        )
        candidate = _apply_refill_pipeline(
            e,
            preferred_slots,
            (candidate[0].copy(), int(candidate[1]), candidate[2].copy(), list(candidate[3])),
        )
        candidate_unscheduled = list(candidate[3])
        if _is_better_refill_result(
            candidate,
            best,
            e.pair_priority,
            pair_radio_type=e.pair_radio_type,
            wifi_radio_id=e.RADIO_WIFI,
            wifi_first=wifi_first,
        ):
            best = (
                candidate[0].copy(),
                int(candidate[1]),
                candidate[2].copy(),
                candidate_unscheduled,
            )
            best_unscheduled = candidate_unscheduled
        if not best_unscheduled:
            break

    if return_ble_stats:
        return (*best, retries_used, _build_wifi_first_ble_stats(e, best[0]) if wifi_first else {})
    return (*best, retries_used)


def print_office_stats(rows):
    print("=== Per-Office WiFi/BLE Pair & Slot Statistics ===")
    print("office_id,wifi_pair_count,ble_pair_count,wifi_slots_used,ble_slots_used")
    for r in rows:
        print(
            f"{r['office_id']},{r['wifi_pair_count']},{r['ble_pair_count']},"
            f"{r['wifi_slots_used']},{r['ble_slots_used']}"
        )


def build_pair_parameter_rows(
    pair_office_id,
    pair_radio_type,
    pair_channel,
    pair_priority,
    pair_release_time_slot,
    pair_deadline_slot,
    ble_channel_mode,
    pair_ble_ce_channels,
    pair_start_time_slot,
    pair_wifi_anchor_slot,
    pair_wifi_period_slots,
    pair_wifi_tx_slots,
    pair_ble_anchor_slot,
    pair_ble_ci_slots,
    pair_ble_ce_slots,
    pair_ble_ce_feasible,
    z_vec,
    occupied_slots,
    macrocycle_slots,
    slot_time,
    wifi_id,
    ble_id,
    ble_slot_stats=None,
):
    rows = []
    ble_slot_stats = {} if ble_slot_stats is None else ble_slot_stats
    for pair_id in range(pair_radio_type.shape[0]):
        is_ble = int(pair_radio_type[pair_id]) == int(ble_id)
        schedule_slot = int(z_vec[pair_id])
        slot_stats = ble_slot_stats.get(schedule_slot, {}) if is_ble and schedule_slot >= 0 else {}
        ce_channel_summary = None
        if is_ble and ble_channel_mode == "per_ce":
            ce_channel_summary = [int(ch) for ch in pair_ble_ce_channels.get(int(pair_id), np.zeros(0, dtype=int))]
        rows.append(
            {
                "pair_id": int(pair_id),
                "office_id": int(pair_office_id[pair_id]),
                "radio": "ble" if is_ble else "wifi",
                "channel": int(pair_channel[pair_id]),
                "priority": float(pair_priority[pair_id]),
                "release_time_slot": int(pair_release_time_slot[pair_id]),
                "deadline_slot": int(pair_deadline_slot[pair_id]),
                "schedule_slot": schedule_slot,
                "schedule_time_ms": float(schedule_slot * slot_time * 1e3),
                "ble_channel_mode": str(ble_channel_mode),
                "ble_ce_channel_summary": ce_channel_summary,
                "start_time_slot": int(pair_start_time_slot[pair_id]),
                "wifi_anchor_slot": int(pair_wifi_anchor_slot[pair_id]) if not is_ble else None,
                "wifi_period_slots": int(pair_wifi_period_slots[pair_id]) if not is_ble else None,
                "wifi_period_ms": float(pair_wifi_period_slots[pair_id] * slot_time * 1e3) if not is_ble else None,
                "wifi_tx_slots": int(pair_wifi_tx_slots[pair_id]) if not is_ble else None,
                "wifi_tx_ms": float(pair_wifi_tx_slots[pair_id] * slot_time * 1e3) if not is_ble else None,
                "ble_anchor_slot": int(pair_ble_anchor_slot[pair_id]) if is_ble else None,
                "ble_ci_slots": int(pair_ble_ci_slots[pair_id]) if is_ble else None,
                "ble_ci_ms": float(pair_ble_ci_slots[pair_id] * slot_time * 1e3) if is_ble else None,
                "ble_ce_slots": int(pair_ble_ce_slots[pair_id]) if is_ble else None,
                "ble_ce_ms": float(pair_ble_ce_slots[pair_id] * slot_time * 1e3) if is_ble else None,
                "ble_ce_feasible": bool(pair_ble_ce_feasible[pair_id]) if is_ble else None,
                "effective_ble_channels": int(slot_stats["effective_ble_channels"]) if slot_stats else None,
                "scheduled_ble_pairs": int(slot_stats["scheduled_ble_pairs"]) if slot_stats else None,
                "no_collision_probability": float(slot_stats["no_collision_probability"]) if slot_stats else None,
                "macrocycle_slots": int(macrocycle_slots),
                "occupied_slots_in_macrocycle": [int(s) for s in np.where(occupied_slots[pair_id])[0]],
            }
        )
    return rows


def compute_pair_parameter_rows(
    e: env,
    z_vec: np.ndarray,
    occupied_slots: np.ndarray,
    macrocycle_slots: int,
    ble_slot_stats=None,
):
    return build_pair_parameter_rows(
        pair_office_id=e.pair_office_id,
        pair_radio_type=e.pair_radio_type,
        pair_channel=e.pair_channel,
        pair_priority=e.pair_priority,
        pair_release_time_slot=e.pair_release_time_slot,
        pair_deadline_slot=e.pair_deadline_slot,
        ble_channel_mode=e.ble_channel_mode,
        pair_ble_ce_channels=e.pair_ble_ce_channels if e.pair_ble_ce_channels is not None else {},
        pair_start_time_slot=e.pair_start_time_slot,
        pair_wifi_anchor_slot=e.pair_wifi_anchor_slot,
        pair_wifi_period_slots=e.pair_wifi_period_slots,
        pair_wifi_tx_slots=e.pair_wifi_tx_slots,
        pair_ble_anchor_slot=e.pair_ble_anchor_slot,
        pair_ble_ci_slots=e.pair_ble_ci_slots,
        pair_ble_ce_slots=e.pair_ble_ce_slots,
        pair_ble_ce_feasible=e.pair_ble_ce_feasible,
        z_vec=z_vec,
        occupied_slots=occupied_slots,
        macrocycle_slots=macrocycle_slots,
        slot_time=e.slot_time,
        wifi_id=e.RADIO_WIFI,
        ble_id=e.RADIO_BLE,
        ble_slot_stats=ble_slot_stats,
    )


def resolve_macrocycle_schedule_status(schedule_start_slots: np.ndarray, occupied_slots: np.ndarray):
    schedule_start_slots = np.asarray(schedule_start_slots, dtype=int).ravel()
    occupied_slots = np.asarray(occupied_slots, dtype=bool)
    if occupied_slots.ndim != 2:
        raise ValueError("occupied_slots must be a 2D boolean array.")
    if occupied_slots.shape[0] != schedule_start_slots.shape[0]:
        raise ValueError("schedule_start_slots and occupied_slots must have the same pair dimension.")

    has_occupancy = np.any(occupied_slots, axis=1)
    is_scheduled = np.logical_and(schedule_start_slots >= 0, has_occupancy)
    scheduled_pair_ids = np.where(is_scheduled)[0].astype(int).tolist()
    unscheduled_pair_ids = np.where(~is_scheduled)[0].astype(int).tolist()
    return scheduled_pair_ids, unscheduled_pair_ids


def filter_pair_rows_by_ids(pair_rows, pair_ids):
    selected = set(int(pair_id) for pair_id in pair_ids)
    return [row for row in pair_rows if int(row["pair_id"]) in selected]


def build_schedule_rows(pair_rows):
    slot_map = {}
    for row in pair_rows:
        for slot in row["occupied_slots_in_macrocycle"]:
            slot_map.setdefault(int(slot), []).append(row)

    rows = []
    for slot in sorted(slot_map):
        grouped = sorted(slot_map[slot], key=lambda r: int(r["pair_id"]))
        wifi_ids = [int(r["pair_id"]) for r in grouped if r["radio"] == "wifi"]
        ble_ids = [int(r["pair_id"]) for r in grouped if r["radio"] == "ble"]
        rows.append(
            {
                "schedule_slot": slot,
                "pair_ids": [int(r["pair_id"]) for r in grouped],
                "wifi_pair_ids": wifi_ids,
                "ble_pair_ids": ble_ids,
                "pair_count": len(grouped),
                "wifi_pair_count": len(wifi_ids),
                "ble_pair_count": len(ble_ids),
            }
        )
    return rows


def get_pair_channel_ranges_mhz(e: env, pair_ids):
    pair_ids = np.asarray(pair_ids, dtype=int).ravel()
    ranges = {}
    for pair_id in pair_ids:
        low_hz, high_hz = e._get_pair_link_range_hz(int(pair_id))
        ranges[int(pair_id)] = (float(low_hz / 1e6), float(high_hz / 1e6))
    return ranges


def build_schedule_plot_rows(pair_rows, pair_channel_ranges, e: env | None = None):
    rows = []
    for row in pair_rows:
        pair_id = int(row["pair_id"])
        if e is not None and row["radio"] == "ble" and getattr(e, "ble_channel_mode", "single") == "per_ce":
            instances = e.expand_pair_event_instances(
                pair_id,
                int(row["macrocycle_slots"]),
                start_slot=int(row["schedule_slot"]),
            )
            for inst in instances:
                low_mhz = float(inst["freq_range_hz"][0] / 1e6)
                high_mhz = float(inst["freq_range_hz"][1] / 1e6)
                label = f"{pair_id} B-ch{int(inst['channel'])} ev{int(inst['event_index'])}"
                wrapped_ranges = inst.get("wrapped_slot_ranges", [inst["slot_range"]])
                for seg_start, seg_end in wrapped_ranges:
                    for slot in range(int(seg_start), int(seg_end)):
                        rows.append(
                            {
                                "pair_id": pair_id,
                                "radio": row["radio"],
                                "channel": int(inst["channel"]),
                                "slot": int(slot),
                                "freq_low_mhz": low_mhz,
                                "freq_high_mhz": high_mhz,
                                "label": label,
                            }
                        )
            continue

        low_mhz, high_mhz = pair_channel_ranges[pair_id]
        radio_tag = "W" if row["radio"] == "wifi" else "B"
        label = f"{pair_id} {radio_tag}-ch{int(row['channel'])}"
        for slot in row["occupied_slots_in_macrocycle"]:
            rows.append(
                {
                    "pair_id": pair_id,
                    "radio": row["radio"],
                    "channel": int(row["channel"]),
                    "slot": int(slot),
                    "freq_low_mhz": float(low_mhz),
                    "freq_high_mhz": float(high_mhz),
                    "label": label,
                }
            )
    rows.extend(build_ble_overlap_plot_rows(rows))
    if e is not None:
        rows.extend(build_ble_advertising_plot_rows(int(pair_rows[0]["macrocycle_slots"]) if pair_rows else 0, e))
    return rows


def build_ble_advertising_plot_rows(macrocycle_slots, e: env):
    macrocycle_slots = int(macrocycle_slots)
    if macrocycle_slots <= 0:
        return []
    rows = []
    for center_mhz in getattr(e, "ble_advertising_center_freq_mhz", []):
        rows.append(
            {
                "pair_id": -2,
                "radio": "ble_adv_idle",
                "channel": -1,
                "slot": 0,
                "slot_width": macrocycle_slots,
                "freq_low_mhz": float(center_mhz - 1.0),
                "freq_high_mhz": float(center_mhz + 1.0),
                "label": f"BLE adv idle {center_mhz:.0f} MHz",
            }
        )
    return rows


def build_ble_overlap_plot_rows(plot_rows):
    overlap_rows = []
    rows_by_slot = {}
    for row in plot_rows:
        rows_by_slot.setdefault(int(row["slot"]), []).append(row)
    for slot, rows in rows_by_slot.items():
        ble_rows = [row for row in rows if row["radio"] == "ble"]
        for idx, left in enumerate(ble_rows):
            for right in ble_rows[idx + 1 :]:
                lo = max(float(left["freq_low_mhz"]), float(right["freq_low_mhz"]))
                hi = min(float(left["freq_high_mhz"]), float(right["freq_high_mhz"]))
                if lo >= hi:
                    continue
                overlap_rows.append(
                    {
                        "pair_id": -1,
                        "radio": "ble_overlap",
                        "channel": -1,
                        "slot": int(slot),
                        "freq_low_mhz": float(lo),
                        "freq_high_mhz": float(hi),
                        "label": f"BLE overlap {min(int(left['pair_id']), int(right['pair_id']))}/{max(int(left['pair_id']), int(right['pair_id']))}",
                    }
                )
    return overlap_rows


def build_ble_ce_event_rows(e: env, pair_rows):
    rows = []
    if getattr(e, "ble_channel_mode", "single") != "per_ce":
        return rows
    for row in pair_rows:
        if row["radio"] != "ble" or int(row["schedule_slot"]) < 0:
            continue
        instances = e.expand_pair_event_instances(
            int(row["pair_id"]),
            int(row["macrocycle_slots"]),
            start_slot=int(row["schedule_slot"]),
        )
        for inst in instances:
            wrapped_ranges = inst.get("wrapped_slot_ranges", [inst["slot_range"]])
            for seg_start, seg_end in wrapped_ranges:
                rows.append(
                    {
                        "pair_id": int(row["pair_id"]),
                        "event_index": int(inst["event_index"]),
                        "channel": int(inst["channel"]),
                        "slot_start": int(seg_start),
                        "slot_end": int(seg_end),
                        "freq_low_mhz": float(inst["freq_range_hz"][0] / 1e6),
                        "freq_high_mhz": float(inst["freq_range_hz"][1] / 1e6),
                    }
                )
    return rows


def render_schedule_plot(plot_rows, output_path, macrocycle_slots):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"wifi": "#0B6E4F", "ble": "#C84C09", "ble_adv_idle": "#D9D9D9", "ble_overlap": "#B80F2A"}
    for row in plot_rows:
        height = row["freq_high_mhz"] - row["freq_low_mhz"]
        rect = Rectangle(
            (row["slot"], row["freq_low_mhz"]),
            float(row.get("slot_width", 1.0)),
            height,
            facecolor=colors.get(row["radio"], "#4C4C4C"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.35 if row["radio"] == "ble_adv_idle" else 0.65,
        )
        ax.add_patch(rect)
        if height >= 1.5:
            ax.text(
                row["slot"] + float(row.get("slot_width", 1.0)) / 2.0,
                row["freq_low_mhz"] + height / 2.0,
                row["label"],
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                clip_on=True,
            )
    ax.set_xlim(0, int(macrocycle_slots))
    if plot_rows:
        ax.set_ylim(min(r["freq_low_mhz"] for r in plot_rows), max(r["freq_high_mhz"] for r in plot_rows))
    ax.set_xlabel("Slot")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("WiFi/BLE Schedule")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(
        handles=[
            Patch(facecolor=colors["wifi"], edgecolor="black", alpha=0.65, label="WiFi"),
            Patch(facecolor=colors["ble"], edgecolor="black", alpha=0.65, label="BLE"),
            Patch(facecolor=colors["ble_adv_idle"], edgecolor="black", alpha=0.35, label="BLE adv idle"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt_cell(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def print_pair_parameter_rows(rows):
    print("=== Pair Parameter Table ===")
    print(
        "pair_id,office_id,radio,channel,priority,release_time_slot,deadline_slot,schedule_slot,schedule_time_ms,"
        "ble_channel_mode,ble_ce_channel_summary,start_time_slot,"
        "wifi_anchor_slot,wifi_period_slots,wifi_period_ms,wifi_tx_slots,wifi_tx_ms,"
        "ble_anchor_slot,ble_ci_slots,ble_ci_ms,ble_ce_slots,ble_ce_ms,ble_ce_feasible,"
        "macrocycle_slots,occupied_slots_in_macrocycle"
    )
    for row in rows:
        print(
            ",".join(
                [
                    _fmt_cell(row["pair_id"]),
                    _fmt_cell(row["office_id"]),
                    _fmt_cell(row["radio"]),
                    _fmt_cell(row["channel"]),
                    _fmt_cell(row["priority"]),
                    _fmt_cell(row["release_time_slot"]),
                    _fmt_cell(row["deadline_slot"]),
                    _fmt_cell(row["schedule_slot"]),
                    _fmt_cell(row["schedule_time_ms"]),
                    _fmt_cell(row["ble_channel_mode"]),
                    _fmt_cell(row["ble_ce_channel_summary"]),
                    _fmt_cell(row["start_time_slot"]),
                    _fmt_cell(row["wifi_anchor_slot"]),
                    _fmt_cell(row["wifi_period_slots"]),
                    _fmt_cell(row["wifi_period_ms"]),
                    _fmt_cell(row["wifi_tx_slots"]),
                    _fmt_cell(row["wifi_tx_ms"]),
                    _fmt_cell(row["ble_anchor_slot"]),
                    _fmt_cell(row["ble_ci_slots"]),
                    _fmt_cell(row["ble_ci_ms"]),
                    _fmt_cell(row["ble_ce_slots"]),
                    _fmt_cell(row["ble_ce_ms"]),
                    _fmt_cell(row["ble_ce_feasible"]),
                    _fmt_cell(row["macrocycle_slots"]),
                    _fmt_cell(row["occupied_slots_in_macrocycle"]),
                ]
            )
        )


def print_schedule_rows(rows):
    print("=== Schedule Table ===")
    print("schedule_slot,pair_ids,wifi_pair_ids,ble_pair_ids,pair_count,wifi_pair_count,ble_pair_count")
    for row in rows:
        print(
            ",".join(
                [
                    _fmt_cell(row["schedule_slot"]),
                    _fmt_cell(row["pair_ids"]),
                    _fmt_cell(row["wifi_pair_ids"]),
                    _fmt_cell(row["ble_pair_ids"]),
                    _fmt_cell(row["pair_count"]),
                    _fmt_cell(row["wifi_pair_count"]),
                    _fmt_cell(row["ble_pair_count"]),
                ]
            )
        )


def write_rows_to_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _fmt_cell(row.get(key))
                    for key in fieldnames
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="pd_mmw_template + office/pair statistics")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件路径；支持带 _comment_* 中文注释字段。",
    )
    parser.add_argument("--cell-size", type=int, default=None, help="办公室网格边长（办公室数量 = cell_size^2）")
    parser.add_argument(
        "--pair-density",
        "--sta-density",
        dest="pair_density",
        type=float,
        default=None,
        help="通信对密度（每平方米）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子；不填则使用当前时间")
    parser.add_argument("--mmw-nit", type=int, default=None, help="MMW 迭代次数")
    parser.add_argument("--mmw-eta", type=float, default=None, help="MMW 步长 eta")
    parser.add_argument("--use-gpu", action="store_true", default=None, help="启用 GPU 设备选择（若 CUDA 可用）")
    parser.add_argument("--gpu-id", type=int, default=None, help="指定使用的 GPU 编号")
    parser.add_argument("--max-slots", type=int, default=None, help="最大允许调度时隙数；超过后输出当前成功调度结果")
    parser.add_argument("--ble-channel-retries", type=int, default=None, help="对未调度 BLE pair 重新选信道并重试宏周期排布的次数")
    parser.add_argument(
        "--ble-channel-mode",
        choices=["single", "per_ce"],
        default=None,
        help="BLE 信道模式：single 为每 pair 单信道，per_ce 为每个 CE 独立信道。",
    )
    parser.add_argument(
        "--ble-schedule-backend",
        choices=["legacy", "macrocycle_hopping_sdp", "macrocycle_hopping_ga"],
        default=None,
        help="BLE 调度后端：legacy 保持现有方案，macrocycle_hopping_sdp 接入 BLE-only 宏周期跳频 SDP，macrocycle_hopping_ga 接入 BLE-only 宏周期跳频 GA。",
    )
    parser.add_argument("--ble-ga-population-size", type=int, default=None, help="BLE GA 种群大小。")
    parser.add_argument("--ble-ga-generations", type=int, default=None, help="BLE GA 迭代代数。")
    parser.add_argument("--ble-ga-mutation-rate", type=float, default=None, help="BLE GA 变异概率。")
    parser.add_argument("--ble-ga-crossover-rate", type=float, default=None, help="BLE GA 交叉概率。")
    parser.add_argument("--ble-ga-elite-count", type=int, default=None, help="BLE GA 精英保留个数。")
    parser.add_argument("--ble-ga-seed", type=int, default=None, help="BLE GA 随机种子。")
    parser.add_argument(
        "--ble-max-offsets-per-pair",
        type=int,
        default=None,
        help="BLE 候选状态剪枝上限；每个 pair 最多保留多少个可行 offset。",
    )
    parser.add_argument(
        "--ble-log-candidate-summary",
        action="store_true",
        default=None,
        help="在 BLE backend 求解前打印候选空间摘要（state_count / offset_count / pattern_count）。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="CSV 输出目录（会写出 pair_parameters.csv 和 wifi_ble_schedule.csv）",
    )
    parser.add_argument(
        "--wifi-first-ble-scheduling",
        action="store_true",
        default=None,
        help="先调度 WiFi，再按剩余 BLE 可用物理信道数约束 BLE 起始时隙调度。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = resolve_runtime_config(args)
    np.set_printoptions(threshold=10)
    np.set_printoptions(linewidth=1000)
    runtime_device = resolve_torch_device(config["use_gpu"], config["gpu_id"])
    pair_density = (
        compute_pair_density_for_manual_pairs(config["cell_size"], len(config["pair_parameters"]))
        if config["pair_generation_mode"] == "manual"
        else config["pair_density"]
    )

    e = env(
        cell_edge=7.0,
        cell_size=config["cell_size"],
        pair_density_per_m2=pair_density,
        seed=int(time.time()) if config["seed"] is None else config["seed"],
        radio_prob=(0.1, 0.9),
        slot_time=1.25e-3,
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=7.5e-3,
        ble_payload_bits=800,
        ble_phy_rate_bps=2e6,
        ble_ci_exp_min=3,
        ble_ci_exp_max=6,
        ble_channel_mode=config["ble_channel_mode"],
    )
    if config["pair_generation_mode"] == "manual":
        apply_manual_pair_parameters(e, config["pair_parameters"])
    defer_wifi_first_ble_backend = bool(
        config["wifi_first_ble_scheduling"]
        and config["ble_schedule_backend"] in {"macrocycle_hopping_sdp", "macrocycle_hopping_ga"}
    )
    if not defer_wifi_first_ble_backend:
        apply_ble_schedule_backend(e, config)

    print("n_pair =", e.n_pair)
    print("n_device =", 2 * e.n_pair)
    print("n_office =", e.n_office)
    print("office_area_m2 =", e.office_area_m2)
    print("pair_density_per_m2 =", e.pair_density_per_m2)
    print("pair_generation_mode =", config["pair_generation_mode"])
    print("ble_schedule_backend =", config["ble_schedule_backend"])
    print("runtime_device =", runtime_device)
    print("wifi_period_quanta_candidates:", e.wifi_period_quanta_candidates.tolist())
    print("ble_ci_quanta_candidates:", e.ble_ci_quanta_candidates.tolist())
    print("n_wifi_pair =", int(np.sum(e.pair_radio_type == e.RADIO_WIFI)))
    print("n_ble_pair  =", int(np.sum(e.pair_radio_type == e.RADIO_BLE)))
    print("wifi_first_ble_scheduling =", bool(config["wifi_first_ble_scheduling"]))

    if e.n_pair == 0:
        z_vec_pref = np.zeros(0, dtype=int)
        Z_fin_mmw = 0
        remainder = 0
        avg_bler = float("nan")
        max_bler = float("nan")
        weighted_bler = float("nan")
        schedule_start_slots = np.zeros(0, dtype=int)
        macrocycle_slots = 0
        occupancy = np.zeros((0, 0), dtype=bool)
        macro_unscheduled = []
        ble_channel_retries_used = 0
        ble_slot_stats = {}
        partial = {
            "slot_cap_hit": False,
            "scheduled_pair_ids": [],
            "unscheduled_pair_ids": [],
        }
        partial_schedule = False
        scheduled_pair_ids = []
        unscheduled_pair_ids = []
        print("MMW result:", Z_fin_mmw, remainder, avg_bler, max_bler, weighted_bler)
        print("macrocycle_slots =", macrocycle_slots)
        print("macrocycle_ms =", float(macrocycle_slots * e.slot_time * 1e3))
        print("partial_schedule =", partial_schedule)
        print("scheduled_pair_ids =", scheduled_pair_ids)
        print("unscheduled_pair_ids =", unscheduled_pair_ids)
        print("ble_channel_retries_used =", ble_channel_retries_used)
    else:
        def make_alg():
            alg = mmw(nit=config["mmw_nit"], eta=config["mmw_eta"], device=runtime_device)
            alg.DEBUG = False
            alg.LOG_GAP = False
            return alg

        bs = binary_search_relaxation()
        bs.force_lower_bound = False
        bs.max_slot_cap = int(config["max_slots"])
        bs.user_priority = e.pair_priority
        bs.slot_mask_builder = lambda Z, state, ee=e: ee.build_slot_compatibility_mask(Z)
        if config["wifi_first_ble_scheduling"]:
            bs.strategy = "wifi_first"
            bs.pair_radio_type = e.pair_radio_type
            bs.wifi_radio_id = e.RADIO_WIFI
            bs.ble_radio_id = e.RADIO_BLE
        else:
            bs.strategy = "joint"
        bs.feasibility_check_alg = make_alg()

        z_vec_pref, Z_fin_mmw, remainder = bs.run(e.generate_S_Q_hmax())
        partial = bs.last_partial_schedule or {
            "slot_cap_hit": False,
            "scheduled_pair_ids": list(range(e.n_pair)),
            "unscheduled_pair_ids": [],
        }
        if config["wifi_first_ble_scheduling"]:
            stage_results = bs.last_stage_results or {"wifi": {"z_fin": 0, "remainder": 0}, "ble": {"z_fin": 0, "remainder": 0}}
            print(
                "WiFi-first stage result:",
                {
                    "wifi_slots": int(stage_results["wifi"]["z_fin"]),
                    "ble_slots": int(stage_results["ble"]["z_fin"]),
                    "wifi_remainder": int(stage_results["wifi"]["remainder"]),
                    "ble_remainder": int(stage_results["ble"]["remainder"]),
                },
            )
        if partial["unscheduled_pair_ids"]:
            avg_bler = float("nan")
            max_bler = float("nan")
            weighted_bler = float("nan")
        else:
            bler_arr = e.evaluate_bler(z_vec_pref, Z_fin_mmw)
            avg_bler = float(np.mean(bler_arr))
            max_bler = float(np.max(bler_arr))
            weighted_bler = float(e.evaluate_weighted_bler(z_vec_pref, Z_fin_mmw))
        print("MMW result:", Z_fin_mmw, remainder, avg_bler, max_bler, weighted_bler)
        z_vec = np.asarray(z_vec_pref, dtype=int)
        if defer_wifi_first_ble_backend:
            wifi_interference_blocks = build_wifi_first_ble_external_interference_blocks(e, z_vec)
            print("wifi_interference_blocks =", len(wifi_interference_blocks))
            apply_ble_schedule_backend(
                e,
                config,
                external_interference_blocks=wifi_interference_blocks,
            )
        if config["wifi_first_ble_scheduling"]:
            schedule_start_slots, macrocycle_slots, occupancy, macro_unscheduled, ble_channel_retries_used, ble_slot_stats = (
                retry_ble_channels_and_assign_macrocycle(
                    e,
                    z_vec,
                    max_ble_channel_retries=config["ble_channel_retries"],
                    wifi_first=True,
                    return_ble_stats=True,
                )
            )
        else:
            schedule_start_slots, macrocycle_slots, occupancy, macro_unscheduled, ble_channel_retries_used = (
                retry_ble_channels_and_assign_macrocycle(
                    e,
                    z_vec,
                    max_ble_channel_retries=config["ble_channel_retries"],
                )
            )
            ble_slot_stats = {}
        scheduled_pair_ids, unscheduled_pair_ids = resolve_macrocycle_schedule_status(schedule_start_slots, occupancy)
        partial_schedule = bool(unscheduled_pair_ids)
        print("macrocycle_slots =", macrocycle_slots)
        print("macrocycle_ms =", float(macrocycle_slots * e.slot_time * 1e3))
        print("partial_schedule =", partial_schedule)
        print("scheduled_pair_ids =", scheduled_pair_ids)
        print("unscheduled_pair_ids =", unscheduled_pair_ids)
        print("ble_channel_retries_used =", ble_channel_retries_used)

    ble_idx = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    if ble_idx.size:
        infeasible = int(np.sum(~e.pair_ble_ce_feasible[ble_idx]))
        print(
            "BLE timing summary:",
            {
                "n_ble_pair": int(ble_idx.size),
                "ci_slots_min": int(np.min(e.pair_ble_ci_slots[ble_idx])),
                "ci_slots_avg": float(np.mean(e.pair_ble_ci_slots[ble_idx])),
                "ci_slots_max": int(np.max(e.pair_ble_ci_slots[ble_idx])),
                "ce_slots_min": int(np.min(e.pair_ble_ce_slots[ble_idx])),
                "ce_slots_avg": float(np.mean(e.pair_ble_ce_slots[ble_idx])),
                "ce_slots_max": int(np.max(e.pair_ble_ce_slots[ble_idx])),
                "infeasible_ble_pair": infeasible,
            },
        )

    pair_rows = compute_pair_parameter_rows(
        e,
        schedule_start_slots,
        occupancy,
        macrocycle_slots,
        ble_slot_stats=ble_slot_stats,
    )
    print_pair_parameter_rows(pair_rows)
    write_rows_to_csv(
        os.path.join(config["output_dir"], "pair_parameters.csv"),
        [
            "pair_id",
            "office_id",
            "radio",
            "channel",
            "priority",
            "release_time_slot",
            "deadline_slot",
            "schedule_slot",
            "schedule_time_ms",
            "ble_channel_mode",
            "ble_ce_channel_summary",
            "start_time_slot",
            "wifi_anchor_slot",
            "wifi_period_slots",
            "wifi_period_ms",
            "wifi_tx_slots",
            "wifi_tx_ms",
            "ble_anchor_slot",
            "ble_ci_slots",
            "ble_ci_ms",
            "ble_ce_slots",
            "ble_ce_ms",
            "ble_ce_feasible",
            "effective_ble_channels",
            "scheduled_ble_pairs",
            "no_collision_probability",
            "macrocycle_slots",
            "occupied_slots_in_macrocycle",
        ],
        pair_rows,
    )

    scheduled_pair_rows = filter_pair_rows_by_ids(pair_rows, scheduled_pair_ids)
    unscheduled_pair_rows = filter_pair_rows_by_ids(pair_rows, unscheduled_pair_ids)
    schedule_rows = build_schedule_rows(scheduled_pair_rows)
    print_schedule_rows(schedule_rows)
    write_rows_to_csv(
        os.path.join(config["output_dir"], "wifi_ble_schedule.csv"),
        [
            "schedule_slot",
            "pair_ids",
            "wifi_pair_ids",
            "ble_pair_ids",
            "pair_count",
            "wifi_pair_count",
            "ble_pair_count",
        ],
        schedule_rows,
    )
    write_rows_to_csv(
        os.path.join(config["output_dir"], "unscheduled_pairs.csv"),
        [
            "pair_id",
            "office_id",
            "radio",
            "channel",
            "priority",
            "release_time_slot",
            "deadline_slot",
            "schedule_slot",
            "schedule_time_ms",
            "ble_channel_mode",
            "ble_ce_channel_summary",
            "start_time_slot",
            "wifi_anchor_slot",
            "wifi_period_slots",
            "wifi_period_ms",
            "wifi_tx_slots",
            "wifi_tx_ms",
            "ble_anchor_slot",
            "ble_ci_slots",
            "ble_ci_ms",
            "ble_ce_slots",
            "ble_ce_ms",
            "ble_ce_feasible",
            "effective_ble_channels",
            "scheduled_ble_pairs",
            "no_collision_probability",
            "macrocycle_slots",
            "occupied_slots_in_macrocycle",
        ],
        unscheduled_pair_rows,
    )
    pair_channel_ranges = get_pair_channel_ranges_mhz(e, scheduled_pair_ids)
    schedule_plot_rows = build_schedule_plot_rows(scheduled_pair_rows, pair_channel_ranges, e=e)
    ble_ce_event_rows = build_ble_ce_event_rows(e, scheduled_pair_rows)
    write_rows_to_csv(
        os.path.join(config["output_dir"], "schedule_plot_rows.csv"),
        [
            "pair_id",
            "radio",
            "channel",
            "slot",
            "slot_width",
            "freq_low_mhz",
            "freq_high_mhz",
            "label",
        ],
        schedule_plot_rows,
    )
    if e.ble_channel_mode == "per_ce":
        write_rows_to_csv(
            os.path.join(config["output_dir"], "ble_ce_channel_events.csv"),
            [
                "pair_id",
                "event_index",
                "channel",
                "slot_start",
                "slot_end",
                "freq_low_mhz",
                "freq_high_mhz",
            ],
            ble_ce_event_rows,
        )
    overview_plot_path, window_plot_paths = render_all_from_csv(
        config["output_dir"],
        macrocycle_slots=macrocycle_slots,
        window_slots=128,
    )
    schedule_plot_path = os.path.join(config["output_dir"], "wifi_ble_schedule.png")
    # Keep the legacy filename for compatibility by copying the overview output.
    with open(overview_plot_path, "rb") as src, open(schedule_plot_path, "wb") as dst:
        dst.write(src.read())
    print(
        "CSV outputs:",
        os.path.join(config["output_dir"], "pair_parameters.csv"),
        os.path.join(config["output_dir"], "wifi_ble_schedule.csv"),
    )
    print("Schedule plot:", schedule_plot_path)
    print("Schedule overview plot:", str(overview_plot_path))
    print("Schedule window plots:", [str(p) for p in window_plot_paths])

    office_rows = compute_office_pair_slot_stats_for_pair_ids(e, schedule_start_slots, scheduled_pair_ids)
    print_office_stats(office_rows)
