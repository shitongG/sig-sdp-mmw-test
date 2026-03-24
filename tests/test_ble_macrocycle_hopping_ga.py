import random
from pathlib import Path

import numpy as np

import ble_macrocycle_hopping_ga as MODULE
from ble_macrocycle_hopping_sdp import CandidateState, ExternalInterferenceBlock, HoppingPattern, PairConfig, build_candidate_states


def _mini_instance():
    pair_configs = [
        PairConfig(pair_id=0, release_time=0, deadline=0, connect_interval=4, event_duration=1, num_events=1),
        PairConfig(pair_id=1, release_time=0, deadline=0, connect_interval=4, event_duration=1, num_events=1),
    ]
    cfg_dict = {cfg.pair_id: cfg for cfg in pair_configs}
    pattern_dict = {
        0: [
            HoppingPattern(pattern_id=0, start_channel=0, hop_increment=0),
            HoppingPattern(pattern_id=1, start_channel=20, hop_increment=0),
        ],
        1: [HoppingPattern(pattern_id=0, start_channel=30, hop_increment=0)],
    }
    states, _, A_k = build_candidate_states(pair_configs, pattern_dict)
    return pair_configs, cfg_dict, pattern_dict, states, A_k


def test_build_pair_candidate_groups_returns_local_choice_lists():
    _, _, _, states, _ = _mini_instance()
    groups = MODULE.build_pair_candidate_groups(candidate_states=states, pair_ids=[0, 1])
    assert [group.pair_id for group in groups] == [0, 1]
    assert [group.size for group in groups] == [2, 1]
    assert groups[0].candidates[0] == CandidateState(pair_id=0, offset=0, pattern_id=0)


def test_decode_chromosome_selects_one_state_per_pair():
    _, _, _, states, _ = _mini_instance()
    groups = MODULE.build_pair_candidate_groups(candidate_states=states, pair_ids=[0, 1])
    selected = MODULE.decode_ga_chromosome(chromosome=[1, 0], pair_candidate_groups=groups)
    assert selected[0] == states[1]
    assert selected[1] == states[2]


def test_ga_fitness_matches_collision_plus_external_penalty():
    _, cfg_dict, pattern_dict, states, _ = _mini_instance()
    groups = MODULE.build_pair_candidate_groups(candidate_states=states, pair_ids=[0, 1])
    result = MODULE.evaluate_ga_chromosome(
        chromosome=[0, 0],
        pair_candidate_groups=groups,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=37,
        external_interference_blocks=[
            ExternalInterferenceBlock(
                start_slot=0,
                end_slot=0,
                freq_low_mhz=2402.0,
                freq_high_mhz=2422.0,
                source_type="wifi",
                source_pair_id=9,
            )
        ],
    )
    assert result["best_fitness"] > 0.0
    assert result["external_cost"] > 0.0


def test_initialize_ga_population_respects_pair_local_choice_ranges():
    _, _, _, states, _ = _mini_instance()
    groups = MODULE.build_pair_candidate_groups(candidate_states=states, pair_ids=[0, 1])
    pop = MODULE.initialize_ga_population(
        pair_candidate_groups=groups,
        population_size=8,
        rng=random.Random(7),
    )
    assert len(pop) == 8
    assert all(len(chrom) == 2 for chrom in pop)
    assert all(0 <= chrom[0] < 2 for chrom in pop)
    assert all(0 <= chrom[1] < 1 + 0 for chrom in pop)


def test_mutation_keeps_gene_values_in_local_range():
    _, _, _, states, _ = _mini_instance()
    groups = MODULE.build_pair_candidate_groups(candidate_states=states, pair_ids=[0, 1])
    child = MODULE.mutate_ga_chromosome(
        chromosome=[0, 0],
        pair_candidate_groups=groups,
        mutation_rate=1.0,
        rng=random.Random(3),
    )
    assert 0 <= child[0] < 2
    assert child[1] == 0


def test_crossover_preserves_length():
    child_a, child_b = MODULE.crossover_ga_chromosomes(
        [0, 1, 0],
        [1, 0, 1],
        rng=random.Random(5),
        crossover_rate=1.0,
    )
    assert len(child_a) == 3
    assert len(child_b) == 3



def test_ga_module_avoids_top_level_import_of_sdp_script():
    source = (Path(__file__).resolve().parents[1] / "ble_macrocycle_hopping_ga.py").read_text()
    assert 'import ble_macrocycle_hopping_sdp as sdp' not in source
    assert 'importlib.import_module("ble_macrocycle_hopping_sdp")' in source


def test_decode_chromosome_sorts_mapping_groups_by_pair_id():
    _, _, _, states, _ = _mini_instance()
    groups = MODULE.build_pair_candidate_groups(candidate_states=states, pair_ids=[0, 1])
    mapping = {1: groups[1], 0: groups[0]}
    selected = MODULE.decode_ga_chromosome(chromosome=[1, 0], pair_candidate_groups=mapping)
    assert selected[0] == states[1]
    assert selected[1] == states[2]

def test_solve_ble_hopping_schedule_ga_returns_selected_schedule_dict():
    _, cfg_dict, pattern_dict, states, _ = _mini_instance()
    result = MODULE.solve_ble_hopping_schedule_ga(
        candidate_states=states,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=37,
        pair_ids=[0, 1],
        population_size=16,
        generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_count=2,
        seed=7,
    )
    assert set(result.selected) == {0, 1}
    assert result.best_fitness >= 0.0
    assert len(result.fitness_history) == 11
    assert result.ce_channel_map[0].shape == (1,)


def test_ga_solver_is_reproducible_for_fixed_seed():
    _, cfg_dict, pattern_dict, states, _ = _mini_instance()
    result_a = MODULE.solve_ble_hopping_schedule_ga(
        candidate_states=states,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=37,
        pair_ids=[0, 1],
        population_size=16,
        generations=10,
        seed=11,
    )
    result_b = MODULE.solve_ble_hopping_schedule_ga(
        candidate_states=states,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=37,
        pair_ids=[0, 1],
        population_size=16,
        generations=10,
        seed=11,
    )
    assert result_a.best_fitness == result_b.best_fitness
    assert result_a.ce_channel_map[0].tolist() == result_b.ce_channel_map[0].tolist()
    assert result_a.ce_channel_map[1].tolist() == result_b.ce_channel_map[1].tolist()
