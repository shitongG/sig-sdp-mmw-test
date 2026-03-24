from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from ble_macrocycle_hopping_sdp import CandidateState, EventBlock, ExternalInterferenceBlock, HoppingPattern, PairConfig


def _sdp_runtime() -> Any:
    return importlib.import_module("ble_macrocycle_hopping_sdp")


@dataclass(frozen=True)
class PairCandidateGroup:
    pair_id: int
    candidates: Tuple[Any, ...]

    @property
    def size(self) -> int:
        return len(self.candidates)


@dataclass
class GASolution:
    selected: Dict[int, Any]
    blocks: List[Any]
    overlap_blocks: List[Any]
    ce_channel_map: Dict[int, np.ndarray]
    best_fitness: float
    fitness_history: List[float]
    best_chromosome: List[int]
    collision_cost: float = 0.0
    external_cost: float = 0.0


def _state_pair_id(state: CandidateState) -> int:
    if hasattr(state, "pair_id"):
        return int(getattr(state, "pair_id"))
    if hasattr(state, "pair"):
        return int(getattr(state, "pair"))
    raise AttributeError("candidate state must expose pair_id or pair")


def _as_group_list(
    pair_candidate_groups: Sequence[PairCandidateGroup] | Mapping[int, PairCandidateGroup],
) -> List[PairCandidateGroup]:
    if isinstance(pair_candidate_groups, Mapping):
        return [pair_candidate_groups[key] for key in sorted(pair_candidate_groups)]
    return list(pair_candidate_groups)


def _rng_random(rng: Any) -> float:
    if hasattr(rng, "random"):
        return float(rng.random())
    raise TypeError("rng must provide a random() method")


def _rng_randrange(rng: Any, stop: int) -> int:
    if stop <= 0:
        raise ValueError("stop must be positive")
    if hasattr(rng, "randrange"):
        return int(rng.randrange(stop))
    if hasattr(rng, "integers"):
        return int(rng.integers(stop))
    raise TypeError("rng must provide randrange() or integers()")


def _rng_sample_without_replacement(rng: Any, population_size: int, sample_size: int) -> List[int]:
    if sample_size <= 0:
        return []
    sample_size = min(sample_size, population_size)
    if hasattr(rng, "sample"):
        return list(rng.sample(range(population_size), sample_size))
    if hasattr(rng, "choice"):
        values = rng.choice(population_size, size=sample_size, replace=False)
        return [int(v) for v in np.asarray(values).tolist()]
    raise TypeError("rng must provide sample() or choice()")


def _normalize_rng(rng: Any = None, seed: Optional[int] = None) -> Any:
    if rng is not None:
        return rng
    return random.Random(seed)


def _validate_groups(pair_candidate_groups: Sequence[PairCandidateGroup]) -> None:
    if not pair_candidate_groups:
        raise ValueError("at least one pair candidate group is required")
    for group in pair_candidate_groups:
        if not group.candidates:
            raise ValueError(f"pair {group.pair_id} has no candidate states")


def build_pair_candidate_groups(
    candidate_states: Sequence[Any],
    pair_ids: Optional[Sequence[int]] = None,
) -> List[PairCandidateGroup]:
    grouped: Dict[int, List[Any]] = {}
    order: List[int] = []

    for state in candidate_states:
        pair_id = _state_pair_id(state)
        if pair_id not in grouped:
            grouped[pair_id] = []
            order.append(pair_id)
        grouped[pair_id].append(state)

    if pair_ids is None:
        ordered_pair_ids = order
    else:
        ordered_pair_ids = [int(pair_id) for pair_id in pair_ids]
        missing = [pair_id for pair_id in ordered_pair_ids if pair_id not in grouped]
        if missing:
            raise ValueError(f"missing candidate states for pairs: {missing}")

    groups = [PairCandidateGroup(pair_id=pair_id, candidates=tuple(grouped[pair_id])) for pair_id in ordered_pair_ids]
    _validate_groups(groups)
    return groups


def decode_ga_chromosome(
    chromosome: Sequence[int],
    pair_candidate_groups: Sequence[PairCandidateGroup] | Mapping[int, PairCandidateGroup],
) -> Dict[int, Any]:
    groups = _as_group_list(pair_candidate_groups)
    if len(chromosome) != len(groups):
        raise ValueError("chromosome length must match the number of pair candidate groups")

    selected: Dict[int, Any] = {}
    for gene, group in zip(chromosome, groups):
        gene_index = int(gene)
        if gene_index < 0 or gene_index >= len(group.candidates):
            raise IndexError(f"gene {gene_index} out of range for pair {group.pair_id}")
        selected[group.pair_id] = group.candidates[gene_index]
    return selected


def evaluate_ga_chromosome(
    chromosome: Sequence[int],
    pair_candidate_groups: Sequence[PairCandidateGroup] | Mapping[int, PairCandidateGroup],
    *,
    cfg_dict: Mapping[int, Any],
    pattern_dict: Mapping[int, Sequence[Any]],
    num_channels: int,
    pair_weight: Optional[Mapping[Tuple[int, int], float]] = None,
    external_interference_blocks: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    sdp = _sdp_runtime()
    selected = decode_ga_chromosome(chromosome, pair_candidate_groups)
    blocks = sdp.build_event_blocks(selected=selected, cfg_dict=dict(cfg_dict), pattern_dict=dict(pattern_dict), num_channels=num_channels)
    overlap_blocks = sdp.build_overlap_blocks(blocks)
    ce_channel_map = sdp.selected_schedule_to_ce_channels(
        selected=selected,
        cfg_dict=dict(cfg_dict),
        pattern_dict=dict(pattern_dict),
        num_channels=num_channels,
    )
    collision_cost = float(
        sdp.compute_total_collision_of_schedule(
            selected=selected,
            cfg_dict=dict(cfg_dict),
            pattern_dict=dict(pattern_dict),
            num_channels=num_channels,
            pair_weight=dict(pair_weight) if pair_weight is not None else None,
        )
    )
    external_cost = 0.0
    for state in selected.values():
        external_cost += float(
            sdp.external_interference_cost_for_state(
                state=state,
                cfg_dict=dict(cfg_dict),
                pattern_dict=dict(pattern_dict),
                num_channels=num_channels,
                interference_blocks=list(external_interference_blocks) if external_interference_blocks is not None else None,
            )
        )

    best_fitness = collision_cost + external_cost
    return {
        "chromosome": list(int(g) for g in chromosome),
        "selected": selected,
        "blocks": blocks,
        "overlap_blocks": overlap_blocks,
        "ce_channel_map": ce_channel_map,
        "collision_cost": collision_cost,
        "external_cost": external_cost,
        "best_fitness": float(best_fitness),
    }


def initialize_ga_population(
    pair_candidate_groups: Sequence[PairCandidateGroup] | Mapping[int, PairCandidateGroup],
    population_size: int,
    rng: Any = None,
) -> List[List[int]]:
    groups = _as_group_list(pair_candidate_groups)
    _validate_groups(groups)
    if population_size <= 0:
        raise ValueError("population_size must be positive")

    rng = _normalize_rng(rng)
    population: List[List[int]] = []
    for _ in range(population_size):
        chromosome = [_rng_randrange(rng, len(group.candidates)) for group in groups]
        population.append(chromosome)
    return population


def crossover_ga_chromosomes(
    parent_a: Sequence[int],
    parent_b: Sequence[int],
    rng: Any = None,
    crossover_rate: float = 0.8,
) -> Tuple[List[int], List[int]]:
    if len(parent_a) != len(parent_b):
        raise ValueError("parents must have the same chromosome length")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be within [0, 1]")
    if len(parent_a) <= 1:
        return list(parent_a), list(parent_b)

    rng = _normalize_rng(rng)
    if _rng_random(rng) >= crossover_rate:
        return list(parent_a), list(parent_b)

    crossover_point = _rng_randrange(rng, len(parent_a) - 1) + 1
    child_a = list(parent_a[:crossover_point]) + list(parent_b[crossover_point:])
    child_b = list(parent_b[:crossover_point]) + list(parent_a[crossover_point:])
    return child_a, child_b


def mutate_ga_chromosome(
    chromosome: Sequence[int],
    pair_candidate_groups: Sequence[PairCandidateGroup] | Mapping[int, PairCandidateGroup],
    mutation_rate: float = 0.05,
    rng: Any = None,
) -> List[int]:
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be within [0, 1]")

    groups = _as_group_list(pair_candidate_groups)
    if len(chromosome) != len(groups):
        raise ValueError("chromosome length must match the number of pair candidate groups")

    rng = _normalize_rng(rng)
    mutated = list(int(gene) for gene in chromosome)
    for idx, group in enumerate(groups):
        if len(group.candidates) <= 1:
            continue
        if _rng_random(rng) >= mutation_rate:
            continue
        current = mutated[idx]
        choices = [choice for choice in range(len(group.candidates)) if choice != current]
        mutated[idx] = choices[_rng_randrange(rng, len(choices))]
    return mutated


def _tournament_select(
    evaluated_population: Sequence[Dict[str, Any]],
    tournament_size: int,
    rng: Any,
) -> Dict[str, Any]:
    if tournament_size <= 1 or len(evaluated_population) == 1:
        return evaluated_population[_rng_randrange(rng, len(evaluated_population))]

    indices = _rng_sample_without_replacement(rng, len(evaluated_population), tournament_size)
    contenders = [evaluated_population[idx] for idx in indices]
    return min(contenders, key=lambda item: float(item["best_fitness"]))


def solve_ble_hopping_schedule_ga(
    candidate_states: Sequence[Any],
    *,
    cfg_dict: Mapping[int, Any],
    pattern_dict: Mapping[int, Sequence[Any]],
    num_channels: int,
    pair_ids: Optional[Sequence[int]] = None,
    pair_candidate_groups: Optional[Sequence[PairCandidateGroup] | Mapping[int, PairCandidateGroup]] = None,
    pair_weight: Optional[Mapping[Tuple[int, int], float]] = None,
    external_interference_blocks: Optional[Sequence[Any]] = None,
    population_size: int = 32,
    generations: int = 40,
    mutation_rate: float = 0.05,
    crossover_rate: float = 0.8,
    elite_count: int = 1,
    tournament_size: int = 2,
    seed: Optional[int] = None,
    rng: Any = None,
) -> GASolution:
    if pair_candidate_groups is None:
        groups = build_pair_candidate_groups(candidate_states, pair_ids=pair_ids)
    else:
        groups = _as_group_list(pair_candidate_groups)
    _validate_groups(groups)
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generations < 0:
        raise ValueError("generations must be non-negative")
    if elite_count < 0:
        raise ValueError("elite_count must be non-negative")
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive")

    rng = _normalize_rng(rng, seed)
    elite_count = min(elite_count, population_size)

    population = initialize_ga_population(groups, population_size=population_size, rng=rng)

    def evaluate_population(pop: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
        evaluated: List[Dict[str, Any]] = []
        for chromosome in pop:
            evaluated.append(
                evaluate_ga_chromosome(
                    chromosome,
                    groups,
                    cfg_dict=cfg_dict,
                    pattern_dict=pattern_dict,
                    num_channels=num_channels,
                    pair_weight=pair_weight,
                    external_interference_blocks=external_interference_blocks,
                )
            )
        return evaluated

    evaluated = evaluate_population(population)
    best = min(evaluated, key=lambda item: float(item["best_fitness"]))
    fitness_history = [float(best["best_fitness"])]

    for _ in range(generations):
        ranked = sorted(evaluated, key=lambda item: float(item["best_fitness"]))
        next_population: List[List[int]] = [list(item["chromosome"]) for item in ranked[:elite_count]]

        while len(next_population) < population_size:
            parent_a = _tournament_select(ranked, tournament_size, rng)
            parent_b = _tournament_select(ranked, tournament_size, rng)
            child_a, child_b = crossover_ga_chromosomes(
                parent_a["chromosome"],
                parent_b["chromosome"],
                rng=rng,
                crossover_rate=crossover_rate,
            )
            child_a = mutate_ga_chromosome(child_a, groups, mutation_rate=mutation_rate, rng=rng)
            next_population.append(child_a)
            if len(next_population) < population_size:
                child_b = mutate_ga_chromosome(child_b, groups, mutation_rate=mutation_rate, rng=rng)
                next_population.append(child_b)

        evaluated = evaluate_population(next_population)
        generation_best = min(evaluated, key=lambda item: float(item["best_fitness"]))
        if float(generation_best["best_fitness"]) < float(best["best_fitness"]):
            best = generation_best
        fitness_history.append(float(best["best_fitness"]))

    return GASolution(
        selected=dict(best["selected"]),
        blocks=list(best["blocks"]),
        overlap_blocks=list(best["overlap_blocks"]),
        ce_channel_map=dict(best["ce_channel_map"]),
        best_fitness=float(best["best_fitness"]),
        fitness_history=fitness_history,
        best_chromosome=list(best["chromosome"]),
        collision_cost=float(best["collision_cost"]),
        external_cost=float(best["external_cost"]),
    )
