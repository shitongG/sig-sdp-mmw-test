"""Joint GA backend for the isolated joint WiFi/BLE experiment."""

from __future__ import annotations

from dataclasses import asdict
from functools import cmp_to_key
from typing import Any, Mapping
import random

import numpy as np

from .joint_wifi_ble_model import (
    JointCandidateSpace,
    JointCandidateState,
    build_joint_candidate_states,
    build_joint_cost_matrix,
    build_joint_forbidden_state_pairs,
    build_payload_by_pair,
    expand_candidate_blocks,
    resolve_joint_objective_policy,
    selected_schedule_has_no_conflicts,
    state_is_idle,
    summarize_selected_schedule_metrics,
)

DEFAULT_POPULATION_SIZE = 48
DEFAULT_GENERATIONS = 60
DEFAULT_MUTATION_RATE = 0.15
DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_ELITE_COUNT = 4
DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_SEED = 7
INVALID_FITNESS = -1e12


class JointGAContext:
    def __init__(self, config: Mapping[str, Any], space: JointCandidateSpace, cost_matrix: np.ndarray):
        self.config = config
        self.space = space
        self.cost_matrix = cost_matrix
        self.forbidden_pairs = build_joint_forbidden_state_pairs(space.states)
        self.pair_ids = sorted(space.pair_to_state_indices)
        self.gene_options = [space.pair_to_state_indices[pair_id] for pair_id in self.pair_ids]
        self.objective_policy = resolve_joint_objective_policy(config)
        self.payload_tie_tolerance = float(self.objective_policy.get("payload_tie_tolerance", 0.0))
        self.metric_cache: dict[tuple[int, ...], dict[str, float] | None] = {}
        self.metric_cache_hits = 0


def chromosome_is_feasible(chromosome: list[int], context: JointGAContext) -> bool:
    for i, left_idx in enumerate(chromosome):
        for right_idx in chromosome[i + 1 :]:
            pair = (min(left_idx, right_idx), max(left_idx, right_idx))
            if pair in context.forbidden_pairs:
                return False
    return True


def chromosome_cost(chromosome: list[int], context: JointGAContext) -> float:
    cost = 0.0
    for i, left_idx in enumerate(chromosome):
        for right_idx in chromosome[i + 1 :]:
            cost += float(context.cost_matrix[left_idx, right_idx])
    return cost


def chromosome_metrics(chromosome: list[int], context: JointGAContext) -> dict[str, float] | None:
    cache_key = tuple(chromosome)
    if cache_key in context.metric_cache:
        context.metric_cache_hits += 1
        cached_metrics = context.metric_cache[cache_key]
        return None if cached_metrics is None else dict(cached_metrics)
    if not chromosome_is_feasible(chromosome, context):
        context.metric_cache[cache_key] = None
        return None
    all_states = [context.space.states[idx] for idx in chromosome]
    selected_states = [state for state in all_states if not state_is_idle(state)]
    metrics = summarize_selected_schedule_metrics(context.config, selected_states, objective=context.config)
    metrics["soft_cost"] = chromosome_cost(chromosome, context)
    metrics["scheduled_count"] = float(len(selected_states))
    context.metric_cache[cache_key] = dict(metrics)
    return metrics


def compare_metric_dicts(left: dict[str, float] | None, right: dict[str, float] | None, tolerance: float) -> int:
    if left is None and right is None:
        return 0
    if left is None:
        return -1
    if right is None:
        return 1
    left_payload = float(left["scheduled_payload_bytes"])
    right_payload = float(right["scheduled_payload_bytes"])
    if left_payload > right_payload + tolerance:
        return 1
    if right_payload > left_payload + tolerance:
        return -1
    for key in ("fill_penalty", "soft_cost", "occupied_slot_count"):
        left_value = float(left.get(key, 0.0))
        right_value = float(right.get(key, 0.0))
        if left_value < right_value - 1e-9:
            return 1
        if right_value < left_value - 1e-9:
            return -1
    if float(left.get("scheduled_count", 0.0)) > float(right.get("scheduled_count", 0.0)):
        return 1
    if float(right.get("scheduled_count", 0.0)) > float(left.get("scheduled_count", 0.0)):
        return -1
    return 0


def summarize_radio_payloads(selected_states: list[Any], task_payloads: Mapping[int, Any]) -> dict[str, float]:
    wifi_payload_bytes = 0.0
    ble_payload_bytes = 0.0
    selected_pairs = 0.0
    for state in selected_states:
        if isinstance(state, Mapping):
            pair_id = int(state["pair_id"])
            medium = str(state.get("medium", ""))
        else:
            pair_id = int(getattr(state, "pair_id"))
            medium = str(getattr(state, "medium", ""))
        task = task_payloads.get(pair_id, {})
        if isinstance(task, Mapping):
            payload_bytes = float(task.get("payload_bytes", 0.0))
            radio = str(task.get("radio", medium))
        else:
            payload_bytes = float(task)
            radio = medium
        if radio == "wifi":
            wifi_payload_bytes += payload_bytes
        else:
            ble_payload_bytes += payload_bytes
        selected_pairs += 1.0
    return {
        "wifi_payload_bytes": wifi_payload_bytes,
        "ble_payload_bytes": ble_payload_bytes,
        "scheduled_payload_bytes": wifi_payload_bytes + ble_payload_bytes,
        "selected_pairs": selected_pairs,
    }


def compare_joint_candidate_scores(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None, wifi_payload_floor: int) -> int:
    if left is None and right is None:
        return 0
    if left is None:
        return -1
    if right is None:
        return 1
    left_wifi = float(left.get("wifi_payload_bytes", 0.0))
    right_wifi = float(right.get("wifi_payload_bytes", 0.0))
    left_valid = left_wifi >= float(wifi_payload_floor)
    right_valid = right_wifi >= float(wifi_payload_floor)
    if left_valid != right_valid:
        return 1 if left_valid else -1
    left_key = (
        left_wifi,
        float(left.get("scheduled_payload_bytes", 0.0)),
        -float(left.get("fill_penalty", 0.0)),
        float(left.get("selected_pairs", 0.0)),
    )
    right_key = (
        right_wifi,
        float(right.get("scheduled_payload_bytes", 0.0)),
        -float(right.get("fill_penalty", 0.0)),
        float(right.get("selected_pairs", 0.0)),
    )
    return (left_key > right_key) - (left_key < right_key)


def _build_task_payloads(space: JointCandidateSpace) -> dict[int, dict[str, Any]]:
    task_payloads: dict[int, dict[str, Any]] = {}
    for pair in space.wifi_pairs:
        task_payloads[int(pair.pair_id)] = {"radio": "wifi", "payload_bytes": int(pair.payload_bytes)}
    for pair in space.ble_pairs:
        task_payloads[int(pair.pair_id)] = {"radio": "ble", "payload_bytes": int(pair.payload_bytes)}
    return task_payloads


def _build_candidate_metrics(
    chromosome: list[int],
    context: JointGAContext,
    task_payloads: Mapping[int, Any],
) -> dict[str, float] | None:
    metrics = chromosome_metrics(chromosome, context)
    if metrics is None:
        return None
    selected_states = [
        context.space.states[state_index]
        for state_index in chromosome
        if not state_is_idle(context.space.states[state_index])
    ]
    radio_summary = summarize_radio_payloads(selected_states, task_payloads)
    candidate_metrics = dict(metrics)
    candidate_metrics.update(radio_summary)
    return candidate_metrics


def _build_seeded_chromosome_from_specs(
    space: JointCandidateSpace,
    seeded_state_specs: list[Mapping[str, Any]],
) -> list[int] | None:
    pair_ids = sorted(space.pair_to_state_indices)
    by_pair = {int(spec["pair_id"]): spec for spec in seeded_state_specs if "pair_id" in spec}
    chromosome: list[int] = []
    for pair_id in pair_ids:
        options = space.pair_to_state_indices[pair_id]
        spec = by_pair.get(pair_id)
        selected_idx = None
        if spec is not None:
            for idx in options:
                state = space.states[idx]
                if state.medium != str(spec.get("medium", state.medium)):
                    continue
                if state.offset != int(spec.get("offset", state.offset)):
                    continue
                if state.channel != spec.get("channel", state.channel):
                    continue
                if state.period_slots != spec.get("period_slots", state.period_slots):
                    continue
                if state.width_slots != spec.get("width_slots", state.width_slots):
                    continue
                if state.num_events != spec.get("num_events", state.num_events):
                    continue
                selected_idx = idx
                break
        if selected_idx is None:
            for idx in options:
                if state_is_idle(space.states[idx]):
                    selected_idx = idx
                    break
        if selected_idx is None:
            return None
        chromosome.append(selected_idx)
    return chromosome


def chromosome_fitness(chromosome: list[int], context: JointGAContext) -> float:
    metrics = chromosome_metrics(chromosome, context)
    if metrics is None:
        return INVALID_FITNESS
    return float(metrics["scheduled_payload_bytes"] - metrics["fill_penalty"] - metrics["soft_cost"])


def initialize_population(
    context: JointGAContext,
    population_size: int,
    rng: random.Random,
    seeded_chromosomes: list[list[int]] | None = None,
) -> list[list[int]]:
    population: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    if seeded_chromosomes:
        for chromosome in seeded_chromosomes:
            if len(population) >= population_size:
                break
            if len(chromosome) != len(context.gene_options):
                continue
            repaired = repair_chromosome(list(chromosome), context, rng)
            if repaired is None:
                continue
            repaired_key = tuple(repaired)
            if repaired_key in seen:
                continue
            population.append(repaired)
            seen.add(repaired_key)

    attempts = 0
    max_attempts = max(100, population_size * 20)
    while len(population) < population_size and attempts < max_attempts:
        chromosome = [rng.choice(options) for options in context.gene_options]
        attempts += 1
        repaired = repair_chromosome(chromosome, context, rng)
        if repaired is not None:
            repaired_key = tuple(repaired)
            if repaired_key in seen:
                continue
            population.append(repaired)
            seen.add(repaired_key)
    return population


def tournament_select(
    population: list[list[int]],
    metrics: list[dict[str, float] | None],
    wifi_payload_floor: int,
    rng: random.Random,
) -> list[int]:
    best_idx = None
    for _ in range(DEFAULT_TOURNAMENT_SIZE):
        candidate_idx = rng.randrange(len(population))
        if best_idx is None or compare_joint_candidate_scores(metrics[candidate_idx], metrics[best_idx], wifi_payload_floor) > 0:
            best_idx = candidate_idx
    return list(population[best_idx])


def crossover(parent_a: list[int], parent_b: list[int], rng: random.Random, crossover_rate: float) -> tuple[list[int], list[int]]:
    if len(parent_a) <= 1 or rng.random() >= crossover_rate:
        return list(parent_a), list(parent_b)
    point = rng.randrange(1, len(parent_a))
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


def repair_chromosome(chromosome: list[int], context: JointGAContext, rng: random.Random) -> list[int] | None:
    repaired: list[int] = []
    for gene_index, selected_idx in enumerate(chromosome):
        options = list(context.gene_options[gene_index])
        idle_options = [idx for idx in options if state_is_idle(context.space.states[idx])]
        active_options = [idx for idx in options if idx != selected_idx and not state_is_idle(context.space.states[idx])]
        rng.shuffle(active_options)
        preferred = []
        if selected_idx in options:
            preferred.append(selected_idx)
        preferred.extend(active_options)
        preferred.extend(idle_options)
        seen = set()
        choice = None
        for candidate_idx in preferred:
            if candidate_idx in seen:
                continue
            seen.add(candidate_idx)
            feasible = True
            for chosen_idx in repaired:
                pair = (min(candidate_idx, chosen_idx), max(candidate_idx, chosen_idx))
                if pair in context.forbidden_pairs:
                    feasible = False
                    break
            if feasible:
                choice = candidate_idx
                break
        if choice is None:
            return None
        repaired.append(choice)
    return repaired


def mutate(chromosome: list[int], context: JointGAContext, rng: random.Random, mutation_rate: float) -> list[int] | None:
    mutated = list(chromosome)
    for gene_index, options in enumerate(context.gene_options):
        if len(options) > 1 and rng.random() < mutation_rate:
            mutated[gene_index] = rng.choice(options)
    return repair_chromosome(mutated, context, rng)


def chromosome_to_states(chromosome: list[int], context: JointGAContext) -> list[JointCandidateState]:
    return [context.space.states[state_index] for state_index in chromosome]


def _empty_result(pair_count: int, state_count: int, status: str) -> dict[str, Any]:
    return {
        "solver": "ga",
        "status": status,
        "task_count": pair_count,
        "state_count": state_count,
        "selected_state_indices": [],
        "selected_states": [],
        "unscheduled_pair_ids": [],
        "blocks": [],
        "best_fitness": 0.0,
        "best_cost": 0.0,
        "fitness_history": [],
        "scheduled_payload_bytes": 0.0,
        "occupied_slot_count": 0.0,
        "occupied_area_mhz_slots": 0.0,
        "fragmentation_penalty": 0.0,
        "idle_area_penalty": 0.0,
        "slot_span_penalty": 0.0,
        "fill_penalty": 0.0,
    }


def solve_joint_wifi_ble_ga(config: Mapping[str, Any]) -> dict[str, Any]:
    """Solve the isolated joint WiFi/BLE scheduling problem with a GA over the mixed state space."""

    space = build_joint_candidate_states(config)
    state_count = len(space.states)
    pair_count = len(space.pair_to_state_indices)
    if state_count == 0:
        return _empty_result(pair_count, 0, "empty")

    ga_cfg = dict(config.get("ga", {})) if isinstance(config, Mapping) else {}
    population_size = int(ga_cfg.get("population_size", DEFAULT_POPULATION_SIZE))
    generations = int(ga_cfg.get("generations", DEFAULT_GENERATIONS))
    mutation_rate = float(ga_cfg.get("mutation_rate", DEFAULT_MUTATION_RATE))
    crossover_rate = float(ga_cfg.get("crossover_rate", DEFAULT_CROSSOVER_RATE))
    elite_count = int(ga_cfg.get("elite_count", DEFAULT_ELITE_COUNT))
    seed = int(ga_cfg.get("seed", DEFAULT_SEED))
    seeded_chromosomes_raw = ga_cfg.get("seeded_chromosomes", [])
    seeded_chromosomes = [
        [int(gene) for gene in chromosome]
        for chromosome in seeded_chromosomes_raw
        if isinstance(chromosome, list)
    ]
    wifi_payload_floor = int(config.get("objective", {}).get("wifi_payload_floor_bytes", 0)) if isinstance(config, Mapping) else 0

    context = JointGAContext(config, space, np.asarray(build_joint_cost_matrix(space.states), dtype=float))
    task_payloads = _build_task_payloads(space)
    seeded_state_specs = ga_cfg.get("seeded_state_specs", [])
    if isinstance(seeded_state_specs, list):
        seeded_from_specs = _build_seeded_chromosome_from_specs(space, seeded_state_specs)
        if seeded_from_specs is not None:
            seeded_chromosomes = [seeded_from_specs] + seeded_chromosomes
    rng = random.Random(seed)
    population = initialize_population(context, max(2, population_size), rng, seeded_chromosomes=seeded_chromosomes)
    if not population:
        return _empty_result(pair_count, state_count, "infeasible")

    best_chromosome = None
    best_metrics = None
    fitness_history: list[float] = []
    if seeded_chromosomes:
        baseline_chromosome = repair_chromosome(list(seeded_chromosomes[0]), context, rng)
        if baseline_chromosome is not None:
            baseline_metrics = _build_candidate_metrics(baseline_chromosome, context, task_payloads)
            if baseline_metrics is not None:
                wifi_payload_floor = max(wifi_payload_floor, int(baseline_metrics["wifi_payload_bytes"]))
                best_chromosome = list(baseline_chromosome)
                best_metrics = baseline_metrics

    for _ in range(max(1, generations)):
        metric_list = [_build_candidate_metrics(chromosome, context, task_payloads) for chromosome in population]
        generation_best_idx = max(
            range(len(population)),
            key=cmp_to_key(lambda a, b: compare_joint_candidate_scores(metric_list[a], metric_list[b], wifi_payload_floor)),
        )
        generation_best_metrics = metric_list[generation_best_idx]
        fitness_history.append(float(generation_best_metrics["scheduled_payload_bytes"]) if generation_best_metrics is not None else INVALID_FITNESS)
        if best_metrics is None or compare_joint_candidate_scores(generation_best_metrics, best_metrics, wifi_payload_floor) > 0:
            best_metrics = generation_best_metrics
            best_chromosome = list(population[generation_best_idx])

        ranked_indices = sorted(
            range(len(population)),
            key=cmp_to_key(lambda a, b: compare_joint_candidate_scores(metric_list[a], metric_list[b], wifi_payload_floor)),
            reverse=True,
        )
        next_population = [list(population[idx]) for idx in ranked_indices[: max(1, min(elite_count, len(population)))] ]
        while len(next_population) < len(population):
            parent_a = tournament_select(population, metric_list, wifi_payload_floor, rng)
            parent_b = tournament_select(population, metric_list, wifi_payload_floor, rng)
            child_a, child_b = crossover(parent_a, parent_b, rng, crossover_rate)
            repaired_a = repair_chromosome(child_a, context, rng)
            repaired_b = repair_chromosome(child_b, context, rng)
            if repaired_a is not None:
                mutated_a = mutate(repaired_a, context, rng, mutation_rate)
                if mutated_a is not None:
                    next_population.append(mutated_a)
            if len(next_population) < len(population) and repaired_b is not None:
                mutated_b = mutate(repaired_b, context, rng, mutation_rate)
                if mutated_b is not None:
                    next_population.append(mutated_b)
            if len(next_population) < len(population) and repaired_a is None and repaired_b is None:
                fallback = initialize_population(context, 1, rng)
                if fallback:
                    next_population.extend(fallback)
                else:
                    break
        if not next_population:
            break
        population = next_population[: len(population)]

    if best_chromosome is None or best_metrics is None:
        return _empty_result(pair_count, state_count, "infeasible")
    all_selected_states = chromosome_to_states(best_chromosome, context)
    selected_states = [state for state in all_selected_states if not state_is_idle(state)]
    if not selected_schedule_has_no_conflicts(selected_states):
        return _empty_result(pair_count, state_count, "infeasible")
    blocks = [block for state in selected_states for block in expand_candidate_blocks(state)]
    best_cost = chromosome_cost(best_chromosome, context)
    unscheduled_pair_ids = [state.pair_id for state in all_selected_states if state_is_idle(state)]
    radio_summary = summarize_radio_payloads(selected_states, task_payloads)
    if int(radio_summary["wifi_payload_bytes"]) < wifi_payload_floor:
        return _empty_result(pair_count, state_count, "wifi_floor_infeasible")

    return {
        "solver": "ga",
        "status": "ok",
        "task_count": pair_count,
        "state_count": state_count,
        "selected_state_indices": list(best_chromosome),
        "selected_states": [asdict(state) for state in selected_states],
        "unscheduled_pair_ids": unscheduled_pair_ids,
        "blocks": [asdict(block) for block in blocks],
        "best_fitness": float(best_metrics["scheduled_payload_bytes"]),
        "best_cost": float(best_cost),
        "fitness_history": fitness_history,
        **radio_summary,
        **best_metrics,
    }
