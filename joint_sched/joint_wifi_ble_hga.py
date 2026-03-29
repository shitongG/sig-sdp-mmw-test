"""Unified heuristic GA backend for the isolated joint WiFi/BLE experiment."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping
import random

from .joint_wifi_ble_ga import solve_joint_wifi_ble_ga
from .joint_wifi_ble_hga_model import (
    build_wifi_state_move_candidates,
    extract_residual_holes,
    identify_blocking_wifi_pairs,
    rank_ble_insertions_for_holes,
    rank_ble_subset_replacements,
    rank_residual_candidate_swaps,
    rank_wifi_state_moves_for_direct_accept_if_better,
    score_ble_state_against_residual_holes,
)
from .joint_wifi_ble_model import (
    JointCandidateSpace,
    JointCandidateState,
    JointSchedulingConfig,
    build_joint_candidate_states,
    build_payload_by_pair,
    ble_data_channel_center_mhz,
    parse_joint_config,
    resolve_joint_objective_policy,
    state_is_idle,
    state_pair_is_feasible,
    summarize_selected_schedule_metrics,
    expand_candidate_blocks,
)

BLE_RESIDUAL_FREQ_GRID_MHZ = tuple(ble_data_channel_center_mhz(channel) for channel in range(37))


def _state_payload_density(state: JointCandidateState, payload_by_pair: Mapping[int, int]) -> float:
    if state_is_idle(state):
        return -1.0
    metrics = summarize_selected_schedule_metrics(
        {
            "macrocycle_slots": 1,
            "wifi_channels": [],
            "ble_channels": [],
            "tasks": [],
        },
        [state],
    )
    area = max(1.0, float(metrics["occupied_area_mhz_slots"]))
    return float(payload_by_pair.get(int(state.pair_id), 0)) / area


def _idle_index_for_pair(pair_indices: Iterable[int], states: list[JointCandidateState]) -> int:
    for idx in pair_indices:
        if state_is_idle(states[idx]):
            return idx
    raise ValueError("Every pair is expected to have an idle state")


def summarize_radio_payloads(
    states: Iterable[JointCandidateState],
    payload_by_pair: Mapping[int, int],
) -> dict[str, float]:
    selected_states = [state for state in states if not state_is_idle(state)]
    wifi_payload_bytes = float(
        sum(payload_by_pair.get(int(state.pair_id), 0) for state in selected_states if state.medium == "wifi")
    )
    scheduled_payload_bytes = float(sum(payload_by_pair.get(int(state.pair_id), 0) for state in selected_states))
    return {
        "wifi_payload_bytes": wifi_payload_bytes,
        "scheduled_payload_bytes": scheduled_payload_bytes,
        "selected_pairs": float(len(selected_states)),
    }


def compare_joint_candidate_scores(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    wifi_payload_floor_bytes: int,
) -> int:
    if left is None and right is None:
        return 0
    if left is None:
        return -1
    if right is None:
        return 1

    left_valid = float(left.get("wifi_payload_bytes", 0.0)) >= float(wifi_payload_floor_bytes)
    right_valid = float(right.get("wifi_payload_bytes", 0.0)) >= float(wifi_payload_floor_bytes)
    if left_valid != right_valid:
        return 1 if left_valid else -1

    left_key = (
        float(left.get("wifi_payload_bytes", 0.0)),
        float(left.get("scheduled_payload_bytes", 0.0)),
        -float(left.get("fill_penalty", 0.0)),
        int(left.get("selected_pairs", 0)),
    )
    right_key = (
        float(right.get("wifi_payload_bytes", 0.0)),
        float(right.get("scheduled_payload_bytes", 0.0)),
        -float(right.get("fill_penalty", 0.0)),
        int(right.get("selected_pairs", 0)),
    )
    return (left_key > right_key) - (left_key < right_key)


def _greedy_seed(
    config: Mapping[str, Any],
    pair_order: list[int],
    candidate_rank_seed: int = 0,
) -> list[int]:
    space = build_joint_candidate_states(config)
    payload_by_pair = build_payload_by_pair(config)
    chosen: list[int] = []
    chosen_states: list[JointCandidateState] = []
    rng = random.Random(candidate_rank_seed)
    pair_to_options = space.pair_to_state_indices
    pair_position = {pair_id: pos for pos, pair_id in enumerate(sorted(pair_to_options))}

    for pair_id in pair_order:
        options = list(pair_to_options[pair_id])
        idle_idx = _idle_index_for_pair(options, space.states)
        active_indices = [idx for idx in options if idx != idle_idx]
        rng.shuffle(active_indices)
        active_indices.sort(
            key=lambda idx: (
                _state_payload_density(space.states[idx], payload_by_pair),
                -float(space.states[idx].offset),
            ),
            reverse=True,
        )
        selected_idx = idle_idx
        for idx in active_indices:
            candidate = space.states[idx]
            feasible = True
            for chosen_state in chosen_states:
                if candidate.pair_id == chosen_state.pair_id:
                    feasible = False
                    break
                from .joint_wifi_ble_model import state_pair_is_feasible

                if not state_pair_is_feasible(candidate, chosen_state):
                    feasible = False
                    break
            if feasible:
                selected_idx = idx
                break
        position = pair_position[pair_id]
        while len(chosen) <= position:
            chosen.append(-1)
        chosen[position] = selected_idx
        chosen_states = [space.states[idx] for idx in chosen if idx >= 0 and not state_is_idle(space.states[idx])]
    return [idx for idx in chosen if idx >= 0]


def _seeded_chromosomes(config: Mapping[str, Any]) -> list[list[int]]:
    space = build_joint_candidate_states(config)
    payload_by_pair = build_payload_by_pair(config)
    pair_ids = sorted(space.pair_to_state_indices)
    order_payload = sorted(pair_ids, key=lambda pair_id: payload_by_pair.get(pair_id, 0), reverse=True)
    order_density = sorted(
        pair_ids,
        key=lambda pair_id: max(
            (_state_payload_density(space.states[idx], payload_by_pair) for idx in space.pair_to_state_indices[pair_id]),
            default=0.0,
        ),
        reverse=True,
    )
    order_release = sorted(
        pair_ids,
        key=lambda pair_id: min(space.states[idx].offset for idx in space.pair_to_state_indices[pair_id]),
    )
    seeds = [
        _greedy_seed(config, order_payload, candidate_rank_seed=11),
        _greedy_seed(config, order_density, candidate_rank_seed=17),
        _greedy_seed(config, order_release, candidate_rank_seed=23),
    ]
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        key = tuple(seed)
        if key in seen:
            continue
        seen.add(key)
        unique.append(seed)
    return unique


def _states_from_result(result: Mapping[str, Any]) -> list[JointCandidateState]:
    return [JointCandidateState(**state) for state in result.get("selected_states", [])]


def _wifi_payload(states: Iterable[JointCandidateState], payload_by_pair: Mapping[int, int]) -> float:
    return float(sum(payload_by_pair.get(int(state.pair_id), 0) for state in states if state.medium == "wifi"))


def _candidate_by_pair(space_states: list[JointCandidateState], selected_states: Iterable[JointCandidateState]) -> dict[int, JointCandidateState]:
    del space_states
    return {int(state.pair_id): state for state in selected_states}


def _residual_freq_grid() -> tuple[float, ...]:
    return BLE_RESIDUAL_FREQ_GRID_MHZ


def _build_wifi_move_seeds(
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    max_seed_count: int = 8,
    max_swap_count: int = 6,
) -> tuple[list[list[int]], int, int]:
    space = build_joint_candidate_states(config)
    selected_states = _states_from_result(result)
    selected_by_pair = _candidate_by_pair(space.states, selected_states)
    selected_wifi_states = [state for state in selected_states if state.medium == "wifi"]
    unscheduled_pairs = {int(pair_id) for pair_id in result.get("unscheduled_pair_ids", [])}
    pair_ids = sorted(space.pair_to_state_indices)
    pair_position = {pair_id: pos for pos, pair_id in enumerate(pair_ids)}
    base = []
    for pair_id in pair_ids:
        options = space.pair_to_state_indices[pair_id]
        idle_idx = _idle_index_for_pair(options, space.states)
        selected_state = selected_by_pair.get(pair_id)
        if selected_state is None:
            base.append(idle_idx)
        else:
            base.append(selected_state.state_id)

    blocker_counts: dict[int, int] = {}
    for pair_id in sorted(unscheduled_pairs):
        active_candidates = [
            space.states[idx]
            for idx in space.pair_to_state_indices.get(pair_id, [])
            if not state_is_idle(space.states[idx]) and space.states[idx].medium == "ble"
        ]
        if not active_candidates:
            continue
        for candidate in active_candidates[:2]:
            blockers = identify_blocking_wifi_pairs(selected_wifi_states, candidate)
            for blocker in blockers:
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1

    residual_seed_count = 0
    wifi_move_repairs_used = 0
    if not blocker_counts:
        return [], 0, 0

    top_blockers = [pair_id for pair_id, _count in sorted(blocker_counts.items(), key=lambda item: item[1], reverse=True)[:2]]
    seeds: list[list[int]] = []
    for blocker_pair_id in top_blockers:
        current_state = selected_by_pair.get(blocker_pair_id)
        if current_state is None or current_state.medium != "wifi":
            continue
        alternatives = [space.states[idx] for idx in space.pair_to_state_indices[blocker_pair_id]]
        unscheduled_ble_candidates = []
        for idle_pair_id in sorted(unscheduled_pairs):
            unscheduled_ble_candidates.extend(
                [
                    space.states[idx]
                    for idx in space.pair_to_state_indices[idle_pair_id]
                    if not state_is_idle(space.states[idx]) and space.states[idx].medium == "ble"
                    ]
                )
        reshuffle_candidates = build_wifi_state_move_candidates(
            current_state,
            alternatives,
            selected_states=selected_states,
            max_candidates=4,
        )
        residual_holes = extract_residual_holes(
            selected_states=selected_states,
            macrocycle_slots=int(result.get("macrocycle_slots", max((int(getattr(state, "macrocycle_slots", 0) or 0) for state in selected_states), default=64))),
            freq_grid_mhz=_residual_freq_grid(),
        )
        ranked_swaps: list[dict[str, Any]] = []
        for replacement in reshuffle_candidates:
            local_selection = [
                replacement if state.pair_id == blocker_pair_id else state
                for state in selected_states
                if not (state.pair_id == blocker_pair_id and state_is_idle(replacement))
            ]
            ranked_subsets = rank_ble_subset_replacements(
                selected_ble_states=[state for state in local_selection if state.medium == "ble"],
                candidate_ble_states=unscheduled_ble_candidates,
                residual_holes=residual_holes,
                protected_wifi_states=[state for state in local_selection if state.medium == "wifi"],
                subset_size_limit=2,
            )
            best_subset = ranked_subsets[0] if ranked_subsets else None
            combined_gain = 0.0
            if best_subset is not None:
                combined_gain = float(best_subset["gain"])
            ranked_swaps.append(
                {
                    "replacement_state": replacement,
                    "best_ble_state": best_subset["insert_states"][0] if best_subset else None,
                    "best_ble_subset": best_subset,
                    "combined_gain": combined_gain,
                }
            )
        for ranked in ranked_swaps[: max(1, max_swap_count)]:
            alternative = ranked["replacement_state"]
            seed = list(base)
            seed[pair_position[blocker_pair_id]] = alternative.state_id
            best_subset = ranked.get("best_ble_subset")
            used_repair = False
            if isinstance(best_subset, dict) and best_subset.get("insert_states"):
                used_repair = True
                for removed_state in best_subset.get("remove_states", []):
                    if isinstance(removed_state, JointCandidateState) and int(removed_state.pair_id) in pair_position:
                        seed[pair_position[int(removed_state.pair_id)]] = _idle_index_for_pair(
                            space.pair_to_state_indices[int(removed_state.pair_id)],
                            space.states,
                        )
                for insert_state in best_subset.get("insert_states", []):
                    if isinstance(insert_state, JointCandidateState) and int(insert_state.pair_id) in pair_position:
                        seed[pair_position[int(insert_state.pair_id)]] = int(insert_state.state_id)
            else:
                repair_applied = False
                for idle_pair_id in sorted(unscheduled_pairs):
                    if idle_pair_id not in pair_position:
                        continue
                    active_candidates = [
                        space.states[idx]
                        for idx in space.pair_to_state_indices[idle_pair_id]
                        if not state_is_idle(space.states[idx]) and space.states[idx].medium == "ble"
                    ]
                    if active_candidates:
                        active_candidates = sorted(
                            active_candidates,
                            key=lambda state: score_ble_state_against_residual_holes(
                                state,
                                [
                                    alternative if sel.pair_id == blocker_pair_id else sel
                                    for sel in selected_states
                                ],
                            ),
                            reverse=True,
                        )
                        seed[pair_position[idle_pair_id]] = active_candidates[0].state_id
                        repair_applied = True
                used_repair = repair_applied
            seeds.append(seed)
            residual_seed_count += 1
            if used_repair:
                wifi_move_repairs_used += 1
            if len(seeds) >= max_seed_count:
                return seeds, residual_seed_count, wifi_move_repairs_used
    return seeds, residual_seed_count, wifi_move_repairs_used


def _current_wifi_payload(states: Iterable[JointCandidateState], payload_by_pair: Mapping[int, int]) -> float:
    return float(sum(payload_by_pair.get(int(state.pair_id), 0) for state in states if state.medium == "wifi"))


def _schedule_metrics_for_states(
    states: list[JointCandidateState],
    payload_by_pair: Mapping[int, int],
    macrocycle_slots: int,
) -> dict[str, Any]:
    summary = summarize_radio_payloads(states, payload_by_pair)
    synthetic_tasks = [
        {
            "task_id": int(pair_id),
            "radio": "ble",
            "payload_bytes": int(payload),
            "release_slot": 0,
            "deadline_slot": max(0, macrocycle_slots - 1),
        }
        for pair_id, payload in sorted(payload_by_pair.items())
    ]
    fill_metrics = summarize_selected_schedule_metrics(
        {
            "macrocycle_slots": macrocycle_slots,
            "wifi_channels": [],
            "ble_channels": [],
            "tasks": synthetic_tasks,
        },
        states,
    )
    return {**summary, **fill_metrics}


def _repair_pack_selected_states(
    *,
    selected_states: list[JointCandidateState],
    space: JointCandidateSpace,
    payload_by_pair: Mapping[int, int],
    wifi_payload_floor_bytes: int,
    insert_budget: int,
    swap_budget: int,
) -> tuple[list[JointCandidateState], int, int]:
    current = [state for state in selected_states if not state_is_idle(state)]
    current_by_pair = {int(state.pair_id): state for state in current}
    insertions_used = 0
    swaps_used = 0
    macrocycle_slots = max(
        (int(getattr(state, "macrocycle_slots", 0) or 0) for state in current),
        default=64,
    )

    def current_metrics(states: list[JointCandidateState]) -> dict[str, Any]:
        return _schedule_metrics_for_states(states, payload_by_pair, macrocycle_slots)

    def is_feasible_against_current(candidate: JointCandidateState, selected: list[JointCandidateState]) -> bool:
        for state in selected:
            if not state_pair_is_feasible(candidate, state):
                return False
        return True

    def macrocycle_slots_for(states: list[JointCandidateState]) -> int:
        return max((int(getattr(state, "macrocycle_slots", 0) or 0) for state in states), default=64)

    candidate_ble_states = [
        state
        for state in space.states
        if not state_is_idle(state) and state.medium == "ble" and int(state.pair_id) not in current_by_pair
    ]
    freq_grid_mhz = _residual_freq_grid()

    for _ in range(max(0, insert_budget)):
        residual_holes = extract_residual_holes(
            selected_states=current,
            macrocycle_slots=macrocycle_slots_for(current),
            freq_grid_mhz=freq_grid_mhz,
        )
        candidate_states = rank_ble_insertions_for_holes(
            candidates=candidate_ble_states,
            residual_holes=residual_holes,
            selected_states=current,
        )
        current_metric_snapshot = current_metrics(current)
        accepted = False
        for candidate in candidate_states:
            if not is_feasible_against_current(candidate, current):
                continue
            trial = current + [candidate]
            if _current_wifi_payload(trial, payload_by_pair) < float(wifi_payload_floor_bytes):
                continue
            gain = compare_joint_candidate_scores(
                current_metrics(trial),
                current_metric_snapshot,
                wifi_payload_floor_bytes,
            )
            if gain > 0:
                current = trial
                current_by_pair = {int(state.pair_id): state for state in current}
                insertions_used += 1
                accepted = True
                break
        if not accepted:
            break

    for _ in range(max(0, swap_budget)):
        residual_holes = extract_residual_holes(
            selected_states=current,
            macrocycle_slots=macrocycle_slots_for(current),
            freq_grid_mhz=freq_grid_mhz,
        )
        ranked_replacements = rank_ble_subset_replacements(
            selected_ble_states=[state for state in current if state.medium == "ble"],
            candidate_ble_states=candidate_ble_states,
            residual_holes=residual_holes,
            protected_wifi_states=[state for state in current if state.medium == "wifi"],
            subset_size_limit=2,
            candidate_pool_limit=max(6, min(12, len(candidate_ble_states))),
        )
        if not ranked_replacements:
            break
        current_metric_snapshot = current_metrics(current)
        accepted = False
        for replacement in ranked_replacements[: max(1, swap_budget)]:
            removed = [state for state in replacement["remove_states"] if isinstance(state, JointCandidateState)]
            inserted = [state for state in replacement["insert_states"] if isinstance(state, JointCandidateState)]
            if not inserted:
                continue
            trial = [state for state in current if state not in removed] + inserted
            if any(
                not is_feasible_against_current(candidate, [state for state in trial if state is not candidate])
                for candidate in inserted
            ):
                continue
            if _current_wifi_payload(trial, payload_by_pair) < float(wifi_payload_floor_bytes):
                continue
            gain = compare_joint_candidate_scores(
                current_metrics(trial),
                current_metric_snapshot,
                wifi_payload_floor_bytes,
            )
            if gain > 0:
                current = trial
                current_by_pair = {int(state.pair_id): state for state in current}
                insertions_used += len(inserted)
                swaps_used += len(removed)
                accepted = True
                break
        if not accepted:
            break
    return current, insertions_used, swaps_used


def _accept_wifi_local_moves(
    *,
    selected_states: list[JointCandidateState],
    space: JointCandidateSpace,
    payload_by_pair: Mapping[int, int],
    wifi_payload_floor_bytes: int,
    move_budget: int,
    insert_budget: int,
    swap_budget: int,
) -> tuple[list[JointCandidateState], int]:
    current = [state for state in selected_states if not state_is_idle(state)]
    accepted_moves = 0
    pair_ids = sorted(space.pair_to_state_indices)
    macrocycle_slots = max(
        (int(getattr(state, "macrocycle_slots", 0) or 0) for state in current),
        default=64,
    )

    def current_metrics(states: list[JointCandidateState]) -> dict[str, Any]:
        return _schedule_metrics_for_states(states, payload_by_pair, macrocycle_slots)

    for _ in range(max(0, move_budget)):
        selected_by_pair = _candidate_by_pair(space.states, current)
        selected_wifi_states = [state for state in current if state.medium == "wifi"]
        if not selected_wifi_states:
            break
        unscheduled_ble_candidates = [
            state
            for state in space.states
            if not state_is_idle(state)
            and state.medium == "ble"
            and int(state.pair_id) not in {int(state.pair_id) for state in current}
        ]
        blocker_counts: dict[int, int] = {}
        for candidate in unscheduled_ble_candidates:
            for blocker in identify_blocking_wifi_pairs(selected_wifi_states, candidate):
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        if not blocker_counts:
            blocker_counts = {int(state.pair_id): 0 for state in selected_wifi_states}

        baseline_metrics = current_metrics(current)
        best_trial: list[JointCandidateState] | None = None
        best_metrics: dict[str, Any] | None = None
        residual_holes = extract_residual_holes(
            selected_states=current,
            macrocycle_slots=macrocycle_slots,
            freq_grid_mhz=_residual_freq_grid(),
        )

        for blocker_pair_id, _count in sorted(blocker_counts.items(), key=lambda item: item[1], reverse=True):
            blocker_state = selected_by_pair.get(blocker_pair_id)
            if blocker_state is None or blocker_state.medium != "wifi":
                continue
            wifi_alternatives = build_wifi_state_move_candidates(
                blocker_state,
                [space.states[idx] for idx in space.pair_to_state_indices[blocker_pair_id]],
                selected_states=current,
                max_candidates=max(4, move_budget),
            )
            ranked_wifi_moves = rank_wifi_state_moves_for_direct_accept_if_better(
                blocker_state,
                wifi_alternatives,
                residual_holes,
                selected_states=current,
            )
            for ranked_move in ranked_wifi_moves:
                wifi_candidate = ranked_move["replacement_state"]
                trial = [wifi_candidate if state.pair_id == blocker_pair_id else state for state in current]
                repaired_trial, _, _ = _repair_pack_selected_states(
                    selected_states=trial,
                    space=space,
                    payload_by_pair=payload_by_pair,
                    wifi_payload_floor_bytes=wifi_payload_floor_bytes,
                    insert_budget=insert_budget,
                    swap_budget=swap_budget,
                )
                if _current_wifi_payload(repaired_trial, payload_by_pair) < float(wifi_payload_floor_bytes):
                    continue
                repaired_metrics = current_metrics(repaired_trial)
                if compare_joint_candidate_scores(repaired_metrics, baseline_metrics, wifi_payload_floor_bytes) > 0:
                    if best_metrics is None or compare_joint_candidate_scores(
                        repaired_metrics,
                        best_metrics,
                        wifi_payload_floor_bytes,
                    ) > 0:
                        best_trial = repaired_trial
                        best_metrics = repaired_metrics
                if best_trial is not None:
                    break
            if best_trial is not None:
                break

        if best_trial is None:
            break
        current = best_trial
        accepted_moves += 1
    return current, accepted_moves


def _better_result(left: Mapping[str, Any], right: Mapping[str, Any] | None) -> bool:
    if right is None:
        return True
    return compare_joint_candidate_scores(left, right, wifi_payload_floor_bytes=0) > 0


def solve_joint_wifi_ble_hga(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    hga_cfg = dict(cfg.get("hga", {}))
    ga_cfg = dict(cfg.get("ga", {}))
    space = build_joint_candidate_states(cfg)
    payload_by_pair = build_payload_by_pair(cfg)

    seeded_chromosomes = _seeded_chromosomes(cfg)
    ga_cfg["seeded_chromosomes"] = seeded_chromosomes
    if "population_size" in hga_cfg:
        ga_cfg["population_size"] = int(hga_cfg["population_size"])
    if "generations" in hga_cfg:
        ga_cfg["generations"] = int(hga_cfg["generations"])
    if "seed" in hga_cfg:
        ga_cfg["seed"] = int(hga_cfg["seed"])
    cfg["ga"] = ga_cfg

    best_result = solve_joint_wifi_ble_ga(cfg)
    best_states = _states_from_result(best_result)
    wifi_seed_payload = _wifi_payload(best_states, payload_by_pair)
    seed_summary = summarize_radio_payloads(best_states, payload_by_pair)
    objective_cfg = resolve_joint_objective_policy(cfg)
    wifi_payload_floor = max(int(objective_cfg.get("wifi_payload_floor_bytes", 0)), int(seed_summary["wifi_payload_bytes"]))
    rounds = max(1, int(hga_cfg.get("coordination_rounds", 2)))
    residual_seed_budget = max(1, int(hga_cfg.get("residual_seed_budget", 4)))
    residual_swap_budget = max(1, int(hga_cfg.get("residual_swap_budget", 6)))
    repair_insert_budget = max(0, int(hga_cfg.get("repair_insert_budget", residual_seed_budget)))
    repair_swap_budget = max(0, int(hga_cfg.get("repair_swap_budget", residual_swap_budget)))
    wifi_local_move_budget = max(0, int(hga_cfg.get("wifi_local_move_budget", 2)))
    rounds_used = 0
    wifi_move_seed_total = 0
    wifi_move_repairs_total = 0

    for _ in range(rounds):
        rounds_used += 1
        local_seeds, wifi_move_seed_count, wifi_move_repairs_used = _build_wifi_move_seeds(
            cfg,
            best_result,
            max_seed_count=residual_seed_budget,
            max_swap_count=residual_swap_budget,
        )
        wifi_move_seed_total += wifi_move_seed_count
        wifi_move_repairs_total += wifi_move_repairs_used
        if not local_seeds:
            continue
        candidate_cfg = dict(cfg)
        candidate_ga = dict(ga_cfg)
        candidate_ga["seeded_chromosomes"] = seeded_chromosomes + local_seeds
        candidate_cfg["ga"] = candidate_ga
        candidate_result = solve_joint_wifi_ble_ga(candidate_cfg)
        candidate_states = _states_from_result(candidate_result)
        candidate_wifi_payload = _wifi_payload(candidate_states, payload_by_pair)
        candidate_summary = summarize_radio_payloads(candidate_states, payload_by_pair)
        candidate_metrics = {
            **candidate_result,
            **candidate_summary,
            "fill_penalty": float(candidate_result.get("fill_penalty", 0.0)),
        }
        best_metrics = {
            **best_result,
            **seed_summary,
            "fill_penalty": float(best_result.get("fill_penalty", 0.0)),
        }
        if candidate_wifi_payload < wifi_payload_floor:
            continue
        if compare_joint_candidate_scores(candidate_metrics, best_metrics, wifi_payload_floor) > 0:
            best_result = candidate_result
            best_states = candidate_states
            seed_summary = candidate_summary

    final_states = _states_from_result(best_result)
    final_states, accepted_wifi_local_moves = _accept_wifi_local_moves(
        selected_states=final_states,
        space=space,
        payload_by_pair=payload_by_pair,
        wifi_payload_floor_bytes=wifi_payload_floor,
        move_budget=wifi_local_move_budget,
        insert_budget=repair_insert_budget,
        swap_budget=repair_swap_budget,
    )
    final_summary = summarize_radio_payloads(final_states, payload_by_pair)
    repaired_states, repair_insertions_used, repair_swaps_used = _repair_pack_selected_states(
        selected_states=final_states,
        space=space,
        payload_by_pair=payload_by_pair,
        wifi_payload_floor_bytes=wifi_payload_floor,
        insert_budget=repair_insert_budget,
        swap_budget=repair_swap_budget,
    )
    repaired_summary = summarize_radio_payloads(repaired_states, payload_by_pair)
    repaired_metrics = {
        **repaired_summary,
        **summarize_selected_schedule_metrics(cfg, repaired_states),
        "fill_penalty": float(summarize_selected_schedule_metrics(cfg, repaired_states)["fill_penalty"]),
    }
    best_metrics = {
        **final_summary,
        **summarize_selected_schedule_metrics(cfg, final_states),
        "fill_penalty": float(summarize_selected_schedule_metrics(cfg, final_states)["fill_penalty"]),
    }
    if compare_joint_candidate_scores(repaired_metrics, best_metrics, wifi_payload_floor) > 0:
        final_states = repaired_states
        final_summary = repaired_summary
        best_result = {
            **best_result,
            "selected_state_indices": [state.state_id for state in final_states],
            "selected_states": [asdict(state) for state in final_states],
            "unscheduled_pair_ids": [pair_id for pair_id in sorted(space.pair_to_state_indices) if pair_id not in {int(state.pair_id) for state in final_states}],
            "blocks": [asdict(block) for state in final_states for block in expand_candidate_blocks(state)],
            "scheduled_payload_bytes": repaired_summary["scheduled_payload_bytes"],
            "occupied_slot_count": repaired_metrics["occupied_slot_count"],
            "occupied_area_mhz_slots": repaired_metrics["occupied_area_mhz_slots"],
            "fragmentation_penalty": repaired_metrics["fragmentation_penalty"],
            "idle_area_penalty": repaired_metrics["idle_area_penalty"],
            "slot_span_penalty": repaired_metrics["slot_span_penalty"],
            "fill_penalty": repaired_metrics["fill_penalty"],
        }
    final_wifi_payload = _wifi_payload(final_states, payload_by_pair)
    if final_wifi_payload < wifi_payload_floor:
        return {
            **best_result,
            "solver": "hga",
            "status": "wifi_floor_infeasible",
            "search_mode": "unified_joint",
            "coordination_rounds_used": rounds_used,
            "wifi_seed_payload_bytes": wifi_seed_payload,
            "final_wifi_payload_bytes": final_wifi_payload,
            "heuristic_seed_count": len(seeded_chromosomes),
            "candidate_state_count": len(space.states),
            "residual_seed_count": wifi_move_seed_total,
            "wifi_move_seed_count": wifi_move_seed_total,
            "wifi_move_repairs_used": wifi_move_repairs_total,
            "accepted_wifi_local_moves": accepted_wifi_local_moves,
            "repair_insertions_used": 0,
            "repair_swaps_used": 0,
        }
    return {
        **best_result,
        "solver": "hga",
        "search_mode": "unified_joint",
        "coordination_rounds_used": rounds_used,
        "wifi_seed_payload_bytes": wifi_seed_payload,
        "final_wifi_payload_bytes": final_wifi_payload,
        "wifi_payload_bytes": final_summary["wifi_payload_bytes"],
        "selected_pairs": int(final_summary["selected_pairs"]),
        "heuristic_seed_count": len(seeded_chromosomes),
        "candidate_state_count": len(space.states),
        "residual_seed_count": wifi_move_seed_total,
        "wifi_move_seed_count": wifi_move_seed_total,
        "wifi_move_repairs_used": wifi_move_repairs_total,
        "accepted_wifi_local_moves": accepted_wifi_local_moves,
        "repair_insertions_used": repair_insertions_used,
        "repair_swaps_used": repair_swaps_used,
    }
