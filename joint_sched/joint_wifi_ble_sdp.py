"""Joint SDP backend for the isolated joint WiFi/BLE experiment."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import cvxpy as cp
import numpy as np

from .joint_wifi_ble_model import (
    JointCandidateSpace,
    JointCandidateState,
    build_joint_candidate_states,
    build_joint_cost_matrix,
    build_joint_forbidden_state_pairs,
    build_payload_by_pair,
    build_state_fill_penalty_vector,
    build_state_utility_vector,
    expand_candidate_blocks,
    resolve_joint_objective_policy,
    selected_schedule_has_no_conflicts,
    state_is_idle,
    summarize_selected_schedule_metrics,
)


def _empty_result(pair_count: int, state_count: int, status: str) -> dict[str, Any]:
    return {
        "solver": "sdp",
        "status": status,
        "task_count": pair_count,
        "state_count": state_count,
        "selected_state_indices": [],
        "selected_states": [],
        "unscheduled_pair_ids": [],
        "blocks": [],
        "objective_value": 0.0,
        "diag_values": [],
        "scheduled_payload_bytes": 0.0,
        "occupied_slot_count": 0.0,
        "occupied_area_mhz_slots": 0.0,
        "fragmentation_penalty": 0.0,
        "idle_area_penalty": 0.0,
        "slot_span_penalty": 0.0,
        "fill_penalty": 0.0,
    }


def round_joint_solution(
    space: JointCandidateSpace,
    diag_values: np.ndarray,
    forbidden_pairs: set[tuple[int, int]],
) -> tuple[list[int], list[JointCandidateState], list[int]] | None:
    pair_ids = sorted(space.pair_to_state_indices, key=lambda pair_id: (len(space.pair_to_state_indices[pair_id]), pair_id))
    selected_indices: list[int] = []

    def is_feasible_choice(candidate_idx: int) -> bool:
        for chosen_idx in selected_indices:
            pair = (min(candidate_idx, chosen_idx), max(candidate_idx, chosen_idx))
            if pair in forbidden_pairs:
                return False
        return True

    def candidate_sort_key(idx: int) -> tuple[int, float]:
        state = space.states[idx]
        return (0 if state_is_idle(state) else 1, float(diag_values[idx]))

    def backtrack(position: int) -> bool:
        if position == len(pair_ids):
            return True
        pair_id = pair_ids[position]
        candidates = sorted(space.pair_to_state_indices[pair_id], key=candidate_sort_key, reverse=True)
        for candidate_idx in candidates:
            if not is_feasible_choice(candidate_idx):
                continue
            selected_indices.append(candidate_idx)
            if backtrack(position + 1):
                return True
            selected_indices.pop()
        return False

    if not backtrack(0):
        return None
    all_selected_states = [space.states[idx] for idx in selected_indices]
    scheduled_states = [state for state in all_selected_states if not state_is_idle(state)]
    if not selected_schedule_has_no_conflicts(scheduled_states):
        return None
    unscheduled_pair_ids = [state.pair_id for state in all_selected_states if state_is_idle(state)]
    return selected_indices, scheduled_states, unscheduled_pair_ids


def solve_joint_wifi_ble_sdp(config: Mapping[str, Any]) -> dict[str, Any]:
    """Solve the isolated joint WiFi/BLE scheduling problem with an SDP relaxation."""

    space = build_joint_candidate_states(config)
    state_count = len(space.states)
    pair_count = len(space.pair_to_state_indices)

    if state_count == 0:
        return _empty_result(pair_count, 0, "empty")

    omega = np.asarray(build_joint_cost_matrix(space.states), dtype=float)
    upper = np.triu(omega, k=1)
    forbidden_pairs = build_joint_forbidden_state_pairs(space.states)
    payload_by_pair = build_payload_by_pair(config)
    objective_policy = resolve_joint_objective_policy(config)
    utility = build_state_utility_vector(space.states, payload_by_pair=payload_by_pair, objective=config)
    fill_penalties = build_state_fill_penalty_vector(space.states, objective=config)
    payload_vector = np.asarray([0.0 if state_is_idle(state) else float(payload_by_pair.get(int(state.pair_id), 0)) for state in space.states], dtype=float)
    Y = cp.Variable((state_count, state_count), PSD=True)
    diag_y = cp.diag(Y)

    constraints: list[Any] = [diag_y >= 0, diag_y <= 1]
    for indices in space.pair_to_state_indices.values():
        constraints.append(cp.sum(diag_y[indices]) == 1)

    for i in range(state_count):
        constraints.append(Y[i, i] == diag_y[i])
        for j in range(i + 1, state_count):
            constraints.append(Y[i, j] <= diag_y[i])
            constraints.append(Y[i, j] <= diag_y[j])
            constraints.append(Y[i, j] >= diag_y[i] + diag_y[j] - 1)
            constraints.append(Y[i, j] >= 0)
            constraints.append(Y[j, i] == Y[i, j])
            if (i, j) in forbidden_pairs:
                constraints.append(Y[i, j] == 0)

    payload_expr = cp.sum(cp.multiply(payload_vector, diag_y))
    soft_cost_expr = cp.sum(cp.multiply(upper, Y))

    if str(objective_policy.get("mode")) == "utility":
        objective = cp.Minimize(soft_cost_expr - cp.sum(cp.multiply(utility, diag_y)))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)
        final_problem = problem
    else:
        stage_one = cp.Problem(cp.Maximize(payload_expr), constraints)
        stage_one.solve(solver=cp.SCS, verbose=False)
        if stage_one.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE} or payload_expr.value is None:
            return _empty_result(pair_count, state_count, "infeasible")
        payload_star = float(payload_expr.value)
        tolerance = float(objective_policy.get("payload_tie_tolerance", 0.0))
        stage_two_constraints = list(constraints) + [payload_expr >= payload_star - tolerance]
        stage_two = cp.Problem(cp.Minimize(cp.sum(cp.multiply(fill_penalties, diag_y)) + soft_cost_expr), stage_two_constraints)
        stage_two.solve(solver=cp.SCS, verbose=False)
        final_problem = stage_two

    if final_problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
        return _empty_result(pair_count, state_count, "infeasible")
    if diag_y.value is None:
        return _empty_result(pair_count, state_count, str(final_problem.status))

    diag_values = np.asarray(diag_y.value, dtype=float).reshape(-1)
    rounded = round_joint_solution(space, diag_values, forbidden_pairs)
    if rounded is None:
        return _empty_result(pair_count, state_count, "infeasible")
    selected_indices, selected_states, unscheduled_pair_ids = rounded
    blocks = [block for state in selected_states for block in expand_candidate_blocks(state)]

    return {
        "solver": "sdp",
        "status": str(final_problem.status),
        "task_count": pair_count,
        "state_count": state_count,
        "selected_state_indices": selected_indices,
        "selected_states": [asdict(state) for state in selected_states],
        "unscheduled_pair_ids": unscheduled_pair_ids,
        "blocks": [asdict(block) for block in blocks],
        "objective_value": float(final_problem.value) if final_problem.value is not None else 0.0,
        "diag_values": diag_values.tolist(),
        **summarize_selected_schedule_metrics(config, selected_states),
    }
