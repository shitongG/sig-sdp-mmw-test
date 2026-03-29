from joint_sched.joint_wifi_ble_hga import _accept_wifi_local_moves
from joint_sched.joint_wifi_ble_hga import _build_wifi_move_seeds
from joint_sched.joint_wifi_ble_hga import _repair_pack_selected_states
from joint_sched.joint_wifi_ble_hga import solve_joint_wifi_ble_hga
from joint_sched.joint_wifi_ble_hga_model import (
    rank_ble_candidates_for_residual_hole,
    rank_residual_candidate_swaps,
    score_residual_hole_fit,
)
from joint_sched.joint_wifi_ble_model import JointCandidateState, build_joint_candidate_states, selected_schedule_has_no_conflicts


def _hga_test_config() -> dict:
    return {
        "macrocycle_slots": 16,
        "wifi_channels": [0, 10],
        "ble_channels": list(range(37)),
        "ga": {
            "population_size": 10,
            "generations": 4,
            "seed": 3,
        },
        "hga": {
            "population_size": 12,
            "generations": 6,
            "coordination_rounds": 3,
            "seed": 5,
        },
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1200,
                "release_slot": 0,
                "deadline_slot": 3,
                "wifi_tx_slots": 4,
                "max_offsets": 1,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 0,
                "preferred_channel": 0,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
        ],
    }


def _states_from_result(result: dict) -> list[JointCandidateState]:
    return [JointCandidateState(**state) for state in result["selected_states"]]


def test_joint_hga_returns_collision_free_schedule():
    result = solve_joint_wifi_ble_hga(_hga_test_config())

    assert result["solver"] == "hga"
    assert result["status"] == "ok"
    assert result["scheduled_payload_bytes"] > 0
    assert result["coordination_rounds_used"] >= 1
    assert result["search_mode"] == "unified_joint"
    assert selected_schedule_has_no_conflicts(_states_from_result(result))


def test_joint_hga_reshuffles_without_dropping_wifi_payload():
    result = solve_joint_wifi_ble_hga(_hga_test_config())

    assert result["wifi_seed_payload_bytes"] == result["final_wifi_payload_bytes"]
    assert result["scheduled_payload_bytes"] >= result["wifi_seed_payload_bytes"]


def test_joint_hga_reports_residual_seed_usage():
    result = solve_joint_wifi_ble_hga(_hga_test_config())

    assert result["search_mode"] == "unified_joint"
    assert result["residual_seed_count"] >= 0
    assert result["wifi_move_seed_count"] >= 0
    assert result["wifi_move_repairs_used"] >= 0
    assert result["accepted_wifi_local_moves"] >= 0
    assert result["wifi_move_seed_count"] == result["residual_seed_count"]


def test_score_residual_hole_fit_prefers_candidate_that_fills_more_of_hole():
    hole = {"slot_start": 10, "slot_end": 14, "freq_low_mhz": 2430.0, "freq_high_mhz": 2432.0}
    tight_fit = {"occupied_area_mhz_slots": 8.0, "overlap_area_mhz_slots": 8.0}
    loose_fit = {"occupied_area_mhz_slots": 4.0, "overlap_area_mhz_slots": 4.0}

    assert score_residual_hole_fit(tight_fit, hole) > score_residual_hole_fit(loose_fit, hole)


def test_rank_ble_candidates_for_residual_hole_prefers_wifi_safe_dense_candidate():
    hole = {"slot_start": 0, "slot_end": 8, "freq_low_mhz": 2450.0, "freq_high_mhz": 2452.0}
    candidates = [
        {"state_index": 1, "wifi_overlap_area": 0.0, "overlap_area_mhz_slots": 6.0},
        {"state_index": 2, "wifi_overlap_area": 2.0, "overlap_area_mhz_slots": 8.0},
    ]

    ranked = rank_ble_candidates_for_residual_hole(candidates, hole)

    assert ranked[0]["state_index"] == 1


def test_rank_residual_candidate_swaps_uses_residual_hole_fit():
    current_wifi = JointCandidateState(
        state_id=10,
        pair_id=0,
        medium="wifi",
        offset=0,
        channel=0,
        period_slots=8,
        width_slots=4,
        num_events=1,
    )
    replacement = JointCandidateState(
        state_id=11,
        pair_id=0,
        medium="wifi",
        offset=8,
        channel=0,
        period_slots=8,
        width_slots=4,
        num_events=1,
    )
    dense_ble = JointCandidateState(
        state_id=20,
        pair_id=1,
        medium="ble",
        offset=0,
        channel=5,
        pattern_id=0,
        ci_slots=8,
        ce_slots=1,
        num_events=1,
    )
    sparse_ble = JointCandidateState(
        state_id=21,
        pair_id=2,
        medium="ble",
        offset=6,
        channel=5,
        pattern_id=0,
        ci_slots=8,
        ce_slots=1,
        num_events=1,
    )

    ranked = rank_residual_candidate_swaps(
        current_selection=[current_wifi],
        blocker_pair_id=0,
        replacement_candidates=[replacement],
        unscheduled_ble_candidates=[dense_ble, sparse_ble],
    )

    assert ranked[0]["best_ble_state"].state_id == 20


def test_wifi_move_seed_builder_counts_whole_wifi_moves_and_repairs():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0, 10],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1200,
                "release_slot": 0,
                "deadline_slot": 3,
                "wifi_tx_slots": 4,
                "max_offsets": 1,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 0,
                "preferred_channel": 0,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
            {
                "task_id": 2,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 0,
                "preferred_channel": 10,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
        ],
    }

    space = build_joint_candidate_states(config)
    selected_states = [
        next(state for state in (space.states[idx] for idx in space.pair_to_state_indices[0]) if state.medium == "wifi" and state.channel == 0),
        next(state for state in (space.states[idx] for idx in space.pair_to_state_indices[2]) if state.medium == "ble"),
    ]
    result = {
        "selected_states": [dict(state.__dict__) for state in selected_states],
        "unscheduled_pair_ids": [1],
        "macrocycle_slots": 16,
    }

    seeds, seed_count, repairs_used = _build_wifi_move_seeds(
        config,
        result,
        max_seed_count=4,
        max_swap_count=4,
    )

    assert seed_count == len(seeds)
    assert seed_count >= 1
    assert repairs_used >= 1


def test_joint_hga_keeps_wifi_payload_at_or_above_seed_floor():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "ga": {
            "population_size": 10,
            "generations": 5,
            "seed": 11,
        },
        "hga": {
            "population_size": 12,
            "generations": 6,
            "coordination_rounds": 3,
            "seed": 5,
        },
        "objective": {
            "mode": "lexicographic",
            "wifi_payload_floor_bytes": 3000,
        },
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1500,
                "release_slot": 0,
                "deadline_slot": 3,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 8,
                "repetitions": 1,
                "max_offsets": 1,
            },
            {
                "task_id": 1,
                "radio": "wifi",
                "payload_bytes": 1500,
                "release_slot": 4,
                "deadline_slot": 7,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 8,
                "repetitions": 1,
                "max_offsets": 1,
            },
            {
                "task_id": 2,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 7,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
        ],
    }

    result = solve_joint_wifi_ble_hga(config)

    assert result["final_wifi_payload_bytes"] >= 3000


def test_repair_pack_selected_states_adds_residual_ble_without_breaking_wifi_floor():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1200,
                "release_slot": 0,
                "deadline_slot": 15,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 8,
                "repetitions": 2,
                "max_offsets": 1,
                "cyclic_periodic": True,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 15,
                "preferred_channel": 30,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
            {
                "task_id": 2,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 8,
                "deadline_slot": 15,
                "preferred_channel": 30,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
        ],
    }

    space = build_joint_candidate_states(config)
    payload_by_pair = {0: 1200, 1: 247, 2: 247}
    selected_states = [
        next(state for state in (space.states[idx] for idx in space.pair_to_state_indices[0]) if state.medium == "wifi"),
        next(state for state in (space.states[idx] for idx in space.pair_to_state_indices[1]) if state.medium == "ble"),
    ]

    repaired_states, insertions_used, swaps_used = _repair_pack_selected_states(
        selected_states=selected_states,
        space=space,
        payload_by_pair=payload_by_pair,
        wifi_payload_floor_bytes=1200,
        insert_budget=4,
        swap_budget=2,
    )

    assert insertions_used >= 1
    assert swaps_used >= 0
    assert any(state.pair_id == 2 for state in repaired_states)
    assert selected_schedule_has_no_conflicts(repaired_states)


def test_repair_pack_selected_states_uses_multi_ble_subset_replacement_when_swap_only():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1200,
                "release_slot": 0,
                "deadline_slot": 15,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 8,
                "repetitions": 2,
                "max_offsets": 1,
                "cyclic_periodic": True,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 2,
                "deadline_slot": 13,
                "preferred_channel": 20,
                "ble_ce_slots": 4,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
            {
                "task_id": 2,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 15,
                "preferred_channel": 20,
                "ble_ce_slots": 4,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
            {
                "task_id": 3,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 4,
                "deadline_slot": 15,
                "preferred_channel": 20,
                "ble_ce_slots": 4,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
        ],
    }

    space = build_joint_candidate_states(config)
    payload_by_pair = {0: 1200, 1: 247, 2: 247, 3: 247}
    wifi_state = next(state for state in (space.states[idx] for idx in space.pair_to_state_indices[0]) if state.medium == "wifi")
    blocker_ble_state = next(state for state in (space.states[idx] for idx in space.pair_to_state_indices[1]) if state.medium == "ble")

    repaired_states, insertions_used, swaps_used = _repair_pack_selected_states(
        selected_states=[wifi_state, blocker_ble_state],
        space=space,
        payload_by_pair=payload_by_pair,
        wifi_payload_floor_bytes=1200,
        insert_budget=0,
        swap_budget=1,
    )

    repaired_pair_ids = {state.pair_id for state in repaired_states}
    assert insertions_used >= 2
    assert swaps_used >= 1
    assert 1 not in repaired_pair_ids
    assert {2, 3}.issubset(repaired_pair_ids)
    assert selected_schedule_has_no_conflicts(repaired_states)


def test_accept_wifi_local_moves_repacks_ble_and_reports_accepted_move(monkeypatch):
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1200,
                "release_slot": 0,
                "deadline_slot": 15,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 8,
                "repetitions": 2,
                "max_offsets": 3,
                "cyclic_periodic": True,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 2,
                "deadline_slot": 13,
                "preferred_channel": 20,
                "ble_ce_slots": 4,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
            {
                "task_id": 2,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 15,
                "preferred_channel": 20,
                "ble_ce_slots": 4,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
            {
                "task_id": 3,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 4,
                "deadline_slot": 15,
                "preferred_channel": 20,
                "ble_ce_slots": 4,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
        ],
    }

    space = build_joint_candidate_states(config)
    payload_by_pair = {0: 1200, 1: 247, 2: 247, 3: 247}
    wifi_state = next(
        state
        for state in (space.states[idx] for idx in space.pair_to_state_indices[0])
        if state.medium == "wifi" and state.offset == 8
    )
    blocker_ble_state = next(state for state in (space.states[idx] for idx in space.pair_to_state_indices[1]) if state.medium == "ble")

    wifi_zero_state = next(
        state
        for state in (space.states[idx] for idx in space.pair_to_state_indices[0])
        if state.medium == "wifi" and state.offset == 0
    )
    inserted_ble_states = [
        next(state for state in (space.states[idx] for idx in space.pair_to_state_indices[2]) if state.medium == "ble"),
        next(state for state in (space.states[idx] for idx in space.pair_to_state_indices[3]) if state.medium == "ble"),
    ]

    def fake_identify_blocking_wifi_pairs(_wifi_states, _ble_state):
        return {0}

    def fake_repair_pack_selected_states(
        *,
        selected_states,
        space,
        payload_by_pair,
        wifi_payload_floor_bytes,
        insert_budget,
        swap_budget,
    ):
        del space, payload_by_pair, wifi_payload_floor_bytes, insert_budget, swap_budget
        wifi_candidate = next(state for state in selected_states if state.medium == "wifi")
        if wifi_candidate.offset == 0:
            return [wifi_zero_state, *inserted_ble_states], 2, 1
        return list(selected_states), 0, 0

    monkeypatch.setattr("joint_sched.joint_wifi_ble_hga.identify_blocking_wifi_pairs", fake_identify_blocking_wifi_pairs)
    monkeypatch.setattr("joint_sched.joint_wifi_ble_hga._repair_pack_selected_states", fake_repair_pack_selected_states)

    repaired_states, accepted_wifi_local_moves = _accept_wifi_local_moves(
        selected_states=[wifi_state, blocker_ble_state],
        space=space,
        payload_by_pair=payload_by_pair,
        wifi_payload_floor_bytes=1200,
        move_budget=1,
        insert_budget=0,
        swap_budget=1,
    )

    repaired_pair_ids = {state.pair_id for state in repaired_states}
    assert accepted_wifi_local_moves >= 1
    assert 1 not in repaired_pair_ids
    assert {2, 3}.issubset(repaired_pair_ids)
    assert selected_schedule_has_no_conflicts(repaired_states)
