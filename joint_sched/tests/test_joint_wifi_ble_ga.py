from joint_sched.joint_wifi_ble_ga import (
    compare_joint_candidate_scores,
    summarize_radio_payloads,
    solve_joint_wifi_ble_ga,
)
from joint_sched.joint_wifi_ble_model import JointCandidateState, selected_schedule_has_no_conflicts



def _states_from_result(result):
    return [JointCandidateState(**state) for state in result["selected_states"]]


def test_compare_joint_candidate_scores_rejects_lower_wifi_payload():
    baseline = {
        "wifi_payload_bytes": 2000,
        "scheduled_payload_bytes": 6000,
        "fill_penalty": 100.0,
        "selected_pairs": 10,
    }
    candidate = {
        "wifi_payload_bytes": 1800,
        "scheduled_payload_bytes": 7000,
        "fill_penalty": 90.0,
        "selected_pairs": 11,
    }

    assert compare_joint_candidate_scores(candidate, baseline, wifi_payload_floor=2000) < 0


def test_compare_joint_candidate_scores_prefers_more_total_payload_when_wifi_floor_equal():
    left = {
        "wifi_payload_bytes": 2000,
        "scheduled_payload_bytes": 6400,
        "fill_penalty": 150.0,
        "selected_pairs": 10,
    }
    right = {
        "wifi_payload_bytes": 2000,
        "scheduled_payload_bytes": 6200,
        "fill_penalty": 80.0,
        "selected_pairs": 10,
    }

    assert compare_joint_candidate_scores(left, right, wifi_payload_floor=2000) > 0


def test_compare_joint_candidate_scores_prefers_lower_fill_penalty_before_pair_count():
    left = {
        "wifi_payload_bytes": 2000,
        "scheduled_payload_bytes": 6400,
        "fill_penalty": 50.0,
        "selected_pairs": 8,
    }
    right = {
        "wifi_payload_bytes": 2000,
        "scheduled_payload_bytes": 6400,
        "fill_penalty": 80.0,
        "selected_pairs": 9,
    }

    assert compare_joint_candidate_scores(left, right, wifi_payload_floor=2000) > 0


def test_summarize_radio_payloads_splits_wifi_and_ble_payloads():
    selected_states = [
        {"pair_id": 0, "medium": "wifi"},
        {"pair_id": 1, "medium": "ble"},
    ]
    task_payloads = {
        0: {"radio": "wifi", "payload_bytes": 1500},
        1: {"radio": "ble", "payload_bytes": 247},
    }

    summary = summarize_radio_payloads(selected_states, task_payloads)

    assert summary["wifi_payload_bytes"] == 1500
    assert summary["ble_payload_bytes"] == 247



def test_joint_wifi_ble_ga_returns_mixed_schedule_and_history():
    config = {
        "macrocycle_slots": 32,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "ga": {
            "population_size": 20,
            "generations": 12,
            "seed": 11,
        },
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1024,
                "release_slot": 0,
                "deadline_slot": 8,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 2,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 20,
                "preferred_channel": 5,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 2,
                "max_offsets": 2,
            },
        ],
    }

    result = solve_joint_wifi_ble_ga(config)

    assert result["solver"] == "ga"
    assert result["status"] == "ok"
    assert len(result["selected_states"]) == 2
    assert len(result["fitness_history"]) == 12
    assert len({state["pair_id"] for state in result["selected_states"]}) == 2
    assert selected_schedule_has_no_conflicts(_states_from_result(result)) is True



def test_joint_wifi_ble_ga_is_reproducible_with_seed():
    config = {
        "macrocycle_slots": 32,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "ga": {
            "population_size": 18,
            "generations": 10,
            "seed": 5,
        },
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1024,
                "release_slot": 0,
                "deadline_slot": 4,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 1,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 20,
                "preferred_channel": 5,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 2,
            },
        ],
    }

    left = solve_joint_wifi_ble_ga(config)
    right = solve_joint_wifi_ble_ga(config)

    assert left["selected_state_indices"] == right["selected_state_indices"]
    assert left["fitness_history"] == right["fitness_history"]


def test_joint_ga_respects_wifi_payload_floor():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "ga": {"population_size": 8, "generations": 4, "seed": 7},
        "objective": {"mode": "lexicographic", "wifi_payload_floor_bytes": 1500},
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1500,
                "release_slot": 0,
                "deadline_slot": 15,
                "preferred_channel": 0,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 8,
                "wifi_num_events": 2,
                "max_offsets": 2,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 15,
                "preferred_channel": 5,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 1,
                "max_offsets": 2,
            },
        ],
    }

    result = solve_joint_wifi_ble_ga(config)

    assert result["status"] == "ok"
    assert result["wifi_payload_bytes"] >= 1500



def test_joint_ga_uses_idle_state_for_fully_conflicting_instance():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "ga": {
            "population_size": 12,
            "generations": 8,
            "seed": 9,
        },
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1024,
                "release_slot": 0,
                "deadline_slot": 1,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 1,
            },
            {
                "task_id": 1,
                "radio": "wifi",
                "payload_bytes": 1024,
                "release_slot": 0,
                "deadline_slot": 1,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 1,
            },
        ],
    }

    result = solve_joint_wifi_ble_ga(config)

    assert result["status"] == "ok"
    assert len(result["selected_states"]) == 1
    assert len(result["unscheduled_pair_ids"]) == 1
    assert len(result["blocks"]) == 1


def test_joint_ga_keeps_higher_payload_when_gap_exceeds_tolerance():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "ga": {
            "population_size": 24,
            "generations": 16,
            "seed": 13,
        },
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1500,
                "release_slot": 0,
                "deadline_slot": 1,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 1,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 50,
                "release_slot": 0,
                "deadline_slot": 0,
                "preferred_channel": 5,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 1,
                "ble_pattern_count": 1,
                "max_offsets": 1,
            },
        ],
    }

    result = solve_joint_wifi_ble_ga(config)

    assert result["status"] == "ok"
    assert {state["pair_id"] for state in result["selected_states"]} == {0}
    assert result["unscheduled_pair_ids"] == [1]


def test_joint_ga_uses_fill_penalty_when_payload_is_tied():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "objective": {
            "mode": "lexicographic",
            "payload_tie_tolerance": 0,
            "fragmentation_penalty": 1.0,
            "idle_area_penalty": 1.0,
            "slot_span_penalty": 1.0,
        },
        "ga": {
            "population_size": 24,
            "generations": 20,
            "seed": 13,
        },
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1000,
                "release_slot": 0,
                "deadline_slot": 5,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 2,
            },
            {
                "task_id": 1,
                "radio": "wifi",
                "payload_bytes": 1000,
                "release_slot": 2,
                "deadline_slot": 3,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 1,
            },
        ],
    }

    result = solve_joint_wifi_ble_ga(config)

    assert result["scheduled_payload_bytes"] == 2000.0
    assert result["fill_penalty"] == 4.0
