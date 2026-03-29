from joint_sched.joint_wifi_ble_model import JointCandidateState, selected_schedule_has_no_conflicts
from joint_sched.joint_wifi_ble_sdp import solve_joint_wifi_ble_sdp



def _states_from_result(result):
    return [JointCandidateState(**state) for state in result["selected_states"]]



def test_joint_wifi_ble_sdp_selects_one_state_per_pair_and_returns_blocks():
    config = {
        "macrocycle_slots": 32,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1200,
                "release_slot": 0,
                "deadline_slot": 8,
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

    result = solve_joint_wifi_ble_sdp(config)

    assert result["solver"] == "sdp"
    assert result["status"] in {"optimal", "optimal_inaccurate"}
    assert result["task_count"] == 2
    assert len(result["selected_state_indices"]) == 2
    assert len(result["selected_states"]) == 2
    assert len({state["pair_id"] for state in result["selected_states"]}) == 2
    assert len(result["blocks"]) >= 2
    assert selected_schedule_has_no_conflicts(_states_from_result(result)) is True



def test_joint_sdp_never_selects_overlapping_state_pair():
    config = {
        "macrocycle_slots": 32,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
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

    result = solve_joint_wifi_ble_sdp(config)
    states = _states_from_result(result)
    ble_state = next(state for state in result["selected_states"] if state["medium"] == "ble")

    assert selected_schedule_has_no_conflicts(states) is True
    assert ble_state["offset"] > 0



def test_joint_sdp_uses_idle_state_for_fully_conflicting_instance():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
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

    result = solve_joint_wifi_ble_sdp(config)

    assert result["status"] in {"optimal", "optimal_inaccurate"}
    assert len(result["selected_states"]) == 1
    assert len(result["unscheduled_pair_ids"]) == 1
    assert len(result["blocks"]) == 1


def test_joint_sdp_keeps_higher_payload_when_gap_exceeds_tolerance():
    config = {
        "macrocycle_slots": 16,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
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

    result = solve_joint_wifi_ble_sdp(config)

    assert result["status"] in {"optimal", "optimal_inaccurate"}
    assert {state["pair_id"] for state in result["selected_states"]} == {0}
    assert result["unscheduled_pair_ids"] == [1]


def test_joint_sdp_breaks_payload_tie_with_fill_objective():
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

    result = solve_joint_wifi_ble_sdp(config)

    assert result["scheduled_payload_bytes"] == 2000.0
    assert result["occupied_slot_count"] == 4.0
    state_by_pair = {state["pair_id"]: state for state in result["selected_states"]}
    assert state_by_pair[0]["offset"] == 0
