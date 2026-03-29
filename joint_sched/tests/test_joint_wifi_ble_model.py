from pathlib import Path

from joint_sched import JointSchedulingConfig, JointTaskSpec, load_joint_config
from joint_sched.joint_wifi_ble_model import (
    BLE_ADV_CHANNELS_MHZ,
    BLEPairConfig,
    ExternalBlock,
    JointCandidateState,
    WiFiPairConfig,
    build_joint_candidate_states,
    build_joint_cost_matrix,
    build_joint_forbidden_state_pairs,
    build_state_fill_penalty_vector,
    build_state_utility_vector,
    resolve_joint_objective_policy,
    summarize_selected_schedule_metrics,
    expand_candidate_blocks,
    parse_joint_config,
    state_pair_is_feasible,
)



def test_model_imports_and_config_loader():
    config_path = Path(__file__).resolve().parents[1] / "joint_wifi_ble_demo_config.json"
    payload = load_joint_config(config_path)

    assert payload["macrocycle_slots"] == 64
    assert len(payload["wifi_channels"]) == 3
    assert len(payload["ble_channels"]) == 37

    task = JointTaskSpec(
        task_id=0,
        radio="wifi",
        payload_bytes=1024,
        release_slot=0,
        deadline_slot=32,
        preferred_channel=0,
    )
    cfg = JointSchedulingConfig(
        macrocycle_slots=64,
        wifi_channels=[0, 5, 10],
        ble_channels=list(range(37)),
        tasks=[task],
    )

    assert cfg.tasks[0].radio == "wifi"



def test_joint_candidate_states_share_one_global_index_space():
    payload = {
        "macrocycle_slots": 64,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1200,
                "release_slot": 0,
                "deadline_slot": 16,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
                "max_offsets": 2,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 4,
                "deadline_slot": 24,
                "preferred_channel": 8,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
                "ble_pattern_count": 2,
                "max_offsets": 2,
            },
        ],
    }

    space = build_joint_candidate_states(payload)

    assert len(space.wifi_pairs) == 1
    assert len(space.ble_pairs) == 1
    assert set(space.pair_to_state_indices) == {0, 1}
    assert all(isinstance(idx, int) for idx in space.pair_to_state_indices[0])
    assert {space.states[idx].medium for idx in space.pair_to_state_indices[0]} == {"idle", "wifi"}
    assert {space.states[idx].medium for idx in space.pair_to_state_indices[1]} == {"idle", "ble"}
    assert [state.state_id for state in space.states] == list(range(len(space.states)))


def test_wifi_candidate_blocks_repeat_periodically_across_macrocycle():
    payload = {
        "macrocycle_slots": 32,
        "wifi_channels": [0],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1000,
                "release_slot": 0,
                "deadline_slot": 31,
                "preferred_channel": 0,
                "wifi_tx_slots": 4,
                "wifi_period_slots": 8,
                "repetitions": 3,
                "max_offsets": 1,
            }
        ],
    }

    space = build_joint_candidate_states(payload)
    wifi_state = next(
        space.states[idx]
        for idx in space.pair_to_state_indices[0]
        if space.states[idx].medium == "wifi"
    )

    assert wifi_state.num_events == 3

    blocks = expand_candidate_blocks(wifi_state)

    assert [(block.slot_start, block.slot_end) for block in blocks] == [(0, 4), (8, 12), (16, 20)]
    assert [block.event_index for block in blocks] == [0, 1, 2]


def test_joint_occupancy_expansion_uses_common_block_model_and_excludes_adv_channels():
    wifi_state = JointCandidateState(
        state_id=0,
        pair_id=0,
        medium="wifi",
        offset=3,
        channel=0,
        width_slots=2,
        period_slots=2,
    )
    ble_state = JointCandidateState(
        state_id=1,
        pair_id=1,
        medium="ble",
        offset=4,
        channel=8,
        pattern_id=1,
        ci_slots=8,
        ce_slots=1,
        num_events=3,
    )

    wifi_blocks = expand_candidate_blocks(wifi_state)
    ble_blocks = expand_candidate_blocks(ble_state)

    assert len(wifi_blocks) == 1
    assert wifi_blocks[0].slot_start == 3
    assert wifi_blocks[0].slot_end == 5

    assert len(ble_blocks) == 3
    assert [block.slot_start for block in ble_blocks] == [4, 12, 20]
    assert all((block.freq_low_mhz + block.freq_high_mhz) / 2.0 not in BLE_ADV_CHANNELS_MHZ for block in ble_blocks)


def test_cyclic_wifi_expansion_wraps_across_macrocycle_boundary():
    wifi_state = JointCandidateState(
        state_id=0,
        pair_id=0,
        medium="wifi",
        offset=13,
        channel=0,
        width_slots=5,
        period_slots=16,
        num_events=4,
        cyclic_periodic=True,
        macrocycle_slots=64,
    )

    blocks = expand_candidate_blocks(wifi_state)

    assert [(block.slot_start, block.slot_end) for block in blocks] == [
        (13, 18),
        (29, 34),
        (45, 50),
        (61, 64),
        (0, 2),
    ]



def test_joint_cost_matrix_counts_wifi_wifi_ble_overlap_and_zero_for_disjoint():
    states = [
        JointCandidateState(
            state_id=0,
            pair_id=0,
            medium="wifi",
            offset=0,
            channel=0,
            width_slots=2,
            period_slots=2,
        ),
        JointCandidateState(
            state_id=1,
            pair_id=1,
            medium="wifi",
            offset=1,
            channel=0,
            width_slots=2,
            period_slots=2,
        ),
        JointCandidateState(
            state_id=2,
            pair_id=2,
            medium="ble",
            offset=1,
            channel=5,
            pattern_id=0,
            ci_slots=8,
            ce_slots=1,
            num_events=1,
        ),
        JointCandidateState(
            state_id=3,
            pair_id=3,
            medium="ble",
            offset=10,
            channel=30,
            pattern_id=0,
            ci_slots=8,
            ce_slots=1,
            num_events=1,
        ),
    ]

    matrix = build_joint_cost_matrix(states)

    assert matrix[0][1] > 0.0
    assert matrix[0][2] > 0.0
    assert matrix[2][3] == 0.0
    assert matrix[0][0] == 0.0



def test_state_pair_feasibility_rejects_wifi_ble_overlap():
    wifi_state = JointCandidateState(
        state_id=0,
        pair_id=0,
        medium="wifi",
        offset=0,
        channel=0,
        width_slots=2,
        period_slots=2,
    )
    ble_state = JointCandidateState(
        state_id=1,
        pair_id=1,
        medium="ble",
        offset=1,
        channel=5,
        pattern_id=0,
        ci_slots=8,
        ce_slots=1,
        num_events=1,
    )

    assert state_pair_is_feasible(wifi_state, ble_state) is False



def test_state_pair_feasibility_rejects_wifi_wifi_and_ble_ble_overlap():
    wifi_left = JointCandidateState(
        state_id=0,
        pair_id=0,
        medium="wifi",
        offset=0,
        channel=0,
        width_slots=2,
        period_slots=2,
    )
    wifi_right = JointCandidateState(
        state_id=1,
        pair_id=1,
        medium="wifi",
        offset=1,
        channel=0,
        width_slots=2,
        period_slots=2,
    )
    ble_left = JointCandidateState(
        state_id=2,
        pair_id=2,
        medium="ble",
        offset=4,
        channel=8,
        pattern_id=0,
        ci_slots=8,
        ce_slots=2,
        num_events=1,
    )
    ble_right = JointCandidateState(
        state_id=3,
        pair_id=3,
        medium="ble",
        offset=5,
        channel=8,
        pattern_id=0,
        ci_slots=8,
        ce_slots=2,
        num_events=1,
    )

    assert state_pair_is_feasible(wifi_left, wifi_right) is False
    assert state_pair_is_feasible(ble_left, ble_right) is False



def test_state_pair_feasibility_accepts_disjoint_pairs_and_builds_forbidden_pairs():
    states = [
        JointCandidateState(
            state_id=0,
            pair_id=0,
            medium="wifi",
            offset=0,
            channel=0,
            width_slots=2,
            period_slots=2,
        ),
        JointCandidateState(
            state_id=1,
            pair_id=1,
            medium="wifi",
            offset=6,
            channel=5,
            width_slots=2,
            period_slots=2,
        ),
        JointCandidateState(
            state_id=2,
            pair_id=2,
            medium="ble",
            offset=1,
            channel=5,
            pattern_id=0,
            ci_slots=8,
            ce_slots=1,
            num_events=1,
        ),
    ]

    assert state_pair_is_feasible(states[0], states[1]) is True
    forbidden_pairs = build_joint_forbidden_state_pairs(states)
    assert (0, 2) in forbidden_pairs
    assert (0, 1) not in forbidden_pairs



def test_model_dataclasses_exist_for_joint_experiment():
    assert WiFiPairConfig.__name__ == "WiFiPairConfig"
    assert BLEPairConfig.__name__ == "BLEPairConfig"
    assert ExternalBlock.__name__ == "ExternalBlock"

    cfg = parse_joint_config(
        {
            "macrocycle_slots": 16,
            "wifi_channels": [0, 5, 10],
            "ble_channels": list(range(37)),
            "tasks": [],
        }
    )
    assert cfg.macrocycle_slots == 16


def test_idle_state_has_zero_utility():
    states = [JointCandidateState(state_id=0, pair_id=0, medium="idle", offset=0)]

    utility = build_state_utility_vector(states, payload_by_pair={0: 1024})

    assert utility.tolist() == [0.0]


def test_wifi_and_ble_states_get_positive_utility_from_payload():
    states = [
        JointCandidateState(state_id=0, pair_id=0, medium="wifi", offset=0, channel=0, width_slots=2, period_slots=2),
        JointCandidateState(state_id=1, pair_id=1, medium="ble", offset=0, channel=5, pattern_id=0, ci_slots=8, ce_slots=1, num_events=1),
    ]

    utility = build_state_utility_vector(states, payload_by_pair={0: 1500, 1: 247})

    assert utility[0] > 0.0
    assert utility[1] > 0.0


def test_wider_wifi_state_is_penalized_more_when_payload_is_equal():
    states = [
        JointCandidateState(state_id=0, pair_id=0, medium="wifi", offset=0, channel=0, width_slots=2, period_slots=2),
        JointCandidateState(state_id=1, pair_id=1, medium="wifi", offset=0, channel=0, width_slots=4, period_slots=4),
    ]

    utility = build_state_utility_vector(states, payload_by_pair={0: 1200, 1: 1200})

    assert utility[0] > utility[1]


def test_resolve_objective_policy_defaults_to_payload_then_fill():
    objective = resolve_joint_objective_policy({})

    assert objective["mode"] == "lexicographic"
    assert objective["primary"] == "payload"
    assert objective["secondary"] == "fill"
    assert objective["payload_tie_tolerance"] >= 0


def test_objective_policy_accepts_fill_penalties():
    objective = resolve_joint_objective_policy(
        {
            "objective": {
                "mode": "lexicographic",
                "primary": "payload",
                "secondary": "fill",
                "payload_tie_tolerance": 64,
                "fragmentation_penalty": 1.0,
                "idle_area_penalty": 0.5,
                "slot_span_penalty": 0.1,
            }
        }
    )

    assert objective["payload_tie_tolerance"] == 64
    assert objective["fragmentation_penalty"] == 1.0
    assert objective["idle_area_penalty"] == 0.5


def test_state_fill_metrics_penalize_wider_and_more_fragmented_states():
    states = [
        JointCandidateState(state_id=0, pair_id=0, medium="wifi", offset=0, channel=0, width_slots=2, period_slots=2),
        JointCandidateState(state_id=1, pair_id=1, medium="ble", offset=0, channel=5, pattern_id=0, ci_slots=8, ce_slots=1, num_events=2),
    ]

    penalties = build_state_fill_penalty_vector(states)

    assert penalties[1] > penalties[0]


def test_selected_schedule_metrics_report_payload_and_idle_area():
    payload = {
        "macrocycle_slots": 32,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1500,
                "release_slot": 0,
                "deadline_slot": 16,
                "preferred_channel": 0,
                "wifi_tx_slots": 2,
            },
            {
                "task_id": 1,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 8,
                "deadline_slot": 31,
                "preferred_channel": 8,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 2,
            },
        ],
    }
    states = [
        JointCandidateState(state_id=0, pair_id=0, medium="wifi", offset=0, channel=0, width_slots=2, period_slots=2),
        JointCandidateState(state_id=1, pair_id=1, medium="ble", offset=8, channel=8, pattern_id=0, ci_slots=8, ce_slots=1, num_events=2),
    ]

    metrics = summarize_selected_schedule_metrics(payload, states)

    assert metrics["scheduled_payload_bytes"] == 1747.0
    assert metrics["idle_area_penalty"] >= 0.0
    assert metrics["fragmentation_penalty"] >= 0.0
