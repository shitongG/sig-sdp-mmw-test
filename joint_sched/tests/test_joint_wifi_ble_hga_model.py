from joint_sched.joint_wifi_ble_hga_model import (
    WIFI_STRIPE_COUNT,
    build_wifi_state_move_candidates,
    build_wifi_local_reshuffle_candidates,
    compute_hole_capacity,
    extract_residual_holes,
    expand_wifi_state_to_stripes,
    identify_blocking_wifi_pairs,
    rank_ble_insertions_for_holes,
    rank_ble_subset_replacements,
    rank_wifi_state_moves_for_ble_holes,
    rank_wifi_state_moves_for_direct_accept_if_better,
    rank_residual_candidate_swaps,
    score_ble_state_against_residual_holes,
)
from joint_sched.joint_wifi_ble_model import JointCandidateState


def test_expand_wifi_state_to_stripes_returns_contiguous_2mhz_blocks():
    state = JointCandidateState(
        state_id=3,
        pair_id=0,
        medium="wifi",
        offset=4,
        channel=0,
        width_slots=4,
        period_slots=16,
    )

    stripes = expand_wifi_state_to_stripes(state)

    assert len(stripes) == WIFI_STRIPE_COUNT
    assert stripes[0].freq_low_mhz == 2402.0
    assert stripes[-1].freq_high_mhz == 2422.0


def test_wifi_stripe_blocks_share_same_offset_and_slot_span():
    state = JointCandidateState(
        state_id=4,
        pair_id=0,
        medium="wifi",
        offset=4,
        channel=5,
        width_slots=4,
        period_slots=16,
    )

    stripes = expand_wifi_state_to_stripes(state)

    assert {block.slot_start for block in stripes} == {4}
    assert {block.slot_end for block in stripes} == {8}


def test_identify_blocking_wifi_pairs_returns_pairs_that_overlap_unscheduled_ble():
    wifi_states = [
        JointCandidateState(
            state_id=0,
            pair_id=0,
            medium="wifi",
            offset=0,
            channel=0,
            width_slots=4,
            period_slots=16,
        ),
        JointCandidateState(
            state_id=1,
            pair_id=4,
            medium="wifi",
            offset=0,
            channel=5,
            width_slots=4,
            period_slots=16,
        ),
    ]
    ble_state = JointCandidateState(
        state_id=2,
        pair_id=9,
        medium="ble",
        offset=1,
        channel=5,
        pattern_id=0,
        ci_slots=8,
        ce_slots=1,
        num_events=1,
    )

    blockers = identify_blocking_wifi_pairs(wifi_states, ble_state)

    assert blockers == {0}


def test_build_wifi_local_reshuffle_candidates_excludes_current_offset():
    current_state = JointCandidateState(
        state_id=0,
        pair_id=0,
        medium="wifi",
        offset=4,
        channel=0,
        width_slots=4,
        period_slots=16,
    )
    alternatives = [
        current_state,
        JointCandidateState(
            state_id=1,
            pair_id=0,
            medium="wifi",
            offset=8,
            channel=0,
            width_slots=4,
            period_slots=16,
        ),
        JointCandidateState(
            state_id=2,
            pair_id=0,
            medium="wifi",
            offset=12,
            channel=5,
            width_slots=4,
            period_slots=16,
        ),
    ]

    candidates = build_wifi_local_reshuffle_candidates(current_state, alternatives, max_candidates=4)

    assert candidates
    assert all(candidate.offset != current_state.offset or candidate.channel != current_state.channel for candidate in candidates)


def test_build_wifi_state_move_candidates_keeps_only_real_same_pair_alternatives():
    current_state = JointCandidateState(
        state_id=0,
        pair_id=4,
        medium="wifi",
        offset=4,
        channel=0,
        width_slots=4,
        period_slots=16,
    )
    candidate_pool = [
        current_state,
        JointCandidateState(
            state_id=1,
            pair_id=4,
            medium="wifi",
            offset=8,
            channel=0,
            width_slots=4,
            period_slots=16,
        ),
        JointCandidateState(
            state_id=2,
            pair_id=4,
            medium="wifi",
            offset=12,
            channel=10,
            width_slots=4,
            period_slots=16,
        ),
        JointCandidateState(
            state_id=3,
            pair_id=9,
            medium="wifi",
            offset=8,
            channel=10,
            width_slots=4,
            period_slots=16,
        ),
    ]

    candidates = build_wifi_state_move_candidates(current_state, candidate_pool, max_candidates=4)

    assert [candidate.state_id for candidate in candidates] == [1, 2]


def test_rank_wifi_state_moves_for_ble_holes_prefers_move_that_leaves_more_hole_capacity():
    current_state = JointCandidateState(
        state_id=0,
        pair_id=4,
        medium="wifi",
        offset=4,
        channel=0,
        width_slots=4,
        period_slots=16,
    )
    keep_hole_open = JointCandidateState(
        state_id=1,
        pair_id=4,
        medium="wifi",
        offset=8,
        channel=10,
        width_slots=4,
        period_slots=16,
    )
    block_hole = JointCandidateState(
        state_id=2,
        pair_id=4,
        medium="wifi",
        offset=8,
        channel=0,
        width_slots=4,
        period_slots=16,
    )
    ranked = rank_wifi_state_moves_for_ble_holes(
        current_state=current_state,
        candidate_wifi_states=[block_hole, keep_hole_open],
        residual_holes=[
            {
                "slot_start": 8.0,
                "slot_end": 12.0,
                "freq_low_mhz": 2452.0,
                "freq_high_mhz": 2472.0,
            }
        ],
        selected_states=[current_state],
    )

    assert ranked[0]["replacement_state"].state_id == 1


def test_rank_wifi_state_moves_for_direct_accept_if_better_prefers_candidate_with_lower_overlap_ratio():
    current_state = JointCandidateState(
        state_id=10,
        pair_id=7,
        medium="wifi",
        offset=8,
        channel=5,
        width_slots=4,
        period_slots=16,
    )
    better_state = JointCandidateState(
        state_id=11,
        pair_id=7,
        medium="wifi",
        offset=8,
        channel=0,
        width_slots=4,
        period_slots=16,
    )
    worse_state = JointCandidateState(
        state_id=12,
        pair_id=7,
        medium="wifi",
        offset=9,
        channel=5,
        width_slots=4,
        period_slots=16,
    )

    ranked = rank_wifi_state_moves_for_direct_accept_if_better(
        current_state=current_state,
        candidate_wifi_states=[worse_state, better_state],
        residual_holes=[
            {
                "slot_start": 8.0,
                "slot_end": 12.0,
                "freq_low_mhz": 2434.0,
                "freq_high_mhz": 2438.0,
            }
        ],
    )

    assert ranked[0]["replacement_state"].state_id == 11
    assert ranked[0]["accept_if_better"] is True
    assert ranked[0]["hole_capacity"] == compute_hole_capacity(
        {
            "slot_start": 8.0,
            "slot_end": 12.0,
            "freq_low_mhz": 2434.0,
            "freq_high_mhz": 2438.0,
        }
    )


def test_residual_hole_scoring_prefers_ble_state_that_fits_open_gap():
    selected = [
        JointCandidateState(
            state_id=0,
            pair_id=0,
            medium="wifi",
            offset=0,
            channel=0,
            width_slots=4,
            period_slots=16,
        ),
        JointCandidateState(
            state_id=1,
            pair_id=1,
            medium="wifi",
            offset=8,
            channel=0,
            width_slots=4,
            period_slots=16,
        ),
    ]
    good = JointCandidateState(
        state_id=2,
        pair_id=2,
        medium="ble",
        offset=4,
        channel=5,
        pattern_id=0,
        ci_slots=8,
        ce_slots=1,
        num_events=1,
    )
    bad = JointCandidateState(
        state_id=3,
        pair_id=2,
        medium="ble",
        offset=1,
        channel=5,
        pattern_id=0,
        ci_slots=8,
        ce_slots=1,
        num_events=1,
    )

    assert score_ble_state_against_residual_holes(good, selected) > score_ble_state_against_residual_holes(bad, selected)


def test_extract_residual_holes_returns_time_gaps_per_frequency_stripe():
    selected = [
        JointCandidateState(
            state_id=0,
            pair_id=0,
            medium="wifi",
            offset=0,
            channel=0,
            width_slots=4,
            period_slots=16,
            num_events=2,
            cyclic_periodic=True,
            macrocycle_slots=64,
        ),
        JointCandidateState(
            state_id=1,
            pair_id=1,
            medium="wifi",
            offset=16,
            channel=5,
            width_slots=4,
            period_slots=16,
            num_events=2,
            cyclic_periodic=True,
            macrocycle_slots=64,
        ),
    ]

    holes = extract_residual_holes(
        selected_states=selected,
        macrocycle_slots=64,
        freq_grid_mhz=[2412.0, 2437.0, 2462.0],
    )

    assert holes
    assert any(hole["slot_start"] == 4.0 for hole in holes)
    assert any(hole["freq_center_mhz"] == 2437.0 for hole in holes)


def test_rank_ble_insertions_for_holes_prefers_state_that_fills_more():
    selected = [
        JointCandidateState(
            state_id=0,
            pair_id=0,
            medium="wifi",
            offset=0,
            channel=0,
            width_slots=4,
            period_slots=16,
            num_events=2,
            cyclic_periodic=True,
            macrocycle_slots=64,
        )
    ]
    good = JointCandidateState(
        state_id=2,
        pair_id=2,
        medium="ble",
        offset=4,
        channel=20,
        pattern_id=0,
        ci_slots=8,
        ce_slots=2,
        num_events=1,
        cyclic_periodic=False,
        macrocycle_slots=64,
    )
    bad = JointCandidateState(
        state_id=3,
        pair_id=3,
        medium="ble",
        offset=6,
        channel=1,
        pattern_id=0,
        ci_slots=8,
        ce_slots=1,
        num_events=1,
        cyclic_periodic=False,
        macrocycle_slots=64,
    )
    ranked = rank_ble_insertions_for_holes(
        candidates=[bad, good],
        residual_holes=[
            {
                "slot_start": 4.0,
                "slot_end": 8.0,
                "freq_low_mhz": 2451.0,
                "freq_high_mhz": 2453.0,
            }
        ],
        selected_states=selected,
    )

    assert ranked[0].pair_id == 2


def test_compute_hole_capacity_returns_slot_times_bandwidth():
    hole = {
        "slot_start": 20.0,
        "slot_end": 28.0,
        "freq_low_mhz": 2450.0,
        "freq_high_mhz": 2452.0,
    }

    assert compute_hole_capacity(hole) == 16.0


def test_rank_ble_insertions_prefers_candidate_that_uses_larger_fraction_of_hole_capacity():
    hole = {
        "slot_start": 20.0,
        "slot_end": 28.0,
        "freq_low_mhz": 2450.0,
        "freq_high_mhz": 2452.0,
    }
    dense = JointCandidateState(
        state_id=10,
        pair_id=10,
        medium="ble",
        offset=20,
        channel=23,
        pattern_id=0,
        ci_slots=16,
        ce_slots=4,
        num_events=1,
        macrocycle_slots=64,
    )
    sparse = JointCandidateState(
        state_id=11,
        pair_id=11,
        medium="ble",
        offset=20,
        channel=23,
        pattern_id=0,
        ci_slots=16,
        ce_slots=1,
        num_events=1,
        macrocycle_slots=64,
    )

    ranked = rank_ble_insertions_for_holes(
        candidates=[dense, sparse],
        residual_holes=[hole],
        selected_states=[],
    )

    assert ranked[0].pair_id == 10


def test_rank_ble_subset_replacements_prefers_two_dense_ble_over_one_sparse_blocker():
    dense_a = JointCandidateState(
        state_id=60,
        pair_id=60,
        medium="ble",
        offset=10,
        channel=22,
        pattern_id=0,
        ci_slots=16,
        ce_slots=2,
        num_events=1,
        macrocycle_slots=64,
    )
    dense_b = JointCandidateState(
        state_id=61,
        pair_id=61,
        medium="ble",
        offset=12,
        channel=22,
        pattern_id=0,
        ci_slots=16,
        ce_slots=2,
        num_events=1,
        macrocycle_slots=64,
    )

    ranked = rank_ble_subset_replacements(
        selected_ble_states=[],
        candidate_ble_states=[dense_a, dense_b],
        residual_holes=[
            {
                "slot_start": 20.0,
                "slot_end": 28.0,
                "freq_low_mhz": 2450.0,
                "freq_high_mhz": 2452.0,
            }
        ],
        protected_wifi_states=[],
        subset_size_limit=2,
    )

    assert ranked
    assert {state.pair_id for state in ranked[0]["insert_states"]} == {60, 61}


def test_rank_residual_candidate_swaps_returns_best_local_joint_moves_first():
    current_selection = [
        JointCandidateState(
            state_id=0,
            pair_id=0,
            medium="wifi",
            offset=0,
            channel=0,
            width_slots=4,
            period_slots=16,
        ),
    ]
    replacements = [
        JointCandidateState(
            state_id=1,
            pair_id=0,
            medium="wifi",
            offset=0,
            channel=0,
            width_slots=4,
            period_slots=16,
        ),
        JointCandidateState(
            state_id=2,
            pair_id=0,
            medium="wifi",
            offset=8,
            channel=0,
            width_slots=4,
            period_slots=16,
        ),
    ]
    ble_candidates = [
        JointCandidateState(
            state_id=3,
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
            state_id=4,
            pair_id=2,
            medium="ble",
            offset=5,
            channel=5,
            pattern_id=0,
            ci_slots=8,
            ce_slots=1,
            num_events=1,
        ),
    ]

    ranked = rank_residual_candidate_swaps(current_selection, blocker_pair_id=0, replacement_candidates=replacements, unscheduled_ble_candidates=ble_candidates)

    assert ranked
    assert ranked[0]["combined_gain"] >= ranked[-1]["combined_gain"]
