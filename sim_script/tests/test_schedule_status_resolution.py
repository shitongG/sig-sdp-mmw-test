import numpy as np

from sim_script.pd_mmw_template_ap_stats import resolve_macrocycle_schedule_status


def test_resolve_macrocycle_schedule_status_uses_final_macrocycle_assignment():
    schedule_start_slots = np.array([5, -1, 9], dtype=int)
    occupied_slots = np.array(
        [
            [False, False, True, False],
            [False, False, False, False],
            [False, True, True, False],
        ],
        dtype=bool,
    )

    scheduled_pair_ids, unscheduled_pair_ids = resolve_macrocycle_schedule_status(
        schedule_start_slots,
        occupied_slots,
    )

    assert scheduled_pair_ids == [0, 2]
    assert unscheduled_pair_ids == [1]


def test_resolve_macrocycle_schedule_status_treats_missing_occupancy_as_unscheduled():
    schedule_start_slots = np.array([7, 3], dtype=int)
    occupied_slots = np.array(
        [
            [False, False, False],
            [True, False, False],
        ],
        dtype=bool,
    )

    scheduled_pair_ids, unscheduled_pair_ids = resolve_macrocycle_schedule_status(
        schedule_start_slots,
        occupied_slots,
    )

    assert scheduled_pair_ids == [1]
    assert unscheduled_pair_ids == [0]
