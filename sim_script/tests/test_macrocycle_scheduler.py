import numpy as np

from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots, build_schedule_rows
from sim_src.env.env import env


def test_macrocycle_schedule_has_no_overlapping_occupancy():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.003,
        seed=5,
        radio_prob=(0.6, 0.4),
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
        ble_ci_exp_min=4,
        ble_ci_exp_max=5,
        ble_phy_rate_bps=2e6,
        ble_ce_max_s=7.5e-3,
    )
    preferred = np.arange(e.n_pair, dtype=int)
    start_slots, macro, occ = assign_macrocycle_start_slots(e, preferred)
    assert start_slots.shape[0] == e.n_pair
    assert np.all(start_slots >= 0)
    assert occ.shape == (e.n_pair, macro)
    assert np.all(np.sum(occ, axis=0) <= 1)


def test_build_schedule_rows_emits_true_slot_occupancy_rows():
    pair_rows = [
        {
            "pair_id": 2,
            "radio": "ble",
            "schedule_slot": 9,
            "occupied_slots_in_macrocycle": [9, 10, 11],
        },
        {
            "pair_id": 14,
            "radio": "wifi",
            "schedule_slot": 12,
            "occupied_slots_in_macrocycle": [12, 13, 14, 15],
        },
    ]
    rows = build_schedule_rows(pair_rows)
    assert rows[0]["schedule_slot"] == 9
    assert rows[0]["pair_ids"] == [2]
    assert rows[1]["schedule_slot"] == 10
    assert rows[1]["pair_ids"] == [2]
    assert rows[3]["schedule_slot"] == 12
    assert rows[3]["wifi_pair_ids"] == [14]
