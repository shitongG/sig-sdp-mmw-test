import numpy as np

from sim_src.env.env import env
from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots


def test_real_env_macrocycle_scheduler_keeps_multiple_pairs_when_conflicts_allow():
    e = env(
        cell_edge=7.0,
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=5,
        radio_prob=(0.0, 1.0),
        slot_time=1.25e-3,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=7.5e-3,
        ble_phy_rate_bps=2e6,
        ble_ci_exp_min=3,
        ble_ci_exp_max=4,
    )

    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(
        e,
        np.zeros(e.n_pair, dtype=int),
        allow_partial=True,
    )

    assert macro > 0
    assert occ.shape[0] == e.n_pair
    assert len(unscheduled) < e.n_pair
