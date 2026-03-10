import numpy as np

from sim_src.env.env import env


def test_macrocycle_interference_state_matches_pair_dimension():
    e = env(
        cell_edge=7.0,
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=7,
        radio_prob=(0.0, 1.0),
        slot_time=1.25e-3,
        ble_ce_max_s=7.5e-3,
        ble_phy_rate_bps=2e6,
    )
    s_gain, q_conflict, h_max = e.get_macrocycle_conflict_state()
    assert s_gain.shape == (e.n_pair, e.n_pair)
    assert q_conflict.shape == (e.n_pair, e.n_pair)
    assert h_max.shape == (e.n_pair,)
    assert np.issubdtype(h_max.dtype, np.floating)
