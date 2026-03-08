import numpy as np

from sim_src.env.env import env


def test_same_office_non_overlapping_pairs_not_forced_mutually_exclusive():
    e = env(cell_edge=7.0, cell_size=1, pair_density_per_m2=0.05, seed=2, radio_prob=(1.0, 0.0))
    if e.n_pair < 2:
        raise AssertionError("test requires at least 2 pairs")

    e.pair_office_id[:] = 0
    e.pair_radio_type[:] = e.RADIO_WIFI
    e.pair_channel[:] = 0
    e.pair_channel[0] = 0
    e.pair_channel[1] = 10
    e.device_radio_type = e.pair_radio_type
    e.device_radio_channel = e.pair_channel
    e.user_radio_type = e.pair_radio_type
    e.user_radio_channel = e.pair_channel

    _, q_conflict, _ = e.generate_S_Q_hmax()
    q = q_conflict.toarray()
    assert q[0, 1] == 0
    assert q[1, 0] == 0
