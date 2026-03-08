import numpy as np

from sim_src.env.env import env


def test_env_builds_pair_level_arrays():
    e = env(cell_edge=7.0, cell_size=2, pair_density_per_m2=0.05, seed=1)
    assert e.n_pair == e.n_sta
    assert e.n_office == e.n_ap
    assert e.pair_radio_type.shape[0] == e.n_pair
    assert e.pair_priority.shape[0] == e.n_pair
    assert e.pair_office_id.shape[0] == e.n_pair
    assert e.pair_tx_locs.shape == (e.n_pair, 2)
    assert e.pair_rx_locs.shape == (e.n_pair, 2)
    assert np.array_equal(e.device_priority, e.pair_priority)
