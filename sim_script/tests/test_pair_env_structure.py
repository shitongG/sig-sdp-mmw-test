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


def test_env_builds_pair_level_bler_parameter_arrays():
    e = env(cell_edge=7.0, cell_size=2, pair_density_per_m2=0.05, seed=1)

    assert e.pair_packet_bits.shape[0] == e.n_pair
    assert e.pair_bandwidth_hz.shape[0] == e.n_pair
    assert np.array_equal(e.device_packet_bits, e.pair_packet_bits)
    assert np.array_equal(e.user_packet_bits, e.pair_packet_bits)
    assert np.array_equal(e.device_bandwidth_hz, e.pair_bandwidth_hz)
    assert np.array_equal(e.user_bandwidth_hz, e.pair_bandwidth_hz)
