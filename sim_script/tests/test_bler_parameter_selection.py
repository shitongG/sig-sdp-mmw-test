import numpy as np
import pytest

from sim_src.env.env import env


def test_env_exposes_radio_specific_user_packet_and_bandwidth_arrays():
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=1,
        wifi_packet_bit=12000,
        ble_packet_bit=320,
    )

    expected_packet_bits = np.where(
        e.user_radio_type == e.RADIO_WIFI,
        12000.0,
        320.0,
    )
    expected_bandwidths = np.where(
        e.user_radio_type == e.RADIO_WIFI,
        e.wifi_channel_bandwidth_hz,
        e.ble_channel_bandwidth_hz,
    )

    assert np.array_equal(e.user_packet_bits, expected_packet_bits)
    assert np.array_equal(e.user_bandwidth_hz, expected_bandwidths)
    assert np.array_equal(e.pair_packet_bits, e.user_packet_bits)
    assert np.array_equal(e.pair_bandwidth_hz, e.user_bandwidth_hz)


def test_evaluate_bler_uses_per_pair_packet_bits_and_bandwidths():
    n_pair = int((2**2) * 0.05 * (7.0**2))
    packet_bits = np.linspace(500.0, 1300.0, n_pair)
    bandwidths = np.full(n_pair, 1e6)
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=2,
        pair_packet_bits=packet_bits,
        pair_bandwidth_hz=bandwidths,
    )
    z = np.arange(e.n_pair, dtype=int)
    Z = int(e.n_pair)

    sinr = e.evaluate_sinr(z, Z)
    expected = np.array(
        [
            env.polyanskiy_model(sinr[k], packet_bits[k], bandwidths[k], e.slot_time)
            for k in range(e.n_pair)
        ]
    )

    assert np.allclose(e.evaluate_bler(z, Z), expected)


def test_env_computes_per_link_min_sinr_from_radio_defaults():
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=3,
        radio_prob=(0.5, 0.5),
        wifi_packet_bit=8000,
        ble_packet_bit=1000,
        wifi_channel_bandwidth_hz=5e6,
        ble_channel_bandwidth_hz=2e6,
    )

    expected = np.array(
        [
            env.db_to_dec(env.bisection_method(e.pair_packet_bits[k], e.pair_bandwidth_hz[k], e.slot_time, e.max_err))
            for k in range(e.n_pair)
        ]
    )

    assert np.allclose(e.pair_min_sinr, expected)
    assert np.array_equal(e.user_min_sinr, e.pair_min_sinr)
    assert np.array_equal(e.device_min_sinr, e.pair_min_sinr)


def test_compute_txp_follows_per_link_min_sinr():
    n_pair = int((2**2) * 0.05 * (7.0**2))
    packet_bits = np.linspace(500.0, 1300.0, n_pair)
    bandwidths = np.full(n_pair, 1e6)
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=4,
        pair_packet_bits=packet_bits,
        pair_bandwidth_hz=bandwidths,
    )

    dis = np.linalg.norm(e.pair_tx_locs - e.pair_rx_locs, axis=1)
    gain = -env.fre_dis_to_loss_dB(e.fre_Hz, dis)
    expected = env.dec_to_db(e.pair_min_sinr) - (
        gain - env.bandwidth_txpr_to_noise_dBm(e.pair_bandwidth_hz)
    )
    expected = expected + env.dec_to_db(e.txp_offset)

    assert np.allclose(e._compute_txp().ravel(), expected)


def test_generate_s_q_hmax_uses_per_link_min_sinr():
    n_pair = int((2**2) * 0.05 * (7.0**2))
    packet_bits = np.linspace(500.0, 1300.0, n_pair)
    bandwidths = np.full(n_pair, 1e6)
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=5,
        pair_packet_bits=packet_bits,
        pair_bandwidth_hz=bandwidths,
    )

    s_gain, _, h_max = e.generate_S_Q_hmax()
    expected = s_gain.diagonal() / e.pair_min_sinr - 1.0

    assert np.allclose(h_max, expected)


def test_bandwidth_txpr_to_noise_dbm_uses_thermal_noise_formula():
    bandwidths = np.array([1e6, 2e6, 5e6], dtype=float)

    expected = -174.0 + 10.0 * np.log10(bandwidths) + env.NOISEFIGURE

    assert np.allclose(env.bandwidth_txpr_to_noise_dBm(bandwidths), expected)


def test_compute_txp_uses_bandwidth_dependent_noise_per_link():
    n_pair = int((2**2) * 0.05 * (7.0**2))
    packet_bits = np.full(n_pair, 1300.0)
    bandwidths = np.linspace(1e6, 3e6, n_pair)
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=7,
        pair_packet_bits=packet_bits,
        pair_bandwidth_hz=bandwidths,
    )

    dis = np.linalg.norm(e.pair_tx_locs - e.pair_rx_locs, axis=1)
    gain = -env.fre_dis_to_loss_dB(e.fre_Hz, dis)
    expected = env.dec_to_db(e.pair_min_sinr) - (
        gain - env.bandwidth_txpr_to_noise_dBm(e.pair_bandwidth_hz)
    )
    expected = expected + env.dec_to_db(e.txp_offset)

    assert np.allclose(e._compute_txp().ravel(), expected)


def test_generate_s_q_hmax_state_changes_with_bandwidth_dependent_noise():
    n_pair = int((2**2) * 0.05 * (7.0**2))
    packet_bits = np.full(n_pair, 1300.0)
    low_bw = np.full(n_pair, 1e6)
    high_bw = np.full(n_pair, 3e6)

    e_low = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=8,
        pair_packet_bits=packet_bits,
        pair_bandwidth_hz=low_bw,
    )
    e_high = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.05,
        seed=8,
        pair_packet_bits=packet_bits,
        pair_bandwidth_hz=high_bw,
    )

    s_low, _, h_low = e_low.generate_S_Q_hmax()
    s_high, _, h_high = e_high.generate_S_Q_hmax()

    assert not np.allclose(s_low.toarray(), s_high.toarray())
    assert np.allclose(h_low, h_high)


def test_bandwidth_txpr_to_noise_dbm_rejects_non_positive_bandwidth():
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        env.bandwidth_txpr_to_noise_dBm(np.array([0.0, 1e6]))
