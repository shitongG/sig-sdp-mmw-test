import numpy as np

from sim_src.env.env import env


def test_ble_available_channels_exclude_wifi_overlapping_spectrum():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=11,
        radio_prob=(1.0, 0.0),
        wifi_channel_bandwidth_hz=20e6,
        ble_channel_bandwidth_hz=2e6,
    )

    e.pair_radio_type[:] = e.RADIO_BLE
    e.pair_radio_type[0] = e.RADIO_WIFI
    e.pair_channel[:] = 0
    e.pair_wifi_period_slots[:] = 0
    e.pair_wifi_period_slots[0] = 8
    e.pair_wifi_tx_slots[:] = 0
    e.pair_wifi_tx_slots[0] = 2
    e.pair_wifi_anchor_slot[:] = 0

    channels = e.get_available_ble_channels_for_start_slot(
        wifi_pair_ids=np.array([0], dtype=int),
        wifi_start_slots=np.array([0], dtype=int),
        start_slot=0,
    )

    expected = np.array([idx for idx in range(e.ble_channel_count) if idx >= 11], dtype=int)
    assert np.array_equal(channels, expected)


def test_ble_start_slot_capacity_matches_available_channel_count():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=12,
        radio_prob=(1.0, 0.0),
        wifi_channel_bandwidth_hz=20e6,
        ble_channel_bandwidth_hz=2e6,
    )

    e.pair_radio_type[:] = e.RADIO_BLE
    e.pair_radio_type[0] = e.RADIO_WIFI
    e.pair_channel[:] = 0
    e.pair_wifi_period_slots[:] = 0
    e.pair_wifi_period_slots[0] = 8
    e.pair_wifi_tx_slots[:] = 0
    e.pair_wifi_tx_slots[0] = 2
    e.pair_wifi_anchor_slot[:] = 0

    c = e.get_ble_start_slot_capacity(
        wifi_pair_ids=np.array([0], dtype=int),
        wifi_start_slots=np.array([0], dtype=int),
        start_slot=0,
    )

    assert c == 26


def test_ble_no_collision_probability_matches_closed_form():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=13)

    assert np.isclose(e.compute_ble_no_collision_probability(c=10, n=3), (1.0 - 0.1) ** 2)
    assert e.compute_ble_no_collision_probability(c=10, n=1) == 1.0
    assert e.compute_ble_no_collision_probability(c=0, n=2) == 0.0


def test_ble_available_channels_respect_wraparound_wifi_occupancy():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=14,
        radio_prob=(1.0, 0.0),
        wifi_channel_bandwidth_hz=20e6,
        ble_channel_bandwidth_hz=2e6,
    )

    e.pair_radio_type[:] = e.RADIO_BLE
    e.pair_radio_type[0] = e.RADIO_WIFI
    e.pair_channel[:] = 0
    e.pair_wifi_period_slots[:] = 0
    e.pair_wifi_period_slots[0] = 32
    e.pair_wifi_tx_slots[:] = 0
    e.pair_wifi_tx_slots[0] = 2
    e.pair_wifi_anchor_slot[:] = 0

    channels = e.get_available_ble_channels_for_start_slot(
        wifi_pair_ids=np.array([0], dtype=int),
        wifi_start_slots=np.array([31], dtype=int),
        start_slot=0,
    )

    expected = np.array([idx for idx in range(e.ble_channel_count) if idx >= 11], dtype=int)
    assert np.array_equal(channels, expected)
