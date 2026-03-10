import numpy as np

from sim_src.env.env import env


def test_pair_start_time_slot_defaults_to_zero_for_all_pairs():
    e = env(cell_size=2, sta_density_per_1m2=0.01, seed=7)
    assert np.array_equal(e.pair_start_time_slot, np.zeros(e.n_pair, dtype=int))


def test_wifi_anchor_respects_start_time_slot_and_tx_width():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=3,
        radio_prob=(1.0, 0.0),
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
    )
    wifi_idx = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    assert wifi_idx.size > 0
    for k in wifi_idx:
        low = int(e.pair_start_time_slot[k])
        high = int(low + e.pair_wifi_period_slots[k] - e.pair_wifi_tx_slots[k])
        assert low <= int(e.pair_wifi_anchor_slot[k]) <= high


def test_ble_anchor_respects_start_time_slot_and_ce_width():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=5,
        radio_prob=(0.0, 1.0),
        ble_ce_max_s=7.5e-3,
        ble_phy_rate_bps=2e6,
    )
    ble_idx = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert ble_idx.size > 0
    for k in ble_idx:
        if not e.pair_ble_ce_feasible[k]:
            continue
        low = int(e.pair_start_time_slot[k])
        high = int(low + e.pair_ble_ci_slots[k] - e.pair_ble_ce_slots[k])
        assert low <= int(e.pair_ble_anchor_slot[k]) <= high
