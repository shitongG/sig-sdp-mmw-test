import numpy as np

from sim_src.env.env import env


def test_ble_retry_resamples_per_ce_channels_without_changing_ci_ce():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=6,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )
    ble_ids = np.where(np.logical_and(e.pair_radio_type == e.RADIO_BLE, e.pair_ble_ce_feasible))[0]
    assert ble_ids.size > 0
    pair_id = int(ble_ids[0])
    ci_before = int(e.pair_ble_ci_slots[pair_id])
    ce_before = int(e.pair_ble_ce_slots[pair_id])
    e.pair_ble_ce_channels[pair_id] = np.zeros_like(e.pair_ble_ce_channels[pair_id])
    channels_before = e.pair_ble_ce_channels[pair_id].copy()

    e.resample_ble_channels(np.array([pair_id], dtype=int))

    assert int(e.pair_ble_ci_slots[pair_id]) == ci_before
    assert int(e.pair_ble_ce_slots[pair_id]) == ce_before
    assert not np.array_equal(e.pair_ble_ce_channels[pair_id], channels_before)
