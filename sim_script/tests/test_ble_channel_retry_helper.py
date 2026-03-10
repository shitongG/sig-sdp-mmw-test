import numpy as np

from sim_src.env.env import env


def test_resample_ble_channels_updates_only_target_ble_pairs_and_preserves_timing():
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
    target = np.array([0], dtype=int)
    old_channel = e.pair_channel.copy()
    old_ci = e.pair_ble_ci_slots.copy()
    old_ce = e.pair_ble_ce_slots.copy()

    e.resample_ble_channels(target)

    assert old_channel[0] != e.pair_channel[0] or e.ble_channel_count == 1
    assert np.array_equal(old_ci, e.pair_ble_ci_slots)
    assert np.array_equal(old_ce, e.pair_ble_ce_slots)
