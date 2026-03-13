import numpy as np

from sim_src.env.env import env


def test_ble_multi_ce_mode_initializes_per_ce_channel_storage():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=2,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
    )
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]

    assert ble_ids.size > 0
    assert e.ble_channel_mode == "per_ce"
    assert isinstance(e.pair_ble_ce_channels, dict)
