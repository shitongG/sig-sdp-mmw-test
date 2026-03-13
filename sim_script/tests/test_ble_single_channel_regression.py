import numpy as np

from sim_src.env.env import env


def test_ble_default_mode_keeps_single_pair_channel():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=1, radio_prob=(0.0, 1.0))
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]

    assert ble_ids.size > 0
    assert e.pair_channel.shape[0] == e.n_pair
    assert not hasattr(e, "pair_ble_ce_channels") or e.pair_ble_ce_channels is None
