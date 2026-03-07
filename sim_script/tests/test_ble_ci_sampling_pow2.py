import numpy as np
from sim_src.env.env import env


def test_ble_user_ci_is_from_discrete_candidates():
    e = env(cell_size=3, sta_density_per_1m2=0.01, seed=2, radio_prob=(0.0, 1.0))
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    ci_quanta = np.rint((e.user_ble_ci_slots[ble_idx] * e.slot_time) / 1.25e-3).astype(int)
    assert np.all(np.isin(ci_quanta, e.ble_ci_quanta_candidates))
