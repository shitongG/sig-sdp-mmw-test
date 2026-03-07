import numpy as np
from sim_src.env.env import env


def test_ble_anchor_quantized_and_window_periodic():
    e = env(cell_size=2, sta_density_per_1m2=0.01, seed=7, radio_prob=(0.0, 1.0))
    Z = 300
    mask = e.build_slot_compatibility_mask(Z)
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    for k in ble_idx:
        if not e.user_ble_ce_feasible[k]:
            assert not mask[k].any()
            continue
        ci = int(e.user_ble_ci_slots[k])
        ce = int(e.user_ble_ce_slots[k])
        anchor = int(e.user_ble_anchor_slot[k])
        assert 0 <= anchor < ci
        assert abs((anchor * e.slot_time) / e.slot_time - anchor) < 1e-12
        assert mask[k, anchor:min(anchor + ce, Z)].all()
