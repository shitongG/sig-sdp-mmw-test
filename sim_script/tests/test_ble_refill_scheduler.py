import numpy as np

import sim_script.pd_mmw_template_ap_stats as mod


class _BleRefillEnv:
    RADIO_WIFI = 0
    RADIO_BLE = 1
    n_pair = 3
    pair_radio_type = np.array([1, 1, 0], dtype=int)
    pair_priority = np.array([3.0, 2.0, 1.0], dtype=float)

    def get_pair_width_slots(self):
        return np.array([2, 1, 1], dtype=int)

    def get_pair_period_slots(self):
        return np.array([4, 4, 4], dtype=int)


def test_ble_refill_prioritizes_unscheduled_ble_pairs(monkeypatch):
    env = _BleRefillEnv()
    preferred = np.array([0, 0, 0], dtype=int)
    initial = (np.array([-1, 0, 1], dtype=int), 4, np.zeros((3, 4), dtype=bool), [0])

    calls = []

    def fake_assign(_env, _preferred, allow_partial=True, pair_order=None):
        calls.append(tuple(pair_order.tolist()))
        if pair_order[0] == 0:
            occ = np.zeros((3, 4), dtype=bool)
            occ[0, 0] = True
            occ[1, 1] = True
            occ[2, 2] = True
            return np.array([0, 1, 2], dtype=int), 4, occ, []
        return initial

    monkeypatch.setattr(mod, "assign_macrocycle_start_slots", fake_assign)

    result = mod._refill_unscheduled_pairs_by_radio(
        env,
        preferred,
        initial,
        target_radio_type=env.RADIO_BLE,
    )

    assert result[3] == []
    assert calls
    assert calls[0][0] == 0
