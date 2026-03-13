import numpy as np

import sim_script.pd_mmw_template_ap_stats as mod


class _WifiRefillEnv:
    RADIO_WIFI = 0
    RADIO_BLE = 1
    n_pair = 3
    pair_radio_type = np.array([0, 0, 1], dtype=int)
    pair_priority = np.array([3.0, 2.0, 1.0], dtype=float)

    def get_pair_width_slots(self):
        return np.array([2, 2, 1], dtype=int)

    def get_pair_period_slots(self):
        return np.array([4, 4, 4], dtype=int)


def test_wifi_refill_prioritizes_unscheduled_wifi_pairs(monkeypatch):
    env = _WifiRefillEnv()
    preferred = np.array([0, 0, 0], dtype=int)
    initial = (np.array([0, -1, 1], dtype=int), 4, np.zeros((3, 4), dtype=bool), [1])

    calls = []

    def fake_assign(_env, _preferred, allow_partial=True, pair_order=None):
        calls.append(tuple(pair_order.tolist()))
        if pair_order[0] == 1:
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
        target_radio_type=env.RADIO_WIFI,
    )

    assert result[3] == []
    assert calls
    assert calls[0][0] == 1


def test_is_better_refill_result_prefers_more_scheduled_wifi_when_wifi_first():
    pair_priority = np.array([5.0, 4.0, 2.0, 1.0], dtype=float)
    pair_radio_type = np.array([0, 0, 1, 1], dtype=int)
    best = (np.array([], dtype=int), 8, np.zeros((4, 8), dtype=bool), [1])
    candidate = (np.array([], dtype=int), 8, np.zeros((4, 8), dtype=bool), [2])

    better = mod._is_better_refill_result(
        candidate,
        best,
        pair_priority,
        pair_radio_type=pair_radio_type,
        wifi_radio_id=0,
        wifi_first=True,
    )

    assert better is True


def test_wifi_first_refill_uses_wifi_aware_candidate_comparison(monkeypatch):
    env = _WifiRefillEnv()
    preferred = np.array([0, 0, 0], dtype=int)
    initial = (np.array([0, -1, 1], dtype=int), 4, np.zeros((3, 4), dtype=bool), [1])

    def fake_assign(_env, _preferred, allow_partial=True, pair_order=None):
        if pair_order[0] == 1:
            return np.array([0, 1, -1], dtype=int), 4, np.zeros((3, 4), dtype=bool), [2]
        return initial

    monkeypatch.setattr(mod, "assign_macrocycle_start_slots", fake_assign)

    result = mod._refill_unscheduled_pairs_by_radio(
        env,
        preferred,
        initial,
        target_radio_type=env.RADIO_WIFI,
        wifi_first=True,
    )

    assert result[3] == [2]
