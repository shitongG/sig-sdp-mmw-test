import numpy as np
import scipy.sparse

from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots


class _WifiFirstEnv:
    n_pair = 4
    RADIO_WIFI = 0
    RADIO_BLE = 1
    ble_channel_mode = "single"
    pair_radio_type = np.array([0, 0, 1, 1], dtype=int)
    pair_priority = np.array([10.0, 9.0, 2.0, 1.0], dtype=float)
    pair_ble_ce_feasible = np.array([True, True, True, True], dtype=bool)

    def compute_macrocycle_slots(self):
        return 8

    def get_pair_period_slots(self):
        return np.array([8, 8, 8, 8], dtype=int)

    def get_pair_width_slots(self):
        return np.array([1, 1, 1, 1], dtype=int)

    def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
        occ = np.zeros(macrocycle_slots, dtype=bool)
        occ[int(start_slot)] = True
        return occ

    def build_pair_conflict_matrix(self):
        return np.zeros((4, 4), dtype=bool)

    def get_macrocycle_conflict_state(self):
        return (
            scipy.sparse.csr_matrix((4, 4), dtype=float),
            scipy.sparse.csr_matrix((4, 4), dtype=bool),
            np.full(4, 1e9, dtype=float),
        )

    def get_ble_start_slot_capacity(self, wifi_pair_ids, wifi_start_slots, start_slot):
        return 1 if int(start_slot) == 0 else 2

    def compute_ble_no_collision_probability(self, c, n):
        if n <= 1:
            return 1.0
        if c <= 0:
            return 0.0
        return float((1.0 - 1.0 / c) ** (n - 1))


def test_wifi_first_assignment_limits_ble_pairs_by_remaining_channel_capacity():
    env = _WifiFirstEnv()
    starts, macro, occ, unscheduled, ble_stats = assign_macrocycle_start_slots(
        env,
        preferred_slots=np.zeros(env.n_pair, dtype=int),
        allow_partial=True,
        wifi_first=True,
        return_ble_stats=True,
    )

    scheduled_ble = [idx for idx in [2, 3] if idx not in unscheduled and starts[idx] == 0]
    assert len(scheduled_ble) == 1
    assert ble_stats[0]["effective_ble_channels"] == 1
    assert ble_stats[0]["scheduled_ble_pairs"] == 1
    assert ble_stats[0]["no_collision_probability"] == 1.0
