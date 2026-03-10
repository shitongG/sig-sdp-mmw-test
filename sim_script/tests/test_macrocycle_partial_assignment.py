import numpy as np

from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots


class _StubEnv:
    n_pair = 3
    pair_priority = np.array([3.0, 2.0, 1.0])
    RADIO_BLE = 1
    pair_radio_type = np.array([0, 0, 0], dtype=int)
    pair_ble_ce_feasible = np.array([True, True, True], dtype=bool)

    def compute_macrocycle_slots(self):
        return 4

    def get_pair_period_slots(self):
        return np.array([4, 4, 4], dtype=int)

    def get_pair_width_slots(self):
        return np.array([2, 2, 2], dtype=int)

    def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
        occ = np.zeros(macrocycle_slots, dtype=bool)
        occ[start_slot : min(start_slot + 2, macrocycle_slots)] = True
        return occ

    def build_pair_conflict_matrix(self):
        return np.ones((self.n_pair, self.n_pair), dtype=bool) ^ np.eye(self.n_pair, dtype=bool)


def test_assign_macrocycle_start_slots_can_return_partial_assignment():
    env = _StubEnv()

    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(
        env,
        np.array([0, 1, 2], dtype=int),
        allow_partial=True,
    )

    assert macro == 4
    assert starts[0] >= 0
    assert starts[1] >= 0
    assert starts[2] == -1
    assert occ.shape == (3, 4)
    assert unscheduled == [2]
