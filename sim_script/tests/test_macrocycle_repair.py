import numpy as np
import scipy.sparse

from sim_script.pd_mmw_template_ap_stats import (
    assign_macrocycle_start_slots,
    retry_ble_channels_and_assign_macrocycle,
)


class _GreedyMissEnv:
    n_pair = 4
    RADIO_WIFI = 0
    RADIO_BLE = 1
    pair_radio_type = np.array([1, 1, 1, 1], dtype=int)
    pair_priority = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    pair_channel = np.array([0, 0, 0, 0], dtype=int)
    pair_ble_ci_slots = np.array([4, 4, 4, 4], dtype=int)
    pair_ble_ce_slots = np.array([1, 1, 1, 2], dtype=int)
    pair_ble_ce_feasible = np.array([True, True, True, True], dtype=bool)

    def compute_macrocycle_slots(self):
        return 4

    def get_pair_period_slots(self):
        return np.array([4, 4, 4, 4], dtype=int)

    def get_pair_width_slots(self):
        return np.array([1, 1, 1, 2], dtype=int)

    def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
        occ = np.zeros(macrocycle_slots, dtype=bool)
        width = int(self.get_pair_width_slots()[pair_id])
        start = int(start_slot % self.get_pair_period_slots()[pair_id])
        occ[start : min(start + width, macrocycle_slots)] = True
        return occ

    def build_pair_conflict_matrix(self):
        return np.ones((self.n_pair, self.n_pair), dtype=bool) ^ np.eye(self.n_pair, dtype=bool)

    def get_macrocycle_conflict_state(self):
        s_gain = scipy.sparse.csr_matrix((self.n_pair, self.n_pair), dtype=float)
        q_conflict = scipy.sparse.csr_matrix((self.n_pair, self.n_pair), dtype=float)
        h_max = np.ones(self.n_pair, dtype=float) * 999.0
        return s_gain, q_conflict, h_max

    def resample_ble_channels(self, pair_ids):
        return


def test_repair_step_can_recover_feasible_solution_without_channel_retry():
    env = _GreedyMissEnv()
    preferred = np.array([0, 0, 0, 0], dtype=int)

    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(
        env,
        preferred,
        allow_partial=True,
    )
    assert macro == 4
    assert unscheduled == [2]

    repaired = retry_ble_channels_and_assign_macrocycle(
        env,
        preferred,
        max_ble_channel_retries=0,
    )

    repaired_unscheduled = repaired[3]
    repaired_starts = repaired[0]
    assert repaired_unscheduled == []
    assert sorted(repaired_starts.tolist()) == [0, 1, 2, 3]
