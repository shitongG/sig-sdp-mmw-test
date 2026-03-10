import numpy as np
import scipy.sparse

from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots


class _NonConflictEnv:
    n_pair = 2
    pair_priority = np.array([2.0, 1.0], dtype=float)
    RADIO_BLE = 1
    pair_radio_type = np.array([0, 0], dtype=int)
    pair_ble_ce_feasible = np.array([True, True], dtype=bool)

    def compute_macrocycle_slots(self):
        return 4

    def get_pair_period_slots(self):
        return np.array([4, 4], dtype=int)

    def get_pair_width_slots(self):
        return np.array([2, 2], dtype=int)

    def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
        occ = np.zeros(macrocycle_slots, dtype=bool)
        occ[0:2] = True
        return occ

    def build_pair_conflict_matrix(self):
        return np.array([[False, False], [False, False]], dtype=bool)

    def get_macrocycle_conflict_state(self):
        s_gain = scipy.sparse.csr_matrix(np.array([[0.4, 0.2], [0.2, 0.4]], dtype=float))
        q_conflict = scipy.sparse.csr_matrix((2, 2))
        h_max = np.array([1.0, 1.0], dtype=float)
        return s_gain, q_conflict, h_max


class _ConflictEnv(_NonConflictEnv):
    def build_pair_conflict_matrix(self):
        return np.array([[False, True], [True, False]], dtype=bool)


def test_macrocycle_scheduler_allows_overlap_for_non_conflicting_pairs():
    env = _NonConflictEnv()

    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(
        env,
        np.array([0, 0], dtype=int),
        allow_partial=False,
    )

    assert unscheduled == []
    assert macro == 4
    assert starts.tolist() == [0, 0]
    assert occ[0, 0] and occ[1, 0]
    assert np.any(np.logical_and(occ[0], occ[1]))


def test_macrocycle_scheduler_still_blocks_overlap_for_conflicting_pairs():
    env = _ConflictEnv()

    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(
        env,
        np.array([0, 0], dtype=int),
        allow_partial=True,
    )

    assert macro == 4
    assert starts[0] >= 0
    assert starts[1] == -1
    assert unscheduled == [1]
    assert not np.any(np.logical_and(occ[0], occ[1]))
