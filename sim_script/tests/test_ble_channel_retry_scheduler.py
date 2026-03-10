import numpy as np
import scipy.sparse

from sim_script.pd_mmw_template_ap_stats import retry_ble_channels_and_assign_macrocycle


class _RetryEnv:
    n_pair = 2
    RADIO_WIFI = 0
    RADIO_BLE = 1
    pair_radio_type = np.array([1, 1], dtype=int)
    pair_priority = np.array([2.0, 1.0], dtype=float)
    pair_channel = np.array([0, 0], dtype=int)
    pair_ble_ci_slots = np.array([8, 8], dtype=int)
    pair_ble_ce_slots = np.array([3, 3], dtype=int)
    pair_ble_ce_feasible = np.array([True, True], dtype=bool)
    retry_count = 0

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
        if self.retry_count == 0:
            return np.array([[False, True], [True, False]], dtype=bool)
        return np.array([[False, False], [False, False]], dtype=bool)

    def get_macrocycle_conflict_state(self):
        s_gain = scipy.sparse.csr_matrix(np.array([[0.4, 0.2], [0.2, 0.4]], dtype=float))
        q_conflict = scipy.sparse.csr_matrix((2, 2))
        h_max = np.array([1.0, 1.0], dtype=float)
        return s_gain, q_conflict, h_max

    def resample_ble_channels(self, pair_ids):
        self.retry_count += 1
        self.pair_channel[pair_ids] = 1


def test_retry_ble_channels_can_reduce_unscheduled_pairs():
    env = _RetryEnv()
    preferred = np.array([0, 0], dtype=int)
    starts, macro, occ, unscheduled, retries_used = retry_ble_channels_and_assign_macrocycle(
        env,
        preferred,
        max_ble_channel_retries=1,
    )
    assert unscheduled == []
    assert retries_used == 1


def test_retry_does_not_change_ci_ce():
    env = _RetryEnv()
    old_ci = env.pair_ble_ci_slots.copy()
    old_ce = env.pair_ble_ce_slots.copy()
    retry_ble_channels_and_assign_macrocycle(env, np.array([0, 0], dtype=int), max_ble_channel_retries=1)
    assert np.array_equal(old_ci, env.pair_ble_ci_slots)
    assert np.array_equal(old_ce, env.pair_ble_ce_slots)
