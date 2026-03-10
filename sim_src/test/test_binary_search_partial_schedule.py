import numpy as np
import scipy.sparse

from sim_src.alg.binary_search_relaxation import binary_search_relaxation


class _StubAlg:
    def run_with_state(self, bs_iteration, Z, state):
        return True, np.eye(state[0].shape[0], dtype=float)

    def rounding(self, Z, gX, state, user_priority=None, slot_mask=None):
        if Z < 3:
            return np.array([0, 1, 0], dtype=int), Z, 1
        return np.array([0, 1, 2], dtype=int), Z, 0


def test_binary_search_returns_partial_result_when_max_slot_cap_blocks_full_schedule():
    state = (
        scipy.sparse.eye(3, format="csr"),
        scipy.sparse.eye(3, format="csr"),
        np.ones(3, dtype=float),
    )
    bs = binary_search_relaxation()
    bs.feasibility_check_alg = _StubAlg()
    bs.max_slot_cap = 2

    z_vec, z_fin, remainder = bs.run(state)
    partial = bs.last_partial_schedule

    assert z_fin == 2
    assert remainder == 1
    assert partial["slot_cap_hit"] is True
    assert partial["scheduled_pair_ids"] == [0, 1]
    assert partial["unscheduled_pair_ids"] == [2]
