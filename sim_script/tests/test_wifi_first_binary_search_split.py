import numpy as np
import scipy.sparse

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
import sim_script.pd_mmw_template_ap_stats as pd_script


def test_binary_search_strategy_defaults_to_joint():
    bs = binary_search_relaxation()

    assert bs.strategy == "joint"
    assert bs.pair_radio_type is None
    assert bs.wifi_radio_id == 0
    assert bs.ble_radio_id == 1
    assert bs.last_stage_results is None


def test_wifi_first_solver_keeps_wifi_pairs_in_front_stage():
    bs = binary_search_relaxation()
    pair_radio_type = np.array([0, 0, 1, 1], dtype=int)

    wifi_idx, ble_idx = bs.split_pair_indices_by_radio_type(pair_radio_type, wifi_id=0, ble_id=1)

    assert np.array_equal(wifi_idx, np.array([0, 1], dtype=int))
    assert np.array_equal(ble_idx, np.array([2, 3], dtype=int))


def test_binary_search_slices_state_for_pair_ids():
    state = (
        scipy.sparse.csr_matrix(np.array([[1.0, 0.2], [0.3, 2.0]])),
        scipy.sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]])),
        np.array([5.0, 6.0]),
    )
    bs = binary_search_relaxation()

    sliced = bs._slice_state_for_pair_ids(state, np.array([1], dtype=int))

    assert sliced[0].shape == (1, 1)
    assert sliced[1].shape == (1, 1)
    assert np.array_equal(sliced[2], np.array([6.0]))


class _StageBs:
    def __init__(self, z_vec, z_fin, remainder):
        self._z_vec = np.asarray(z_vec, dtype=int)
        self._z_fin = int(z_fin)
        self._remainder = int(remainder)
        self.user_priority = None
        self.slot_mask_builder = None
        self.force_lower_bound = False
        self.max_slot_cap = None
        self.feasibility_check_alg = None
        self.last_partial_schedule = {
            "slot_cap_hit": False,
            "scheduled_pair_ids": list(range(len(self._z_vec) - self._remainder)),
            "unscheduled_pair_ids": list(range(len(self._z_vec) - self._remainder, len(self._z_vec))),
        }

    def run(self, state):
        return self._z_vec.copy(), self._z_fin, self._remainder


class _SplitEnv:
    RADIO_WIFI = 0
    RADIO_BLE = 1
    n_pair = 4
    pair_radio_type = np.array([0, 0, 1, 1], dtype=int)
    pair_priority = np.array([5.0, 4.0, 3.0, 2.0], dtype=float)

    def generate_S_Q_hmax(self):
        import scipy.sparse

        return (
            scipy.sparse.eye(4, format="csr"),
            scipy.sparse.csr_matrix((4, 4), dtype=float),
            np.ones(4, dtype=float),
        )

    def build_slot_compatibility_mask(self, z):
        return np.ones((4, int(z)), dtype=bool)


def test_script_no_longer_exports_wifi_first_feasibility_helper():
    assert not hasattr(pd_script, "solve_wifi_first_feasibility")


def test_run_uses_wifi_first_strategy_when_configured():
    state = (
        scipy.sparse.eye(4, format="csr"),
        scipy.sparse.csr_matrix((4, 4), dtype=float),
        np.ones(4, dtype=float),
    )
    bs = binary_search_relaxation()
    bs.strategy = "wifi_first"
    bs.pair_radio_type = np.array([0, 0, 1, 1], dtype=int)
    calls = []

    def fake_search(left, right, stage_state):
        calls.append(stage_state[0].shape[0])
        if len(calls) == 1:
            return 2, np.array([0, 1], dtype=int), 0, 1
        return 2, np.array([1, 0], dtype=int), 0, 1

    bs.search = fake_search

    z_vec, z_fin, remainder = bs.run(state)

    assert np.array_equal(z_vec, np.array([0, 1, 1, 0], dtype=int))
    assert z_fin == 2
    assert remainder == 0
    assert set(bs.last_stage_results.keys()) == {"wifi", "ble"}
