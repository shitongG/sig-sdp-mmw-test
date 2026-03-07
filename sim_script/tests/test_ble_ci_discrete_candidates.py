import numpy as np
from sim_src.env.env import env


def test_ble_ci_candidates_follow_pow2_rule():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.005,
        seed=1,
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
    )
    expected = np.array([2**n for n in range(3, 12)], dtype=int)
    assert np.array_equal(e.ble_ci_quanta_candidates, expected)
