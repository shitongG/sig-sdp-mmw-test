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


def test_script_ble_config_exposes_full_pow2_ci_range():
    e = env(
        cell_edge=7.0,
        cell_size=2,
        pair_density_per_m2=0.1,
        seed=1,
        radio_prob=(0.6, 0.4),
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=7.5e-3,
        ble_payload_bits=800,
        ble_phy_rate_bps=2e6,
        ble_ci_exp_min=3,
        ble_ci_exp_max=11,
    )
    expected = np.array([2**n for n in range(3, 12)], dtype=int)
    assert np.array_equal(e.ble_ci_quanta_candidates, expected)
    assert e.ble_phy_rate_bps == 2e6
    assert e.ble_ce_max_s == 7.5e-3
