import numpy as np

from sim_src.alg.mmw import mmw
from sim_src.env.env import env


def test_mmw_cpu_baseline_returns_stable_shape_and_finite_values():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=3,
        slot_time=1.25e-3,
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
        ble_ce_max_s=7.5e-3,
        ble_phy_rate_bps=2e6,
    )
    state = e.generate_S_Q_hmax()
    alg = mmw(nit=3, eta=0.05)
    ok, x_half = alg.run_with_state(0, 4, state)
    assert ok is True
    assert x_half.shape[0] == e.n_pair
    assert np.isfinite(x_half).all()
