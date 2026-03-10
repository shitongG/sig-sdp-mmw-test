import numpy as np

from sim_src.env.env import env


def test_base_slot_time_is_1250us():
    e = env(cell_size=2, sta_density_per_1m2=0.01, seed=1)
    assert e.slot_time == 1.25e-3


def test_wifi_period_candidates_follow_pow2_rule():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=1,
        radio_prob=(1.0, 0.0),
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
    )
    expected = np.array([16, 32], dtype=int)
    assert np.array_equal(e.wifi_period_quanta_candidates, expected)


def test_wifi_tx_duration_candidates_follow_125ms_grid():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=2,
        radio_prob=(1.0, 0.0),
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
    )
    expected_ms = np.array([5.0, 6.25, 7.5, 8.75, 10.0])
    actual_ms = e.wifi_tx_quanta_candidates * e.slot_time * 1e3
    assert np.allclose(actual_ms, expected_ms)


def test_wifi_pairs_sample_periods_and_tx_duration_candidates():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=2,
        radio_prob=(1.0, 0.0),
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
    )
    wifi_idx = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    assert wifi_idx.size > 0
    period_quanta = np.rint((e.pair_wifi_period_slots[wifi_idx] * e.slot_time) / 1.25e-3).astype(int)
    assert np.all(np.isin(period_quanta, [16, 32]))
    assert set(e.pair_wifi_tx_slots[wifi_idx]).issubset({4, 5, 6, 7, 8})
