import numpy as np

from sim_src.env.env import env


def test_build_slot_occupancy_mask_marks_full_wifi_and_ble_blocks():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=3,
        radio_prob=(0.5, 0.5),
        wifi_period_exp_min=4,
        wifi_period_exp_max=4,
        ble_ci_exp_min=4,
        ble_ci_exp_max=4,
        ble_phy_rate_bps=2e6,
        ble_ce_max_s=7.5e-3,
    )
    Z = 256
    mask = e.build_slot_occupancy_mask(Z)
    for k in range(e.n_pair):
        if e.pair_radio_type[k] == e.RADIO_WIFI:
            assert mask[k].any()
            assert int(mask[k].sum()) >= int(e.pair_wifi_tx_slots[k])
        else:
            if e.pair_ble_ce_feasible[k]:
                assert int(mask[k].sum()) >= int(e.pair_ble_ce_slots[k])


def test_occupancy_mask_repeats_at_period_boundaries():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=4,
        radio_prob=(1.0, 0.0),
        wifi_period_exp_min=4,
        wifi_period_exp_max=4,
    )
    Z = 128
    mask = e.build_slot_occupancy_mask(Z)
    wifi_idx = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    for k in wifi_idx:
        period = int(e.pair_wifi_period_slots[k])
        tx = int(e.pair_wifi_tx_slots[k])
        anchor = int(e.pair_wifi_anchor_slot[k] % period)
        assert mask[k, anchor:anchor + tx].all()
        if anchor + period + tx <= Z:
            assert mask[k, anchor + period:anchor + period + tx].all()


def test_compute_macrocycle_slots_is_lcm_of_active_periods():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=5,
        radio_prob=(0.5, 0.5),
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
        ble_ci_exp_min=4,
        ble_ci_exp_max=5,
        ble_phy_rate_bps=2e6,
        ble_ce_max_s=7.5e-3,
    )
    macro = e.compute_macrocycle_slots()
    for p in e.get_active_period_slots():
        assert macro % int(p) == 0
