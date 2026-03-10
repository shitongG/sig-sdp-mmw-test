import numpy as np
from sim_src.env.env import env


def test_ble_anchor_quantized_and_window_periodic():
    e = env(cell_size=2, sta_density_per_1m2=0.01, seed=7, radio_prob=(0.0, 1.0))
    Z = 300
    mask = e.build_slot_compatibility_mask(Z)
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    for k in ble_idx:
        if not e.user_ble_ce_feasible[k]:
            assert not mask[k].any()
            continue
        ci = int(e.user_ble_ci_slots[k])
        ce = int(e.user_ble_ce_slots[k])
        anchor = int(e.user_ble_anchor_slot[k])
        assert 0 <= anchor < ci
        assert abs((anchor * e.slot_time) / e.slot_time - anchor) < 1e-12
        assert mask[k, anchor:min(anchor + ce, Z)].all()


def test_ble_ce_constraints_preserved_with_2mbps_phy():
    e = env(
        cell_size=3,
        sta_density_per_1m2=0.02,
        seed=5,
        radio_prob=(0.0, 1.0),
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=7.5e-3,
        ble_payload_bits=800,
        ble_phy_rate_bps=2e6,
        ble_ci_exp_min=3,
        ble_ci_exp_max=11,
    )
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    assert ble_idx.size > 0
    assert e.ble_ce_required_s == 1.25e-3

    for k in ble_idx:
        ci_s = e.user_ble_ci_slots[k] * e.slot_time
        if not e.user_ble_ce_feasible[k]:
            assert e.user_ble_ce_slots[k] == 0
            continue
        ce_s = e.user_ble_ce_slots[k] * e.slot_time
        assert ce_s >= e.ble_ce_required_s
        assert ce_s >= e.ble_ce_min_s
        assert ce_s <= e.ble_ce_max_s + e.slot_time
        assert ce_s <= ci_s + e.slot_time
        assert np.isclose(ce_s / 1.25e-3, round(ce_s / 1.25e-3))


def test_ble_mask_exposes_window_when_ci_exceeds_schedule_horizon():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=11,
        radio_prob=(0.0, 1.0),
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=7.5e-3,
        ble_payload_bits=800,
        ble_phy_rate_bps=2e6,
        ble_ci_exp_min=11,
        ble_ci_exp_max=11,
    )
    Z = 32
    mask = e.build_slot_compatibility_mask(Z)
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    assert ble_idx.size > 0

    for k in ble_idx:
        assert e.user_ble_ci_slots[k] > Z
        assert mask[k].any()


def test_ble_mask_marks_full_ce_window():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=7,
        radio_prob=(0.0, 1.0),
        ble_ci_exp_min=4,
        ble_ci_exp_max=4,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=7.5e-3,
        ble_phy_rate_bps=2e6,
    )
    Z = 300
    mask = e.build_slot_occupancy_mask(Z)
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    for k in ble_idx:
        if not e.user_ble_ce_feasible[k]:
            assert not mask[k].any()
            continue
        anchor = int(e.user_ble_anchor_slot[k] % e.user_ble_ci_slots[k])
        ce = int(e.user_ble_ce_slots[k])
        assert mask[k, anchor:anchor + ce].all()


def test_ble_ce_range_is_quantized_to_125ms_slots():
    e = env(
        cell_size=2,
        sta_density_per_1m2=0.01,
        seed=3,
        radio_prob=(0.0, 1.0),
        ble_ce_max_s=7.5e-3,
        ble_phy_rate_bps=2e6,
    )
    ble_idx = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert ble_idx.size > 0
    ce_s = e.pair_ble_ce_slots[ble_idx] * e.slot_time
    assert np.all(ce_s >= 1.25e-3)
    assert np.all(ce_s <= 7.5e-3 + 1e-12)
    assert np.allclose(ce_s / 1.25e-3, np.round(ce_s / 1.25e-3))
