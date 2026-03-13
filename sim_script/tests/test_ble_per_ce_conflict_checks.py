import numpy as np
import scipy.sparse

from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots
from sim_src.env.env import env


def test_ble_per_ce_mode_allows_same_slot_when_event_channels_do_not_overlap():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=5,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )
    ble_ids = np.where(np.logical_and(e.pair_radio_type == e.RADIO_BLE, e.pair_ble_ce_feasible))[0]
    assert ble_ids.size >= 2
    pair_a, pair_b = int(ble_ids[0]), int(ble_ids[1])

    e.pair_ble_ci_slots[pair_a] = 8
    e.pair_ble_ci_slots[pair_b] = 8
    e.pair_ble_ce_slots[pair_a] = 1
    e.pair_ble_ce_slots[pair_b] = 1
    e.pair_ble_anchor_slot[pair_a] = 0
    e.pair_ble_anchor_slot[pair_b] = 0
    e.pair_channel[pair_a] = 0
    e.pair_channel[pair_b] = 0
    e.pair_ble_ce_channels[pair_a] = np.array([0], dtype=int)
    e.pair_ble_ce_channels[pair_b] = np.array([20], dtype=int)

    static_conflict = scipy.sparse.lil_matrix((e.n_pair, e.n_pair), dtype=bool)
    static_conflict[pair_a, pair_b] = True
    static_conflict[pair_b, pair_a] = True
    e.build_pair_conflict_matrix = lambda: static_conflict.tocsr()
    e.get_macrocycle_conflict_state = lambda: (
        scipy.sparse.csr_matrix((e.n_pair, e.n_pair), dtype=float),
        scipy.sparse.csr_matrix((e.n_pair, e.n_pair), dtype=bool),
        np.full(e.n_pair, 1e9, dtype=float),
    )

    starts, macrocycle_slots, occ, unscheduled = assign_macrocycle_start_slots(
        e,
        preferred_slots=np.zeros(e.n_pair, dtype=int),
        allow_partial=True,
        pair_order=np.array([pair_a, pair_b], dtype=int),
    )

    assert pair_a not in unscheduled
    assert pair_b not in unscheduled
    assert starts[pair_a] == 0
    assert starts[pair_b] == 0


def test_ble_per_ce_wraparound_event_conflicts_with_wifi_slot_after_cycle_boundary():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=6,
        radio_prob=(1.0, 0.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )

    e.pair_radio_type[:] = e.RADIO_BLE
    e.pair_radio_type[0] = e.RADIO_WIFI
    e.pair_radio_type[1] = e.RADIO_BLE

    wifi_id = 0
    ble_id = 1
    e.pair_channel[wifi_id] = 1
    e.pair_channel[ble_id] = 0
    e.pair_wifi_period_slots[wifi_id] = 64
    e.pair_wifi_tx_slots[wifi_id] = 5
    e.pair_ble_ci_slots[ble_id] = 64
    e.pair_ble_ce_slots[ble_id] = 3
    e.pair_ble_ce_feasible[ble_id] = True
    e.pair_ble_ce_channels[ble_id] = np.array([5], dtype=int)  # overlaps WiFi ch1

    assert e.get_pair_channel_for_slot(wifi_id, slot=0, start_slot=63) == 1
    assert e.get_pair_channel_for_slot(ble_id, slot=0, start_slot=63) == 5
    assert e.is_slot_channel_conflict(wifi_id, start_a=63, pair_b=ble_id, start_b=63, slot=0) is True
