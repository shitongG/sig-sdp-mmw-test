import numpy as np

from sim_src.env.env import env


def test_expand_pair_occupancy_wraps_across_macrocycle_boundary():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=21,
        radio_prob=(1.0, 0.0),
    )
    pair_id = int(np.where(e.pair_radio_type == e.RADIO_WIFI)[0][0])
    e.pair_wifi_period_slots[pair_id] = 32
    e.pair_wifi_tx_slots[pair_id] = 5

    occ = e.expand_pair_occupancy(pair_id, start_slot=31, macrocycle_slots=32)

    assert np.where(occ)[0].tolist() == [0, 1, 2, 3, 31]


def test_expand_pair_event_instances_wraps_last_event():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=22,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )
    pair_id = int(np.where(np.logical_and(e.pair_radio_type == e.RADIO_BLE, e.pair_ble_ce_feasible))[0][0])
    e.pair_ble_ci_slots[pair_id] = 32
    e.pair_ble_ce_slots[pair_id] = 5
    e.pair_ble_ce_channels[pair_id] = np.array([3], dtype=int)

    instances = e.expand_pair_event_instances(pair_id, macrocycle_slots=32, start_slot=31)

    assert instances[0]["slot_range"] == (31, 36)
    assert instances[0]["wrapped_slot_ranges"] == [(31, 32), (0, 4)]
