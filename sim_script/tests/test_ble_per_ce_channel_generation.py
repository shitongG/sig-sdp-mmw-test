import numpy as np

from sim_src.env.env import env


def test_ble_per_ce_mode_assigns_channels_per_connection_event():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=3,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    feasible_ble_ids = [int(pair_id) for pair_id in ble_ids if e.pair_ble_ce_feasible[pair_id]]
    assert feasible_ble_ids
    pair_id = feasible_ble_ids[0]
    ce_channels = e.pair_ble_ce_channels[pair_id]

    assert len(ce_channels) > 0
    assert all(0 <= ch < e.ble_channel_count for ch in ce_channels)
    assert e.pair_ble_ci_slots[pair_id] > 0
    assert e.pair_ble_ce_slots[pair_id] > 0


def test_ble_per_ce_mode_uses_injected_channel_map_for_expansion():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=3,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    feasible_ble_ids = [int(pair_id) for pair_id in ble_ids if e.pair_ble_ce_feasible[pair_id]]
    assert feasible_ble_ids
    pair_id = feasible_ble_ids[0]
    macrocycle_slots = max(int(e.compute_macrocycle_slots()), int(e.pair_ble_ci_slots[pair_id]))
    event_count = max(1, macrocycle_slots // int(e.pair_ble_ci_slots[pair_id]))
    manual_channels = np.array([(3 + idx * 4) % e.ble_channel_count for idx in range(event_count)], dtype=int)

    e.set_ble_ce_channel_map({pair_id: manual_channels})
    instances = e.expand_pair_event_instances(pair_id, macrocycle_slots=macrocycle_slots)

    assert e.pair_ble_ce_channels[pair_id].tolist() == manual_channels.tolist()
    assert [inst["channel"] for inst in instances] == manual_channels.tolist()


def test_ble_per_ce_mode_preserves_injected_channel_map_during_resample():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=3,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    feasible_ble_ids = [int(pair_id) for pair_id in ble_ids if e.pair_ble_ce_feasible[pair_id]]
    assert feasible_ble_ids
    pair_id = feasible_ble_ids[0]
    macrocycle_slots = max(int(e.compute_macrocycle_slots()), int(e.pair_ble_ci_slots[pair_id]))
    event_count = max(1, macrocycle_slots // int(e.pair_ble_ci_slots[pair_id]))
    manual_channels = np.array([(7 + idx * 5) % e.ble_channel_count for idx in range(event_count)], dtype=int)

    e.set_ble_ce_channel_map({pair_id: manual_channels})
    e.resample_ble_channels(np.array([pair_id], dtype=int))

    assert e.pair_ble_ce_channels[pair_id].tolist() == manual_channels.tolist()
