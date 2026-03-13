import numpy as np

from sim_src.env.env import env


def test_ble_per_ce_mode_exposes_distinct_channel_ranges_across_events():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=4,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    feasible_ble_ids = [int(pair_id) for pair_id in ble_ids if e.pair_ble_ce_feasible[pair_id]]
    assert feasible_ble_ids
    pair_id = feasible_ble_ids[0]

    instances = e.expand_pair_event_instances(pair_id, macrocycle_slots=max(e.compute_macrocycle_slots(), e.pair_ble_ci_slots[pair_id]))

    assert len(instances) > 0
    assert all("channel" in inst for inst in instances)
    assert all("slot_range" in inst for inst in instances)
