import numpy as np

from sim_script.pd_mmw_template_ap_stats import build_pair_parameter_rows, build_schedule_plot_rows
from sim_src.env.env import env


def test_ble_per_ce_mode_exports_event_level_channel_rows():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=7,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )
    ble_ids = np.where(np.logical_and(e.pair_radio_type == e.RADIO_BLE, e.pair_ble_ce_feasible))[0]
    assert ble_ids.size > 0
    pair_id = int(ble_ids[0])
    macrocycle_slots = max(e.compute_macrocycle_slots(), e.pair_ble_ci_slots[pair_id] * 2)
    e.pair_ble_ce_channels[pair_id] = np.array([1, 20], dtype=int)

    schedule_start_slots = np.full(e.n_pair, -1, dtype=int)
    schedule_start_slots[pair_id] = 0
    occupied = np.zeros((e.n_pair, macrocycle_slots), dtype=bool)
    occupied[pair_id] = e.expand_pair_occupancy(pair_id, 0, macrocycle_slots)

    pair_rows = build_pair_parameter_rows(
        pair_office_id=e.pair_office_id,
        pair_radio_type=e.pair_radio_type,
        pair_channel=e.pair_channel,
        pair_priority=e.pair_priority,
        ble_channel_mode=e.ble_channel_mode,
        pair_ble_ce_channels=e.pair_ble_ce_channels,
        pair_start_time_slot=e.pair_start_time_slot,
        pair_wifi_anchor_slot=e.pair_wifi_anchor_slot,
        pair_wifi_period_slots=e.pair_wifi_period_slots,
        pair_wifi_tx_slots=e.pair_wifi_tx_slots,
        pair_ble_anchor_slot=e.pair_ble_anchor_slot,
        pair_ble_ci_slots=e.pair_ble_ci_slots,
        pair_ble_ce_slots=e.pair_ble_ce_slots,
        pair_ble_ce_feasible=e.pair_ble_ce_feasible,
        z_vec=schedule_start_slots,
        occupied_slots=occupied,
        macrocycle_slots=macrocycle_slots,
        slot_time=e.slot_time,
        wifi_id=e.RADIO_WIFI,
        ble_id=e.RADIO_BLE,
    )
    pair_rows = [row for row in pair_rows if row["pair_id"] == pair_id]
    rows = build_schedule_plot_rows(pair_rows, {}, e=e)

    assert len(rows) > 0
    assert len({row["channel"] for row in rows}) >= 2
    assert any("ev0" in row["label"] for row in rows)
    assert any("ev1" in row["label"] for row in rows)


def test_build_schedule_plot_rows_emits_ble_overlap_rows():
    pair_rows = [
        {
            "pair_id": 0,
            "radio": "ble",
            "channel": 5,
            "schedule_slot": 0,
            "macrocycle_slots": 4,
            "occupied_slots_in_macrocycle": [0],
        },
        {
            "pair_id": 1,
            "radio": "ble",
            "channel": 6,
            "schedule_slot": 0,
            "macrocycle_slots": 4,
            "occupied_slots_in_macrocycle": [0],
        },
    ]
    pair_channel_ranges = {
        0: (2440.0, 2444.0),
        1: (2442.0, 2446.0),
    }

    rows = build_schedule_plot_rows(pair_rows, pair_channel_ranges, e=None)
    overlap_rows = [row for row in rows if row["radio"] == "ble_overlap"]

    assert overlap_rows == [
        {
            "pair_id": -1,
            "radio": "ble_overlap",
            "channel": -1,
            "slot": 0,
            "freq_low_mhz": 2442.0,
            "freq_high_mhz": 2444.0,
            "label": "BLE overlap 0/1",
        }
    ]


def test_ble_per_ce_plot_rows_wrap_across_macrocycle_boundary():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=23,
        radio_prob=(0.0, 1.0),
        ble_channel_mode="per_ce",
        ble_phy_rate_bps=2e6,
    )
    pair_id = int(np.where(np.logical_and(e.pair_radio_type == e.RADIO_BLE, e.pair_ble_ce_feasible))[0][0])
    e.pair_ble_ci_slots[pair_id] = 32
    e.pair_ble_ce_slots[pair_id] = 5
    e.pair_ble_ce_channels[pair_id] = np.array([2], dtype=int)

    pair_rows = [
        {
            "pair_id": pair_id,
            "radio": "ble",
            "channel": int(e.pair_channel[pair_id]),
            "schedule_slot": 31,
            "macrocycle_slots": 32,
            "occupied_slots_in_macrocycle": [31, 0, 1, 2, 3],
        }
    ]

    rows = build_schedule_plot_rows(pair_rows, {}, e=e)
    slots = sorted(row["slot"] for row in rows if row["radio"] == "ble")

    assert slots == [0, 1, 2, 3, 31]
