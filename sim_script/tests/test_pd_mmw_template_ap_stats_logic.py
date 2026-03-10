import numpy as np

from sim_script.pd_mmw_template_ap_stats import (
    _aggregate_office_stats_from_arrays,
    build_pair_parameter_rows,
    build_schedule_rows,
)


def test_aggregate_office_stats_from_arrays():
    office_id = np.array([0, 0, 1, 1], dtype=int)
    radio = np.array([0, 1, 0, 1], dtype=int)  # 0=WiFi, 1=BLE
    z_vec = np.array([1, 2, 1, 3], dtype=int)

    rows = _aggregate_office_stats_from_arrays(
        office_id=office_id,
        radio=radio,
        z_vec=z_vec,
        n_office=2,
        wifi_id=0,
        ble_id=1,
    )

    assert rows[0]["wifi_pair_count"] == 1
    assert rows[0]["ble_pair_count"] == 1
    assert rows[0]["wifi_slots_used"] == 1
    assert rows[0]["ble_slots_used"] == 1

    assert rows[1]["wifi_pair_count"] == 1
    assert rows[1]["ble_pair_count"] == 1
    assert rows[1]["wifi_slots_used"] == 1
    assert rows[1]["ble_slots_used"] == 1


def test_build_pair_parameter_rows_contains_wifi_and_ble_fields():
    rows = build_pair_parameter_rows(
        pair_office_id=np.array([0, 0]),
        pair_radio_type=np.array([0, 1]),
        pair_channel=np.array([6, 12]),
        pair_priority=np.array([3.0, 1.0]),
        pair_start_time_slot=np.array([0, 0]),
        pair_wifi_anchor_slot=np.array([4, 0]),
        pair_wifi_period_slots=np.array([16, 0]),
        pair_wifi_tx_slots=np.array([5, 0]),
        pair_ble_anchor_slot=np.array([0, 4]),
        pair_ble_ci_slots=np.array([0, 16]),
        pair_ble_ce_slots=np.array([0, 3]),
        pair_ble_ce_feasible=np.array([True, True]),
        z_vec=np.array([2, 5]),
        occupied_slots=np.array(
            [
                [False, False, True, True, False, False],
                [False, False, False, False, False, True],
            ],
            dtype=bool,
        ),
        macrocycle_slots=6,
        slot_time=1.25e-3,
        wifi_id=0,
        ble_id=1,
    )

    assert rows[0]["pair_id"] == 0
    assert rows[0]["radio"] == "wifi"
    assert rows[0]["schedule_slot"] == 2
    assert rows[0]["channel"] == 6
    assert rows[0]["start_time_slot"] == 0
    assert rows[0]["wifi_period_slots"] == 16
    assert rows[0]["wifi_period_ms"] == 20.0
    assert rows[0]["wifi_tx_slots"] == 5
    assert rows[0]["wifi_tx_ms"] == 6.25
    assert rows[0]["ble_ci_slots"] is None

    assert rows[1]["pair_id"] == 1
    assert rows[1]["radio"] == "ble"
    assert rows[1]["schedule_slot"] == 5
    assert rows[1]["start_time_slot"] == 0
    assert rows[1]["ble_ci_slots"] == 16
    assert rows[1]["ble_ce_slots"] == 3
    assert rows[1]["ble_anchor_slot"] == 4
    assert rows[1]["ble_ci_ms"] == 20.0
    assert rows[1]["ble_ce_ms"] == 3.75


def test_build_schedule_rows_orders_by_min_schedule_slot():
    pair_rows = [
        {"pair_id": 3, "radio": "ble", "schedule_slot": 5, "office_id": 1, "channel": 2, "occupied_slots_in_macrocycle": [5]},
        {"pair_id": 0, "radio": "wifi", "schedule_slot": 1, "office_id": 0, "channel": 6, "occupied_slots_in_macrocycle": [1]},
        {"pair_id": 2, "radio": "wifi", "schedule_slot": 5, "office_id": 1, "channel": 1, "occupied_slots_in_macrocycle": [5]},
    ]

    rows = build_schedule_rows(pair_rows)

    assert rows[0]["schedule_slot"] == 1
    assert rows[0]["pair_ids"] == [0]
    assert rows[0]["wifi_pair_ids"] == [0]
    assert rows[0]["ble_pair_ids"] == []

    assert rows[1]["schedule_slot"] == 5
    assert rows[1]["pair_ids"] == [2, 3]
    assert rows[1]["wifi_pair_ids"] == [2]
    assert rows[1]["ble_pair_ids"] == [3]
