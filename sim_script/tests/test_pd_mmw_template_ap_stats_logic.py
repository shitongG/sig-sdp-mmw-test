import numpy as np

from sim_script.pd_mmw_template_ap_stats import _aggregate_ap_stats_from_arrays


def test_aggregate_ap_stats_from_arrays():
    asso = np.array([0, 0, 1, 1], dtype=int)
    radio = np.array([0, 1, 0, 1], dtype=int)  # 0=WiFi, 1=BLE
    z_vec = np.array([1, 2, 1, 3], dtype=int)

    rows = _aggregate_ap_stats_from_arrays(
        asso=asso,
        radio=radio,
        z_vec=z_vec,
        n_ap=2,
        wifi_id=0,
        ble_id=1,
    )

    assert rows[0]["wifi_user_count"] == 1
    assert rows[0]["ble_user_count"] == 1
    assert rows[0]["wifi_slots_used"] == 1
    assert rows[0]["ble_slots_used"] == 1

    assert rows[1]["wifi_user_count"] == 1
    assert rows[1]["ble_user_count"] == 1
    assert rows[1]["wifi_slots_used"] == 1
    assert rows[1]["ble_slots_used"] == 1
