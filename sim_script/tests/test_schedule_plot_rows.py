from sim_script.pd_mmw_template_ap_stats import build_schedule_plot_rows


def test_schedule_plot_rows_include_slot_channel_and_frequency_band():
    pair_rows = [
        {
            "pair_id": 3,
            "radio": "ble",
            "channel": 19,
            "occupied_slots_in_macrocycle": [5, 6],
        }
    ]
    rows = build_schedule_plot_rows(pair_rows, pair_channel_ranges={3: (2440.0, 2442.0)})
    assert rows[0]["slot"] == 5
    assert rows[0]["pair_id"] == 3
    assert rows[0]["freq_low_mhz"] == 2440.0
    assert rows[0]["freq_high_mhz"] == 2442.0
    assert rows[0]["label"] == "3 B-ch19"
