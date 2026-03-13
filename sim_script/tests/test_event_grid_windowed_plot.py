from sim_script.plot_schedule_from_csv import render_windowed_event_grid_plots


def test_render_windowed_plots_splits_macrocycle_into_multiple_pngs(tmp_path):
    spans = [
        {"pair_id": 1, "radio": "wifi", "channel": 9, "slot_start": 0, "slot_end": 20,
         "freq_low_mhz": 2447.0, "freq_high_mhz": 2467.0, "label": "1 W-ch9"},
        {"pair_id": 2, "radio": "ble", "channel": 3, "event_index": 0, "slot_start": 130, "slot_end": 134,
         "freq_low_mhz": 2408.0, "freq_high_mhz": 2410.0, "label": "2 B-ch3 ev0"},
    ]
    outputs = render_windowed_event_grid_plots(spans, tmp_path, macrocycle_slots=256, window_slots=128)
    assert len(outputs) == 2
