from sim_script.plot_schedule_from_csv import build_event_text_annotations


def test_event_grid_plot_labels_once_per_event():
    event_rows = [
        {
            "pair_id": 7,
            "radio": "ble",
            "channel": 3,
            "event_index": 0,
            "slot_start": 10,
            "slot_end": 14,
            "freq_low_mhz": 2408.0,
            "freq_high_mhz": 2410.0,
            "label": "7 B-ch3 ev0",
        }
    ]
    labels = build_event_text_annotations(event_rows)
    assert len(labels) == 1
    assert labels[0]["text"] == "7 B-ch3 ev0"
