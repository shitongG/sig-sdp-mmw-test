from sim_script.plot_schedule_from_csv import group_slot_rows_into_event_spans


def test_group_contiguous_slot_rows_into_one_event_span():
    rows = [
        {"pair_id": "1", "radio": "wifi", "channel": "9", "slot": "10", "freq_low_mhz": "2447.0", "freq_high_mhz": "2467.0", "label": "1 W-ch9"},
        {"pair_id": "1", "radio": "wifi", "channel": "9", "slot": "11", "freq_low_mhz": "2447.0", "freq_high_mhz": "2467.0", "label": "1 W-ch9"},
    ]
    spans = group_slot_rows_into_event_spans(rows)
    assert len(spans) == 1
    assert spans[0]["slot_start"] == 10
    assert spans[0]["slot_end"] == 12
