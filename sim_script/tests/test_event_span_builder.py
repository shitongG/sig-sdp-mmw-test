from sim_script.plot_schedule_from_csv import build_ble_event_spans


def test_build_event_spans_from_ble_event_csv():
    rows = [{
        "pair_id": "5", "event_index": "2", "channel": "17",
        "slot_start": "20", "slot_end": "24",
        "freq_low_mhz": "2435.0", "freq_high_mhz": "2437.0",
    }]
    spans = build_ble_event_spans(rows)
    assert spans[0]["pair_id"] == 5
    assert spans[0]["slot_start"] == 20
    assert spans[0]["slot_end"] == 24
    assert spans[0]["label"] == "5 B-ch17 ev2"
