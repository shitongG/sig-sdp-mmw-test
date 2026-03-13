import pathlib

from sim_script.plot_schedule_from_csv import render_event_grid_plot


def test_render_overview_plot_writes_png_and_places_one_label_per_event(tmp_path):
    spans = [{
        "pair_id": 7, "radio": "ble", "channel": 3, "event_index": 0,
        "slot_start": 10, "slot_end": 14,
        "freq_low_mhz": 2408.0, "freq_high_mhz": 2410.0,
        "label": "7 B-ch3 ev0",
    }]
    out = pathlib.Path(tmp_path) / "overview.png"
    render_event_grid_plot(spans, out, macrocycle_slots=64)
    assert out.exists()
