import pathlib

import matplotlib.pyplot as plt

from sim_script.plot_schedule_from_csv import render_event_grid_plot


def test_render_event_grid_plot_adds_ble_overlap_legend_entry(tmp_path):
    spans = [
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
        },
        {
            "pair_id": -1,
            "radio": "ble_overlap",
            "channel": -1,
            "event_index": -1,
            "slot_start": 11,
            "slot_end": 13,
            "freq_low_mhz": 2408.5,
            "freq_high_mhz": 2409.5,
            "label": "BLE overlap 7/9",
        },
    ]
    out = pathlib.Path(tmp_path) / "overview.png"
    original_close = plt.close
    try:
        plt.close = lambda *args, **kwargs: None
        render_event_grid_plot(spans, out, macrocycle_slots=64)
        fig = plt.gcf()
        legend = fig.axes[0].get_legend()
        assert legend is not None
        assert any(text.get_text() == "BLE overlap" for text in legend.get_texts())
    finally:
        plt.close = original_close
        plt.close("all")
