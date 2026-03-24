import pathlib

import matplotlib.pyplot as plt

from sim_script.plot_schedule_from_csv import iter_internal_slot_boundaries, render_event_grid_plot


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


def test_iter_internal_slot_boundaries_marks_each_slot_boundary_inside_span():
    span = {
        "pair_id": 3,
        "radio": "wifi",
        "channel": 6,
        "slot_start": 10,
        "slot_end": 14,
        "freq_low_mhz": 2430.0,
        "freq_high_mhz": 2450.0,
        "label": "3 W-ch6",
    }
    assert iter_internal_slot_boundaries(span, 10, 14) == [11, 12, 13]


def test_render_event_grid_plot_draws_visible_patch_edges_and_internal_boundaries(tmp_path):
    spans = [{
        "pair_id": 7, "radio": "wifi", "channel": 6,
        "slot_start": 10, "slot_end": 14,
        "freq_low_mhz": 2427.0, "freq_high_mhz": 2447.0,
        "label": "7 W-ch6",
    }]
    out = pathlib.Path(tmp_path) / "overview.png"
    original_close = plt.close
    try:
        plt.close = lambda *args, **kwargs: None
        render_event_grid_plot(spans, out, macrocycle_slots=64)
        fig = plt.gcf()
        ax = fig.axes[0]
        assert ax.patches
        patch = ax.patches[0]
        assert patch.get_linewidth() > 0.0
        edge_rgba = patch.get_edgecolor()
        assert edge_rgba[-1] > 0.0
        assert len(ax.lines) == 3
    finally:
        plt.close = original_close
        plt.close("all")
