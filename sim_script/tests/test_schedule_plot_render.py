import pathlib
import tempfile

import matplotlib.pyplot as plt

from sim_script.pd_mmw_template_ap_stats import render_schedule_plot


def test_render_schedule_plot_writes_png_file():
    rows = [
        {
            "pair_id": 3,
            "radio": "ble",
            "slot": 5,
            "freq_low_mhz": 2440.0,
            "freq_high_mhz": 2442.0,
            "channel": 19,
            "label": "3 B-ch19",
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = pathlib.Path(tmpdir) / "schedule_plot.png"
        render_schedule_plot(rows, out, macrocycle_slots=16)
        assert out.exists()


def test_render_schedule_plot_adds_text_labels():
    rows = [
        {
            "pair_id": 3,
            "radio": "ble",
            "slot": 5,
            "freq_low_mhz": 2440.0,
            "freq_high_mhz": 2444.0,
            "channel": 19,
            "label": "3 B-ch19",
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = pathlib.Path(tmpdir) / "schedule_plot.png"
        original_close = plt.close
        try:
            plt.close = lambda *args, **kwargs: None
            render_schedule_plot(rows, out, macrocycle_slots=16)
            fig = plt.gcf()
            assert fig.axes[0].texts
        finally:
            plt.close = original_close
            plt.close("all")
