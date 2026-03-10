import pathlib
import tempfile

import matplotlib.pyplot as plt

from sim_script.pd_mmw_template_ap_stats import render_schedule_plot


def test_render_schedule_plot_uses_full_macrocycle_xlim():
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
        original_close = plt.close
        try:
            plt.close = lambda *args, **kwargs: None
            render_schedule_plot(rows, out, macrocycle_slots=64)
            fig = plt.gcf()
            xmin, xmax = fig.axes[0].get_xlim()
            assert xmin == 0
            assert xmax == 64
        finally:
            plt.close = original_close
            plt.close("all")


def test_render_schedule_plot_keeps_empty_tail_slots_visible():
    rows = [
        {
            "pair_id": 8,
            "radio": "wifi",
            "slot": 1,
            "freq_low_mhz": 2420.0,
            "freq_high_mhz": 2440.0,
            "channel": 3,
            "label": "8 W-ch3",
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = pathlib.Path(tmpdir) / "schedule_plot.png"
        original_close = plt.close
        try:
            plt.close = lambda *args, **kwargs: None
            render_schedule_plot(rows, out, macrocycle_slots=32)
            fig = plt.gcf()
            xmin, xmax = fig.axes[0].get_xlim()
            assert xmax - xmin == 32
        finally:
            plt.close = original_close
            plt.close("all")
