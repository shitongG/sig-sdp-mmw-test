import numpy as np

from sim_script.pd_mmw_template_ap_stats import _is_better_refill_result


def test_refill_prefers_fewer_unscheduled_pairs():
    priorities = np.array([3.0, 2.0, 1.0], dtype=float)
    best = (None, None, None, [0, 1])
    candidate = (None, None, None, [2])
    assert _is_better_refill_result(candidate, best, priorities)


def test_refill_prefers_saving_higher_priority_pairs_when_counts_tie():
    priorities = np.array([3.0, 2.0, 1.0], dtype=float)
    best = (None, None, None, [0])
    candidate = (None, None, None, [2])
    assert _is_better_refill_result(candidate, best, priorities)
