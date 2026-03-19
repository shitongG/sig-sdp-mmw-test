import importlib.util
import pathlib
import sys
import unittest

import numpy as np


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "test.py"
SPEC = importlib.util.spec_from_file_location("ble_state_scheduler", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BleStateSchedulerTest(unittest.TestCase):
    def test_collision_matrix_counts_same_channel_overlap(self):
        config = MODULE.SchedulingConfig(
            pairs=(
                MODULE.PairConfig(
                    pair_id=0,
                    offset_candidates=(0,),
                    pattern_count=1,
                    interval=4,
                    duration=2,
                    event_count=2,
                    channel_seed=0,
                ),
                MODULE.PairConfig(
                    pair_id=1,
                    offset_candidates=(1,),
                    pattern_count=1,
                    interval=4,
                    duration=2,
                    event_count=2,
                    channel_seed=7,
                ),
            ),
            channel_count=8,
        )

        model = MODULE.build_model(config)
        expected = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)
        np.testing.assert_allclose(model.collision_matrix, expected)

    def test_bruteforce_solver_finds_zero_collision_schedule(self):
        config = MODULE.SchedulingConfig(
            pairs=(
                MODULE.PairConfig(
                    pair_id=0,
                    offset_candidates=(0, 1),
                    pattern_count=2,
                    interval=4,
                    duration=2,
                    event_count=2,
                    channel_seed=0,
                ),
                MODULE.PairConfig(
                    pair_id=1,
                    offset_candidates=(0, 1),
                    pattern_count=2,
                    interval=4,
                    duration=2,
                    event_count=2,
                    channel_seed=3,
                ),
            ),
            channel_count=8,
        )

        model = MODULE.build_model(config)
        solution = MODULE.solve_bruteforce(model)

        self.assertEqual(solution.total_collision, 0.0)
        self.assertEqual(set(solution.selected_states.keys()), {0, 1})


if __name__ == "__main__":
    unittest.main()
