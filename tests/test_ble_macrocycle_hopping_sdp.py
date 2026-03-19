import os
import io
import contextlib
import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "ble_macrocycle_hopping_sdp.py"
SPEC = importlib.util.spec_from_file_location("ble_macrocycle_hopping_sdp", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EventBlockExpansionTest(unittest.TestCase):
    def test_build_sdp_relaxation_annotation_does_not_reference_runtime_cp_alias(self):
        return_annotation = MODULE.build_sdp_relaxation.__annotations__["return"]
        self.assertNotIn("cp.", str(return_annotation))

    def test_solve_ble_hopping_schedule_exposes_selected_blocks_and_objective(self):
        pair_configs, cfg_dict, pattern_dict, pair_weight, num_channels = MODULE.build_demo_instance()
        pair_ids = [cfg.pair_id for cfg in pair_configs]
        states, _, A_k = MODULE.build_candidate_states(pair_configs, pattern_dict)

        result = MODULE.solve_ble_hopping_schedule(
            pair_configs=pair_configs,
            cfg_dict=cfg_dict,
            pattern_dict=pattern_dict,
            pair_ids=pair_ids,
            A_k=A_k,
            states=states,
            num_channels=num_channels,
            pair_weight=pair_weight,
        )

        self.assertIn("selected", result)
        self.assertIn("blocks", result)
        self.assertIn("overlap_blocks", result)
        self.assertIn("objective_value", result)
        self.assertTrue(result["selected"])
        self.assertTrue(result["blocks"])

    def test_selected_schedule_to_ce_channels_returns_per_pair_channel_map(self):
        pair_configs, cfg_dict, pattern_dict, pair_weight, num_channels = MODULE.build_demo_instance()
        pair_ids = [cfg.pair_id for cfg in pair_configs]
        states, _, A_k = MODULE.build_candidate_states(pair_configs, pattern_dict)
        selected = {k: states[idxs[0]] for k, idxs in A_k.items()}

        ce_channel_map = MODULE.selected_schedule_to_ce_channels(
            selected=selected,
            cfg_dict=cfg_dict,
            pattern_dict=pattern_dict,
            num_channels=num_channels,
        )

        self.assertEqual(set(ce_channel_map), set(pair_ids))
        self.assertTrue(all(hasattr(channels, "tolist") for channels in ce_channel_map.values()))

    def test_selected_schedule_expands_to_event_blocks(self):
        cfg = MODULE.PairConfig(
            pair_id=7,
            release_time=0,
            deadline=10,
            connect_interval=3,
            event_duration=2,
            num_events=3,
        )
        pattern = MODULE.HoppingPattern(pattern_id=2, start_channel=5, hop_increment=4)
        selected = {7: MODULE.CandidateState(pair_id=7, offset=1, pattern_id=2)}
        cfg_dict = {7: cfg}
        pattern_dict = {7: [pattern]}

        blocks = MODULE.build_event_blocks(selected, cfg_dict, pattern_dict, num_channels=37)

        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0].pair_id, 7)
        self.assertEqual(blocks[0].event_index, 0)
        self.assertEqual(blocks[0].start_slot, 1)
        self.assertEqual(blocks[0].end_slot, 2)
        self.assertEqual(blocks[0].channel, 5)
        self.assertEqual(blocks[0].frequency_mhz, 2412.0)
        self.assertEqual(blocks[1].start_slot, 4)
        self.assertEqual(blocks[1].channel, 9)
        self.assertEqual(blocks[1].frequency_mhz, 2420.0)

    def test_same_channel_time_overlap_builds_overlap_block(self):
        blocks = [
            MODULE.EventBlock(
                pair_id=0,
                event_index=0,
                start_slot=2,
                end_slot=5,
                channel=10,
                frequency_mhz=2422.0,
                offset=2,
                pattern_id=0,
            ),
            MODULE.EventBlock(
                pair_id=1,
                event_index=0,
                start_slot=4,
                end_slot=6,
                channel=10,
                frequency_mhz=2422.0,
                offset=4,
                pattern_id=1,
            ),
        ]

        overlaps = MODULE.build_overlap_blocks(blocks)

        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0].start_slot, 4)
        self.assertEqual(overlaps[0].end_slot, 5)
        self.assertEqual(overlaps[0].channel, 10)
        self.assertEqual(overlaps[0].frequency_mhz, 2422.0)

    def test_same_overlap_window_is_deduplicated(self):
        blocks = [
            MODULE.EventBlock(
                pair_id=0,
                event_index=0,
                start_slot=2,
                end_slot=5,
                channel=10,
                frequency_mhz=2422.0,
                offset=2,
                pattern_id=0,
            ),
            MODULE.EventBlock(
                pair_id=1,
                event_index=0,
                start_slot=2,
                end_slot=5,
                channel=10,
                frequency_mhz=2422.0,
                offset=2,
                pattern_id=1,
            ),
            MODULE.EventBlock(
                pair_id=2,
                event_index=0,
                start_slot=2,
                end_slot=5,
                channel=10,
                frequency_mhz=2422.0,
                offset=2,
                pattern_id=2,
            ),
        ]

        overlaps = MODULE.build_overlap_blocks(blocks)

        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0].start_slot, 2)
        self.assertEqual(overlaps[0].end_slot, 5)
        self.assertEqual(overlaps[0].channel, 10)

    def test_render_event_grid_writes_png(self):
        blocks = [
            MODULE.EventBlock(
                pair_id=3,
                event_index=1,
                start_slot=8,
                end_slot=10,
                channel=12,
                frequency_mhz=2426.0,
                offset=2,
                pattern_id=0,
            )
        ]
        overlaps = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "schedule.png"
            with mock.patch.object(matplotlib, "use", wraps=matplotlib.use) as use_mock:
                MODULE.render_event_grid(blocks, overlaps, output_path, title="Test Grid")
            use_mock.assert_called_once_with("Agg", force=True)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_demo_blocks_are_printable_for_main_flow(self):
        pair_configs, cfg_dict, pattern_dict, _, num_channels = MODULE.build_demo_instance()
        states, _, A_k = MODULE.build_candidate_states(pair_configs, pattern_dict)
        selected = {k: states[idxs[0]] for k, idxs in A_k.items()}

        blocks = MODULE.build_event_blocks(selected, cfg_dict, pattern_dict, num_channels)

        self.assertGreater(len(blocks), 0)
        self.assertTrue(all(hasattr(block, "frequency_mhz") for block in blocks))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            MODULE.print_event_block_table(blocks)

        printed = buf.getvalue()
        self.assertIn("事件级时频块表", printed)
        self.assertIn("pair=0", printed)

    def test_render_selected_schedule_smoke_without_solver(self):
        pair_configs, cfg_dict, pattern_dict, _, num_channels = MODULE.build_demo_instance()
        states, _, A_k = MODULE.build_candidate_states(pair_configs, pattern_dict)
        selected = {k: states[idxs[0]] for k, idxs in A_k.items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "selected-grid.png"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                blocks, overlaps = MODULE.render_selected_schedule(
                    selected=selected,
                    cfg_dict=cfg_dict,
                    pattern_dict=pattern_dict,
                    num_channels=num_channels,
                    output_path=output_path,
                    title="Smoke Grid",
                )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertGreater(len(blocks), 0)
            self.assertGreaterEqual(len(overlaps), 0)
            self.assertIn("事件级时频块表", buf.getvalue())
            self.assertIn("pair=0", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
