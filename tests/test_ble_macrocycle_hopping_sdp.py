import os
import io
import contextlib
import importlib.util
import inspect
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
import numpy as np


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "ble_macrocycle_hopping_sdp.py"
SPEC = importlib.util.spec_from_file_location("ble_macrocycle_hopping_sdp", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EventBlockExpansionTest(unittest.TestCase):
    def test_readme_documents_inputs_and_optimization_variables(self):
        readme = pathlib.Path("README.md").read_text(encoding="utf-8")

        self.assertIn("输入参数与优化变量", readme)
        self.assertIn("输入参数（Inputs）", readme)
        self.assertIn("优化变量（Decision Variables）", readme)
        self.assertIn("主调度脚本", readme)
        self.assertIn("BLE-only 宏周期跳频求解器", readme)
        self.assertIn("联合调度模型", readme)
        self.assertIn("x_k", readme)
        self.assertIn("s_k", readme)
        self.assertIn("c_{k,m}", readme)
        self.assertIn("r_k", readme)

    def test_readme_documents_unified_joint_ga_hga_principles(self):
        readme = pathlib.Path("README.md").read_text(encoding="utf-8")

        self.assertIn("统一联合 GA", readme)
        self.assertIn("统一联合 HGA", readme)
        self.assertIn("染色体编码", readme)
        self.assertIn("WiFi floor", readme)
        self.assertIn("residual-hole", readme)
        self.assertIn("accept-if-better", readme)
        self.assertIn("whole-WiFi-state move", readme)

    def test_build_sdp_relaxation_annotation_does_not_reference_runtime_cp_alias(self):
        return_annotation = MODULE.build_sdp_relaxation.__annotations__["return"]
        self.assertNotIn("cp.", str(return_annotation))

    def test_ble_data_channel_frequency_mapping_matches_two_segment_model(self):
        self.assertEqual(MODULE.ble_channel_to_frequency_mhz(0), 2404.0)
        self.assertEqual(MODULE.ble_channel_to_frequency_mhz(10), 2424.0)
        self.assertEqual(MODULE.ble_channel_to_frequency_mhz(11), 2428.0)
        self.assertEqual(MODULE.ble_channel_to_frequency_mhz(36), 2478.0)

    def test_load_ble_standalone_config_parses_fifty_pair_json(self):
        config_path = pathlib.Path(__file__).resolve().parents[1] / "ble_macrocycle_hopping_sdp_config.json"

        config = MODULE.load_ble_standalone_config(config_path)
        states, _, A_k = MODULE.build_candidate_states(config.pair_configs, config.pattern_dict)

        self.assertEqual(config.num_channels, 37)
        self.assertEqual(len(config.pair_configs), 50)
        self.assertEqual(len(config.pattern_dict), 50)
        self.assertEqual(config.output_path.name, "ble_macrocycle_hopping_sdp_schedule.png")
        self.assertEqual(len(states), 112)
        self.assertEqual(len(A_k), 50)

    def test_resolve_standalone_config_uses_demo_fallback(self):
        config = MODULE.resolve_standalone_config(None)

        self.assertEqual(len(config.pair_configs), 4)
        self.assertEqual(config.plot_title, "BLE Event Grid")
        self.assertTrue(config.output_path.name.endswith(".png"))

    def test_merge_or_load_config_accepts_ga_solver_fields(self):
        config = {
            "solver": "ga",
            "ga_population_size": 24,
            "ga_generations": 30,
            "ga_mutation_rate": 0.15,
            "ga_crossover_rate": 0.8,
            "ga_elite_count": 2,
            "ga_seed": 7,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "ga-config.json"
            config_path.write_text(json.dumps(config))

            merged = MODULE.merge_or_load_config(config_path)

        self.assertEqual(merged["solver"], "ga")
        self.assertEqual(merged["ga_population_size"], 24)
        self.assertEqual(merged["ga_generations"], 30)
        self.assertEqual(merged["ga_mutation_rate"], 0.15)
        self.assertEqual(merged["ga_crossover_rate"], 0.8)
        self.assertEqual(merged["ga_elite_count"], 2)
        self.assertEqual(merged["ga_seed"], 7)

    def test_parse_args_accepts_solver_flag(self):
        args = MODULE.parse_args(["--solver", "ga"])
        self.assertEqual(args.solver, "ga")

    def test_run_ble_macrocycle_hopping_sdp_dispatches_to_ga_without_cvxpy(self):
        fake_ga_result = mock.Mock()
        fake_ga_result.selected = {0: MODULE.CandidateState(pair_id=0, offset=0, pattern_id=0)}
        fake_ga_result.blocks = []
        fake_ga_result.overlap_blocks = []
        fake_ga_result.collision_cost = 0.0
        fake_ga_result.best_fitness = 0.0

        with mock.patch("ble_macrocycle_hopping_ga.solve_ble_hopping_schedule_ga", return_value=fake_ga_result) as solve_mock, \
             mock.patch.object(MODULE, "render_event_grid"), \
             mock.patch.object(MODULE, "print_selected_schedule"), \
             mock.patch.object(MODULE, "print_event_block_table"), \
             mock.patch.object(MODULE, "require_cvxpy", side_effect=AssertionError("require_cvxpy should not be called for GA")):
            result = MODULE.run_ble_macrocycle_hopping_sdp(None, solver_override="ga")

        solve_mock.assert_called_once()
        self.assertIs(result["ga_result"], fake_ga_result)
        self.assertEqual(result["total_collision"], 0.0)


    def test_build_ble_advertising_idle_blocks_cover_three_band_centers(self):
        idle_blocks = MODULE.build_ble_advertising_idle_blocks(24)

        self.assertEqual(len(idle_blocks), 3)
        self.assertEqual({block.frequency_mhz for block in idle_blocks}, set(MODULE.BLE_ADVERTISING_CENTER_FREQ_MHZ))
        self.assertTrue(all(block.start_slot == 0 for block in idle_blocks))
        self.assertTrue(all(block.end_slot == 23 for block in idle_blocks))

    def test_external_interference_cost_is_positive_when_wifi_block_overlaps_ble_event(self):
        cfg = MODULE.PairConfig(
            pair_id=0,
            release_time=0,
            deadline=7,
            connect_interval=4,
            event_duration=2,
            num_events=1,
        )
        pattern = MODULE.HoppingPattern(pattern_id=0, start_channel=0, hop_increment=0)
        interference = [
            MODULE.ExternalInterferenceBlock(
                start_slot=0,
                end_slot=1,
                freq_low_mhz=2402.0,
                freq_high_mhz=2422.0,
                source_type="wifi",
                source_pair_id=99,
            )
        ]
        cost = MODULE.external_interference_cost_for_state(
            state=MODULE.CandidateState(pair_id=0, offset=0, pattern_id=0),
            cfg_dict={0: cfg},
            pattern_dict={0: [pattern]},
            num_channels=37,
            interference_blocks=interference,
        )
        self.assertGreater(cost, 0.0)

    def test_external_interference_cost_is_zero_without_time_frequency_overlap(self):
        cfg = MODULE.PairConfig(
            pair_id=0,
            release_time=0,
            deadline=7,
            connect_interval=4,
            event_duration=2,
            num_events=1,
        )
        pattern = MODULE.HoppingPattern(pattern_id=0, start_channel=20, hop_increment=0)
        interference = [
            MODULE.ExternalInterferenceBlock(
                start_slot=0,
                end_slot=1,
                freq_low_mhz=2402.0,
                freq_high_mhz=2422.0,
                source_type="wifi",
                source_pair_id=99,
            )
        ]
        cost = MODULE.external_interference_cost_for_state(
            state=MODULE.CandidateState(pair_id=0, offset=4, pattern_id=0),
            cfg_dict={0: cfg},
            pattern_dict={0: [pattern]},
            num_channels=37,
            interference_blocks=interference,
        )
        self.assertEqual(cost, 0.0)

    def test_build_sdp_relaxation_objective_is_vectorized(self):
        source = inspect.getsource(MODULE.build_sdp_relaxation)
        self.assertIn("np.triu", source)
        self.assertIn("cp.sum(cp.multiply", source)
        self.assertNotIn("objective_expr +=", source)

    def test_vectorized_upper_triangle_objective_matches_reference(self):
        omega = np.array(
            [
                [0.0, 0.0, 1.5, 2.0],
                [0.0, 0.0, 3.0, 4.5],
                [1.5, 3.0, 0.0, 5.0],
                [2.0, 4.5, 5.0, 0.0],
            ]
        )
        y_value = np.array(
            [
                [1.0, 0.2, 0.3, 0.4],
                [0.2, 1.0, 0.5, 0.6],
                [0.3, 0.5, 1.0, 0.7],
                [0.4, 0.6, 0.7, 1.0],
            ]
        )

        reference = 0.0
        for i in range(omega.shape[0]):
            for j in range(i + 1, omega.shape[1]):
                reference += omega[i, j] * y_value[i, j]

        vectorized = np.sum(np.triu(omega, k=1) * y_value)
        self.assertAlmostEqual(reference, vectorized)


    def test_rounding_prefers_state_with_lower_external_interference_when_ble_ble_cost_equal(self):
        pair_configs = [
            MODULE.PairConfig(pair_id=0, release_time=0, deadline=3, connect_interval=4, event_duration=1, num_events=1),
            MODULE.PairConfig(pair_id=1, release_time=0, deadline=3, connect_interval=4, event_duration=1, num_events=1),
        ]
        cfg_dict = {cfg.pair_id: cfg for cfg in pair_configs}
        pattern_dict = {
            0: [
                MODULE.HoppingPattern(pattern_id=0, start_channel=0, hop_increment=0),
                MODULE.HoppingPattern(pattern_id=1, start_channel=20, hop_increment=0),
            ],
            1: [MODULE.HoppingPattern(pattern_id=0, start_channel=30, hop_increment=0)],
        }
        states, _, A_k = MODULE.build_candidate_states(pair_configs, pattern_dict)
        result = MODULE.solve_ble_hopping_schedule(
            pair_configs=pair_configs,
            cfg_dict=cfg_dict,
            pattern_dict=pattern_dict,
            pair_ids=[0, 1],
            A_k=A_k,
            states=states,
            num_channels=37,
            external_interference_blocks=[
                MODULE.ExternalInterferenceBlock(
                    start_slot=0,
                    end_slot=0,
                    freq_low_mhz=2402.0,
                    freq_high_mhz=2422.0,
                    source_type="wifi",
                    source_pair_id=50,
                )
            ],
        )
        self.assertEqual(result["selected"][0].pattern_id, 1)

    def test_forbidden_state_indices_drop_wifi_overlapping_candidates_when_pair_has_wifi_free_option(self):
        costs = np.array([2.0, 0.0, 1.0], dtype=float)
        forbidden = MODULE.build_external_interference_forbidden_state_indices(
            pair_ids=[0, 1],
            A_k={0: [0, 1], 1: [2]},
            candidate_external_cost=costs,
        )
        self.assertEqual(forbidden, [0])


    def test_event_blocks_never_use_ble_advertising_frequencies(self):
        pair_configs, cfg_dict, pattern_dict, _, num_channels = MODULE.build_demo_instance()
        states, _, A_k = MODULE.build_candidate_states(pair_configs, pattern_dict)
        selected = {k: states[idxs[0]] for k, idxs in A_k.items()}

        blocks = MODULE.build_event_blocks(selected, cfg_dict, pattern_dict, num_channels)
        freqs = {block.frequency_mhz for block in blocks}

        self.assertTrue(freqs.isdisjoint(set(MODULE.BLE_ADVERTISING_CENTER_FREQ_MHZ)))

    def test_prune_feasible_offsets_limits_count_and_keeps_edges(self):
        offsets = list(range(1, 11))

        pruned = MODULE.prune_feasible_offsets(offsets, max_offsets=4)

        self.assertEqual(pruned, [1, 4, 7, 10])

    def test_build_candidate_states_respects_max_offsets_per_pair(self):
        pair_configs, _, pattern_dict, _, _ = MODULE.build_demo_instance()

        full_states, _, full_A_k = MODULE.build_candidate_states(pair_configs, pattern_dict)
        pruned_states, _, pruned_A_k = MODULE.build_candidate_states(
            pair_configs,
            pattern_dict,
            max_offsets_per_pair=2,
        )

        self.assertLess(len(pruned_states), len(full_states))
        for pair_id, idxs in pruned_A_k.items():
            self.assertLessEqual(len(idxs), len(pattern_dict[pair_id]) * 2)
        self.assertEqual(set(full_A_k), set(pruned_A_k))

    def test_summarize_candidate_space_reports_state_count_and_offsets(self):
        pair_configs, _, pattern_dict, _, _ = MODULE.build_demo_instance()
        summary = MODULE.summarize_candidate_space(
            pair_configs=pair_configs,
            pattern_dict=pattern_dict,
            max_offsets_per_pair=2,
        )

        expected_state_count = sum(
            len(MODULE.prune_feasible_offsets(MODULE.compute_feasible_offsets(cfg), 2))
            * len(pattern_dict[cfg.pair_id])
            for cfg in pair_configs
        )
        self.assertEqual(summary["state_count"], expected_state_count)
        self.assertEqual(summary["pair_count"], len(pair_configs))
        self.assertEqual(summary["pairs"][0]["offset_count"], 2)
        self.assertEqual(summary["pairs"][0]["pattern_count"], len(pattern_dict[0]))

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
        self.assertEqual(blocks[0].frequency_mhz, 2414.0)
        self.assertEqual(blocks[1].start_slot, 4)
        self.assertEqual(blocks[1].channel, 9)
        self.assertEqual(blocks[1].frequency_mhz, 2422.0)

    def test_same_channel_time_overlap_builds_overlap_block(self):
        blocks = [
            MODULE.EventBlock(
                pair_id=0,
                event_index=0,
                start_slot=2,
                end_slot=5,
                channel=10,
                frequency_mhz=2424.0,
                offset=2,
                pattern_id=0,
            ),
            MODULE.EventBlock(
                pair_id=1,
                event_index=0,
                start_slot=4,
                end_slot=6,
                channel=10,
                frequency_mhz=2424.0,
                offset=4,
                pattern_id=1,
            ),
        ]

        overlaps = MODULE.build_overlap_blocks(blocks)

        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0].start_slot, 4)
        self.assertEqual(overlaps[0].end_slot, 5)
        self.assertEqual(overlaps[0].channel, 10)
        self.assertEqual(overlaps[0].frequency_mhz, 2424.0)

    def test_same_overlap_window_is_deduplicated(self):
        blocks = [
            MODULE.EventBlock(
                pair_id=0,
                event_index=0,
                start_slot=2,
                end_slot=5,
                channel=10,
                frequency_mhz=2424.0,
                offset=2,
                pattern_id=0,
            ),
            MODULE.EventBlock(
                pair_id=1,
                event_index=0,
                start_slot=2,
                end_slot=5,
                channel=10,
                frequency_mhz=2424.0,
                offset=2,
                pattern_id=1,
            ),
            MODULE.EventBlock(
                pair_id=2,
                event_index=0,
                start_slot=2,
                end_slot=5,
                channel=10,
                frequency_mhz=2424.0,
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
                frequency_mhz=2430.0,
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
