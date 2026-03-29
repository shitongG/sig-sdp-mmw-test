"""Standalone runner for the isolated joint WiFi/BLE scheduler."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_sched.joint_wifi_ble_model import load_joint_config, resolve_joint_objective_policy
from joint_sched.joint_wifi_ble_plot import export_joint_output_artifacts
from joint_sched.joint_wifi_ble_ga import solve_joint_wifi_ble_ga
from joint_sched.joint_wifi_ble_hga import solve_joint_wifi_ble_hga
from joint_sched.joint_wifi_ble_adapter import build_joint_runtime_config_from_mainline_pair_parameters_csv
from joint_sched.joint_wifi_ble_sdp import solve_joint_wifi_ble_sdp
from joint_sched.joint_wifi_ble_random import BLE_CHANNELS, WIFI_CHANNELS, generate_joint_tasks_from_main_style_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated joint WiFi/BLE scheduler demo")
    parser.add_argument("--config", default="joint_sched/joint_wifi_ble_demo_config.json")
    parser.add_argument("--solver", choices=["sdp", "ga", "hga"], default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)



def is_native_joint_config(config: Mapping[str, Any]) -> bool:
    return all(key in config for key in ("tasks", "wifi_channels", "ble_channels"))



def _resolve_pair_parameters_csv_path(config: Mapping[str, Any], config_path: Path) -> Path | None:
    csv_value = config.get("pair_parameters_csv")
    if csv_value is not None:
        csv_path = Path(str(csv_value))
        if not csv_path.is_absolute():
            csv_path = (config_path.parent / csv_path).resolve()
        return csv_path

    output_dir_value = config.get("output_dir")
    if output_dir_value is None:
        return None
    output_dir = Path(str(output_dir_value))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    candidate = output_dir / "pair_parameters.csv"
    return candidate if candidate.exists() else None


def resolve_joint_runtime_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if config_path.suffix.lower() == ".csv":
        runtime = build_joint_runtime_config_from_mainline_pair_parameters_csv(config_path)
        runtime["_joint_generation_mode"] = "faithful_mainline_csv"
        return runtime

    config = load_joint_config(config_path)
    if is_native_joint_config(config):
        return dict(config)

    pair_parameters_csv = _resolve_pair_parameters_csv_path(config, config_path)
    if pair_parameters_csv is not None:
        runtime = build_joint_runtime_config_from_mainline_pair_parameters_csv(pair_parameters_csv, base_config=config)
        runtime["_joint_generation_mode"] = "faithful_mainline_csv"
        return runtime

    tasks = generate_joint_tasks_from_main_style_config(config)
    macrocycle_slots = 64
    return {
        "macrocycle_slots": macrocycle_slots,
        "wifi_channels": list(WIFI_CHANNELS),
        "ble_channels": list(BLE_CHANNELS),
        "tasks": [asdict(task) for task in tasks],
        "solver": str(config.get("solver", "sdp")),
        "objective": dict(config.get("objective", {"mode": "lexicographic", "primary": "payload", "secondary": "fill", "payload_tie_tolerance": 0, "fragmentation_penalty": 1.0, "idle_area_penalty": 1.0, "slot_span_penalty": 0.1, "occupied_area_penalty": 0.0})),
        "_source_config": str(config_path),
        "_joint_generation_mode": "simplified_random",
    }





def _normalized_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for task in config.get("tasks", []):
        if isinstance(task, dict):
            tasks.append(dict(task))
        else:
            tasks.append(asdict(task))
    return tasks


def _occupied_slot_count(blocks: list[Mapping[str, Any]]) -> int:
    occupied: set[tuple[int, int, str, int]] = set()
    for block in blocks:
        for slot in range(int(block["slot_start"]), int(block["slot_end"])):
            occupied.add((slot, int(block["pair_id"]), str(block["medium"]), int(block.get("event_index", 0))))
    return len(occupied)


def _summary_metrics(config: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    tasks = {int(task["task_id"]): task for task in _normalized_tasks(config)}
    selected_pair_ids = {int(state["pair_id"]) for state in result.get("selected_states", [])}
    wifi_payload_bytes = float(
        result.get(
            "wifi_payload_bytes",
            sum(
                int(tasks[pair_id].get("payload_bytes", 0))
                for pair_id in selected_pair_ids
                if pair_id in tasks and str(tasks[pair_id].get("radio", "")).lower() == "wifi"
            ),
        )
    )
    scheduled_payload_bytes = float(result.get("scheduled_payload_bytes", sum(int(tasks[pair_id].get("payload_bytes", 0)) for pair_id in selected_pair_ids if pair_id in tasks)))
    occupied_slot_count = float(result.get("occupied_slot_count", _occupied_slot_count(list(result.get("blocks", [])))))
    active_channel_count = max(1, len(config.get("wifi_channels", [])) + len(config.get("ble_channels", [])))
    utilization = occupied_slot_count / float(int(config["macrocycle_slots"]) * active_channel_count)
    objective_cfg = resolve_joint_objective_policy(config if isinstance(config, Mapping) else None)
    hga_cfg = dict(config.get("hga", {})) if isinstance(config, Mapping) else {}
    return {
        "objective_mode": str(objective_cfg.get("mode", "lexicographic")),
        "wifi_payload_floor_bytes": float(objective_cfg.get("wifi_payload_floor_bytes", 0.0)),
        "wifi_payload_bytes": wifi_payload_bytes,
        "scheduled_payload_bytes": scheduled_payload_bytes,
        "occupied_slot_count": occupied_slot_count,
        "occupied_area_mhz_slots": float(result.get("occupied_area_mhz_slots", 0.0)),
        "fragmentation_penalty": float(result.get("fragmentation_penalty", 0.0)),
        "idle_area_penalty": float(result.get("idle_area_penalty", 0.0)),
        "slot_span_penalty": float(result.get("slot_span_penalty", 0.0)),
        "fill_penalty": float(result.get("fill_penalty", 0.0)),
        "resource_utilization": max(0.0, min(float(utilization), 1.0)),
        "residual_seed_budget": int(hga_cfg.get("residual_seed_budget", 4)),
        "residual_swap_budget": int(hga_cfg.get("residual_swap_budget", 6)),
    }



def _write_joint_summary(output_dir: Path, summary: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "joint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def _write_experiment_record(output_dir: Path, summary: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "experiment_record.md"
    lines = [
        f"# {output_dir.name}",
        "",
        f"- solver: {summary.get('solver')}",
        f"- scheduled_pairs: {summary.get('selected_pairs')}",
        f"- scheduled_payload_bytes: {summary.get('scheduled_payload_bytes')}",
        f"- final_wifi_payload_bytes: {summary.get('final_wifi_payload_bytes')}",
        f"- wifi_payload_floor_bytes: {summary.get('wifi_payload_floor_bytes')}",
        f"- wifi_move_seed_count: {summary.get('wifi_move_seed_count', 0)}",
        f"- wifi_move_repairs_used: {summary.get('wifi_move_repairs_used', 0)}",
        f"- accepted_wifi_local_moves: {summary.get('accepted_wifi_local_moves', 0)}",
        f"- repair_insertions_used: {summary.get('repair_insertions_used', 0)}",
        f"- repair_swaps_used: {summary.get('repair_swaps_used', 0)}",
        f"- overview_path: {summary.get('overview_path')}",
    ]
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return record_path

def run_joint_demo(config_path: str | Path, solver: str | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = resolve_joint_runtime_config(config_path)
    solver_name = solver or str(config.get("solver", "sdp"))
    solver_config: Mapping[str, Any] = config

    start = time.perf_counter()
    if solver_name == "ga":
        result = solve_joint_wifi_ble_ga(solver_config)
    elif solver_name == "hga":
        result = solve_joint_wifi_ble_hga(solver_config)
    else:
        result = solve_joint_wifi_ble_sdp(solver_config)
    elapsed = time.perf_counter() - start

    resolved_output_dir = Path(output_dir) if output_dir is not None else config_path.parent / f"joint_wifi_ble_{solver_name}_output"
    artifacts = export_joint_output_artifacts(
        result,
        tasks=list(config["tasks"]),
        output_dir=resolved_output_dir,
        macrocycle_slots=int(config["macrocycle_slots"]),
    )
    scheduled_pairs = sum(1 for state in result.get("selected_states", []) if str(state.get("medium")) != "idle")
    metrics = _summary_metrics(config, result)
    summary = {
        "solver": solver_name,
        "status": result.get("status"),
        "task_count": result.get("task_count"),
        "state_count": result.get("state_count"),
        "selected_pairs": scheduled_pairs,
        "output_dir": str(resolved_output_dir),
        "output_path": str(artifacts["overview_path"]),
        "overview_path": str(artifacts["overview_path"]),
        "schedule_csv_path": str(artifacts["schedule_plot_rows_path"]),
        "wifi_ble_schedule_csv": str(artifacts["wifi_ble_schedule_csv"]),
        "pair_parameters_csv": str(artifacts["pair_parameters_csv"]),
        "unscheduled_pairs_csv": str(artifacts["unscheduled_pairs_csv"]),
        "ble_ce_channel_events_csv": str(artifacts["ble_ce_channel_events_csv"]),
        "wifi_ble_schedule_png": str(artifacts["wifi_ble_schedule_png"]),
        "window_count": len(artifacts["window_paths"]),
        "elapsed_sec": round(elapsed, 4),
        **metrics,
    }
    if config.get("pair_parameters_csv") is not None:
        summary["input_pair_parameters_csv"] = str(config["pair_parameters_csv"])
    summary["experiment_tag"] = resolved_output_dir.name
    for key in (
        "search_mode",
        "wifi_seed_payload_bytes",
        "final_wifi_payload_bytes",
        "coordination_rounds_used",
        "heuristic_seed_count",
        "candidate_state_count",
        "residual_seed_count",
        "wifi_move_seed_count",
        "wifi_move_repairs_used",
        "accepted_wifi_local_moves",
        "repair_insertions_used",
        "repair_swaps_used",
    ):
        if key in result:
            summary[key] = result[key]
    if "_joint_generation_mode" in config:
        summary["_joint_generation_mode"] = config["_joint_generation_mode"]
    summary.setdefault("final_wifi_payload_bytes", float(summary.get("wifi_payload_bytes", 0.0)))
    summary_path = _write_joint_summary(resolved_output_dir, summary)
    experiment_record_path = _write_experiment_record(resolved_output_dir, summary)
    summary["joint_summary_json"] = str(summary_path)
    summary["experiment_record_md"] = str(experiment_record_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_joint_demo(args.config, solver=args.solver, output_dir=args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
