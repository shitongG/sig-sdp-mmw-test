"""Shared data model for the isolated joint WiFi/BLE demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math

import numpy as np

WIFI_CHANNEL_TO_MHZ = {
    0: 2412.0,
    5: 2437.0,
    10: 2462.0,
}
BLE_ADV_CHANNELS_MHZ = (2402.0, 2426.0, 2480.0)
WIFI_BANDWIDTH_MHZ = 20.0
BLE_BANDWIDTH_MHZ = 2.0
DEFAULT_MAX_OFFSETS = 4
DEFAULT_BLE_PATTERN_COUNT = 3
DEFAULT_WIFI_BYTES_PER_SLOT = 750
DEFAULT_BLE_BYTES_PER_CE_SLOT = 247
DEFAULT_ALPHA_PAYLOAD = 1.0
DEFAULT_BETA_SLOTS = 0.15
DEFAULT_GAMMA_AREA = 0.02
DEFAULT_PAYLOAD_TIE_TOLERANCE = 0.0
DEFAULT_FRAGMENTATION_PENALTY = 1.0
DEFAULT_IDLE_AREA_PENALTY = 1.0
DEFAULT_SLOT_SPAN_PENALTY = 0.1
DEFAULT_OCCUPIED_AREA_PENALTY = 0.0


@dataclass(frozen=True)
class JointTaskSpec:
    """Raw task description parsed from JSON."""

    task_id: int
    radio: str
    payload_bytes: int
    release_slot: int
    deadline_slot: int
    preferred_channel: int | None = None
    repetitions: int = 1
    wifi_tx_slots: int | None = None
    wifi_period_slots: int | None = None
    ble_ce_slots: int | None = None
    ble_ci_slots_options: tuple[int, ...] = ()
    ble_num_events: int | None = None
    ble_pattern_count: int = DEFAULT_BLE_PATTERN_COUNT
    max_offsets: int = DEFAULT_MAX_OFFSETS
    cyclic_periodic: bool = False


@dataclass(frozen=True)
class WiFiPairConfig:
    pair_id: int
    payload_bytes: int
    release_slot: int
    deadline_slot: int
    tx_slots: int
    period_slots: int
    num_events: int
    channel_options: tuple[int, ...]
    max_offsets: int = DEFAULT_MAX_OFFSETS
    cyclic_periodic: bool = False


@dataclass(frozen=True)
class BLEPairConfig:
    pair_id: int
    payload_bytes: int
    release_slot: int
    deadline_slot: int
    ce_slots: int
    ci_slots_options: tuple[int, ...]
    num_events: int
    preferred_channel: int
    pattern_count: int = DEFAULT_BLE_PATTERN_COUNT
    max_offsets: int = DEFAULT_MAX_OFFSETS


@dataclass(frozen=True)
class JointSchedulingConfig:
    """Configuration for the isolated joint WiFi/BLE experiment."""

    macrocycle_slots: int
    wifi_channels: list[int]
    ble_channels: list[int]
    tasks: list[JointTaskSpec] = field(default_factory=list)
    solver: str = "sdp"


@dataclass(frozen=True)
class JointCandidateState:
    state_id: int
    pair_id: int
    medium: str
    offset: int
    channel: int | None = None
    period_slots: int | None = None
    width_slots: int | None = None
    pattern_id: int | None = None
    ci_slots: int | None = None
    ce_slots: int | None = None
    num_events: int | None = None
    cyclic_periodic: bool = False
    macrocycle_slots: int | None = None


@dataclass(frozen=True)
class ResourceBlock:
    state_id: int
    pair_id: int
    medium: str
    event_index: int
    slot_start: int
    slot_end: int
    freq_low_mhz: float
    freq_high_mhz: float
    label: str


@dataclass(frozen=True)
class ExternalBlock:
    slot_start: int
    slot_end: int
    freq_low_mhz: float
    freq_high_mhz: float
    label: str


@dataclass(frozen=True)
class JointCandidateSpace:
    states: list[JointCandidateState]
    pair_to_state_indices: dict[int, list[int]]
    wifi_pairs: list[WiFiPairConfig]
    ble_pairs: list[BLEPairConfig]


def state_is_idle(state: JointCandidateState) -> bool:
    return state.medium == "idle"


def load_joint_config(path: str | Path) -> dict[str, Any]:
    """Load the isolated demo configuration from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {key: value for key, value in payload.items() if not str(key).startswith("_comment")}
    return payload


def parse_joint_config(payload: Mapping[str, Any]) -> JointSchedulingConfig:
    tasks = [parse_joint_task_spec(item) for item in payload.get("tasks", [])]
    return JointSchedulingConfig(
        macrocycle_slots=int(payload["macrocycle_slots"]),
        wifi_channels=[int(channel) for channel in payload.get("wifi_channels", [])],
        ble_channels=[int(channel) for channel in payload.get("ble_channels", [])],
        tasks=tasks,
        solver=str(payload.get("solver", "sdp")),
    )


def parse_joint_task_spec(payload: Mapping[str, Any]) -> JointTaskSpec:
    return JointTaskSpec(
        task_id=int(payload["task_id"]),
        radio=str(payload["radio"]),
        payload_bytes=int(payload["payload_bytes"]),
        release_slot=int(payload["release_slot"]),
        deadline_slot=int(payload["deadline_slot"]),
        preferred_channel=None if payload.get("preferred_channel") is None else int(payload["preferred_channel"]),
        repetitions=int(payload.get("repetitions", 1)),
        wifi_tx_slots=None if payload.get("wifi_tx_slots") is None else int(payload["wifi_tx_slots"]),
        wifi_period_slots=None if payload.get("wifi_period_slots") is None else int(payload["wifi_period_slots"]),
        ble_ce_slots=None if payload.get("ble_ce_slots") is None else int(payload["ble_ce_slots"]),
        ble_ci_slots_options=tuple(int(value) for value in payload.get("ble_ci_slots_options", [])),
        ble_num_events=None if payload.get("ble_num_events") is None else int(payload["ble_num_events"]),
        ble_pattern_count=int(payload.get("ble_pattern_count", DEFAULT_BLE_PATTERN_COUNT)),
        max_offsets=int(payload.get("max_offsets", DEFAULT_MAX_OFFSETS)),
        cyclic_periodic=bool(payload.get("cyclic_periodic", False)),
    )


def build_joint_candidate_states(config: Mapping[str, Any] | JointSchedulingConfig) -> JointCandidateSpace:
    cfg = config if isinstance(config, JointSchedulingConfig) else parse_joint_config(config)
    wifi_pairs, ble_pairs = build_pair_configs(cfg)
    states: list[JointCandidateState] = []
    pair_to_state_indices: dict[int, list[int]] = {}
    next_state_id = 0

    for wifi_pair in wifi_pairs:
        pair_indices: list[int] = []
        idle_state = JointCandidateState(
            state_id=next_state_id,
            pair_id=wifi_pair.pair_id,
            medium="idle",
            offset=wifi_pair.release_slot,
            cyclic_periodic=wifi_pair.cyclic_periodic,
            macrocycle_slots=cfg.macrocycle_slots,
        )
        states.append(idle_state)
        pair_indices.append(next_state_id)
        next_state_id += 1
        offsets = prune_offsets(
                compute_feasible_offsets(
                    release_slot=wifi_pair.release_slot,
                    deadline_slot=wifi_pair.deadline_slot,
                    step_slots=wifi_pair.period_slots,
                    width_slots=wifi_pair.tx_slots,
                    num_events=wifi_pair.num_events,
                    cyclic_periodic=wifi_pair.cyclic_periodic,
                    macrocycle_slots=cfg.macrocycle_slots,
                ),
                wifi_pair.max_offsets,
            )
        for offset in offsets:
            for channel in wifi_pair.channel_options:
                state = JointCandidateState(
                    state_id=next_state_id,
                    pair_id=wifi_pair.pair_id,
                    medium="wifi",
                    offset=offset,
                    channel=channel,
                    period_slots=wifi_pair.period_slots,
                    width_slots=wifi_pair.tx_slots,
                    num_events=wifi_pair.num_events,
                    cyclic_periodic=wifi_pair.cyclic_periodic,
                    macrocycle_slots=cfg.macrocycle_slots,
                )
                states.append(state)
                pair_indices.append(next_state_id)
                next_state_id += 1
        pair_to_state_indices[wifi_pair.pair_id] = pair_indices

    for ble_pair in ble_pairs:
        pair_indices = []
        idle_state = JointCandidateState(
            state_id=next_state_id,
            pair_id=ble_pair.pair_id,
            medium="idle",
            offset=ble_pair.release_slot,
            macrocycle_slots=cfg.macrocycle_slots,
        )
        states.append(idle_state)
        pair_indices.append(next_state_id)
        next_state_id += 1
        for ci_slots in ble_pair.ci_slots_options:
            offsets = prune_offsets(
                compute_feasible_offsets(
                    release_slot=ble_pair.release_slot,
                    deadline_slot=ble_pair.deadline_slot,
                    step_slots=ci_slots,
                    width_slots=ble_pair.ce_slots,
                    num_events=ble_pair.num_events,
                ),
                ble_pair.max_offsets,
            )
            for offset in offsets:
                for pattern_id in range(ble_pair.pattern_count):
                    state = JointCandidateState(
                        state_id=next_state_id,
                        pair_id=ble_pair.pair_id,
                        medium="ble",
                        offset=offset,
                        channel=ble_pair.preferred_channel,
                        pattern_id=pattern_id,
                        ci_slots=ci_slots,
                        ce_slots=ble_pair.ce_slots,
                        num_events=ble_pair.num_events,
                        macrocycle_slots=cfg.macrocycle_slots,
                    )
                    states.append(state)
                    pair_indices.append(next_state_id)
                    next_state_id += 1
        pair_to_state_indices[ble_pair.pair_id] = pair_indices

    return JointCandidateSpace(
        states=states,
        pair_to_state_indices=pair_to_state_indices,
        wifi_pairs=wifi_pairs,
        ble_pairs=ble_pairs,
    )


def build_pair_configs(config: JointSchedulingConfig) -> tuple[list[WiFiPairConfig], list[BLEPairConfig]]:
    wifi_pairs: list[WiFiPairConfig] = []
    ble_pairs: list[BLEPairConfig] = []
    for task in sorted(config.tasks, key=lambda item: item.task_id):
        if task.radio == "wifi":
            wifi_pairs.append(build_wifi_pair_config(task, config.wifi_channels))
        elif task.radio == "ble":
            ble_pairs.append(build_ble_pair_config(task, config.ble_channels))
        else:
            raise ValueError(f"Unsupported radio: {task.radio}")
    return wifi_pairs, ble_pairs


def build_wifi_pair_config(task: JointTaskSpec, wifi_channels: Iterable[int]) -> WiFiPairConfig:
    tx_slots = task.wifi_tx_slots or max(1, math.ceil(task.payload_bytes / DEFAULT_WIFI_BYTES_PER_SLOT))
    candidate_channels = tuple([task.preferred_channel] if task.preferred_channel is not None else sorted(set(int(ch) for ch in wifi_channels)))
    return WiFiPairConfig(
        pair_id=task.task_id,
        payload_bytes=task.payload_bytes,
        release_slot=task.release_slot,
        deadline_slot=task.deadline_slot,
        tx_slots=tx_slots,
        period_slots=task.wifi_period_slots or tx_slots,
        num_events=max(1, task.repetitions),
        channel_options=candidate_channels,
        max_offsets=task.max_offsets,
        cyclic_periodic=task.cyclic_periodic,
    )


def build_ble_pair_config(task: JointTaskSpec, ble_channels: Iterable[int]) -> BLEPairConfig:
    ce_slots = task.ble_ce_slots or max(1, math.ceil(task.payload_bytes / DEFAULT_BLE_BYTES_PER_CE_SLOT))
    ci_options = task.ble_ci_slots_options or derive_default_ble_ci_options(ce_slots)
    preferred_channel = task.preferred_channel if task.preferred_channel is not None else min(int(ch) for ch in ble_channels)
    return BLEPairConfig(
        pair_id=task.task_id,
        payload_bytes=task.payload_bytes,
        release_slot=task.release_slot,
        deadline_slot=task.deadline_slot,
        ce_slots=ce_slots,
        ci_slots_options=tuple(sorted(set(int(value) for value in ci_options))),
        num_events=task.ble_num_events or max(1, task.repetitions),
        preferred_channel=int(preferred_channel),
        pattern_count=max(1, task.ble_pattern_count),
        max_offsets=task.max_offsets,
    )


def derive_default_ble_ci_options(ce_slots: int) -> tuple[int, ...]:
    return tuple(value for value in (max(ce_slots, 8), max(ce_slots, 16)) if value >= ce_slots)


def compute_feasible_offsets(
    *,
    release_slot: int,
    deadline_slot: int,
    step_slots: int,
    width_slots: int,
    num_events: int,
    cyclic_periodic: bool = False,
    macrocycle_slots: int | None = None,
) -> list[int]:
    if cyclic_periodic and macrocycle_slots is not None and macrocycle_slots > 0:
        max_offset = min(deadline_slot, macrocycle_slots - 1)
        if max_offset < release_slot:
            return []
        return list(range(release_slot, max_offset + 1))
    max_offset = deadline_slot - (num_events - 1) * step_slots - width_slots + 1
    if max_offset < release_slot:
        return []
    return list(range(release_slot, max_offset + 1))


def prune_offsets(offsets: list[int], limit: int) -> list[int]:
    if limit <= 0 or len(offsets) <= limit:
        return offsets
    if limit == 1:
        return [offsets[0]]
    positions = [round(i * (len(offsets) - 1) / (limit - 1)) for i in range(limit)]
    pruned = [offsets[idx] for idx in positions]
    return list(dict.fromkeys(pruned))


def ble_data_channel_center_mhz(channel: int) -> float:
    if 0 <= channel <= 10:
        return 2404.0 + 2.0 * channel
    if 11 <= channel <= 36:
        return 2428.0 + 2.0 * (channel - 11)
    raise ValueError(f"BLE data channel out of range: {channel}")


def ble_pattern_channel(base_channel: int, pattern_id: int, event_index: int) -> int:
    hop = (5 + 2 * pattern_id) % 37
    if hop == 0:
        hop = 1
    return (base_channel + event_index * hop) % 37


def expand_candidate_blocks(state: JointCandidateState) -> list[ResourceBlock]:
    if state_is_idle(state):
        return []
    if state.medium == "wifi":
        return expand_wifi_candidate_blocks(state)
    if state.medium == "ble":
        return expand_ble_candidate_blocks(state)
    raise ValueError(f"Unsupported medium: {state.medium}")


def expand_wifi_candidate_blocks(state: JointCandidateState) -> list[ResourceBlock]:
    if state.channel is None or state.width_slots is None:
        raise ValueError("WiFi state missing channel or width")
    center = WIFI_CHANNEL_TO_MHZ[state.channel]
    num_events = max(1, int(state.num_events or 1))
    period_slots = int(state.period_slots or state.width_slots)
    macrocycle_slots = int(state.macrocycle_slots or 0)
    blocks: list[ResourceBlock] = []
    for event_index in range(num_events):
        slot_start = state.offset + event_index * period_slots
        if state.cyclic_periodic and macrocycle_slots > 0:
            slot_start %= macrocycle_slots
            slot_end = slot_start + state.width_slots
            if slot_end <= macrocycle_slots:
                blocks.append(
                    ResourceBlock(
                        state_id=state.state_id,
                        pair_id=state.pair_id,
                        medium="wifi",
                        event_index=event_index,
                        slot_start=slot_start,
                        slot_end=slot_end,
                        freq_low_mhz=center - WIFI_BANDWIDTH_MHZ / 2.0,
                        freq_high_mhz=center + WIFI_BANDWIDTH_MHZ / 2.0,
                        label=f"wifi-{state.pair_id}-ev{event_index}",
                    )
                )
            else:
                blocks.append(
                    ResourceBlock(
                        state_id=state.state_id,
                        pair_id=state.pair_id,
                        medium="wifi",
                        event_index=event_index,
                        slot_start=slot_start,
                        slot_end=macrocycle_slots,
                        freq_low_mhz=center - WIFI_BANDWIDTH_MHZ / 2.0,
                        freq_high_mhz=center + WIFI_BANDWIDTH_MHZ / 2.0,
                        label=f"wifi-{state.pair_id}-ev{event_index}",
                    )
                )
                blocks.append(
                    ResourceBlock(
                        state_id=state.state_id,
                        pair_id=state.pair_id,
                        medium="wifi",
                        event_index=event_index,
                        slot_start=0,
                        slot_end=slot_end - macrocycle_slots,
                        freq_low_mhz=center - WIFI_BANDWIDTH_MHZ / 2.0,
                        freq_high_mhz=center + WIFI_BANDWIDTH_MHZ / 2.0,
                        label=f"wifi-{state.pair_id}-ev{event_index}",
                    )
                )
        else:
            blocks.append(
                ResourceBlock(
                    state_id=state.state_id,
                    pair_id=state.pair_id,
                    medium="wifi",
                    event_index=event_index,
                    slot_start=slot_start,
                    slot_end=slot_start + state.width_slots,
                    freq_low_mhz=center - WIFI_BANDWIDTH_MHZ / 2.0,
                    freq_high_mhz=center + WIFI_BANDWIDTH_MHZ / 2.0,
                    label=f"wifi-{state.pair_id}-ev{event_index}",
                )
            )
    return blocks


def expand_ble_candidate_blocks(state: JointCandidateState) -> list[ResourceBlock]:
    if state.channel is None or state.pattern_id is None or state.ci_slots is None or state.ce_slots is None or state.num_events is None:
        raise ValueError("BLE state missing hopping parameters")
    blocks: list[ResourceBlock] = []
    for event_index in range(state.num_events):
        channel = ble_pattern_channel(state.channel, state.pattern_id, event_index)
        center = ble_data_channel_center_mhz(channel)
        blocks.append(
            ResourceBlock(
                state_id=state.state_id,
                pair_id=state.pair_id,
                medium="ble",
                event_index=event_index,
                slot_start=state.offset + event_index * state.ci_slots,
                slot_end=state.offset + event_index * state.ci_slots + state.ce_slots,
                freq_low_mhz=center - BLE_BANDWIDTH_MHZ / 2.0,
                freq_high_mhz=center + BLE_BANDWIDTH_MHZ / 2.0,
                label=f"ble-{state.pair_id}-ev{event_index}",
            )
        )
    return blocks


def build_payload_by_pair(config: Mapping[str, Any] | JointSchedulingConfig) -> dict[int, int]:
    cfg = config if isinstance(config, JointSchedulingConfig) else parse_joint_config(config)
    return {int(task.task_id): int(task.payload_bytes) for task in cfg.tasks}


def resolve_joint_objective_policy(config: Mapping[str, Any] | None = None) -> dict[str, float | str]:
    objective = dict(config.get("objective", {})) if isinstance(config, Mapping) else {}
    mode = str(objective.get("mode", "lexicographic"))
    if mode == "utility":
        return {
            "mode": "utility",
            "alpha_payload": float(objective.get("alpha_payload", DEFAULT_ALPHA_PAYLOAD)),
            "beta_slots": float(objective.get("beta_slots", DEFAULT_BETA_SLOTS)),
            "gamma_area": float(objective.get("gamma_area", DEFAULT_GAMMA_AREA)),
        }
    return {
        "mode": "lexicographic",
        "primary": str(objective.get("primary", "payload")),
        "secondary": str(objective.get("secondary", "fill")),
        "wifi_payload_floor_bytes": float(objective.get("wifi_payload_floor_bytes", 0.0)),
        "payload_tie_tolerance": float(objective.get("payload_tie_tolerance", DEFAULT_PAYLOAD_TIE_TOLERANCE)),
        "fragmentation_penalty": float(objective.get("fragmentation_penalty", DEFAULT_FRAGMENTATION_PENALTY)),
        "idle_area_penalty": float(objective.get("idle_area_penalty", DEFAULT_IDLE_AREA_PENALTY)),
        "slot_span_penalty": float(objective.get("slot_span_penalty", DEFAULT_SLOT_SPAN_PENALTY)),
        "occupied_area_penalty": float(objective.get("occupied_area_penalty", DEFAULT_OCCUPIED_AREA_PENALTY)),
        "alpha_payload": float(objective.get("alpha_payload", DEFAULT_ALPHA_PAYLOAD)),
        "beta_slots": float(objective.get("beta_slots", DEFAULT_BETA_SLOTS)),
        "gamma_area": float(objective.get("gamma_area", DEFAULT_GAMMA_AREA)),
    }


def state_slot_span(state: JointCandidateState) -> int:
    blocks = expand_candidate_blocks(state)
    if not blocks:
        return 0
    return max(int(block.slot_end) for block in blocks) - min(int(block.slot_start) for block in blocks)


def state_fragmentation_penalty(state: JointCandidateState) -> float:
    if state_is_idle(state):
        return 0.0
    return float(max(0, state_slot_span(state) - state_occupied_slot_count(state)))


def build_state_fill_penalty_vector(
    states: list[JointCandidateState],
    objective: Mapping[str, Any] | None = None,
) -> np.ndarray:
    policy = resolve_joint_objective_policy(objective)
    frag_weight = float(policy.get("fragmentation_penalty", DEFAULT_FRAGMENTATION_PENALTY))
    span_weight = float(policy.get("slot_span_penalty", DEFAULT_SLOT_SPAN_PENALTY))
    area_weight = float(policy.get("occupied_area_penalty", DEFAULT_OCCUPIED_AREA_PENALTY))
    penalties: list[float] = []
    for state in states:
        if state_is_idle(state):
            penalties.append(0.0)
            continue
        penalties.append(
            frag_weight * state_fragmentation_penalty(state)
            + span_weight * float(state_slot_span(state))
            + area_weight * float(state_occupied_area(state))
        )
    return np.asarray(penalties, dtype=float)


def summarize_selected_schedule_metrics(
    config: Mapping[str, Any] | JointSchedulingConfig,
    states: Iterable[JointCandidateState],
    objective: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    cfg = config if isinstance(config, JointSchedulingConfig) else parse_joint_config(config)
    payload_by_pair = build_payload_by_pair(cfg)
    policy = resolve_joint_objective_policy(objective if objective is not None else config if isinstance(config, Mapping) else None)
    selected_states = [state for state in states if not state_is_idle(state)]
    blocks = [block for state in selected_states for block in expand_candidate_blocks(state)]
    scheduled_payload_bytes = float(sum(payload_by_pair.get(int(state.pair_id), 0) for state in selected_states))
    occupied_area = float(sum((float(block.freq_high_mhz) - float(block.freq_low_mhz)) * float(int(block.slot_end) - int(block.slot_start)) for block in blocks))
    occupied_slots: set[int] = set()
    for block in blocks:
        occupied_slots.update(range(int(block.slot_start), int(block.slot_end)))
    occupied_slot_count = float(len(occupied_slots))
    if occupied_slots:
        slot_min = min(occupied_slots)
        slot_max = max(occupied_slots) + 1
        slot_span = float(slot_max - slot_min)
        sorted_slots = sorted(occupied_slots)
        contiguous_segments = 1
        for left, right in zip(sorted_slots, sorted_slots[1:]):
            if right != left + 1:
                contiguous_segments += 1
        fragmentation_penalty = float(max(0, contiguous_segments - 1))
    else:
        slot_span = 0.0
        fragmentation_penalty = 0.0
    if blocks:
        freq_low = min(float(block.freq_low_mhz) for block in blocks)
        freq_high = max(float(block.freq_high_mhz) for block in blocks)
        idle_area_penalty = max(0.0, slot_span * max(0.0, freq_high - freq_low) - occupied_area)
    else:
        idle_area_penalty = 0.0
    fill_penalty = (
        float(policy.get("fragmentation_penalty", DEFAULT_FRAGMENTATION_PENALTY)) * fragmentation_penalty
        + float(policy.get("idle_area_penalty", DEFAULT_IDLE_AREA_PENALTY)) * idle_area_penalty
        + float(policy.get("slot_span_penalty", DEFAULT_SLOT_SPAN_PENALTY)) * slot_span
    )
    return {
        "scheduled_payload_bytes": scheduled_payload_bytes,
        "occupied_slot_count": occupied_slot_count,
        "occupied_area_mhz_slots": occupied_area,
        "fragmentation_penalty": fragmentation_penalty,
        "idle_area_penalty": float(idle_area_penalty),
        "slot_span_penalty": float(slot_span),
        "fill_penalty": float(fill_penalty),
    }


def state_occupied_slot_count(state: JointCandidateState) -> int:
    blocks = expand_candidate_blocks(state)
    return sum(int(block.slot_end) - int(block.slot_start) for block in blocks)


def state_occupied_area(state: JointCandidateState) -> float:
    blocks = expand_candidate_blocks(state)
    area = 0.0
    for block in blocks:
        duration = float(int(block.slot_end) - int(block.slot_start))
        bandwidth = float(block.freq_high_mhz) - float(block.freq_low_mhz)
        area += duration * bandwidth
    return area


def resolve_objective_weights(config: Mapping[str, Any] | None = None) -> tuple[float, float, float]:
    objective = resolve_joint_objective_policy(config)
    return (
        float(objective.get("alpha_payload", DEFAULT_ALPHA_PAYLOAD)),
        float(objective.get("beta_slots", DEFAULT_BETA_SLOTS)),
        float(objective.get("gamma_area", DEFAULT_GAMMA_AREA)),
    )


def build_state_utility_vector(
    states: list[JointCandidateState],
    payload_by_pair: Mapping[int, int],
    objective: Mapping[str, Any] | None = None,
) -> np.ndarray:
    alpha_payload, beta_slots, gamma_area = resolve_objective_weights(objective)
    utility: list[float] = []
    for state in states:
        if state_is_idle(state):
            utility.append(0.0)
            continue
        payload_bytes = float(payload_by_pair.get(int(state.pair_id), 0))
        slot_cost = float(state_occupied_slot_count(state))
        area_cost = float(state_occupied_area(state))
        utility.append(alpha_payload * payload_bytes - beta_slots * slot_cost - gamma_area * area_cost)
    return np.asarray(utility, dtype=float)


def blocks_overlap_cost(left: ResourceBlock, right: ResourceBlock) -> float:
    slot_overlap = max(0, min(left.slot_end, right.slot_end) - max(left.slot_start, right.slot_start))
    if slot_overlap <= 0:
        return 0.0
    freq_overlap = max(0.0, min(left.freq_high_mhz, right.freq_high_mhz) - max(left.freq_low_mhz, right.freq_low_mhz))
    if freq_overlap <= 0.0:
        return 0.0
    return float(slot_overlap) * float(freq_overlap)


def blocks_conflict(left: ResourceBlock, right: ResourceBlock) -> bool:
    return blocks_overlap_cost(left, right) > 0.0


def state_pair_is_feasible(left: JointCandidateState, right: JointCandidateState) -> bool:
    if left.pair_id == right.pair_id:
        return False
    left_blocks = expand_candidate_blocks(left)
    right_blocks = expand_candidate_blocks(right)
    for left_block in left_blocks:
        for right_block in right_blocks:
            if blocks_conflict(left_block, right_block):
                return False
    return True


def build_joint_forbidden_state_pairs(states: list[JointCandidateState]) -> set[tuple[int, int]]:
    forbidden_pairs: set[tuple[int, int]] = set()
    for i, left in enumerate(states):
        for j in range(i + 1, len(states)):
            right = states[j]
            if not state_pair_is_feasible(left, right):
                forbidden_pairs.add((i, j))
    return forbidden_pairs


def selected_schedule_has_no_conflicts(states: Iterable[JointCandidateState]) -> bool:
    states_list = list(states)
    for i, left in enumerate(states_list):
        for right in states_list[i + 1 :]:
            if not state_pair_is_feasible(left, right):
                return False
    return True


def build_joint_cost_matrix(states: list[JointCandidateState]) -> list[list[float]]:
    blocks_by_state = [expand_candidate_blocks(state) for state in states]
    size = len(states)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            cost = 0.0
            for left in blocks_by_state[i]:
                for right in blocks_by_state[j]:
                    cost += blocks_overlap_cost(left, right)
            matrix[i][j] = cost
            matrix[j][i] = cost
    return matrix
