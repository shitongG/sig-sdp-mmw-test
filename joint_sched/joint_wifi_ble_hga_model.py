"""Helpers for the unified heuristic joint WiFi/BLE GA backend."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping

from .joint_wifi_ble_model import (
    JointCandidateState,
    ResourceBlock,
    WIFI_CHANNEL_TO_MHZ,
    blocks_conflict,
    blocks_overlap_cost,
    expand_ble_candidate_blocks,
    expand_candidate_blocks,
    expand_wifi_candidate_blocks,
    state_is_idle,
    state_occupied_area,
    state_pair_is_feasible,
)

WIFI_STRIPE_WIDTH_MHZ = 2.0
WIFI_STRIPE_COUNT = 10


def _hole_to_block(hole: Mapping[str, Any] | Any) -> ResourceBlock:
    if isinstance(hole, Mapping):
        slot_start = int(hole.get("slot_start", 0))
        slot_end = int(hole.get("slot_end", 0))
        freq_low_mhz = float(hole.get("freq_low_mhz", 0.0))
        freq_high_mhz = float(hole.get("freq_high_mhz", 0.0))
    else:
        slot_start = int(getattr(hole, "slot_start", 0))
        slot_end = int(getattr(hole, "slot_end", 0))
        freq_low_mhz = float(getattr(hole, "freq_low_mhz", 0.0))
        freq_high_mhz = float(getattr(hole, "freq_high_mhz", 0.0))
    return ResourceBlock(
        state_id=-1,
        pair_id=-1,
        medium="hole",
        event_index=0,
        slot_start=slot_start,
        slot_end=slot_end,
        freq_low_mhz=freq_low_mhz,
        freq_high_mhz=freq_high_mhz,
        label="residual-hole",
    )


def compute_hole_capacity(hole: Mapping[str, Any] | Any) -> float:
    if isinstance(hole, Mapping):
        slot_start = float(hole.get("slot_start", 0.0))
        slot_end = float(hole.get("slot_end", 0.0))
        freq_low_mhz = float(hole.get("freq_low_mhz", 0.0))
        freq_high_mhz = float(hole.get("freq_high_mhz", 0.0))
    else:
        slot_start = float(getattr(hole, "slot_start", 0.0))
        slot_end = float(getattr(hole, "slot_end", 0.0))
        freq_low_mhz = float(getattr(hole, "freq_low_mhz", 0.0))
        freq_high_mhz = float(getattr(hole, "freq_high_mhz", 0.0))
    return max(0.0, slot_end - slot_start) * max(0.0, freq_high_mhz - freq_low_mhz)


def expand_wifi_state_to_stripes(state: JointCandidateState) -> list[ResourceBlock]:
    if state.medium != "wifi" or state.channel is None or state.width_slots is None:
        raise ValueError("WiFi stripe expansion requires a WiFi state with channel and width")
    center = WIFI_CHANNEL_TO_MHZ[int(state.channel)]
    low = center - 10.0
    blocks: list[ResourceBlock] = []
    for idx in range(WIFI_STRIPE_COUNT):
        stripe_low = low + idx * WIFI_STRIPE_WIDTH_MHZ
        stripe_high = stripe_low + WIFI_STRIPE_WIDTH_MHZ
        blocks.append(
            ResourceBlock(
                state_id=state.state_id,
                pair_id=state.pair_id,
                medium="wifi",
                event_index=idx,
                slot_start=state.offset,
                slot_end=state.offset + int(state.width_slots),
                freq_low_mhz=stripe_low,
                freq_high_mhz=stripe_high,
                label=f"wifi-{state.pair_id}-stripe{idx}",
            )
        )
    return blocks


def identify_blocking_wifi_pairs(
    wifi_states: Iterable[JointCandidateState],
    ble_state: JointCandidateState,
) -> set[int]:
    if state_is_idle(ble_state):
        return set()
    ble_blocks = expand_ble_candidate_blocks(ble_state)
    blockers: set[int] = set()
    for wifi_state in wifi_states:
        if state_is_idle(wifi_state):
            continue
        for stripe in expand_wifi_state_to_stripes(wifi_state):
            if any(blocks_conflict(stripe, ble_block) for ble_block in ble_blocks):
                blockers.add(int(wifi_state.pair_id))
                break
    return blockers


def build_wifi_local_reshuffle_candidates(
    current_state: JointCandidateState,
    all_wifi_states: Iterable[JointCandidateState],
    max_candidates: int = 4,
) -> list[JointCandidateState]:
    return build_wifi_state_move_candidates(
        current_state=current_state,
        all_wifi_states=all_wifi_states,
        max_candidates=max_candidates,
    )


def build_wifi_state_move_candidates(
    current_state: JointCandidateState,
    all_wifi_states: Iterable[JointCandidateState],
    *,
    selected_states: Iterable[JointCandidateState] | None = None,
    max_candidates: int = 4,
) -> list[JointCandidateState]:
    candidates: list[JointCandidateState] = []
    seen: set[tuple[int | None, int | None]] = set()
    protected_states = [
        state
        for state in (selected_states or [])
        if not state_is_idle(state) and int(state.pair_id) != int(current_state.pair_id)
    ]
    for candidate in all_wifi_states:
        if candidate.pair_id != current_state.pair_id or state_is_idle(candidate) or candidate.medium != "wifi":
            continue
        signature = (candidate.offset, candidate.channel)
        if signature == (current_state.offset, current_state.channel) or signature in seen:
            continue
        if any(not state_pair_is_feasible(candidate, protected_state) for protected_state in protected_states):
            continue
        seen.add(signature)
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return candidates


def extract_residual_holes(
    *,
    selected_states: Iterable[JointCandidateState],
    macrocycle_slots: int,
    freq_grid_mhz: Iterable[float],
    stripe_width_mhz: float = WIFI_STRIPE_WIDTH_MHZ,
) -> list[dict[str, float]]:
    if macrocycle_slots <= 0:
        return []
    occupied_blocks = [block for state in selected_states if not state_is_idle(state) for block in expand_candidate_blocks(state)]
    holes: list[dict[str, float]] = []
    for center_mhz in sorted({float(freq) for freq in freq_grid_mhz}):
        stripe_low = center_mhz - stripe_width_mhz / 2.0
        stripe_high = center_mhz + stripe_width_mhz / 2.0
        stripe_block = ResourceBlock(
            state_id=-1,
            pair_id=-1,
            medium="stripe",
            event_index=0,
            slot_start=0,
            slot_end=1,
            freq_low_mhz=stripe_low,
            freq_high_mhz=stripe_high,
            label=f"stripe-{center_mhz}",
        )
        busy_slots: set[int] = set()
        for block in occupied_blocks:
            if blocks_overlap_cost(block, stripe_block) <= 0.0:
                continue
            busy_slots.update(range(max(0, int(block.slot_start)), min(macrocycle_slots, int(block.slot_end))))
        if not busy_slots:
            holes.append(
                {
                    "slot_start": 0.0,
                    "slot_end": float(macrocycle_slots),
                    "freq_low_mhz": stripe_low,
                    "freq_high_mhz": stripe_high,
                    "freq_center_mhz": center_mhz,
                    "stripe_width_mhz": stripe_width_mhz,
                }
            )
            continue
        cursor = 0
        for slot in sorted(busy_slots):
            if slot > cursor:
                holes.append(
                    {
                        "slot_start": float(cursor),
                        "slot_end": float(slot),
                        "freq_low_mhz": stripe_low,
                        "freq_high_mhz": stripe_high,
                        "freq_center_mhz": center_mhz,
                        "stripe_width_mhz": stripe_width_mhz,
                    }
                )
            cursor = max(cursor, slot + 1)
        if cursor < macrocycle_slots:
            holes.append(
                {
                    "slot_start": float(cursor),
                    "slot_end": float(macrocycle_slots),
                    "freq_low_mhz": stripe_low,
                    "freq_high_mhz": stripe_high,
                    "freq_center_mhz": center_mhz,
                    "stripe_width_mhz": stripe_width_mhz,
                }
            )
    return [hole for hole in holes if float(hole["slot_end"]) > float(hole["slot_start"])]


def score_ble_state_against_residual_holes(
    ble_state: JointCandidateState,
    selected_states: Iterable[JointCandidateState],
) -> float:
    if state_is_idle(ble_state) or ble_state.medium != "ble":
        return float("-inf")
    ble_blocks = expand_ble_candidate_blocks(ble_state)
    overlap_cost = 0.0
    occupied_slots = 0
    for block in ble_blocks:
        occupied_slots += int(block.slot_end) - int(block.slot_start)
    for selected_state in selected_states:
        for selected_block in expand_candidate_blocks(selected_state):
            for ble_block in ble_blocks:
                overlap_cost += blocks_overlap_cost(ble_block, selected_block)
    return -float(overlap_cost) - 0.01 * float(occupied_slots)


def score_wifi_state_against_residual_holes(
    wifi_state: JointCandidateState,
    residual_holes: Iterable[Mapping[str, Any]],
) -> float:
    if state_is_idle(wifi_state) or wifi_state.medium != "wifi":
        return float("-inf")
    wifi_blocks = expand_wifi_candidate_blocks(wifi_state)
    overlap_area = 0.0
    for hole in residual_holes:
        hole_block = _hole_to_block(hole)
        for block in wifi_blocks:
            overlap_area += blocks_overlap_cost(block, hole_block)
    return -float(overlap_area)


def score_wifi_state_against_residual_holes_by_capacity(
    wifi_state: JointCandidateState,
    residual_holes: Iterable[Mapping[str, Any]],
) -> float:
    if state_is_idle(wifi_state) or wifi_state.medium != "wifi":
        return float("-inf")
    holes = list(residual_holes)
    total_capacity = sum(compute_hole_capacity(hole) for hole in holes)
    if total_capacity <= 0.0:
        return 0.0
    wifi_blocks = expand_wifi_candidate_blocks(wifi_state)
    overlap_area = 0.0
    for hole in holes:
        hole_block = _hole_to_block(hole)
        for block in wifi_blocks:
            overlap_area += blocks_overlap_cost(block, hole_block)
    return -float(overlap_area / total_capacity)


def rank_wifi_state_moves_for_ble_holes(
    current_state: JointCandidateState,
    candidate_wifi_states: Iterable[JointCandidateState],
    residual_holes: Iterable[Mapping[str, Any]],
    selected_states: Iterable[JointCandidateState] | None = None,
) -> list[dict[str, object]]:
    current_overlap = score_wifi_state_against_residual_holes(current_state, residual_holes)
    ranked: list[dict[str, object]] = []
    protected_states = [
        state
        for state in (selected_states or [])
        if not state_is_idle(state) and int(state.pair_id) != int(current_state.pair_id)
    ]
    for candidate in candidate_wifi_states:
        if state_is_idle(candidate) or candidate.medium != "wifi" or int(candidate.pair_id) != int(current_state.pair_id):
            continue
        if candidate.offset == current_state.offset and candidate.channel == current_state.channel:
            continue
        if any(not state_pair_is_feasible(candidate, protected_state) for protected_state in protected_states):
            continue
        candidate_overlap = score_wifi_state_against_residual_holes(candidate, residual_holes)
        ranked.append(
            {
                "replacement_state": candidate,
                "current_overlap_area": float(-current_overlap),
                "candidate_overlap_area": float(-candidate_overlap),
                "gain": float(current_overlap - candidate_overlap),
            }
        )
    ranked.sort(
        key=lambda item: (
            float(item["gain"]),
            -float(item["candidate_overlap_area"]),
            -int(getattr(item["replacement_state"], "state_id", 0)),
        ),
        reverse=True,
    )
    return ranked


def rank_wifi_state_moves_for_direct_accept_if_better(
    current_state: JointCandidateState,
    candidate_wifi_states: Iterable[JointCandidateState],
    residual_holes: Iterable[Mapping[str, Any]],
    selected_states: Iterable[JointCandidateState] | None = None,
) -> list[dict[str, object]]:
    current_capacity_score = score_wifi_state_against_residual_holes_by_capacity(current_state, residual_holes)
    current_overlap_score = score_wifi_state_against_residual_holes(current_state, residual_holes)
    hole_capacity = float(sum(compute_hole_capacity(hole) for hole in residual_holes))
    ranked: list[dict[str, object]] = []
    protected_states = [
        state
        for state in (selected_states or [])
        if not state_is_idle(state) and int(state.pair_id) != int(current_state.pair_id)
    ]
    for candidate in candidate_wifi_states:
        if state_is_idle(candidate) or candidate.medium != "wifi" or int(candidate.pair_id) != int(current_state.pair_id):
            continue
        if any(not state_pair_is_feasible(candidate, protected_state) for protected_state in protected_states):
            continue
        candidate_capacity_score = score_wifi_state_against_residual_holes_by_capacity(candidate, residual_holes)
        candidate_overlap_score = score_wifi_state_against_residual_holes(candidate, residual_holes)
        gain = float(candidate_capacity_score - current_capacity_score)
        ranked.append(
            {
                "replacement_state": candidate,
                "hole_capacity": hole_capacity,
                "current_overlap_area": float(-current_overlap_score),
                "candidate_overlap_area": float(-candidate_overlap_score),
                "current_overlap_ratio": float(-current_capacity_score),
                "candidate_overlap_ratio": float(-candidate_capacity_score),
                "gain": gain,
                "accept_if_better": gain > 0.0,
            }
        )
    ranked.sort(
        key=lambda item: (
            bool(item["accept_if_better"]),
            float(item["gain"]),
            -float(item["candidate_overlap_ratio"]),
            -float(item["candidate_overlap_area"]),
            -int(getattr(item["replacement_state"], "state_id", 0)),
        ),
        reverse=True,
    )
    return ranked


def score_residual_hole_fit(candidate_metrics: Mapping[str, Any], hole: Mapping[str, Any]) -> float:
    hole_area = compute_hole_capacity(hole)
    overlap_area = float(candidate_metrics.get("overlap_area_mhz_slots", 0.0))
    wifi_overlap_area = float(candidate_metrics.get("wifi_overlap_area", 0.0))
    fill_ratio = 0.0 if hole_area <= 0.0 else overlap_area / hole_area
    return fill_ratio - 10.0 * wifi_overlap_area


def score_candidate_state_against_hole(
    candidate: JointCandidateState | Mapping[str, Any],
    hole: Mapping[str, Any] | Any,
    selected_states: Iterable[JointCandidateState] | None = None,
) -> float:
    if isinstance(candidate, Mapping):
        return score_residual_hole_fit(candidate, hole)  # type: ignore[arg-type]
    if state_is_idle(candidate) or candidate.medium != "ble":
        return float("-inf")
    candidate_blocks = expand_ble_candidate_blocks(candidate)
    hole_block = _hole_to_block(hole)
    overlap_area = sum(blocks_overlap_cost(block, hole_block) for block in candidate_blocks)
    wifi_overlap_area = 0.0
    if selected_states is not None:
        selected_blocks = [block for state in selected_states for block in expand_candidate_blocks(state)]
        for block in candidate_blocks:
            for selected_block in selected_blocks:
                wifi_overlap_area += blocks_overlap_cost(block, selected_block)
    return score_residual_hole_fit(
        {
            "state_index": candidate.state_id,
            "overlap_area_mhz_slots": float(overlap_area),
            "wifi_overlap_area": float(wifi_overlap_area),
        },
        hole,
    )


def rank_ble_candidates_for_residual_hole(
    candidates: Iterable[Mapping[str, Any]],
    hole: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("wifi_overlap_area", 0.0)) > 0.0,
            -score_residual_hole_fit(item, hole),
            -float(item.get("overlap_area_mhz_slots", 0.0)),
            int(item.get("state_index", 0)),
        ),
    )


def rank_ble_insertions_for_holes(
    candidates: Iterable[JointCandidateState],
    residual_holes: Iterable[Mapping[str, Any]],
    selected_states: Iterable[JointCandidateState] | None = None,
) -> list[JointCandidateState]:
    holes = list(residual_holes)
    ranked: list[tuple[float, int, JointCandidateState]] = []
    for candidate in candidates:
        if state_is_idle(candidate) or candidate.medium != "ble":
            continue
        score = 0.0
        for hole in holes:
            score += score_candidate_state_against_hole(candidate, hole, selected_states=selected_states)
        ranked.append((score, int(candidate.state_id), candidate))
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [candidate for _, _, candidate in ranked]


def rank_ble_subset_replacements(
    *,
    selected_ble_states: Iterable[JointCandidateState],
    candidate_ble_states: Iterable[JointCandidateState],
    residual_holes: Iterable[Mapping[str, Any]],
    protected_wifi_states: Iterable[JointCandidateState] | None = None,
    subset_size_limit: int = 2,
    candidate_pool_limit: int = 8,
) -> list[dict[str, Any]]:
    protected_wifi = [state for state in (protected_wifi_states or []) if not state_is_idle(state)]
    current_ble = [state for state in selected_ble_states if not state_is_idle(state) and state.medium == "ble"]
    candidates = [state for state in candidate_ble_states if not state_is_idle(state) and state.medium == "ble"]
    holes = list(residual_holes)
    ranked: list[dict[str, Any]] = []
    if subset_size_limit <= 0:
        return ranked
    if holes:
        candidates = rank_ble_insertions_for_holes(
            candidates=candidates,
            residual_holes=holes,
            selected_states=protected_wifi + current_ble,
        )[: max(1, candidate_pool_limit)]

    for subset_size in range(1, min(subset_size_limit, len(candidates)) + 1):
        for insert_subset in combinations(candidates, subset_size):
            if any(not state_pair_is_feasible(left, right) for left, right in combinations(insert_subset, 2)):
                continue
            if any(not state_pair_is_feasible(candidate, wifi) for candidate in insert_subset for wifi in protected_wifi):
                continue

            conflicting_ble = [
                state
                for state in current_ble
                if any(not state_pair_is_feasible(state, candidate) for candidate in insert_subset)
            ]
            retained_ble = [state for state in current_ble if state not in conflicting_ble]
            if any(not state_pair_is_feasible(candidate, state) for candidate in insert_subset for state in retained_ble):
                continue

            hole_score = 0.0
            for candidate in insert_subset:
                hole_score += sum(
                    score_candidate_state_against_hole(candidate, hole, selected_states=protected_wifi + retained_ble)
                    for hole in holes
                )

            removal_penalty = float(sum(state_occupied_area(state) for state in conflicting_ble))
            ranked.append(
                {
                    "remove_states": conflicting_ble,
                    "insert_states": list(insert_subset),
                    "hole_score": float(hole_score),
                    "removal_penalty": removal_penalty,
                    "gain": float(hole_score - removal_penalty),
                    "subset_size": subset_size,
                }
            )

    ranked.sort(
        key=lambda item: (
            float(item["gain"]),
            float(item["hole_score"]),
            -float(item["removal_penalty"]),
            int(item["subset_size"]),
        ),
        reverse=True,
    )
    return ranked


def rank_residual_candidate_swaps(
    current_selection: Iterable[JointCandidateState],
    blocker_pair_id: int,
    replacement_candidates: Iterable[JointCandidateState],
    unscheduled_ble_candidates: Iterable[JointCandidateState],
) -> list[dict[str, object]]:
    selected_states = list(current_selection)
    blocker_state = next((state for state in selected_states if int(state.pair_id) == int(blocker_pair_id)), None)
    residual_holes = extract_residual_holes(
        selected_states=([blocker_state] if blocker_state is not None else []),
        macrocycle_slots=max((int(state.offset) for state in selected_states), default=64) + 64,
        freq_grid_mhz=(),
    )
    ranked: list[dict[str, object]] = []
    for replacement in replacement_candidates:
        if state_is_idle(replacement):
            continue
        local_selection = [
            replacement if state.pair_id == blocker_pair_id else state
            for state in selected_states
            if not (state.pair_id == blocker_pair_id and state_is_idle(replacement))
        ]
        ranked_subsets = rank_ble_subset_replacements(
            selected_ble_states=[state for state in local_selection if state.medium == "ble"],
            candidate_ble_states=unscheduled_ble_candidates,
            residual_holes=residual_holes,
            protected_wifi_states=[state for state in local_selection if state.medium == "wifi"],
            subset_size_limit=2,
        )
        best_subset = ranked_subsets[0] if ranked_subsets else None
        best_ble_state = None
        best_ble_score = float("-inf")
        if best_subset is not None and best_subset["insert_states"]:
            best_ble_state = best_subset["insert_states"][0]
            best_ble_score = float(best_subset["gain"])
        ranked.append(
            {
                "replacement_state": replacement,
                "best_ble_state": best_ble_state,
                "combined_gain": float(best_ble_score),
            }
        )
    return sorted(ranked, key=lambda item: float(item["combined_gain"]), reverse=True)
