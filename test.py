from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PairConfig:
    pair_id: int
    offset_candidates: Tuple[int, ...]
    pattern_count: int
    interval: int
    duration: int
    event_count: int
    channel_seed: int = 0
    release_time: int = 0
    deadline: int | None = None


@dataclass(frozen=True)
class SchedulingConfig:
    pairs: Tuple[PairConfig, ...]
    channel_count: int = 37
    macro_horizon: int | None = None


@dataclass(frozen=True)
class Event:
    pair_id: int
    state_index: int
    event_index: int
    channel: int
    interval: Tuple[int, int]


@dataclass(frozen=True)
class Model:
    config: SchedulingConfig
    states: Tuple[Tuple[int, int, int], ...]
    state_to_idx: Dict[Tuple[int, int, int], int]
    pair_state_indices: Dict[int, Tuple[int, ...]]
    collision_matrix: np.ndarray


@dataclass(frozen=True)
class Solution:
    selected_indices: Tuple[int, ...]
    selected_states: Dict[int, Tuple[int, int, int]]
    total_collision: float


def _validate_pair(pair: PairConfig) -> None:
    if not pair.offset_candidates:
        raise ValueError(f"pair {pair.pair_id} has no offset candidates")
    if pair.pattern_count <= 0:
        raise ValueError(f"pair {pair.pair_id} must have at least one pattern")
    if pair.interval <= 0:
        raise ValueError(f"pair {pair.pair_id} interval must be positive")
    if pair.duration <= 0:
        raise ValueError(f"pair {pair.pair_id} duration must be positive")
    if pair.event_count <= 0:
        raise ValueError(f"pair {pair.pair_id} event_count must be positive")


def infer_macro_horizon(config: SchedulingConfig) -> int:
    if config.macro_horizon is not None:
        return config.macro_horizon
    return max(pair.offset_candidates[-1] + pair.interval * pair.event_count for pair in config.pairs)


def event_start(pair: PairConfig, offset: int, event_index: int) -> int:
    return offset + event_index * pair.interval


def event_interval(pair: PairConfig, offset: int, event_index: int) -> Tuple[int, int]:
    start = event_start(pair, offset, event_index)
    end = start + pair.duration - 1
    return start, end


def overlap_length(interval1: Tuple[int, int], interval2: Tuple[int, int]) -> int:
    left = max(interval1[0], interval2[0])
    right = min(interval1[1], interval2[1])
    return max(0, right - left + 1)


def channel_of_event(
    pair: PairConfig,
    pattern_index: int,
    event_index: int,
    channel_count: int,
) -> int:
    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    return (pair.channel_seed + 5 * pattern_index + 3 * event_index + pair.pair_id) % channel_count


def enumerate_states(config: SchedulingConfig) -> Tuple[Tuple[int, int, int], ...]:
    states: List[Tuple[int, int, int]] = []
    for pair in config.pairs:
        _validate_pair(pair)
        for offset in pair.offset_candidates:
            for pattern_index in range(pair.pattern_count):
                states.append((pair.pair_id, offset, pattern_index))
    return tuple(states)


def state_events(
    config: SchedulingConfig,
    state_index: int,
    state: Tuple[int, int, int],
) -> List[Event]:
    pair_id, offset, pattern_index = state
    pair = next(pair for pair in config.pairs if pair.pair_id == pair_id)
    events: List[Event] = []
    for event_index in range(pair.event_count):
        interval = event_interval(pair, offset, event_index)
        channel = channel_of_event(pair, pattern_index, event_index, config.channel_count)
        events.append(
            Event(
                pair_id=pair_id,
                state_index=state_index,
                event_index=event_index,
                channel=channel,
                interval=interval,
            )
        )
    return events


def collision_cost(events1: Sequence[Event], events2: Sequence[Event]) -> float:
    total = 0.0
    for event1 in events1:
        for event2 in events2:
            if event1.channel != event2.channel:
                continue
            total += overlap_length(event1.interval, event2.interval)
    return total


def build_model(config: SchedulingConfig) -> Model:
    states = enumerate_states(config)
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    pair_state_indices: Dict[int, List[int]] = {pair.pair_id: [] for pair in config.pairs}
    cached_events: Dict[int, List[Event]] = {}

    for idx, state in enumerate(states):
        pair_state_indices[state[0]].append(idx)
        cached_events[idx] = state_events(config, idx, state)

    collision_matrix = np.zeros((len(states), len(states)), dtype=float)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            if states[i][0] == states[j][0]:
                continue
            cost = collision_cost(cached_events[i], cached_events[j])
            collision_matrix[i, j] = cost
            collision_matrix[j, i] = cost

    return Model(
        config=config,
        states=states,
        state_to_idx=state_to_idx,
        pair_state_indices={pair_id: tuple(indices) for pair_id, indices in pair_state_indices.items()},
        collision_matrix=collision_matrix,
    )


def _total_collision_for_selection(model: Model, selected_indices: Sequence[int]) -> float:
    total = 0.0
    for left, right in itertools.combinations(selected_indices, 2):
        total += model.collision_matrix[left, right]
    return total


def solve_bruteforce(model: Model) -> Solution:
    pair_ids = [pair.pair_id for pair in model.config.pairs]
    candidate_groups = [model.pair_state_indices[pair_id] for pair_id in pair_ids]

    best_indices: Tuple[int, ...] | None = None
    best_cost: float | None = None

    for selection in itertools.product(*candidate_groups):
        total = _total_collision_for_selection(model, selection)
        if best_cost is None or total < best_cost:
            best_cost = total
            best_indices = tuple(selection)

    assert best_indices is not None
    assert best_cost is not None
    return Solution(
        selected_indices=best_indices,
        selected_states={model.states[idx][0]: model.states[idx] for idx in best_indices},
        total_collision=best_cost,
    )


def round_from_scores(model: Model, scores: Sequence[float]) -> Dict[int, Tuple[int, int, int]]:
    if len(scores) != len(model.states):
        raise ValueError("scores length must equal number of states")

    selected: Dict[int, Tuple[int, int, int]] = {}
    for pair_id, indices in model.pair_state_indices.items():
        best_idx = max(indices, key=lambda idx: scores[idx])
        selected[pair_id] = model.states[best_idx]
    return selected


def describe_solution(model: Model, solution: Solution) -> Dict[str, object]:
    return {
        "macro_horizon": infer_macro_horizon(model.config),
        "channel_count": model.config.channel_count,
        "state_count": len(model.states),
        "total_collision": solution.total_collision,
        "selected_states": {
            str(pair_id): {
                "pair_id": state[0],
                "offset": state[1],
                "pattern_index": state[2],
            }
            for pair_id, state in solution.selected_states.items()
        },
    }


def build_demo_config() -> SchedulingConfig:
    return SchedulingConfig(
        pairs=(
            PairConfig(
                pair_id=0,
                offset_candidates=(0, 1, 2),
                pattern_count=2,
                interval=6,
                duration=2,
                event_count=3,
                channel_seed=0,
            ),
            PairConfig(
                pair_id=1,
                offset_candidates=(0, 2, 4),
                pattern_count=2,
                interval=6,
                duration=2,
                event_count=3,
                channel_seed=4,
            ),
            PairConfig(
                pair_id=2,
                offset_candidates=(1, 3, 5),
                pattern_count=2,
                interval=6,
                duration=2,
                event_count=3,
                channel_seed=9,
            ),
        ),
        channel_count=10,
    )


def main() -> None:
    config = build_demo_config()
    model = build_model(config)
    solution = solve_bruteforce(model)
    print(json.dumps(describe_solution(model, solution), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
