#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLE-only 宏周期 hopping 轨迹调度：完整可运行示例
================================================

实现内容：
1. 系统参数与数据结构定义
2. 候选状态 (offset, pattern) 枚举
3. 宏周期内轨迹碰撞代价 Omega 的预计算
4. Candidate-state lifted SDP 松弛
5. 从 SDP 解进行简单 rounding，恢复离散调度
6. 打印调度结果与碰撞统计

依赖：
    pip install numpy cvxpy

说明：
- hopping 规则这里使用简化版本：
    channel = (start_channel + hop_increment * event_index) mod num_channels
- 你后续替换真实 BLE hopping，只需要改 channel_of_event()。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Optional
import numpy as np

if TYPE_CHECKING:
    import cvxpy

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - fallback for environments without cvxpy
    cp = None


BLE_DATA_CHANNEL_COUNT = 37
BLE_ADVERTISING_CENTER_FREQ_MHZ = (2402.0, 2426.0, 2480.0)


@dataclass(frozen=True)
class PairConfig:
    pair_id: int
    release_time: int
    deadline: int
    connect_interval: int
    event_duration: int
    num_events: int


@dataclass(frozen=True)
class HoppingPattern:
    pattern_id: int
    start_channel: int
    hop_increment: int


@dataclass(frozen=True)
class CandidateState:
    pair_id: int
    offset: int
    pattern_id: int


@dataclass(frozen=True)
class EventBlock:
    pair_id: int
    event_index: int
    start_slot: int
    end_slot: int
    channel: int
    frequency_mhz: float
    offset: int
    pattern_id: int


@dataclass(frozen=True)
class ExternalInterferenceBlock:
    start_slot: int
    end_slot: int
    freq_low_mhz: float
    freq_high_mhz: float
    source_type: str
    source_pair_id: int


@dataclass(frozen=True)
class BLEStandaloneConfig:
    config_path: Optional[Path]
    num_channels: int
    pair_configs: List[PairConfig]
    cfg_dict: Dict[int, PairConfig]
    pattern_dict: Dict[int, List[HoppingPattern]]
    pair_weight: Dict[Tuple[int, int], float]
    hard_collision_threshold: Optional[float]
    plot_title: str
    output_path: Path


def compute_feasible_offsets(cfg: PairConfig) -> List[int]:
    """
    计算合法 offset 集合 S_k:
        S_k = { s | r_k <= s <= D_k - (M_k - 1) * Delta_k - d_k + 1 }
    """
    latest_start = (
        cfg.deadline
        - (cfg.num_events - 1) * cfg.connect_interval
        - cfg.event_duration
        + 1
    )
    if latest_start < cfg.release_time:
        return []
    return list(range(cfg.release_time, latest_start + 1))


def prune_feasible_offsets(offsets: List[int], max_offsets: Optional[int]) -> List[int]:
    """Deterministically down-sample feasible offsets while keeping edges."""
    if max_offsets is None or max_offsets <= 0 or len(offsets) <= max_offsets:
        return list(offsets)
    if max_offsets == 1:
        return [offsets[0]]
    if max_offsets == 2:
        return [offsets[0], offsets[-1]]

    raw_idx = np.linspace(0, len(offsets) - 1, num=max_offsets)
    chosen: List[int] = []
    seen = set()
    for idx in np.rint(raw_idx).astype(int):
        idx = int(max(0, min(len(offsets) - 1, idx)))
        if idx in seen:
            continue
        seen.add(idx)
        chosen.append(idx)

    if len(chosen) < max_offsets:
        for idx in range(len(offsets)):
            if idx in seen:
                continue
            chosen.append(idx)
            seen.add(idx)
            if len(chosen) == max_offsets:
                break

    chosen.sort()
    return [offsets[idx] for idx in chosen]


def event_start_time(cfg: PairConfig, offset: int, event_idx: int) -> int:
    """t_{k,m}(s) = s + m * Delta_k"""
    return offset + event_idx * cfg.connect_interval


def event_interval(cfg: PairConfig, offset: int, event_idx: int) -> Tuple[int, int]:
    """I_{k,m}(s) = [start, end]，闭区间表示"""
    start = event_start_time(cfg, offset, event_idx)
    end = start + cfg.event_duration - 1
    return start, end


def interval_overlap_length(interval1: Tuple[int, int], interval2: Tuple[int, int]) -> int:
    """两个闭区间的重叠长度"""
    l1, r1 = interval1
    l2, r2 = interval2
    left = max(l1, l2)
    right = min(r1, r2)
    return max(0, right - left + 1)


def channel_of_event(pattern: HoppingPattern, event_idx: int, num_channels: int) -> int:
    """
    简化 hopping 规则：
        c_{k,m}^{(ell)} = (start_channel + hop_increment * m) mod num_channels
    """
    return (pattern.start_channel + pattern.hop_increment * event_idx) % num_channels


def ble_channel_to_frequency_mhz(channel: int) -> float:
    """Map BLE data-channel index to its center frequency in MHz."""
    if not 0 <= channel < BLE_DATA_CHANNEL_COUNT:
        raise ValueError(f"BLE data channel must be within [0, {BLE_DATA_CHANNEL_COUNT - 1}].")
    if channel <= 10:
        return 2404.0 + 2.0 * channel
    return 2428.0 + 2.0 * (channel - 11)


def build_candidate_states(
    pair_configs: List[PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    max_offsets_per_pair: Optional[int] = None,
) -> Tuple[List[CandidateState], Dict[CandidateState, int], Dict[int, List[int]]]:
    """
    构造候选状态全集 A = union_k {(k, s, ell)}
    返回：
        states: 全部候选状态
        state_to_idx: 状态到索引
        A_k: 每个 pair 的候选状态索引集合
    """
    states: List[CandidateState] = []
    A_k: Dict[int, List[int]] = {}

    for cfg in pair_configs:
        k = cfg.pair_id
        feasible_offsets = prune_feasible_offsets(
            compute_feasible_offsets(cfg),
            max_offsets=max_offsets_per_pair,
        )
        if not feasible_offsets:
            raise ValueError(f"Pair {k} has no feasible offset.")

        if k not in pattern_dict or len(pattern_dict[k]) == 0:
            raise ValueError(f"Pair {k} has no candidate hopping pattern.")

        A_k[k] = []
        for s in feasible_offsets:
            for pat in pattern_dict[k]:
                st = CandidateState(pair_id=k, offset=s, pattern_id=pat.pattern_id)
                A_k[k].append(len(states))
                states.append(st)

    state_to_idx = {st: i for i, st in enumerate(states)}
    return states, state_to_idx, A_k


def summarize_candidate_space(
    pair_configs: List[PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    max_offsets_per_pair: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize the BLE candidate space before solving."""
    pair_summaries: List[Dict[str, Any]] = []
    total_state_count = 0
    for cfg in pair_configs:
        offsets = prune_feasible_offsets(
            compute_feasible_offsets(cfg),
            max_offsets=max_offsets_per_pair,
        )
        pattern_count = len(pattern_dict.get(cfg.pair_id, ()))
        state_count = len(offsets) * pattern_count
        total_state_count += state_count
        pair_summaries.append(
            {
                "pair_id": cfg.pair_id,
                "offsets": offsets,
                "offset_count": len(offsets),
                "pattern_count": pattern_count,
                "state_count": state_count,
            }
        )

    return {
        "pair_count": len(pair_configs),
        "state_count": total_state_count,
        "max_offsets_per_pair": max_offsets_per_pair,
        "pairs": pair_summaries,
    }


def lookup_pattern(
    pattern_dict: Dict[int, List[HoppingPattern]],
    pair_id: int,
    pattern_id: int,
) -> HoppingPattern:
    """按 (pair_id, pattern_id) 查找对应 pattern"""
    for pat in pattern_dict[pair_id]:
        if pat.pattern_id == pattern_id:
            return pat
    raise KeyError(f"Pattern {pattern_id} not found for pair {pair_id}.")


def build_event_blocks(
    selected: Dict[int, CandidateState],
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
) -> List[EventBlock]:
    """Expand selected candidate states into event-level time-frequency blocks."""
    blocks: List[EventBlock] = []
    for pair_id, state in sorted(selected.items()):
        cfg = cfg_dict[pair_id]
        pattern = lookup_pattern(pattern_dict, pair_id, state.pattern_id)
        for event_idx in range(cfg.num_events):
            start = event_start_time(cfg, state.offset, event_idx)
            end = start + cfg.event_duration - 1
            channel = channel_of_event(pattern, event_idx, num_channels)
            blocks.append(
                EventBlock(
                    pair_id=pair_id,
                    event_index=event_idx,
                    start_slot=start,
                    end_slot=end,
                    channel=channel,
                    frequency_mhz=ble_channel_to_frequency_mhz(channel),
                    offset=state.offset,
                    pattern_id=state.pattern_id,
                )
            )
    return blocks


def build_ble_advertising_idle_blocks(max_slot: int) -> List[EventBlock]:
    """Build full-width BLE advertising-channel bands for plotting."""
    if max_slot <= 0:
        return []
    return [
        EventBlock(
            pair_id=-1,
            event_index=-1,
            start_slot=0,
            end_slot=max_slot - 1,
            channel=-1,
            frequency_mhz=freq,
            offset=-1,
            pattern_id=-1,
        )
        for freq in BLE_ADVERTISING_CENTER_FREQ_MHZ
    ]


def _frequency_overlap_mhz(low0: float, high0: float, low1: float, high1: float) -> float:
    return max(0.0, min(high0, high1) - max(low0, low1))


def external_interference_cost_for_state(
    state: CandidateState,
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
    interference_blocks: Optional[List[ExternalInterferenceBlock]] = None,
) -> float:
    interference_blocks = interference_blocks or []
    if not interference_blocks:
        return 0.0

    blocks = build_event_blocks(
        selected={state.pair_id: state},
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=num_channels,
    )

    total = 0.0
    for block in blocks:
        block_low = float(block.frequency_mhz - 1.0)
        block_high = float(block.frequency_mhz + 1.0)
        for ext in interference_blocks:
            time_overlap = max(0, min(block.end_slot, ext.end_slot) - max(block.start_slot, ext.start_slot) + 1)
            if time_overlap <= 0:
                continue
            freq_overlap = _frequency_overlap_mhz(block_low, block_high, ext.freq_low_mhz, ext.freq_high_mhz)
            if freq_overlap <= 0.0:
                continue
            total += float(time_overlap)
    return total


def build_external_interference_cost_vector(
    states: List[CandidateState],
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
    interference_blocks: Optional[List[ExternalInterferenceBlock]] = None,
) -> np.ndarray:
    return np.asarray([
        external_interference_cost_for_state(
            state=state,
            cfg_dict=cfg_dict,
            pattern_dict=pattern_dict,
            num_channels=num_channels,
            interference_blocks=interference_blocks,
        )
        for state in states
    ], dtype=float)


def build_external_interference_forbidden_state_indices(
    pair_ids: List[int],
    A_k: Dict[int, List[int]],
    candidate_external_cost: np.ndarray,
) -> List[int]:
    candidate_external_cost = np.asarray(candidate_external_cost, dtype=float)
    forbidden = []
    for pair_id in pair_ids:
        indices = [int(idx) for idx in A_k[int(pair_id)]]
        if not indices:
            continue
        if any(candidate_external_cost[idx] <= 0.0 for idx in indices):
            forbidden.extend(idx for idx in indices if candidate_external_cost[idx] > 0.0)
    return forbidden


def build_overlap_blocks(blocks: List[EventBlock]) -> List[EventBlock]:
    """Build time-frequency segments where same-channel blocks overlap."""
    overlap_blocks: List[EventBlock] = []
    seen: set[tuple[int, int, int]] = set()
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            left = blocks[i]
            right = blocks[j]
            if left.channel != right.channel:
                continue

            start = max(left.start_slot, right.start_slot)
            end = min(left.end_slot, right.end_slot)
            if start > end:
                continue

            key = (left.channel, start, end)
            if key in seen:
                continue
            seen.add(key)
            overlap_blocks.append(
                EventBlock(
                    pair_id=-1,
                    event_index=-1,
                    start_slot=start,
                    end_slot=end,
                    channel=left.channel,
                    frequency_mhz=left.frequency_mhz,
                    offset=-1,
                    pattern_id=-1,
                )
            )
    return overlap_blocks


def selected_schedule_to_ce_channels(
    selected: Dict[int, CandidateState],
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
) -> Dict[int, np.ndarray]:
    """Convert selected states into per-pair event-channel arrays."""
    blocks = build_event_blocks(
        selected=selected,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=num_channels,
    )
    ce_channel_map: Dict[int, np.ndarray] = {}
    for pair_id in sorted(selected.keys()):
        cfg = cfg_dict[pair_id]
        pair_blocks = [block for block in blocks if block.pair_id == pair_id]
        channels = np.full(cfg.num_events, -1, dtype=int)
        for block in pair_blocks:
            if 0 <= block.event_index < cfg.num_events:
                channels[block.event_index] = int(block.channel)
        if np.any(channels < 0):
            raise ValueError(f"Pair {pair_id} does not have a complete event-channel map.")
        ce_channel_map[pair_id] = channels
    return ce_channel_map


def render_event_grid(
    blocks: List[EventBlock],
    overlap_blocks: List[EventBlock],
    output_path,
    title: str = "BLE Event Grid",
) -> None:
    """Render event blocks and overlaps to a PNG using matplotlib's Agg backend."""
    mpl_config_dir = Path(tempfile.gettempdir()) / "sig_sdp_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(14, 6))

    def _draw_block(block: EventBlock, color: str, alpha: float, label: str | None = None) -> None:
        width = block.end_slot - block.start_slot + 1
        rect = Rectangle(
            (block.start_slot, block.frequency_mhz - 0.8),
            width,
            1.6,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)
        if label is not None:
            ax.text(
                block.start_slot + width / 2.0,
                block.frequency_mhz,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    all_blocks = blocks + overlap_blocks
    max_slot = max((block.end_slot for block in all_blocks), default=-1) + 1
    idle_blocks = build_ble_advertising_idle_blocks(max_slot)

    for block in idle_blocks:
        _draw_block(block, color="#c7c7c7", alpha=0.45)

    for block in blocks:
        _draw_block(
            block,
            color="#dd8452",
            alpha=0.9,
            label=f"{block.pair_id} B-ch{block.channel} ev{block.event_index}",
        )

    for block in overlap_blocks:
        _draw_block(block, color="#e15759", alpha=0.75)

    plot_blocks = all_blocks + idle_blocks
    if plot_blocks:
        min_freq = min(block.frequency_mhz for block in plot_blocks) - 2.0
        max_freq = max(block.frequency_mhz for block in plot_blocks) + 2.0
        ax.set_ylim(min_freq, max_freq)
        ax.set_xlim(0, max_slot)

    ax.set_title(title)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Frequency (MHz)")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(
        handles=[
            Patch(color="#c7c7c7", label="BLE adv idle"),
            Patch(color="#dd8452", label="BLE"),
            Patch(color="#e15759", label="BLE overlap"),
        ]
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def strip_comment_keys(obj: Any) -> Any:
    """Remove `_comment_*` keys recursively from a JSON object."""
    if isinstance(obj, dict):
        return {
            key: strip_comment_keys(value)
            for key, value in obj.items()
            if not (isinstance(key, str) and key.startswith("_comment"))
        }
    if isinstance(obj, list):
        return [strip_comment_keys(item) for item in obj]
    return obj


def parse_pair_weight_map(raw_pair_weight: Any) -> Dict[Tuple[int, int], float]:
    """Parse optional pair-weight encodings from JSON."""
    if raw_pair_weight in (None, {}):
        return {}
    if not isinstance(raw_pair_weight, dict):
        raise ValueError("pair_weight must be a JSON object.")

    pair_weight: Dict[Tuple[int, int], float] = {}
    for raw_key, raw_value in raw_pair_weight.items():
        if isinstance(raw_key, str):
            if "," in raw_key:
                left, right = raw_key.split(",", 1)
            elif "-" in raw_key:
                left, right = raw_key.split("-", 1)
            else:
                raise ValueError(
                    "pair_weight keys must be encoded as 'i,j' or 'i-j' strings."
                )
            pair_key = (int(left.strip()), int(right.strip()))
        elif isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
            pair_key = (int(raw_key[0]), int(raw_key[1]))
        else:
            raise ValueError("Unsupported pair_weight key format.")

        if isinstance(raw_value, dict):
            if "weight" not in raw_value:
                raise ValueError("pair_weight entry dict must contain a 'weight' field.")
            weight_value = float(raw_value["weight"])
        else:
            weight_value = float(raw_value)

        pair_weight[(min(pair_key), max(pair_key))] = weight_value

    return pair_weight


def load_ble_standalone_config(config_path: Path) -> BLEStandaloneConfig:
    """Load the standalone BLE-only JSON config used by the demo script."""
    raw = json.loads(Path(config_path).read_text())
    data = strip_comment_keys(raw)
    if not isinstance(data, dict):
        raise ValueError("Standalone BLE config must be a JSON object.")

    num_channels = int(data.get("num_channels", BLE_DATA_CHANNEL_COUNT))
    hard_collision_threshold = data.get("hard_collision_threshold", None)
    if hard_collision_threshold is not None:
        hard_collision_threshold = float(hard_collision_threshold)

    pair_configs = [PairConfig(**item) for item in data.get("pair_configs", [])]
    pair_configs.sort(key=lambda cfg: cfg.pair_id)
    cfg_dict = {cfg.pair_id: cfg for cfg in pair_configs}

    raw_pattern_dict = data.get("pattern_dict", {})
    if not isinstance(raw_pattern_dict, dict):
        raise ValueError("pattern_dict must be a JSON object.")
    pattern_dict: Dict[int, List[HoppingPattern]] = {}
    for key, value in raw_pattern_dict.items():
        pair_id = int(key)
        if not isinstance(value, list):
            raise ValueError(f"pattern_dict[{pair_id}] must be a list.")
        patterns = [HoppingPattern(**item) for item in value]
        patterns.sort(key=lambda pat: pat.pattern_id)
        pattern_dict[pair_id] = patterns

    pair_weight = parse_pair_weight_map(data.get("pair_weight", {}))

    plot_title = str(data.get("plot_title", "BLE Event Grid"))
    output_path = data.get("output_path", "ble_macrocycle_hopping_sdp_schedule.png")
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = Path(config_path).resolve().parent / output_path

    return BLEStandaloneConfig(
        config_path=Path(config_path).resolve(),
        num_channels=num_channels,
        pair_configs=pair_configs,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        pair_weight=pair_weight,
        hard_collision_threshold=hard_collision_threshold,
        plot_title=plot_title,
        output_path=output_path,
    )


def build_demo_standalone_config() -> BLEStandaloneConfig:
    """Build the legacy demo configuration used when no JSON config is supplied."""
    pair_configs, cfg_dict, pattern_dict, pair_weight, num_channels = build_demo_instance()
    return BLEStandaloneConfig(
        config_path=None,
        num_channels=num_channels,
        pair_configs=pair_configs,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        pair_weight=pair_weight,
        hard_collision_threshold=None,
        plot_title="BLE Event Grid",
        output_path=Path(__file__).with_name("ble_macrocycle_hopping_sdp_schedule.png"),
    )


def resolve_standalone_config(config_path: Optional[Path]) -> BLEStandaloneConfig:
    """Resolve either the JSON config path or the built-in demo fallback."""
    if config_path is None:
        return build_demo_standalone_config()
    return load_ble_standalone_config(Path(config_path))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLE-only macrocycle hopping SDP demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config file such as ble_macrocycle_hopping_sdp_config.json",
    )
    return parser.parse_args(argv)


def require_cvxpy() -> None:
    """Fail fast when the SDP path is invoked without cvxpy installed."""
    if cp is None:
        raise RuntimeError("cvxpy is required for the SDP path. Install cvxpy to run main() or build_sdp_relaxation().")


def weighted_collision_cost_between_states(
    state_a: CandidateState,
    state_b: CandidateState,
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
    pair_weight: Optional[Dict[Tuple[int, int], float]] = None,
) -> float:
    """
    计算两个候选状态 a=(k,s,ell), b=(j,s',ell') 的轨迹碰撞代价 Omega_ab。

    Omega_ab = sum_m sum_n overlap(I_{k,m}(s), I_{j,n}(s'))
                         * 1{channel_{k,m} == channel_{j,n}}
    """
    k = state_a.pair_id
    j = state_b.pair_id

    cfg_k = cfg_dict[k]
    cfg_j = cfg_dict[j]
    pat_k = lookup_pattern(pattern_dict, k, state_a.pattern_id)
    pat_j = lookup_pattern(pattern_dict, j, state_b.pattern_id)

    total = 0.0
    for m in range(cfg_k.num_events):
        interval_k = event_interval(cfg_k, state_a.offset, m)
        channel_k = channel_of_event(pat_k, m, num_channels)

        for n in range(cfg_j.num_events):
            interval_j = event_interval(cfg_j, state_b.offset, n)
            channel_j = channel_of_event(pat_j, n, num_channels)

            overlap = interval_overlap_length(interval_k, interval_j)
            same_channel = 1 if channel_k == channel_j else 0
            total += overlap * same_channel

    if pair_weight is not None:
        key = (min(k, j), max(k, j))
        total *= pair_weight.get(key, 1.0)

    return total


def build_collision_matrix(
    states: List[CandidateState],
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
    pair_weight: Optional[Dict[Tuple[int, int], float]] = None,
) -> np.ndarray:
    """构造全部候选状态之间的碰撞代价矩阵 Omega"""
    A = len(states)
    Omega = np.zeros((A, A), dtype=float)

    for i in range(A):
        for j in range(i + 1, A):
            if states[i].pair_id == states[j].pair_id:
                continue

            cost = weighted_collision_cost_between_states(
                states[i],
                states[j],
                cfg_dict=cfg_dict,
                pattern_dict=pattern_dict,
                num_channels=num_channels,
                pair_weight=pair_weight,
            )
            Omega[i, j] = cost
            Omega[j, i] = cost

    return Omega


def build_sdp_relaxation(
    pair_ids: List[int],
    A_k: Dict[int, List[int]],
    Omega: np.ndarray,
    candidate_external_cost: Optional[np.ndarray] = None,
    forbidden_state_indices: Optional[List[int]] = None,
    hard_collision_threshold: Optional[float] = None,
) -> Tuple["cvxpy.Problem", "cvxpy.Variable"] | Tuple[Any, Any]:
    """
    构建 candidate-state lifted SDP 松弛：

        min   sum_{a<b} Omega_ab * Y_ab
        s.t.  sum_{a in A_k} Y_aa = 1
              Y_ab = 0,  for a != b and a,b in A_k
              Y >= 0 (PSD)

    可选：
        若 Omega_ab > hard_collision_threshold，则强制 Y_ab = 0
    """
    require_cvxpy()

    A = Omega.shape[0]
    Y = cp.Variable((A, A), symmetric=True)

    constraints = [Y >> 0]
    diagY = cp.diag(Y)

    # 每个 pair 只能选一个候选状态
    for k in pair_ids:
        constraints.append(cp.sum(diagY[A_k[k]]) == 1)

    # 同一个 pair 的不同候选状态不能同时选择
    for k in pair_ids:
        idxs = A_k[k]
        for i in idxs:
            for j in idxs:
                if i != j:
                    constraints.append(Y[i, j] == 0)

    # 可选：高碰撞状态对直接禁掉
    if hard_collision_threshold is not None:
        for i in range(A):
            for j in range(i + 1, A):
                if Omega[i, j] > hard_collision_threshold:
                    constraints.append(Y[i, j] == 0)
                    constraints.append(Y[j, i] == 0)

    forbidden_state_indices = [] if forbidden_state_indices is None else [int(idx) for idx in forbidden_state_indices]
    for idx in forbidden_state_indices:
        constraints.append(diagY[idx] == 0)

    # 只保留上三角，避免双层标量累加带来的 CVXPY 子表达式膨胀。
    objective_expr = cp.sum(cp.multiply(np.triu(Omega, k=1), Y))
    if candidate_external_cost is not None:
        objective_expr = objective_expr + cp.sum(cp.multiply(np.asarray(candidate_external_cost, dtype=float), diagY))

    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    return problem, Y


def round_solution_from_Y(
    Y_value: np.ndarray,
    states: List[CandidateState],
    A_k: Dict[int, List[int]],
) -> Dict[int, CandidateState]:
    """
    最简单的 rounding：
        对每个 pair k，选取对角元 Y_aa 最大的候选状态
    """
    diag_scores = np.diag(Y_value)
    selected: Dict[int, CandidateState] = {}

    for k, idxs in A_k.items():
        best_idx = max(idxs, key=lambda idx: diag_scores[idx])
        selected[k] = states[best_idx]

    return selected


def compute_total_collision_of_schedule(
    selected: Dict[int, CandidateState],
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
    pair_weight: Optional[Dict[Tuple[int, int], float]] = None,
) -> float:
    """计算离散调度结果的总碰撞代价"""
    pair_ids = sorted(selected.keys())
    total = 0.0

    for i in range(len(pair_ids)):
        for j in range(i + 1, len(pair_ids)):
            k = pair_ids[i]
            kp = pair_ids[j]
            total += weighted_collision_cost_between_states(
                selected[k],
                selected[kp],
                cfg_dict=cfg_dict,
                pattern_dict=pattern_dict,
                num_channels=num_channels,
                pair_weight=pair_weight,
            )
    return total


def solve_ble_hopping_schedule(
    pair_configs: List[PairConfig],
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    pair_ids: List[int],
    A_k: Dict[int, List[int]],
    states: List[CandidateState],
    num_channels: int,
    pair_weight: Optional[Dict[Tuple[int, int], float]] = None,
    external_interference_blocks: Optional[List[ExternalInterferenceBlock]] = None,
    hard_collision_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve the BLE hopping SDP and return the rounded discrete schedule."""
    require_cvxpy()
    Omega = build_collision_matrix(
        states=states,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=num_channels,
        pair_weight=pair_weight,
    )
    candidate_external_cost = build_external_interference_cost_vector(
        states=states,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=num_channels,
        interference_blocks=external_interference_blocks,
    )
    forbidden_state_indices = build_external_interference_forbidden_state_indices(
        pair_ids=pair_ids,
        A_k=A_k,
        candidate_external_cost=candidate_external_cost,
    )
    problem, Y = build_sdp_relaxation(
        pair_ids=pair_ids,
        A_k=A_k,
        Omega=Omega,
        candidate_external_cost=candidate_external_cost,
        forbidden_state_indices=forbidden_state_indices,
        hard_collision_threshold=hard_collision_threshold,
    )
    problem.solve(solver=cp.SCS, verbose=False)
    if Y.value is None:
        raise RuntimeError("SDP solver did not return a solution.")
    selected = round_solution_from_Y(Y.value, states, A_k)
    blocks = build_event_blocks(selected, cfg_dict, pattern_dict, num_channels)
    overlap_blocks = build_overlap_blocks(blocks)
    ce_channel_map = selected_schedule_to_ce_channels(selected, cfg_dict, pattern_dict, num_channels)
    return {
        "problem": problem,
        "Y": Y,
        "selected": selected,
        "blocks": blocks,
        "overlap_blocks": overlap_blocks,
        "ce_channel_map": ce_channel_map,
        "objective_value": float(problem.value),
    }


def print_candidate_summary(
    pair_configs: List[PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    max_offsets_per_pair: Optional[int] = None,
) -> None:
    """打印每个 pair 的合法 offset 集与可选 pattern"""
    print("=" * 72)
    print("BLE candidate summary")
    print("=" * 72)
    summary = summarize_candidate_space(
        pair_configs=pair_configs,
        pattern_dict=pattern_dict,
        max_offsets_per_pair=max_offsets_per_pair,
    )
    cfg_by_id = {cfg.pair_id: cfg for cfg in pair_configs}
    print(
        f"pair_count={summary['pair_count']}, "
        f"state_count={summary['state_count']}, "
        f"max_offsets_per_pair={summary['max_offsets_per_pair']}"
    )
    for row in summary["pairs"]:
        cfg = cfg_by_id[row["pair_id"]]
        pats = pattern_dict[cfg.pair_id]
        offsets = row["offsets"]
        print(
            f"Pair {cfg.pair_id}: offset_count={row['offset_count']}, "
            f"pattern_count={row['pattern_count']}, "
            f"state_count={row['state_count']}, "
            f"offsets={offsets}, "
            f"patterns={[p.pattern_id for p in pats]}, "
            f"(r={cfg.release_time}, D={cfg.deadline}, "
            f"Delta={cfg.connect_interval}, d={cfg.event_duration}, M={cfg.num_events})"
        )
    print()


def print_selected_schedule(
    selected: Dict[int, CandidateState],
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
) -> None:
    """打印 rounding 后的调度结果"""
    print("=" * 72)
    print("离散调度结果")
    print("=" * 72)

    for k in sorted(selected.keys()):
        st = selected[k]
        cfg = cfg_dict[k]
        pat = lookup_pattern(pattern_dict, k, st.pattern_id)

        print(
            f"Pair {k}: offset={st.offset}, pattern={st.pattern_id}, "
            f"start_channel={pat.start_channel}, hop_increment={pat.hop_increment}"
        )
        for m in range(cfg.num_events):
            start = event_start_time(cfg, st.offset, m)
            end = start + cfg.event_duration - 1
            ch = channel_of_event(pat, m, num_channels)
            print(f"    Event {m}: time=[{start}, {end}], channel={ch}")
        print()


def print_event_block_table(blocks: List[EventBlock]) -> None:
    """Print the expanded event-level time-frequency blocks."""
    print("=" * 72)
    print("事件级时频块表")
    print("=" * 72)
    for block in blocks:
        print(
            f"pair={block.pair_id}, ev={block.event_index}, "
            f"time=[{block.start_slot}, {block.end_slot}], "
            f"channel={block.channel}, freq={block.frequency_mhz:.1f} MHz, "
            f"offset={block.offset}, pattern={block.pattern_id}"
    )
    print()


def render_selected_schedule(
    selected: Dict[int, CandidateState],
    cfg_dict: Dict[int, PairConfig],
    pattern_dict: Dict[int, List[HoppingPattern]],
    num_channels: int,
    output_path: Path,
    title: str = "BLE Event Grid",
) -> Tuple[List[EventBlock], List[EventBlock]]:
    """Expand, summarize, and render a selected schedule without solving it."""
    blocks = build_event_blocks(
        selected=selected,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=num_channels,
    )
    overlap_blocks = build_overlap_blocks(blocks)
    print_event_block_table(blocks)
    render_event_grid(
        blocks=blocks,
        overlap_blocks=overlap_blocks,
        output_path=output_path,
        title=title,
    )
    return blocks, overlap_blocks


def build_demo_instance():
    """
    构造一个最小可运行示例。
    你后续可以把这里替换成自己的实验参数。
    """
    num_channels = 37

    pair_configs = [
        PairConfig(pair_id=0, release_time=1, deadline=18, connect_interval=4, event_duration=1, num_events=4),
        PairConfig(pair_id=1, release_time=2, deadline=19, connect_interval=4, event_duration=1, num_events=4),
        PairConfig(pair_id=2, release_time=1, deadline=16, connect_interval=3, event_duration=1, num_events=4),
        PairConfig(pair_id=3, release_time=4, deadline=20, connect_interval=5, event_duration=1, num_events=3),
    ]
    cfg_dict = {cfg.pair_id: cfg for cfg in pair_configs}

    pattern_dict = {
        0: [
            HoppingPattern(pattern_id=0, start_channel=1, hop_increment=5),
            HoppingPattern(pattern_id=1, start_channel=7, hop_increment=9),
        ],
        1: [
            HoppingPattern(pattern_id=0, start_channel=2, hop_increment=5),
            HoppingPattern(pattern_id=1, start_channel=10, hop_increment=11),
        ],
        2: [
            HoppingPattern(pattern_id=0, start_channel=3, hop_increment=7),
            HoppingPattern(pattern_id=1, start_channel=9, hop_increment=5),
            HoppingPattern(pattern_id=2, start_channel=15, hop_increment=9),
        ],
        3: [
            HoppingPattern(pattern_id=0, start_channel=4, hop_increment=5),
            HoppingPattern(pattern_id=1, start_channel=20, hop_increment=7),
        ],
    }

    pair_weight = {
        (0, 1): 1.0,
        (0, 2): 1.2,
        (0, 3): 1.0,
        (1, 2): 1.5,
        (1, 3): 1.0,
        (2, 3): 1.3,
    }

    return pair_configs, cfg_dict, pattern_dict, pair_weight, num_channels


def run_ble_macrocycle_hopping_sdp(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Solve and render the BLE-only schedule either from a JSON config or from the demo fallback.
    """
    require_cvxpy()

    runtime = resolve_standalone_config(config_path)
    pair_configs = runtime.pair_configs
    cfg_dict = runtime.cfg_dict
    pattern_dict = runtime.pattern_dict
    pair_weight = runtime.pair_weight
    num_channels = runtime.num_channels
    pair_ids = [cfg.pair_id for cfg in pair_configs]

    print_candidate_summary(pair_configs, pattern_dict)

    states, state_to_idx, A_k = build_candidate_states(pair_configs, pattern_dict)
    A = len(states)

    print("=" * 72)
    print(f"候选状态总数 |A| = {A}")
    print("=" * 72)
    for idx, st in enumerate(states):
        print(f"idx={idx:2d} -> (pair={st.pair_id}, offset={st.offset}, pattern={st.pattern_id})")
    print()

    Omega = build_collision_matrix(
        states=states,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=num_channels,
        pair_weight=pair_weight,
    )

    print("=" * 72)
    print("Omega 的前 10x10 子块")
    print("=" * 72)
    max_show = min(10, A)
    print(np.array_str(Omega[:max_show, :max_show], precision=2, suppress_small=True))
    print()

    problem, Y = build_sdp_relaxation(
        pair_ids=pair_ids,
        A_k=A_k,
        Omega=Omega,
        hard_collision_threshold=None,  # 你也可以改成 0 / 1 / 2 等阈值试验
    )

    print("=" * 72)
    print("开始求解 SDP 松弛")
    print("=" * 72)
    problem.solve(solver=cp.SCS, verbose=False)

    print(f"求解状态: {problem.status}")
    print(f"SDP 目标值: {problem.value:.4f}")
    print()

    if Y.value is None:
        raise RuntimeError("求解器没有返回有效解，请检查 cvxpy / solver。")

    selected = round_solution_from_Y(Y.value, states, A_k)

    print_selected_schedule(selected, cfg_dict, pattern_dict, num_channels)

    blocks = build_event_blocks(selected, cfg_dict, pattern_dict, num_channels)
    overlap_blocks = build_overlap_blocks(blocks)
    print_event_block_table(blocks)

    total_collision = compute_total_collision_of_schedule(
        selected=selected,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        num_channels=num_channels,
        pair_weight=pair_weight,
    )

    print("=" * 72)
    print(f"rounding 后离散调度的总碰撞代价: {total_collision:.4f}")
    print("=" * 72)

    render_event_grid(blocks, overlap_blocks, runtime.output_path, title=runtime.plot_title)
    print(f"调度图已保存: {runtime.output_path}")

    print("\n各候选状态的对角元分数 Y_aa:")
    diag_scores = np.diag(Y.value)
    for idx, score in enumerate(diag_scores):
        st = states[idx]
        print(
            f"idx={idx:2d}, pair={st.pair_id}, offset={st.offset}, "
            f"pattern={st.pattern_id}, score={score:.4f}"
        )

    return {
        "runtime": runtime,
        "states": states,
        "state_to_idx": state_to_idx,
        "A_k": A_k,
        "Omega": Omega,
        "problem": problem,
        "Y": Y,
        "selected": selected,
        "blocks": blocks,
        "overlap_blocks": overlap_blocks,
        "total_collision": total_collision,
    }


def main(argv: Optional[List[str]] = None) -> None:
    """
    主流程：
        1. 从 JSON 或 demo 构造实例
        2. 枚举候选状态
        3. 预计算 Omega
        4. 解 SDP
        5. rounding 恢复离散调度
        6. 打印结果
    """
    args = parse_args(argv)
    run_ble_macrocycle_hopping_sdp(args.config)


if __name__ == "__main__":
    main()
