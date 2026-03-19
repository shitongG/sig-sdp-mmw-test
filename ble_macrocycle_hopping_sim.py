
#!/usr/bin/env python3
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from pathlib import Path
import itertools
import json
import math
import csv

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class BLEPair:
    name: str
    r: int
    D: int
    Delta: int
    d: int
    M: int
    patterns: List[List[int]]


@dataclass(frozen=True)
class CandidateState:
    pair_idx: int
    pair_name: str
    offset: int
    pattern_idx: int
    pattern: Tuple[int, ...]

    def label(self) -> str:
        return f"{self.pair_name}|s={self.offset}|p={self.pattern_idx}"


def event_start(offset: int, m: int, Delta: int) -> int:
    return offset + m * Delta


def event_interval(offset: int, m: int, Delta: int, d: int) -> Tuple[int, int]:
    st = event_start(offset, m, Delta)
    ed = st + d - 1
    return st, ed


def interval_overlap_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    return max(0, right - left + 1)


def feasible_offsets(pair: BLEPair) -> List[int]:
    s_max = pair.D - (pair.M - 1) * pair.Delta - pair.d + 1
    if s_max < pair.r:
        return []
    return list(range(pair.r, s_max + 1))


def build_candidate_states(pairs: List[BLEPair]) -> Dict[int, List[CandidateState]]:
    out: Dict[int, List[CandidateState]] = {}
    for k, pair in enumerate(pairs):
        states: List[CandidateState] = []
        offsets = feasible_offsets(pair)
        for s in offsets:
            for ell, patt in enumerate(pair.patterns):
                if len(patt) != pair.M:
                    raise ValueError(
                        f"Pattern length mismatch for pair {pair.name}: "
                        f"expected M={pair.M}, got {len(patt)}"
                    )
                states.append(
                    CandidateState(
                        pair_idx=k,
                        pair_name=pair.name,
                        offset=s,
                        pattern_idx=ell,
                        pattern=tuple(patt),
                    )
                )
        out[k] = states
    return out


def pair_collision_cost(
    state_a: CandidateState,
    pair_a: BLEPair,
    state_b: CandidateState,
    pair_b: BLEPair,
) -> int:
    total = 0
    for m in range(pair_a.M):
        ia = event_interval(state_a.offset, m, pair_a.Delta, pair_a.d)
        ca = state_a.pattern[m]
        for n in range(pair_b.M):
            ib = event_interval(state_b.offset, n, pair_b.Delta, pair_b.d)
            cb = state_b.pattern[n]
            if ca == cb:
                total += interval_overlap_len(ia, ib)
    return total


def precompute_collision_matrix(
    pairs: List[BLEPair],
    candidates: Dict[int, List[CandidateState]],
) -> Dict[Tuple[str, str], int]:
    omega: Dict[Tuple[str, str], int] = {}
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            for a in candidates[i]:
                for b in candidates[j]:
                    key = (a.label(), b.label())
                    omega[key] = pair_collision_cost(a, pairs[i], b, pairs[j])
    return omega


def total_schedule_cost(
    assignment: Dict[int, CandidateState],
    pairs: List[BLEPair],
) -> int:
    total = 0
    K = len(pairs)
    for i in range(K):
        for j in range(i + 1, K):
            total += pair_collision_cost(assignment[i], pairs[i], assignment[j], pairs[j])
    return total


def exhaustive_search_best_schedule(
    pairs: List[BLEPair],
    candidates: Dict[int, List[CandidateState]],
) -> Tuple[Dict[int, CandidateState], int]:
    index_lists = [list(range(len(candidates[k]))) for k in range(len(pairs))]
    best_assignment = None
    best_cost = math.inf
    for combo in itertools.product(*index_lists):
        assignment = {k: candidates[k][combo[k]] for k in range(len(pairs))}
        cost = total_schedule_cost(assignment, pairs)
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment
    if best_assignment is None:
        raise RuntimeError("No feasible assignment found.")
    return best_assignment, int(best_cost)


def build_slot_channel_occupancy(
    assignment: Dict[int, CandidateState],
    pairs: List[BLEPair],
) -> Dict[Tuple[int, int], List[str]]:
    occ: Dict[Tuple[int, int], List[str]] = {}
    for k, state in assignment.items():
        pair = pairs[k]
        for m in range(pair.M):
            ch = state.pattern[m]
            st, ed = event_interval(state.offset, m, pair.Delta, pair.d)
            for t in range(st, ed + 1):
                occ.setdefault((t, ch), []).append(pair.name)
    return occ


def extract_collision_records(
    assignment: Dict[int, CandidateState],
    pairs: List[BLEPair],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            si = assignment[i]
            sj = assignment[j]
            pi = pairs[i]
            pj = pairs[j]
            for m in range(pi.M):
                ii = event_interval(si.offset, m, pi.Delta, pi.d)
                ci = si.pattern[m]
                for n in range(pj.M):
                    ij = event_interval(sj.offset, n, pj.Delta, pj.d)
                    cj = sj.pattern[n]
                    overlap = interval_overlap_len(ii, ij)
                    if overlap > 0 and ci == cj:
                        rows.append({
                            "pair_a": pi.name,
                            "event_a": m,
                            "pair_b": pj.name,
                            "event_b": n,
                            "channel": ci,
                            "interval_a": ii,
                            "interval_b": ij,
                            "collision_slots": overlap,
                        })
    return rows


def plot_gantt(
    assignment: Dict[int, CandidateState],
    pairs: List[BLEPair],
    macro_horizon: int,
    output_path: Path,
) -> None:
    occ = build_slot_channel_occupancy(assignment, pairs)
    fig, ax = plt.subplots(figsize=(14, 1.2 * len(pairs) + 2))
    row_height = 0.8

    for row, k in enumerate(range(len(pairs))):
        state = assignment[k]
        pair = pairs[k]
        y = len(pairs) - 1 - row
        for m in range(pair.M):
            st, ed = event_interval(state.offset, m, pair.Delta, pair.d)
            width = ed - st + 1
            ch = state.pattern[m]
            collides = False
            for t in range(st, ed + 1):
                users = occ.get((t, ch), [])
                if len(users) > 1:
                    collides = True
                    break
            rect = Rectangle(
                (st - 0.5, y - row_height / 2),
                width,
                row_height,
                fill=False,
                hatch='///' if collides else None,
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                st - 0.5 + width / 2,
                y,
                f"ch{ch}",
                ha="center",
                va="center",
                fontsize=9,
            )
        ax.text(-0.5, y, pair.name, ha="right", va="center", fontsize=10)

    ax.set_xlim(0.5, macro_horizon + 0.5)
    ax.set_ylim(-1, len(pairs))
    ax.set_xticks(range(1, macro_horizon + 1))
    ax.set_yticks([])
    ax.set_xlabel("Time slot")
    ax.set_title("BLE Macro-Cycle Hopping Schedule (hatched blocks indicate collisions)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_demo_instance() -> Tuple[List[BLEPair], int]:
    pairs = [
        BLEPair(
            name="BLE_1",
            r=1,
            D=18,
            Delta=4,
            d=2,
            M=4,
            patterns=[
                [5, 9, 13, 17],
                [5, 10, 15, 20],
                [8, 9, 10, 11],
            ],
        ),
        BLEPair(
            name="BLE_2",
            r=2,
            D=20,
            Delta=4,
            d=2,
            M=4,
            patterns=[
                [9, 13, 17, 21],
                [10, 15, 20, 25],
                [8, 10, 12, 14],
            ],
        ),
        BLEPair(
            name="BLE_3",
            r=1,
            D=17,
            Delta=3,
            d=1,
            M=5,
            patterns=[
                [5, 9, 13, 17, 21],
                [6, 10, 14, 18, 22],
                [8, 9, 10, 11, 12],
            ],
        ),
    ]
    H = max(p.D for p in pairs)
    return pairs, H


def main() -> None:
    out_dir = Path.cwd()
    pairs, H = build_demo_instance()
    candidates = build_candidate_states(pairs)
    for k, cand_list in candidates.items():
        if not cand_list:
            raise RuntimeError(f"No feasible candidate states for pair {pairs[k].name}")

    omega = precompute_collision_matrix(pairs, candidates)
    best_assignment, best_cost = exhaustive_search_best_schedule(pairs, candidates)
    collision_rows = extract_collision_records(best_assignment, pairs)

    summary = {
        "macro_horizon": H,
        "best_cost": best_cost,
        "pairs": [asdict(p) for p in pairs],
        "chosen_states": {
            pairs[k].name: {
                "offset": best_assignment[k].offset,
                "pattern_idx": best_assignment[k].pattern_idx,
                "pattern": list(best_assignment[k].pattern),
            }
            for k in range(len(pairs))
        },
        "collision_records": collision_rows,
    }

    json_path = out_dir / "ble_macrocycle_best_schedule.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "ble_macrocycle_collision_pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_a", "candidate_b", "omega_collision_cost"])
        for (a, b), val in sorted(omega.items()):
            writer.writerow([a, b, val])

    png_path = out_dir / "ble_macrocycle_gantt.png"
    plot_gantt(best_assignment, pairs, H, png_path)

    print("=== BLE Macro-Cycle Hopping Scheduling Demo ===")
    print(f"Macro horizon H = {H}")
    for k in range(len(pairs)):
        st = best_assignment[k]
        print(
            f"{pairs[k].name}: offset={st.offset}, pattern_idx={st.pattern_idx}, pattern={list(st.pattern)}"
        )
    print(f"Best total collision cost = {best_cost}")
    print(f"Wrote: {json_path.name}")
    print(f"Wrote: {csv_path.name}")
    print(f"Wrote: {png_path.name}")


if __name__ == "__main__":
    main()
