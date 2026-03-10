import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.mmw import mmw
from sim_src.env.env import env
from sim_src.util import resolve_torch_device


def _aggregate_office_stats_from_arrays(
    office_id: np.ndarray,
    radio: np.ndarray,
    z_vec: np.ndarray,
    n_office: int,
    wifi_id: int,
    ble_id: int,
):
    """Aggregate pair counts and slot usage per office."""
    rows = []
    for office in range(n_office):
        office_pairs = np.where(office_id == office)[0]
        wifi_pairs = office_pairs[radio[office_pairs] == wifi_id]
        ble_pairs = office_pairs[radio[office_pairs] == ble_id]
        rows.append(
            {
                "office_id": int(office),
                "wifi_pair_count": int(wifi_pairs.size),
                "ble_pair_count": int(ble_pairs.size),
                "wifi_slots_used": int(np.unique(z_vec[wifi_pairs]).size) if wifi_pairs.size else 0,
                "ble_slots_used": int(np.unique(z_vec[ble_pairs]).size) if ble_pairs.size else 0,
            }
        )
    return rows


def compute_office_pair_slot_stats(e: env, z_vec: np.ndarray):
    return _aggregate_office_stats_from_arrays(
        office_id=e.pair_office_id,
        radio=e.pair_radio_type,
        z_vec=z_vec,
        n_office=e.n_office,
        wifi_id=e.RADIO_WIFI,
        ble_id=e.RADIO_BLE,
    )


def compute_office_pair_slot_stats_for_pair_ids(e: env, z_vec: np.ndarray, pair_ids):
    pair_ids = np.asarray(pair_ids, dtype=int)
    if pair_ids.size == 0:
        return _aggregate_office_stats_from_arrays(
            office_id=np.array([], dtype=int),
            radio=np.array([], dtype=int),
            z_vec=np.array([], dtype=int),
            n_office=e.n_office,
            wifi_id=e.RADIO_WIFI,
            ble_id=e.RADIO_BLE,
        )
    return _aggregate_office_stats_from_arrays(
        office_id=e.pair_office_id[pair_ids],
        radio=e.pair_radio_type[pair_ids],
        z_vec=np.asarray(z_vec, dtype=int)[pair_ids],
        n_office=e.n_office,
        wifi_id=e.RADIO_WIFI,
        ble_id=e.RADIO_BLE,
    )


def _sorted_candidate_starts(period_slots: int, preferred_slot: int):
    period_slots = int(period_slots)
    preferred_slot = int(preferred_slot % period_slots)
    candidates = list(range(period_slots))
    candidates.sort(key=lambda s: (min((s - preferred_slot) % period_slots, (preferred_slot - s) % period_slots), s))
    return candidates


def assign_macrocycle_start_slots(
    e: env,
    preferred_slots: np.ndarray,
    allow_partial: bool = False,
    pair_order: np.ndarray | None = None,
):
    preferred_slots = np.asarray(preferred_slots, dtype=int).ravel()
    macrocycle_slots = e.compute_macrocycle_slots()
    if macrocycle_slots <= 0:
        return np.zeros(e.n_pair, dtype=int), 0, np.zeros((e.n_pair, 0), dtype=bool), []

    assigned_starts = np.full(e.n_pair, -1, dtype=int)
    occupancies = np.zeros((e.n_pair, macrocycle_slots), dtype=bool)
    conflict = e.build_pair_conflict_matrix()
    slot_asn = [[] for _ in range(macrocycle_slots)]
    s_gain, _, h_max = e.get_macrocycle_conflict_state()
    s_gain = s_gain.copy().tocsr()
    s_gain.setdiag(0.0)
    s_gain.eliminate_zeros()
    slot_gain_sum = [np.zeros(e.n_pair, dtype=float) for _ in range(macrocycle_slots)]
    unscheduled = []

    period_slots = e.get_pair_period_slots()
    width_slots = e.get_pair_width_slots()
    if pair_order is None:
        order = np.lexsort((preferred_slots, -e.pair_priority, -width_slots))
    else:
        order = np.asarray(pair_order, dtype=int).ravel()
    for pair_id in order:
        pair_id = int(pair_id)
        if e.pair_radio_type[pair_id] == e.RADIO_BLE and not e.pair_ble_ce_feasible[pair_id]:
            if allow_partial:
                unscheduled.append(pair_id)
                continue
            raise ValueError(f"BLE pair {pair_id} is infeasible under CE constraints.")

        period = int(period_slots[pair_id])
        if period <= 0:
            if allow_partial:
                unscheduled.append(pair_id)
                continue
            raise ValueError(f"Pair {pair_id} has invalid period {period}.")

        assigned = False
        for start_slot in _sorted_candidate_starts(period, preferred_slots[pair_id]):
            occ = e.expand_pair_occupancy(pair_id, start_slot, macrocycle_slots)
            occ_slots = np.where(occ)[0]
            tmp_h = np.asarray(s_gain[pair_id].toarray()).ravel()
            violates = False
            for slot in occ_slots:
                for other_pair in slot_asn[int(slot)]:
                    if conflict[pair_id, other_pair]:
                        violates = True
                        break
                if violates:
                    break
                neighbor_index = np.append(np.array(slot_asn[int(slot)], dtype=int), pair_id).astype(int)
                vio_h = (slot_gain_sum[int(slot)][neighbor_index] + tmp_h[neighbor_index]) > h_max[neighbor_index]
                if np.any(vio_h):
                    violates = True
                    break
                if violates:
                    break
            if violates:
                continue
            assigned_starts[pair_id] = int(start_slot)
            occupancies[pair_id] = occ
            for slot in occ_slots:
                slot_asn[int(slot)].append(pair_id)
                slot_gain_sum[int(slot)] += tmp_h
            assigned = True
            break

        if not assigned:
            unscheduled.append(pair_id)
            if allow_partial:
                continue
            raise ValueError(f"Unable to assign non-overlapping macrocycle start slot for pair {pair_id}.")

    return assigned_starts, macrocycle_slots, occupancies, unscheduled


def _repair_macrocycle_assignment_by_reordering(
    e: env,
    preferred_slots: np.ndarray,
    best_result,
    max_repair_passes: int = 2,
):
    best_starts, best_macrocycle_slots, best_occupancies, best_unscheduled = best_result
    best_unscheduled = list(best_unscheduled)
    if not best_unscheduled:
        return best_starts, best_macrocycle_slots, best_occupancies, best_unscheduled

    for _ in range(int(max_repair_passes)):
        if not best_unscheduled:
            break

        period_slots = e.get_pair_period_slots()
        unscheduled_set = set(best_unscheduled)
        scheduled = [pair_id for pair_id in range(e.n_pair) if pair_id not in unscheduled_set]
        candidate_orders = [
            np.array(best_unscheduled + scheduled, dtype=int),
            np.array(
                best_unscheduled
                + sorted(
                    scheduled,
                    key=lambda pair_id: (
                        -float(e.pair_priority[pair_id]),
                        -int(e.get_pair_width_slots()[pair_id]),
                        int(preferred_slots[pair_id]),
                    ),
                ),
                dtype=int,
            ),
            np.array(
                best_unscheduled
                + sorted(
                    scheduled,
                    key=lambda pair_id: (
                        -int(e.get_pair_width_slots()[pair_id]),
                        -float(e.pair_priority[pair_id]),
                        int(preferred_slots[pair_id]),
                    ),
                ),
                dtype=int,
            ),
        ]

        improved = False
        for candidate_order in candidate_orders:
            staggered_preferred = np.asarray(preferred_slots, dtype=int).copy()
            for rank, pair_id in enumerate(candidate_order):
                period = int(period_slots[int(pair_id)])
                if period > 0:
                    staggered_preferred[int(pair_id)] = rank % period
            for candidate_preferred in (np.asarray(preferred_slots, dtype=int), staggered_preferred):
                candidate = assign_macrocycle_start_slots(
                    e,
                    candidate_preferred,
                    allow_partial=True,
                    pair_order=candidate_order,
                )
                candidate_unscheduled = list(candidate[3])
                if len(candidate_unscheduled) < len(best_unscheduled):
                    best_starts = candidate[0].copy()
                    best_macrocycle_slots = int(candidate[1])
                    best_occupancies = candidate[2].copy()
                    best_unscheduled = candidate_unscheduled
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_starts, best_macrocycle_slots, best_occupancies, best_unscheduled


def _is_better_refill_result(candidate, best, pair_priority: np.ndarray):
    candidate_unscheduled = list(candidate[3])
    best_unscheduled = list(best[3])
    if len(candidate_unscheduled) != len(best_unscheduled):
        return len(candidate_unscheduled) < len(best_unscheduled)
    candidate_priority = float(np.sum(np.asarray(pair_priority)[candidate_unscheduled])) if candidate_unscheduled else 0.0
    best_priority = float(np.sum(np.asarray(pair_priority)[best_unscheduled])) if best_unscheduled else 0.0
    return candidate_priority < best_priority


def _refill_unscheduled_pairs_by_radio(
    e: env,
    preferred_slots: np.ndarray,
    best_result,
    target_radio_type: int,
    max_refill_passes: int = 2,
):
    best = (
        best_result[0].copy(),
        int(best_result[1]),
        best_result[2].copy(),
        list(best_result[3]),
    )
    for _ in range(int(max_refill_passes)):
        best_unscheduled = list(best[3])
        target_unscheduled = [
            pair_id for pair_id in best_unscheduled if int(e.pair_radio_type[pair_id]) == int(target_radio_type)
        ]
        if not target_unscheduled:
            break

        other_unscheduled = [pair_id for pair_id in best_unscheduled if pair_id not in set(target_unscheduled)]
        scheduled = [pair_id for pair_id in range(e.n_pair) if pair_id not in set(best_unscheduled)]
        period_slots = e.get_pair_period_slots()
        candidate_orders = [
            np.array(target_unscheduled + other_unscheduled + scheduled, dtype=int),
            np.array(
                target_unscheduled
                + sorted(
                    other_unscheduled + scheduled,
                    key=lambda pair_id: (
                        -float(e.pair_priority[pair_id]),
                        -int(e.get_pair_width_slots()[pair_id]),
                        int(preferred_slots[pair_id]),
                    ),
                ),
                dtype=int,
            ),
            np.array(
                target_unscheduled
                + sorted(
                    other_unscheduled + scheduled,
                    key=lambda pair_id: (
                        -int(e.get_pair_width_slots()[pair_id]),
                        -float(e.pair_priority[pair_id]),
                        int(preferred_slots[pair_id]),
                    ),
                ),
                dtype=int,
            ),
        ]

        improved = False
        for candidate_order in candidate_orders:
            staggered_preferred = np.asarray(preferred_slots, dtype=int).copy()
            for rank, pair_id in enumerate(candidate_order):
                period = int(period_slots[int(pair_id)])
                if period > 0:
                    staggered_preferred[int(pair_id)] = rank % period
            for candidate_preferred in (np.asarray(preferred_slots, dtype=int), staggered_preferred):
                candidate = assign_macrocycle_start_slots(
                    e,
                    candidate_preferred,
                    allow_partial=True,
                    pair_order=candidate_order,
                )
                if _is_better_refill_result(candidate, best, e.pair_priority):
                    best = (
                        candidate[0].copy(),
                        int(candidate[1]),
                        candidate[2].copy(),
                        list(candidate[3]),
                    )
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _apply_refill_pipeline(e: env, preferred_slots: np.ndarray, initial_result):
    best = _repair_macrocycle_assignment_by_reordering(
        e,
        preferred_slots,
        (
            initial_result[0].copy(),
            int(initial_result[1]),
            initial_result[2].copy(),
            list(initial_result[3]),
        ),
    )
    best = _refill_unscheduled_pairs_by_radio(
        e,
        preferred_slots,
        best,
        target_radio_type=e.RADIO_WIFI,
    )
    best = _refill_unscheduled_pairs_by_radio(
        e,
        preferred_slots,
        best,
        target_radio_type=e.RADIO_BLE,
    )
    return best


def retry_ble_channels_and_assign_macrocycle(e: env, preferred_slots: np.ndarray, max_ble_channel_retries: int = 0):
    starts, macrocycle_slots, occupancies, unscheduled = assign_macrocycle_start_slots(
        e,
        preferred_slots,
        allow_partial=True,
    )
    best = _apply_refill_pipeline(
        e,
        preferred_slots,
        (starts.copy(), int(macrocycle_slots), occupancies.copy(), list(unscheduled)),
    )
    best_unscheduled = list(best[3])
    retries_used = 0

    for _ in range(int(max_ble_channel_retries)):
        ble_unscheduled = np.array(
            [pair_id for pair_id in best_unscheduled if e.pair_radio_type[pair_id] == e.RADIO_BLE],
            dtype=int,
        )
        if ble_unscheduled.size == 0:
            break

        e.resample_ble_channels(ble_unscheduled)
        retries_used += 1
        candidate = assign_macrocycle_start_slots(
            e,
            preferred_slots,
            allow_partial=True,
        )
        candidate = _apply_refill_pipeline(
            e,
            preferred_slots,
            (candidate[0].copy(), int(candidate[1]), candidate[2].copy(), list(candidate[3])),
        )
        candidate_unscheduled = list(candidate[3])
        if _is_better_refill_result(candidate, best, e.pair_priority):
            best = (
                candidate[0].copy(),
                int(candidate[1]),
                candidate[2].copy(),
                candidate_unscheduled,
            )
            best_unscheduled = candidate_unscheduled
        if not best_unscheduled:
            break

    return (*best, retries_used)


def print_office_stats(rows):
    print("=== Per-Office WiFi/BLE Pair & Slot Statistics ===")
    print("office_id,wifi_pair_count,ble_pair_count,wifi_slots_used,ble_slots_used")
    for r in rows:
        print(
            f"{r['office_id']},{r['wifi_pair_count']},{r['ble_pair_count']},"
            f"{r['wifi_slots_used']},{r['ble_slots_used']}"
        )


def build_pair_parameter_rows(
    pair_office_id,
    pair_radio_type,
    pair_channel,
    pair_priority,
    pair_start_time_slot,
    pair_wifi_anchor_slot,
    pair_wifi_period_slots,
    pair_wifi_tx_slots,
    pair_ble_anchor_slot,
    pair_ble_ci_slots,
    pair_ble_ce_slots,
    pair_ble_ce_feasible,
    z_vec,
    occupied_slots,
    macrocycle_slots,
    slot_time,
    wifi_id,
    ble_id,
):
    rows = []
    for pair_id in range(pair_radio_type.shape[0]):
        is_ble = int(pair_radio_type[pair_id]) == int(ble_id)
        schedule_slot = int(z_vec[pair_id])
        rows.append(
            {
                "pair_id": int(pair_id),
                "office_id": int(pair_office_id[pair_id]),
                "radio": "ble" if is_ble else "wifi",
                "channel": int(pair_channel[pair_id]),
                "priority": float(pair_priority[pair_id]),
                "schedule_slot": schedule_slot,
                "schedule_time_ms": float(schedule_slot * slot_time * 1e3),
                "start_time_slot": int(pair_start_time_slot[pair_id]),
                "wifi_anchor_slot": int(pair_wifi_anchor_slot[pair_id]) if not is_ble else None,
                "wifi_period_slots": int(pair_wifi_period_slots[pair_id]) if not is_ble else None,
                "wifi_period_ms": float(pair_wifi_period_slots[pair_id] * slot_time * 1e3) if not is_ble else None,
                "wifi_tx_slots": int(pair_wifi_tx_slots[pair_id]) if not is_ble else None,
                "wifi_tx_ms": float(pair_wifi_tx_slots[pair_id] * slot_time * 1e3) if not is_ble else None,
                "ble_anchor_slot": int(pair_ble_anchor_slot[pair_id]) if is_ble else None,
                "ble_ci_slots": int(pair_ble_ci_slots[pair_id]) if is_ble else None,
                "ble_ci_ms": float(pair_ble_ci_slots[pair_id] * slot_time * 1e3) if is_ble else None,
                "ble_ce_slots": int(pair_ble_ce_slots[pair_id]) if is_ble else None,
                "ble_ce_ms": float(pair_ble_ce_slots[pair_id] * slot_time * 1e3) if is_ble else None,
                "ble_ce_feasible": bool(pair_ble_ce_feasible[pair_id]) if is_ble else None,
                "macrocycle_slots": int(macrocycle_slots),
                "occupied_slots_in_macrocycle": [int(s) for s in np.where(occupied_slots[pair_id])[0]],
            }
        )
    return rows


def compute_pair_parameter_rows(e: env, z_vec: np.ndarray, occupied_slots: np.ndarray, macrocycle_slots: int):
    return build_pair_parameter_rows(
        pair_office_id=e.pair_office_id,
        pair_radio_type=e.pair_radio_type,
        pair_channel=e.pair_channel,
        pair_priority=e.pair_priority,
        pair_start_time_slot=e.pair_start_time_slot,
        pair_wifi_anchor_slot=e.pair_wifi_anchor_slot,
        pair_wifi_period_slots=e.pair_wifi_period_slots,
        pair_wifi_tx_slots=e.pair_wifi_tx_slots,
        pair_ble_anchor_slot=e.pair_ble_anchor_slot,
        pair_ble_ci_slots=e.pair_ble_ci_slots,
        pair_ble_ce_slots=e.pair_ble_ce_slots,
        pair_ble_ce_feasible=e.pair_ble_ce_feasible,
        z_vec=z_vec,
        occupied_slots=occupied_slots,
        macrocycle_slots=macrocycle_slots,
        slot_time=e.slot_time,
        wifi_id=e.RADIO_WIFI,
        ble_id=e.RADIO_BLE,
    )


def resolve_macrocycle_schedule_status(schedule_start_slots: np.ndarray, occupied_slots: np.ndarray):
    schedule_start_slots = np.asarray(schedule_start_slots, dtype=int).ravel()
    occupied_slots = np.asarray(occupied_slots, dtype=bool)
    if occupied_slots.ndim != 2:
        raise ValueError("occupied_slots must be a 2D boolean array.")
    if occupied_slots.shape[0] != schedule_start_slots.shape[0]:
        raise ValueError("schedule_start_slots and occupied_slots must have the same pair dimension.")

    has_occupancy = np.any(occupied_slots, axis=1)
    is_scheduled = np.logical_and(schedule_start_slots >= 0, has_occupancy)
    scheduled_pair_ids = np.where(is_scheduled)[0].astype(int).tolist()
    unscheduled_pair_ids = np.where(~is_scheduled)[0].astype(int).tolist()
    return scheduled_pair_ids, unscheduled_pair_ids


def filter_pair_rows_by_ids(pair_rows, pair_ids):
    selected = set(int(pair_id) for pair_id in pair_ids)
    return [row for row in pair_rows if int(row["pair_id"]) in selected]


def build_schedule_rows(pair_rows):
    slot_map = {}
    for row in pair_rows:
        for slot in row["occupied_slots_in_macrocycle"]:
            slot_map.setdefault(int(slot), []).append(row)

    rows = []
    for slot in sorted(slot_map):
        grouped = sorted(slot_map[slot], key=lambda r: int(r["pair_id"]))
        wifi_ids = [int(r["pair_id"]) for r in grouped if r["radio"] == "wifi"]
        ble_ids = [int(r["pair_id"]) for r in grouped if r["radio"] == "ble"]
        rows.append(
            {
                "schedule_slot": slot,
                "pair_ids": [int(r["pair_id"]) for r in grouped],
                "wifi_pair_ids": wifi_ids,
                "ble_pair_ids": ble_ids,
                "pair_count": len(grouped),
                "wifi_pair_count": len(wifi_ids),
                "ble_pair_count": len(ble_ids),
            }
        )
    return rows


def get_pair_channel_ranges_mhz(e: env, pair_ids):
    pair_ids = np.asarray(pair_ids, dtype=int).ravel()
    ranges = {}
    for pair_id in pair_ids:
        low_hz, high_hz = e._get_pair_link_range_hz(int(pair_id))
        ranges[int(pair_id)] = (float(low_hz / 1e6), float(high_hz / 1e6))
    return ranges


def build_schedule_plot_rows(pair_rows, pair_channel_ranges):
    rows = []
    for row in pair_rows:
        pair_id = int(row["pair_id"])
        low_mhz, high_mhz = pair_channel_ranges[pair_id]
        radio_tag = "W" if row["radio"] == "wifi" else "B"
        label = f"{pair_id} {radio_tag}-ch{int(row['channel'])}"
        for slot in row["occupied_slots_in_macrocycle"]:
            rows.append(
                {
                    "pair_id": pair_id,
                    "radio": row["radio"],
                    "channel": int(row["channel"]),
                    "slot": int(slot),
                    "freq_low_mhz": float(low_mhz),
                    "freq_high_mhz": float(high_mhz),
                    "label": label,
                }
            )
    return rows


def render_schedule_plot(plot_rows, output_path, macrocycle_slots):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"wifi": "#0B6E4F", "ble": "#C84C09"}
    for row in plot_rows:
        height = row["freq_high_mhz"] - row["freq_low_mhz"]
        rect = Rectangle(
            (row["slot"], row["freq_low_mhz"]),
            1.0,
            height,
            facecolor=colors.get(row["radio"], "#4C4C4C"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.65,
        )
        ax.add_patch(rect)
        if height >= 1.5:
            ax.text(
                row["slot"] + 0.5,
                row["freq_low_mhz"] + height / 2.0,
                row["label"],
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                clip_on=True,
            )
    ax.set_xlim(0, int(macrocycle_slots))
    if plot_rows:
        ax.set_ylim(min(r["freq_low_mhz"] for r in plot_rows), max(r["freq_high_mhz"] for r in plot_rows))
    ax.set_xlabel("Slot")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("WiFi/BLE Schedule")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(
        handles=[
            Patch(facecolor=colors["wifi"], edgecolor="black", alpha=0.65, label="WiFi"),
            Patch(facecolor=colors["ble"], edgecolor="black", alpha=0.65, label="BLE"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt_cell(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def print_pair_parameter_rows(rows):
    print("=== Pair Parameter Table ===")
    print(
        "pair_id,office_id,radio,channel,priority,schedule_slot,schedule_time_ms,"
        "start_time_slot,"
        "wifi_anchor_slot,wifi_period_slots,wifi_period_ms,wifi_tx_slots,wifi_tx_ms,"
        "ble_anchor_slot,ble_ci_slots,ble_ci_ms,ble_ce_slots,ble_ce_ms,ble_ce_feasible,"
        "macrocycle_slots,occupied_slots_in_macrocycle"
    )
    for row in rows:
        print(
            ",".join(
                [
                    _fmt_cell(row["pair_id"]),
                    _fmt_cell(row["office_id"]),
                    _fmt_cell(row["radio"]),
                    _fmt_cell(row["channel"]),
                    _fmt_cell(row["priority"]),
                    _fmt_cell(row["schedule_slot"]),
                    _fmt_cell(row["schedule_time_ms"]),
                    _fmt_cell(row["start_time_slot"]),
                    _fmt_cell(row["wifi_anchor_slot"]),
                    _fmt_cell(row["wifi_period_slots"]),
                    _fmt_cell(row["wifi_period_ms"]),
                    _fmt_cell(row["wifi_tx_slots"]),
                    _fmt_cell(row["wifi_tx_ms"]),
                    _fmt_cell(row["ble_anchor_slot"]),
                    _fmt_cell(row["ble_ci_slots"]),
                    _fmt_cell(row["ble_ci_ms"]),
                    _fmt_cell(row["ble_ce_slots"]),
                    _fmt_cell(row["ble_ce_ms"]),
                    _fmt_cell(row["ble_ce_feasible"]),
                    _fmt_cell(row["macrocycle_slots"]),
                    _fmt_cell(row["occupied_slots_in_macrocycle"]),
                ]
            )
        )


def print_schedule_rows(rows):
    print("=== Schedule Table ===")
    print("schedule_slot,pair_ids,wifi_pair_ids,ble_pair_ids,pair_count,wifi_pair_count,ble_pair_count")
    for row in rows:
        print(
            ",".join(
                [
                    _fmt_cell(row["schedule_slot"]),
                    _fmt_cell(row["pair_ids"]),
                    _fmt_cell(row["wifi_pair_ids"]),
                    _fmt_cell(row["ble_pair_ids"]),
                    _fmt_cell(row["pair_count"]),
                    _fmt_cell(row["wifi_pair_count"]),
                    _fmt_cell(row["ble_pair_count"]),
                ]
            )
        )


def write_rows_to_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _fmt_cell(row.get(key))
                    for key in fieldnames
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="pd_mmw_template + office/pair statistics")
    parser.add_argument("--cell-size", type=int, default=2, help="办公室网格边长（办公室数量 = cell_size^2）")
    parser.add_argument(
        "--pair-density",
        "--sta-density",
        dest="pair_density",
        type=float,
        default=0.05,
        help="通信对密度（每平方米）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子；不填则使用当前时间")
    parser.add_argument("--mmw-nit", type=int, default=200, help="MMW 迭代次数")
    parser.add_argument("--mmw-eta", type=float, default=0.05, help="MMW 步长 eta")
    parser.add_argument("--use-gpu", action="store_true", help="启用 GPU 设备选择（若 CUDA 可用）")
    parser.add_argument("--gpu-id", type=int, default=0, help="指定使用的 GPU 编号")
    parser.add_argument("--max-slots", type=int, default=300, help="最大允许调度时隙数；超过后输出当前成功调度结果")
    parser.add_argument("--ble-channel-retries", type=int, default=0, help="对未调度 BLE pair 重新选信道并重试宏周期排布的次数")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sim_script/output",
        help="CSV 输出目录（会写出 pair_parameters.csv 和 wifi_ble_schedule.csv）",
    )
    args = parser.parse_args()
    if args.max_slots < 2:
        parser.error("--max-slots must be at least 2.")
    return args


if __name__ == "__main__":
    args = parse_args()
    np.set_printoptions(threshold=10)
    np.set_printoptions(linewidth=1000)
    runtime_device = resolve_torch_device(args.use_gpu, args.gpu_id)

    e = env(
        cell_edge=7.0,
        # cell_size=args.cell_size,
        cell_size=1,
        # pair_density_per_m2=args.pair_density,
        pair_density_per_m2=0.5,
        seed=int(time.time()) if args.seed is None else args.seed,
        radio_prob=(0.3, 0.7),
        slot_time=1.25e-3,
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=7.5e-3,
        ble_payload_bits=800,
        ble_phy_rate_bps=2e6,
        ble_ci_exp_min=3,
        # ble_ci_exp_max=11,
        ble_ci_exp_max=6,
    )

    print("n_pair =", e.n_pair)
    print("n_device =", 2 * e.n_pair)
    print("n_office =", e.n_office)
    print("office_area_m2 =", e.office_area_m2)
    print("pair_density_per_m2 =", e.pair_density_per_m2)
    print("runtime_device =", runtime_device)
    print("wifi_period_quanta_candidates:", e.wifi_period_quanta_candidates.tolist())
    print("ble_ci_quanta_candidates:", e.ble_ci_quanta_candidates.tolist())
    print("n_wifi_pair =", int(np.sum(e.pair_radio_type == e.RADIO_WIFI)))
    print("n_ble_pair  =", int(np.sum(e.pair_radio_type == e.RADIO_BLE)))

    bs = binary_search_relaxation()
    bs.user_priority = e.pair_priority
    bs.slot_mask_builder = lambda Z, state, ee=e: ee.build_slot_compatibility_mask(Z)
    bs.force_lower_bound = False
    # bs.max_slot_cap = args.max_slots
    bs.max_slot_cap = 1000  # 不强制限制最大时隙数，改为在结果分析阶段判断是否超过 args.max_slots

    alg = mmw(nit=args.mmw_nit, eta=args.mmw_eta, device=runtime_device)
    alg.DEBUG = False
    alg.LOG_GAP = False
    bs.feasibility_check_alg = alg

    z_vec_pref, Z_fin_mmw, remainder = bs.run(e.generate_S_Q_hmax())
    partial = bs.last_partial_schedule or {"slot_cap_hit": False, "scheduled_pair_ids": list(range(e.n_pair)), "unscheduled_pair_ids": []}
    if partial["unscheduled_pair_ids"]:
        avg_bler = float("nan")
        max_bler = float("nan")
        weighted_bler = float("nan")
    else:
        bler_arr = e.evaluate_bler(z_vec_pref, Z_fin_mmw)
        avg_bler = float(np.mean(bler_arr))
        max_bler = float(np.max(bler_arr))
        weighted_bler = float(e.evaluate_weighted_bler(z_vec_pref, Z_fin_mmw))
    print("MMW result:", Z_fin_mmw, remainder, avg_bler, max_bler, weighted_bler)
    z_vec = np.asarray(z_vec_pref, dtype=int)
    schedule_start_slots, macrocycle_slots, occupancy, macro_unscheduled, ble_channel_retries_used = retry_ble_channels_and_assign_macrocycle(
        e,
        z_vec,
        max_ble_channel_retries=args.ble_channel_retries,
    )
    scheduled_pair_ids, unscheduled_pair_ids = resolve_macrocycle_schedule_status(schedule_start_slots, occupancy)
    partial_schedule = bool(unscheduled_pair_ids)
    print("macrocycle_slots =", macrocycle_slots)
    print("macrocycle_ms =", float(macrocycle_slots * e.slot_time * 1e3))
    print("partial_schedule =", partial_schedule)
    print("scheduled_pair_ids =", scheduled_pair_ids)
    print("unscheduled_pair_ids =", unscheduled_pair_ids)
    print("ble_channel_retries_used =", ble_channel_retries_used)

    ble_idx = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    if ble_idx.size:
        infeasible = int(np.sum(~e.pair_ble_ce_feasible[ble_idx]))
        print(
            "BLE timing summary:",
            {
                "n_ble_pair": int(ble_idx.size),
                "ci_slots_min": int(np.min(e.pair_ble_ci_slots[ble_idx])),
                "ci_slots_avg": float(np.mean(e.pair_ble_ci_slots[ble_idx])),
                "ci_slots_max": int(np.max(e.pair_ble_ci_slots[ble_idx])),
                "ce_slots_min": int(np.min(e.pair_ble_ce_slots[ble_idx])),
                "ce_slots_avg": float(np.mean(e.pair_ble_ce_slots[ble_idx])),
                "ce_slots_max": int(np.max(e.pair_ble_ce_slots[ble_idx])),
                "infeasible_ble_pair": infeasible,
            },
        )

    pair_rows = compute_pair_parameter_rows(e, schedule_start_slots, occupancy, macrocycle_slots)
    print_pair_parameter_rows(pair_rows)
    write_rows_to_csv(
        os.path.join(args.output_dir, "pair_parameters.csv"),
        [
            "pair_id",
            "office_id",
            "radio",
            "channel",
            "priority",
            "schedule_slot",
            "schedule_time_ms",
            "start_time_slot",
            "wifi_anchor_slot",
            "wifi_period_slots",
            "wifi_period_ms",
            "wifi_tx_slots",
            "wifi_tx_ms",
            "ble_anchor_slot",
            "ble_ci_slots",
            "ble_ci_ms",
            "ble_ce_slots",
            "ble_ce_ms",
            "ble_ce_feasible",
            "macrocycle_slots",
            "occupied_slots_in_macrocycle",
        ],
        pair_rows,
    )

    scheduled_pair_rows = filter_pair_rows_by_ids(pair_rows, scheduled_pair_ids)
    unscheduled_pair_rows = filter_pair_rows_by_ids(pair_rows, unscheduled_pair_ids)
    schedule_rows = build_schedule_rows(scheduled_pair_rows)
    print_schedule_rows(schedule_rows)
    write_rows_to_csv(
        os.path.join(args.output_dir, "wifi_ble_schedule.csv"),
        [
            "schedule_slot",
            "pair_ids",
            "wifi_pair_ids",
            "ble_pair_ids",
            "pair_count",
            "wifi_pair_count",
            "ble_pair_count",
        ],
        schedule_rows,
    )
    write_rows_to_csv(
        os.path.join(args.output_dir, "unscheduled_pairs.csv"),
        [
            "pair_id",
            "office_id",
            "radio",
            "channel",
            "priority",
            "schedule_slot",
            "schedule_time_ms",
            "start_time_slot",
            "wifi_anchor_slot",
            "wifi_period_slots",
            "wifi_period_ms",
            "wifi_tx_slots",
            "wifi_tx_ms",
            "ble_anchor_slot",
            "ble_ci_slots",
            "ble_ci_ms",
            "ble_ce_slots",
            "ble_ce_ms",
            "ble_ce_feasible",
            "macrocycle_slots",
            "occupied_slots_in_macrocycle",
        ],
        unscheduled_pair_rows,
    )
    pair_channel_ranges = get_pair_channel_ranges_mhz(e, scheduled_pair_ids)
    schedule_plot_rows = build_schedule_plot_rows(scheduled_pair_rows, pair_channel_ranges)
    write_rows_to_csv(
        os.path.join(args.output_dir, "schedule_plot_rows.csv"),
        [
            "pair_id",
            "radio",
            "channel",
            "slot",
            "freq_low_mhz",
            "freq_high_mhz",
            "label",
        ],
        schedule_plot_rows,
    )
    schedule_plot_path = os.path.join(args.output_dir, "wifi_ble_schedule.png")
    render_schedule_plot(schedule_plot_rows, schedule_plot_path, macrocycle_slots=macrocycle_slots)
    print(
        "CSV outputs:",
        os.path.join(args.output_dir, "pair_parameters.csv"),
        os.path.join(args.output_dir, "wifi_ble_schedule.csv"),
    )
    print("Schedule plot:", schedule_plot_path)

    office_rows = compute_office_pair_slot_stats_for_pair_ids(e, schedule_start_slots, scheduled_pair_ids)
    print_office_stats(office_rows)
