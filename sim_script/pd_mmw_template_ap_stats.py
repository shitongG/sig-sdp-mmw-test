import argparse
import time

import numpy as np

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.mmw import mmw
from sim_src.env.env import env


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


def print_office_stats(rows):
    print("=== Per-Office WiFi/BLE Pair & Slot Statistics ===")
    print("office_id,wifi_pair_count,ble_pair_count,wifi_slots_used,ble_slots_used")
    for r in rows:
        print(
            f"{r['office_id']},{r['wifi_pair_count']},{r['ble_pair_count']},"
            f"{r['wifi_slots_used']},{r['ble_slots_used']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="pd_mmw_template + office/pair statistics")
    parser.add_argument("--cell-size", type=int, default=10, help="办公室网格边长（办公室数量 = cell_size^2）")
    parser.add_argument(
        "--pair-density",
        "--sta-density",
        dest="pair_density",
        type=float,
        default=0.5,
        help="通信对密度（每平方米）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子；不填则使用当前时间")
    parser.add_argument("--mmw-nit", type=int, default=200, help="MMW 迭代次数")
    parser.add_argument("--mmw-eta", type=float, default=0.05, help="MMW 步长 eta")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.set_printoptions(threshold=10)
    np.set_printoptions(linewidth=1000)

    e = env(
        cell_edge=7.0,
        cell_size=args.cell_size,
        pair_density_per_m2=args.pair_density,
        seed=int(time.time()) if args.seed is None else args.seed,
        radio_prob=(0.6, 0.4),
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=15e-3,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=2.5e-3,
        ble_payload_bits=800,
        ble_phy_rate_bps=1e6,
    )

    print("n_pair =", e.n_pair)
    print("n_device =", 2 * e.n_pair)
    print("n_office =", e.n_office)
    print("office_area_m2 =", e.office_area_m2)
    print("pair_density_per_m2 =", e.pair_density_per_m2)
    print("ble_ci_quanta_candidates:", e.ble_ci_quanta_candidates.tolist())
    print("n_wifi_pair =", int(np.sum(e.pair_radio_type == e.RADIO_WIFI)))
    print("n_ble_pair  =", int(np.sum(e.pair_radio_type == e.RADIO_BLE)))

    bs = binary_search_relaxation()
    bs.user_priority = e.pair_priority
    bs.slot_mask_builder = lambda Z, state, ee=e: ee.build_slot_compatibility_mask(Z)
    bs.force_lower_bound = False

    alg = mmw(nit=args.mmw_nit, eta=args.mmw_eta)
    alg.DEBUG = False
    alg.LOG_GAP = True
    bs.feasibility_check_alg = alg

    z_vec, Z_fin_mmw, remainder = bs.run(e.generate_S_Q_hmax())
    bler_arr = e.evaluate_bler(z_vec, Z_fin_mmw)
    avg_bler = float(np.mean(bler_arr))
    max_bler = float(np.max(bler_arr))
    weighted_bler = float(e.evaluate_weighted_bler(z_vec, Z_fin_mmw))
    print("MMW result:", Z_fin_mmw, remainder, avg_bler, max_bler, weighted_bler)

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

    office_rows = compute_office_pair_slot_stats(e, z_vec)
    print_office_stats(office_rows)
