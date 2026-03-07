import argparse
import time

import numpy as np

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.mmw import mmw
from sim_src.env.env import env


def _aggregate_ap_stats_from_arrays(
    asso: np.ndarray,
    radio: np.ndarray,
    z_vec: np.ndarray,
    n_ap: int,
    wifi_id: int,
    ble_id: int,
):
    """按 AP 聚合用户数和时隙占用统计。"""
    rows = []
    for ap in range(n_ap):
        ap_users = np.where(asso == ap)[0]
        wifi_users = ap_users[radio[ap_users] == wifi_id]
        ble_users = ap_users[radio[ap_users] == ble_id]
        rows.append(
            {
                "ap_id": int(ap),
                "wifi_user_count": int(wifi_users.size),
                "ble_user_count": int(ble_users.size),
                "wifi_slots_used": int(np.unique(z_vec[wifi_users]).size) if wifi_users.size else 0,
                "ble_slots_used": int(np.unique(z_vec[ble_users]).size) if ble_users.size else 0,
            }
        )
    return rows


def compute_ap_user_slot_stats(e: env, z_vec: np.ndarray):
    # 按当前接收功率最大 AP 做用户关联
    rxpr = e._compute_state_real().toarray()
    asso = np.argmax(rxpr, axis=1)
    return _aggregate_ap_stats_from_arrays(
        asso=asso,
        radio=e.user_radio_type,
        z_vec=z_vec,
        n_ap=e.n_ap,
        wifi_id=e.RADIO_WIFI,
        ble_id=e.RADIO_BLE,
    )


def print_ap_stats(rows):
    print("=== Per-AP WiFi/BLE User & Slot Statistics ===")
    print("ap_id,wifi_user_count,ble_user_count,wifi_slots_used,ble_slots_used")
    for r in rows:
        print(
            f"{r['ap_id']},{r['wifi_user_count']},{r['ble_user_count']},"
            f"{r['wifi_slots_used']},{r['ble_slots_used']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="pd_mmw_template + AP 统计输出")
    parser.add_argument("--cell-size", type=int, default=10, help="蜂窝边长（AP 数量 = cell_size^2）")
    parser.add_argument("--sta-density", type=float, default=1e-2, help="用户密度（每平方米）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子；不填则使用当前时间")
    parser.add_argument("--mmw-nit", type=int, default=200, help="MMW 迭代次数")
    parser.add_argument("--mmw-eta", type=float, default=0.05, help="MMW 步长 eta")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.set_printoptions(threshold=10)
    np.set_printoptions(linewidth=1000)

    # 实验配置：复用 pd_mmw_template 主流程，并加 BLE 参数
    e = env(
        cell_size=args.cell_size,
        sta_density_per_1m2=args.sta_density,
        seed=int(time.time()) if args.seed is None else args.seed,
        radio_prob=(0.6, 0.4),
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=15e-3,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=2.5e-3,
        ble_payload_bits=800,
        ble_phy_rate_bps=1e6,
    )

    print("n_sta =", e.n_sta)
    print("n_ap  =", e.n_ap)
    print("ble_ci_quanta_candidates:", e.ble_ci_quanta_candidates.tolist())

    bs = binary_search_relaxation()
    bs.user_priority = e.user_priority
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

    # 输出 BLE 随机 CI/CE 的摘要
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    if ble_idx.size:
        infeasible = int(np.sum(~e.user_ble_ce_feasible[ble_idx]))
        print(
            "BLE timing summary:",
            {
                "n_ble_users": int(ble_idx.size),
                "ci_slots_min": int(np.min(e.user_ble_ci_slots[ble_idx])),
                "ci_slots_avg": float(np.mean(e.user_ble_ci_slots[ble_idx])),
                "ci_slots_max": int(np.max(e.user_ble_ci_slots[ble_idx])),
                "ce_slots_min": int(np.min(e.user_ble_ce_slots[ble_idx])),
                "ce_slots_avg": float(np.mean(e.user_ble_ce_slots[ble_idx])),
                "ce_slots_max": int(np.max(e.user_ble_ce_slots[ble_idx])),
                "infeasible_ble_users": infeasible,
            },
        )

    # 输出“同 AP 下 WiFi/BLE 用户数 + 各自时隙占用数”
    ap_rows = compute_ap_user_slot_stats(e, z_vec)
    print_ap_stats(ap_rows)
