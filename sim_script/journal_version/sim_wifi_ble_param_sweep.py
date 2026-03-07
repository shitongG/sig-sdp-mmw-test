import argparse
import csv
import itertools
import os
import time
from datetime import datetime

import numpy as np

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.mmw import mmw
from sim_src.env.env import env

# ===================== 可调参数编号总表（中文） =====================
# [参数01] cell_sizes: 小区边长网格规模（AP 数量约等于 cell_size^2）
# [参数02] rho_list: 用户空间密度（sta_density_per_1m2）
# [参数03] seeds: 随机种子范围/列表
# [参数04] nit: MMW 迭代次数
# [参数05] eta: MMW 学习率
# [参数06] enable_slot_mask: 是否启用 BLE 的 CI/CE 时隙掩码约束
#
# [参数07] wifi_ratio_list: WiFi 用户比例（BLE 比例 = 1 - WiFi 比例）
# [参数08] wifi_channel_count: WiFi 可用信道数
# [参数09] wifi_bw_mhz_list: WiFi 单信道带宽（MHz）
# [参数10] wifi_reuse_channels: AP 固定复用的 WiFi 信道索引集合（如 0,5,10 对应 1/6/11）
#
# [参数11] ble_channel_count: BLE 可用信道数
# [参数12] ble_bw_mhz_list: BLE 单信道带宽（MHz）
# [参数13] ble_ci_min_ms_list: BLE CI 下界（ms），必须是 1.25ms 整数倍
# [参数14] ble_ci_max_ms_list: BLE CI 上界（ms），必须是 1.25ms 整数倍
# [参数15] ble_ci_fixed: 是否固定所有 BLE 用户使用同一个 CI（用于 A/B 对比）
# [参数16] ble_ce_min_ms_list: BLE CE 下界（ms，默认 >= 1.25ms）
# [参数17] ble_ce_max_ms_list: BLE CE 上界（ms）
# [参数18] ble_payload_bits_list: BLE 每次连接事件需承载的 payload（bit）
# [参数19] ble_phy_rate_mbps_list: BLE 物理层速率（Mbps）
#
# [参数20] output: 输出 CSV 路径
# ===============================================================

def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_seed_spec(text):
    s = text.strip()
    if ":" in s:
        a, b = s.split(":")
        start = int(a.strip())
        end = int(b.strip())
        return list(range(start, end + 1))
    return parse_int_list(s)


def now_tag():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_output_path(user_output):
    if user_output:
        return user_output
    base = os.path.join("sim_script", "journal_version", "sweep_output")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"wifi_ble_param_sweep-{now_tag()}.csv")


def run_one_case(cfg, seed, nit, eta, enable_slot_mask):
    # 中文说明：这里把当前组合参数实例化到 env，然后跑一次 “二分 + MMW + 评估”
    wifi_ratio = cfg["wifi_ratio"]
    ble_ratio = 1.0 - wifi_ratio
    ble_ci_max_ms = cfg["ble_ci_min_ms"] if cfg["ble_ci_fixed"] else cfg["ble_ci_max_ms"]
    e = env(
        cell_size=cfg["cell_size"],
        sta_density_per_1m2=cfg["rho"],
        seed=seed,
        radio_prob=(wifi_ratio, ble_ratio),
        wifi_channel_count=cfg["wifi_channel_count"],
        wifi_channel_bandwidth_hz=cfg["wifi_bw_mhz"] * 1e6,
        ble_channel_count=cfg["ble_channel_count"],
        ble_channel_bandwidth_hz=cfg["ble_bw_mhz"] * 1e6,
        wifi_reuse_channels=tuple(cfg["wifi_reuse_channels"]),
        ble_ci_min_s=cfg["ble_ci_min_ms"] * 1e-3,
        ble_ci_max_s=ble_ci_max_ms * 1e-3,
        ble_ce_max_s=cfg["ble_ce_max_ms"] * 1e-3,
        ble_ce_min_s=cfg["ble_ce_min_ms"] * 1e-3,
        ble_payload_bits=cfg["ble_payload_bits"],
        ble_phy_rate_bps=cfg["ble_phy_rate_mbps"] * 1e6,
    )

    bs = binary_search_relaxation()
    bs.user_priority = e.user_priority
    if enable_slot_mask:
        bs.slot_mask_builder = lambda Z, state, ee=e: ee.build_slot_compatibility_mask(Z)

    alg = mmw(nit=nit, eta=eta)
    bs.feasibility_check_alg = alg

    tic = time.perf_counter()
    z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
    runtime_s = time.perf_counter() - tic

    bler = e.evaluate_bler(z_vec, Z_fin)
    weighted_bler = e.evaluate_weighted_bler(z_vec, Z_fin)
    radio_stats = e.get_radio_conflict_stats()
    slot_mask = e.build_slot_compatibility_mask(int(Z_fin))
    ble_idx = np.where(e.user_radio_type == e.RADIO_BLE)[0]
    ble_slot_allowed_ratio = float(np.mean(slot_mask[ble_idx])) if ble_idx.size > 0 else 1.0

    if ble_idx.size > 0:
        ble_ci_slots_avg = float(np.mean(e.user_ble_ci_slots[ble_idx]))
        ble_ci_slots_min = int(np.min(e.user_ble_ci_slots[ble_idx]))
        ble_ci_slots_max = int(np.max(e.user_ble_ci_slots[ble_idx]))
        ble_ce_slots_avg = float(np.mean(e.user_ble_ce_slots[ble_idx]))
        ble_ce_slots_min = int(np.min(e.user_ble_ce_slots[ble_idx]))
        ble_ce_slots_max = int(np.max(e.user_ble_ce_slots[ble_idx]))
        ble_ce_required_avg = float(np.mean(e.user_ble_ce_required_s[ble_idx]))
        ble_infeasible_users = int(np.sum(~e.user_ble_ce_feasible[ble_idx]))
    else:
        ble_ci_slots_avg = 0.0
        ble_ci_slots_min = 0
        ble_ci_slots_max = 0
        ble_ce_slots_avg = 0.0
        ble_ce_slots_min = 0
        ble_ce_slots_max = 0
        ble_ce_required_avg = 0.0
        ble_infeasible_users = 0

    return {
        "status": "ok",
        "seed": seed,
        "runtime_s": runtime_s,
        "Z_fin": int(Z_fin),
        "remainder": int(remainder),
        "avg_bler": float(np.mean(bler)),
        "max_bler": float(np.max(bler)),
        "weighted_bler": float(weighted_bler),
        "ble_ce_required_avg_s": ble_ce_required_avg,
        "ble_ce_max_s": float(e.ble_ce_max_s),
        "ble_infeasible_users": ble_infeasible_users,
        "ble_ci_slots_avg": ble_ci_slots_avg,
        "ble_ci_slots_min": ble_ci_slots_min,
        "ble_ci_slots_max": ble_ci_slots_max,
        "ble_ce_slots_avg": ble_ce_slots_avg,
        "ble_ce_slots_min": ble_ce_slots_min,
        "ble_ce_slots_max": ble_ce_slots_max,
        "ble_slot_allowed_ratio": ble_slot_allowed_ratio,
        **radio_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="WiFi/BLE 参数扫频仿真脚本（支持 CI/CE/QoS 与时隙掩码）"
    )
    parser.add_argument("--cell-sizes", default="5", help="[参数01] 小区网格规模，逗号分隔，如 5,8,10")
    parser.add_argument("--rho-list", default="0.0075", help="[参数02] 用户密度列表，逗号分隔")
    parser.add_argument("--seeds", default="0:2", help="[参数03] 种子范围 a:b 或种子列表 1,3,5")
    parser.add_argument("--nit", type=int, default=40, help="[参数04] MMW 迭代次数")
    parser.add_argument("--eta", type=float, default=0.04, help="[参数05] MMW eta")
    parser.add_argument("--enable-slot-mask", action="store_true", help="[参数06] 启用 BLE CI/CE 时隙掩码约束")

    parser.add_argument("--wifi-ratio-list", default="0.6", help="[参数07] WiFi 用户比例列表（范围 0~1）")
    parser.add_argument("--wifi-channel-count", type=int, default=13, help="[参数08] WiFi 信道数")
    parser.add_argument("--wifi-bw-mhz-list", default="20", help="[参数09] WiFi 信道带宽（MHz）列表")
    parser.add_argument("--wifi-reuse-channels", default="0,5,10", help="[参数10] AP 固定复用 WiFi 信道索引")

    parser.add_argument("--ble-channel-count", type=int, default=37, help="[参数11] BLE 信道数")
    parser.add_argument("--ble-bw-mhz-list", default="2", help="[参数12] BLE 信道带宽（MHz）列表")
    parser.add_argument("--ble-ci-min-ms-list", default="7.5", help="[参数13] BLE CI 下界（ms）列表")
    parser.add_argument("--ble-ci-max-ms-list", default="7.5", help="[参数14] BLE CI 上界（ms）列表")
    parser.add_argument("--ble-ci-fixed", action="store_true", help="[参数15] 固定所有 BLE 用户使用同一个 CI（用 CI 下界值）")
    parser.add_argument("--ble-ce-min-ms-list", default="1.25", help="[参数16] BLE CE 下界（ms）列表")
    parser.add_argument("--ble-ce-max-ms-list", default="1.25", help="[参数17] BLE CE 上界（ms）列表")
    parser.add_argument("--ble-payload-bits-list", default="800", help="[参数18] BLE payload（bit）列表")
    parser.add_argument("--ble-phy-rate-mbps-list", default="1", help="[参数19] BLE PHY 速率（Mbps）列表")

    parser.add_argument("--output", default="", help="[参数20] 输出 CSV 路径")
    args = parser.parse_args()

    cell_sizes = parse_int_list(args.cell_sizes)
    rho_list = parse_float_list(args.rho_list)
    seeds = parse_seed_spec(args.seeds)
    wifi_ratios = parse_float_list(args.wifi_ratio_list)
    wifi_bw_mhz_list = parse_float_list(args.wifi_bw_mhz_list)
    ble_bw_mhz_list = parse_float_list(args.ble_bw_mhz_list)
    ble_ci_min_ms_list = parse_float_list(args.ble_ci_min_ms_list)
    ble_ci_max_ms_list = parse_float_list(args.ble_ci_max_ms_list)
    ble_ce_min_ms_list = parse_float_list(args.ble_ce_min_ms_list)
    ble_ce_max_ms_list = parse_float_list(args.ble_ce_max_ms_list)
    ble_payload_bits_list = parse_float_list(args.ble_payload_bits_list)
    ble_phy_rate_mbps_list = parse_float_list(args.ble_phy_rate_mbps_list)
    wifi_reuse_channels = parse_int_list(args.wifi_reuse_channels)

    out_path = make_output_path(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    grid = list(
        itertools.product(
            cell_sizes,
            rho_list,
            wifi_ratios,
            wifi_bw_mhz_list,
            ble_bw_mhz_list,
            ble_ci_min_ms_list,
            ble_ci_max_ms_list,
            ble_ce_min_ms_list,
            ble_ce_max_ms_list,
            ble_payload_bits_list,
            ble_phy_rate_mbps_list,
        )
    )
    total_runs = len(grid) * len(seeds)
    print(f"[INFO] cases={len(grid)}, seeds={len(seeds)}, total_runs={total_runs}")
    print(f"[INFO] output={out_path}")

    header = [
        "status",
        "error",
        "seed",
        "runtime_s",
        "cell_size",
        "rho",
        "wifi_ratio",
        "wifi_channel_count",
        "wifi_bw_mhz",
        "wifi_reuse_channels",
        "ble_channel_count",
        "ble_bw_mhz",
        "ble_ci_min_ms",
        "ble_ci_max_ms",
        "ble_ci_fixed",
        "ble_ce_min_ms",
        "ble_ce_max_ms",
        "ble_payload_bits",
        "ble_phy_rate_mbps",
        "enable_slot_mask",
        "nit",
        "eta",
        "Z_fin",
        "remainder",
        "avg_bler",
        "max_bler",
        "weighted_bler",
        "ble_ce_required_avg_s",
        "ble_ce_max_s",
        "ble_infeasible_users",
        "ble_ci_slots_avg",
        "ble_ci_slots_min",
        "ble_ci_slots_max",
        "ble_ce_slots_avg",
        "ble_ce_slots_min",
        "ble_ce_slots_max",
        "ble_slot_allowed_ratio",
        "n_user",
        "n_wifi_user",
        "n_ble_user",
        "wifi_wifi_edges",
        "wifi_ble_edges",
        "ble_ble_edges",
        "total_radio_conflict_edges",
    ]

    done = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for (
            cell_size,
            rho,
            wifi_ratio,
            wifi_bw_mhz,
            ble_bw_mhz,
            ble_ci_min_ms,
            ble_ci_max_ms,
            ble_ce_min_ms,
            ble_ce_max_ms,
            ble_payload_bits,
            ble_phy_rate_mbps,
        ) in grid:
            cfg = {
                "cell_size": int(cell_size),
                "rho": float(rho),
                "wifi_ratio": float(wifi_ratio),
                "wifi_channel_count": int(args.wifi_channel_count),
                "wifi_bw_mhz": float(wifi_bw_mhz),
                "wifi_reuse_channels": wifi_reuse_channels,
                "ble_channel_count": int(args.ble_channel_count),
                "ble_bw_mhz": float(ble_bw_mhz),
                "ble_ci_min_ms": float(ble_ci_min_ms),
                "ble_ci_max_ms": float(ble_ci_max_ms),
                "ble_ci_fixed": bool(args.ble_ci_fixed),
                "ble_ce_min_ms": float(ble_ce_min_ms),
                "ble_ce_max_ms": float(ble_ce_max_ms),
                "ble_payload_bits": float(ble_payload_bits),
                "ble_phy_rate_mbps": float(ble_phy_rate_mbps),
            }
            for seed in seeds:
                done += 1
                base = {
                    "status": "error",
                    "error": "",
                    "seed": int(seed),
                    "cell_size": cfg["cell_size"],
                    "rho": cfg["rho"],
                    "wifi_ratio": cfg["wifi_ratio"],
                    "wifi_channel_count": cfg["wifi_channel_count"],
                    "wifi_bw_mhz": cfg["wifi_bw_mhz"],
                    "wifi_reuse_channels": "-".join(str(x) for x in cfg["wifi_reuse_channels"]),
                    "ble_channel_count": cfg["ble_channel_count"],
                    "ble_bw_mhz": cfg["ble_bw_mhz"],
                    "ble_ci_min_ms": cfg["ble_ci_min_ms"],
                    "ble_ci_max_ms": cfg["ble_ci_max_ms"],
                    "ble_ci_fixed": cfg["ble_ci_fixed"],
                    "ble_ce_min_ms": cfg["ble_ce_min_ms"],
                    "ble_ce_max_ms": cfg["ble_ce_max_ms"],
                    "ble_payload_bits": cfg["ble_payload_bits"],
                    "ble_phy_rate_mbps": cfg["ble_phy_rate_mbps"],
                    "enable_slot_mask": bool(args.enable_slot_mask),
                    "nit": int(args.nit),
                    "eta": float(args.eta),
                }
                try:
                    row = run_one_case(
                        cfg=cfg,
                        seed=int(seed),
                        nit=int(args.nit),
                        eta=float(args.eta),
                        enable_slot_mask=bool(args.enable_slot_mask),
                    )
                    base.update(row)
                    base["error"] = ""
                except Exception as ex:
                    base["error"] = str(ex)
                w.writerow(base)
                print(
                    f"[{done}/{total_runs}] seed={seed}, cell={cfg['cell_size']}, "
                    f"wifi_ratio={cfg['wifi_ratio']}, ci_min_ms={cfg['ble_ci_min_ms']}, status={base['status']}"
                )

    print(f"[DONE] saved={out_path}")


if __name__ == "__main__":
    main()
