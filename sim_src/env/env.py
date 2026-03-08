import math

import numpy as np
import scipy


class env:
    C = 299792458.0
    PI = 3.14159265358979323846
    HIDDEN_LOSS = 200.0
    NOISE_FLOOR_DBM = -94.0
    BOLTZMANN = 1.3803e-23
    NOISEFIGURE = 13

    def __init__(
        self,
        cell_edge=20.0,
        cell_size=20,
        pair_density_per_m2=5e-3,
        sta_density_per_1m2=None,
        fre_Hz=4e9,
        txp_dbm_hi=5.0,
        txp_offset=2.0,
        min_s_n_ratio=0.1,
        packet_bit=800,
        bandwidth=5e6,
        slot_time=1.25e-4,
        max_err=1e-5,
        seed=1,
        prio_prob=(0.2, 0.3, 0.5),
        prio_value=(3.0, 2.0, 1.0),
        radio_prob=(0.5, 0.5),
        wifi_channel_count=13,
        wifi_channel_bandwidth_hz=20e6,
        ble_channel_count=37,
        ble_channel_bandwidth_hz=2e6,
        wifi_reuse_channels=(0, 5, 10),
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=1.25e-3,
        ble_payload_bits=None,
        ble_phy_rate_bps=1e6,
        ble_ci_exp_min=3,
        ble_ci_exp_max=11,
    ):
        """
        可调参数（中文说明）：
        1) cell_size/pair_density_per_m2: 办公室网格规模与通信对密度
        2) radio_prob: WiFi/BLE 用户比例
        3) wifi_channel_count/wifi_channel_bandwidth_hz/wifi_reuse_channels: WiFi 频谱配置
        4) ble_channel_count/ble_channel_bandwidth_hz: BLE 频谱配置
        5) ble_ci_min_s/ble_ci_max_s: BLE 连接间隔 CI 范围（周期）
        6) ble_ce_min_s/ble_ce_max_s: BLE 连接事件时长范围
        7) ble_payload_bits/ble_phy_rate_bps: BLE 负载与速率，用于计算 CE 所需时长
        """
        self.rand_gen_loc = np.random.default_rng(seed)
        self.rand_gen_fad = np.random.default_rng(seed)
        self.rand_gen_mob = np.random.default_rng(seed)

        self.cell_edge = cell_edge
        self.cell_size = cell_size
        self.grid_edge = self.cell_edge * self.cell_size

        self.n_office = int(self.cell_size**2)
        self.n_ap = self.n_office
        self.office_offset = self.cell_edge / 2.0
        self.ap_offset = self.office_offset

        if sta_density_per_1m2 is not None:
            pair_density_per_m2 = sta_density_per_1m2
        self.pair_density_per_m2 = pair_density_per_m2
        self.office_area_m2 = self.cell_edge**2
        self.pair_density_per_office = self.pair_density_per_m2 * self.office_area_m2
        self.n_pair = int(self.cell_size**2 * self.pair_density_per_office)
        self.n_sta = self.n_pair

        self.fre_Hz = fre_Hz
        self.lam = self.C / self.fre_Hz
        self.txp_dbm_hi = txp_dbm_hi
        self.txp_offset = txp_offset
        self.min_s_n_ratio = min_s_n_ratio
        self.packet_bit = packet_bit
        self.bandwidth = bandwidth
        self.slot_time = slot_time
        self.max_err = max_err

        self.office_locs = None
        self.device_locs = None
        self.device_dirs = None
        self.pair_tx_locs = None
        self.pair_rx_locs = None
        self.pair_office_id = None

        self.device_priority = None
        self.user_priority = None
        self.pair_priority = None
        self.prio_prob = np.array(prio_prob, dtype=float)
        self.prio_value = np.array(prio_value, dtype=float)

        self.RADIO_WIFI = 0
        self.RADIO_BLE = 1
        self.radio_prob = np.array(radio_prob, dtype=float)
        self.wifi_channel_count = int(wifi_channel_count)
        self.ble_channel_count = int(ble_channel_count)
        self.wifi_channel_bandwidth_hz = float(wifi_channel_bandwidth_hz)
        self.ble_channel_bandwidth_hz = float(ble_channel_bandwidth_hz)
        self.wifi_reuse_channels = np.array(wifi_reuse_channels, dtype=int)

        self.device_radio_type = None
        self.device_radio_channel = None
        self.device_ble_anchor_slot = None
        self.user_radio_type = None
        self.user_radio_channel = None
        self.user_ble_anchor_slot = None
        self.pair_radio_type = None
        self.pair_channel = None
        self.pair_ble_anchor_slot = None

        self.office_wifi_channel = None
        self.office_ble_channel = None
        self.ap_wifi_channel = None
        self.ap_ble_channel = None

        # BLE timing model
        self.ble_ci_min_s = float(ble_ci_min_s)
        self.ble_ci_max_s = float(ble_ci_max_s)
        self.ble_ce_min_s = float(ble_ce_min_s)
        self.ble_ce_max_s = float(ble_ce_max_s)
        self.ble_payload_bits = float(packet_bit if ble_payload_bits is None else ble_payload_bits)
        self.ble_phy_rate_bps = float(ble_phy_rate_bps)
        self.ble_ci_quanta_min = None
        self.ble_ci_quanta_max = None
        self.ble_ci_quanta_candidates = None
        self.ble_ce_required_s = None
        self.ble_ci_exp_min = int(ble_ci_exp_min)
        self.ble_ci_exp_max = int(ble_ci_exp_max)

        # 用户级 BLE 时序参数（仅对 BLE 用户有效）
        self.device_ble_ci_slots = None
        self.device_ble_ce_slots = None
        self.device_ble_ce_required_s = None
        self.device_ble_ce_feasible = None
        self.user_ble_ci_slots = None
        self.user_ble_ce_slots = None
        self.user_ble_ce_required_s = None
        self.user_ble_ce_feasible = None
        self.pair_ble_ci_slots = None
        self.pair_ble_ce_slots = None
        self.pair_ble_ce_required_s = None
        self.pair_ble_ce_feasible = None

        self.min_sinr = None
        self.loss = None

        self._config_ap_locs()
        self._config_ap_radio_channel()
        self._config_ble_timing()
        self._config_pairs()

    def _config_ap_locs(self):
        x = np.linspace(0 + self.office_offset, self.grid_edge - self.office_offset, self.cell_size)
        y = np.linspace(0 + self.office_offset, self.grid_edge - self.office_offset, self.cell_size)
        xx, yy = np.meshgrid(x, y)
        self.office_locs = np.array((xx.ravel(), yy.ravel())).T
        self.ap_locs = self.office_locs

    def _config_ap_radio_channel(self):
        # 中文：为每个 AP 固定分配 WiFi/BLE 信道。WiFi 默认复用 1/6/11（索引 0/5/10）。
        if self.wifi_reuse_channels.size == 0:
            raise ValueError("wifi_reuse_channels must not be empty.")
        if np.any(self.wifi_reuse_channels < 0) or np.any(self.wifi_reuse_channels >= self.wifi_channel_count):
            raise ValueError("wifi_reuse_channels must be within [0, wifi_channel_count).")
        office_ids = np.arange(self.n_office)
        self.office_wifi_channel = self.wifi_reuse_channels[office_ids % self.wifi_reuse_channels.size]
        self.office_ble_channel = office_ids % self.ble_channel_count
        self.ap_wifi_channel = self.office_wifi_channel
        self.ap_ble_channel = self.office_ble_channel

    def _sample_priority(self, n_obj):
        if self.prio_prob.size != self.prio_value.size:
            raise ValueError("prio_prob and prio_value must have the same length.")
        prob_sum = float(self.prio_prob.sum())
        if prob_sum <= 0:
            raise ValueError("prio_prob must have a positive sum.")
        prob = self.prio_prob / prob_sum
        idx = self.rand_gen_loc.choice(self.prio_value.size, size=n_obj, p=prob)
        return self.prio_value[idx]

    def _sample_pair_endpoint_in_office(self, office_id):
        office_x = office_id % self.cell_size
        office_y = office_id // self.cell_size
        x0 = office_x * self.cell_edge
        y0 = office_y * self.cell_edge
        return np.array(
            [
                self.rand_gen_loc.uniform(x0, x0 + self.cell_edge),
                self.rand_gen_loc.uniform(y0, y0 + self.cell_edge),
            ]
        )

    def _config_pairs(self):
        # 中文：直接生成通信对，不再先生成单设备。
        if self.radio_prob.size != 2:
            raise ValueError("radio_prob must contain two probabilities: (wifi, ble).")
        p_sum = float(np.sum(self.radio_prob))
        if p_sum <= 0:
            raise ValueError("radio_prob sum must be positive.")
        if self.wifi_channel_count <= 0 or self.ble_channel_count <= 0:
            raise ValueError("wifi_channel_count and ble_channel_count must be positive.")

        radio_p = self.radio_prob / p_sum
        self.pair_radio_type = self.rand_gen_loc.choice(
            np.array([self.RADIO_WIFI, self.RADIO_BLE], dtype=int),
            size=self.n_pair,
            p=radio_p,
        )

        self.pair_office_id = self.rand_gen_loc.integers(low=0, high=self.n_office, size=self.n_pair)
        self.pair_channel = np.zeros(self.n_pair, dtype=int)
        self.pair_ble_anchor_slot = np.zeros(self.n_pair, dtype=int)
        self.pair_ble_ci_slots = np.zeros(self.n_pair, dtype=int)
        self.pair_ble_ce_slots = np.zeros(self.n_pair, dtype=int)
        self.pair_ble_ce_required_s = np.zeros(self.n_pair, dtype=float)
        self.pair_ble_ce_feasible = np.zeros(self.n_pair, dtype=bool)
        self.pair_tx_locs = np.zeros((self.n_pair, 2), dtype=float)
        self.pair_rx_locs = np.zeros((self.n_pair, 2), dtype=float)

        for k in range(self.n_pair):
            office_id = int(self.pair_office_id[k])
            self.pair_tx_locs[k] = self._sample_pair_endpoint_in_office(office_id)
            self.pair_rx_locs[k] = self._sample_pair_endpoint_in_office(office_id)

        wifi_mask = self.pair_radio_type == self.RADIO_WIFI
        ble_mask = self.pair_radio_type == self.RADIO_BLE

        if np.any(wifi_mask):
            self.pair_channel[wifi_mask] = self.rand_gen_loc.integers(
                low=0, high=self.wifi_channel_count, size=int(np.sum(wifi_mask))
            )
        if np.any(ble_mask):
            self.pair_channel[ble_mask] = self.rand_gen_loc.integers(
                low=0, high=self.ble_channel_count, size=int(np.sum(ble_mask))
            )
            ble_pairs = np.where(ble_mask)[0]
            self._config_ble_pair_timing(ble_pairs)

        self.pair_priority = self._sample_priority(self.n_pair)
        self.device_priority = self.pair_priority

        # 兼容旧接口：把 pair 语义映射到 device_* 与 user_* 字段。
        self.device_radio_type = self.pair_radio_type
        self.device_radio_channel = self.pair_channel
        self.device_ble_anchor_slot = self.pair_ble_anchor_slot
        self.device_ble_ci_slots = self.pair_ble_ci_slots
        self.device_ble_ce_slots = self.pair_ble_ce_slots
        self.device_ble_ce_required_s = self.pair_ble_ce_required_s
        self.device_ble_ce_feasible = self.pair_ble_ce_feasible
        self.user_priority = self.pair_priority
        self.user_radio_type = self.pair_radio_type
        self.user_radio_channel = self.pair_channel
        self.user_ble_anchor_slot = self.pair_ble_anchor_slot
        self.user_ble_ci_slots = self.pair_ble_ci_slots
        self.user_ble_ce_slots = self.pair_ble_ce_slots
        self.user_ble_ce_required_s = self.pair_ble_ce_required_s
        self.user_ble_ce_feasible = self.pair_ble_ce_feasible
        self.device_locs = self.pair_tx_locs
        self.device_dirs = np.zeros((self.n_pair, 2), dtype=float)
        self.sta_locs = self.device_locs
        self.sta_dirs = self.device_dirs

    def _config_ble_timing(self):
        # 中文：IFS/MSS 不再作为可调参数体现在 env 中。
        # 这里仅维护 BLE CI/CE 的全局约束范围；具体 CI/CE 在每个 BLE 用户上随机生成。
        ci_base = 1.25e-3
        ci_min = max(7.5e-3, self.ble_ci_min_s)
        ci_max = min(4.0, self.ble_ci_max_s)
        if ci_min > ci_max:
            raise ValueError("BLE CI range is invalid: ble_ci_min_s > ble_ci_max_s after clamping.")

        self.ble_ci_quanta_min = int(math.ceil(ci_min / ci_base))
        self.ble_ci_quanta_max = int(math.floor(ci_max / ci_base))
        if self.ble_ci_quanta_min > self.ble_ci_quanta_max:
            raise ValueError("BLE CI range must be integer multiples of 1.25ms.")
        if self.ble_ci_exp_min > self.ble_ci_exp_max:
            raise ValueError("ble_ci_exp_min must be <= ble_ci_exp_max.")

        # 中文：CI 候选集合采用离散规则 CI = 2^n * 1.25ms (n in [ble_ci_exp_min, ble_ci_exp_max])。
        pow2_quanta = np.array([2**n for n in range(self.ble_ci_exp_min, self.ble_ci_exp_max + 1)], dtype=int)
        pow2_quanta = pow2_quanta[
            np.logical_and(pow2_quanta >= self.ble_ci_quanta_min, pow2_quanta <= self.ble_ci_quanta_max)
        ]
        if pow2_quanta.size == 0:
            raise ValueError("No valid BLE CI candidates under 2^n*1.25ms rule.")
        self.ble_ci_quanta_candidates = pow2_quanta

        if self.ble_phy_rate_bps <= 0:
            raise ValueError("ble_phy_rate_bps must be positive.")
        if self.ble_ce_min_s < 1.25e-3:
            raise ValueError("ble_ce_min_s must be >= 1.25ms.")
        if self.ble_ce_max_s < self.ble_ce_min_s:
            raise ValueError("ble_ce_max_s must be >= ble_ce_min_s.")

        # QoS 所需时长：至少覆盖 payload 发送时间与 CE 最小时长
        payload_tx_s = self.ble_payload_bits / self.ble_phy_rate_bps
        self.ble_ce_required_s = float(max(self.ble_ce_min_s, payload_tx_s))

    def _config_ble_pair_timing(self, ble_pairs):
        # 中文：为每个 BLE pair 随机生成满足约束的 CI 与 CE。
        # CI: [ble_ci_quanta_min, ble_ci_quanta_max] * 1.25ms
        # CE: 在 [ce_required, ble_ce_max] 内随机；若 ce_required > ble_ce_max 则该用户不可满足。
        ci_base = 1.25e-3
        for k in ble_pairs:
            # 中文：CI 只从离散候选集合采样，不再用连续整数区间采样。
            ci_quanta = int(self.rand_gen_loc.choice(self.ble_ci_quanta_candidates))
            ci_s = ci_quanta * ci_base
            ci_slots = int(round(ci_s / self.slot_time))
            if abs(ci_s / self.slot_time - ci_slots) > 1e-9:
                raise ValueError("Sampled BLE CI is not an integer multiple of slot_time.")
            self.pair_ble_ci_slots[k] = ci_slots

            ce_required = self.ble_ce_required_s
            self.pair_ble_ce_required_s[k] = ce_required
            feasible = ce_required <= self.ble_ce_max_s
            self.pair_ble_ce_feasible[k] = feasible
            if feasible:
                # 在可行范围内随机 CE，并且限制 CE 不超过 CI
                ce_high = min(self.ble_ce_max_s, ci_s)
                if ce_required > ce_high:
                    self.pair_ble_ce_feasible[k] = False
                    self.pair_ble_ce_slots[k] = 0
                else:
                    ce_s = float(self.rand_gen_loc.uniform(low=ce_required, high=ce_high))
                    self.pair_ble_ce_slots[k] = max(1, int(math.ceil(ce_s / self.slot_time)))
            else:
                self.pair_ble_ce_slots[k] = 0

            # 锚点按用户自己的 CI 周期随机
            if self.pair_ble_ci_slots[k] > 0:
                # 中文：anchor 以 slot 索引表示，天然是 slot_time(当前 0.125ms=125us) 的整数倍。
                self.pair_ble_anchor_slot[k] = int(self.rand_gen_loc.integers(low=0, high=self.pair_ble_ci_slots[k]))

    def build_slot_compatibility_mask(self, Z):
        # 中文：构建“用户-时隙可用掩码”。
        # WiFi 用户：所有时隙可用；
        # BLE 用户：仅在 anchor + n*CI 的 CE 连续窗口内可用；
        # 若 CE 不可满足，则 BLE 全部时隙不可用。
        Z = int(Z)
        K = int(self.n_pair)
        mask = np.ones((K, Z), dtype=bool)
        ble_idx = np.where(self.pair_radio_type == self.RADIO_BLE)[0]
        for k in ble_idx:
            mask[k, :] = False
            if not self.pair_ble_ce_feasible[k]:
                continue
            ci_slots = int(self.pair_ble_ci_slots[k])
            ce_slots = int(self.pair_ble_ce_slots[k])
            if ci_slots <= 0 or ce_slots <= 0:
                continue
            anchor = int(self.pair_ble_anchor_slot[k] % ci_slots)
            z = anchor
            while z < Z:
                end_z = min(z + ce_slots, Z)
                mask[k, z:end_z] = True
                z += ci_slots
        return mask

    def _get_wifi_channel_range_hz(self, wifi_idx):
        center = (2412.0 + 5.0 * float(wifi_idx)) * 1e6
        half_bw = self.wifi_channel_bandwidth_hz / 2.0
        return center - half_bw, center + half_bw

    def _get_ble_channel_range_hz(self, ble_idx):
        center = (2402.0 + 2.0 * float(ble_idx)) * 1e6
        half_bw = self.ble_channel_bandwidth_hz / 2.0
        return center - half_bw, center + half_bw

    def _get_pair_link_range_hz(self, pair_idx):
        if self.pair_radio_type[pair_idx] == self.RADIO_WIFI:
            return self._get_wifi_channel_range_hz(self.pair_channel[pair_idx])
        return self._get_ble_channel_range_hz(self.pair_channel[pair_idx])

    def _build_link_overlap_mask(self):
        # 中文：只保留频段重叠链路的干扰项；非重叠频段干扰置零。
        K = self.n_pair
        mask = np.zeros((K, K), dtype=bool)
        tx_ranges = [self._get_pair_link_range_hz(i) for i in range(K)]
        rx_ranges = [self._get_pair_link_range_hz(j) for j in range(K)]
        for i in range(K):
            t0, t1 = tx_ranges[i]
            for j in range(K):
                if i == j:
                    mask[i, j] = True
                    continue
                r0, r1 = rx_ranges[j]
                if env._is_range_overlap(t0, t1, r0, r1):
                    mask[i, j] = True
        return mask

    @staticmethod
    def _is_range_overlap(a0, a1, b0, b1):
        return not (a1 <= b0 or b1 <= a0)

    def _build_radio_interference_constraints(self, K):
        # 中文：构建冲突图（WiFi-WiFi, WiFi-BLE, BLE-BLE），依据频谱区间是否重叠。
        Q_radio = scipy.sparse.lil_matrix((K, K))
        wifi_pairs = np.where(self.pair_radio_type == self.RADIO_WIFI)[0]
        ble_pairs = np.where(self.pair_radio_type == self.RADIO_BLE)[0]

        for ii in range(wifi_pairs.size):
            ui = wifi_pairs[ii]
            ch_i = self.pair_channel[ui]
            i0, i1 = self._get_wifi_channel_range_hz(ch_i)
            for jj in range(ii + 1, wifi_pairs.size):
                uj = wifi_pairs[jj]
                ch_j = self.pair_channel[uj]
                j0, j1 = self._get_wifi_channel_range_hz(ch_j)
                if env._is_range_overlap(i0, i1, j0, j1):
                    Q_radio[ui, uj] = 1
                    Q_radio[uj, ui] = 1

        for wi in wifi_pairs:
            wch = self.pair_channel[wi]
            w0, w1 = self._get_wifi_channel_range_hz(wch)
            for bj in ble_pairs:
                bch = self.pair_channel[bj]
                b0, b1 = self._get_ble_channel_range_hz(bch)
                if env._is_range_overlap(w0, w1, b0, b1):
                    Q_radio[wi, bj] = 1
                    Q_radio[bj, wi] = 1

        for ii in range(ble_pairs.size):
            ui = ble_pairs[ii]
            ch_i = self.pair_channel[ui]
            i0, i1 = self._get_ble_channel_range_hz(ch_i)
            for jj in range(ii + 1, ble_pairs.size):
                uj = ble_pairs[jj]
                ch_j = self.pair_channel[uj]
                j0, j1 = self._get_ble_channel_range_hz(ch_j)
                if env._is_range_overlap(i0, i1, j0, j1):
                    Q_radio[ui, uj] = 1
                    Q_radio[uj, ui] = 1

        Q_radio.setdiag(0)
        return Q_radio.tocsr()

    def get_radio_conflict_stats(self):
        # 中文：统计三类冲突边数量，便于调参与结果解释。
        K = self.n_pair
        Q_radio = self._build_radio_interference_constraints(K).tocsr()

        wifi_idx = np.where(self.pair_radio_type == self.RADIO_WIFI)[0]
        ble_idx = np.where(self.pair_radio_type == self.RADIO_BLE)[0]

        wifi_ble_edges = int(Q_radio[np.ix_(wifi_idx, ble_idx)].nnz) if wifi_idx.size and ble_idx.size else 0
        wifi_wifi_nnz = int(Q_radio[np.ix_(wifi_idx, wifi_idx)].nnz) if wifi_idx.size else 0
        ble_ble_nnz = int(Q_radio[np.ix_(ble_idx, ble_idx)].nnz) if ble_idx.size else 0
        wifi_wifi_edges = wifi_wifi_nnz // 2
        ble_ble_edges = ble_ble_nnz // 2
        total_edges = int(Q_radio.nnz // 2)

        return {
            "n_pair": int(K),
            "n_wifi_pair": int(wifi_idx.size),
            "n_ble_pair": int(ble_idx.size),
            "wifi_wifi_edges": int(wifi_wifi_edges),
            "wifi_ble_edges": int(wifi_ble_edges),
            "ble_ble_edges": int(ble_ble_edges),
            "total_radio_conflict_edges": int(total_edges),
        }

    def _get_random_dir(self):
        dd = self.rand_gen_mob.standard_normal(2)
        return dd / np.linalg.norm(dd)

    def _compute_min_sinr(self):
        min_sinr_db = env.bisection_method(self.packet_bit, self.bandwidth, self.slot_time, self.max_err)
        self.min_sinr = env.db_to_dec(min_sinr_db)
        return self.min_sinr

    def rand_device_mobility(self, mobility_in_meter_s=0.0, t_us=0, resolution_us=1.0):
        if mobility_in_meter_s == 0.0 or t_us == 0:
            return
        n_step = math.ceil(t_us / resolution_us)
        for _ in range(n_step):
            for i in range(self.n_pair):
                dd = self.device_dirs[i] * mobility_in_meter_s * resolution_us / 1e6
                x = self.device_locs[i][0] + dd[0]
                y = self.device_locs[i][1] + dd[1]
                if 0 <= x <= self.grid_edge and 0 <= y <= self.grid_edge:
                    self.device_locs[i] = np.array([x, y])
                else:
                    self.device_dirs[i] = self._get_random_dir()
        self.sta_locs = self.device_locs
        self.sta_dirs = self.device_dirs

    @classmethod
    def bandwidth_txpr_to_noise_dBm(cls, B):
        return env.NOISE_FLOOR_DBM

    @staticmethod
    def fre_dis_to_loss_dB(fre_Hz, dis):
        L = 20.0 * math.log10(fre_Hz / 1e6) + 16 - 28
        return L + 28 * np.log10(dis + 1)

    @staticmethod
    def db_to_dec(snr_db):
        return 10.0 ** (snr_db / 10.0)

    @staticmethod
    def dec_to_db(snr_dec):
        return 10.0 * math.log10(snr_dec)

    @staticmethod
    def polyanskiy_model(snr_dec, L, B, T):
        nu = -L * math.log(2.0) + B * T * math.log(1 + snr_dec)
        do = math.sqrt(B * T * (1.0 - 1.0 / ((1.0 + snr_dec) ** 2)))
        return scipy.stats.norm.sf(nu / do)

    @staticmethod
    def err(x, L, B, T, max_err):
        snr = env.db_to_dec(x)
        return env.polyanskiy_model(snr, L, B, T) / max_err - 1.0

    @staticmethod
    def bisection_method(L, B, T, max_err=1e-5, a=-5.0, b=30.0, tol=0.1):
        if env.err(a, L, B, T, max_err) * env.err(b, L, B, T, max_err) >= 0:
            print("Bisection method fails.")
            return None

        while (env.err(a, L, B, T, max_err) - env.err(b, L, B, T, max_err)) > tol:
            midpoint = (a + b) / 2
            if env.err(midpoint, L, B, T, max_err) == 0:
                return midpoint
            if env.err(a, L, B, T, max_err) * env.err(midpoint, L, B, T, max_err) < 0:
                b = midpoint
            else:
                a = midpoint
        return (a + b) / 2

    def _compute_txp(self):
        dis = np.linalg.norm(self.pair_tx_locs - self.pair_rx_locs, axis=1)
        gain = -env.fre_dis_to_loss_dB(self.fre_Hz, dis)
        t = env.dec_to_db(self._compute_min_sinr()) - (gain - self.bandwidth_txpr_to_noise_dBm(self.bandwidth))
        return np.reshape(t + env.dec_to_db(self.txp_offset), (self.n_pair, -1))

    def _compute_state(self):
        dis = scipy.spatial.distance.cdist(self.pair_tx_locs, self.pair_rx_locs)
        self.loss = env.fre_dis_to_loss_dB(self.fre_Hz, dis)
        rxpr_db = self._compute_txp() - self.loss - self.bandwidth_txpr_to_noise_dBm(self.bandwidth)
        rxpr_hi = 10 ** (rxpr_db / 10.0)
        rxpr_hi[rxpr_hi < self.min_s_n_ratio] = 0.0
        return scipy.sparse.csr_matrix(rxpr_hi)

    def _compute_state_real(self):
        dis = scipy.spatial.distance.cdist(self.pair_tx_locs, self.pair_rx_locs)
        self.loss = env.fre_dis_to_loss_dB(self.fre_Hz, dis)
        rxpr_db = self._compute_txp() - self.loss - self.bandwidth_txpr_to_noise_dBm(self.bandwidth)
        rxpr_hi = 10 ** (rxpr_db / 10.0)
        return scipy.sparse.csr_matrix(rxpr_hi)

    def generate_S_Q_hmax(self, real: bool = False) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, np.ndarray]:
        rxpr = self._compute_state_real() if real else self._compute_state()

        K = rxpr.shape[0]
        Q_conflict = self._build_radio_interference_constraints(K).tocsr()
        Q_conflict.setdiag(0.0)
        Q_conflict.sort_indices()
        Q_conflict.eliminate_zeros()

        S_gain = rxpr.copy()
        mask = self._build_link_overlap_mask()
        S_gain = scipy.sparse.csr_matrix(S_gain.toarray() * mask.astype(float))
        S_gain.eliminate_zeros()
        S_gain.sort_indices()

        h_max = S_gain.diagonal() / self._compute_min_sinr() - 1.0
        return S_gain, Q_conflict, h_max

    def evaluate_sinr(self, z, Z):
        rxpr = self._compute_state_real()
        S_gain, _, _ = self.generate_S_Q_hmax(real=True)
        S_gain = np.array(S_gain.toarray())
        S_gain_T_no_diag = S_gain.copy().transpose()
        np.fill_diagonal(S_gain_T_no_diag, 0)

        K = rxpr.shape[0]
        sinr = np.zeros(K) + 1e-3
        for zz in range(Z):
            kidx = np.arange(K)[z == zz]
            signal = S_gain.diagonal()[kidx]
            interference = np.asarray(S_gain_T_no_diag[kidx][:, kidx].sum(axis=1)).ravel()
            sinr[kidx] = signal / (interference + 1)

        return sinr

    def evaluate_bler(self, z, Z):
        sinr = self.evaluate_sinr(z, Z)
        bler = np.zeros(sinr.size)
        for k in range(sinr.size):
            bler[k] = env.polyanskiy_model(sinr[k], self.packet_bit, self.bandwidth, self.slot_time)
        return bler

    def evaluate_weighted_bler(self, z, Z, weights=None):
        bler = self.evaluate_bler(z, Z)
        if weights is None:
            weights = self.pair_priority
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != bler.shape[0]:
            raise ValueError("weights length must match number of users.")
        sw = float(np.sum(weights))
        if sw <= 0:
            raise ValueError("weights sum must be positive.")
        return float(np.sum(weights * bler) / sw)

    def evaluate_pckl(self, z, Z):
        bler = self.evaluate_bler(z, Z)
        return np.array([np.random.choice([0, 1], p=[1 - prob, prob]) for prob in bler])

    def check_cell_edge_snr_err(self):
        l = env.fre_dis_to_loss_dB(self.fre_Hz, self.cell_edge / 2 * math.sqrt(2))
        s_db = self.txp_dbm_hi - l - env.bandwidth_txpr_to_noise_dBm(self.bandwidth)
        s_dec = env.db_to_dec(s_db)
        err = env.polyanskiy_model(s_dec, self.packet_bit, self.bandwidth, self.slot_time)
        print("snr_db", s_db, "snr_dec", s_dec, "err", err)


if __name__ == '__main__':
    e = env(cell_size=5, seed=2)
    print('n_pair', e.n_pair, 'n_office', e.n_office)
    ble_idx = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    if ble_idx.size:
        print(
            'ble_ci_slots(min/avg/max)',
            int(np.min(e.pair_ble_ci_slots[ble_idx])),
            float(np.mean(e.pair_ble_ci_slots[ble_idx])),
            int(np.max(e.pair_ble_ci_slots[ble_idx])),
        )
        print(
            'ble_ce_slots(min/avg/max)',
            int(np.min(e.pair_ble_ce_slots[ble_idx])),
            float(np.mean(e.pair_ble_ce_slots[ble_idx])),
            int(np.max(e.pair_ble_ce_slots[ble_idx])),
        )
        print('ble_infeasible_pair', int(np.sum(~e.pair_ble_ce_feasible[ble_idx])))
