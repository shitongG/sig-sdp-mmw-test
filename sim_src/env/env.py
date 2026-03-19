import math

import numpy as np
import scipy


def sample_ble_pair_timing(
    *,
    rand_gen,
    slot_time: float,
    ble_ci_quanta_candidates: np.ndarray,
    ble_ce_required_s: float,
    ble_ce_max_s: float,
    start_time_slot: int,
):
    ci_base = 1.25e-3
    ci_quanta = int(rand_gen.choice(ble_ci_quanta_candidates))
    ci_s = ci_quanta * ci_base
    ci_slots = int(round(ci_s / slot_time))
    if abs(ci_s / slot_time - ci_slots) > 1e-9:
        raise ValueError("Sampled BLE CI is not an integer multiple of slot_time.")

    ce_required = float(ble_ce_required_s)
    feasible = ce_required <= float(ble_ce_max_s)
    ce_slots = 0
    if feasible:
        ce_high = min(float(ble_ce_max_s), ci_s)
        if ce_required > ce_high:
            feasible = False
        else:
            ce_low_slots = int(math.ceil(ce_required / slot_time))
            ce_high_slots = int(math.floor(ce_high / slot_time))
            if ce_low_slots > ce_high_slots:
                feasible = False
            else:
                ce_slots = int(rand_gen.integers(low=ce_low_slots, high=ce_high_slots + 1))

    anchor_slot = int(start_time_slot)
    if ci_slots > 0:
        high = int(start_time_slot + ci_slots - ce_slots)
        anchor_slot = int(rand_gen.integers(low=int(start_time_slot), high=high + 1))

    return {
        "ci_slots": ci_slots,
        "ce_slots": ce_slots,
        "anchor_slot": anchor_slot,
        "ce_required_s": ce_required,
        "feasible": bool(feasible),
    }


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
        packet_bit=8000,
        bandwidth=5e6,
        wifi_packet_bit=None,
        ble_packet_bit=None,
        pair_packet_bits=None,
        user_packet_bits=None,
        pair_bandwidth_hz=None,
        user_bandwidth_hz=None,
        slot_time=1.25e-3,
        max_err=1e-5,
        seed=1,
        prio_prob=(0.2, 0.3, 0.5),
        prio_value=(3.0, 2.0, 1.0),
        radio_prob=(0.5, 0.5),
        wifi_channel_count=13,
        wifi_channel_bandwidth_hz=20e6,
        wifi_tx_min_s=5e-3,
        wifi_tx_max_s=10e-3,
        wifi_period_exp_min=4,
        wifi_period_exp_max=5,
        ble_channel_count=37,
        ble_channel_bandwidth_hz=2e6,
        wifi_reuse_channels=(0, 5, 10),
        ble_ci_min_s=7.5e-3,
        ble_ci_max_s=4.0,
        ble_ce_min_s=1.25e-3,
        ble_ce_max_s=7.5e-3,
        ble_payload_bits=None,
        ble_phy_rate_bps=1e6,
        ble_ci_exp_min=3,
        ble_ci_exp_max=11,
        ble_channel_mode="single",
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
        self.seed = seed
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
        self.wifi_packet_bit = float(packet_bit if wifi_packet_bit is None else wifi_packet_bit)
        self.ble_packet_bit = float(packet_bit if ble_packet_bit is None else ble_packet_bit)
        self._pair_packet_bits_input = pair_packet_bits
        self._user_packet_bits_input = user_packet_bits
        self._pair_bandwidth_hz_input = pair_bandwidth_hz
        self._user_bandwidth_hz_input = user_bandwidth_hz
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
        self.device_packet_bits = None
        self.user_packet_bits = None
        self.pair_packet_bits = None
        self.device_bandwidth_hz = None
        self.user_bandwidth_hz = None
        self.pair_bandwidth_hz = None
        self.device_min_sinr = None
        self.user_min_sinr = None
        self.pair_min_sinr = None
        self.prio_prob = np.array(prio_prob, dtype=float)
        self.prio_value = np.array(prio_value, dtype=float)

        self.RADIO_WIFI = 0
        self.RADIO_BLE = 1
        self.radio_prob = np.array(radio_prob, dtype=float)
        self.wifi_channel_count = int(wifi_channel_count)
        self.ble_channel_count = int(ble_channel_count)
        self.wifi_channel_bandwidth_hz = float(wifi_channel_bandwidth_hz)
        self.ble_channel_bandwidth_hz = float(ble_channel_bandwidth_hz)
        if ble_channel_mode not in {"single", "per_ce"}:
            raise ValueError("ble_channel_mode must be 'single' or 'per_ce'.")
        self.ble_channel_mode = ble_channel_mode
        self.wifi_reuse_channels = np.array(wifi_reuse_channels, dtype=int)
        self.wifi_fixed_channel_indices = np.array([0, 5, 10], dtype=int)
        self.ble_data_channel_indices = np.arange(37, dtype=int)
        self.ble_advertising_center_freq_mhz = [2402.0, 2426.0, 2480.0]
        self.wifi_tx_min_s = float(wifi_tx_min_s)
        self.wifi_tx_max_s = float(wifi_tx_max_s)
        self.wifi_period_exp_min = int(wifi_period_exp_min)
        self.wifi_period_exp_max = int(wifi_period_exp_max)
        self.wifi_period_quanta_candidates = None
        self.wifi_tx_quanta_candidates = None

        self.device_radio_type = None
        self.device_radio_channel = None
        self.device_start_time_slot = None
        self.device_release_time_slot = None
        self.device_deadline_slot = None
        self.device_wifi_anchor_slot = None
        self.device_wifi_period_slots = None
        self.device_wifi_tx_slots = None
        self.device_ble_anchor_slot = None
        self.user_radio_type = None
        self.user_radio_channel = None
        self.user_wifi_anchor_slot = None
        self.user_wifi_period_slots = None
        self.user_wifi_tx_slots = None
        self.user_ble_anchor_slot = None
        self.user_start_time_slot = None
        self.user_release_time_slot = None
        self.user_deadline_slot = None
        self.pair_radio_type = None
        self.pair_channel = None
        self.pair_start_time_slot = None
        self.pair_release_time_slot = None
        self.pair_deadline_slot = None
        self.pair_wifi_anchor_slot = None
        self.pair_wifi_period_slots = None
        self.pair_wifi_tx_slots = None
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
        self.pair_ble_ce_channels = None
        self._manual_ble_ce_channel_pairs = None

        self.min_sinr = None
        self.loss = None

        self._config_ap_locs()
        self._config_ap_radio_channel()
        self._config_wifi_timing()
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
        if np.any(self.wifi_fixed_channel_indices < 0) or np.any(self.wifi_fixed_channel_indices >= self.wifi_channel_count):
            raise ValueError("wifi_fixed_channel_indices must be within [0, wifi_channel_count).")
        office_ids = np.arange(self.n_office)
        self.office_wifi_channel = self.wifi_fixed_channel_indices[office_ids % self.wifi_fixed_channel_indices.size]
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
        self.pair_start_time_slot = np.zeros(self.n_pair, dtype=int)
        self.pair_release_time_slot = np.zeros(self.n_pair, dtype=int)
        self.pair_deadline_slot = np.zeros(self.n_pair, dtype=int)
        self.pair_wifi_anchor_slot = np.zeros(self.n_pair, dtype=int)
        self.pair_wifi_period_slots = np.zeros(self.n_pair, dtype=int)
        self.pair_wifi_tx_slots = np.zeros(self.n_pair, dtype=int)
        self.pair_ble_anchor_slot = np.zeros(self.n_pair, dtype=int)
        self.pair_ble_ci_slots = np.zeros(self.n_pair, dtype=int)
        self.pair_ble_ce_slots = np.zeros(self.n_pair, dtype=int)
        self.pair_ble_ce_required_s = np.zeros(self.n_pair, dtype=float)
        self.pair_ble_ce_feasible = np.zeros(self.n_pair, dtype=bool)
        self.pair_ble_ce_channels = {} if self.ble_channel_mode == "per_ce" else None
        self._manual_ble_ce_channel_pairs = set() if self.ble_channel_mode == "per_ce" else None
        self.pair_tx_locs = np.zeros((self.n_pair, 2), dtype=float)
        self.pair_rx_locs = np.zeros((self.n_pair, 2), dtype=float)

        for k in range(self.n_pair):
            office_id = int(self.pair_office_id[k])
            self.pair_tx_locs[k] = self._sample_pair_endpoint_in_office(office_id)
            self.pair_rx_locs[k] = self._sample_pair_endpoint_in_office(office_id)

        wifi_mask = self.pair_radio_type == self.RADIO_WIFI
        ble_mask = self.pair_radio_type == self.RADIO_BLE

        if np.any(wifi_mask):
            self.pair_channel[wifi_mask] = self.rand_gen_loc.choice(
                self.wifi_fixed_channel_indices,
                size=int(np.sum(wifi_mask)),
            )
            wifi_pairs = np.where(wifi_mask)[0]
            self._config_wifi_pair_timing(wifi_pairs)
        if np.any(ble_mask):
            self.pair_channel[ble_mask] = self.rand_gen_loc.choice(
                self.ble_data_channel_indices,
                size=int(np.sum(ble_mask)),
            )
            ble_pairs = np.where(ble_mask)[0]
            self._config_ble_pair_timing(ble_pairs)

        macrocycle_slots = int(self.compute_macrocycle_slots())
        self.pair_release_time_slot = np.zeros(self.n_pair, dtype=int)
        self.pair_deadline_slot = np.full(self.n_pair, max(macrocycle_slots - 1, 0), dtype=int)

        self.pair_priority = self._sample_priority(self.n_pair)
        self.device_priority = self.pair_priority
        self.pair_packet_bits = self._resolve_pair_float_array(
            pair_values=self._pair_packet_bits_input,
            user_values=self._user_packet_bits_input,
            wifi_default=self.wifi_packet_bit,
            ble_default=self.ble_packet_bit,
            attr_name="packet bits",
        )
        self.pair_bandwidth_hz = self._resolve_pair_float_array(
            pair_values=self._pair_bandwidth_hz_input,
            user_values=self._user_bandwidth_hz_input,
            wifi_default=self.wifi_channel_bandwidth_hz,
            ble_default=self.ble_channel_bandwidth_hz,
            attr_name="bandwidth",
        )

        # 兼容旧接口：把 pair 语义映射到 device_* 与 user_* 字段。
        self.device_radio_type = self.pair_radio_type
        self.device_radio_channel = self.pair_channel
        self.device_start_time_slot = self.pair_start_time_slot
        self.device_release_time_slot = self.pair_release_time_slot
        self.device_deadline_slot = self.pair_deadline_slot
        self.device_wifi_anchor_slot = self.pair_wifi_anchor_slot
        self.device_wifi_period_slots = self.pair_wifi_period_slots
        self.device_wifi_tx_slots = self.pair_wifi_tx_slots
        self.device_ble_anchor_slot = self.pair_ble_anchor_slot
        self.device_ble_ci_slots = self.pair_ble_ci_slots
        self.device_ble_ce_slots = self.pair_ble_ce_slots
        self.device_ble_ce_required_s = self.pair_ble_ce_required_s
        self.device_ble_ce_feasible = self.pair_ble_ce_feasible
        self.device_packet_bits = self.pair_packet_bits
        self.device_bandwidth_hz = self.pair_bandwidth_hz
        self.user_priority = self.pair_priority
        self.user_radio_type = self.pair_radio_type
        self.user_radio_channel = self.pair_channel
        self.user_start_time_slot = self.pair_start_time_slot
        self.user_release_time_slot = self.pair_release_time_slot
        self.user_deadline_slot = self.pair_deadline_slot
        self.user_wifi_anchor_slot = self.pair_wifi_anchor_slot
        self.user_wifi_period_slots = self.pair_wifi_period_slots
        self.user_wifi_tx_slots = self.pair_wifi_tx_slots
        self.user_ble_anchor_slot = self.pair_ble_anchor_slot
        self.user_ble_ci_slots = self.pair_ble_ci_slots
        self.user_ble_ce_slots = self.pair_ble_ce_slots
        self.user_ble_ce_required_s = self.pair_ble_ce_required_s
        self.user_ble_ce_feasible = self.pair_ble_ce_feasible
        self.user_packet_bits = self.pair_packet_bits
        self.user_bandwidth_hz = self.pair_bandwidth_hz
        self.device_locs = self.pair_tx_locs
        self.device_dirs = np.zeros((self.n_pair, 2), dtype=float)
        self.sta_locs = self.device_locs
        self.sta_dirs = self.device_dirs
        self._compute_min_sinr()

    def _resolve_pair_float_array(self, pair_values, user_values, wifi_default, ble_default, attr_name):
        values = pair_values if pair_values is not None else user_values
        if values is not None:
            arr = np.asarray(values, dtype=float)
            if arr.shape != (self.n_pair,):
                raise ValueError(f"{attr_name} override length must match number of pairs.")
            return arr.copy()

        defaults = np.where(
            self.pair_radio_type == self.RADIO_WIFI,
            float(wifi_default),
            float(ble_default),
        )
        return np.asarray(defaults, dtype=float)

    def _config_wifi_timing(self):
        base = 1.25e-3
        if self.wifi_period_exp_min > self.wifi_period_exp_max:
            raise ValueError("wifi_period_exp_min must be <= wifi_period_exp_max.")
        self.wifi_period_quanta_candidates = np.array(
            [2**n for n in range(self.wifi_period_exp_min, self.wifi_period_exp_max + 1)],
            dtype=int,
        )
        if self.wifi_period_quanta_candidates.size == 0:
            raise ValueError("No valid WiFi period candidates under 2^n*1.25ms rule.")
        if self.wifi_tx_min_s <= 0:
            raise ValueError("wifi_tx_min_s must be positive.")
        if self.wifi_tx_max_s < self.wifi_tx_min_s:
            raise ValueError("wifi_tx_max_s must be >= wifi_tx_min_s.")
        tx_min_slots = int(round(self.wifi_tx_min_s / self.slot_time))
        tx_max_slots = int(round(self.wifi_tx_max_s / self.slot_time))
        if tx_min_slots <= 0 or tx_max_slots <= 0:
            raise ValueError("WiFi TX duration candidates must map to positive slot counts.")
        self.wifi_tx_quanta_candidates = np.arange(tx_min_slots, tx_max_slots + 1, dtype=int)

    def _config_wifi_pair_timing(self, wifi_pairs):
        base = 1.25e-3
        for k in wifi_pairs:
            period_quanta = int(self.rand_gen_loc.choice(self.wifi_period_quanta_candidates))
            period_s = period_quanta * base
            period_slots = int(round(period_s / self.slot_time))
            if abs(period_s / self.slot_time - period_slots) > 1e-9:
                raise ValueError("Sampled WiFi period is not an integer multiple of slot_time.")
            self.pair_wifi_period_slots[k] = period_slots
            tx_slots = int(self.rand_gen_loc.choice(self.wifi_tx_quanta_candidates))
            if tx_slots > period_slots:
                raise ValueError("WiFi minimum transmission duration exceeds sampled WiFi period.")
            self.pair_wifi_tx_slots[k] = tx_slots
            low = int(self.pair_start_time_slot[k])
            high = int(low + period_slots - tx_slots)
            self.pair_wifi_anchor_slot[k] = int(self.rand_gen_loc.integers(low=low, high=high + 1))

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
        self.ble_ce_required_s = float(math.ceil(self.ble_ce_required_s / self.slot_time) * self.slot_time)

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
                    ce_low_slots = int(math.ceil(ce_required / self.slot_time))
                    ce_high_slots = int(math.floor(ce_high / self.slot_time))
                    if ce_low_slots > ce_high_slots:
                        self.pair_ble_ce_feasible[k] = False
                        self.pair_ble_ce_slots[k] = 0
                    else:
                        self.pair_ble_ce_slots[k] = int(
                            self.rand_gen_loc.integers(low=ce_low_slots, high=ce_high_slots + 1)
                        )
            else:
                self.pair_ble_ce_slots[k] = 0

            # 锚点按用户自己的 CI 周期随机
            if self.pair_ble_ci_slots[k] > 0:
                low = int(self.pair_start_time_slot[k])
                high = int(low + self.pair_ble_ci_slots[k] - self.pair_ble_ce_slots[k])
                # 中文：anchor 以 slot 索引表示，天然是 slot_time 的整数倍。
                self.pair_ble_anchor_slot[k] = int(self.rand_gen_loc.integers(low=low, high=high + 1))
                if self.ble_channel_mode == "per_ce":
                    self._assign_ble_ce_channels(int(k))

    def _assign_ble_ce_channels(self, pair_id):
        if self.ble_channel_mode != "per_ce":
            return
        ci_slots = int(self.pair_ble_ci_slots[pair_id])
        if ci_slots <= 0:
            self.pair_ble_ce_channels[pair_id] = np.zeros(0, dtype=int)
            return

        macrocycle_slots = int(self.compute_macrocycle_slots())
        event_count = max(1, macrocycle_slots // ci_slots) if macrocycle_slots > 0 else 1
        self.pair_ble_ce_channels[pair_id] = self.rand_gen_loc.choice(self.ble_data_channel_indices, size=event_count)

    def set_ble_ce_channel_map(self, channel_map):
        if self.ble_channel_mode != "per_ce":
            return
        if self.pair_ble_ce_channels is None:
            self.pair_ble_ce_channels = {}
        if self._manual_ble_ce_channel_pairs is None:
            self._manual_ble_ce_channel_pairs = set()

        for pair_id, channels in channel_map.items():
            pair_id = int(pair_id)
            if self.pair_radio_type[pair_id] != self.RADIO_BLE:
                raise ValueError("BLE CE channel map can only be set for BLE pairs.")
            ci_slots = int(self.pair_ble_ci_slots[pair_id])
            macrocycle_slots = int(self.compute_macrocycle_slots())
            event_count = max(1, macrocycle_slots // ci_slots) if macrocycle_slots > 0 else 1
            channel_arr = np.asarray(channels, dtype=int).ravel()
            if channel_arr.size != event_count:
                raise ValueError("BLE CE channel map length must match event count.")
            if np.any(channel_arr < 0) or np.any(channel_arr >= self.ble_channel_count):
                raise ValueError("BLE CE channel map contains invalid channel indices.")
            self.pair_ble_ce_channels[pair_id] = channel_arr.copy()
            self._manual_ble_ce_channel_pairs.add(pair_id)

    def build_slot_occupancy_mask(self, Z):
        # 中文：构建“用户-时隙占用窗口掩码”。
        # WiFi / BLE 都在统一 base slot 网格上扩展成真实连续占用窗口。
        Z = int(Z)
        K = int(self.n_pair)
        mask = np.zeros((K, Z), dtype=bool)
        for k in range(K):
            if self.pair_radio_type[k] == self.RADIO_WIFI:
                period_slots = int(self.pair_wifi_period_slots[k])
                tx_slots = int(self.pair_wifi_tx_slots[k])
                if period_slots <= 0 or tx_slots <= 0:
                    continue
                anchor = int(self.pair_wifi_anchor_slot[k] % period_slots)
                if Z > 0 and anchor >= Z:
                    anchor = int(anchor % Z)
                z = anchor
                while z < Z:
                    end_z = min(z + tx_slots, Z)
                    mask[k, z:end_z] = True
                    z += period_slots
            else:
                if not self.pair_ble_ce_feasible[k]:
                    continue
                ci_slots = int(self.pair_ble_ci_slots[k])
                ce_slots = int(self.pair_ble_ce_slots[k])
                if ci_slots <= 0 or ce_slots <= 0:
                    continue
                anchor = int(self.pair_ble_anchor_slot[k] % ci_slots)
                if Z > 0 and anchor >= Z:
                    anchor = int(anchor % Z)
                z = anchor
                while z < Z:
                    end_z = min(z + ce_slots, Z)
                    mask[k, z:end_z] = True
                    z += ci_slots
        return mask

    def build_slot_compatibility_mask(self, Z):
        return self.build_slot_occupancy_mask(Z)

    def get_pair_period_slots(self):
        period_slots = np.zeros(self.n_pair, dtype=int)
        wifi_mask = self.pair_radio_type == self.RADIO_WIFI
        ble_mask = self.pair_radio_type == self.RADIO_BLE
        period_slots[wifi_mask] = self.pair_wifi_period_slots[wifi_mask]
        period_slots[ble_mask] = self.pair_ble_ci_slots[ble_mask]
        return period_slots

    def get_pair_width_slots(self):
        width_slots = np.zeros(self.n_pair, dtype=int)
        wifi_mask = self.pair_radio_type == self.RADIO_WIFI
        ble_mask = self.pair_radio_type == self.RADIO_BLE
        width_slots[wifi_mask] = self.pair_wifi_tx_slots[wifi_mask]
        width_slots[ble_mask] = self.pair_ble_ce_slots[ble_mask]
        return width_slots

    def get_active_period_slots(self):
        periods = self.get_pair_period_slots()
        if np.any(self.pair_radio_type == self.RADIO_BLE):
            feasible_ble = np.logical_or(
                self.pair_radio_type != self.RADIO_BLE,
                self.pair_ble_ce_feasible,
            )
            periods = periods[feasible_ble]
        return periods[periods > 0]

    def compute_macrocycle_slots(self):
        periods = self.get_active_period_slots()
        if periods.size == 0:
            return 0
        return int(np.lcm.reduce(periods.astype(np.int64)))

    def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
        macrocycle_slots = int(macrocycle_slots)
        occ = np.zeros(macrocycle_slots, dtype=bool)
        if macrocycle_slots <= 0:
            return occ

        pair_id = int(pair_id)
        if self.pair_radio_type[pair_id] == self.RADIO_BLE and not self.pair_ble_ce_feasible[pair_id]:
            return occ

        period_slots = int(self.get_pair_period_slots()[pair_id])
        width_slots = int(self.get_pair_width_slots()[pair_id])
        if period_slots <= 0 or width_slots <= 0:
            return occ

        start = int(start_slot % period_slots)
        z = start
        while z < macrocycle_slots:
            for offset in range(width_slots):
                occ[(z + offset) % macrocycle_slots] = True
            z += period_slots
        return occ

    def get_wifi_channel_center_mhz(self, wifi_idx):
        wifi_idx = int(wifi_idx)
        if wifi_idx in {1, 6, 11}:
            wifi_idx -= 1
        if wifi_idx not in {0, 5, 10}:
            raise ValueError("WiFi channel index must be one of {0, 1, 5, 6, 10, 11}.")
        return 2412.0 + 5.0 * float(wifi_idx)

    def get_ble_data_channel_center_mhz(self, ble_idx):
        ble_idx = int(ble_idx)
        if ble_idx < 0 or ble_idx >= 37:
            raise ValueError("BLE data channel index must be in [0, 36].")
        if ble_idx <= 10:
            return 2404.0 + 2.0 * float(ble_idx)
        return 2428.0 + 2.0 * float(ble_idx - 11)

    def _get_wifi_channel_range_hz(self, wifi_idx):
        center = self.get_wifi_channel_center_mhz(wifi_idx) * 1e6
        half_bw = self.wifi_channel_bandwidth_hz / 2.0
        return center - half_bw, center + half_bw

    def _get_ble_channel_range_hz(self, ble_idx):
        center = self.get_ble_data_channel_center_mhz(ble_idx) * 1e6
        half_bw = self.ble_channel_bandwidth_hz / 2.0
        return center - half_bw, center + half_bw

    def _get_pair_link_range_hz(self, pair_idx):
        if self.pair_radio_type[pair_idx] == self.RADIO_WIFI:
            return self._get_wifi_channel_range_hz(self.pair_channel[pair_idx])
        return self._get_ble_channel_range_hz(self.pair_channel[pair_idx])

    def _get_pair_link_range_hz_for_channel(self, pair_idx, channel):
        if self.pair_radio_type[pair_idx] == self.RADIO_WIFI:
            return self._get_wifi_channel_range_hz(int(channel))
        return self._get_ble_channel_range_hz(int(channel))

    def get_available_ble_channels_for_start_slot(self, wifi_pair_ids, wifi_start_slots, start_slot):
        wifi_pair_ids = np.asarray(wifi_pair_ids, dtype=int).ravel()
        wifi_start_slots = np.asarray(wifi_start_slots, dtype=int).ravel()
        if wifi_pair_ids.shape != wifi_start_slots.shape:
            raise ValueError("wifi_pair_ids and wifi_start_slots must have the same shape.")

        blocked = np.zeros(self.ble_channel_count, dtype=bool)
        macrocycle_slots = max(int(self.compute_macrocycle_slots()), 1)
        slot_idx = int(start_slot) % macrocycle_slots
        for pair_id, pair_start in zip(wifi_pair_ids, wifi_start_slots):
            pair_id = int(pair_id)
            if self.pair_radio_type[pair_id] != self.RADIO_WIFI:
                continue
            occ = self.expand_pair_occupancy(pair_id, int(pair_start), macrocycle_slots)
            if not occ[slot_idx]:
                continue
            w0, w1 = self._get_wifi_channel_range_hz(int(self.pair_channel[pair_id]))
            for ble_idx in range(self.ble_channel_count):
                b0, b1 = self._get_ble_channel_range_hz(int(ble_idx))
                if env._is_range_overlap(w0, w1, b0, b1):
                    blocked[ble_idx] = True
        return np.where(~blocked)[0]

    def get_ble_start_slot_capacity(self, wifi_pair_ids, wifi_start_slots, start_slot):
        return int(
            self.get_available_ble_channels_for_start_slot(
                wifi_pair_ids=wifi_pair_ids,
                wifi_start_slots=wifi_start_slots,
                start_slot=start_slot,
            ).size
        )

    @staticmethod
    def compute_ble_no_collision_probability(c, n):
        c = int(c)
        n = int(n)
        if n <= 1:
            return 1.0
        if c <= 0:
            return 0.0
        return float((1.0 - 1.0 / c) ** (n - 1))

    def get_pair_channel_for_slot(self, pair_id, slot, start_slot):
        pair_id = int(pair_id)
        slot = int(slot)
        period_slots = int(self.get_pair_period_slots()[pair_id])
        if period_slots <= 0:
            return None
        macrocycle_slots = int(self.compute_macrocycle_slots())
        if macrocycle_slots <= 0:
            return None
        slot_idx = int(slot % macrocycle_slots)
        instances = self.expand_pair_event_instances(pair_id, macrocycle_slots, start_slot=start_slot)
        for inst in instances:
            for seg_start, seg_end in inst.get("wrapped_slot_ranges", [inst["slot_range"]]):
                if int(seg_start) <= slot_idx < int(seg_end):
                    return int(inst["channel"])
        return int(self.pair_channel[pair_id])

    def is_slot_channel_conflict(self, pair_a, start_a, pair_b, start_b, slot):
        ch_a = self.get_pair_channel_for_slot(pair_a, slot, start_a)
        ch_b = self.get_pair_channel_for_slot(pair_b, slot, start_b)
        if ch_a is None or ch_b is None:
            return False
        a0, a1 = self._get_pair_link_range_hz_for_channel(pair_a, ch_a)
        b0, b1 = self._get_pair_link_range_hz_for_channel(pair_b, ch_b)
        return env._is_range_overlap(a0, a1, b0, b1)

    def expand_pair_event_instances(self, pair_id, macrocycle_slots, start_slot=None):
        pair_id = int(pair_id)
        macrocycle_slots = int(macrocycle_slots)
        instances = []
        if macrocycle_slots <= 0:
            return instances
        if self.pair_radio_type[pair_id] == self.RADIO_BLE and not self.pair_ble_ce_feasible[pair_id]:
            return instances

        period_slots = int(self.get_pair_period_slots()[pair_id])
        width_slots = int(self.get_pair_width_slots()[pair_id])
        if period_slots <= 0 or width_slots <= 0:
            return instances

        if start_slot is None:
            if self.pair_radio_type[pair_id] == self.RADIO_WIFI:
                start = int(self.pair_wifi_anchor_slot[pair_id] % period_slots)
            else:
                start = int(self.pair_ble_anchor_slot[pair_id] % period_slots)
        else:
            start = int(start_slot % period_slots)

        event_idx = 0
        z = start
        while z < macrocycle_slots:
            end_z = z + width_slots
            wrapped_ranges = []
            remaining = width_slots
            cursor = z
            while remaining > 0:
                seg_start = cursor % macrocycle_slots
                seg_len = min(remaining, macrocycle_slots - seg_start)
                wrapped_ranges.append((int(seg_start), int(seg_start + seg_len)))
                cursor += seg_len
                remaining -= seg_len
            if self.pair_radio_type[pair_id] == self.RADIO_BLE and self.ble_channel_mode == "per_ce":
                ce_channels = self.pair_ble_ce_channels.get(pair_id, np.zeros(0, dtype=int))
                if ce_channels.size == 0:
                    channel = int(self.pair_channel[pair_id])
                else:
                    channel = int(ce_channels[min(event_idx, ce_channels.size - 1)])
            else:
                channel = int(self.pair_channel[pair_id])
            low_hz, high_hz = self._get_pair_link_range_hz_for_channel(pair_id, channel)
            instances.append(
                {
                    "pair_id": pair_id,
                    "event_index": event_idx,
                    "channel": channel,
                    "slot_range": (int(z), int(end_z)),
                    "wrapped_slot_ranges": wrapped_ranges,
                    "freq_range_hz": (float(low_hz), float(high_hz)),
                }
            )
            event_idx += 1
            z += period_slots
        return instances

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

    def build_pair_conflict_matrix(self):
        conflict = self._build_radio_interference_constraints(self.n_pair).astype(bool).tocsr()
        conflict.setdiag(False)
        conflict.eliminate_zeros()
        return conflict.toarray().astype(bool)

    def get_macrocycle_conflict_state(self):
        s_gain, q_conflict, h_max = self.generate_S_Q_hmax(real=False)
        return s_gain.tocsr(), q_conflict.tocsr(), np.asarray(h_max, dtype=float).ravel()

    def resample_ble_channels(self, pair_ids):
        pair_ids = np.asarray(pair_ids, dtype=int).ravel()
        if pair_ids.size == 0:
            return
        ble_ids = pair_ids[self.pair_radio_type[pair_ids] == self.RADIO_BLE]
        if ble_ids.size == 0:
            return
        if self.ble_channel_mode == "per_ce":
            for pair_id in ble_ids:
                if self._manual_ble_ce_channel_pairs is not None and int(pair_id) in self._manual_ble_ce_channel_pairs:
                    continue
                prev = self.pair_ble_ce_channels.get(int(pair_id), np.zeros(0, dtype=int))
                if prev.size == 0:
                    self._assign_ble_ce_channels(int(pair_id))
                    continue
                new_channels = self.rand_gen_loc.choice(self.ble_data_channel_indices, size=prev.size)
                if self.ble_data_channel_indices.size > 1 and np.array_equal(new_channels, prev):
                    new_channels = self.ble_data_channel_indices[
                        (np.searchsorted(self.ble_data_channel_indices, new_channels) + 1) % self.ble_data_channel_indices.size
                    ]
                self.pair_ble_ce_channels[int(pair_id)] = new_channels
            return
        self.pair_channel[ble_ids] = self.rand_gen_loc.choice(self.ble_data_channel_indices, size=int(ble_ids.size))
        self.device_radio_channel = self.pair_channel
        self.user_radio_channel = self.pair_channel

    def _get_random_dir(self):
        dd = self.rand_gen_mob.standard_normal(2)
        return dd / np.linalg.norm(dd)

    def _compute_min_sinr(self):
        min_sinr_db = np.array(
            [
                env.bisection_method(self.pair_packet_bits[k], self.pair_bandwidth_hz[k], self.slot_time, self.max_err)
                for k in range(self.n_pair)
            ],
            dtype=float,
        )
        self.pair_min_sinr = env.db_to_dec(min_sinr_db)
        self.device_min_sinr = self.pair_min_sinr
        self.user_min_sinr = self.pair_min_sinr
        self.min_sinr = self.pair_min_sinr
        return self.pair_min_sinr

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
        bandwidth_hz = np.asarray(B, dtype=float)
        if np.any(bandwidth_hz <= 0):
            raise ValueError("bandwidth must be positive.")
        return -174.0 + 10.0 * np.log10(bandwidth_hz) + cls.NOISEFIGURE

    @staticmethod
    def fre_dis_to_loss_dB(fre_Hz, dis):
        L = 20.0 * math.log10(fre_Hz / 1e6) + 16 - 28
        return L + 28 * np.log10(dis + 1)

    @staticmethod
    def db_to_dec(snr_db):
        return np.power(10.0, np.asarray(snr_db, dtype=float) / 10.0)

    @staticmethod
    def dec_to_db(snr_dec):
        return 10.0 * np.log10(np.asarray(snr_dec, dtype=float))

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
        fa = env.err(a, L, B, T, max_err)
        fb = env.err(b, L, B, T, max_err)
        while fa < 0 and a > -120.0:
            a -= 5.0
            fa = env.err(a, L, B, T, max_err)
        while fb > 0 and b < 120.0:
            b += 5.0
            fb = env.err(b, L, B, T, max_err)
        if fa * fb >= 0:
            print("Bisection method fails.")
            return a if fa < 0 else b

        while (fa - fb) > tol:
            midpoint = (a + b) / 2
            fm = env.err(midpoint, L, B, T, max_err)
            if fm == 0:
                return midpoint
            if fa * fm < 0:
                b = midpoint
                fb = fm
            else:
                a = midpoint
                fa = fm
        return (a + b) / 2

    def _compute_txp(self):
        dis = np.linalg.norm(self.pair_tx_locs - self.pair_rx_locs, axis=1)
        gain = -env.fre_dis_to_loss_dB(self.fre_Hz, dis)
        noise_dbm = np.asarray(self.bandwidth_txpr_to_noise_dBm(self.pair_bandwidth_hz), dtype=float)
        if noise_dbm.ndim == 0:
            noise_dbm = np.full(self.n_pair, float(noise_dbm), dtype=float)
        t = env.dec_to_db(self._compute_min_sinr()) - (gain - noise_dbm)
        return np.reshape(t + env.dec_to_db(self.txp_offset), (self.n_pair, -1))

    def _compute_state(self):
        dis = scipy.spatial.distance.cdist(self.pair_tx_locs, self.pair_rx_locs)
        self.loss = env.fre_dis_to_loss_dB(self.fre_Hz, dis)
        noise_dbm = np.asarray(self.bandwidth_txpr_to_noise_dBm(self.pair_bandwidth_hz), dtype=float)
        if noise_dbm.ndim == 0:
            noise_dbm = np.full(self.n_pair, float(noise_dbm), dtype=float)
        noise_dbm = noise_dbm.reshape(self.n_pair, -1)
        rxpr_db = self._compute_txp() - self.loss - noise_dbm
        rxpr_hi = 10 ** (rxpr_db / 10.0)
        rxpr_hi[rxpr_hi < self.min_s_n_ratio] = 0.0
        return scipy.sparse.csr_matrix(rxpr_hi)

    def _compute_state_real(self):
        dis = scipy.spatial.distance.cdist(self.pair_tx_locs, self.pair_rx_locs)
        self.loss = env.fre_dis_to_loss_dB(self.fre_Hz, dis)
        noise_dbm = np.asarray(self.bandwidth_txpr_to_noise_dBm(self.pair_bandwidth_hz), dtype=float)
        if noise_dbm.ndim == 0:
            noise_dbm = np.full(self.n_pair, float(noise_dbm), dtype=float)
        noise_dbm = noise_dbm.reshape(self.n_pair, -1)
        rxpr_db = self._compute_txp() - self.loss - noise_dbm
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
            bler[k] = env.polyanskiy_model(
                sinr[k],
                self.user_packet_bits[k],
                self.user_bandwidth_hz[k],
                self.slot_time,
            )
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
