import math
import numpy as np

from sim_src.alg import alg_interface
from sim_src.util import STATS_OBJECT


class binary_search_relaxation(alg_interface,STATS_OBJECT):
    def __init__(self):
        self.feasibility_check_alg = None
        self.strategy = "joint"
        self.pair_radio_type = None
        self.wifi_radio_id = 0
        self.ble_radio_id = 1
        self.force_lower_bound = False
        self.force_full_bound = False
        self.user_priority = None  # [PRIO-BS] 新增优先级透传字段：会传给 solver.rounding(...)。
        self.slot_mask_builder = None  # [BLE-TIMING] 可选函数：输入 Z/state，输出 (K,Z) 时隙可用掩码。
        self.max_slot_cap = None
        self.last_partial_schedule = None
        self.last_stage_results = None

    @staticmethod
    def split_pair_indices_by_radio_type(pair_radio_type, wifi_id=0, ble_id=1):
        pair_radio_type = np.asarray(pair_radio_type, dtype=int).ravel()
        return np.where(pair_radio_type == int(wifi_id))[0], np.where(pair_radio_type == int(ble_id))[0]

    @staticmethod
    def _slice_state_for_pair_ids(state, pair_ids):
        pair_ids = np.asarray(pair_ids, dtype=int).ravel()
        s_gain = state[0][pair_ids][:, pair_ids].tocsr()
        q_conflict = state[1][pair_ids][:, pair_ids].tocsr()
        h_max = np.asarray(state[2], dtype=float).ravel()[pair_ids]
        return s_gain, q_conflict, h_max
    def set_bounds(self,state):
        if self.force_lower_bound:
            nnz_per_row = np.diff(state[1].indptr)
            lb = np.max(nnz_per_row)+1
            return lb,lb
        if self.force_full_bound:
            return 1, state[0].shape[0]

        S_gain = state[0].copy()
        S = S_gain + S_gain.transpose()
        S.setdiag(0)
        S.sort_indices()
        nnz_per_row = np.diff(S.indptr)
        ub = np.max(nnz_per_row)+1
        nnz_per_row = np.diff(state[1].indptr)
        lb = np.max(nnz_per_row)+1
        lb = max(int(lb), 2)
        ub = max(int(ub), 2)
        return lb, ub

    def _run_stage_for_pair_ids(self, state, pair_ids):
        pair_ids = np.asarray(pair_ids, dtype=int).ravel()
        if pair_ids.size == 0:
            return {
                "pair_ids": pair_ids,
                "z_vec": np.zeros(0, dtype=int),
                "z_fin": 0,
                "remainder": 0,
                "partial": {"slot_cap_hit": False, "scheduled_pair_ids": [], "unscheduled_pair_ids": []},
            }
        if pair_ids.size == 1:
            return {
                "pair_ids": pair_ids,
                "z_vec": np.array([0], dtype=int),
                "z_fin": 1,
                "remainder": 0,
                "partial": {"slot_cap_hit": False, "scheduled_pair_ids": [0], "unscheduled_pair_ids": []},
            }

        sliced_state = self._slice_state_for_pair_ids(state, pair_ids)
        prev_priority = self.user_priority
        prev_slot_mask_builder = self.slot_mask_builder
        self.user_priority = None if prev_priority is None else np.asarray(prev_priority, dtype=float)[pair_ids]
        if prev_slot_mask_builder is None:
            self.slot_mask_builder = None
        else:
            self.slot_mask_builder = lambda Z, _state, idx=pair_ids, builder=prev_slot_mask_builder, full_state=state: np.asarray(
                builder(Z, full_state)
            )[idx]
        try:
            left, right = self.set_bounds(sliced_state)
            if self.max_slot_cap is not None:
                cap = int(self.max_slot_cap)
                left = min(left, cap)
                right = min(right, cap)
            z_fin, z_vec, remainder, _ = self.search(left, right, sliced_state)
            partial = self.last_partial_schedule or self._build_partial_schedule(z_vec, z_fin, remainder)
        finally:
            self.user_priority = prev_priority
            self.slot_mask_builder = prev_slot_mask_builder

        return {
            "pair_ids": pair_ids,
            "z_vec": np.asarray(z_vec, dtype=int).ravel(),
            "z_fin": int(z_fin),
            "remainder": int(remainder),
            "partial": partial,
        }

    def _run_wifi_first(self, state):
        if self.pair_radio_type is None:
            raise ValueError("pair_radio_type must be set when strategy='wifi_first'.")
        wifi_idx, ble_idx = self.split_pair_indices_by_radio_type(
            self.pair_radio_type,
            wifi_id=self.wifi_radio_id,
            ble_id=self.ble_radio_id,
        )
        wifi_result = self._run_stage_for_pair_ids(state, wifi_idx)
        ble_result = self._run_stage_for_pair_ids(state, ble_idx)

        z_vec = np.full(state[0].shape[0], -1, dtype=int)
        z_vec[wifi_result["pair_ids"]] = wifi_result["z_vec"]
        z_vec[ble_result["pair_ids"]] = ble_result["z_vec"]
        self.last_stage_results = {"wifi": wifi_result, "ble": ble_result}
        self.last_partial_schedule = {
            "slot_cap_hit": bool(wifi_result["partial"]["slot_cap_hit"] or ble_result["partial"]["slot_cap_hit"]),
            "scheduled_pair_ids": [int(wifi_result["pair_ids"][i]) for i in wifi_result["partial"]["scheduled_pair_ids"]]
            + [int(ble_result["pair_ids"][i]) for i in ble_result["partial"]["scheduled_pair_ids"]],
            "unscheduled_pair_ids": [int(wifi_result["pair_ids"][i]) for i in wifi_result["partial"]["unscheduled_pair_ids"]]
            + [int(ble_result["pair_ids"][i]) for i in ble_result["partial"]["unscheduled_pair_ids"]],
        }
        return (
            z_vec,
            max(int(wifi_result["z_fin"]), int(ble_result["z_fin"])),
            int(wifi_result["remainder"] + ble_result["remainder"]),
        )

    def run(self,state):
        if self.strategy == "wifi_first":
            return self._run_wifi_first(state)

        bd_tic = self._get_tic()
        left, right = self.set_bounds(state)
        if self.max_slot_cap is not None:
            cap = int(self.max_slot_cap)
            left = min(left, cap)
            right = min(right, cap)
        tim = self._get_tim(bd_tic)
        self._add_np_log("bs_set_bounds",0,np.array([left,right,tim]))

        bs_tic = self._get_tic()
        Z, z_vec, rem, it = self.search(left, right, state)
        tim = self._get_tim(bs_tic)
        self._add_np_log("bs_search",0,np.array([left,right,Z,rem,it,tim]))

        self.last_partial_schedule = self._build_partial_schedule(z_vec, Z, rem)
        self.last_stage_results = None
        return z_vec, Z, rem

    def _build_partial_schedule(self, z_vec, Z, rem):
        z_vec = np.asarray(z_vec, dtype=int).ravel()
        K = int(z_vec.shape[0])
        rem = int(rem)
        scheduled_count = max(0, K - rem)
        scheduled_pair_ids = np.arange(scheduled_count, dtype=int).tolist()
        unscheduled_pair_ids = np.arange(scheduled_count, K, dtype=int).tolist()
        return {
            "slot_cap_hit": bool(self.max_slot_cap is not None and int(Z) >= int(self.max_slot_cap) and rem > 0),
            "scheduled_pair_ids": scheduled_pair_ids,
            "unscheduled_pair_ids": unscheduled_pair_ids,
        }

    def search(self, left, right, state):
        it = 0
        to_break=False
        while True:
            mid = math.floor(float(left+right)/2.)
            bs_slv_tic = self._get_tic()
            f, gX = self.feasibility_check_alg.run_with_state(it,mid,state)
            bs_slv_tim = self._get_tim(bs_slv_tic)
            bs_rnd_tic = self._get_tic()
            slot_mask = self.slot_mask_builder(mid, state) if self.slot_mask_builder is not None else None
            # [PRIO-BS] 在二分可行性检查中启用优先级分配：把 user_priority 传给 rounding。
            z_vec, Z, rem = self.feasibility_check_alg.rounding(mid,gX,state,user_priority=self.user_priority,slot_mask=slot_mask)
            bs_rnd_tim = self._get_tim(bs_rnd_tic)
            self._add_np_log("bs_search_per_it",it,np.array([left,right,mid,Z,rem,bs_slv_tim,bs_rnd_tim]))
            it += 1
            if left < right and rem > 0:
                left = mid+1
            elif left + 1 < right and rem == 0:
                right = mid
            elif left + 1 == right and rem == 0:
                to_break = True
            elif left >= right and rem == 0:
                to_break = True
            elif left >= right and rem > 0:
                if self.max_slot_cap is not None and int(right) >= int(self.max_slot_cap):
                    to_break = True
                else:
                    left+=1
                    right+=1

            self._printalltime(left,right,mid,Z,rem,"++++++++++++++++++++")
            if to_break:
                break
        return Z, z_vec, rem, it
