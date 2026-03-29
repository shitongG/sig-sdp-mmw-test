"""Random mixed WiFi/BLE task generation for the isolated joint scheduler."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .joint_wifi_ble_model import JointTaskSpec

OFFICE_CELL_EDGE_M = 7.0
WIFI_CHANNELS = (0, 5, 10)
BLE_CHANNELS = tuple(range(37))
RADIO_PROB = (0.1, 0.9)
SLOT_TIME_S = 1.25e-3
WIFI_TX_SLOT_CHOICES = tuple(range(4, 9))
WIFI_PERIOD_SLOT_CHOICES = (16, 32)
BLE_CI_SLOT_CHOICES = (8, 16, 32, 64)
BLE_CE_SLOT_CHOICES = (1, 2, 3, 4, 5, 6)
WIFI_PAYLOAD_RANGE_BYTES = (512, 1500)
BLE_PAYLOAD_RANGE_BYTES = (1, 247)


def compute_main_style_pair_count(cell_size: int, pair_density: float) -> int:
    office_area_m2 = OFFICE_CELL_EDGE_M ** 2
    return int((int(cell_size) ** 2) * float(pair_density) * office_area_m2)



def _resolve_seed(config: Mapping[str, Any]) -> int:
    seed = config.get("seed")
    if seed is None:
        return int(time.time())
    return int(seed)



def generate_joint_tasks_from_main_style_config(config: Mapping[str, Any]) -> list[JointTaskSpec]:
    cell_size = int(config.get("cell_size", 1))
    pair_density = float(config.get("pair_density", 0.05))
    seed = _resolve_seed(config)
    rng = np.random.default_rng(seed)

    n_pair = compute_main_style_pair_count(cell_size, pair_density)
    if n_pair <= 0:
        return []

    radio_types = rng.choice(np.array(["wifi", "ble"], dtype=object), size=n_pair, p=np.asarray(RADIO_PROB, dtype=float))
    if n_pair >= 2:
        radio_types[0] = "wifi"
        radio_types[1] = "ble"

    tasks: list[JointTaskSpec] = []
    macrocycle_slots = max(int(config.get("macrocycle_slots", 64)), 64)
    for task_id, radio in enumerate(radio_types.tolist()):
        if radio == "wifi":
            tx_slots = int(rng.choice(np.asarray(WIFI_TX_SLOT_CHOICES, dtype=int)))
            period_slots = int(rng.choice(np.asarray(WIFI_PERIOD_SLOT_CHOICES, dtype=int)))
            latest_release = max(0, macrocycle_slots - tx_slots)
            release_slot = int(rng.integers(0, latest_release + 1))
            deadline_slot = int(min(macrocycle_slots - 1, release_slot + period_slots + tx_slots + int(rng.integers(0, 9))))
            preferred_channel = int(rng.choice(np.asarray(WIFI_CHANNELS, dtype=int)))
            payload_bytes = int(rng.integers(WIFI_PAYLOAD_RANGE_BYTES[0], WIFI_PAYLOAD_RANGE_BYTES[1] + 1))
            tasks.append(
                JointTaskSpec(
                    task_id=task_id,
                    radio="wifi",
                    payload_bytes=payload_bytes,
                    release_slot=release_slot,
                    deadline_slot=max(deadline_slot, release_slot + tx_slots - 1),
                    preferred_channel=preferred_channel,
                    repetitions=1,
                    wifi_tx_slots=tx_slots,
                    wifi_period_slots=period_slots,
                    max_offsets=1,
                )
            )
            continue

        ci_slots = int(rng.choice(np.asarray(BLE_CI_SLOT_CHOICES, dtype=int)))
        ce_slots = int(rng.choice(np.asarray([value for value in BLE_CE_SLOT_CHOICES if value <= ci_slots], dtype=int)))
        max_num_events = max(1, min(3, (macrocycle_slots - ce_slots) // max(ci_slots, 1) + 1))
        num_events = int(rng.integers(1, max_num_events + 1))
        required_span = (num_events - 1) * ci_slots + ce_slots
        latest_release = max(0, macrocycle_slots - required_span)
        release_slot = int(rng.integers(0, latest_release + 1))
        slack = int(rng.integers(0, max(1, ci_slots + 1)))
        deadline_slot = int(min(macrocycle_slots - 1, release_slot + required_span + slack - 1))
        preferred_channel = int(rng.choice(np.asarray(BLE_CHANNELS, dtype=int)))
        payload_bytes = int(rng.integers(BLE_PAYLOAD_RANGE_BYTES[0], BLE_PAYLOAD_RANGE_BYTES[1] + 1))
        tasks.append(
            JointTaskSpec(
                task_id=task_id,
                radio="ble",
                payload_bytes=payload_bytes,
                release_slot=release_slot,
                deadline_slot=max(deadline_slot, release_slot + ce_slots - 1),
                preferred_channel=preferred_channel,
                repetitions=num_events,
                ble_ce_slots=ce_slots,
                ble_ci_slots_options=(ci_slots,),
                ble_num_events=num_events,
                ble_pattern_count=1,
                max_offsets=1,
            )
        )

    return tasks
