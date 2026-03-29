from __future__ import annotations


def medium_joint_instance() -> dict:
    return {
        "macrocycle_slots": 64,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "ga": {
            "population_size": 24,
            "generations": 20,
            "seed": 17,
        },
        "tasks": [
            {
                "task_id": 0,
                "radio": "wifi",
                "payload_bytes": 1200,
                "release_slot": 0,
                "deadline_slot": 20,
                "preferred_channel": 0,
                "wifi_tx_slots": 4,
                "max_offsets": 2,
            },
            {
                "task_id": 1,
                "radio": "wifi",
                "payload_bytes": 900,
                "release_slot": 8,
                "deadline_slot": 40,
                "preferred_channel": 5,
                "wifi_tx_slots": 4,
                "max_offsets": 2,
            },
            {
                "task_id": 2,
                "radio": "ble",
                "payload_bytes": 247,
                "release_slot": 0,
                "deadline_slot": 48,
                "preferred_channel": 8,
                "ble_ce_slots": 1,
                "ble_ci_slots_options": [8],
                "ble_num_events": 3,
                "ble_pattern_count": 2,
                "max_offsets": 2,
            },
        ],
    }
