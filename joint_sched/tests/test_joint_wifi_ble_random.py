from joint_sched.joint_wifi_ble_random import generate_joint_tasks_from_main_style_config



def test_random_joint_task_generation_matches_config_shape_and_is_deterministic():
    config = {
        "cell_size": 1,
        "pair_density": 0.1,
        "seed": 123,
    }

    left = generate_joint_tasks_from_main_style_config(config)
    right = generate_joint_tasks_from_main_style_config(config)

    assert left
    assert any(task.radio == "wifi" for task in left)
    assert any(task.radio == "ble" for task in left)
    assert len(left) == len(right)
    assert left == right



def test_random_joint_task_generation_respects_expected_pair_count_formula():
    config = {
        "cell_size": 1,
        "pair_density": 1,
        "seed": 7,
    }

    tasks = generate_joint_tasks_from_main_style_config(config)

    assert len(tasks) == 49
