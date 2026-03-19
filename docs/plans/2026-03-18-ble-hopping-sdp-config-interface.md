# BLE Hopping SDP Config Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a JSON config-file interface to `ble_macrocycle_hopping_sdp.py` so users can change all scheduling parameters without editing Python source, with parameter meanings documented in Chinese.

**Architecture:** Keep the existing solver, event-block, overlap, and plotting pipeline intact. Replace the hardcoded demo-only entry path with a config loader that parses a default or user-specified JSON file into `PairConfig`, `HoppingPattern`, `pair_weight`, plotting, and solver settings, then routes those values into the existing `main()` flow.

**Tech Stack:** Python, `argparse`, `json`, `pathlib`, `numpy`, `cvxpy`, `matplotlib`, standard library `unittest`

---

### Task 1: Add a failing test for config loading and pair-weight parsing

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Modify: `ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
import json
import tempfile


    def test_load_config_from_json_parses_pairs_patterns_and_weights(self):
        config_payload = {
            "num_channels": 20,
            "hard_collision_threshold": 1.5,
            "plot_title": "测试标题",
            "output_path": "out.png",
            "pair_configs": [
                {
                    "pair_id": 0,
                    "release_time": 1,
                    "deadline": 10,
                    "connect_interval": 3,
                    "event_duration": 1,
                    "num_events": 3,
                }
            ],
            "pattern_dict": {
                "0": [
                    {"pattern_id": 0, "start_channel": 2, "hop_increment": 5}
                ]
            },
            "pair_weight": {
                "0-1": 1.2
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")

            config = MODULE.load_config_from_json(config_path)

        self.assertEqual(config["num_channels"], 20)
        self.assertEqual(config["hard_collision_threshold"], 1.5)
        self.assertEqual(config["plot_title"], "测试标题")
        self.assertEqual(config["output_path"], "out.png")
        self.assertEqual(config["pair_configs"][0].pair_id, 0)
        self.assertEqual(config["pattern_dict"][0][0].hop_increment, 5)
        self.assertEqual(config["pair_weight"][(0, 1)], 1.2)
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python -m unittest tests.test_ble_macrocycle_hopping_sdp.EventBlockExpansionTest.test_load_config_from_json_parses_pairs_patterns_and_weights -v`

Expected: FAIL with `AttributeError` because `load_config_from_json` does not exist yet.

**Step 3: Write minimal implementation**

```python
def parse_pair_weight_map(raw_pair_weight: dict[str, float]) -> Dict[Tuple[int, int], float]:
    parsed = {}
    for key, value in raw_pair_weight.items():
        left_text, right_text = key.split("-")
        left = int(left_text)
        right = int(right_text)
        parsed[(min(left, right), max(left, right))] = float(value)
    return parsed


def load_config_from_json(config_path: Path) -> dict:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    pair_configs = [PairConfig(**item) for item in raw["pair_configs"]]
    pattern_dict = {
        int(pair_id): [HoppingPattern(**pattern) for pattern in patterns]
        for pair_id, patterns in raw["pattern_dict"].items()
    }
    pair_weight = parse_pair_weight_map(raw.get("pair_weight", {}))
    return {
        "num_channels": int(raw["num_channels"]),
        "hard_collision_threshold": raw.get("hard_collision_threshold"),
        "plot_title": raw.get("plot_title", "BLE Event Grid"),
        "output_path": raw.get("output_path", "ble_macrocycle_hopping_sdp_schedule.png"),
        "pair_configs": pair_configs,
        "cfg_dict": {cfg.pair_id: cfg for cfg in pair_configs},
        "pattern_dict": pattern_dict,
        "pair_weight": pair_weight,
    }
```

**Step 4: Run test to verify it passes**

Run the same unittest command.

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "feat: add BLE SDP JSON config loader"
```

### Task 2: Add a failing test for the default config file interface

**Files:**
- Create: `ble_macrocycle_hopping_sdp_config.json`
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Modify: `ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
    def test_default_config_file_exists_and_is_loadable(self):
        config_path = pathlib.Path(MODULE.__file__).with_name("ble_macrocycle_hopping_sdp_config.json")
        self.assertTrue(config_path.exists())

        config = MODULE.load_config_from_json(config_path)

        self.assertGreater(len(config["pair_configs"]), 0)
        self.assertGreater(len(config["pattern_dict"]), 0)
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python -m unittest tests.test_ble_macrocycle_hopping_sdp.EventBlockExpansionTest.test_default_config_file_exists_and_is_loadable -v`

Expected: FAIL because the default config file does not exist yet.

**Step 3: Write minimal implementation**

Create `ble_macrocycle_hopping_sdp_config.json` with Chinese parameter descriptions embedded in `_comment_*` fields and data matching the current demo instance.

Example:

```json
{
  "_comment_num_channels": "物理信道总数",
  "num_channels": 37,
  "_comment_hard_collision_threshold": "硬碰撞阈值，设为 null 表示不启用",
  "hard_collision_threshold": null,
  "_comment_plot_title": "输出时频调度图标题",
  "plot_title": "BLE Event Grid",
  "_comment_output_path": "输出图片路径",
  "output_path": "ble_macrocycle_hopping_sdp_schedule.png",
  "_comment_pair_configs": "每条 BLE 链路的时序参数",
  "pair_configs": [...],
  "_comment_pattern_dict": "每条链路的候选跳频模式",
  "pattern_dict": {...},
  "_comment_pair_weight": "链路对之间的碰撞代价权重，键格式为 i-j",
  "pair_weight": {...}
}
```

**Step 4: Run test to verify it passes**

Run the same unittest command.

Expected: PASS

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp_config.json tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: add default BLE SDP config file"
```

### Task 3: Add a failing test for CLI argument parsing

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Modify: `ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
    def test_parse_args_accepts_config_path(self):
        args = MODULE.parse_args(["--config", "custom.json"])
        self.assertEqual(args.config, "custom.json")
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python -m unittest tests.test_ble_macrocycle_hopping_sdp.EventBlockExpansionTest.test_parse_args_accepts_config_path -v`

Expected: FAIL because `parse_args` does not exist yet.

**Step 3: Write minimal implementation**

```python
def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="BLE hopping SDP 调度与时频块绘图")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("ble_macrocycle_hopping_sdp_config.json")),
        help="JSON 配置文件路径",
    )
    return parser.parse_args(argv)
```

**Step 4: Run test to verify it passes**

Run the same unittest command.

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "feat: add BLE SDP config CLI"
```

### Task 4: Route main() through the config interface

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
    def test_run_from_config_returns_loaded_demo_values(self):
        config_path = pathlib.Path(MODULE.__file__).with_name("ble_macrocycle_hopping_sdp_config.json")
        config = MODULE.load_config_from_json(config_path)
        self.assertEqual(config["num_channels"], 37)
        self.assertIn(0, config["cfg_dict"])
```

**Step 2: Run test to verify it fails or expose missing integration**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python -m unittest tests.test_ble_macrocycle_hopping_sdp.EventBlockExpansionTest.test_run_from_config_returns_loaded_demo_values -v`

Expected: FAIL if default config contents are not yet wired to current main flow assumptions.

**Step 3: Write minimal implementation**

```python
def build_demo_instance():
    default_config_path = Path(__file__).with_name("ble_macrocycle_hopping_sdp_config.json")
    config = load_config_from_json(default_config_path)
    return (
        config["pair_configs"],
        config["cfg_dict"],
        config["pattern_dict"],
        config["pair_weight"],
        config["num_channels"],
    )


def main(argv: Optional[List[str]] = None) -> None:
    require_cvxpy()
    args = parse_args(argv)
    config = load_config_from_json(Path(args.config))
    ...
    pair_configs = config["pair_configs"]
    cfg_dict = config["cfg_dict"]
    pattern_dict = config["pattern_dict"]
    pair_weight = config["pair_weight"]
    num_channels = config["num_channels"]
    hard_collision_threshold = config["hard_collision_threshold"]
    plot_title = config["plot_title"]
    output_path = Path(config["output_path"])
```

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp_config.json
git commit -m "feat: drive BLE SDP demo from JSON config"
```

### Task 5: Final verification and usage validation

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Modify: `ble_macrocycle_hopping_sdp_config.json`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Run focused automated verification**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v
```

Expected: All tests PASS.

**Step 2: Run the default config interface**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python ble_macrocycle_hopping_sdp.py
```

Expected:
- script loads `ble_macrocycle_hopping_sdp_config.json`
- SDP solves successfully
- event-block table prints
- schedule image is written to the configured `output_path`

**Step 3: Run the explicit config interface**

Run:

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python ble_macrocycle_hopping_sdp.py --config ble_macrocycle_hopping_sdp_config.json
```

Expected: Same behavior as default run.

**Step 4: Check generated artifact**

Run:

```bash
ls -l ble_macrocycle_hopping_sdp_schedule.png
```

Expected: File exists and size is greater than zero.

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp_config.json tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: add JSON config interface to BLE hopping SDP script"
```
