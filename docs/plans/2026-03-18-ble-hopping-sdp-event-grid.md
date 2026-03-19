# BLE Hopping SDP Event Grid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the BLE-only SDP prototype to export event-level time-frequency blocks and generate a PNG schedule plot directly from the rounded scheduling result.

**Architecture:** Keep the existing candidate-state enumeration, collision matrix construction, SDP relaxation, and rounding flow in [ble_macrocycle_hopping_sdp.py](/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py). Add a post-processing layer that expands each selected candidate state into event-level blocks, computes overlap segments for visualization, and renders a matplotlib figure using BLE channel center frequencies on the y-axis.

**Tech Stack:** Python, `numpy`, `matplotlib`, `cvxpy`, standard library `unittest`

---

### Task 1: Add a failing test for event-block expansion

**Files:**
- Create: `tests/test_ble_macrocycle_hopping_sdp.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "ble_macrocycle_hopping_sdp.py"
SPEC = importlib.util.spec_from_file_location("ble_macrocycle_hopping_sdp", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EventBlockExpansionTest(unittest.TestCase):
    def test_selected_schedule_expands_to_event_blocks(self):
        cfg = MODULE.PairConfig(
            pair_id=7,
            release_time=0,
            deadline=10,
            connect_interval=3,
            event_duration=2,
            num_events=3,
        )
        pattern = MODULE.HoppingPattern(pattern_id=2, start_channel=5, hop_increment=4)
        selected = {7: MODULE.CandidateState(pair_id=7, offset=1, pattern_id=2)}
        cfg_dict = {7: cfg}
        pattern_dict = {7: [pattern]}

        blocks = MODULE.build_event_blocks(selected, cfg_dict, pattern_dict, num_channels=37)

        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0].pair_id, 7)
        self.assertEqual(blocks[0].event_index, 0)
        self.assertEqual(blocks[0].start_slot, 1)
        self.assertEqual(blocks[0].end_slot, 2)
        self.assertEqual(blocks[0].channel, 5)
        self.assertEqual(blocks[1].start_slot, 4)
        self.assertEqual(blocks[1].channel, 9)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: FAIL with `AttributeError` because `build_event_blocks` and its result type do not exist yet.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class EventBlock:
    pair_id: int
    event_index: int
    start_slot: int
    end_slot: int
    channel: int
    frequency_mhz: float
    offset: int
    pattern_id: int


def ble_channel_to_frequency_mhz(channel: int) -> float:
    return 2402.0 + 2.0 * channel


def build_event_blocks(selected, cfg_dict, pattern_dict, num_channels):
    blocks = []
    for pair_id, state in sorted(selected.items()):
        cfg = cfg_dict[pair_id]
        pat = lookup_pattern(pattern_dict, pair_id, state.pattern_id)
        for event_idx in range(cfg.num_events):
            start = event_start_time(cfg, state.offset, event_idx)
            end = start + cfg.event_duration - 1
            channel = channel_of_event(pat, event_idx, num_channels)
            blocks.append(
                EventBlock(
                    pair_id=pair_id,
                    event_index=event_idx,
                    start_slot=start,
                    end_slot=end,
                    channel=channel,
                    frequency_mhz=ble_channel_to_frequency_mhz(channel),
                    offset=state.offset,
                    pattern_id=state.pattern_id,
                )
            )
    return blocks
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: PASS for `test_selected_schedule_expands_to_event_blocks`

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "test: cover BLE SDP event block expansion"
```

### Task 2: Add a failing test for overlap segment detection

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
    def test_overlap_segments_capture_same_channel_time_conflicts(self):
        blocks = [
            MODULE.EventBlock(
                pair_id=0,
                event_index=0,
                start_slot=2,
                end_slot=5,
                channel=10,
                frequency_mhz=2422.0,
                offset=2,
                pattern_id=0,
            ),
            MODULE.EventBlock(
                pair_id=1,
                event_index=0,
                start_slot=4,
                end_slot=6,
                channel=10,
                frequency_mhz=2422.0,
                offset=4,
                pattern_id=1,
            ),
        ]

        overlaps = MODULE.build_overlap_blocks(blocks)

        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0].start_slot, 4)
        self.assertEqual(overlaps[0].end_slot, 5)
        self.assertEqual(overlaps[0].channel, 10)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: FAIL with `AttributeError` because `build_overlap_blocks` does not exist.

**Step 3: Write minimal implementation**

```python
def build_overlap_blocks(blocks):
    overlaps = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if blocks[i].channel != blocks[j].channel:
                continue
            start = max(blocks[i].start_slot, blocks[j].start_slot)
            end = min(blocks[i].end_slot, blocks[j].end_slot)
            if start <= end:
                overlaps.append(
                    EventBlock(
                        pair_id=-1,
                        event_index=-1,
                        start_slot=start,
                        end_slot=end,
                        channel=blocks[i].channel,
                        frequency_mhz=blocks[i].frequency_mhz,
                        offset=-1,
                        pattern_id=-1,
                    )
                )
    return overlaps
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: PASS for both tests

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "test: cover BLE SDP overlap segment detection"
```

### Task 3: Add a failing test for schedule rendering output

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
import tempfile


    def test_render_schedule_plot_writes_png(self):
        blocks = [
            MODULE.EventBlock(
                pair_id=3,
                event_index=1,
                start_slot=8,
                end_slot=10,
                channel=12,
                frequency_mhz=2426.0,
                offset=2,
                pattern_id=0,
            )
        ]
        overlaps = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "schedule.png"
            MODULE.render_event_grid(blocks, overlaps, output_path, title="Test Grid")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: FAIL with `AttributeError` because `render_event_grid` does not exist.

**Step 3: Write minimal implementation**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


def render_event_grid(blocks, overlaps, output_path, title="BLE Event Grid"):
    fig, ax = plt.subplots(figsize=(14, 6))
    for block in blocks:
        width = block.end_slot - block.start_slot + 1
        rect = Rectangle((block.start_slot, block.frequency_mhz - 0.8), width, 1.6, color="#dd8452", alpha=0.9)
        ax.add_patch(rect)
        ax.text(block.start_slot + width / 2.0, block.frequency_mhz, f"{block.pair_id} B-ch{block.channel} ev{block.event_index}", ha="center", va="center", fontsize=8)
    for block in overlaps:
        width = block.end_slot - block.start_slot + 1
        rect = Rectangle((block.start_slot, block.frequency_mhz - 0.8), width, 1.6, color="#e15759", alpha=0.75)
        ax.add_patch(rect)
    ax.set_title(title)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Frequency (MHz)")
    ax.legend(handles=[Patch(color="#dd8452", label="BLE"), Patch(color="#e15759", label="BLE overlap")])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: PASS, and the temporary PNG file exists.

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "feat: render BLE SDP event grid"
```

### Task 4: Wire the new block-table and plot flow into the main script

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
    def test_schedule_summary_contains_event_blocks(self):
        pair_configs, cfg_dict, pattern_dict, pair_weight, num_channels = MODULE.build_demo_instance()
        states, _, A_k = MODULE.build_candidate_states(pair_configs, pattern_dict)
        Omega = MODULE.build_collision_matrix(states, cfg_dict, pattern_dict, num_channels, pair_weight)
        selected = {
            0: states[A_k[0][0]],
            1: states[A_k[1][0]],
            2: states[A_k[2][0]],
            3: states[A_k[3][0]],
        }

        blocks = MODULE.build_event_blocks(selected, cfg_dict, pattern_dict, num_channels)

        self.assertGreater(len(blocks), 0)
        self.assertTrue(all(hasattr(block, "frequency_mhz") for block in blocks))
```

**Step 2: Run test to verify it fails or exposes missing integration**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: FAIL if event blocks are not exposed or helper contracts are still incomplete.

**Step 3: Write minimal implementation**

```python
def print_event_block_table(blocks):
    print("=" * 72)
    print("事件级时频块表")
    print("=" * 72)
    for block in blocks:
        print(
            f"pair={block.pair_id}, ev={block.event_index}, "
            f"time=[{block.start_slot}, {block.end_slot}], "
            f"ch={block.channel}, freq={block.frequency_mhz:.1f} MHz, "
            f"offset={block.offset}, pattern={block.pattern_id}"
        )


def main():
    ...
    selected = round_solution_from_Y(Y.value, states, A_k)
    blocks = build_event_blocks(selected, cfg_dict, pattern_dict, num_channels)
    overlaps = build_overlap_blocks(blocks)
    print_selected_schedule(selected, cfg_dict, pattern_dict, num_channels)
    print_event_block_table(blocks)
    render_event_grid(
        blocks,
        overlaps,
        output_path="ble_macrocycle_hopping_sdp_schedule.png",
        title="BLE Event Grid",
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: PASS, and helper outputs are available for main flow integration.

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "feat: export BLE SDP event blocks in main flow"
```

### Task 5: Add a smoke test for the full plotting pipeline

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
    def test_demo_schedule_plot_can_be_generated_without_solver(self):
        pair_configs, cfg_dict, pattern_dict, _, num_channels = MODULE.build_demo_instance()
        selected = {
            0: MODULE.CandidateState(pair_id=0, offset=1, pattern_id=0),
            1: MODULE.CandidateState(pair_id=1, offset=2, pattern_id=0),
        }

        blocks = MODULE.build_event_blocks(selected, cfg_dict, pattern_dict, num_channels)
        overlaps = MODULE.build_overlap_blocks(blocks)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "demo-grid.png"
            MODULE.render_event_grid(blocks, overlaps, output_path)
            self.assertTrue(output_path.exists())
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: FAIL until all helper functions work together.

**Step 3: Write minimal implementation**

```python
def render_selected_schedule(selected, cfg_dict, pattern_dict, num_channels, output_path):
    blocks = build_event_blocks(selected, cfg_dict, pattern_dict, num_channels)
    overlaps = build_overlap_blocks(blocks)
    render_event_grid(blocks, overlaps, output_path)
    return blocks, overlaps
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: PASS, confirming the post-processing path works independently of the solver.

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "test: add BLE SDP plotting pipeline smoke coverage"
```

### Task 6: Final verification and manual output check

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Run focused automated verification**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: All tests PASS.

**Step 2: Run the script end-to-end**

Run: `python ble_macrocycle_hopping_sdp.py`

Expected:
- SDP solves successfully if `cvxpy` and `SCS` are installed
- terminal prints selected schedule and event-block table
- file `ble_macrocycle_hopping_sdp_schedule.png` is created in the repo root

**Step 3: Inspect the generated artifact**

Run: `ls -l ble_macrocycle_hopping_sdp_schedule.png`

Expected: The PNG exists and size is greater than zero.

**Step 4: If `cvxpy` is missing locally, verify the non-solver path**

Run: `python -m unittest tests/test_ble_macrocycle_hopping_sdp.py -v`

Expected: Tests still PASS because block expansion and plotting helpers do not require the SDP solver.

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp_schedule.png
git commit -m "feat: visualize BLE SDP schedule as event grid"
```
