# Macrocycle Wraparound Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix macrocycle occupancy/event expansion so transmissions that cross the macrocycle boundary wrap around correctly, and make all downstream scheduling, conflict checks, and plotting use the same wraparound semantics.

**Architecture:** The root cause lives in the environment helpers that expand periodic occupancy into macrocycle slots. The fix should start there, then verify all downstream consumers such as WiFi-first BLE capacity checks, event export, and schedule plotting inherit the corrected wraparound behavior without re-implementing special cases.

**Tech Stack:** Python, NumPy, matplotlib, pytest

---

### Task 1: Capture the wraparound bug with failing environment tests

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pair_env_structure.py`
- Create: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_macrocycle_wraparound.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_macrocycle_wraparound.py`

**Step 1: Write the failing occupancy test**

Add a small environment-level test that forces a pair to start near the end of the macrocycle and expects wrapped occupancy.

```python
def test_expand_pair_occupancy_wraps_across_macrocycle_boundary():
    e = env(...)
    pair_id = ...
    occ = e.expand_pair_occupancy(pair_id, start_slot=31, macrocycle_slots=32)
    assert np.where(occ)[0].tolist() == [0, 1, 2, 3, 31]
```

**Step 2: Write the failing event-instance test**

Add a `per_ce` BLE test that expects `expand_pair_event_instances(...)` to return wrapped slot ranges instead of truncating at `macrocycle_slots`.

```python
def test_expand_pair_event_instances_wraps_last_event():
    instances = e.expand_pair_event_instances(pair_id, macrocycle_slots=32, start_slot=31)
    assert instances[0]["slot_range"] == (31, 36)
    assert instances[0]["wrapped_slot_ranges"] == [(31, 32), (0, 4)]
```

**Step 3: Run tests to verify they fail**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_pair_env_structure.py \
  sim_script/tests/test_macrocycle_wraparound.py -v
```

Expected: FAIL because occupancy and event expansion currently truncate at the macrocycle boundary.

**Step 4: Re-run to verify clean red**

Run the same command again and confirm the failures are still due to truncation, not malformed tests.

**Step 5: Commit**

```bash
git add sim_script/tests/test_pair_env_structure.py sim_script/tests/test_macrocycle_wraparound.py
git commit -m "test: capture macrocycle wraparound bug"
```

### Task 2: Fix wraparound in the environment helpers

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/env/env.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_macrocycle_wraparound.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pair_env_structure.py`

**Step 1: Implement wrapped occupancy**

Update `expand_pair_occupancy(...)` so each repeated transmission width is written modulo `macrocycle_slots` instead of being clipped.

Minimal sketch:

```python
for offset in range(width_slots):
    occ[(z + offset) % macrocycle_slots] = True
```

**Step 2: Implement wrapped event segments**

Update `expand_pair_event_instances(...)` so event instances keep enough information for plotting/export across the boundary.

Minimal sketch:

```python
wrapped_ranges = []
remaining = width_slots
cursor = z
while remaining > 0:
    seg_start = cursor % macrocycle_slots
    seg_len = min(remaining, macrocycle_slots - seg_start)
    wrapped_ranges.append((seg_start, seg_start + seg_len))
    cursor += seg_len
    remaining -= seg_len
```

Store both:

- the logical unwrapped range
- the wrapped segment list used by plotting/export

**Step 3: Run tests to verify they pass**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_pair_env_structure.py \
  sim_script/tests/test_macrocycle_wraparound.py -v
```

Expected: PASS.

**Step 4: Check for local regressions in existing BLE occupancy tests**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_ble_per_ce_occupancy_expansion.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pair_env_structure.py sim_script/tests/test_macrocycle_wraparound.py
git commit -m "fix: wrap occupancy and events across macrocycle boundary"
```

### Task 3: Propagate wraparound semantics to scheduling and plotting

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_src/env/env.py`
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/pd_mmw_template_ap_stats.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_wifi_first_ble_channel_availability.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_ble_per_ce_export_and_plot.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_macrocycle_real_conflict_integration.py`

**Step 1: Fix WiFi-first BLE channel availability probing**

`get_available_ble_channels_for_start_slot(...)` currently probes only up to `start_slot + 1`, which is incompatible with wraparound semantics near the boundary. Rework it so it checks occupancy modulo the macrocycle.

Minimal sketch:

```python
macrocycle_slots = int(self.compute_macrocycle_slots())
occ = self.expand_pair_occupancy(pair_id, pair_start, macrocycle_slots)
if occ[start_slot % macrocycle_slots]:
    ...
```

**Step 2: Fix slot-level event export**

Update `build_schedule_plot_rows(...)` and `build_ble_ce_event_rows(...)` so wrapped event segments are emitted as multiple slot rows / CSV rows covering both the tail and the wrapped head of the macrocycle.

**Step 3: Add/adjust tests**

Write one test that proves:

- a WiFi transmission starting at slot `31` blocks BLE capacity at slot `31` and at wrapped slot `0` when appropriate

Write another test that proves:

- a wrapped BLE event generates rows on both sides of the boundary in the plot/export layer

**Step 4: Run the targeted tests**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_wifi_first_ble_channel_availability.py \
  sim_script/tests/test_ble_per_ce_export_and_plot.py \
  sim_script/tests/test_macrocycle_real_conflict_integration.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_wifi_first_ble_channel_availability.py sim_script/tests/test_ble_per_ce_export_and_plot.py sim_script/tests/test_macrocycle_real_conflict_integration.py
git commit -m "fix: propagate macrocycle wraparound through scheduling and export"
```

### Task 4: Verify the user-visible outputs and document the behavior

**Files:**
- Modify: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/README.md`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_ble_channel_modes.py`
- Test: `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test/sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Add a small documentation note**

In `README.md`, add one short note in the outputs/scheduling section that macrocycle exports and plots use wrapped occupancy semantics across the cycle boundary.

**Step 2: Run a focused CLI regression**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 -m pytest \
  sim_script/tests/test_pd_mmw_ble_channel_modes.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_run.py -v
```

Expected: PASS.

**Step 3: Run one real script smoke test**

Run:

```bash
/data/home/public/anaconda3/envs/sig-sdp/bin/python3.10 sim_script/pd_mmw_template_ap_stats.py \
  --cell-size 1 \
  --pair-density 0.5 \
  --seed 123 \
  --mmw-nit 5 \
  --wifi-first-ble-scheduling \
  --output-dir sim_script/output
```

Expected:

- exit code `0`
- regenerated `pair_parameters.csv`
- a wrapped WiFi transmission near the boundary now appears on both ends of the macrocycle in `occupied_slots_in_macrocycle`
- plots reflect the wrapped occupancy

**Step 4: Inspect the output manually**

Verify at least one boundary-crossing case such as a WiFi pair starting near the last slot of the macrocycle now has wrapped occupancy like `[31, 0, 1, ...]` instead of `[31]`.

**Step 5: Commit**

```bash
git add README.md sim_src/env/env.py sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_ble_channel_modes.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "fix: render macrocycle schedules with wraparound semantics"
```
