# Start Time Slot, CSV Pair Input, and WiFi Refill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-pair `start_time_slot` support with bounded anchor sampling, support pair parameter input from random generation or CSV import, generate a multi-task pair CSV fixture for testing, and add WiFi-specific refill logic for unscheduled pairs.

**Architecture:** Extend `sim_src/env/env.py` so pair timing generation can either sample from RNG or read from a normalized CSV schema. Introduce `start_time_slot` as an explicit pair-level timing input and constrain `wifi_anchor_slot` / `ble_anchor_slot` to valid windows derived from `start_time_slot`, period, and width. Then extend `sim_script/pd_mmw_template_ap_stats.py` with a WiFi-specific post-processing refill stage on top of the existing macrocycle assignment and BLE retry flow.

**Tech Stack:** Python, NumPy, SciPy sparse, existing `env` timing model, `pytest`, CSV I/O via stdlib `csv`.

---

**Assumption:** The duplicated requirement “对未调度 WiFi pair 做局部重排回填” is interpreted as one WiFi-specific refill feature, not two different WiFi recovery features. This plan implements one explicit WiFi refill stage after the current generic macrocycle repair. If you actually want a second WiFi-specific strategy such as WiFi channel retry, that should be added in a follow-up plan.

### Task 1: Freeze the CSV schema and failing anchor-bound tests

**Files:**
- Modify: `sim_script/tests/test_wifi_periodic_timing.py`
- Create: `sim_script/tests/test_pair_csv_schema.py`
- Create: `sim_script/tests/test_start_time_slot_anchor_bounds.py`
- Reference: `sim_src/env/env.py:323-336`

**Step 1: Write the failing test for WiFi anchor bounds**

```python
def test_wifi_anchor_respects_start_time_slot_and_tx_width():
    e = env(...)
    k = int(np.where(e.pair_radio_type == e.RADIO_WIFI)[0][0])
    low = e.pair_start_time_slot[k]
    high = low + e.pair_wifi_period_slots[k] - e.pair_wifi_tx_slots[k]
    assert low <= e.pair_wifi_anchor_slot[k] <= high
```

**Step 2: Write the failing test for BLE anchor bounds**

```python
def test_ble_anchor_respects_start_time_slot_and_ce_width():
    e = env(...)
    k = int(np.where(e.pair_radio_type == e.RADIO_BLE)[0][0])
    low = e.pair_start_time_slot[k]
    high = low + e.pair_ble_ci_slots[k] - e.pair_ble_ce_slots[k]
    assert low <= e.pair_ble_anchor_slot[k] <= high
```

**Step 3: Write the failing schema test**

```python
def test_pair_csv_schema_has_start_time_slot_and_timing_columns():
    required = {
        "pair_id", "office_id", "radio", "channel", "priority", "start_time_slot",
        "wifi_period_slots", "wifi_tx_slots", "wifi_anchor_slot",
        "ble_ci_slots", "ble_ce_slots", "ble_anchor_slot",
    }
    assert required.issubset(PAIR_CSV_COLUMNS)
```

**Step 4: Run tests to verify they fail**

Run:
```bash
pytest sim_script/tests/test_start_time_slot_anchor_bounds.py sim_script/tests/test_pair_csv_schema.py -q
```

Expected:
- FAIL because `pair_start_time_slot` and CSV schema constants do not exist yet

**Step 5: Commit**

```bash
git add sim_script/tests/test_wifi_periodic_timing.py sim_script/tests/test_pair_csv_schema.py sim_script/tests/test_start_time_slot_anchor_bounds.py
git commit -m "test: add start time slot anchor bound coverage"
```

### Task 2: Add `start_time_slot` to the environment model

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_start_time_slot_anchor_bounds.py`
- Test: `sim_script/tests/test_wifi_periodic_timing.py`

**Step 1: Add pair-level storage for `start_time_slot`**

Implement minimal fields alongside existing pair timing arrays:

```python
self.pair_start_time_slot = None
self.device_start_time_slot = None
self.user_start_time_slot = None
```

**Step 2: Initialize default `start_time_slot=0` for all pairs**

```python
self.pair_start_time_slot = np.zeros(self.n_pair, dtype=int)
self.device_start_time_slot = self.pair_start_time_slot
self.user_start_time_slot = self.pair_start_time_slot
```

**Step 3: Constrain WiFi anchor sampling to the bounded interval**

Use:

```python
low = int(self.pair_start_time_slot[k])
high = int(low + period_slots - tx_slots)
self.pair_wifi_anchor_slot[k] = int(self.rand_gen_loc.integers(low=low, high=high + 1))
```

**Step 4: Constrain BLE anchor sampling to the bounded interval**

Use:

```python
low = int(self.pair_start_time_slot[k])
high = int(low + ci_slots - ce_slots)
self.pair_ble_anchor_slot[k] = int(self.rand_gen_loc.integers(low=low, high=high + 1))
```

**Step 5: Run tests to verify they pass**

Run:
```bash
pytest sim_script/tests/test_start_time_slot_anchor_bounds.py sim_script/tests/test_wifi_periodic_timing.py -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_start_time_slot_anchor_bounds.py sim_script/tests/test_wifi_periodic_timing.py
git commit -m "feat: add pair start time slot anchor constraints"
```

### Task 3: Add CSV import/export schema for pair timing inputs

**Files:**
- Modify: `sim_src/env/env.py`
- Create: `sim_script/tests/test_pair_csv_roundtrip.py`
- Create: `sim_script/tests/test_pair_csv_import.py`

**Step 1: Write the failing CSV import test**

```python
def test_env_can_load_pair_timing_from_csv(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(...)
    e = env(..., pair_config_csv=str(csv_path))
    assert e.pair_start_time_slot[0] == 0
    assert e.pair_wifi_anchor_slot[0] == 5
```

**Step 2: Write the failing roundtrip test**

```python
def test_env_can_export_random_pair_config_csv(tmp_path):
    e = env(...)
    out = tmp_path / "pairs.csv"
    e.export_pair_config_csv(out)
    text = out.read_text()
    assert "start_time_slot" in text
```

**Step 3: Implement a normalized schema constant and loader**

Add:
- `PAIR_CONFIG_COLUMNS`
- `load_pair_config_csv(path)`
- `export_pair_config_csv(path)`

Keep schema flat and explicit. Use one row per pair.

**Step 4: Gate timing generation by source mode**

If `pair_config_csv` is provided:
- do not randomly sample timing/channel/priority for those columns
- validate imported ranges and dtypes

Otherwise:
- keep current random generation path

**Step 5: Run tests to verify they pass**

Run:
```bash
pytest sim_script/tests/test_pair_csv_schema.py sim_script/tests/test_pair_csv_import.py sim_script/tests/test_pair_csv_roundtrip.py -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pair_csv_schema.py sim_script/tests/test_pair_csv_import.py sim_script/tests/test_pair_csv_roundtrip.py
git commit -m "feat: add pair config csv import export"
```

### Task 4: Generate a random multi-task pair CSV fixture for testing

**Files:**
- Create: `sim_script/testdata/random_pair_tasks.csv`
- Create: `sim_script/tests/test_random_pair_tasks_fixture.py`
- Modify: `sim_script/pd_mmw_template_ap_stats.py`

**Step 1: Write the failing fixture test**

```python
def test_random_pair_tasks_fixture_has_multiple_pairs_and_start_time_slot():
    rows = list(csv.DictReader(open("sim_script/testdata/random_pair_tasks.csv")))
    assert len(rows) >= 8
    assert all("start_time_slot" in row for row in rows)
```

**Step 2: Create the fixture generator command path**

Add a script entry or CLI flag in `pd_mmw_template_ap_stats.py` to export a random pair CSV, for example:
- `--export-random-pair-csv PATH`
- `--random-pair-csv-count N`

**Step 3: Generate and check in a stable fixture**

Create `sim_script/testdata/random_pair_tasks.csv` with:
- both WiFi and BLE rows
- nontrivial `priority`
- explicit `start_time_slot`
- valid anchor bounds

**Step 4: Run tests to verify they pass**

Run:
```bash
pytest sim_script/tests/test_random_pair_tasks_fixture.py -q
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add sim_script/testdata/random_pair_tasks.csv sim_script/tests/test_random_pair_tasks_fixture.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "test: add random pair task csv fixture"
```

### Task 5: Add WiFi-specific refill for unscheduled WiFi pairs

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:164-275`
- Create: `sim_script/tests/test_wifi_refill_scheduler.py`
- Reference: `sim_script/tests/test_macrocycle_repair.py`

**Step 1: Write the failing WiFi refill test**

```python
def test_wifi_refill_can_schedule_wifi_pair_left_unscheduled_by_initial_pass():
    env = _WifiRefillEnv()
    preferred = np.array([...], dtype=int)
    result = retry_ble_channels_and_assign_macrocycle(env, preferred, max_ble_channel_retries=0)
    assert result[3] == []
```

Use a minimal env stub where:
- initial greedy pass leaves one WiFi pair unscheduled
- a second WiFi-focused refill pass can place it without changing BLE parameters

**Step 2: Run test to verify it fails**

Run:
```bash
pytest sim_script/tests/test_wifi_refill_scheduler.py -q
```

Expected:
- FAIL because no WiFi-specific refill exists yet

**Step 3: Implement a WiFi refill helper**

Add a helper after `_repair_macrocycle_assignment_by_reordering(...)`, for example:

```python
def _refill_unscheduled_wifi_pairs(...):
    ...
```

Required behavior:
- only target currently unscheduled WiFi pairs
- keep existing `period/tx/anchor` fixed
- try local reordering against already scheduled pairs
- accept only strictly better results

**Step 4: Call WiFi refill before BLE channel retry**

Update `retry_ble_channels_and_assign_macrocycle(...)` flow to:
1. initial assignment
2. generic repair
3. WiFi-specific refill
4. BLE channel retry loop

**Step 5: Run tests to verify they pass**

Run:
```bash
pytest sim_script/tests/test_wifi_refill_scheduler.py sim_script/tests/test_macrocycle_repair.py sim_script/tests/test_ble_channel_retry_scheduler.py -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_wifi_refill_scheduler.py sim_script/tests/test_macrocycle_repair.py sim_script/tests/test_ble_channel_retry_scheduler.py
git commit -m "feat: add wifi refill for unscheduled pairs"
```

### Task 6: Wire CSV input and random CSV export into the main script

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Create: `sim_script/tests/test_pd_mmw_pair_csv_cli.py`
- Update: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing CLI tests**

```python
def test_script_accepts_pair_config_csv_flag():
    ...

def test_script_can_export_random_pair_csv_without_running_full_schedule():
    ...
```

**Step 2: Add parser flags**

Add:
- `--pair-config-csv`
- `--export-random-pair-csv`
- `--random-pair-csv-count`

**Step 3: Hook flags into `env(...)` construction and export path**

Rules:
- if `--export-random-pair-csv` is passed, generate and write CSV then exit cleanly
- if `--pair-config-csv` is passed, load pair parameters from CSV
- otherwise keep random generation

**Step 4: Run tests to verify they pass**

Run:
```bash
pytest sim_script/tests/test_pd_mmw_pair_csv_cli.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_pair_csv_cli.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
 git commit -m "feat: add pair csv cli workflow"
```

### Task 7: End-to-end verification in `sig-sdp`

**Files:**
- Verify: `sim_script/pd_mmw_template_ap_stats.py`
- Verify: `sim_script/testdata/random_pair_tasks.csv`

**Step 1: Run focused test suite**

Run:
```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && \
pytest \
  sim_script/tests/test_start_time_slot_anchor_bounds.py \
  sim_script/tests/test_pair_csv_schema.py \
  sim_script/tests/test_pair_csv_import.py \
  sim_script/tests/test_pair_csv_roundtrip.py \
  sim_script/tests/test_random_pair_tasks_fixture.py \
  sim_script/tests/test_wifi_refill_scheduler.py \
  sim_script/tests/test_pd_mmw_pair_csv_cli.py -q
```

Expected:
- PASS

**Step 2: Run random CSV export smoke**

Run:
```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && \
python sim_script/pd_mmw_template_ap_stats.py \
  --cell-size 1 \
  --pair-density 0.05 \
  --seed 7 \
  --export-random-pair-csv /tmp/random_pair_tasks.csv \
  --random-pair-csv-count 12
```

Expected:
- exit code 0
- `/tmp/random_pair_tasks.csv` exists
- contains both WiFi and BLE rows

**Step 3: Run CSV-import scheduling smoke**

Run:
```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && \
python sim_script/pd_mmw_template_ap_stats.py \
  --pair-config-csv sim_script/testdata/random_pair_tasks.csv \
  --mmw-nit 5 \
  --output-dir /tmp/pair_csv_schedule
```

Expected:
- exit code 0
- emits pair/schedule CSVs and plot
- honors imported `start_time_slot` and anchor bounds

**Step 4: Commit**

```bash
git add docs/plans/2026-03-10-start-time-slot-anchor-import-wifi-refill.md
 git commit -m "docs: add start time slot and wifi refill implementation plan"
```
