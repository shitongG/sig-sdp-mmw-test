# WiFi-First Iterative Hard Guard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the iterative WiFi-BLE coordination mode preserve WiFi-first semantics by forbidding any candidate schedule that reduces the number of scheduled WiFi pairs relative to the baseline WiFi-first result.

**Architecture:** Keep the current iterative coordination framework, but change the comparison and acceptance logic from “maximize total scheduled count” to a lexicographic WiFi-first policy. The baseline WiFi scheduled count becomes a hard floor for all iterative attempts; only candidates that keep or improve WiFi scheduling are eligible, and only then are total scheduled count, BLE scheduled count, and overlap used as secondary objectives.

**Tech Stack:** Python, `sim_script/pd_mmw_template_ap_stats.py`, JSON config files, pytest

---

### Task 1: Add failing tests for strict WiFi-first scoring

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

Add logic tests that assert:
- a candidate with fewer scheduled WiFi pairs than baseline is rejected even if total scheduled count is higher
- a candidate with equal WiFi scheduled count and higher total scheduled count is accepted
- run output prints baseline/final WiFi and BLE scheduled counts for coordination mode

Example test skeleton:

```python
def test_iterative_coordination_rejects_candidate_that_drops_wifi():
    baseline = FakeAttempt(total=36, wifi=5, ble=31, overlap=0)
    candidate = FakeAttempt(total=38, wifi=4, ble=34, overlap=0)
    assert not _is_better_schedule_attempt(candidate, baseline, baseline_wifi_count=5)
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "drops_wifi or coordination_breakdown"
```

Expected: FAIL because the current scoring allows WiFi to drop and current logging omits the WiFi/BLE breakdown.

**Step 3: Write minimal implementation**

Do not change behavior yet beyond the minimum needed for the tests.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "drops_wifi or coordination_breakdown"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "test: add strict wifi-first coordination expectations"
```

### Task 2: Change attempt scoring to enforce a WiFi hard floor

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

Add focused tests around the scoring helpers asserting the new ordering:
1. reject if `candidate.wifi_scheduled_count < baseline_wifi_count`
2. among candidates with `wifi_scheduled_count >= baseline_wifi_count`, prefer larger `wifi_scheduled_count`
3. then prefer larger total scheduled count
4. then prefer larger BLE scheduled count
5. then prefer smaller overlap count

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "strict_wifi_first_score"
```

Expected: FAIL because `_score_schedule_attempt` and `_is_better_schedule_attempt` do not implement this policy.

**Step 3: Write minimal implementation**

In `sim_script/pd_mmw_template_ap_stats.py`:
- change `_score_schedule_attempt(...)` to include WiFi scheduled count explicitly
- change `_is_better_schedule_attempt(...)` to accept `baseline_wifi_count`
- reject any candidate whose WiFi scheduled count is below the baseline WiFi count
- compare eligible candidates lexicographically by:
  - `wifi_scheduled_count`
  - `total_scheduled_count`
  - `ble_scheduled_count`
  - `-overlap_row_count`

Keep the change local to iterative coordination; do not alter non-iterative scheduling.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "strict_wifi_first_score"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "fix: enforce wifi-first hard floor in iterative scoring"
```

### Task 3: Gate iterative candidates before full acceptance

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

Add a test for `run_iterative_wifi_ble_coordination(...)` asserting that:
- if all reshuffle attempts reduce WiFi below baseline, the baseline result is retained
- if one attempt preserves WiFi and improves BLE/total, that attempt is accepted

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "iterative_retains_baseline_wifi"
```

Expected: FAIL because the current loop accepts higher-total candidates even when WiFi drops.

**Step 3: Write minimal implementation**

In `run_iterative_wifi_ble_coordination(...)`:
- record `baseline_wifi_count = baseline.wifi_scheduled_count`
- pass it into all candidate comparisons
- ensure the final selected attempt also satisfies the WiFi floor
- keep `tested_candidates` counting unchanged

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "iterative_retains_baseline_wifi"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "fix: retain baseline wifi count during iterative coordination"
```

### Task 4: Improve coordination logging and CSV-facing explainability

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

Add a run test asserting the coordination summary includes:
- `baseline_wifi_scheduled`
- `baseline_ble_scheduled`
- `final_wifi_scheduled`
- `final_ble_scheduled`
- whether a WiFi-dropping candidate was rejected, if available

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k coordination_breakdown
```

Expected: FAIL because the output only prints total counts today.

**Step 3: Write minimal implementation**

Update the printed `wifi_ble_coordination = {...}` summary to include WiFi/BLE breakdown. If practical, also report:
- `baseline_total_scheduled`
- `final_total_scheduled`
- `wifi_floor_enforced: true`

Do not add noisy per-candidate logs.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k coordination_breakdown
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: add wifi-first coordination breakdown logging"
```

### Task 5: Update README and config comments to match the stricter semantics

**Files:**
- Modify: `README.md`
- Modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`

**Step 1: Write the failing doc expectation**

List the required README corrections:
- iterative mode is now a strict WiFi-first coordinator
- already scheduled WiFi count is treated as a hard floor
- BLE gains are only considered among candidates that preserve WiFi

**Step 2: Verify docs are stale**

Open the current README and config comments; confirm they still imply “total scheduled count first”.

**Step 3: Write minimal documentation**

Update README and JSON comments so the documented objective is:
- WiFi scheduled count must not decrease
- then maximize total scheduled count
- then maximize BLE scheduled count
- then minimize overlap

**Step 4: Verify docs are consistent with code**

Check config key names and wording against the implementation.

**Step 5: Commit**

```bash
git add README.md sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
git commit -m "docs: describe strict wifi-first iterative coordination"
```

### Task 6: Regression verification on the user-observed case

**Files:**
- Modify: none unless fixes are needed
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Run focused test suites**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "iterative or coordination or wifi_first"
```

Expected: PASS.

**Step 2: Re-run the real config the user cited**

Run:

```bash
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json
```

Expected: completes successfully and prints coordination summary showing WiFi baseline/final counts.

**Step 3: Compare output against the previous concern**

Inspect the generated `pair_parameters.csv` in the configured output directory and confirm:
- `final_wifi_scheduled >= baseline_wifi_scheduled`
- if BLE improved, it happened without losing WiFi
- if no such candidate exists, baseline is retained

**Step 4: Record the result in the final response**

Summarize whether the strict WiFi-first guard changed the observed trade-off.

**Step 5: Commit**

```bash
git add -A
git commit -m "test: verify strict wifi-first iterative coordination on real config"
```
