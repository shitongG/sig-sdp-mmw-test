# WiFi-BLE Iterative Coordination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a configurable WiFi-BLE iterative coordination mode that starts from the existing WiFi-first pipeline, locally reshuffles selected WiFi placements, reruns BLE hopping scheduling, and keeps the best mixed scheduling result by total scheduled pair count.

**Architecture:** Extend the current `wifi_first_ble_scheduling` path instead of replacing it. The new mode should wrap the existing WiFi-first scheduler, identify BLE failures caused by WiFi occupancy, generate a small set of WiFi-local rearrangement candidates, rerun BLE scheduling against each candidate, and keep the best solution. Expose the behavior as config/CLI switches so the legacy WiFi-first and current BLE backends (`macrocycle_hopping_sdp`, `macrocycle_hopping_ga`) remain selectable.

**Tech Stack:** Python, existing scheduler pipeline in `sim_script/pd_mmw_template_ap_stats.py`, existing BLE backends in `ble_macrocycle_hopping_sdp.py` and `ble_macrocycle_hopping_ga.py`, pytest

---

### Task 1: Define config surface for iterative coordination

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

Add tests that assert `resolve_runtime_config(...)` accepts and normalizes:
- `wifi_ble_coordination_mode`
- `wifi_ble_coordination_rounds`
- `wifi_ble_coordination_top_k_wifi_pairs`
- `wifi_ble_coordination_candidate_start_limit`

Also assert invalid values raise `ValueError`.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k coordination_mode
```

Expected: FAIL because the new keys are unknown.

**Step 3: Write minimal implementation**

In `sim_script/pd_mmw_template_ap_stats.py`:
- Add the four config keys to `DEFAULT_CONFIG`
- Parse matching CLI options
- Validate:
  - `wifi_ble_coordination_mode in {"off", "iterative"}`
  - integer limits are non-negative

Update the three JSON files with commented examples:
- default config keeps `"off"`
- at least one config enables `"iterative"`

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k coordination_mode
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add iterative coordination config surface"
```

### Task 2: Isolate the current WiFi-first baseline into a reusable result object

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

Add a test that calls a new helper, for example `run_wifi_first_baseline_schedule(...)`, and asserts it returns a structured result with:
- scheduled pair ids
- unscheduled pair ids
- WiFi scheduled ids
- BLE scheduled ids
- occupancy matrices / rows needed for downstream evaluation

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k wifi_first_baseline_result
```

Expected: FAIL because helper/result type does not exist.

**Step 3: Write minimal implementation**

Refactor the existing WiFi-first path into:
- a small dataclass or named structure, e.g. `ScheduleAttemptResult`
- a helper that runs one full attempt and returns all artifacts needed by:
  - BLE backend
  - CSV/plot output
  - iterative coordination scoring

Do not change behavior yet; this task is only extraction.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k wifi_first_baseline_result
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "refactor: extract wifi-first schedule attempt result"
```

### Task 3: Add diagnostics for BLE rejection reasons under WiFi-first scheduling

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

Add a test for a new helper like `diagnose_unscheduled_ble_pairs(...)` that, on a small synthetic case, classifies each failed BLE pair into one dominant reason such as:
- `wifi_capacity_blocked`
- `slot_conflict`
- `channel_conflict`
- `no_candidate_after_wifi_filter`

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k ble_rejection_reason
```

Expected: FAIL because the helper is missing.

**Step 3: Write minimal implementation**

Implement the diagnostic helper using the existing scheduling checks. Keep the classification coarse and deterministic. Store enough metadata to rank:
- most blocking WiFi pair ids
- most affected BLE pair ids

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k ble_rejection_reason
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add BLE rejection diagnostics for wifi-first scheduling"
```

### Task 4: Generate local WiFi reshuffle candidates

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing test**

Add a test for a helper such as `build_wifi_local_reshuffle_candidates(...)` that:
- takes the current scheduled WiFi pairs
- uses diagnostics to pick at most `top_k` WiFi pairs
- returns a bounded list of alternative start-slot assignments
- does not modify unrelated WiFi pairs

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k wifi_local_reshuffle
```

Expected: FAIL because candidate builder is missing.

**Step 3: Write minimal implementation**

Implement a conservative generator:
- only touch top blocking WiFi pairs
- for each selected WiFi pair, try a bounded set of nearby alternative start slots
- keep assignments valid with respect to WiFi-WiFi conflicts
- deduplicate candidate schedules

The goal is bounded search, not exhaustive search.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k wifi_local_reshuffle
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add local WiFi reshuffle candidate generation"
```

### Task 5: Add iterative coordination loop

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing test**

Add:
- a logic test that a crafted scenario improves or stays equal under `iterative`
- a run test that config mode `iterative` executes and reports coordination statistics

Assertions should cover:
- best attempt chosen by total scheduled count
- BLE scheduled count is non-decreasing relative to baseline in the crafted case
- logs/summary include number of coordination rounds and chosen attempt index

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k iterative_coordination
```

Expected: FAIL because the loop does not exist.

**Step 3: Write minimal implementation**

Implement:
- baseline attempt
- up to `wifi_ble_coordination_rounds` rounds
- each round:
  - diagnose BLE failures
  - build WiFi reshuffle candidates
  - rerun BLE backend against each candidate WiFi occupancy
  - score by:
    1. total scheduled pairs
    2. BLE scheduled pairs
    3. lower collision/overlap secondary tie-break
- keep the best attempt and stop early if no improvement

Expose concise logging:
- baseline totals
- per-round tested candidate count
- best round improvement

**Step 4: Run test to verify it passes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k iterative_coordination
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: add iterative WiFi-BLE coordination loop"
```

### Task 6: Keep BLE backends compatible with iterative coordination

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `ble_macrocycle_hopping_sdp.py`
- Modify: `ble_macrocycle_hopping_ga.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_ga.py`

**Step 1: Write the failing test**

Add tests asserting both backends can consume external WiFi interference blocks repeatedly across multiple attempts without mutating shared state incorrectly.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_ga.py -q -k repeated_external_interference
```

Expected: FAIL if backends reuse mutable artifacts unsafely.

**Step 3: Write minimal implementation**

Ensure:
- repeated calls are pure from the caller perspective
- no leaked state across attempts
- candidate summaries and logs remain attempt-local

Keep algorithm behavior unchanged beyond safety.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_ga.py -q -k repeated_external_interference
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_ga.py tests/test_ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_ga.py
git commit -m "test: harden BLE backends for iterative coordination retries"
```

### Task 7: Document the new coordination mode

**Files:**
- Modify: `README.md`

**Step 1: Write the failing doc expectation**

Add a checklist in your scratch notes for README updates:
- explain why WiFi-first can cap BLE
- explain iterative coordination idea
- list config keys
- show example command
- explain limitations vs full joint optimization

**Step 2: Verify docs are currently missing**

Open `README.md` and confirm the coordination mode is not described.

**Step 3: Write minimal documentation**

Add a new README subsection covering:
- rationale
- algorithm steps
- config examples
- when to prefer `off` vs `iterative`
- note that this is still not full joint optimization

**Step 4: Verify docs render and are consistent**

Check headings, math/Markdown formatting, and config key names against code.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: describe iterative WiFi-BLE coordination mode"
```

### Task 8: Full verification on representative configs

**Files:**
- Modify: none unless fixes are needed
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_ga.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Run focused test suites**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_ga.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "iterative_coordination or ble_ga or macrocycle_hopping"
```

Expected: PASS.

**Step 2: Run a smoke command with SDP backend**

Run:

```bash
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json --ble-schedule-backend macrocycle_hopping_sdp
```

Expected: completes and prints coordination summary when config enables `iterative`.

**Step 3: Run a smoke command with GA backend**

Run:

```bash
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json --ble-schedule-backend macrocycle_hopping_ga
```

Expected: completes and prints coordination summary when config enables `iterative`.

**Step 4: Compare baseline vs iterative on one bounded config**

Run both:

```bash
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json --ble-schedule-backend macrocycle_hopping_sdp
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json --ble-schedule-backend macrocycle_hopping_sdp --wifi-ble-coordination-mode iterative
```

Expected: iterative result is equal or better in total scheduled count on the bounded benchmark, or logs clearly show no-improvement termination.

**Step 5: Commit**

```bash
git add -A
git commit -m "test: verify iterative WiFi-BLE coordination mode"
```
