# Joint Heuristic GA And Baseline Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the existing isolated joint WiFi/BLE GA baseline and add a new heuristic joint GA backend that reuses the earlier WiFi-first / BLE / WiFi-reshuffle intuition while treating each WiFi transmission as a bundle of BLE-like 2 MHz stripes.

**Architecture:** Keep all work isolated under `joint_sched/` and do not alter the existing mainline `sim_script/pd_mmw_template_ap_stats.py` experiment flow during the first implementation pass. The baseline `joint_wifi_ble_ga.py` remains the exact mixed-state GA, but gets speed-oriented internals and benchmark coverage. A new `joint_wifi_ble_hga.py` implements the heuristic coordinator: it expands each WiFi state into synchronized 2 MHz pseudo-links, schedules BLE against the induced residual spectrum, locally reshuffles blocking WiFi states, and repeats until no improvement.

**Tech Stack:** Python 3.10, NumPy, pytest, pandas, existing `joint_sched/` isolated model/plot/runner pipeline.

---

## File Structure

- `joint_sched/joint_wifi_ble_model.py`
  Shared exact joint state model. Keep this as the baseline exact candidate-state builder and hard-feasibility oracle.
- `joint_sched/joint_wifi_ble_ga.py`
  Existing exact mixed-state GA baseline. Optimize internals here, but keep the public `solve_joint_wifi_ble_ga(config)` contract stable.
- `joint_sched/joint_wifi_ble_hga_model.py`
  New helper module for heuristic scheduling. Responsible for converting WiFi states into BLE-like 2 MHz stripe occupancy, building residual-spectrum views, and identifying blocking WiFi states.
- `joint_sched/joint_wifi_ble_hga.py`
  New heuristic joint GA backend. Responsible for the iterative `WiFi seed -> BLE -> WiFi local reshuffle -> BLE` loop and returning the same artifact-friendly result schema as the existing solvers.
- `joint_sched/run_joint_wifi_ble_demo.py`
  Extend runner to support `--solver hga` and write benchmark/summary fields for the new backend.
- `joint_sched/tests/test_joint_wifi_ble_ga.py`
  Baseline GA regression and optimization tests.
- `joint_sched/tests/test_joint_wifi_ble_hga_model.py`
  New model tests for WiFi stripe expansion and blocker diagnosis.
- `joint_sched/tests/test_joint_wifi_ble_hga.py`
  New heuristic scheduler tests.
- `joint_sched/tests/test_joint_wifi_ble_runner.py`
  Runner support for the new solver and benchmark summaries.
- `README.md`
  Add a new paper-style subsection describing the heuristic joint GA and the WiFi-as-striped-BLE approximation.

This stays in one plan because the “baseline optimization” and the “heuristic backend” are tightly coupled: both operate on the same isolated `joint_sched/` subsystem, share metrics, and need direct comparison on the same random instances.

### Task 1: Add a reproducible benchmark fixture for the current isolated joint GA

**Files:**
- Create: `joint_sched/tests/test_joint_wifi_ble_ga_benchmark_fixture.py`
- Modify: `joint_sched/tests/test_joint_wifi_ble_ga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`

- [ ] **Step 1: Write the failing test**

Add a benchmark-style regression that fixes a medium random instance and asserts the current GA reports stable summary fields:

```python
def test_joint_ga_reports_stable_medium_instance_metrics():
    config = {
        "macrocycle_slots": 64,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "ga": {"population_size": 24, "generations": 20, "seed": 17},
        "tasks": [
            {"task_id": 0, "radio": "wifi", "payload_bytes": 1200, "release_slot": 0, "deadline_slot": 20, "preferred_channel": 0, "wifi_tx_slots": 4, "max_offsets": 2},
            {"task_id": 1, "radio": "wifi", "payload_bytes": 900, "release_slot": 8, "deadline_slot": 40, "preferred_channel": 5, "wifi_tx_slots": 4, "max_offsets": 2},
            {"task_id": 2, "radio": "ble", "payload_bytes": 247, "release_slot": 0, "deadline_slot": 48, "preferred_channel": 8, "ble_ce_slots": 1, "ble_ci_slots_options": [8], "ble_num_events": 3, "ble_pattern_count": 2, "max_offsets": 2},
        ],
    }
    result = solve_joint_wifi_ble_ga(config)
    assert result["status"] == "ok"
    assert result["scheduled_payload_bytes"] > 0
    assert len(result["fitness_history"]) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q -k stable_medium_instance`
Expected: FAIL because the benchmark fixture/test does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create a tiny helper fixture file with one reusable `medium_joint_instance()` config payload:

```python
def medium_joint_instance() -> dict:
    return {
        "macrocycle_slots": 64,
        "wifi_channels": [0, 5, 10],
        "ble_channels": list(range(37)),
        "ga": {"population_size": 24, "generations": 20, "seed": 17},
        "tasks": [...],
    }
```

Use it from `test_joint_wifi_ble_ga.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q -k stable_medium_instance`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/tests/test_joint_wifi_ble_ga_benchmark_fixture.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "test: add reproducible joint GA benchmark fixture"
```

### Task 2: Optimize the baseline exact joint GA internals without changing its public contract

**Files:**
- Modify: `joint_sched/joint_wifi_ble_ga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`

- [ ] **Step 1: Write the failing test**

Add tests that force the baseline GA to use cached evaluation and better seeding hooks:

```python
def test_joint_ga_reuses_cached_chromosome_metrics():
    context = make_test_context()
    chromosome = [1, 0, 2]
    left = chromosome_metrics(chromosome, context)
    right = chromosome_metrics(chromosome, context)
    assert left == right
    assert context.metric_cache_hits >= 1
```

```python
def test_joint_ga_supports_seeded_population_members():
    population = initialize_population(context, 8, rng, seeded_chromosomes=[[1, 0, 2]])
    assert population[0] == [1, 0, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q -k "cached_chromosome or seeded_population"`
Expected: FAIL because caching/seeded population are not implemented.

- [ ] **Step 3: Write minimal implementation**

Extend `JointGAContext` with:

```python
self.metric_cache: dict[tuple[int, ...], dict[str, float] | None] = {}
self.metric_cache_hits = 0
```

Update `chromosome_metrics(...)` to cache by `tuple(chromosome)`, and update `initialize_population(...)` to accept:

```python
def initialize_population(context, population_size, rng, seeded_chromosomes: list[list[int]] | None = None) -> list[list[int]]:
```

Seed valid repaired chromosomes first, then fill the remainder randomly.

- [ ] **Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q -k "cached_chromosome or seeded_population"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_ga.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "feat: optimize baseline joint GA caching and seeding"
```

### Task 3: Add the WiFi stripe approximation model for heuristic joint scheduling

**Files:**
- Create: `joint_sched/joint_wifi_ble_hga_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`

- [ ] **Step 1: Write the failing test**

Add tests for WiFi stripe expansion:

```python
def test_expand_wifi_state_to_stripes_returns_contiguous_2mhz_blocks():
    state = JointCandidateState(state_id=3, pair_id=0, medium="wifi", offset=4, channel=0, width_slots=4, period_slots=16)
    stripes = expand_wifi_state_to_stripes(state)
    assert len(stripes) == 10
    assert stripes[0].freq_low_mhz == 2402.0
    assert stripes[-1].freq_high_mhz == 2422.0
```

```python
def test_wifi_stripe_blocks_share_same_offset_and_slot_span():
    state = JointCandidateState(...)
    stripes = expand_wifi_state_to_stripes(state)
    assert {block.slot_start for block in stripes} == {4}
    assert {block.slot_end for block in stripes} == {8}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py -q`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `joint_sched/joint_wifi_ble_hga_model.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .joint_wifi_ble_model import JointCandidateState, ResourceBlock, WIFI_CHANNEL_TO_MHZ

WIFI_STRIPE_WIDTH_MHZ = 2.0
WIFI_STRIPE_COUNT = 10


def expand_wifi_state_to_stripes(state: JointCandidateState) -> list[ResourceBlock]:
    center = WIFI_CHANNEL_TO_MHZ[int(state.channel)]
    low = center - 10.0
    blocks = []
    for idx in range(WIFI_STRIPE_COUNT):
        stripe_low = low + idx * WIFI_STRIPE_WIDTH_MHZ
        stripe_high = stripe_low + WIFI_STRIPE_WIDTH_MHZ
        blocks.append(
            ResourceBlock(
                state_id=state.state_id,
                pair_id=state.pair_id,
                medium="wifi",
                event_index=idx,
                slot_start=state.offset,
                slot_end=state.offset + int(state.width_slots),
                freq_low_mhz=stripe_low,
                freq_high_mhz=stripe_high,
                label=f"wifi-{state.pair_id}-stripe{idx}",
            )
        )
    return blocks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_hga_model.py joint_sched/tests/test_joint_wifi_ble_hga_model.py
git commit -m "feat: add wifi stripe model for heuristic joint GA"
```

### Task 4: Build blocker diagnosis and local WiFi reshuffle helpers

**Files:**
- Modify: `joint_sched/joint_wifi_ble_hga_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga_model.py`

- [ ] **Step 1: Write the failing test**

Add tests for blocker identification and local alternatives:

```python
def test_identify_blocking_wifi_pairs_returns_pairs_that_overlap_unscheduled_ble():
    blockers = identify_blocking_wifi_pairs(wifi_states, ble_state)
    assert blockers == {0, 4}
```

```python
def test_build_wifi_local_reshuffle_candidates_excludes_current_offset():
    candidates = build_wifi_local_reshuffle_candidates(current_state, alternatives)
    assert all(candidate.offset != current_state.offset for candidate in candidates)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py -q -k "blocking_wifi_pairs or reshuffle_candidates"`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Add helpers:

```python
def identify_blocking_wifi_pairs(wifi_states, ble_state) -> set[int]:
    ...


def build_wifi_local_reshuffle_candidates(current_state, all_wifi_states, max_candidates: int = 4) -> list[JointCandidateState]:
    ...
```

Use stripe-expanded WiFi blocks against BLE blocks to decide blockers.

- [ ] **Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_hga_model.py -q -k "blocking_wifi_pairs or reshuffle_candidates"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_hga_model.py joint_sched/tests/test_joint_wifi_ble_hga_model.py
git commit -m "feat: add blocker diagnosis for heuristic joint GA"
```

### Task 5: Implement the heuristic joint GA backend

**Files:**
- Create: `joint_sched/joint_wifi_ble_hga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_hga.py`

- [ ] **Step 1: Write the failing test**

Add a smoke test for the iterative heuristic flow:

```python
def test_joint_hga_returns_collision_free_schedule():
    result = solve_joint_wifi_ble_hga(config)
    assert result["solver"] == "hga"
    assert result["status"] == "ok"
    assert result["scheduled_payload_bytes"] > 0
    assert result["coordination_rounds_used"] >= 1
```

Add a behavioral test that the heuristic can keep WiFi and improve BLE relative to the seed WiFi-only placement:

```python
def test_joint_hga_reshuffles_wifi_instead_of_dropping_payload():
    result = solve_joint_wifi_ble_hga(config)
    assert result["wifi_seed_payload_bytes"] == result["final_wifi_payload_bytes"]
    assert result["scheduled_payload_bytes"] >= result["wifi_seed_payload_bytes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_hga.py -q`
Expected: FAIL because the solver file does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `joint_sched/joint_wifi_ble_hga.py` with this skeleton:

```python
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .joint_wifi_ble_ga import solve_joint_wifi_ble_ga
from .joint_wifi_ble_hga_model import identify_blocking_wifi_pairs, build_wifi_local_reshuffle_candidates
from .joint_wifi_ble_model import expand_candidate_blocks, selected_schedule_has_no_conflicts, state_is_idle


def solve_joint_wifi_ble_hga(config: Mapping[str, Any]) -> dict[str, Any]:
    # 1. run baseline joint GA as WiFi-seeded initializer
    # 2. separate selected wifi states and unscheduled ble tasks
    # 3. identify blocking wifi pairs for those BLE tasks
    # 4. try small local wifi reshuffles
    # 5. rerun BLE-aware selection and keep best payload-preserving improvement
    return {
        "solver": "hga",
        "status": "ok",
        ...
    }
```

Keep the result schema aligned with the existing runner: `selected_states`, `unscheduled_pair_ids`, `blocks`, `scheduled_payload_bytes`, `occupied_slot_count`, `fill_penalty`.

- [ ] **Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_hga.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_hga.py joint_sched/tests/test_joint_wifi_ble_hga.py
git commit -m "feat: add heuristic joint GA backend"
```

### Task 6: Wire the new heuristic solver into the isolated runner

**Files:**
- Modify: `joint_sched/__init__.py`
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Modify: `joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing test**

Add runner coverage:

```python
def test_runner_accepts_hga_solver(tmp_path):
    summary = run_joint_demo(config_path="joint_sched/joint_wifi_ble_demo_config.json", solver="hga", output_dir=tmp_path / "hga")
    assert summary["solver"] == "hga"
    assert summary["status"] == "ok"
    assert Path(summary["overview_path"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q -k accepts_hga_solver`
Expected: FAIL because runner only supports `sdp` and `ga`.

- [ ] **Step 3: Write minimal implementation**

Update `parse_args()`:

```python
parser.add_argument("--solver", choices=["sdp", "ga", "hga"], default=None)
```

Update dispatch in `run_joint_demo(...)`:

```python
if solver_name == "ga":
    result = solve_joint_wifi_ble_ga(config)
elif solver_name == "hga":
    result = solve_joint_wifi_ble_hga(config)
else:
    result = solve_joint_wifi_ble_sdp(config)
```

Export `solve_joint_wifi_ble_hga` from `joint_sched/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q -k accepts_hga_solver`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/__init__.py joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_runner.py
git commit -m "feat: wire heuristic joint GA into isolated runner"
```

### Task 7: Compare baseline exact GA and heuristic GA on the same random config

**Files:**
- Modify: `joint_sched/run_joint_wifi_ble_demo.py` (only if summary fields are missing)
- Test: `joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing test**

Add a summary-field test for heuristic diagnostics:

```python
def test_hga_summary_reports_seed_and_final_payloads(tmp_path):
    summary = run_joint_demo(..., solver="hga", output_dir=tmp_path / "hga")
    assert summary["wifi_seed_payload_bytes"] >= 0
    assert summary["final_wifi_payload_bytes"] >= 0
    assert summary["coordination_rounds_used"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q -k seed_and_final_payloads`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Add the extra fields from the HGA result into runner summary and `joint_summary.json`.

Then run two actual experiments:

```bash
python joint_sched/run_joint_wifi_ble_demo.py --config joint_sched/joint_wifi_ble_main_config_payload_fill.json --solver ga --output joint_sched/output_main_config_ga_optimized
python joint_sched/run_joint_wifi_ble_demo.py --config joint_sched/joint_wifi_ble_main_config_payload_fill.json --solver hga --output joint_sched/output_main_config_hga
```

Expected: both complete and write full artifact families.

- [ ] **Step 4: Run test to verify it passes**

Run: `env PYTHONPATH=. pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q -k seed_and_final_payloads`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/output_main_config_ga_optimized joint_sched/output_main_config_hga
git commit -m "test: compare optimized GA and heuristic joint GA"
```

### Task 8: Document the heuristic approximation in paper style

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the documentation diff**

Add a subsection under `joint_sched` that explains:
- why the exact mixed-state GA is still useful as a baseline
- why the earlier heuristic intuition is now being reintroduced in a joint formulation
- the approximation: a `20 MHz` WiFi transmission is modeled as `10` synchronized `2 MHz` stripes
- the iterative loop:

```math
\text{WiFi seed} \rightarrow \text{BLE schedule} \rightarrow \text{identify blockers} \rightarrow \text{WiFi local reshuffle} \rightarrow \text{BLE reschedule}
```

and the invariant:

```math
P_{\mathrm{wifi}}^{(t+1)} \ge P_{\mathrm{wifi}}^{(t)}
```

for the strict deterministic-traffic case where WiFi payload is not allowed to drop.

- [ ] **Step 2: Inspect the updated section locally**

Run: `sed -n '900,1120p' README.md`
Expected: the new heuristic subsection appears in the `joint_sched` section and uses GitHub-safe math blocks.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe heuristic joint GA and wifi stripe approximation"
```

### Task 9: Final regression and comparison sanity check

**Files:**
- Modify: none expected
- Test: `joint_sched/tests/*`

- [ ] **Step 1: Run syntax checks**

Run:

```bash
python -m py_compile \
  joint_sched/joint_wifi_ble_model.py \
  joint_sched/joint_wifi_ble_ga.py \
  joint_sched/joint_wifi_ble_hga_model.py \
  joint_sched/joint_wifi_ble_hga.py \
  joint_sched/run_joint_wifi_ble_demo.py
```

Expected: no output.

- [ ] **Step 2: Run the full isolated joint test suite**

Run:

```bash
env PYTHONPATH=. pytest joint_sched/tests -q
```

Expected: PASS.

- [ ] **Step 3: Run no-conflict sanity checks on both GA backends**

Run a short snippet that loads the generated `schedule_plot_rows.csv` files and asserts no positive time-frequency overlap remains for:
- `joint_sched/output_main_config_ga_optimized`
- `joint_sched/output_main_config_hga`

Expected: both report `no_conflicts = True`.

- [ ] **Step 4: Commit**

```bash
git add joint_sched README.md
git commit -m "chore: finalize joint GA optimization and heuristic backend"
```
