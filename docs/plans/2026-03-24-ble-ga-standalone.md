# BLE GA Standalone Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a BLE-only standalone genetic-algorithm solver that searches over `(offset, pattern)` assignments as a fast heuristic alternative to the current SDP + pruning path in `ble_macrocycle_hopping_sdp.py`.

**Architecture:** Keep the current candidate-state generation, event expansion, collision-cost computation, config loading, and plotting pipeline. Add a GA search layer that works directly on per-pair genes, where each gene selects one candidate state index from that pair's local candidate set. The new GA path lives first in the standalone BLE-only solver and is invoked by a new solver/backend choice, without yet integrating it into `sim_script/pd_mmw_template_ap_stats.py`.

**Tech Stack:** Python, NumPy, existing BLE candidate-state utilities in `ble_macrocycle_hopping_sdp.py`, `pytest`/`unittest`, JSON config loading, matplotlib output already in repo.

---

### Task 1: Freeze the GA interface and config surface

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Modify: `ble_macrocycle_hopping_sdp_config.json`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

Add tests that define the public surface for the GA path.

```python
def test_merge_or_load_config_accepts_ga_solver_fields(tmp_path):
    config = {
        "solver": "ga",
        "ga_population_size": 24,
        "ga_generations": 30,
        "ga_mutation_rate": 0.15,
        "ga_crossover_rate": 0.8,
        "ga_elite_count": 2,
        "ga_seed": 7,
    }
    # load config and assert fields are preserved


def test_parse_args_accepts_solver_flag():
    args = MODULE.parse_args(["--solver", "ga"])
    assert args.solver == "ga"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "ga_solver_fields or solver_flag"`
Expected: FAIL because the standalone solver does not yet accept GA config/CLI fields.

**Step 3: Write minimal implementation**

Add the minimal config/CLI plumbing:
- new CLI flag `--solver {sdp,ga}`
- new config fields:
  - `solver`
  - `ga_population_size`
  - `ga_generations`
  - `ga_mutation_rate`
  - `ga_crossover_rate`
  - `ga_elite_count`
  - `ga_seed`
- default solver remains `sdp`
- JSON config continues to work for the existing SDP path unchanged

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "ga_solver_fields or solver_flag"`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp_config.json
git commit -m "feat: add GA solver config surface"
```

### Task 2: Make candidate-state groups explicit for chromosome encoding

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

Add tests for a helper that converts `A_k` into a GA-friendly per-pair candidate table.

```python
def test_build_pair_candidate_groups_returns_local_choice_lists():
    groups = MODULE.build_pair_candidate_groups(pair_ids=[0, 1], A_k={0: [0, 1], 1: [2]}, states=states)
    assert groups[0] == [0, 1]
    assert groups[1] == [2]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k pair_candidate_groups`
Expected: FAIL with missing helper.

**Step 3: Write minimal implementation**

Implement a small helper that:
- takes `pair_ids`, `A_k`, `states`
- returns deterministic per-pair candidate index lists in a stable order
- validates every pair has at least one candidate state

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k pair_candidate_groups`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "refactor: expose pair candidate groups for GA encoding"
```

### Task 3: Define chromosome decoding and fitness calculation

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

Add tests that define how a chromosome maps to a schedule and objective.

```python
def test_decode_chromosome_selects_one_state_per_pair():
    chromosome = np.array([1, 0], dtype=int)
    selected = MODULE.decode_ga_chromosome(chromosome, pair_ids=[0, 1], pair_candidate_groups={0: [0, 1], 1: [2]}, states=states)
    assert selected[0] == states[1]
    assert selected[1] == states[2]


def test_ga_fitness_matches_total_collision_plus_external_penalty():
    score = MODULE.evaluate_ga_chromosome(...)
    assert score == pytest.approx(expected_value)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "decode_ga_chromosome or ga_fitness"`
Expected: FAIL because decode/evaluate helpers do not yet exist.

**Step 3: Write minimal implementation**

Implement helpers:
- `decode_ga_chromosome(...)`
- `evaluate_ga_chromosome(...)`

Fitness should be minimized and should include:
- BLE-BLE total collision cost via existing collision helpers
- external interference penalty via existing `external_interference_cost_for_state(...)`
- optional large penalty when a chromosome picks a forbidden WiFi-overlapping state for a pair that has a zero-penalty alternative

Keep this logic pure and reusable.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "decode_ga_chromosome or ga_fitness"`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "feat: add GA chromosome decode and fitness helpers"
```

### Task 4: Implement GA population initialization

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
def test_initialize_ga_population_respects_pair_local_choice_ranges():
    pop = MODULE.initialize_ga_population(
        pair_ids=[0, 1],
        pair_candidate_groups={0: [0, 1, 2], 1: [3, 4]},
        population_size=8,
        rng=np.random.default_rng(7),
    )
    assert pop.shape == (8, 2)
    assert np.all(pop[:, 0] >= 0) and np.all(pop[:, 0] < 3)
    assert np.all(pop[:, 1] >= 0) and np.all(pop[:, 1] < 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k initialize_ga_population`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement population initialization over local gene choices, not global state ids. Add one deterministic seed path for reproducibility.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k initialize_ga_population`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "feat: add GA population initialization"
```

### Task 5: Implement crossover, mutation, and elitism

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
def test_mutation_keeps_gene_values_in_local_range():
    child = MODULE.mutate_ga_chromosome(
        chromosome=np.array([0, 1, 2]),
        gene_choice_sizes=[2, 3, 4],
        mutation_rate=1.0,
        rng=np.random.default_rng(3),
    )
    assert 0 <= child[0] < 2
    assert 0 <= child[1] < 3
    assert 0 <= child[2] < 4


def test_crossover_preserves_length_and_local_gene_space():
    child_a, child_b = MODULE.crossover_ga_chromosomes(...)
    assert child_a.shape == parent_a.shape
    assert child_b.shape == parent_b.shape
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "mutation_keeps_gene_values or crossover_preserves"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:
- one-point or uniform crossover
- per-gene mutation
- elitism helper that preserves top `elite_count` chromosomes

Keep the first version simple; do not add adaptive rates yet.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "mutation_keeps_gene_values or crossover_preserves"`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "feat: add GA evolution operators"
```

### Task 6: Implement the main GA solver loop

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
def test_solve_ble_hopping_schedule_ga_returns_selected_schedule_dict():
    result = MODULE.solve_ble_hopping_schedule_ga(
        pair_configs=pair_configs,
        cfg_dict=cfg_dict,
        pattern_dict=pattern_dict,
        pair_ids=[0, 1],
        A_k=A_k,
        states=states,
        num_channels=37,
        ga_population_size=16,
        ga_generations=20,
        ga_mutation_rate=0.1,
        ga_crossover_rate=0.8,
        ga_elite_count=2,
        ga_seed=7,
    )
    assert set(result["selected"]) == {0, 1}
    assert "best_fitness" in result
    assert "fitness_history" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k solve_ble_hopping_schedule_ga`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement `solve_ble_hopping_schedule_ga(...)` with:
- deterministic RNG from `ga_seed`
- initialization
- fitness evaluation
- tournament selection or rank selection
- crossover + mutation + elitism
- best-solution tracking
- final decode into the same return shape currently used by `solve_ble_hopping_schedule(...)`:
  - `selected`
  - `blocks`
  - `overlap_blocks`
  - `ce_channel_map`
  - `objective_value` or `best_fitness`

Do not remove or rewrite the SDP path.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k solve_ble_hopping_schedule_ga`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "feat: add standalone BLE genetic solver"
```

### Task 7: Wire the standalone entrypoint to choose SDP or GA

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Modify: `ble_macrocycle_hopping_sdp_config.json`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

```python
def test_run_ble_macrocycle_hopping_sdp_dispatches_to_ga_solver_when_requested(monkeypatch):
    calls = {}
    def fake_ga(**kwargs):
        calls["kwargs"] = kwargs
        return fake_result
    monkeypatch.setattr(MODULE, "solve_ble_hopping_schedule_ga", fake_ga)
    result = MODULE.run_ble_macrocycle_hopping_sdp(config_path=config_path)
    assert calls["kwargs"]["ga_generations"] == 12
    assert result is fake_result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k dispatches_to_ga_solver`
Expected: FAIL

**Step 3: Write minimal implementation**

Update standalone execution flow so that:
- `solver = sdp` -> current path unchanged
- `solver = ga` -> call `solve_ble_hopping_schedule_ga(...)`
- output printing and plotting reuse the existing post-processing path

Also update `ble_macrocycle_hopping_sdp_config.json` to include a documented GA example.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k dispatches_to_ga_solver`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp_config.json
git commit -m "feat: wire standalone BLE solver selection"
```

### Task 8: Add a baseline quality/performance regression test

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Optionally Modify: `ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing test**

Add a bounded regression that checks the GA is reproducible and returns a finite result on a medium-sized synthetic instance.

```python
def test_ga_solver_is_reproducible_for_fixed_seed():
    result_a = MODULE.solve_ble_hopping_schedule_ga(..., ga_seed=11)
    result_b = MODULE.solve_ble_hopping_schedule_ga(..., ga_seed=11)
    assert result_a["best_fitness"] == result_b["best_fitness"]
    assert result_a["ce_channel_map"] == result_b["ce_channel_map"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k reproducible_for_fixed_seed`
Expected: FAIL

**Step 3: Write minimal implementation**

Make any RNG plumbing deterministic enough to satisfy the test. Keep the implementation simple and local.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k reproducible_for_fixed_seed`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp.py
git commit -m "test: lock GA solver reproducibility"
```

### Task 9: Document the GA path and how to compare it with SDP

**Files:**
- Modify: `README.md`
- Modify: `ble_macrocycle_hopping_sdp_config.json`
- Test: none

**Step 1: Write the documentation update**

Add a short subsection under the BLE-only standalone solver that explains:
- GA optimizes over `(offset, pattern)` candidate states
- GA is a heuristic alternative to SDP
- GA does not guarantee global optimality
- when to prefer GA over SDP
- the meaning of each GA hyperparameter

Include two runnable examples:

```bash
python ble_macrocycle_hopping_sdp.py --solver sdp
python ble_macrocycle_hopping_sdp.py --solver ga --config ble_macrocycle_hopping_sdp_config.json
```

**Step 2: Sanity-check the examples**

Run:
- `python ble_macrocycle_hopping_sdp.py --help`
- `python ble_macrocycle_hopping_sdp.py --solver ga --config ble_macrocycle_hopping_sdp_config.json`

Expected:
- CLI help shows `--solver`
- GA example completes and prints a valid schedule summary

**Step 3: Commit**

```bash
git add README.md ble_macrocycle_hopping_sdp_config.json ble_macrocycle_hopping_sdp.py
git commit -m "docs: describe standalone BLE GA solver"
```

### Task 10: Final regression sweep

**Files:**
- Modify: none unless a failure is found
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Run the focused BLE test suite**

Run: `pytest tests/test_ble_macrocycle_hopping_sdp.py -q`
Expected: PASS

**Step 2: Run one standalone SDP smoke and one GA smoke**

Run:

```bash
python ble_macrocycle_hopping_sdp.py --solver sdp
python ble_macrocycle_hopping_sdp.py --solver ga --config ble_macrocycle_hopping_sdp_config.json
```

Expected:
- both commands finish
- both emit a schedule summary and PNG
- GA path does not require `max_offsets_per_pair` pruning to be meaningful, though it may still reuse candidate pruning as an optional speed knob

**Step 3: Final commit if needed**

```bash
git add -A
git commit -m "chore: finalize standalone BLE GA solver"
```
