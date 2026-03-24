# BLE GA Standalone And Main Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the standalone BLE GA workflow with JSON/README support, then add the GA solver as an optional BLE backend in the main `pd_mmw_template_ap_stats.py` pipeline.

**Architecture:** Keep the genetic algorithm in the dedicated module `ble_macrocycle_hopping_ga.py` and treat `ble_macrocycle_hopping_sdp.py` as a thin standalone dispatcher that can select `sdp` or `ga`. For main-pipeline integration, mirror the existing `macrocycle_hopping_sdp` adapter path: build BLE candidate states from `env`, call the GA solver, write the selected per-CE channels back to `env.pair_ble_ce_channels`, and preserve legacy behavior behind explicit config switches.

**Tech Stack:** Python 3, NumPy, pytest, existing BLE standalone solver utilities, existing `sim_script` JSON config system.

---

### Task 1: Finalize standalone GA JSON configuration behavior

**Files:**
- Modify: `ble_macrocycle_hopping_sdp.py`
- Modify: `ble_macrocycle_hopping_sdp_config.json`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

**Step 1: Write the failing tests**

Add tests that assert:

```python
def test_run_ble_macrocycle_hopping_sdp_uses_ga_solver_from_json_config(tmp_path):
    config = {
        "solver": "ga",
        "ga_population_size": 8,
        "ga_generations": 4,
        "ga_mutation_rate": 0.1,
        "ga_crossover_rate": 0.8,
        "ga_elite_count": 1,
        "ga_seed": 5,
    }
    config_path = tmp_path / "ga.json"
    config_path.write_text(json.dumps(config))
    result = MODULE.run_ble_macrocycle_hopping_sdp(config_path)
    assert result["runtime"].solver == "ga"
    assert "ga_result" in result


def test_default_ble_standalone_config_exposes_ga_fields():
    config = MODULE.merge_or_load_config(None)
    assert "solver" in config
    assert "ga_population_size" in config
    assert "ga_generations" in config
```

**Step 2: Run tests to verify they fail**

Run: `env PYTHONPATH=$PWD pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "uses_ga_solver_from_json_config or exposes_ga_fields"`
Expected: FAIL because at least one path still does not assert the complete standalone config behavior.

**Step 3: Write minimal implementation**

Update `ble_macrocycle_hopping_sdp.py` so that:
- `merge_or_load_config(None)` returns defaults including GA fields.
- `load_ble_standalone_config(...)` preserves GA settings from JSON.
- `run_ble_macrocycle_hopping_sdp(config_path)` respects `solver="ga"` from JSON without requiring CLI override.
- `build_demo_standalone_config()` and `ble_macrocycle_hopping_sdp_config.json` expose matching GA defaults.

Update `ble_macrocycle_hopping_sdp_config.json` to document these keys clearly:

```json
{
  "solver": "ga",
  "ga_population_size": 48,
  "ga_generations": 80,
  "ga_mutation_rate": 0.1,
  "ga_crossover_rate": 0.85,
  "ga_elite_count": 2,
  "ga_seed": 7
}
```

**Step 4: Run tests to verify they pass**

Run: `env PYTHONPATH=$PWD pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "uses_ga_solver_from_json_config or exposes_ga_fields or solver_flag or dispatches_to_ga"`
Expected: PASS.

**Step 5: Commit**

```bash
git add ble_macrocycle_hopping_sdp.py ble_macrocycle_hopping_sdp_config.json tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: finalize standalone GA config behavior"
```

### Task 2: Document standalone GA usage in README

**Files:**
- Modify: `README.md`
- Test: manual verification only

**Step 1: Write the failing doc checklist**

Create a local checklist in the plan execution notes requiring README to include:
- `python ble_macrocycle_hopping_sdp.py --solver ga`
- `python ble_macrocycle_hopping_sdp.py --config ble_macrocycle_hopping_sdp_config.json`
- explanation of GA chromosome = one `(offset, pattern)` choice per BLE pair
- explanation that GA is an approximate fast alternative to SDP
- explanation of GA config keys and intended scaling tradeoffs

**Step 2: Verify README is currently missing at least one item**

Run: `rg -n "ga_population_size|--solver ga|genetic|遗传|approximate|近似" README.md`
Expected: Missing or incomplete coverage for at least one required item.

**Step 3: Write minimal implementation**

Add a README subsection such as:
- `3.x BLE-only GA backend`
- algorithm summary
- chromosome / fitness / mutation / elitism explanation
- standalone commands
- tuning guidance for population size and generations

Use standard GitHub-renderable Markdown/math only.

**Step 4: Verify the README coverage**

Run: `rg -n "ga_population_size|--solver ga|GA|遗传算法|population|generations" README.md`
Expected: Matching lines for all major GA usage and tuning topics.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add standalone GA usage guide"
```

### Task 3: Add main-pipeline config surface for a GA BLE backend

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing tests**

Add tests such as:

```python
def test_default_config_accepts_macrocycle_hopping_ga_backend():
    config = MODULE.merge_config_with_defaults({
        "ble_schedule_backend": "macrocycle_hopping_ga",
        "ble_ga_population_size": 32,
        "ble_ga_generations": 40,
        "ble_ga_mutation_rate": 0.1,
        "ble_ga_crossover_rate": 0.85,
        "ble_ga_elite_count": 2,
        "ble_ga_seed": 7,
    })
    assert config["ble_schedule_backend"] == "macrocycle_hopping_ga"


def test_parse_args_accepts_ble_ga_cli_overrides():
    args = MODULE.parse_args([
        "--ble-schedule-backend", "macrocycle_hopping_ga",
        "--ble-ga-population-size", "16",
        "--ble-ga-generations", "20",
    ])
    assert args.ble_schedule_backend == "macrocycle_hopping_ga"
    assert args.ble_ga_population_size == 16
```

**Step 2: Run tests to verify they fail**

Run: `env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "macrocycle_hopping_ga or ble_ga"`
Expected: FAIL because config/CLI keys are not present yet.

**Step 3: Write minimal implementation**

In `sim_script/pd_mmw_template_ap_stats.py`:
- extend `DEFAULT_CONFIG`
- extend config validation / merge normalization
- extend `parse_args()` with:
  - `--ble-ga-population-size`
  - `--ble-ga-generations`
  - `--ble-ga-mutation-rate`
  - `--ble-ga-crossover-rate`
  - `--ble-ga-elite-count`
  - `--ble-ga-seed`
- allow `ble_schedule_backend` choices:
  - `legacy`
  - `macrocycle_hopping_sdp`
  - `macrocycle_hopping_ga`

In JSON configs, add documented GA keys while keeping legacy defaults unchanged unless the file is explicitly for GA experiments.

**Step 4: Run tests to verify they pass**

Run: `env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "macrocycle_hopping_ga or ble_ga"`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: add GA backend config to main scheduler"
```

### Task 4: Implement BLE GA adapter for the main scheduler

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Reuse: `ble_macrocycle_hopping_ga.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the failing tests**

Add tests that mock the GA module and assert the main scheduler uses it correctly:

```python
def test_apply_ble_schedule_backend_dispatches_to_macrocycle_hopping_ga(monkeypatch):
    env = build_small_test_env_with_ble_pairs()
    called = {}

    def fake_solver(**kwargs):
        called.update(kwargs)
        return FakeGAResult(...)

    monkeypatch.setattr(MODULE.ble_macrocycle_hopping_ga, "solve_ble_hopping_schedule_ga", fake_solver)
    MODULE.apply_ble_schedule_backend(env, {"ble_schedule_backend": "macrocycle_hopping_ga", ...})
    assert called["population_size"] == 16
    assert len(env.pair_ble_ce_channels) > 0
```

Also add a WiFi-first test if that path is meant to remain exclusive to the SDP solver for now.

**Step 2: Run tests to verify they fail**

Run: `env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "dispatches_to_macrocycle_hopping_ga"`
Expected: FAIL because the backend dispatch does not exist yet.

**Step 3: Write minimal implementation**

In `sim_script/pd_mmw_template_ap_stats.py`:
- add a helper analogous to the SDP adapter, for example `solve_ble_hopping_ga_for_env(...)`
- build `pair_configs`, `cfg_dict`, `pattern_dict`, `states`, `A_k` the same way as the SDP path
- call `ble_macrocycle_hopping_ga.solve_ble_hopping_schedule_ga(...)`
- write the selected CE channels back to `env.set_ble_ce_channel_map(...)`
- keep the return shape similar to the SDP helper so downstream code stays simple

Guardrails:
- preserve legacy backend behavior exactly
- preserve SDP backend behavior exactly
- only force `ble_channel_mode = "per_ce"` for GA if needed, and do it explicitly
- if `wifi_first_ble_scheduling` is enabled but GA does not yet model WiFi external interference, either document and reject the combination or adapt the existing interference-block builder before enabling it

**Step 4: Run tests to verify they pass**

Run: `env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "dispatches_to_macrocycle_hopping_ga or apply_ble_schedule_backend"`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: integrate BLE GA backend into main scheduler"
```

### Task 5: Add run-level tests for the GA backend in the main pipeline

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- Optionally modify: `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`

**Step 1: Write the failing tests**

Add a smoke test similar to the existing macrocycle hopping backend tests:

```python
def test_macrocycle_hopping_ga_backend_json_config_runs(tmp_path):
    config_path = write_small_ga_backend_config(tmp_path)
    completed = run_script(config_path)
    assert completed.returncode == 0
    assert "macrocycle_hopping_ga" in completed.stdout
```

Add one CLI override test if useful:

```python
def test_ble_schedule_backend_cli_override_to_ga_runs(tmp_path):
    ...
```

**Step 2: Run tests to verify they fail**

Run: `env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "macrocycle_hopping_ga"`
Expected: FAIL because the smoke path is not implemented yet.

**Step 3: Write minimal implementation**

- Reuse the smallest stable config possible.
- If necessary, add a dedicated smoke JSON file with a few WiFi/BLE pairs and bounded candidate counts.
- Ensure the script prints backend selection clearly enough for the smoke assertion.

**Step 4: Run tests to verify they pass**

Run: `env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "macrocycle_hopping_ga"`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
git commit -m "test: add main-pipeline GA backend smoke coverage"
```

### Task 6: End-to-end verification and documentation cleanup

**Files:**
- Modify: `README.md`
- Optionally modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Optionally modify: `docs/plans/2026-03-24-ble-ga-standalone-and-main-integration.md`

**Step 1: Add final README usage entries for the main scheduler**

Document:
- how to enable `macrocycle_hopping_ga`
- which GA knobs exist in the main config
- current limitation vs SDP path, especially around WiFi-first interference handling if applicable

**Step 2: Run end-to-end commands**

Run these exact commands:

```bash
env PYTHONPATH=$PWD pytest tests/test_ble_macrocycle_hopping_sdp.py tests/test_ble_macrocycle_hopping_ga.py -q
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "ble_ga or macrocycle_hopping_ga or apply_ble_schedule_backend"
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "macrocycle_hopping_ga"
env PYTHONPATH=$PWD python ble_macrocycle_hopping_sdp.py --solver ga
env PYTHONPATH=$PWD python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
```

Expected:
- all selected tests PASS
- standalone GA run succeeds
- main GA backend run succeeds or fails only for a documented unsupported combination that is now explicitly rejected

**Step 3: Final commit**

```bash
git add README.md sim_script/pd_mmw_template_ap_stats_config.json docs/plans/2026-03-24-ble-ga-standalone-and-main-integration.md
git commit -m "docs: document GA standalone and main backend usage"
```
