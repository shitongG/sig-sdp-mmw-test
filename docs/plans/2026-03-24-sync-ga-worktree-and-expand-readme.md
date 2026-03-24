# Sync GA Worktree And Expand README Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Sync the verified GA worktree changes back into the main workspace and expand `README.md` with a more detailed, paper-style explanation of the genetic algorithm design.

**Architecture:** First verify the worktree is internally consistent and identify the exact commits/files to bring back. Then copy the code, config, tests, and docs from the `ble-ga-standalone` worktree into the main workspace, preserving the standalone GA module boundary and the new main-scheduler backend switch. Finally, deepen the README with a structured GA design section that explains encoding, candidate grouping, fitness, selection, crossover, mutation, elitism, and where the GA sits relative to SDP.

**Tech Stack:** Git worktree workflow, Python 3, pytest, Markdown documentation.

---

### Task 1: Verify the worktree state and collect the exact sync set

**Files:**
- Inspect: `.worktrees/ble-ga-standalone/ble_macrocycle_hopping_ga.py`
- Inspect: `.worktrees/ble-ga-standalone/ble_macrocycle_hopping_sdp.py`
- Inspect: `.worktrees/ble-ga-standalone/sim_script/pd_mmw_template_ap_stats.py`
- Inspect: `.worktrees/ble-ga-standalone/README.md`
- Inspect: `.worktrees/ble-ga-standalone/git log`

**Step 1: Write the checklist**

Create a checklist in execution notes for these expected sync items:
- standalone GA module exists in `ble_macrocycle_hopping_ga.py`
- standalone dispatcher supports `--solver ga`
- main scheduler supports `ble_schedule_backend = macrocycle_hopping_ga`
- tests exist for standalone GA and main backend integration
- README contains standalone GA usage notes
- only generated image artifacts remain uncommitted in the worktree

**Step 2: Run verification commands**

Run:

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/ble-ga-standalone status --short
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/ble-ga-standalone log --oneline --decorate -5
```

Expected:
- committed GA worktree history is visible
- only generated plot/image artifacts are still modified or untracked

**Step 3: Record the exact file sync list**

List the files that must be copied back to main:
- `ble_macrocycle_hopping_ga.py`
- `ble_macrocycle_hopping_sdp.py`
- `ble_macrocycle_hopping_sdp_config.json`
- `sim_script/pd_mmw_template_ap_stats.py`
- `sim_script/pd_mmw_template_ap_stats_config.json`
- `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
- `tests/test_ble_macrocycle_hopping_ga.py`
- `tests/test_ble_macrocycle_hopping_sdp.py`
- `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
- `README.md`

**Step 4: Commit**

No commit in this task.

### Task 2: Write failing verification for the main workspace before syncing

**Files:**
- Test: `tests/test_ble_macrocycle_hopping_ga.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Run targeted tests from the main workspace before copying files**

Run from `/data/home/Jie_Wan/mycode/sig-sdp-mmw-test`:

```bash
env PYTHONPATH=$PWD pytest tests/test_ble_macrocycle_hopping_ga.py -q
env PYTHONPATH=$PWD pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "macrocycle_hopping_ga or ble_ga"
```

Expected:
- FAIL or file-not-found, demonstrating the main workspace is missing the worktree GA implementation.

**Step 2: Save the failure notes**

Record which tests fail and why:
- missing file
- missing backend choice
- missing config keys
- missing README coverage is not a test failure but a docs gap

**Step 3: Commit**

No commit in this task.

### Task 3: Sync code, config, and tests from the worktree to main workspace

**Files:**
- Copy/modify: `ble_macrocycle_hopping_ga.py`
- Copy/modify: `ble_macrocycle_hopping_sdp.py`
- Copy/modify: `ble_macrocycle_hopping_sdp_config.json`
- Copy/modify: `sim_script/pd_mmw_template_ap_stats.py`
- Copy/modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Copy/modify: `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Copy/modify: `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
- Copy/modify: `tests/test_ble_macrocycle_hopping_ga.py`
- Copy/modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Copy/modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Copy/modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Copy the files exactly from worktree to main workspace**

Use direct file copy or checkout-by-path logic; do not manually retype large edits.

**Step 2: Run syntax verification**

Run:

```bash
python -m py_compile \
  ble_macrocycle_hopping_ga.py \
  ble_macrocycle_hopping_sdp.py \
  sim_script/pd_mmw_template_ap_stats.py
```

Expected: PASS.

**Step 3: Run focused tests**

Run:

```bash
env PYTHONPATH=$PWD pytest \
  tests/test_ble_macrocycle_hopping_sdp.py \
  tests/test_ble_macrocycle_hopping_ga.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_logic.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_run.py \
  -q -k "ble_ga or macrocycle_hopping_ga or dispatches_to_ga or ga_solver_fields or solver_flag or apply_ble_schedule_backend or cli_override_to_ga"
```

Expected: PASS.

**Step 4: Commit**

```bash
git add \
  ble_macrocycle_hopping_ga.py \
  ble_macrocycle_hopping_sdp.py \
  ble_macrocycle_hopping_sdp_config.json \
  sim_script/pd_mmw_template_ap_stats.py \
  sim_script/pd_mmw_template_ap_stats_config.json \
  sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json \
  sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json \
  tests/test_ble_macrocycle_hopping_ga.py \
  tests/test_ble_macrocycle_hopping_sdp.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_logic.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_run.py

git commit -m "feat: sync standalone and main GA BLE schedulers"
```

### Task 4: Expand README with a detailed GA design section

**Files:**
- Modify: `README.md`

**Step 1: Write the failing doc checklist**

The README must explicitly explain all of the following:
- why GA was added: high-density BLE scenarios where SDP is too slow
- chromosome definition: one gene per BLE pair, each gene choosing one local `(offset, pattern)` candidate
- candidate grouping: `PairCandidateGroup`
- fitness definition: BLE-BLE collision cost plus WiFi external interference cost when present
- population initialization
- selection: tournament selection
- crossover: single-point crossover
- mutation: local re-sampling within one pair’s candidate list
- elitism and `fitness_history`
- standalone GA path vs main `macrocycle_hopping_ga` backend
- approximation/performance tradeoff relative to SDP

**Step 2: Verify the current README is still missing at least one detail**

Run:

```bash
rg -n -- "tournament|elitism|fitness_history|PairCandidateGroup|mutation|crossover|approximation|近似" README.md
```

Expected: at least one algorithm detail is missing or under-explained.

**Step 3: Write minimal implementation**

Add a dedicated subsection, for example:
- `2.3 BLE-only GA standalone backend`
- `3.4 macrocycle_hopping_ga BLE 后端`
- `3.4.1 GA encoding`
- `3.4.2 GA fitness`
- `3.4.3 GA operators`
- `3.4.4 Why GA is faster than SDP in dense cases`

Use GitHub-friendly Markdown and math only.

Suggested bullets to include verbatim or near-verbatim:

```markdown
- 染色体长度等于 BLE pair 数量。
- 第 k 个基因只在第 k 个 pair 的局部候选状态集合中取值。
- 一个候选状态就是一个 `(offset, pattern)` 组合。
- 适应度 = BLE-BLE 碰撞代价 + 外部干扰代价。
- 选择算子使用 tournament selection。
- 交叉算子使用 single-point crossover。
- 变异算子只会在该 pair 的局部候选集合内重新采样。
- 每代保留 elite 个最优染色体，以避免最优解退化。
```

**Step 4: Verify README coverage**

Run:

```bash
rg -n -- "tournament|elitism|fitness_history|PairCandidateGroup|mutation|crossover|approximation|近似|macrocycle_hopping_ga" README.md
```

Expected: matching lines for all major GA design concepts.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: expand genetic algorithm design in README"
```

### Task 5: Run end-to-end verification in the main workspace

**Files:**
- Verify only

**Step 1: Run the standalone GA command**

Run:

```bash
env PYTHONPATH=$PWD python ble_macrocycle_hopping_sdp.py --solver ga
```

Expected: PASS and generate/update `ble_macrocycle_hopping_sdp_schedule.png`.

**Step 2: Run the main GA backend smoke command**

Run:

```bash
cat > /tmp/pd_mmw_macrocycle_ga_smoke.json <<'EOF'
{
  "cell_size": 1,
  "seed": 123,
  "mmw_nit": 1,
  "mmw_eta": 0.05,
  "ble_schedule_backend": "macrocycle_hopping_ga",
  "ble_ga_population_size": 12,
  "ble_ga_generations": 6,
  "ble_ga_mutation_rate": 0.1,
  "ble_ga_crossover_rate": 0.8,
  "ble_ga_elite_count": 1,
  "ble_ga_seed": 7,
  "pair_generation_mode": "manual",
  "output_dir": "/tmp/pd_mmw_macrocycle_ga_out",
  "pair_parameters": [
    {
      "pair_id": 0,
      "office_id": 0,
      "radio": "ble",
      "channel": 8,
      "priority": 1.0,
      "release_time_slot": 0,
      "deadline_slot": 31,
      "start_time_slot": 0,
      "ble_anchor_slot": 0,
      "ble_timing_mode": "auto"
    },
    {
      "pair_id": 1,
      "office_id": 0,
      "radio": "wifi",
      "channel": 0,
      "priority": 1.0,
      "release_time_slot": 0,
      "deadline_slot": 15,
      "start_time_slot": 0,
      "wifi_anchor_slot": 0,
      "wifi_period_slots": 16,
      "wifi_tx_slots": 2
    }
  ]
}
EOF

env PYTHONPATH=$PWD python sim_script/pd_mmw_template_ap_stats.py --config /tmp/pd_mmw_macrocycle_ga_smoke.json
```

Expected: PASS and print `ble_schedule_backend = macrocycle_hopping_ga`.

**Step 3: Run the targeted regression suite again**

Run:

```bash
env PYTHONPATH=$PWD pytest \
  tests/test_ble_macrocycle_hopping_sdp.py \
  tests/test_ble_macrocycle_hopping_ga.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_logic.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_run.py \
  -q -k "ble_ga or macrocycle_hopping_ga or dispatches_to_ga or ga_solver_fields or solver_flag or apply_ble_schedule_backend or cli_override_to_ga"
```

Expected: PASS.

**Step 4: Commit**

No new code changes expected.

### Task 6: Sync completion and cleanup decision

**Files:**
- Optionally remove or ignore: generated `ble_macrocycle_hopping_sdp_schedule.png`

**Step 1: Check main workspace git status**

Run:

```bash
git status --short
```

Expected:
- only intended code/docs changes are staged or committed
- generated plot artifacts are either intentionally excluded or explicitly acknowledged

**Step 2: Decide whether to keep generated PNG artifacts**

If the PNG is not meant to be versioned, leave it uncommitted and note that in the final summary.

**Step 3: Final summary**

Report:
- which worktree commits were synced
- which new main-workspace commits were created
- whether the worktree can now be deleted
