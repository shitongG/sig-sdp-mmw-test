# Audit Remaining Changes, README Formula Fixes, and Worktree Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the remaining modified files in the current workspace, update `README.md` to reflect any meaningful unsynced functionality, fix GitHub-incompatible math macros such as `\operatorname`, commit all intended changes, push them to GitHub, and remove worktree content that has already been synced to `main`.

**Architecture:** Treat this as a repository finalization pass. First inventory the remaining dirty workspace and classify each change as code, output artifact, documentation, or scratch. Then add README regression tests that lock formula compatibility and joint-scheduler documentation expectations. After that, update `README.md`, sync any still-unsynced code/output changes, verify, commit, push, and finally clean up already-synced worktrees or synced files inside them without touching unrelated work.

**Tech Stack:** Git, Markdown, pytest, Python, existing repo test suite

---

## File Structure

- Modify: `README.md`
  - Add missing documentation for any remaining unsynced behavior discovered in the workspace audit.
  - Replace GitHub-incompatible math macros such as `\operatorname`.
  - Re-check unified joint GA/HGA explanation after the audit.
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
  - Add or update README regression tests for formula compatibility and wording.
- Review and potentially sync remaining tracked files:
  - `sim_script/pd_mmw_template_ap_stats.py`
  - `sim_script/pd_mmw_template_ap_stats_config.json`
  - `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
  - `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
  - `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
  - `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
  - `sim_script/output_ga/`
  - `sim_script/output_ga_wifi_reschedule/`
- Review and potentially remove synced worktree content:
  - `.worktrees/joint-wifi-ble/`
  - any other `.worktrees/*` still present

## Task 1: Audit the remaining dirty workspace

**Files:**
- Review only: repository root

- [ ] **Step 1: Capture current workspace status**

Run:

```bash
git status --short
```

Expected: a concise list of remaining modified, deleted, or untracked files outside the already-pushed `joint_sched/` sync.

- [ ] **Step 2: Capture file-level diff summary for remaining tracked files**

Run:

```bash
git diff --stat -- \
  sim_script/pd_mmw_template_ap_stats.py \
  sim_script/pd_mmw_template_ap_stats_config.json \
  sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json \
  sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json \
  sim_script/tests/test_pd_mmw_template_ap_stats_logic.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_run.py \
  README.md
```

Expected: a focused diff summary that shows whether these remaining changes are real feature work or just incidental drift.

- [ ] **Step 3: Classify the remaining changes**

Create a short audit note in your working notes using this structure:

```text
Keep and sync:
- files that contain code or test logic not yet pushed
- output directories explicitly requested by the user

Document only:
- README changes needed to explain audited functionality

Ignore or leave untouched:
- unrelated editor config
- user-owned scratch state
```

- [ ] **Step 4: Inspect remaining output directories**

Run:

```bash
find sim_script/output_ga -maxdepth 2 -type f | sort
find sim_script/output_ga_wifi_reschedule -maxdepth 2 -type f | sort
```

Expected: a concrete view of which experimental outputs remain unsynced and may need to be preserved.

- [ ] **Step 5: Commit**

This task is audit-only. Do not commit yet.

## Task 2: Lock README formula compatibility with tests

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

- [ ] **Step 1: Write the failing test**

Add this test near the existing README regression tests:

```python
def test_readme_avoids_github_unsupported_math_macros(self):
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")

    self.assertNotIn("\\operatorname{", readme)
    self.assertIn("\\mathrm{", readme)
```

- [ ] **Step 2: Run the focused test to verify current state**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py::EventBlockExpansionTest::test_readme_avoids_github_unsupported_math_macros -q
```

Expected:
- FAIL if unsupported macros still exist
- PASS if a previous fix already removed them; in that case keep the test as regression coverage

- [ ] **Step 3: Add an explicit README regression for joint GA/HGA wording if needed**

If the wording is not already locked tightly enough, add this companion test:

```python
def test_readme_documents_joint_ga_hga_key_ideas(self):
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")

    assert "统一联合 GA" in readme
    assert "统一联合 HGA" in readme
    assert "染色体编码" in readme
    assert "WiFi floor" in readme
    assert "residual-hole" in readme
    assert "accept-if-better" in readme
```

- [ ] **Step 4: Run the focused README regression tests**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "readme"
```

Expected: all README-related tests run and report clear pass/fail status.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "test: lock README formula compatibility"
```

## Task 3: Update README based on audited remaining changes

**Files:**
- Modify: `README.md`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

- [ ] **Step 1: Remove or replace GitHub-incompatible math macros**

Search and replace unsupported macros such as:

```text
\operatorname{diag}(Y) -> \mathrm{diag}(Y)
\operatorname{OverlapArea}(...) -> \mathrm{OverlapArea}(...)
\operatorname{Area}(...) -> \mathrm{Area}(...)
```

If more unsupported macros appear, convert them to GitHub-safe forms such as `\mathrm{...}` or plain identifiers inside math blocks.

- [ ] **Step 2: Add README notes for any remaining unsynced code behavior**

If Task 1 finds real unsynced functionality in `sim_script/pd_mmw_template_ap_stats.py` or related configs/tests, add a short subsection documenting it. Use concrete bullets like:

```md
### X.X [Feature Name]

- 作用：
- 关键配置：
- 输出文件：
- 与 joint scheduler 的关系：
```

Do not add filler text; only document audited behavior that actually exists in code.

- [ ] **Step 3: Re-read the unified joint GA/HGA section and tighten wording**

Ensure the README explicitly states:

```text
- unified joint GA and HGA optimize the same mixed candidate-state space
- WiFi stripe expansion is diagnostic only
- whole-WiFi-state move operates on an entire periodic WiFi state
- accept-if-better move is applied only if WiFi floor is preserved and score improves
```

- [ ] **Step 4: Run README regression tests**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q -k "readme"
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "docs: finalize README formulas and joint scheduler notes"
```

## Task 4: Sync remaining code and experiment outputs

**Files:**
- Modify or add:
  - `sim_script/pd_mmw_template_ap_stats.py`
  - `sim_script/pd_mmw_template_ap_stats_config.json`
  - `sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json`
  - `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
  - `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
  - `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`
  - `sim_script/output_ga/*`
  - `sim_script/output_ga_wifi_reschedule/*`

- [ ] **Step 1: Review remaining tracked diffs one file at a time**

Run:

```bash
git diff -- sim_script/pd_mmw_template_ap_stats.py
git diff -- sim_script/pd_mmw_template_ap_stats_config.json
git diff -- sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json
git diff -- sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json
git diff -- sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git diff -- sim_script/tests/test_pd_mmw_template_ap_stats_run.py
```

Expected: enough information to decide whether each file should be committed now or left out.

- [ ] **Step 2: Stage only the audited remaining files that should ship**

Use explicit file paths, for example:

```bash
git add \
  sim_script/pd_mmw_template_ap_stats.py \
  sim_script/pd_mmw_template_ap_stats_config.json \
  sim_script/pd_mmw_template_ap_stats_macrocycle_hopping_9wifi_16ble.json \
  sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json \
  sim_script/tests/test_pd_mmw_template_ap_stats_logic.py \
  sim_script/tests/test_pd_mmw_template_ap_stats_run.py \
  sim_script/output_ga \
  sim_script/output_ga_wifi_reschedule
```

Stage only what the audit supports.

- [ ] **Step 3: Run targeted verification for the staged mainline scheduler changes**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q
```

Expected: PASS for the synced mainline scheduler logic and run-path tests.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: sync remaining scheduler changes and outputs"
```

## Task 5: Clean up worktrees that are fully synced to main

**Files:**
- Remove: synced worktrees only

- [ ] **Step 1: List existing worktrees**

Run:

```bash
git worktree list
```

Expected: a list of the main workspace plus any remaining feature worktrees.

- [ ] **Step 2: Verify each remaining worktree is fully synced**

For each non-main worktree, inspect:

```bash
git -C <worktree-path> status --short
git -C <worktree-path> log --oneline -n 5
```

Expected: only delete worktrees whose relevant changes are already on `main`.

- [ ] **Step 3: Remove synced worktrees**

Use:

```bash
git worktree remove <worktree-path>
```

If a linked branch is no longer needed:

```bash
git branch -D <branch-name>
```

Only remove worktrees that are confirmed synced.

- [ ] **Step 4: Re-list worktrees**

Run:

```bash
git worktree list
```

Expected: only unsynced or intentionally kept worktrees remain.

- [ ] **Step 5: Commit**

No git commit is required for worktree removal unless tracked files changed in the main workspace.

## Task 6: Final verification, commit remaining work, and push to GitHub

**Files:**
- Modify: none unless review finds final issues

- [ ] **Step 1: Run final repository checks**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
python -m py_compile ble_macrocycle_hopping_sdp.py sim_script/pd_mmw_template_ap_stats.py joint_sched/run_joint_wifi_ble_demo.py
```

Expected: PASS

- [ ] **Step 2: Review final git status**

Run:

```bash
git status --short
```

Expected: clean, or only intentionally untracked/scratch items that you explicitly chose not to ship.

- [ ] **Step 3: Push to GitHub**

Run:

```bash
git push origin main
```

Expected: remote `main` updates successfully.

- [ ] **Step 4: Record final handoff summary**

Write a short handoff note containing:

```text
- remaining files audited
- README formula/macros fixed
- remaining code/output synced
- worktrees removed
- tests executed
- final commit SHA
```

- [ ] **Step 5: Commit**

Only if final review found additional tracked changes after the previous commits:

```bash
git add -A
git commit -m "chore: finalize repo cleanup and README compatibility"
```

## Self-Review

- Spec coverage:
  - Audit remaining workspace changes: covered in Task 1 and Task 4.
  - Update README accordingly: covered in Task 3.
  - Fix GitHub-incompatible formulas: covered in Task 2 and Task 3.
  - Commit everything intended and push to GitHub: covered in Task 4 and Task 6.
  - Delete synced worktrees/files: covered in Task 5.
- Placeholder scan:
  - No `TODO`, `TBD`, or vague “handle later” language remains.
  - All commands and file paths are concrete.
- Type consistency:
  - README notation remains consistent with existing symbols `x_k`, `s_k`, `r_k`, `\ell_k`, `c_{k,m}`, and `Y`.
  - Joint scheduler terminology uses `WiFi floor`, `residual-hole`, `accept-if-better`, and `whole-WiFi-state move` consistently.

