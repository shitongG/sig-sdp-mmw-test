# Sync Joint Worktree and README GA/HGA Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Review the `joint-wifi-ble` worktree, sync validated code and experiment outputs back to main, strengthen `README.md` so the unified joint GA/HGA principles are clearly documented in paper style, and then push the final result to GitHub.

**Architecture:** First inventory the worktree delta and identify exactly which files should be synced versus ignored. Then add focused README regression tests for unified joint GA/HGA documentation, patch the README until the tests pass, and only after that merge the worktree content into the main workspace. Finish with verification, git review, commit, push, and an explicit note about which experiment outputs were intentionally synced.

**Tech Stack:** Git worktrees, Markdown, pytest, Python, existing repo test suite

---

## File Structure

- Modify: `README.md`
  - Strengthen the unified joint `GA` / `HGA` sections with clearer algorithm principles, chromosome encoding, WiFi floor protection, residual-hole repair, and accept-if-better WiFi local move logic.
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
  - Add README regression checks for joint GA/HGA documentation so later edits do not silently weaken the paper-style explanation.
- Review and selectively sync from worktree:
  - `.worktrees/joint-wifi-ble/joint_sched/*.py`
  - `.worktrees/joint-wifi-ble/joint_sched/tests/*.py`
  - `.worktrees/joint-wifi-ble/README.md`
  - `.worktrees/joint-wifi-ble/joint_sched/output_*/*`
  - `.worktrees/joint-wifi-ble/joint_sched/experiments/*`
- Modify or create in main workspace after sync:
  - `joint_sched/...`
  - `joint_sched/tests/...`
  - `joint_sched/output...` / `joint_sched/experiments...` as needed

## Task 1: Audit the joint worktree before syncing

**Files:**
- Review only: `.worktrees/joint-wifi-ble`

- [ ] **Step 1: Capture the worktree status**

Run:

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble status --short
```

Expected: a concise list of modified, added, or untracked files in the worktree.

- [ ] **Step 2: Capture the diff summary against main**

Run:

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble diff --stat
```

Expected: a file-level summary showing which code, docs, and output artifacts differ from main.

- [ ] **Step 3: Record the files to sync**

Create a short checklist in your notes with these categories:

```text
Code to sync:
- joint_sched/*.py
- joint_sched/tests/*.py

Docs to sync:
- README.md changes related to joint GA/HGA

Outputs to sync:
- joint_sched/output_compare_*
- joint_sched/experiments/*

Do not sync:
- transient tmp outputs
- unrelated worktree-only scratch files
```

- [ ] **Step 4: Review the worktree README delta**

Run:

```bash
git -C /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble diff -- README.md
```

Expected: a readable diff showing exactly what the worktree added to README for joint scheduling.

- [ ] **Step 5: Commit**

This task is audit-only. Do not commit yet.

## Task 2: Add failing regression tests for README joint GA/HGA documentation

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

- [ ] **Step 1: Write the failing test**

Add this test near the existing README documentation tests:

```python
def test_readme_documents_unified_joint_ga_hga_principles(self):
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")

    self.assertIn("统一联合 GA", readme)
    self.assertIn("统一联合 HGA", readme)
    self.assertIn("染色体编码", readme)
    self.assertIn("WiFi floor", readme)
    self.assertIn("residual-hole", readme)
    self.assertIn("accept-if-better", readme)
    self.assertIn("whole-WiFi-state move", readme)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py::EventBlockExpansionTest::test_readme_documents_unified_joint_ga_hga_principles -q
```

Expected: FAIL because the current README wording is not explicit enough.

- [ ] **Step 3: Save the test in minimal form**

Use the exact test body above. Do not add broader assertions yet.

- [ ] **Step 4: Re-run the test to verify failure reason**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py::EventBlockExpansionTest::test_readme_documents_unified_joint_ga_hga_principles -q
```

Expected: FAIL on missing README content, not on import or syntax issues.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "test: require README joint GA HGA documentation"
```

## Task 3: Strengthen README joint GA/HGA algorithm explanation

**Files:**
- Modify: `README.md`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

- [ ] **Step 1: Add a paper-style subsection under the joint scheduling documentation**

Insert a new subsection in `README.md` that explicitly describes unified joint `GA/HGA` using this content pattern:

```md
#### X.X 统一联合 GA / HGA 的算法思想

联合调度不再采用“先 WiFi、后 BLE”的顺序式决策，而是为每个任务 `k` 构造统一候选状态集合 `\mathcal{A}_k`，并在同一搜索空间中同时决定：

```math
x_{k,a} \in \{0,1\}, \qquad a \in \mathcal{A}_k,
\qquad
\sum_{a \in \mathcal{A}_k} x_{k,a} = 1.
```

其中候选状态 `a` 对 WiFi 表示一个周期流 state（channel, offset, period, width），对 BLE 表示一个跳频 state（offset, pattern, CE sequence）。

#### X.X.X 染色体编码

在联合 GA / HGA 中，染色体按 pair 或 task 编码。第 `k` 个基因只允许取 `\mathcal{A}_k` 中的一个状态索引，因此单个染色体天然满足“一任务一状态”约束。若选择 `idle` 状态，则该任务在本轮调度中被舍弃。

#### X.X.X 目标函数与保护约束

联合搜索首先满足硬无碰撞约束，其次优化 payload 与填充度；在 faithful protected 实验中，还要求：

```math
P_{\mathrm{WiFi}}(x) \ge P_{\mathrm{WiFi}}^{\mathrm{baseline}}.
```

这就是 README 中所说的 WiFi floor。

#### X.X.X HGA 相比 GA 的增强

统一联合 HGA 并不退回到顺序调度，而是在同一联合状态空间中加入三类启发式：

1. residual-hole diagnosis：把当前已选状态映射成时频占用，并识别可容纳 BLE 周期事件序列的剩余空洞；
2. whole-WiFi-state move：把 WiFi 周期流当作整体 state 移动，而不是拆成独立小任务；
3. accept-if-better local move：对当前 best 解直接尝试 WiFi local move + BLE repack，只有在不降低 WiFi floor 且总评分更优时才接受。

#### X.X.X 为什么 HGA 仍属于联合调度

WiFi stripe 展开只用于几何诊断与 residual-hole 评分，不构成独立调度对象。真正的优化单元始终是 task 级联合 state，因此 HGA 依然是 unified joint scheduling，而不是 WiFi-first heuristic 的回退版本。
```

- [ ] **Step 2: Run focused README tests**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py::EventBlockExpansionTest::test_readme_documents_unified_joint_ga_hga_principles -q
pytest tests/test_ble_macrocycle_hopping_sdp.py::EventBlockExpansionTest::test_readme_documents_inputs_and_optimization_variables -q
```

Expected: both PASS

- [ ] **Step 3: Read the updated README slice**

Run:

```bash
sed -n '300,520p' README.md
```

Expected:
- unified joint GA/HGA principle is explicit,
- chromosome encoding is explicit,
- WiFi floor is explicit,
- residual-hole and accept-if-better WiFi local move are explicit.

- [ ] **Step 4: Commit**

```bash
git add README.md tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "docs: clarify unified joint GA and HGA principles"
```

## Task 4: Sync validated joint scheduling code from worktree to main

**Files:**
- Create or modify in main:
  - `joint_sched/*.py`
  - `joint_sched/tests/*.py`
  - `joint_sched/output_compare_*/*`
  - `joint_sched/experiments/*`

- [ ] **Step 1: Copy the joint scheduling source tree from the worktree**

Run:

```bash
rsync -a --delete \
  /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/.worktrees/joint-wifi-ble/joint_sched/ \
  /data/home/Jie_Wan/mycode/sig-sdp-mmw-test/joint_sched/
```

Expected: main workspace `joint_sched/` matches the validated worktree implementation.

- [ ] **Step 2: Review the synced files**

Run:

```bash
git status --short joint_sched README.md tests/test_ble_macrocycle_hopping_sdp.py
```

Expected: only the intended `joint_sched/`, `README.md`, and test changes show up.

- [ ] **Step 3: Verify no accidental scratch paths were copied**

Run:

```bash
find joint_sched -maxdepth 2 -type f | sort
```

Expected: only code, tests, configs, and the intended experiment/output folders.

- [ ] **Step 4: Commit**

```bash
git add joint_sched README.md tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "feat: sync unified joint scheduling worktree"
```

## Task 5: Verify code and experiment outputs after sync

**Files:**
- Test: `joint_sched/tests/*.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

- [ ] **Step 1: Run the README regression tests**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected: PASS

- [ ] **Step 2: Run the joint scheduling test suite**

Run:

```bash
PYTHONPATH=. pytest joint_sched/tests -q
```

Expected: PASS

- [ ] **Step 3: Run a faithful HGA smoke command**

Run:

```bash
PYTHONPATH=. python joint_sched/run_joint_wifi_ble_demo.py \
  --solver hga \
  --config sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --output joint_sched/experiments/final_faithful_hga_sync
```

Expected:
- command exits successfully,
- `joint_summary.json` exists,
- `wifi_ble_schedule_overview.png` exists.

- [ ] **Step 4: Inspect the summary**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("joint_sched/experiments/final_faithful_hga_sync/joint_summary.json").read_text())
print({
    "selected_pairs": summary.get("selected_pairs"),
    "final_wifi_payload_bytes": summary.get("final_wifi_payload_bytes"),
    "accepted_wifi_local_moves": summary.get("accepted_wifi_local_moves"),
})
PY
```

Expected: prints a compact dictionary proving the synced code is runnable and produces summary fields.

- [ ] **Step 5: Commit**

If the smoke output is intended to ship:

```bash
git add joint_sched/experiments/final_faithful_hga_sync
git commit -m "test: record final faithful joint HGA sync output"
```

If not intended to ship, do not commit it.

## Task 6: Final review, push to GitHub, and workspace cleanup

**Files:**
- Modify: none unless review finds issues

- [ ] **Step 1: Review the final diff**

Run:

```bash
git diff --stat HEAD~3..HEAD
```

Expected: a coherent final diff covering README, tests, joint scheduler code, and intended outputs only.

- [ ] **Step 2: Check workspace status**

Run:

```bash
git status --short
```

Expected: clean, or only intentionally untracked scratch artifacts.

- [ ] **Step 3: Push to GitHub**

Run:

```bash
git push origin main
```

Expected: push succeeds and remote is updated.

- [ ] **Step 4: Record the synced outputs in the handoff note**

Write a final handoff note with:

```text
- synced code directories
- synced experiment output directories
- README sections updated
- tests run and their outcomes
- final commit SHA
```

- [ ] **Step 5: Commit**

Do not create an extra empty commit if everything is already committed. Only commit if review uncovered final fixes:

```bash
git add -A
git commit -m "chore: finalize joint scheduler sync and docs"
```

## Self-Review

- Spec coverage:
  - Check worktree content carefully: covered by Task 1 and Task 4.
  - Sync code and outputs to main: covered by Task 4.
  - Check README clearly documents unified joint GA/HGA principles and ideas: covered by Task 2 and Task 3.
  - Update/push to GitHub: covered by Task 6.
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to previous task” placeholders remain.
  - Commands, paths, and test names are concrete.
- Type consistency:
  - The plan consistently uses `x_k`, `x_{k,a}`, `s_k`, `l_k`, `c_{k,m}`, and `Y`.
  - “WiFi floor”, “residual-hole”, and “accept-if-better” terminology is stable across tasks.

