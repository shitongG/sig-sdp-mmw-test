# README Inputs vs Optimization Variables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new paper-style README section that clearly separates current model input parameters from optimization variables for the main scheduler, BLE-only scheduler, and joint scheduler.

**Architecture:** Treat this as a documentation-only feature with verification against live code. First lock the source of truth by adding a focused README test that checks the required terminology and structure. Then update `README.md` with a notation-oriented section that maps JSON/CLI configuration inputs to mathematical symbols and separately lists decision variables chosen by the schedulers.

**Tech Stack:** Markdown, pytest, existing Python test suite

---

## File Structure

- Modify: `README.md`
  - Add a new standalone subsection near the current modeling/method sections.
  - Separate “输入参数” from “优化变量”.
  - Cover three scopes explicitly: main mixed scheduler, BLE-only standalone scheduler, and joint scheduler.
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
  - Add a focused README regression test so the notation section is not accidentally deleted or renamed later.

## Task 1: Add a README regression test for the new notation section

**Files:**
- Modify: `tests/test_ble_macrocycle_hopping_sdp.py`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

- [ ] **Step 1: Write the failing test**

Add this test near the existing README/documentation tests:

```python
def test_readme_documents_inputs_and_optimization_variables() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "输入参数与优化变量" in readme
    assert "输入参数（Inputs）" in readme
    assert "优化变量（Decision Variables）" in readme
    assert "主调度脚本" in readme
    assert "BLE-only 宏周期跳频求解器" in readme
    assert "联合调度模型" in readme
    assert "x_k" in readme
    assert "s_k" in readme
    assert "c_{k,m}" in readme
    assert "r_k" in readme
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py::test_readme_documents_inputs_and_optimization_variables -q
```

Expected: FAIL because the new README section does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Keep the test imports consistent. If `Path` is not already imported in the test file, add:

```python
from pathlib import Path
```

Insert the test exactly as written above.

- [ ] **Step 4: Run test to verify it still fails for the right reason**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py::test_readme_documents_inputs_and_optimization_variables -q
```

Expected: FAIL on README content assertions, not on syntax/import errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "test: lock README inputs vs variables section"
```

## Task 2: Add the paper-style README section

**Files:**
- Modify: `README.md`
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

- [ ] **Step 1: Write the README section in paper style**

Add a new section with this structure and content pattern:

```md
### X.X 输入参数与优化变量

为避免将“实验输入”与“调度器决策变量”混淆，本文将当前模型中的量分为输入参数（inputs）与优化变量（decision variables）两类。

#### X.X.1 输入参数（Inputs）

主调度脚本 `sim_script/pd_mmw_template_ap_stats.py` 的输入参数记为

```math
\Theta_{\mathrm{main}} =
\{
\texttt{cell\_size},\,
\texttt{pair\_density},\,
\texttt{seed},\,
\texttt{pair\_generation\_mode},\,
\texttt{pair\_parameters},\,
\texttt{ble\_schedule\_backend},\,
\texttt{wifi\_first\_ble\_scheduling}
\}.
```

其中：

- `cell_size`、`pair_density`、`seed` 决定随机实例规模与分布；
- `pair_generation_mode` 与 `pair_parameters` 决定 pair 是随机生成还是手工给定；
- 若采用手工模式，则每个 pair 的已知输入还包括时间窗、业务类型以及周期参数。

对第 `k` 个业务对，已知输入可写为

```math
\theta_k =
\left(
r_k,\,
D_k,\,
\rho_k,\,
\pi_k,\,
\chi_k
\right),
```

其中 `r_k` 与 `D_k` 分别表示 release time 与 deadline，`\rho_k \in \{\mathrm{WiFi}, \mathrm{BLE}\}` 表示业务类型，`\pi_k` 表示该业务在对应介质下的周期参数集合，`\chi_k` 表示可选信道/初始偏好等先验信息。

BLE-only 宏周期跳频求解器 `ble_macrocycle_hopping_sdp.py` 的输入参数记为

```math
\Theta_{\mathrm{ble}} =
\{
\texttt{num\_channels},\,
\texttt{pair\_configs},\,
\texttt{pattern\_dict},\,
\texttt{pair\_weight},\,
\texttt{hard\_collision\_threshold}
\}.
```

联合调度模型的实例输入则由 WiFi/BLE 任务集合与其时间窗、周期流参数、候选信道集合共同给出。

#### X.X.2 优化变量（Decision Variables）

主调度与联合调度中，真正由优化器决定的是“是否调度、何时开始、选择哪个候选状态、选择哪种介质/信道轨迹”。形式化地，对任务 `k` 定义：

```math
x_k \in \{0,1\},
\qquad
s_k \in \mathcal{S}_k,
\qquad
r_k \in \{\mathrm{WiFi}, \mathrm{BLE}\}.
```

其中 `x_k=1` 表示任务被成功调度，`s_k` 表示起始 offset，`r_k` 表示最终选中的无线介质。

若任务 `k` 选择 BLE，则其跳频相关优化变量可写为

```math
\ell_k \in \mathcal{L}_k,
\qquad
c_{k,m},\; m=0,\dots,M_k-1,
```

其中 `\ell_k` 是 hopping pattern 编号，`c_{k,m}` 是第 `m` 个 BLE connect event 的数据信道。

若任务 `k` 选择 WiFi，则优化器决定的是 WiFi 周期流 state，即其起始 offset、周期重复位置以及所选信道。

在 lifted SDP 中，上述离散选择被编码为矩阵变量

```math
Y \succeq 0,
```

而在 GA/HGA 中，上述离散选择被编码为染色体中的 state index。

#### X.X.3 为什么必须区分这两类量

输入参数描述“任务需求与场景约束”，优化变量描述“调度器实际搜索和决定的量”。例如，deadline、payload、CI/CE 候选属于输入；offset、pattern、每个 event 的信道序列、是否接纳某个任务则属于优化器输出。
```

Place this section near the existing modeling/algorithm description so readers encounter it before the detailed SDP/GA formulas.

- [ ] **Step 2: Run the focused test**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py::test_readme_documents_inputs_and_optimization_variables -q
```

Expected: PASS

- [ ] **Step 3: Run broader regression for README-sensitive tests**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected: PASS

- [ ] **Step 4: Review README placement and wording**

Open the relevant README slice and verify the new section reads cleanly in context:

```bash
sed -n '220,420p' README.md
```

Expected:
- the new section appears as a standalone subsection,
- symbols are consistent with existing notation,
- “输入参数” and “优化变量” are clearly separated.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "docs: add README section for inputs and optimization variables"
```

## Task 3: Final verification

**Files:**
- Modify: none
- Test: `tests/test_ble_macrocycle_hopping_sdp.py`

- [ ] **Step 1: Run final verification**

Run:

```bash
pytest tests/test_ble_macrocycle_hopping_sdp.py -q
```

Expected: PASS

- [ ] **Step 2: Optional syntax-free doc sanity check**

Run:

```bash
python -m py_compile ble_macrocycle_hopping_sdp.py sim_script/pd_mmw_template_ap_stats.py
```

Expected: PASS

- [ ] **Step 3: Commit verification-only checkpoint if needed**

If the previous commit already contains the final doc/test changes, do not create an extra empty commit. Otherwise:

```bash
git add README.md tests/test_ble_macrocycle_hopping_sdp.py
git commit -m "test: verify README notation documentation"
```

## Self-Review

- Spec coverage:
  - Add a README section: covered in Task 2.
  - Separate inputs from optimization variables: covered in Task 2 with two explicit subsections.
  - Use paper-style notation: covered in Task 2 via math blocks and symbol definitions.
  - Keep the section from regressing: covered in Task 1.
- Placeholder scan:
  - No `TODO`, `TBD`, or “implement later” placeholders remain.
  - Commands and test names are concrete.
- Type consistency:
  - Symbols `x_k`, `s_k`, `r_k`, `\ell_k`, `c_{k,m}`, and `Y` are defined once and reused consistently.

