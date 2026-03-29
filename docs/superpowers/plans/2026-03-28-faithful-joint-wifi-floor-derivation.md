# Faithful Joint WiFi Floor Derivation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive a real WiFi baseline payload floor from the mainline scheduling instance, inject it automatically into faithful `joint_sched`, and rerun `GA/HGA` comparisons on the same protected floor.

**Architecture:** Extend the faithful-mainline CSV adapter so it computes a WiFi baseline payload floor from the legacy scheduled WiFi rows in `pair_parameters.csv`, writes that value into the joint objective policy, and exposes it in runner summaries. Then lock the behavior with focused tests and rerun the faithful `GA/HGA` experiments so the comparison is no longer based on a zero WiFi floor.

**Tech Stack:** Python 3.10, pandas, dataclasses, pytest, existing `joint_sched/` adapter/runner/GA/HGA stack

---

## File Map

- Modify: `joint_sched/joint_wifi_ble_adapter.py`
  - Derive WiFi baseline payload floor from mainline `pair_parameters.csv`.
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
  - Surface the derived floor and generation mode diagnostics in summaries.
- Modify: `joint_sched/tests/test_joint_wifi_ble_adapter.py`
  - Add adapter-level regression tests for WiFi floor derivation.
- Modify: `joint_sched/tests/test_joint_wifi_ble_runner.py`
  - Add runner-level tests proving faithful mode auto-injects a nonzero floor when WiFi is scheduled in the source CSV.
- Modify: `README.md`
  - Document how faithful `joint_sched` derives the protected WiFi floor from mainline outputs.

---

### Task 1: Derive WiFi Baseline Payload Floor In The Faithful Adapter

**Files:**
- Modify: `joint_sched/joint_wifi_ble_adapter.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_adapter.py`

- [ ] **Step 1: Write the failing adapter test**

```python
import pandas as pd

from joint_sched.joint_wifi_ble_adapter import (
    build_joint_runtime_config_from_mainline_pair_parameters_csv,
)


def test_adapter_derives_wifi_payload_floor_from_scheduled_wifi_rows(tmp_path):
    csv_path = tmp_path / "pair_parameters.csv"
    pd.DataFrame(
        [
            {
                "pair_id": 1,
                "radio": "wifi",
                "schedule_slot": 10,
                "wifi_tx_slots": 5,
                "wifi_period_slots": 16,
                "occupied_slots_in_macrocycle": "[10,11,12,13,14,26,27,28,29,30]",
                "deadline_slot": 63,
                "release_time_slot": 0,
                "channel": 0,
            },
            {
                "pair_id": 2,
                "radio": "ble",
                "schedule_slot": 20,
                "ble_ce_slots": 1,
                "ble_ci_slots": 8,
                "occupied_slots_in_macrocycle": "[20,28,36,44,52,60]",
                "deadline_slot": 63,
                "release_time_slot": 0,
                "channel": 5,
            },
        ]
    ).to_csv(csv_path, index=False)

    runtime = build_joint_runtime_config_from_mainline_pair_parameters_csv(csv_path)

    assert runtime["objective"]["wifi_payload_floor_bytes"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_adapter.py -q
```

Expected: FAIL because the adapter currently does not auto-derive `wifi_payload_floor_bytes`.

- [ ] **Step 3: Write minimal implementation**

```python
def derive_wifi_payload_floor_from_mainline_rows(rows: list[dict[str, str]]) -> int:
    total = 0
    for row in rows:
        if str(row.get("radio", "")) != "wifi":
            continue
        if int(float(row.get("schedule_slot", -1) or -1)) < 0:
            continue
        tx_slots = int(float(row.get("wifi_tx_slots", 0) or 0))
        occupied_slots = parse_mainline_slot_list(row.get("occupied_slots_in_macrocycle", "[]"))
        repetitions = max(1, len(occupied_slots) // max(1, tx_slots))
        total += repetitions * tx_slots * DEFAULT_WIFI_BYTES_PER_SLOT
    return total


objective = dict(base_config.get("objective", {}))
objective.setdefault("wifi_payload_floor_bytes", derive_wifi_payload_floor_from_mainline_rows(rows))
runtime["objective"] = objective
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_adapter.py -q
```

Expected: PASS with a nonzero derived WiFi floor.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_adapter.py joint_sched/tests/test_joint_wifi_ble_adapter.py
git commit -m "feat: derive wifi floor from faithful mainline csv"
```

---

### Task 2: Preserve Explicit Objective Overrides While Auto-Injecting The Floor

**Files:**
- Modify: `joint_sched/joint_wifi_ble_adapter.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_adapter.py`

- [ ] **Step 1: Write the failing override test**

```python
import pandas as pd

from joint_sched.joint_wifi_ble_adapter import (
    build_joint_runtime_config_from_mainline_pair_parameters_csv,
)


def test_adapter_preserves_explicit_wifi_floor_override(tmp_path):
    csv_path = tmp_path / "pair_parameters.csv"
    pd.DataFrame(
        [
            {
                "pair_id": 1,
                "radio": "wifi",
                "schedule_slot": 0,
                "wifi_tx_slots": 5,
                "wifi_period_slots": 16,
                "occupied_slots_in_macrocycle": "[0,1,2,3,4]",
                "deadline_slot": 63,
                "release_time_slot": 0,
                "channel": 0,
            }
        ]
    ).to_csv(csv_path, index=False)

    runtime = build_joint_runtime_config_from_mainline_pair_parameters_csv(
        csv_path,
        base_config={"objective": {"wifi_payload_floor_bytes": 99999}},
    )

    assert runtime["objective"]["wifi_payload_floor_bytes"] == 99999
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_adapter.py -q
```

Expected: FAIL if the derived floor overwrites explicit user intent.

- [ ] **Step 3: Write minimal implementation**

```python
objective = dict(base_config.get("objective", {}))
if "wifi_payload_floor_bytes" not in objective:
    objective["wifi_payload_floor_bytes"] = derive_wifi_payload_floor_from_mainline_rows(rows)
runtime["objective"] = objective
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_adapter.py -q
```

Expected: PASS with explicit override preserved.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_adapter.py joint_sched/tests/test_joint_wifi_ble_adapter.py
git commit -m "fix: preserve explicit wifi floor overrides"
```

---

### Task 3: Surface Derived Floor In Faithful Runner Summaries

**Files:**
- Modify: `joint_sched/run_joint_wifi_ble_demo.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing runner test**

```python
from pathlib import Path

from joint_sched.run_joint_wifi_ble_demo import run_joint_demo


def test_faithful_runner_reports_nonzero_derived_wifi_floor(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    summary = run_joint_demo(
        config_path=repo_root / "sim_script/output_ga_wifi_reschedule/pair_parameters.csv",
        solver="ga",
        output_dir=tmp_path / "faithful_ga",
    )

    assert summary["_joint_generation_mode"] == "faithful_mainline_csv"
    assert summary["wifi_payload_floor_bytes"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: FAIL because faithful mode currently reports a zero floor in summary.

- [ ] **Step 3: Write minimal implementation**

```python
summary["wifi_payload_floor_bytes"] = float(metrics["wifi_payload_floor_bytes"])
summary["_joint_generation_mode"] = config.get("_joint_generation_mode", "native_joint")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: PASS and faithful runner summary now exposes a nonzero floor.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_runner.py
git commit -m "feat: report derived wifi floor in faithful runner"
```

---

### Task 4: Lock HGA Behavior Against The Auto-Derived WiFi Floor

**Files:**
- Modify: `joint_sched/tests/test_joint_wifi_ble_runner.py`

- [ ] **Step 1: Write the failing HGA faithful-floor test**

```python
from pathlib import Path

from joint_sched.run_joint_wifi_ble_demo import run_joint_demo


def test_faithful_hga_keeps_final_wifi_payload_above_derived_floor(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    summary = run_joint_demo(
        config_path=repo_root / "sim_script/output_ga_wifi_reschedule/pair_parameters.csv",
        solver="hga",
        output_dir=tmp_path / "faithful_hga",
    )

    assert summary["wifi_payload_floor_bytes"] > 0
    assert summary["final_wifi_payload_bytes"] >= summary["wifi_payload_floor_bytes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: FAIL until faithful HGA actually consumes the auto-derived floor through the adapter/runner pipeline.

- [ ] **Step 3: Write minimal plumbing if needed**

```python
config = resolve_joint_runtime_config(config_path)
result = solve_joint_wifi_ble_hga(config)
summary["wifi_payload_floor_bytes"] = float(resolve_joint_objective_policy(config).get("wifi_payload_floor_bytes", 0.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: PASS and HGA faithful mode respects the derived WiFi floor.

- [ ] **Step 5: Commit**

```bash
git add joint_sched/tests/test_joint_wifi_ble_runner.py joint_sched/run_joint_wifi_ble_demo.py
git commit -m "test: enforce derived wifi floor in faithful hga"
```

---

### Task 5: Document Faithful WiFi-Floor Derivation In README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the documentation block**

```markdown
#### 11.x Faithful Mainline WiFi Floor Derivation

当 `joint_sched` 直接读取主线 `pair_parameters.csv` 时，实验不再假定 `wifi_payload_floor_bytes = 0`。相反，适配器会先从主线已调度 WiFi pair 中恢复一个基线 WiFi payload：

```math
P_{\mathrm{wifi}}^{\min}
=
\sum_{k \in \mathcal{K}_{\mathrm{wifi}}^{\mathrm{scheduled}}}
M_k^{\mathrm{wifi}} \cdot w_k \cdot \rho_{\mathrm{wifi}}
```

其中：
- $`M_k^{\mathrm{wifi}}`$ 为主线宏周期内 WiFi 事件重复次数
- $`w_k`$ 为每次 WiFi 事件的 `tx_slots`
- $`\rho_{\mathrm{wifi}}`$ 为代码中的 `DEFAULT_WIFI_BYTES_PER_SLOT`

这个量随后自动写入 faithful `joint_sched` 的目标配置：

```math
P_{\mathrm{wifi}}(x) \ge P_{\mathrm{wifi}}^{\min}
```

因此 faithful `GA/HGA` 的比较不再依赖手工输入的 floor，也不会因为默认零阈值而再次退化成“牺牲 WiFi 换 BLE”。
```

- [ ] **Step 2: Save the README changes**

Insert the block in the `joint_sched` section immediately after the protected WiFi payload floor subsection.

- [ ] **Step 3: Verify the README section exists**

Run:

```bash
rg -n "Faithful Mainline WiFi Floor Derivation|P_\\{\\\\mathrm\\{wifi\\}\\}\\^\\{\\\\min\\}" README.md
```

Expected: the new subsection is present once.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: explain faithful wifi floor derivation"
```

---

### Task 6: Rerun Protected Faithful GA/HGA Comparison

**Files:**
- Modify: `joint_sched/output_compare_ga_faithful_periodic_protected/`
- Modify: `joint_sched/output_compare_hga_faithful_periodic_protected/`

- [ ] **Step 1: Run the focused adapter and runner tests**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_adapter.py joint_sched/tests/test_joint_wifi_ble_runner.py -q
```

Expected: PASS for the new faithful floor derivation tests.

- [ ] **Step 2: Run faithful GA and HGA on the same mainline CSV**

Run:

```bash
python joint_sched/run_joint_wifi_ble_demo.py \
  --config sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --solver ga \
  --output joint_sched/output_compare_ga_faithful_periodic_protected

python joint_sched/run_joint_wifi_ble_demo.py \
  --config sim_script/output_ga_wifi_reschedule/pair_parameters.csv \
  --solver hga \
  --output joint_sched/output_compare_hga_faithful_periodic_protected
```

Expected:
- both commands exit `0`
- both summaries report `wifi_payload_floor_bytes > 0`
- `HGA` reports `final_wifi_payload_bytes >= wifi_payload_floor_bytes`

- [ ] **Step 3: Inspect the summaries**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

for name in [
    "joint_sched/output_compare_ga_faithful_periodic_protected/joint_summary.json",
    "joint_sched/output_compare_hga_faithful_periodic_protected/joint_summary.json",
]:
    data = json.loads(Path(name).read_text(encoding="utf-8"))
    print(name, {
        "selected_pairs": data["selected_pairs"],
        "scheduled_payload_bytes": data["scheduled_payload_bytes"],
        "wifi_payload_floor_bytes": data["wifi_payload_floor_bytes"],
        "wifi_seed_payload_bytes": data.get("wifi_seed_payload_bytes"),
        "final_wifi_payload_bytes": data.get("final_wifi_payload_bytes"),
        "fill_penalty": data["fill_penalty"],
    })
PY
```

Expected: the printed results now reflect a nonzero faithful WiFi floor and make GA/HGA comparison meaningful.

- [ ] **Step 4: Commit**

```bash
git add joint_sched/joint_wifi_ble_adapter.py joint_sched/run_joint_wifi_ble_demo.py joint_sched/tests/test_joint_wifi_ble_adapter.py joint_sched/tests/test_joint_wifi_ble_runner.py README.md joint_sched/output_compare_ga_faithful_periodic_protected joint_sched/output_compare_hga_faithful_periodic_protected
git commit -m "feat: auto-derive wifi floor for faithful joint scheduling"
```

---

## Self-Review

- Spec coverage: The plan covers all requested items: derive a real WiFi baseline floor from the mainline instance, auto-inject it into faithful `joint_sched`, and rerun protected `GA/HGA` comparisons.
- Placeholder scan: No placeholders remain; every task includes exact files, code snippets, commands, and expected outcomes.
- Type consistency: `wifi_payload_floor_bytes` is used consistently across adapter, runner, tests, README, and final comparison tasks.

