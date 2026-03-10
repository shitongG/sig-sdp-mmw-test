# Macrocycle Concurrent Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the macrocycle scheduler's global `reserved` exclusion with a conflict-aware scheduler that allows concurrent occupancy when pairs do not share channel/association/interference conflicts.

**Architecture:** Reuse the existing conflict information already produced by `env.generate_S_Q_hmax()` and/or the derived conflict matrices in `env.py`. The macrocycle scheduler should maintain per-slot assigned pair sets and reject overlap only when a candidate pair conflicts with already assigned pairs in the same occupied slot. Keep the current partial-assignment behavior, but make it much less conservative by allowing safe concurrency.

**Tech Stack:** Python 3.10, NumPy, SciPy sparse matrices, existing `env`, `binary_search_relaxation`, `sdp_solver`, script-level CSV output.

---

### Task 1: Define a reusable pair-conflict predicate for macrocycle scheduling

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_macrocycle_conflict_predicate.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_src.env.env import env


def test_pair_conflict_predicate_distinguishes_safe_concurrency_from_true_conflict():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=3, radio_prob=(1.0, 0.0))
    pair_a = 0
    pair_b = 1
    conflict = e.build_pair_conflict_matrix()
    assert conflict.shape == (e.n_pair, e.n_pair)
    assert np.all(np.diag(conflict) == 0)
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because `build_pair_conflict_matrix()` does not exist yet.

**Step 3: Write minimal implementation**

```python
def build_pair_conflict_matrix(self):
    _, q_conflict, _ = self.generate_S_Q_hmax()
    conflict = q_conflict.copy().astype(bool)
    conflict.setdiag(False)
    return conflict.toarray().astype(bool)
```

If needed, combine `Q_conflict` with any additional hard mutual-exclusion constraints that must still serialize in macrocycle placement.

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_macrocycle_conflict_predicate.py
git commit -m "feat: expose pair conflict matrix for macrocycle scheduling"
```

### Task 2: Replace global `reserved` with per-slot conflict-aware assignment

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:80-126`
- Test: `sim_script/tests/test_macrocycle_concurrent_assignment.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots


class _StubEnv:
    n_pair = 2
    pair_priority = np.array([2.0, 1.0])
    RADIO_BLE = 1
    pair_radio_type = np.array([0, 0], dtype=int)
    pair_ble_ce_feasible = np.array([True, True], dtype=bool)

    def compute_macrocycle_slots(self):
        return 4

    def get_pair_period_slots(self):
        return np.array([4, 4], dtype=int)

    def get_pair_width_slots(self):
        return np.array([2, 2], dtype=int)

    def expand_pair_occupancy(self, pair_id, start_slot, macrocycle_slots):
        occ = np.zeros(macrocycle_slots, dtype=bool)
        occ[0:2] = True
        return occ

    def build_pair_conflict_matrix(self):
        return np.array([[False, False], [False, False]], dtype=bool)


def test_macrocycle_scheduler_allows_overlap_for_non_conflicting_pairs():
    env = _StubEnv()
    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(env, np.array([0, 0]), allow_partial=False)
    assert unscheduled == []
    assert occ[0, 0] and occ[1, 0]
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL because current implementation blocks any shared occupied slot through `reserved`.

**Step 3: Write minimal implementation**

```python
conflict = e.build_pair_conflict_matrix()
slot_asn = [[] for _ in range(macrocycle_slots)]
...
for start_slot in ...:
    occ = e.expand_pair_occupancy(...)
    occ_slots = np.where(occ)[0]
    violates = False
    for slot in occ_slots:
        for other in slot_asn[slot]:
            if conflict[pair_id, other]:
                violates = True
                break
        if violates:
            break
    if violates:
        continue
    assigned_starts[pair_id] = ...
    occupancies[pair_id] = occ
    for slot in occ_slots:
        slot_asn[slot].append(pair_id)
```

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_macrocycle_concurrent_assignment.py
git commit -m "feat: allow non-conflicting macrocycle overlap"
```

### Task 3: Preserve blocking for true conflicts

**Files:**
- Modify: `sim_script/tests/test_macrocycle_concurrent_assignment.py`
- Modify: `sim_script/pd_mmw_template_ap_stats.py` if needed

**Step 1: Write the failing test**

```python
def test_macrocycle_scheduler_still_blocks_overlap_for_conflicting_pairs():
    env = _StubEnv()
    env.build_pair_conflict_matrix = lambda: np.array([[False, True], [True, False]], dtype=bool)
    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(env, np.array([0, 0]), allow_partial=True)
    assert starts[0] >= 0
    assert starts[1] == -1 or not np.any(np.logical_and(occ[0], occ[1]))
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL if the new scheduler ignores actual conflicts.

**Step 3: Write minimal implementation**

Tighten the slot-by-slot conflict check so overlap is rejected only when `conflict[pair_id, other]` is true for any already assigned pair sharing an occupied slot.

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/tests/test_macrocycle_concurrent_assignment.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "test: preserve true conflict blocking in macrocycle scheduler"
```

### Task 4: Use real conflict matrix from `env.generate_S_Q_hmax()` in script flow

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_macrocycle_real_conflict_integration.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sim_src.env.env import env
from sim_script.pd_mmw_template_ap_stats import assign_macrocycle_start_slots


def test_real_env_macrocycle_scheduler_can_keep_multiple_non_conflicting_pairs():
    e = env(cell_size=1, pair_density_per_m2=0.05, seed=5, radio_prob=(0.0, 1.0), ble_ci_exp_min=3, ble_ci_exp_max=4)
    preferred = np.zeros(e.n_pair, dtype=int)
    starts, macro, occ, unscheduled = assign_macrocycle_start_slots(e, preferred, allow_partial=True)
    assert len(unscheduled) < e.n_pair
```

**Step 2: Run test to verify it fails**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL or show no improvement before the real conflict matrix is wired in.

**Step 3: Write minimal implementation**

- Call `e.build_pair_conflict_matrix()` once per scheduler invocation.
- Use that matrix for overlap checks.
- Do not rebuild `generate_S_Q_hmax()` inside the inner candidate loop.

**Step 4: Run test to verify it passes**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_macrocycle_real_conflict_integration.py
git commit -m "feat: use real conflict graph for macrocycle concurrency"
```

### Task 5: Keep partial-schedule and CSV behavior intact under concurrency

**Files:**
- Modify: `sim_script/tests/test_macrocycle_partial_assignment.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/pd_mmw_template_ap_stats.py` if needed

**Step 1: Write the failing regression tests**

```python
def test_partial_assignment_still_returns_unscheduled_pairs_under_conflict_pressure():
    ...


def test_schedule_rows_can_include_multiple_pairs_in_same_schedule_slot_when_non_conflicting():
    pair_rows = [
        {"pair_id": 0, "radio": "ble", "schedule_slot": 3, "occupied_slots_in_macrocycle": [3, 4]},
        {"pair_id": 1, "radio": "ble", "schedule_slot": 3, "occupied_slots_in_macrocycle": [3, 4]},
    ]
    rows = build_schedule_rows(pair_rows)
    assert rows[0]["pair_count"] == 2
```

**Step 2: Run tests to verify failures**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: FAIL if helper logic still assumes one pair per occupied slot.

**Step 3: Write minimal implementation**

- Keep `build_schedule_rows()` unchanged if it already groups multiple pair rows per slot correctly.
- Only adjust tests/helper glue where the old assumption was serialized occupancy.

**Step 4: Run tests to verify pass**

Run: `source /data/home/public/anaconda3/etc/profile.d/conda.sh && conda activate sig-sdp && python - <<'PY' ... PY`
Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/tests/test_macrocycle_partial_assignment.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "test: cover concurrent macrocycle occupancy outputs"
```

### Task 6: Focused verification in `sig-sdp`

**Files:**
- Modify: none unless fixes are needed
- Test: all touched tests and one script smoke run

**Step 1: Run focused test suite**

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python - <<'PY'
from sim_script.tests.test_macrocycle_conflict_predicate import test_pair_conflict_predicate_distinguishes_safe_concurrency_from_true_conflict
from sim_script.tests.test_macrocycle_concurrent_assignment import (
    test_macrocycle_scheduler_allows_overlap_for_non_conflicting_pairs,
    test_macrocycle_scheduler_still_blocks_overlap_for_conflicting_pairs,
)
from sim_script.tests.test_macrocycle_partial_assignment import test_assign_macrocycle_start_slots_can_return_partial_assignment

test_pair_conflict_predicate_distinguishes_safe_concurrency_from_true_conflict()
test_macrocycle_scheduler_allows_overlap_for_non_conflicting_pairs()
test_macrocycle_scheduler_still_blocks_overlap_for_conflicting_pairs()
test_assign_macrocycle_start_slots_can_return_partial_assignment()
print('macrocycle_concurrency: PASS')
PY
```

Expected: `macrocycle_concurrency: PASS`

**Step 2: Run one real script smoke**

```bash
source /data/home/public/anaconda3/etc/profile.d/conda.sh
conda activate sig-sdp
python sim_script/pd_mmw_template_ap_stats.py --cell-size 1 --pair-density 0.05 --mmw-nit 5 --seed 7
```

Expected:
- exit code `0`
- `partial_schedule` false or improved compared with the old global-reserved behavior for the same seed
- schedule table may contain multiple pair ids on the same `schedule_slot`

**Step 3: Commit verification-only fixes if needed**

```bash
git add <touched-files>
git commit -m "fix: stabilize conflict-aware macrocycle verification"
```
