# Joint WiFi-BLE Scheduling SDP/GA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a new isolated joint WiFi-BLE scheduler that treats WiFi and BLE candidate states in one unified optimization problem, first with an SDP-based solver and then with a GA-based solver, without modifying the existing WiFi-first experiment chain.

**Architecture:** Create a new experiment folder with standalone joint-scheduling modules, configs, tests, and outputs. The joint scheduler should generate candidate states for both WiFi and BLE, construct a unified collision/interference cost model over all candidate states, and solve “one state per pair” jointly. Implement the shared model builder first, then add two independent solvers on top of it: a lifted SDP relaxation plus rounding, and a GA searcher over the same mixed candidate-state space.

**Tech Stack:** Python, NumPy, CVXPY, matplotlib, pytest

---

### Task 1: Create isolated joint-scheduling experiment layout

**Files:**
- Create: `joint_sched/__init__.py`
- Create: `joint_sched/joint_wifi_ble_model.py`
- Create: `joint_sched/joint_wifi_ble_sdp.py`
- Create: `joint_sched/joint_wifi_ble_ga.py`
- Create: `joint_sched/joint_wifi_ble_plot.py`
- Create: `joint_sched/joint_wifi_ble_demo_config.json`
- Create: `joint_sched/tests/test_joint_wifi_ble_model.py`
- Create: `joint_sched/tests/test_joint_wifi_ble_sdp.py`
- Create: `joint_sched/tests/test_joint_wifi_ble_ga.py`

**Step 1: Write the failing test**

Create a smoke import test in `joint_sched/tests/test_joint_wifi_ble_model.py` that imports the new package modules and asserts placeholder public symbols exist.

```python
def test_joint_modules_exist():
    import joint_sched.joint_wifi_ble_model as model
    assert hasattr(model, "build_joint_candidate_states")
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_model.py::test_joint_modules_exist -v
```

Expected: FAIL because the new package/files do not exist yet.

**Step 3: Write minimal implementation**

Create the new directory and files with placeholder functions/classes only. Do not copy existing WiFi-first code into them yet.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_model.py::test_joint_modules_exist -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/__init__.py joint_sched/joint_wifi_ble_model.py joint_sched/joint_wifi_ble_sdp.py joint_sched/joint_wifi_ble_ga.py joint_sched/joint_wifi_ble_plot.py joint_sched/joint_wifi_ble_demo_config.json joint_sched/tests/test_joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_sdp.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "feat: scaffold isolated joint wifi-ble scheduling package"
```

### Task 2: Define unified task and candidate-state model

**Files:**
- Modify: `joint_sched/joint_wifi_ble_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_model.py`

**Step 1: Write the failing test**

Add tests for shared dataclasses such as:
- `WiFiPairConfig`
- `BLEPairConfig`
- `JointCandidateState`
- `ExternalBlock` if needed for future constraints

Also assert `build_joint_candidate_states(...)` returns:
- one global state list
- per-pair candidate index map
- both WiFi and BLE states in the same index space

**Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k candidate_states
```

Expected: FAIL because the data model and builder are placeholders.

**Step 3: Write minimal implementation**

Implement a unified candidate-state model:
- WiFi state encodes `(pair_id, medium='wifi', offset, channel, period, width)`
- BLE state encodes `(pair_id, medium='ble', offset, pattern_id, ci, ce, num_events)`
- shared `pair_to_state_indices` mapping

Keep the builder deterministic and bounded.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k candidate_states
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_model.py
git commit -m "feat: add unified wifi-ble candidate state model"
```

### Task 3: Implement joint time-frequency occupancy expansion

**Files:**
- Modify: `joint_sched/joint_wifi_ble_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_model.py`

**Step 1: Write the failing test**

Add tests asserting that a WiFi candidate and a BLE candidate can both be expanded into comparable event/resource blocks with:
- slot ranges
- frequency ranges
- per-event labels

Also verify BLE blocks never occupy advertising channels.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k occupancy
```

Expected: FAIL because shared expansion utilities do not exist.

**Step 3: Write minimal implementation**

Implement:
- `expand_wifi_candidate_blocks(...)`
- `expand_ble_candidate_blocks(...)`
- `expand_candidate_blocks(...)`

Use the same MHz semantics already used in the main repo:
- WiFi fixed `1/6/11`
- BLE data channel two-segment mapping
- BLE advertising channels excluded from data hopping

**Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k occupancy
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_model.py
git commit -m "feat: add shared occupancy expansion for joint states"
```

### Task 4: Build unified collision/interference cost matrix

**Files:**
- Modify: `joint_sched/joint_wifi_ble_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_model.py`

**Step 1: Write the failing test**

Add tests for a new helper like `build_joint_cost_matrix(...)` asserting:
- WiFi-WiFi overlap contributes cost
- BLE-BLE overlap contributes cost
- WiFi-BLE overlap contributes cost
- non-overlapping states contribute zero

**Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k cost_matrix
```

Expected: FAIL because the joint cost builder does not exist.

**Step 3: Write minimal implementation**

Implement a unified cost matrix `Omega_joint[i, j]` over the mixed state list. Keep the first version simple:
- cost = total overlap length in time × overlapping bandwidth indicator/weight
- no fairness terms yet

**Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_model.py -q -k cost_matrix
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_model.py
git commit -m "feat: add unified wifi-ble collision cost matrix"
```

### Task 5: Add the joint SDP solver

**Files:**
- Modify: `joint_sched/joint_wifi_ble_sdp.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_sdp.py`

**Step 1: Write the failing test**

Add tests asserting the SDP solver:
- consumes the shared mixed candidate-state model
- enforces “one state per pair” across both WiFi and BLE pairs
- returns a selected mixed schedule and block list

**Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_sdp.py -q
```

Expected: FAIL because the solver is still placeholder code.

**Step 3: Write minimal implementation**

Implement:
- lifted SDP relaxation over the mixed state space
- candidate selection rounding back to one state per pair
- output structure similar to the BLE-only solver

Reuse the vectorized objective style, not the old nested-expression style.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_sdp.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_sdp.py joint_sched/tests/test_joint_wifi_ble_sdp.py
git commit -m "feat: add joint wifi-ble SDP solver"
```

### Task 6: Add the joint GA solver

**Files:**
- Modify: `joint_sched/joint_wifi_ble_ga.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`

**Step 1: Write the failing test**

Add tests asserting the GA solver:
- consumes the same mixed candidate-state space as the SDP solver
- keeps one gene per pair
- returns a selected mixed schedule and fitness history

**Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: FAIL because the GA solver is placeholder code.

**Step 3: Write minimal implementation**

Implement a mixed-medium GA:
- one chromosome gene per pair
- each gene chooses one local candidate state for either WiFi or BLE according to that pair's candidate set
- fitness uses the same `Omega_joint`
- include tournament selection, crossover, mutation, elitism

**Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_ga.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "feat: add joint wifi-ble GA solver"
```

### Task 7: Add standalone plotting and config-driven demo entrypoints

**Files:**
- Modify: `joint_sched/joint_wifi_ble_plot.py`
- Modify: `joint_sched/joint_wifi_ble_sdp.py`
- Modify: `joint_sched/joint_wifi_ble_ga.py`
- Modify: `joint_sched/joint_wifi_ble_demo_config.json`
- Test: `joint_sched/tests/test_joint_wifi_ble_sdp.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`

**Step 1: Write the failing test**

Add run-level tests asserting both standalone solvers can:
- load the demo JSON
- solve the joint problem
- emit event-level CSV/PNG outputs
- show BLE advertising channels as idle, not occupiable

**Step 2: Run test to verify it fails**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_sdp.py joint_sched/tests/test_joint_wifi_ble_ga.py -q -k demo_config
```

Expected: FAIL because config IO and plotting entrypoints are not wired yet.

**Step 3: Write minimal implementation**

Implement:
- JSON loader for mixed WiFi/BLE demo inputs
- standalone `main()` for SDP and GA scripts
- plot renderer aligned with the repo’s current plotting semantics

**Step 4: Run test to verify it passes**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_sdp.py joint_sched/tests/test_joint_wifi_ble_ga.py -q -k demo_config
```

Expected: PASS.

**Step 5: Commit**

```bash
git add joint_sched/joint_wifi_ble_plot.py joint_sched/joint_wifi_ble_sdp.py joint_sched/joint_wifi_ble_ga.py joint_sched/joint_wifi_ble_demo_config.json joint_sched/tests/test_joint_wifi_ble_sdp.py joint_sched/tests/test_joint_wifi_ble_ga.py
git commit -m "feat: add standalone joint wifi-ble demo runners and plotting"
```

### Task 8: Document the new isolated joint experiment

**Files:**
- Modify: `README.md`

**Step 1: Write the failing doc expectation**

List the required README additions:
- new `joint_sched/` experiment folder
- difference from the existing WiFi-first chain
- SDP and GA entrypoints
- limitation that this is an isolated experiment path for now

**Step 2: Verify docs are missing**

Open `README.md` and confirm the new joint path is not documented.

**Step 3: Write minimal documentation**

Add a dedicated README section describing:
- why joint scheduling exists
- how it differs from iterative WiFi-first coordination
- the new entrypoints and example commands

**Step 4: Verify docs render and paths are correct**

Check all paths and command examples.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: describe isolated joint wifi-ble scheduling experiment"
```

### Task 9: Full verification on both solvers

**Files:**
- Modify: none unless fixes are needed
- Test: `joint_sched/tests/test_joint_wifi_ble_model.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_sdp.py`
- Test: `joint_sched/tests/test_joint_wifi_ble_ga.py`

**Step 1: Run the full joint test suite**

Run:

```bash
pytest joint_sched/tests/test_joint_wifi_ble_model.py joint_sched/tests/test_joint_wifi_ble_sdp.py joint_sched/tests/test_joint_wifi_ble_ga.py -q
```

Expected: PASS.

**Step 2: Run the standalone joint SDP demo**

Run:

```bash
python joint_sched/joint_wifi_ble_sdp.py --config joint_sched/joint_wifi_ble_demo_config.json
```

Expected: PASS and generate mixed WiFi/BLE scheduling outputs.

**Step 3: Run the standalone joint GA demo**

Run:

```bash
python joint_sched/joint_wifi_ble_ga.py --config joint_sched/joint_wifi_ble_demo_config.json
```

Expected: PASS and generate mixed WiFi/BLE scheduling outputs.

**Step 4: Compare SDP vs GA outputs on the same demo**

Record:
- total scheduled pair count
- WiFi scheduled count
- BLE scheduled count
- runtime

**Step 5: Commit**

```bash
git add -A
git commit -m "test: verify isolated joint wifi-ble SDP and GA solvers"
```
