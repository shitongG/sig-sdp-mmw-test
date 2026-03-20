# Packet-Size-Driven Radio Timing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace direct manual input of `wifi_tx_slots`, `ble_ce_slots`, and `ble_ci_slots` with a packet-size-driven model so users mainly provide packet size and radio choice, while keeping the old slot-based interface backward compatible.

**Architecture:** Add a timing-derivation layer between JSON/config parsing and env mutation. The new layer will accept per-pair packet-size fields plus radio selection, derive WiFi TX duration and BLE CE/CI timing, then populate the existing `pair_wifi_tx_slots`, `pair_ble_ce_slots`, and `pair_ble_ci_slots` arrays so the downstream scheduler and plotting logic remain unchanged. Random generation will keep working when no JSON is supplied, but instead of sampling direct slot counts it will sample packet sizes and derive slot counts from them.

**Tech Stack:** Python, NumPy, existing `env` timing helpers, `pytest`, JSON config files, existing CSV/plot pipeline.

---

### Task 1: Freeze the desired timing formulas in tests

**Files:**
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Modify: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write the failing unit tests for timing derivation helpers**

Add tests that document the new model explicitly:

```python
def test_derive_wifi_tx_slots_from_packet_bytes_uses_airtime_and_ceiling():
    slots = derive_wifi_tx_slots_from_packet_bytes(
        packet_bytes=1500,
        slot_time=1.25e-3,
        phy_rate_bps=20e6,
    )
    assert slots >= 1


def test_derive_ble_ce_slots_from_packet_bytes_uses_247b_pdu_chunks():
    assert derive_ble_ce_slots_from_packet_bytes(0, slot_time=1.25e-3) == 1
    assert derive_ble_ce_slots_from_packet_bytes(247, slot_time=1.25e-3) == 1
    assert derive_ble_ce_slots_from_packet_bytes(248, slot_time=1.25e-3) == 2


def test_derive_ble_ci_slots_is_discrete_and_not_smaller_than_ce():
    ci_slots = derive_ble_ci_slots_from_packet_bytes(
        packet_bytes=247,
        slot_time=1.25e-3,
        ble_ci_quanta_candidates=np.array([8, 16, 32, 64], dtype=int),
        ce_slots=1,
    )
    assert ci_slots in {8, 16, 32, 64}
    assert ci_slots >= 1
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "derive_wifi_tx_slots or derive_ble_ce_slots or derive_ble_ci_slots"
```

Expected: FAIL with import or name errors because the helpers do not exist yet.

**Step 3: Write the minimal implementation signatures**

In `sim_script/pd_mmw_template_ap_stats.py`, add placeholder helper functions with the target names and arguments so the tests can import them:

```python
def derive_wifi_tx_slots_from_packet_bytes(packet_bytes, slot_time, phy_rate_bps):
    raise NotImplementedError


def derive_ble_ce_slots_from_packet_bytes(packet_bytes, slot_time, ble_payload_limit_bytes=247):
    raise NotImplementedError


def derive_ble_ci_slots_from_packet_bytes(packet_bytes, slot_time, ble_ci_quanta_candidates, ce_slots):
    raise NotImplementedError
```

**Step 4: Run test to verify it still fails for behavior, not import**

Run the same `pytest` command.

Expected: FAIL with `NotImplementedError`.

**Step 5: Commit**

```bash
git add sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/pd_mmw_template_ap_stats.py
git commit -m "test: pin packet-size timing formulas"
```

### Task 2: Implement the packet-size-to-slot conversion helpers

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write the minimal implementation**

Implement the helpers with these rules:

```python
def derive_wifi_tx_slots_from_packet_bytes(packet_bytes, slot_time, phy_rate_bps):
    bits = max(0, int(packet_bytes)) * 8
    airtime_s = bits / float(phy_rate_bps)
    return max(1, int(math.ceil(airtime_s / float(slot_time))))


def derive_ble_ce_slots_from_packet_bytes(packet_bytes, slot_time, ble_payload_limit_bytes=247):
    payload_bytes = max(0, int(packet_bytes))
    pdu_count = max(1, int(math.ceil(payload_bytes / float(ble_payload_limit_bytes))))
    return pdu_count


def derive_ble_ci_slots_from_packet_bytes(packet_bytes, slot_time, ble_ci_quanta_candidates, ce_slots):
    candidates = np.asarray(ble_ci_quanta_candidates, dtype=int)
    feasible = candidates[candidates >= int(ce_slots)]
    if feasible.size == 0:
        return int(candidates.max())
    return int(feasible.min())
```

Implementation notes:
- WiFi uses bytes to airtime via `phy_rate_bps`, then rounds up to slots.
- BLE assumes one request + ACK exchange per PDU chunk consumes one slot of `1.25 ms`.
- BLE CI remains on the existing discrete candidate set and becomes the smallest feasible CI that can contain the CE.

**Step 2: Run unit tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "derive_wifi_tx_slots or derive_ble_ce_slots or derive_ble_ci_slots"
```

Expected: PASS.

**Step 3: Add a regression test for edge values**

Add one more test for:
- WiFi `512B`
- WiFi `1500B`
- BLE `0B`
- BLE `247B`
- BLE `248B`

Expected:
- WiFi slots monotonic with bytes
- BLE `0/247 -> 1 slot`
- BLE `248 -> 2 slots`

**Step 4: Run tests again**

Run the same `pytest` command.

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: derive wifi and ble timing from packet size"
```

### Task 3: Extend manual JSON schema to accept packet size instead of direct slots

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:72-160`
- Modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write failing schema-validation tests**

Add tests for the new schema rules:

```python
def test_manual_wifi_pair_accepts_packet_bytes_without_wifi_tx_slots():
    config = merge_config_with_defaults({
        "pair_generation_mode": "manual",
        "pair_parameters": [{
            "pair_id": 0,
            "office_id": 0,
            "radio": "wifi",
            "channel": 0,
            "priority": 1.0,
            "release_time_slot": 0,
            "deadline_slot": 31,
            "start_time_slot": 0,
            "wifi_anchor_slot": 0,
            "wifi_period_slots": 32,
            "packet_bytes": 1200,
        }],
    })
    assert config["pair_parameters"][0]["packet_bytes"] == 1200
```

Also add failure tests:
- WiFi `packet_bytes < 512` rejects
- WiFi `packet_bytes > 1500` rejects
- BLE `packet_bytes < 0` rejects
- BLE `packet_bytes > 247` rejects in the strict single-PDU mode

**Step 2: Run tests to verify failure**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "packet_bytes"
```

Expected: FAIL because validation still requires direct slot fields.

**Step 3: Implement schema evolution**

Change `_validate_pair_parameters(...)` so that:
- common field `packet_bytes` is allowed and preferred
- WiFi manual rows can provide `packet_bytes` instead of `wifi_tx_slots`
- BLE manual rows can provide `packet_bytes` instead of `ble_ce_slots` and `ble_ci_slots`
- old slot-based fields remain valid for backward compatibility

Recommended validation rules:

```python
required_wifi = {"wifi_anchor_slot", "wifi_period_slots"}
required_ble = {"ble_anchor_slot"}

if radio == "wifi":
    if "packet_bytes" not in row and "wifi_tx_slots" not in row:
        raise ValueError("wifi row must provide packet_bytes or wifi_tx_slots.")
if radio == "ble":
    if ble_timing_mode == "manual" and "packet_bytes" not in row and (
        "ble_ci_slots" not in row or "ble_ce_slots" not in row
    ):
        raise ValueError("ble row must provide packet_bytes or both ble_ci_slots and ble_ce_slots.")
```

**Step 4: Update example JSON files**

Document the new field using comment keys:

```json
"_comment_packet_bytes": "WiFi: 512~1500 B, BLE: 0~247 B; if present, timing is derived from packet size"
```

Keep at least one old-style row or config comment to show backward compatibility.

**Step 5: Run tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "packet_bytes or manual_pair"
```

Expected: PASS.

**Step 6: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: accept packet-size driven manual pair config"
```

### Task 4: Apply packet-size-driven timing during manual env construction

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py:300-390`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write failing integration tests around `apply_manual_pair_parameters(...)`**

Add tests that assert:

```python
def test_apply_manual_pair_parameters_derives_wifi_tx_slots_from_packet_bytes():
    row = {
        "pair_id": 0,
        "office_id": 0,
        "radio": "wifi",
        "channel": 0,
        "priority": 1.0,
        "release_time_slot": 0,
        "deadline_slot": 31,
        "start_time_slot": 0,
        "wifi_anchor_slot": 0,
        "wifi_period_slots": 32,
        "packet_bytes": 1500,
    }
    ...
    assert e.pair_wifi_tx_slots[0] >= 1
```

```python
def test_apply_manual_pair_parameters_derives_ble_ci_ce_from_packet_bytes():
    row = {
        "pair_id": 0,
        "office_id": 0,
        "radio": "ble",
        "channel": 8,
        "priority": 1.0,
        "release_time_slot": 0,
        "deadline_slot": 31,
        "start_time_slot": 0,
        "ble_anchor_slot": 0,
        "packet_bytes": 247,
    }
    ...
    assert e.pair_ble_ce_slots[0] == 1
    assert e.pair_ble_ci_slots[0] in {8, 16, 32, 64}
```

**Step 2: Run tests to verify failure**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "derives_wifi_tx_slots_from_packet_bytes or derives_ble_ci_ce_from_packet_bytes"
```

Expected: FAIL because `apply_manual_pair_parameters(...)` still reads direct slot fields.

**Step 3: Implement the minimal code path**

In `apply_manual_pair_parameters(...)`:
- If WiFi row includes `packet_bytes`, derive `pair_wifi_tx_slots`
- If BLE row includes `packet_bytes`, derive `pair_ble_ce_slots` and `pair_ble_ci_slots`
- Keep existing `ble_timing_mode == "auto"` path untouched
- Only fall back to explicit slot fields if `packet_bytes` is absent

Suggested shape:

```python
if radio == "wifi":
    if "packet_bytes" in row:
        e.pair_wifi_tx_slots[pair_id] = derive_wifi_tx_slots_from_packet_bytes(
            packet_bytes=row["packet_bytes"],
            slot_time=e.slot_time,
            phy_rate_bps=e.wifi_channel_bandwidth_hz,
        )
    else:
        e.pair_wifi_tx_slots[pair_id] = int(row["wifi_tx_slots"])
elif ble_timing_mode == "manual":
    if "packet_bytes" in row:
        ce_slots = derive_ble_ce_slots_from_packet_bytes(...)
        ci_slots = derive_ble_ci_slots_from_packet_bytes(...)
        e.pair_ble_ce_slots[pair_id] = ce_slots
        e.pair_ble_ci_slots[pair_id] = ci_slots
    else:
        ...
```

Also update `pair_packet_bits` so that, when `packet_bytes` exists, `pair_packet_bits[pair_id] = packet_bytes * 8`.

**Step 4: Run the tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "apply_manual_pair_parameters"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: derive manual wifi and ble timing from packet bytes"
```

### Task 5: Switch random generation from direct slots to packet-size-driven timing

**Files:**
- Modify: `sim_src/env/env.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`

**Step 1: Write failing tests against random env generation**

Add tests that pin the new random behavior:

```python
def test_random_env_wifi_packet_bytes_are_within_512_to_1500():
    e = env(cell_size=1, pair_density_per_m2=0.1, seed=1)
    wifi_ids = np.where(e.pair_radio_type == e.RADIO_WIFI)[0]
    assert np.all(e.pair_packet_bytes[wifi_ids] >= 512)
    assert np.all(e.pair_packet_bytes[wifi_ids] <= 1500)
```

```python
def test_random_env_ble_packet_bytes_are_within_0_to_247():
    e = env(cell_size=1, pair_density_per_m2=0.1, seed=1)
    ble_ids = np.where(e.pair_radio_type == e.RADIO_BLE)[0]
    assert np.all(e.pair_packet_bytes[ble_ids] >= 0)
    assert np.all(e.pair_packet_bytes[ble_ids] <= 247)
```

```python
def test_random_env_derived_slots_match_packet_size():
    e = env(cell_size=1, pair_density_per_m2=0.1, seed=1)
    ...
```

Expected assertions:
- WiFi `pair_wifi_tx_slots` equals helper-derived result from sampled packet bytes
- BLE `pair_ble_ce_slots` and `pair_ble_ci_slots` equal helper-derived results from sampled packet bytes

**Step 2: Run test to verify it fails**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "random_env_wifi_packet_bytes or random_env_ble_packet_bytes or derived_slots_match_packet_size"
```

Expected: FAIL because `env` does not yet store `pair_packet_bytes` as the sampled source of truth.

**Step 3: Implement random packet-size sampling in `env`**

In `sim_src/env/env.py`:
- add radio-specific packet-byte sampling ranges:
  - WiFi: `[512, 1500]`
  - BLE: `[0, 247]`
- sample packet bytes before timing configuration
- derive:
  - WiFi `pair_wifi_tx_slots`
  - BLE `pair_ble_ce_slots`
  - BLE `pair_ble_ci_slots`
- keep existing release/deadline/anchor logic

Recommended helper split:

```python
def _sample_pair_packet_bytes(self):
    ...

def _derive_wifi_tx_slots_for_pair(self, packet_bytes):
    ...

def _derive_ble_timing_for_pair(self, packet_bytes, start_time_slot):
    ...
```

Implementation note:
- Keep `sample_ble_pair_timing(...)` but refactor it so it can accept an optional `packet_bytes` and derive `ce_required_s` from the PDU rule instead of always using the old payload duration path.

**Step 4: Run tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "random_env_wifi_packet_bytes or random_env_ble_packet_bytes or derived_slots_match_packet_size or auto_ble_timing"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add sim_src/env/env.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py
git commit -m "feat: drive random wifi and ble timing from packet size"
```

### Task 6: Update CSV/output schema to expose packet size as the primary user-facing parameter

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_logic.py`
- Test: `sim_script/tests/test_pd_mmw_template_ap_stats_run.py`

**Step 1: Write failing tests for output rows**

Add tests asserting that `compute_pair_parameter_rows(...)` includes:
- `packet_bytes`
- `packet_bits`
- derived `wifi_tx_slots` or `ble_ci_slots` / `ble_ce_slots`

Example:

```python
def test_compute_pair_parameter_rows_includes_packet_size_and_derived_slots():
    rows = compute_pair_parameter_rows(...)
    assert rows[0]["packet_bytes"] == 247
    assert rows[0]["packet_bits"] == 1976
```

**Step 2: Run tests to verify failure**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q -k "packet_size_and_derived_slots"
```

Expected: FAIL because rows do not yet surface packet bytes explicitly.

**Step 3: Implement row/schema update**

Ensure output rows include:
- `packet_bytes`
- `packet_bits`
- `timing_source` with values like `manual_slots`, `manual_packet_bytes`, `auto_packet_bytes`

This makes post-processing and debugging easier.

**Step 4: Add a smoke test for JSON-driven execution**

In `test_pd_mmw_template_ap_stats_run.py`, run the script with a config containing packet-byte-driven rows and assert the emitted `pair_parameters.csv` has the new columns.

**Step 5: Run tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q -k "packet_bytes or timing_source"
```

Expected: PASS.

**Step 6: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py
git commit -m "feat: expose packet-size-driven timing in outputs"
```

### Task 7: Refresh sample configs and README for the new user-facing interface

**Files:**
- Modify: `sim_script/pd_mmw_template_ap_stats_config.json`
- Modify: `sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json`
- Modify: `README.md`

**Step 1: Update JSON examples**

Make the examples show the new preferred interface:

```json
{
  "pair_id": 0,
  "radio": "wifi",
  "packet_bytes": 1200,
  "wifi_anchor_slot": 4,
  "wifi_period_slots": 32
}
```

```json
{
  "pair_id": 1,
  "radio": "ble",
  "packet_bytes": 247,
  "ble_anchor_slot": 8
}
```

Also retain comments saying the old direct slot fields are still supported for debugging/backward compatibility.

**Step 2: Update README**

Add a short section:
- “现在用户主要提供什么”
- WiFi `packet_bytes` 范围 `512~1500B`
- BLE `packet_bytes` 范围 `0~247B`
- 这些字段如何映射到 `wifi_tx_slots / ble_ce_slots / ble_ci_slots`
- 不提供 JSON 时，随机模式会先随机生成 `packet_bytes`，再推导时序参数

**Step 3: Verify docs and examples**

Run:

```bash
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json
```

Expected: command succeeds and `pair_parameters.csv` shows packet-size-driven fields.

**Step 4: Commit**

```bash
git add sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json README.md
git commit -m "docs: document packet-size driven wifi and ble timing"
```

### Task 8: Final regression pass

**Files:**
- No code changes expected

**Step 1: Run focused logic tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_logic.py -q
```

Expected: PASS.

**Step 2: Run focused runtime tests**

Run:

```bash
pytest sim_script/tests/test_pd_mmw_template_ap_stats_run.py -q
```

Expected: PASS.

**Step 3: Run one random-mode end-to-end execution**

Run:

```bash
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_config.json
```

Expected:
- command succeeds
- random mode samples packet sizes first
- derived timing fields appear in `pair_parameters.csv`

**Step 4: Run one manual-mode end-to-end execution**

Run:

```bash
python sim_script/pd_mmw_template_ap_stats.py --config sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json
```

Expected:
- command succeeds
- manual rows using `packet_bytes` derive valid WiFi/BLE timing

**Step 5: Final commit**

```bash
git add sim_script/pd_mmw_template_ap_stats.py sim_src/env/env.py sim_script/tests/test_pd_mmw_template_ap_stats_logic.py sim_script/tests/test_pd_mmw_template_ap_stats_run.py sim_script/pd_mmw_template_ap_stats_config.json sim_script/pd_mmw_template_ap_stats_manual_pairs_config.json README.md
git commit -m "feat: drive wifi and ble scheduling from packet sizes"
```
