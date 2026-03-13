# Per-Link Scheduling Constraints Design

**Goal:** Make scheduling constraints use per-link WiFi/BLE or override-specific packet length and bandwidth, so feasibility checks and BLER evaluation share the same PHY assumptions.

## Context

The current environment model already supports per-link packet length and bandwidth for `evaluate_bler()`, but the scheduling constraint path still uses global scalar `packet_bit` and `bandwidth`. That means `min_sinr`, transmit power normalization, and `h_max` are still derived from one shared PHY configuration even when WiFi and BLE links now evaluate BLER with different parameters.

## Approved Direction

Use per-link thresholds everywhere in the scheduling constraint path.

- Compute `pair_min_sinr[k]` from each link's own `packet_bits[k]`, `bandwidth_hz[k]`, `slot_time`, and `max_err`.
- Use per-link bandwidth in transmit-power and received-power normalization.
- Produce per-link `h_max[k] = S_gain[k, k] / pair_min_sinr[k] - 1`.
- Keep the public shape of `generate_S_Q_hmax()` unchanged: it still returns `(S_gain, Q_conflict, h_max)`, but `h_max` now reflects per-link PHY requirements.

## Data Model

Add normalized arrays alongside the recently added BLER parameter arrays:

- `pair_min_sinr`
- `device_min_sinr`
- `user_min_sinr`

These remain aligned with the existing pair/device/user one-to-one compatibility mapping.

## Constraint Flow

1. `pair_packet_bits` and `pair_bandwidth_hz` are already normalized during pair setup.
2. `_compute_min_sinr()` becomes a per-link array computation using `bisection_method()` per link.
3. `_compute_txp()` uses `pair_min_sinr[k]` and `pair_bandwidth_hz[k]` for each transmitting link.
4. `_compute_state()` and `_compute_state_real()` use the transmitter-link bandwidth array when converting received power against noise.
5. `generate_S_Q_hmax()` computes `h_max` elementwise with the per-link threshold array.

## Scope Boundaries

This change updates scheduling-feasibility math only inside `env`.

- In scope: `min_sinr`, transmit-power normalization, state matrices, `h_max`, compatibility arrays, and tests.
- Out of scope: algorithm code in `sim_src/alg`, CSV schema changes, plotting logic, and changes to caller APIs beyond optional constructor parameters that already exist.

## Testing Strategy

Use TDD with focused regression tests:

- Verify per-link `min_sinr` differs for WiFi and BLE defaults.
- Verify per-link override arrays are honored by `_compute_min_sinr()`.
- Verify `generate_S_Q_hmax()` uses per-link `pair_min_sinr` instead of one shared scalar.
- Re-run the recently added BLER parameter tests and a representative script-facing regression test.
