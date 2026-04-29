# Phase 5 Boundary Baseline

**Date:** 2026-04-28
**Branch:** `feat/uber-kernel-migration`
**Reference commit:** `e7c9c38e9` (Phase 5 SHIPPED)

## One-line baseline

Phase 5 reduces the full-attn route to **one replay-time Python
boundary per full-attn layer**; remaining boundary overhead is
dominated by **architecture-level 16 full-attn + 48 GDN splitting
ops per token**.

## Per-token boundary inventory (from Phase 5 trace)

Source:
`benchmarks/nvllm/traces/phase_5_paged_skip/2026-04-28-restored/profile_kernels.txt`

| Op | Calls / 32-iteration trace | Calls / token |
|---|---|---|
| `vllm::cute_beta_coop_run` | 512 | 16 (one per full-attn layer) |
| `vllm::gdn_attention_core` | 1536 | 48 (one per GDN layer) |
| `_C_cache_ops::reshape_and_cache_flash` | 512 | 16 |

## Why this matters for downstream phases

- **β-coop is at the architectural minimum.** Reducing β-coop boundary
  count further requires layer-fusion, not op-fusion. The remaining
  lever for β-coop specifically is per-call Python overhead inside the
  boundary (Phase 6a).
- **GDN dominates the architectural splitting count.** 48 GDN
  boundaries / token is 3× the β-coop count. If post-Phase-6a steady
  state is still slow, GDN is the next target (Phase 6b).
- **Non-splitting ops (`cute_residual_mirror`, `cute_mlp_forward`,
  `cute_phase_e_dispatch`) are NOT boundaries.** Their Python bodies
  run only at capture time, not per replay. They affect graph size
  / capture cost, not steady-state Python dispatch.

## How to interpret future phase deltas against this baseline

A perf delta from a future phase can claim "boundary-count reduction"
only if the splitting-op counts in the trace's `profile_kernels.txt`
fall below this table. A delta claiming "Python overhead reduction"
within an existing boundary should show a Self CPU drop on the same
splitting op — not a count change.
