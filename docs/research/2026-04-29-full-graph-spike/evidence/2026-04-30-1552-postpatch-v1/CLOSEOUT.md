# β-coop Persistent Buffers Patch v1 — Closeout (2026-04-30)

## Verdict

**Patch v1 does not solve C2.** Hoisting four of five workspace tensors
to persistent attributes on `CutePagedAttentionImpl` (and preserving the
three captured counter `.zero_()` calls before launch) was insufficient
to close the FULL-graph replay-divergence bug class on lower-8 layers.

**Strongest remaining suspect: `wo_output`** — the only of the five
buffers that received no captured reset in v1. Each replay of the FULL
graph hits the same persistent address for `wo_output`, but with stale
contents from the prior decode step (4 attn-CTAs per seq accumulate into
it via atomic_add at Phase 1, and v1 has no reset before re-accumulation
on the next replay).

**Production recommendation: PIECEWISE + β-coop remains the supported
path** (v0.3.0 status quo). C0 and C2-PIECEWISE both PASS post-patch —
no PIECEWISE regression. FULL+β-coop blocker remains open; v2 work
required.

## Evidence table

Code commit: `1cc51ab95` on branch `feat/cute-beta-coop-persistent-buffers`.

| Configuration | C2 result | Evidence |
|---|---|---|
| C0 PIECEWISE+β-coop GSM8K-sanity (8/8) | PASS | `evidence/2026-04-30-1527/c0_summary.md` |
| C2 PIECEWISE+β-coop control (replay coherence) | PASS (unique=1, cross indep) | `evidence/2026-04-30-1537/c2_replay_coherence.md` |
| **C2 FULL+β-coop lower-8 (Gate 1)** | **FAIL (unique=2, cross dep)** | `evidence/2026-04-30-1548/c2_replay_coherence.md` |
| C2 FULL+β-coop all-16 (Gate 2) | not run (Gate 1 FAIL) | — |

## What v1 changed

Hoisted five buffers from per-call `torch.zeros(...)` inside
`run_beta_coop_full` to persistent attributes on
`CutePagedAttentionImpl`, allocated in `attach_mlp_fusion`:

| Buffer | Shape | Reset mechanism (v1) |
|---|---|---|
| `_phase_e_coop_wo_output` | `[max_num_seqs, 4, hidden]` f32 | **none — no captured reset** |
| `_phase_e_coop_mlp_partial_fp32` | `[max_num_seqs, slc, hidden]` f32 | in-kernel CTA-local (Phase 3.2.5) |
| `_phase_e_coop_mlp_arrival_count` | `[max_num_seqs, num_k_tiles]` u32 | host-side `.zero_()` before launch + in-kernel atomic decrement |
| `_phase_e_coop_grid_barrier_i32` | `[max_num_seqs]` i32 | host-side `.zero_()` before launch |
| `_phase_e_coop_phase1_arrival_count` | `[max_num_seqs]` i32 | host-side `.zero_()` before launch |

The three host-side `.zero_()` calls at `phase_e_kernel.py:3036-3038` are
captured cudaMemsetAsync nodes (per `feedback_log_line_vs_cache_state`
and the original v3 narrowing finding that small counter `.zero_()` is
graph-safe but large FP32 tensor `.zero_()` hangs capture). v1 deferred
the FP32 buffer reset to the in-kernel CTA-local mechanism for
`mlp_partial_fp32` (each CTA zeros its own slot before atomic
accumulation, Phase 3.2.5). `wo_output` got no reset on the assumption
that 4 attn-CTAs per seq would fully overwrite the output during Phase 1
Phase A — that assumption appears wrong under FULL replay.

## Failure pattern (Gate 1)

Same prompt repeated 8 times under FULL graph + β-coop on lower-8 layers:
- Replays 0,2,5: continued with `"Q: What is the capital of Japan? A: Tokyo."`
- Replays 1,3,4,6,7: continued with `"Q: What is the capital of Italy? A: Rome."`

Cross-prompt independence violated:
- Prompt A (capitals) run first → `Italy/Rome` continuation
- Prompt A run after prompt B (different topic) → `Japan/Tokyo` continuation

Both grammatical, both correct as far as content, but the model alternates
between equally-likely tokens at the divergence point in a pattern
correlated with prior replay state — characteristic of stale-state from
shared workspace persisting across replays.

## Why v1 failed despite hoisting

v1 fixed only HALF the bug class:
1. ✅ **Address stability** — persistent buffers ensure each replay hits
   the same captured address. Plain `torch.zeros` returned recycled
   graph-pool addresses, breaking the captured-address contract.
2. ❌ **Content freshness** — addresses are stable, but `wo_output` (and
   to a lesser extent the other f32 buffers, despite the in-kernel
   reset for `mlp_partial_fp32`) is not zeroed before the second replay.
   The kernel atomic-adds into a buffer that already contains the prior
   replay's accumulation.

The pre-patch failure was "stale capture address." The post-patch
failure is "stale content at stable address." Both produce the same
end-user symptom (replay divergence) but require different fixes.

## Next step (out of scope for this plan)

**v2: small captured reset op for `wo_output` before the cooperative
launch** (spec §6.4, deferred from v1). The reset must:

- Run on the same stream as the kernel launch (so it's a captured graph
  node, not a host-side allocator action).
- Zero only `[:nat]` rows (slice-aware to avoid wasting bandwidth on
  unused tail).
- Be small enough to capture without hanging (per v3 narrowing finding,
  large FP32 `.zero_()` hangs capture — use a custom `direct_register_custom_op`
  that calls `cudaMemsetAsync` directly, NOT `torch.Tensor.zero_()`).

A new spec + plan are required for v2 — DO NOT attempt v2 inside this
plan's branch. Spec §6.4 sketches the design; production-ready spec
would also need to address whether `mlp_partial_fp32`'s in-kernel reset
needs to be replaced with a host-captured reset for the same reason.

## Followup investigation candidates

1. **Bisect v2** — once v2 reset for `wo_output` lands, re-run Gate 1
   lower-8. If still FAIL, escalate to also reset `mlp_partial_fp32`
   host-side (replacing the in-kernel CTA-local reset, which only runs
   inside Phase 3 and may not preempt Phase 1 atomic-add into stale
   `wo_output`).
2. **Track #35175 pattern variants** — upstream's fix copies fresh data
   into persistent buffers per call. Our case has no fresh data to copy
   (the buffers are pure scratch); the right shape is "captured zero,
   not captured copy_in."
3. **Track #40969** — still OPEN upstream. Same hardware (GB10/SM12.1),
   same cudagraph_mode FULL_AND_PIECEWISE. Watch for an upstream fix
   that may inform our v2 approach.

## Status

- Branch `feat/cute-beta-coop-persistent-buffers` HEAD: `1cc51ab95`
- Patch v1 is **shippable as a no-op-for-FULL but PIECEWISE-clean
  refactor**. PIECEWISE is unaffected and the persistent-buffer
  scaffolding is ready for v2 to extend.
- FULL+β-coop blocker (`project_full_graph_blocked.md`) remains open,
  pending v2.
