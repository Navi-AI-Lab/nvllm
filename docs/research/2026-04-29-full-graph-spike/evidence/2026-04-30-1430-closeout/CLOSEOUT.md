# FULL_AND_PIECEWISE + β-coop — Closeout (2026-04-30)

## Verdict

**FULL_AND_PIECEWISE + β-coop is BLOCKED.**

C1 proves FULL dispatch reaches the intended flat V1 runner. C2 controls show
β-coop is deterministic under PIECEWISE and FULL is deterministic without
β-coop. A single β-coop layer under FULL is deterministic, but lower-8 β-coop
layers fail replay coherence and another 8-layer set hung during capture on
the first attempt (passed on retry, same code). Therefore the failure is
**cumulative / FULL-integration-specific, not a standalone β-coop kernel
math bug**.

**Production recommendation: PIECEWISE + β-coop remains the supported path.**
C3 for FULL remains not meaningful until C2 passes.

## Evidence table

| Configuration | C2 multi-token | Evidence |
|---|---|---|
| FULL + β-coop OFF | PASS | `evidence/2026-04-30-0850/c2_replay_coherence.md` (control) |
| FULL + β-coop [3] (1 layer) | PASS | `evidence/2026-04-30-1303/c2_replay_coherence.md` |
| FULL + β-coop [3..31] (lower-8) | **FAIL (3 unique)** | `evidence/2026-04-30-1311/c2_replay_coherence.md` |
| FULL + β-coop [35..63] (upper-8) | PASS (retry) | `evidence/2026-04-30-1406/c2_replay_coherence.md` |
| FULL + β-coop all 16 | FAIL (2 unique) | `evidence/2026-04-30-0826/c2_replay_coherence.md` |
| PIECEWISE + β-coop all 16 | PASS | `evidence/2026-04-30-1144/c2_replay_coherence.md` |

## Capture-side flake (separate signal)

Upper-8 hung 40+ min on the first attempt at the FULL probe (96% GPU, 101% CPU,
zero log progress), then READY in 4 min on a clean retry with identical code,
identical sync, identical model. Two earlier hangs of similar shape:
- Probe v1 (`setattr(self, ...)` in `dispatch_cudagraph` neighborhood) — fixed
  by switching to module-level booleans (`feedback_no_self_mut_in_cudagraph_dispatch`).
- Schema-arg variant (added `attn_input_scratch` to op signature with
  `mutates_args` declaration) — reverted; capture hung 51 min.

FULL graph capture on β-coop is fragile to *small* surface changes and
sometimes hangs even with no changes.

## Patches that were tried and reverted

1. **v1–v7 workspace-reset patches** (counter zeroing, mlp_partial reset,
   wo_output reset, atomic→load-add-store) — all FAIL or capture hang.
   Reverted to v6 baseline (in-kernel CTA-local mlp_partial reset +
   mlp_arrival_count decrement; atomics preserved on accumulation).
2. **Scratch-arg explicit custom-op param** — added `attn_input_scratch` to
   `cute_beta_coop_run` signature with mutates_args, plumbed through
   `_beta_coop_op.py` → `impl.forward(...)` → `run_beta_coop_full`. **FULL
   capture hung 51 min**, never reached `Application startup complete`.
   Reverted (`_beta_coop_op.py`, `_backend.py`, `qwen3_5.py`,
   `_sync_host_edits.sh`).

## Upstream survey (2026-04-30) — pattern matches

Verified (numbers + states cross-checked against `gh` API, not just subagent
summary):

- **vllm-project/vllm#35175 (MERGED, 2026-03-26)** — "Restore CUDA graph
  persistent buffers for FP8 FlashMLA decode". Verbatim symptom from PR body:
  *"Under the default O2 optimization level (FULL_AND_PIECEWISE CUDA graphs),
  the graph captures tensor addresses during recording. On replay,
  freshly-allocated tensors live at different addresses, so the kernel reads
  stale metadata from the originally-captured addresses, producing garbled
  output that starts normal then degenerates after ~50 tokens."* Direct
  analog of our "first 8 chars stable, decode tokens 2+ diverge." Fix: copy
  fresh data into pre-allocated persistent CUDA-graph buffers (allocate in
  `__init__`, reuse with copy-in per call).

- **vllm-project/vllm#40969 (OPEN)** — "DeepSeek-V4-Flash hangs after ~6
  requests with cudagraph_mode=FULL_AND_PIECEWISE + chunked prefill on SM 12.x
  (GB10)". Same hardware (GB10/SM12.1), same cudagraph mode, same hang shape
  (100% SM, no token output). Not yet fixed upstream.

- **vllm-project/vllm#26678 (CLOSED)** — `mutates_args` ops do not correctly
  set `origin_node` in `torch/_inductor/graph.py:1865-1881`, breaking inductor
  graph_partition. Our `_beta_coop_op.py` declares
  `mutates_args=["output_rmsnorm","output_residual","output_mlp"]` — exactly
  the construct flagged in the issue.

- **vllm-project/vllm#37363 (OPEN)** — "fix piecewise CUDA graph bugs with
  splitting_ops". When a splitting_op allocates new tensors, the next piece's
  CUDA graph replays with stale addresses → silent data corruption. β-coop
  is registered as a splitting_op via `direct_register_custom_op`.

Full evidence: `upstream_search.md` (this directory).

## Hypothesis (informed by upstream)

Our symptom is `#35175` in CuTe form. The β-coop kernel allocates workspaces
inside `run_beta_coop_full` via `torch.zeros(...)` for `mlp_partial_fp32`,
`mlp_arrival_count`, `phase1_arrival_count`, `grid_barrier_i32`. These
allocations are observed via the graph-pool allocator at FULL capture; their
addresses are baked into the captured graph. On replay, fresh `torch.zeros`
calls return *different* graph-pool addresses; the kernel reads/writes to the
*originally-captured* addresses, which contain stale/aliased state by the time
later layers run. Cumulative-with-layer-count (1L PASS, 8L FAIL) tracks the
"more layers reuse the same recycled pool" pattern.

The fix shape, per #35175: pre-allocate all four workspaces in
`CutePagedAttentionImpl.__init__` (or first-call cache), and inside
`run_beta_coop_full` `.copy_(0)` / `zero_()` / use a persistent reset op
*captured into the graph*, never `torch.zeros(...)` per call.

This was the direction we tried via in-kernel CTA-local resets (v1–v7), but
those resets were inside the kernel body — they don't fix the fundamental
issue that the *python-side allocator returns a fresh tensor each call*. The
right fix is at the Python wrapper layer: persistent buffers + explicit
zero ops, not in-kernel reset.

## Why we are stopping today

Friend's framing (paraphrased from review): finding whether the threshold N
is 2, 4, or 6 does not create a shippable path unless we plan to support
"β-coop on a few layers only," which is unattractive as production. We have
enough to declare: cumulative drift, not a single bad layer; PIECEWISE
β-coop is correct and shipping; FULL+β-coop integration needs a structural
persistent-buffer rewrite per #35175 pattern.

## Next-action options (not pursued today)

1. **Cherry-pick #35175 pattern** — pre-allocate `mlp_partial_fp32`,
   `mlp_arrival_count`, `phase1_arrival_count`, `grid_barrier_i32` on the
   backend impl, copy/zero in before launch. Targeted, well-scoped, has
   upstream precedent. Estimated 1-day rewrite + re-verify C1/C2.
2. **Track #40969** — same hardware + same cudagraph mode, currently OPEN
   upstream. If/when it gets fixed, retest FULL+β-coop with the upstream
   patch applied.
3. **Accept the v0.3.0 status quo** — PIECEWISE+β-coop is in production. FULL
   is not necessary for current goals (single-user serving). Document as a
   known-limitation and revisit if perf needs it.
