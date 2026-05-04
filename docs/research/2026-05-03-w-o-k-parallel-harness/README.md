# Standalone W_O K-parallel harness — design (ratified)

**Status:** design ratified 2026-05-03 with user overrides on all six
open questions; ready for implementation.

## Purpose

Validate that splitting the β-coop kernel's W_O GEMV across more CTAs
reduces W_O wall-clock and (via R2-R4 coupling) total kernel time. The
same vehicle is also the only viable path to an NCU
`--replay-mode application` roofline classification of W_O.

## Scope

**v1 = standalone W_O-only microkernel + harness, with synthetic
pre-populated `attn_output`.** The microkernel is a self-contained
CuTe DSL kernel that lives under
`docs/research/2026-05-03-w-o-k-parallel-harness/`, **not** a
modification to the production β-coop kernel in
`phase_e_kernel.py`. v1 deliberately does not touch the production
kernel: the harness validates W_O scaling in isolation; landing
`wo_split` into the full β-coop kernel (with the new pre-W_O
barrier) is v2 production integration, gated on v1 evidence.

| In v1 | Deferred to v2+ |
|---|---|
| Standalone W_O-only microkernel (Constexpr `wo_split` ∈ {1, 2, 4, 8}) | `wo_split` integrated into the full β-coop kernel |
| Scratchpad-slot reduction (extending the existing 4-slot pattern, no attention) | Atomic-add reduction |
| Synthetic W_O-focused inputs (`attn_output`, W_O FP4 weights/scales) | Real-input snapshots from `vllm serve` |
| Post-W_O grid barrier (existing primitive) | New pre-W_O attention→W_O barrier (production variant only — see §1b) |
| `--ncu` self-re-exec under `ncu --replay-mode application` | NCU on the in-server kernel (deferred indefinitely; structural incompatibility) |
| 4-CTA baseline correctness anchor: Torch FP32 reference | Bit-exact equivalence (impossible) |

## Why this exists

PR #6 verdict
(`benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-beta-region-breakdown/summary.md`,
commit `46ad9bbc5`):
landed at 36% K-reducible — strict matrix gate
(`docs/research/2026-05-02-beta-region-breakdown/README.md:19`)
was not met. W_O proceeds only to validation because R2-R4 coupling
creates a plausible recoverable path; the matrix-gate's NCU half
also remained unsatisfied.

Two follow-up NCU attempts on 2026-05-03 (`ncu-attempt1/`,
`ncu-attempt2/` under the breakdown dir) showed in-server NCU is
structurally incompatible with this β-coop kernel:
`--replay-mode kernel` deadlocks on the cooperative-launch grid
barrier (CLAUDE.md §8); `--replay-mode application` requires the
app to exit between metric passes, which `vllm serve` does not.
Both problems disappear in a short-lived process. The harness is
therefore the only viable vehicle for both:

1. **W_O CTA-scaling evidence.** Per-launch wall-clock and
   effective bytes/s at 4 / 8 / 16 / 32-CTA W_O.
2. **NCU roofline classification.** A self-re-exec under
   `ncu --replay-mode application` for memory-bound vs
   compute-bound determination.

These are **distinct claims**. Bandwidth telemetry from the
harness's own per-launch numbers is **not** a substitute for an
NCU roofline classification — it answers "does W_O scale at higher
CTA counts?", not "is W_O memory-bound?". The harness is the
*vehicle* for both.

## Contract

| # | Requirement | Detail |
|---:|---|---|
| 1 | Single cold process | Python script that launches the kernel a finite number of times per config and exits. No daemon, no server. NCU `--replay-mode application` requires this property. |
| 2 | Configurable W_O CTA count | `wo_split` ∈ {1, 2, 4, 8} → `total_wo_ctas = num_kv_heads * wo_split` ∈ {4, 8, 16, 32}. Constexpr-parameterized in the β-coop kernel source (see §1). |
| 3 | Per-launch reports | wall-clock μs (CUDA event-timed), effective bytes/s (W_O weight read + activation read + slot writes), reduction mode used (`scratchpad` for v1), correctness delta vs 4-CTA baseline (rtol/atol on the W_O output buffer). |
| 4 | `--ncu` re-exec | Flag that re-execs the same script under `ncu --replay-mode application <python> <self> <args without --ncu>`, with NCU sections sufficient for memory-bound / compute-bound classification. |
| 5 | Cooperative launch optional | `--no-cooperative` flag. Default is `cooperative=True` (required for the kernel's grid barrier). The flag exists so kernel-replay deadlock can be reproduced separately if needed. |
| 6 | Artifacts under existing trace tree | All outputs under `benchmarks/nvllm/traces/cute_paged_attn/2026-05-03-w-o-k-parallel-harness/`. No `/tmp` writes for evidence files. |

## Ratified design (with user overrides)

### §1. W_O variants — parameterized via `wo_split` Constexpr (v1 microkernel)

The naïve `bx==0 && by<num_wo_ctas` model would conflate W_O
splitting with the existing attention producer gate and cannot
represent 16/32 CTAs cleanly (4 KV heads × wo_split slots is the
natural decomposition; flat `num_wo_ctas` ignores per-KV-head
structure).

**Gate model (ratified):**

| Symbol | Value | Notes |
|---|---|---|
| `wo_split` | Constexpr int ∈ {1, 2, 4, 8} | Sub-split per KV head |
| `num_kv_heads` | 4 (Qwen3.5-27B GQA) | Unchanged |
| `total_wo_ctas` | `num_kv_heads * wo_split` | 4 / 8 / 16 / 32 |
| W_O consumer gate | `bx < wo_split && by < 4` | New |
| Slot id | `wo_slot = by * wo_split + bx` | KV-head-major; preserves baseline at `wo_split=1` (slot id == `by`) |
| K range per CTA | `k_start = (k_dim * bx) // wo_split`, `k_end = (k_dim * (bx + 1)) // wo_split` | Robust integer-divide bounds (no clean-divide assumption) |

**v1 microkernel** lives in `microkernel.py` under this dir; it
adapts the existing β-coop W_O+gather portion (`phase_e_kernel.py`,
commit `46ad9bbc5`):

- `:2861` — `total_ctas_per_seq_attn = 4`; the microkernel uses
  `total_wo_ctas = num_kv_heads * wo_split`. `wo_output` shape is
  `(nat, total_wo_ctas, hidden)`.
- `:4103` — slot id `cta_idx = bx * num_kv_heads + by`; the
  microkernel uses `wo_slot = by * wo_split + bx`.
- `:4053-4101` — W_O K loop; the microkernel uses the robust K
  bounds above to slice the K range per `bx`.
- `:4233-4242` — gather loop bound becomes `total_wo_ctas`;
  semantics unchanged (sum all slots).

**v1 microkernel does NOT need attention→W_O sync.** The harness
pre-populates `attn_output` synthetically before launch, so
there is no producer→consumer cross-CTA dependency inside the
microkernel — every consumer CTA reads from a buffer that is
already valid in global memory. The post-W_O grid barrier (between
W_O slot writes and the gather) uses the **existing** primitive at
`:4371` unchanged.

### §1a. Compile-time elision for `wo_split == 1` (v1 microkernel)

To make the round-trip gate (caveat #7) meaningful, the
`wo_split == 1` path of the microkernel must be code-structurally
equivalent to the unmodified W_O+gather portion of the production
kernel — no extra checks, no idle code paths. This means the new
gates (`bx < wo_split` etc.) and the K-bound math must be
**Constexpr-elided** at `wo_split == 1`:

- `bx < wo_split` collapses to `bx < 1` ≡ `bx == 0` at compile
  time → identical to today.
- `k_start = (k_dim * 0) // 1 = 0`, `k_end = (k_dim * 1) // 1 =
  k_dim` → loop bounds identical to today's `while k_idx < k_dim`.
- Slot id `by * 1 + 0 = by` → identical to today's
  `cta_idx = bx * num_kv_heads + by` with `bx==0`.

Use `cutlass.const_expr(...)` / Constexpr Python-side branching
where the DSL supports it, so the wo_split=1 binary is byte-
equivalent in the W_O+gather region (modulo unavoidable codegen
diffs).

### §1b. v2 production integration — pre-W_O barrier with distinct counter (out of scope for v1)

When v1 evidence shows W_O scales and we land `wo_split` into the
full β-coop kernel, the production-shaped variant requires a new
pre-W_O grid barrier (because at `wo_split > 1`, consumer CTAs
with `bx >= 1` must wait for `bx == 0` to finish writing
`attn_output`). Three guardrails for that downstream work:

1. **Distinct counter, not a shared slot.** The current
   `grid_barrier_i32` is host-zeroed once before launch
   (`phase_e_kernel.py:3091`); the post-W_O barrier increments it
   and waits on `total_ctas_per_seq_grid` (`:4371`). There is no
   in-kernel reset before reuse. The pre-W_O barrier therefore
   needs **either a distinct `pre_wo_barrier_i32` buffer or a
   distinct per-seq counter slot**. Sharing the existing post-W_O
   counter would require phase thresholds, which is fragile.
2. **Compile-time elide the pre-W_O barrier at `wo_split == 1`,**
   so baseline cost is genuinely zero. Same Constexpr pattern as
   §1a.
3. **Robust K bounds** identical to v1 microkernel.

Production-side touchpoints (commit `46ad9bbc5`):

- `:3091` — `grid_barrier_i32` host-zero allocation (add second
  buffer or extend slot count).
- `:3542` — Phase 1 entry gate (`bx==0 && by<4`); attention
  producers stay here.
- `:4006` — region 2 entry comment update.
- `:4371` — existing post-W_O barrier; new pre-W_O barrier is a
  sibling primitive at the attention→W_O boundary.
- All §1 microkernel touchpoints listed above also apply.

### §2. Reduction mode — scratchpad-first

Existing per-CTA FP32 slot writes (`phase_e_kernel.py:4103-4111`)
and gather (`phase_e_kernel.py:4233-4242`) already implement the
scratchpad-reduction pattern at `total_wo_ctas = 4`. Extending the
slot count to `total_wo_ctas = num_kv_heads * wo_split` is a
quantitative extension of existing infrastructure.

Atomic-add reduction is **deferred to v2** — gated on whether
scratchpad establishes that W_O scales *at all* at 8/16/32 CTAs.
If scaling stalls, atomics won't help (contention will be worse,
not better); if scaling holds, atomics may or may not improve over
scratchpad (separate experiment).

### §3. Input synthesis — W_O-focused (production-equivalent dequant)

Synthesise everything the production W_O dequant consumes — the
microkernel must mirror `w_dequant = w_f32 * sf * wo_gs` exactly
(`phase_e_kernel.py:4078-4082`, commit `46ad9bbc5`):

- `attn_output`: post-attention activations
  (post-pre-attn-RMSNorm + post-A+B+C compute), bf16, shape
  `[num_active_tokens, num_q_heads * head_dim]`.
- `wo_weight`: NVFP4 packed bytes (uint8), shape
  `[hidden_size, num_q_heads * head_dim // 2]`.
- `wo_scales`: per-K-group block scales in **FP8 E4M3** swizzled
  layout (`_ld_swizzled_scale` accessor at `kernel.py:773-803`).
  Shape `[numMTiles, numKTiles, 32, 4, 4]`, dtype `uint8`. For
  Qwen3.5-27B W_O: 40 × 96 × 32 × 4 × 4 bytes ≈ 1.9 MB.
- `wo_gs`: scalar fp32, single value (already inverted at load
  time per `feedback_nvfp4_dequant_convention.md`).

Per-K-group scales **must** be included — dropping them makes the
memory stream ~11% lighter and biases the NCU memory-vs-compute
classification (a borderline verdict that would otherwise go
"memory-bound" can flip "compute-bound" under a lighter stream).
Production parity on the dequant is load-bearing for both the
round-trip claim (caveat #7) and the classification claim (NCU
roofline).

**Do NOT synthesise:** Q/K/V activations, KV cache pages,
seq/routing metadata. The harness is not testing the full Phase 1
path — it's testing W_O scaling.

If at v1 evaluation the synthetic-input scaling curve is
suspicious (sub-linear in unexpected ways), that's the trigger to
add v2 real-input snapshots from a captured `vllm serve` decode
step.

### §4. Correctness — rtol/atol + Torch FP32 reference (chained-order is the gate)

Two Torch FP32 references in `torch_reference.py`:

- **`reference_chained_fma`** — authoritative correctness gate.
  Mirrors the kernel's reduction order: per-KV-head chained FMA
  along K (sequential `a = a + w_dequant * attn_val` over the
  full per-head K range), then a 4-way sum across KV heads. Same
  FP4 decode, same swizzled FP8 E4M3 scale lookup, same global
  scale, same K-iteration order, same per-output-row accumulation
  order. At `wo_split == 1` the kernel and the reference share
  the *same* algorithm, so any rtol-failure here is a real bug.
- **`reference_matmul`** — diagnostic only. Vectorised
  `attn_fp32 @ weighted.T`. Mathematically equivalent to the
  kernel's W_O dot product but reduction-order is whatever
  cuBLAS picks (typically tree-reduction). Sub-ULP FP32 reorder
  drift between this and the kernel is real and bounded
  (~1e-3 relative on K=6144 dot products with mixed-sign data, per
  our smoke test) — exactly the "knife-edge argmax" pattern in
  `feedback_distilled_knife_edge.md`. **Used for reporting only**:
  artifacts include max abs / max rel against this reference so
  reorder drift is visible, but it is **not** a pass/fail oracle.

**Pass/fail gate** (per launch, per CTA count):

- `wo_split == 1`: `torch.allclose(kernel_out, reference_chained_fma_out, rtol=1e-3, atol=1e-4)` — strict (kernel and chained reference share reduction order; passes bit-exact in practice).
- `wo_split` ∈ {2, 4, 8}: `torch.allclose(kernel_out, reference_split_order(wo_split, …), rtol=1e-3, atol=1e-4)` — strict against the variant's *own* reduction tree (per-CTA chained K with the same `k_start/k_end = (K_per_head * bx) // wo_split` partition; gather sums slots `0..total_wo_ctas-1` in `wo_slot` order, matching `phase_e_kernel.py:4103, 4233-4242`). 

This split:

| | wo_split=1 | wo_split ∈ {2,4,8} |
|---|---|---|
| Authoritative gate | `reference_chained_fma` (production-order) | `reference_split_order(wo_split)` (kernel-order) |
| Diagnostic drift | vs `reference_matmul` | vs `reference_chained_fma` AND vs `reference_matmul` |

Cross-split drift (variant vs `reference_chained_fma`) is **expected FP32 reorder drift** on K=6144 shapes with mixed-sign data — measured and reported, not used as the implementation bug gate. The scaling-evidence claim is not blocked by a known reduction-tree change; correctness for variants is gated by the direct oracle for *their* tree.

Per-launch CSV / JSON reports:
- `max_abs_split_order`, `max_rel_split_order`, `passes_split_order` (pass/fail) — authoritative
- `max_abs_chained`, `max_rel_chained` (diagnostic — production-order drift)
- `max_abs_matmul`, `max_rel_matmul` (diagnostic — cuBLAS-tree drift)

### §5. NCU sections

Reuse the same sections `run_ncu.sh:46-52` already specifies:

- `MemoryWorkloadAnalysis` — DRAM/L1/L2 throughput; load-bearing
  for memory-bound classification.
- `ComputeWorkloadAnalysis` — SM throughput, SM utilisation.
- `LaunchStats` + `Occupancy` — sanity context.
- `SchedulerStats` — warp utilisation.

`ncu --replay-mode application` replays the entire process per
metric pass. The harness `--ncu` mode therefore launches once per
invocation and exits; NCU re-execs the script as many times as
the sections require.

**Wording rule (carried forward):** Harness telemetry and NCU
classification are kept separate — telemetry answers "does it
scale?", NCU answers "is it memory-bound?".

### §6. Artifact layout — scratchpad-only initially

```
benchmarks/nvllm/traces/cute_paged_attn/2026-05-03-w-o-k-parallel-harness/
  metadata.json                       # commit, image, host, env, kernel-config hashes
  baseline_4cta_scratchpad/
    timing.csv                        # per-launch wall-clock + bytes/s
    correctness_baseline.npy          # 4-CTA reference output (W_O buffer)
    correctness_vs_torch_fp32.json    # max_abs, max_rel, pass/fail vs Torch FP32 ref
  variant_8cta_scratchpad/
    timing.csv
    correctness_delta.json            # max_abs/max_rel vs baseline AND vs Torch FP32 ref
  variant_16cta_scratchpad/
    ...
  variant_32cta_scratchpad/
    ...
  ncu/
    variant_<N>cta_scratchpad/
      command.txt                     # exact ncu invocation
      config.json                     # harness args used
      ncu_stdout.log
      ncu_stderr.log
      <kernel>.ncu-rep                # binary trace
      <kernel>_ncu.csv                # CSV export
  summary.md                          # written by hand after the run
```

Atomic variants are **not** emitted in v1. The atomic v2 follow-up
will add `variant_<N>cta_atomic/` siblings under the same date dir
when implemented.

Pre-run scripts live in
`docs/research/2026-05-03-w-o-k-parallel-harness/`; evidence
outputs live under `benchmarks/...` per
`feedback_benchmarks_evidence_only.md`.

## Runtime environment

CuTe DSL is not in the host `.venv` (verified 2026-05-03:
`ModuleNotFoundError: No module named 'cute'`). The harness runs
inside the `nvllm:gb10` container via `docker run --rm` per
invocation. Disk-cache the JIT artefacts (set
`B12X_CUTE_COMPILE_CACHE_DIR=/work/harness/.cute_cache` and call
`apply_disk_cache_patch` at process start) so NCU
`--replay-mode application` doesn't pay the 24 s cold-compile cost
on every metric pass.

## Reproduction (eventual)

```bash
# Single config (one CTA count, no NCU)
.venv/bin/python docs/research/2026-05-03-w-o-k-parallel-harness/run_harness.py \
  --wo-split 2 --launches 50 \
  --out benchmarks/nvllm/traces/cute_paged_attn/2026-05-03-w-o-k-parallel-harness/variant_8cta_scratchpad/

# Full sweep (all CTA counts, scratchpad only — v1)
bash docs/research/2026-05-03-w-o-k-parallel-harness/run_sweep.sh

# NCU classification of one config
.venv/bin/python docs/research/2026-05-03-w-o-k-parallel-harness/run_harness.py \
  --wo-split 8 --launches 1 --ncu \
  --out benchmarks/nvllm/traces/cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/variant_32cta_scratchpad/
```

## Caveats baked in

1. **Synthetic inputs assume bandwidth/SM behaviour transfers to
   real activation distributions.** If the scaling curve is
   suspicious, switch to snapshotted real inputs (Open
   question §3).
2. **Cooperative launch is required for the β-coop kernel's grid
   barrier.** Default `cooperative=True`; `--no-cooperative` is
   intended only for reproducing kernel-replay deadlock.
3. **NCU `--replay-mode application` works only if the harness is
   side-effect-free across passes.** No file writes that depend on
   prior passes; no GPU state carried between invocations. Each
   NCU pass starts from a cold process.
4. **Wall-clock is per-launch, not per-step.** The β-coop kernel
   runs `n_layers_fused × decode_steps` times in production. The
   harness runs ONE call per launch. Multiplying by 8 (lower-8
   fused) to get a per-step number is the consumer's
   responsibility.
5. **W_O is one site of many.** Validating W_O scaling does NOT
   predict total kernel speedup. Amdahl's law on the unaccounted
   ~26% (kernel epilogue + launch overhead) and the pre-W_O
   Phase 1 work (~0.65%) caps the achievable kernel-level speedup
   independent of W_O's own scaling factor.
6. **Harness telemetry is not a roofline classification.** Per
   the user's wording rule: bandwidth telemetry from the harness
   answers the scaling question, not the bottleneck-shape
   question. Both answers come from the same vehicle but are
   distinct claims.
7. **`wo_split=1` baseline must pass the Torch FP32 reference
   gate FIRST.** Before any 8/16/32 variant runs, the
   parameterized microkernel at `wo_split=1` must match the Torch
   FP32 reference within `rtol=1e-3, atol=1e-4` on the synthetic
   inputs. Failure here indicates the parameterization itself
   broke the W_O+gather math — fail fast and stop the sweep.
   Variants 2/4/8 then compare against BOTH the wo_split=1
   microkernel output AND the Torch FP32 reference. (True
   bit-equivalence to the unmodified production kernel cannot be
   tested directly because the production kernel needs full
   Phase 1 inputs the harness deliberately does not synthesise;
   the Torch FP32 reference is the meaningful anchor for v1.)

## Decision-record line (carried forward)

> Matrix gate was not met; W_O proceeds only to validation because
> R2-R4 coupling creates a plausible recoverable path.
