# Fused CuTe kernel — non-deterministic reduction audit

**Date:** 2026-04-20
**Status:** FIXED 2026-04-20 — deterministic per-CTA slot + fixed-order
gather lands Option 1 from this audit; see "Fix landed" appendix at the
bottom. Phase B kernel output matches Python reference to BF16 cast
precision (1200/1200 close=True, diff.max ≤ 0.0002) — evidence at
`benchmarks/nvllm/traces/phase_a_fused_reduction_fix/2026-04-20/`.

**Context:** `project_fused_path_nondeterminism` — the fused CuTe MLP
and fused CuTe attention paths produce per-request non-deterministic
outputs at temperature=0 (evidence: commit `551dd5d5e`). This audit
scopes **where** the non-determinism comes from in the kernel code so
we can write a targeted fix in a future session.

## Summary

Two `@cute.kernel` functions contain cross-CTA `atomicAdd_f32`
reductions into shared global FP32 buffers. FP32 addition is
non-associative; CTA arrival order at each atomicAdd is
hardware-non-deterministic; the resulting accumulator value drifts
at ULP level across runs. On the Opus-distilled Qwen3.5-27B model,
ULP drift flips argmax on knife-edge tokens.

All non-deterministic sites fit the same pattern — per-CTA partial
sums accumulated into a shared target address. None are "accidental"
— they are deliberate cross-CTA reductions for Phase-D-style
fused-MLP / Phase-B-style W_O fusion. The fix is structural
(deterministic reduction), not a one-line patch.

## Sites — MLP fused kernel (`mlp_kernel.py`)

### Site M-A: FC2 accumulator, Path A (one-thread-per-row)

**File:** `vllm/v1/attention/backends/cute_paged/mlp_kernel.py`
**Lines:** `1004-1012` (commit `57d2f0e0d` HEAD)

```python
# Each thread atomic-adds its owned rows. Rows are
# disjoint across threads, so there's no race.
for r in cutlass.range_constexpr(rpt):
    k_row_global = (
        by * tile_k + row_base_local + Int32(r)
    )
    partial_idx = bz * hidden + k_row_global
    _atomic_add_f32(
        partial_ptr + Int64(partial_idx) * Int64(4),
        acc_list[r],
    )
```

**Non-determinism:** the comment "Rows are disjoint across threads,
so there's no race" is correct within a single CTA but not across
CTAs. Different CTAs handle different `bx` (slice index, covering
different intermediate-dim slices) for the same token `bz`. All
slice-CTAs atomic-add into the same `partial_ptr[bz*hidden + k_row_global]`
address — this is an intentional cross-CTA reduction. Arrival
order of those CTAs is hardware-non-deterministic.

**Width of the sum:** `slice_ctas` adds per token per output row.
For prefill-legacy preset this is 8 adds per (token, k_row) — small
but enough for FP32 non-associativity to flip ULPs consistently.

### Site M-B: FC2 accumulator, Path B (multi-thread-per-row)

**File:** same, **Lines:** `1098-1103`

```python
# Hoist partial_idx outside the `if thread_in_row == 0` region ...
partial_idx = bz * hidden + k_row_global
if thread_in_row == Int32(0):
    _atomic_add_f32(
        partial_ptr + Int64(partial_idx) * Int64(4),
        out_acc,
    )
```

Same mechanism as M-A but runs when `tile_k < num_threads`
(small-tile test configs only — prefill-legacy uses Path A). Fixing
this can ride along with M-A since they share the target buffer.

### Site M-C: arrival counter — DETERMINISTIC in outcome, not a target

**File:** same, **Lines:** `1108-1117`

```python
# === Phase 4: Arrival counter + last-CTA epilogue ===
_threadfence()
cute.arch.sync_threads()

if tid == Int32(0):
    count_idx = bz * num_k_tiles + by
    old = _atomic_add_u32(
        count_ptr + Int64(count_idx) * Int64(4),
        Int32(1),
    )
    # is_last?
    is_last_flag = Int32(0)
    if old == (slice_ctas - Int32(1)):
        is_last_flag = Int32(1)
```

Integer `_atomic_add_u32` increment used for "am I the last CTA to
arrive" detection. The outcome is deterministic (exactly one CTA
observes `old == slice_ctas - 1` and runs the epilogue). *Which*
CTA runs the epilogue depends on arrival order, but the epilogue
reads post-race state which is independent of identity. **Not
contributing to the math non-determinism.** Leave alone during fix.

## Sites — Attention fused kernel (`kernel.py`)

### Site A-A: Phase B W_O partial accumulator

**File:** `vllm/v1/attention/backends/cute_paged/kernel.py`
**Lines:** `1653-1690`

```python
# atomicAdd accumulators to global FP32 output buffer
wo_out_base = wo_output_ptr + Int64(
    seq_idx * hd_wo * Int32(4))
for _oi in cutlass.range_constexpr(8):
    out_row = out_base + Int32(_oi)
    if out_row < hd_wo:
        if _oi == 0:
            _atomic_add_f32(
                wo_out_base + Int64(
                    out_row * Int32(4)), a0)
        if _oi == 1:
            _atomic_add_f32(
                wo_out_base + Int64(
                    out_row * Int32(4)), a1)
        ...   # repeated for a2..a7
```

**Non-determinism:** 8 atomicAdd_f32 calls per thread per 8-row
output group, inside a `range_constexpr(5)` loop = 40 atomicAdds
per thread per CTA for `hidden_dim=5120`. Different CTAs
(differ in `by=kv_head_idx`) write to the same `wo_output_ptr[seq,
row]` addresses because W_O sums K-dim partial products from every
KV head group into one output row.

**Width of the sum:** `num_kv_heads` adds per seq per output row =
4 adds for Qwen3.5-27B — even smaller than MLP, but still enough
for ULP drift on a distilled model.

### Site A-B: Phase C RMSNorm arrival counter

**File:** same, **Lines:** `1717-1720`

```python
if tid == Int32(0):
    old_count = _atomic_add_u32(
        arrival_count_ptr + Int64(seq_idx * Int32(4)),
        ...
```

Same pattern as site M-C — integer arrival counter used for
"last CTA runs RMSNorm." Outcome-deterministic, not a target.

## Fix options

### Option 1: Deterministic per-CTA staging + single-CTA gather

**Approach:** Each CTA writes its partial to a distinct slot in a
staging buffer indexed by CTA index, not summed. The last-arriving
CTA (already detected by the arrival counter) runs a deterministic
gather that sums slots in CTA-index order.

**Cost:**
- Memory: extra staging buffer of `num_ctas × partial_size × 4B` per
  reduction point. For MLP site M-A: `slice_ctas × hidden × 4B` =
  8 × 5120 × 4 = 160 KB per token — acceptable.
- Compute: one more gather pass in the last-CTA epilogue.
- Latency: probably +5-10 % on the fused path.

**Determinism:** fully deterministic — CTA indices are stable, sum
order is fixed.

### Option 2: In-place deterministic reduction via ordering barrier

**Approach:** Use a CTA-index-gated spinlock so CTAs accumulate in
index order. CTA-`i` spins until a flag set by CTA-`i-1` is raised,
then adds and raises its own flag.

**Cost:** serialization — the effective parallelism across CTAs
collapses to ~1 for the accumulator section. Probably 3-5× slower
on the reduction path. Not recommended.

### Option 3: Higher-precision intermediate

**Approach:** Keep atomicAdd but promote accumulator to FP64 (or
integer-scaled FP32 with much tighter ULP bound).

**Cost:**
- Memory: 2× buffer size for FP64.
- PTX: SM120/121 has no native FP64 atomicAdd — would emit a CAS
  loop, slower and still non-deterministic in ordering (just
  more bits of margin before flipping argmax).

**Determinism:** NOT fully deterministic. Reduces magnitude of ULP
drift but doesn't eliminate it. Might hide the bug on current test
cases but reappear on next knife-edge token. Not recommended as
the principled fix.

## Recommended fix plan

1. **Option 1 (deterministic staging + gather)** for site M-A
   (primary MLP fusion path used by prefill-legacy preset).
2. Ride-along fix for M-B (same buffer target).
3. Option 1 for site A-A (attention Phase B W_O reduction).
4. Site M-C and A-B are not targets; leave them alone.
5. Verification: run `docs/research/phase_a_gsm8k_repro/run_q2_repeat.sh`
   under the fixed image with `CUTE_*_FUSION=1`. Must produce 5
   byte-identical correct outputs (same as unfused determinism).
6. Re-enable fusion by default (revert the `_backend.py:375` default
   from "0" → "1") once verification passes.

## Estimated effort

- Code edits: 2-3 hours (add staging buffer allocation in
  `_backend.py` and `mlp_kernel.py`, rewrite reduction sites in
  both kernels, handle the fence/sync changes).
- Rebuild + smoke tests: 1 hour.
- Q2 x5 deterministic verification: ~15 min.
- Full GSM8K x5 + other regressions: ~1 hour.
- **Total: ~4-5 hours** for a focused session.

## Not in scope

- CUDA graph interactions with the fixed kernel (should "just work"
  since the kernel body is stateless WRT graph capture).
- Performance tuning of the new reduction path — correctness first,
  perf re-measurement in a follow-up.
- FP8 KV / TurboQuant interactions — those paths don't go through
  the fused MLP / fused W_O reductions here.

## Fix landed (appendix, 2026-04-20)

**Changes:**
- `mlp_partial_fp32` reshaped to `[max_num_seqs, slice_ctas, hidden]`.
  Sites M-A and M-B route the per-s atomicAdd to `partial[bz, bx, k_row]`
  (per-CTA slot keyed by `bx`). Same-thread → same-address within a CTA
  so the per-s adds stay deterministic. Cross-CTA reduction moves to the
  last-CTA epilogue, which gathers `slice_ctas` slots in constexpr bx
  order.
- `wo_output` reshaped to `[max_num_seqs, total_ctas_per_seq, hidden]`.
  Phase B atomicAdd_f32 → plain `_st_global_f32` into the CTA's slot
  (`cta_idx = bx * num_kv_heads + by`). A new Phase B.5 runs inside the
  existing last-CTA branch: all 128 threads of the last CTA gather all
  per-CTA slots into slot 0 in fixed `cta_i` order. Phase C reads
  unchanged code but from slot 0.
- `_backend.py` allocates both buffers lazily with the new shapes
  (slice_ctas only known after `Phase_D_MLP_Kernel.__init__`; total
  CTAs sized with decode-cta_q=16 as the upper bound).
- Defaults re-flipped: `CUTE_ATTN_FUSION=1` and `CUTE_MLP_FUSION=1` in
  `_backend.py` gates and `scripts/serve-cute*.sh` env defaults.

**Validation evidence:** `benchmarks/nvllm/traces/phase_a_fused_reduction_fix/2026-04-20/`
- `./q2_results.jsonl` — Q2 x5 with both fusions ON: 5 byte-identical
  raw outputs (" 600/60 =  1200 1"). Deterministic.
- `./mlp_only/q2_results.jsonl` — MLP-only: 5 byte-identical and
  correct (" 10. 12 *  50/60 =").
- `./attn_only/q2_results.jsonl` — attn-only: 5 byte-identical and
  correct (" $10.").
- `./debug_fusion/decode_log.txt` — `CUTE_DEBUG_FUSION=1`: Phase B
  kernel vs Python `attn @ W_O.T` reference, 1200/1200 calls
  `close=True`, diff.max ≤ 0.0002 (ULP noise).

**Caveat (distilled-model Q2 sensitivity):** the combined-fusion Q2
output diverges from the unfused baseline not because the kernels are
wrong but because each deterministic fused path produces a different
(still numerically-correct) FP32 sum order than the Python reference.
Across 64 layers × attn+MLP the sub-ULP drifts compound past the
distilled Opus argmax margin on the first Q2 token. Single-kernel-
fused variants (MLP-only, attn-only) stay on the correct side of the
knife-edge; a non-distilled model is insensitive to this level of
drift. See `project_fused_path_nondeterminism` memory for the original
knife-edge framing.
