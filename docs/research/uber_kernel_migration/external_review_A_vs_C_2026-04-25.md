# β-lite kill (Option A) vs collapse (Option C) — external evidence review

**Date:** 2026-04-25
**Branch:** `fix/phase-d-mlp-decode-tile-preset` (HEAD)
**Author:** research subagent (read-only)
**Scope:** Decode-path full-attention layer for Qwen3.5-27B on SM120 (DGX Spark). Prefill out of scope.

> **2026-04-25 correction (post-review):** report originally cited "12-SM chip" for GB10. Actual count verified via `torch.cuda.get_device_properties`: **48 SMs, 102400 B SMEM/SM**. Conclusions in this report survive the correction because the binding constraint for cooperative β-coop at large batch is **SMEM-per-CTA**, not SM count: at 102400/45568 ≈ 2 CTAs/SM × 48 SMs = resident_cap=96, num_seqs=8 (512 CTAs needed) requires shrinking SMEM/CTA to ~9 KB — a 5× reduction not achievable on SM120. Realistic shrink to 16-20 KB Phase 1 covers num_seqs=3-4 cooperatively; long-tail batch still needs the non-cooperative branch. Recommendation (C now, A as follow-up) holds.

---

## TL;DR

External evidence strongly favors **Option C — collapse β-lite into a launch-config knob on the same kernel object** as a near-term step, with **Option A as a longer-term goal contingent on real SMEM-shrink work**. Production fused-decode kernels in FlashAttention-3, FlashInfer, TRT-LLM XQA, and CUTLASS all share a common pattern: a single source-of-truth kernel body whose dispatch chooses between a "small-batch / split-KV / persistent" mode and a "large-batch / data-parallel" mode at launch time. None of them ships a single launch config that handles every batch regime — each has at least two — but all invest in keeping the math in one place. Killing β-lite without first proving SMEM headroom is in conflict with the FlashInfer / FA3 precedent.

---

## How other kernels handle this

### FlashAttention-3 (Tri Dao et al.)

FA3 ships **one templated forward kernel** but **two scheduler classes selected at compile time**, plus an optional **separate combine kernel** for split-KV reductions. The persistent-vs-single-tile choice is a `static constexpr bool UsePersistentScheduler` heuristic embedded in the launch template:

> `static constexpr bool UsePersistentScheduler = Arch >= 90 ? !(Split && !Varlen) : ((Is_causal && !Varlen) || (Varlen && Split));`
> — `hopper/flash_fwd_launch_template.h` (commit `27f501dbe011f4371bff938fe7e09311ab3002fa`)

The schedulers themselves live in `hopper/tile_scheduler.hpp` (same commit) as three classes: `SingleTileScheduler`, `StaticPersistentTileScheduler`, `DynamicPersistentTileScheduler`. The first uses a grid of `{num_blocks, num_head*num_splits, num_batch}`; the latter two use a grid of `{num_sm}`. **Different launch grids, same kernel body.** Split-KV decode adds a *second* kernel — `FlashAttnFwdCombine` in `hopper/flash_fwd_combine_kernel.h` — that runs only when `num_splits > 1`. The split count is per-batch via `num_splits_dynamic_ptr`, and the small-batch threshold is documented as `L_K ≤ 512` ([arxiv 2604.00028](https://arxiv.org/html/2604.00028)).

**Takeaway:** Even FA3 — the de-facto reference — does not have one launch config. It has a compile-time scheduler-class swap plus an optional combine kernel. Same-math-different-launch is exactly the pattern.

### FlashInfer (Zihao Ye / Catalyst)

FlashInfer's `BatchDecodeWithPagedKVCache` is **one templated kernel** (`BatchDecodeWithPagedKVCacheKernel`, lines 981-985 of `include/flashinfer/attention/decode.cuh`, [v0.2.4](https://github.com/flashinfer-ai/flashinfer/blob/v0.2.4/include/flashinfer/attention/decode.cuh)) with a runtime `params.partition_kv` flag set by the host dispatcher (`BatchDecodeWithPagedKVCacheDispatched`, lines 1013-1088). When `tmp_v == nullptr`, partition-KV is off and the grid is `{padded_batch_size, num_kv_heads}`; otherwise partition-KV is on and the work is split across `num_chunks` along the KV axis. **No cooperative launch** — `cudaLaunchKernel` only.

vLLM's V1 FlashInfer backend wraps this with **separate decode and prefill wrapper objects** (`BatchDecodeWithPagedKVCacheWrapper` line 597, `BatchPrefillWithPagedKVCacheWrapper` line 560 of `vllm/v1/attention/backends/flashinfer.py` at [vllm v0.11.0](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/attention/backends/flashinfer.py)) and **separate cudagraph-specific decode wrappers selected per padded batch size** (lines 846-858).

**Takeaway:** FlashInfer accepts *two* host-side wrappers and a *runtime branch* inside one kernel rather than one universal kernel. Decode and prefill are not fused; small-batch and large-batch decode are not fused either at the wrapper level.

### TRT-LLM XQA decoder

TRT-LLM's XQA path has an explicit **multi-block / split-KV mode** that activates only when batch is small enough that the full batch grid would underutilize the GPU. The threshold is occupancy-based, not a hardcoded batch size:

> `float(block_count) * kEnableMinBlockFactor >= float(mRunner->mMultiProcessorCount)`
> where `block_count = num_kv_heads * batch_size * multi_block_count`
> — `decoderXQAImplJIT.cpp` lines ~105-115, 280-282 ([v0.13.0](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.13.0/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/decoderXQAImplJIT.cpp))

When the predicate fires, multi-block runs an additional reduction kernel after the main XQA kernel.

**Takeaway:** Same shape as FA3 — **occupancy-driven** dispatch into a split-KV path with a separate reduction. Both paths share JIT-generated kernel bodies with different launch configs.

### CUTLASS (NVIDIA)

CUTLASS's "cooperative" warp-specialized GEMM is **not** `cudaLaunchCooperativeKernel`; "cooperative" refers to **two consumer warp groups inside one CTA splitting one output tile** — see `sm90_gemm_tma_warpspecialized_cooperative.hpp` at [v3.6.0](https://github.com/NVIDIA/cutlass/blob/v3.6.0/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp). Persistent kernels exist (grid sized to SM count) but again use standard launches. Stream-K is a *scheduling* strategy, not a launch mode. The CUTLASS tutorial ([Colfax](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)) explicitly documents that Stream-K "only rarely performs well relative to the other two schedulers (typically when there are so few tiles that the GPU would be severely underutilized without splitting)" — i.e., the small-batch regime is exactly where persistent + split shines, and the large-batch regime is exactly where naive data-parallel wins.

**Takeaway:** CUTLASS's persistent kernels are **the same kernel body with a tile-scheduler swap**, not a different kernel. The persistent regime targets *small problem* — equivalent to small batch in our world.

### Mamba-2 SSD (Tri Dao / Albert Gu)

Mamba-2's SSD combined Triton kernel uses **one kernel** with a 3D launch grid `(triton.cdiv(chunk_size, BLOCK_M) * triton.cdiv(headdim, BLOCK_N), batch * nchunks, nheads)` — `mamba_ssm/ops/triton/ssd_combined.py` line 563-565 ([v2.2.4](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py)). **No cooperative launch**, no separate paths for batch size — Triton's autotuner picks tile sizes per shape. This is a counter-example: when SMEM budget is comfortable, one kernel suffices. Mamba-2's SSD is also not as SMEM-constrained as FP4 attention + 17K-intermediate MLP fused.

### vLLM upstream V1 attention

V1 has **separate backends** for FlashInfer, FlashAttention, FA3, and Triton, each with its own decode/prefill split. The Triton backend ([deep dive blog 2026-03-04](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)) documents that even within one backend, decode and prefill use different kernels and CUDA-graph capture is decode-only. CLI flags `--decode-attention-backend` and `--prefill-attention-backend` exist precisely because operators want to mix-and-match per phase.

**Takeaway:** vLLM upstream is even more split than FA3 / FlashInfer — multiple *backends*, not just multiple *paths inside one backend*.

---

## Cooperative-launch state of the art

`cudaLaunchCooperativeKernel` is rarely used in production decode kernels. The grid-stride / persistent-CTA pattern dominates instead, because:

1. Cooperative launch caps the grid at the resident occupancy (the same SMEM constraint that bites β-coop today).
2. Persistent CTAs achieve cross-CTA coordination via *atomics on global counters* (FA3's `DynamicPersistentTileScheduler`, CUTLASS's Stream-K) rather than `__grid_sync()`.
3. Cooperative launch interacts badly with CUDA graphs in some configurations (vLLM upstream warns about graph capture vs dynamic-grid attention).

Where cooperative launch *does* appear: DeepSeek's MLA, some research SSM kernels, and our own β-coop. The threshold pattern across all of them: cooperative **iff** the entire problem fits the resident cap, otherwise fall back to a non-cooperative grid-stride or split-KV variant. This is exactly what β-coop / β-lite already implement — the question is whether to keep both or unify.

For SM89/SM90/SM120 the threshold is the same in spirit but the SMEM budget differs: SM90 has up to 228 KB opt-in SMEM/CTA, SM120 caps at ~100 KB opt-in (49152 default). β-coop's 45568 B is already near the default ceiling.

---

## Option A pros (with evidence)

- **One source of truth → fewer correctness divergences.** β-coop and β-lite have already accumulated subtle differences (e.g., Phase 0 input-LN baking is dead in β-coop's consume path but live in β-lite). Single-kernel projects like Mamba-2 SSD report fewer math-bug bisections (Phase F.1 bisection on this repo found one such drift across paths).
- **CUDA graph capture is simpler with one launch shape per layer.** vLLM upstream cudagraph wrappers exist *because* multiple paths complicate capture. Cf. `feedback_cudagraph_patterns` and the Triton backend deep-dive's note that "attention kernels present challenges because their launch grids often depend on batch size."
- **Removes the 64×num_seqs ≤ resident_cap cliff.** Today, num_seqs=2 may or may not fit cooperatively depending on resident_cap (probed at startup). Killing β-lite forces the SMEM diet that makes the cliff vanish for the realistic batch range.
- **β-lite has a known +25 ms perf regression (Phase D MLP) at num_seqs=1** per `project_fused_path_perf_collapse` (memory note). Removing it eliminates a path that today is *slower* than the cooperative path on the workloads it covers.

## Option A cons (with evidence)

- **No precedent ships a single decode launch config covering all batch sizes** when the kernel is SMEM-bound. FA3, FlashInfer, TRT-LLM XQA, and CUTLASS Stream-K all keep at least two paths (single-tile vs persistent, partition-KV on/off, multi-block on/off). Our hardware is *more* SMEM-constrained than H100, not less.
- **SMEM shrink work is non-trivial.** Phase 1 attn at 45568 B is dominated by Q/K/V SMEM tiles and grid-barrier scratch. Packed-FP8 K/V ping-pong needs an asynchronous double-buffer plus dequant on read — production precedent exists (FA3's K/V SMEM pipelining at Hopper), but on SM120 we don't have TMA, so the pipelining is a hand-rolled `cp.async`. Risk of multi-week stall.
- **Cooperative-launch SMEM cliff returns at long context.** At max_model_len=131072 with 96-block resident cap, num_seqs=1 already needs 64 CTAs. If a future model raises CTAs/seq, the cliff is back — and β-lite would be the *natural* mitigation. Removing it forecloses that escape hatch.
- **SM120 opt-in SMEM ceiling is ~100 KB.** Even after a shrink, fitting num_seqs=8 cooperatively at 64 CTAs/seq = 512 CTAs needs a resident cap of 512, which on a 12-SM chip means 42 CTAs/SM — likely impossible regardless of SMEM. So β-coop alone cannot serve num_seqs=8 cooperatively today; killing β-lite without **also** adding a non-cooperative β-coop launch mode (= Option C) leaves no path for large batches.

## Option C pros (with evidence)

- **Direct precedent in every production decode kernel surveyed.** FA3's `UsePersistentScheduler` boolean, FlashInfer's `params.partition_kv`, TRT-LLM's `multi_block_mode`, CUTLASS Stream-K vs data-parallel — all are "same kernel body, different launch config / runtime flag." The C structure is the *industry default*.
- **Lowest engineering risk.** The β-coop kernel body already exists and works. C is a host-side dispatch refactor: pick `cooperative=True` for small batch, `cooperative=False` + appropriate grid-stride for large. The CuTe DSL supports `cooperative=` as a launch kwarg natively (`reference_cute_cooperative_launch` memory note).
- **Preserves the β-lite escape hatch** without keeping β-lite's distinct math. If the cooperative path hits an SMEM/occupancy cliff, the same kernel body falls through to non-cooperative — no second source tree to maintain.
- **Compatible with future Option A.** The collapse is a strict prerequisite for safely killing one path: once both are the same body, comparison and shrink work has a single target.
- **CUDA-graph-friendly.** One kernel object, two launch shapes — graph capture handles this; what it dislikes is *runtime kernel selection by name*. Compare vLLM cudagraph wrappers' per-batch-size approach.

## Option C cons (with evidence)

- **Doesn't unify behavior, only code.** Two launch shapes still mean two SMEM budgets and two grid-barrier requirements. Phase 2 grid_sync semantics differ between cooperative and non-cooperative; the kernel must branch on that — tested in FA3's persistent vs single-tile schedulers (separate scheduler classes, not just flags).
- **Tile-scheduler complexity moves into the kernel.** FA3 absorbs this via three scheduler classes; CUTLASS via a tile-scheduler abstraction. We'd need a CuTe DSL equivalent — and CuTe DSL has known limits around constexpr loops and multi-mode kernel bodies (`feedback_constexpr_oom`). Risk of compile-time blowup if the if-branches are large.
- **Maintenance two-headedness.** Even with one source, adding a Phase E feature (e.g., Phase 0 prologue for QKV fusion) requires testing both launch modes. β-lite today gets less testing than β-coop; that asymmetry persists under C.
- **Does not address the existing β-lite math/perf bugs** — those need fixing regardless. Without a fix, C ships a known-broken second mode.

---

## SMEM shrink feasibility for Option A

Phase 1 SMEM at 45568 B on SM120 with hidden=5120, intermediate=17408, head_dim=256, FP8 KV:

| Bucket | Current bytes | Shrink potential |
|---|---|---|
| Q SMEM (BF16, 24 heads × 256) | 12288 | Halve via FP8 storage + dequant on read (~6144) |
| K SMEM (FP8, paged tile) | ~8192 | Already FP8; ping-pong with `cp.async` halves resident at cost of one extra stage |
| V SMEM (FP8, paged tile) | ~8192 | Same as K |
| Sync / m-d / output partials | ~16896 | Hard to shrink — grid-barrier counter + log-sum-exp state |

**Production precedent for packed-FP8 ping-pong:** FA3 on Hopper (`hopper/flash_fwd_kernel_sm90.h`, same commit) uses `cp.async`-driven K/V double-buffering with dequant in registers — this is the standard idiom. SM120 lacks TMA but supports `cp.async.bulk` and `cp.async.cg/ca`; the same idiom ports with more boilerplate.

**Realistic budget after shrink:** ~32-36 KB Phase 1, opening room for ~2x resident_cap. That gets β-coop from num_seqs=1 to num_seqs=2-3 cooperatively. **It does NOT get to num_seqs=8 cooperatively** — at 64 CTAs/seq × 8 = 512 CTAs total, you'd need impossible occupancy on a 12-SM chip. Conclusion: SMEM shrink alone cannot make β-coop universal. Option A *also* needs a non-cooperative mode of β-coop for large batches, i.e., it presupposes Option C as a substep.

The MLP Phase 3 SMEM (smem_x = 5120×4 = 20480 B, plus reduction/intermediate) is independent from Phase 1 because they're temporally separated by the grid barrier. The `max(phase_01, phase_3)` in `_smem_bytes_phase_coop_full` (phase_e_kernel.py:278-281) means MLP is not currently the bottleneck — Phase 1 is. A Phase 1 shrink is the highest-leverage move; Phase 3 shrink is a follow-up.

---

## Recommendation

**Adopt Option C now, with Option A as a tracked follow-up gated on SMEM-shrink evidence.**

The collapse is the directly precedented move:

1. **Every production decode kernel surveyed** (FA3, FlashInfer, TRT-LLM XQA, CUTLASS persistent) ships at least two launch configurations sharing one kernel body. None ships a single-config decode kernel under SMEM pressure comparable to ours.
2. **Killing β-lite outright requires SMEM shrink work that has not happened**, and even after a realistic shrink the cooperative path cannot cover num_seqs=8 on a 12-SM/100KB-SMEM chip. So Option A in its strong form ("β-coop for ALL batch sizes") is mathematically unattainable without *also* adding a non-cooperative β-coop launch mode — which **is** Option C.
3. **Option C eliminates the β-lite math divergence** (Phase E β-coop math fix from `project_phase_e_beta_math_bug` only needs one place to land) without committing to a multi-week SMEM diet.
4. **Option A becomes a clean follow-up** once C is in place: after collapse, the question "do we still need the non-cooperative mode" becomes a measurable per-batch-size benchmark, not a structural refactor.

**Risk acknowledgment.** This recommendation rests on the assumption that adding `cooperative=False` as a launch path through the existing β-coop body is feasible in CuTe DSL without triggering the constexpr-loop blowup pattern we hit before. If it turns out the kernel body cannot host a Phase 2 grid-barrier branch cleanly (cooperative grid_sync vs non-cooperative atomic-counter spin-wait), C devolves into "two near-identical kernel bodies in one file," which retains most of A's downsides. A 1-day spike to confirm the grid-barrier branch compiles cleanly under both modes is the highest-value next step before committing to either A or C.

**Open uncertainty.** I have not measured the per-num_seqs latency of β-coop (cooperative) vs a hypothetical β-coop (non-cooperative). The Phase E shipped trace (`benchmarks/nvllm/traces/phase_e/2026-04-23-initial/`) covers the cooperative path only. If non-cooperative β-coop turns out to be slower than current β-lite at num_seqs ≥ 4, Option A's "kill β-lite" framing is correct — but the path to it still routes through Option C as a transitional state.

---

## Pinned source references

- FlashAttention-3 launch template — [hopper/flash_fwd_launch_template.h @ 27f501dbe](https://github.com/Dao-AILab/flash-attention/blob/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_fwd_launch_template.h)
- FA3 tile scheduler — [hopper/tile_scheduler.hpp @ 27f501dbe](https://github.com/Dao-AILab/flash-attention/blob/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/tile_scheduler.hpp)
- FA3 combine kernel — [hopper/flash_fwd_combine_kernel.h @ 27f501dbe](https://github.com/Dao-AILab/flash-attention/blob/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_fwd_combine_kernel.h)
- FlashInfer decode — [include/flashinfer/attention/decode.cuh @ v0.2.4](https://github.com/flashinfer-ai/flashinfer/blob/v0.2.4/include/flashinfer/attention/decode.cuh)
- TRT-LLM XQA dispatch — [decoderXQAImplJIT.cpp @ v0.13.0](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.13.0/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/decoderXQAImplJIT.cpp)
- CUTLASS sm90 cooperative GEMM — [include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp @ v3.6.0](https://github.com/NVIDIA/cutlass/blob/v3.6.0/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp)
- Mamba-2 SSD — [mamba_ssm/ops/triton/ssd_combined.py @ v2.2.4](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py)
- vLLM V1 FlashInfer backend — [vllm/v1/attention/backends/flashinfer.py @ v0.11.0](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/attention/backends/flashinfer.py)
- Colfax persistent + Stream-K tutorial — [research.colfax-intl.com](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- Sequence-aware split heuristic for FA3 low-head-count decode — [arxiv 2604.00028](https://arxiv.org/html/2604.00028)
- Flash-Decoding pytorch blog — [pytorch.org/blog/flash-decoding](https://pytorch.org/blog/flash-decoding/)

## Local code anchors (HEAD of `fix/phase-d-mlp-decode-tile-preset`)

- β-coop dispatch predicate — `vllm/v1/attention/backends/cute_paged/_backend.py:1131-1145`
- β-coop SMEM totals — `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:221-281`
- β-coop run entry — `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2685` (`run_beta_coop_full`)
- β-lite MLP kernel — `vllm/v1/attention/backends/cute_paged/mlp_kernel.py` (`Phase_D_MLP_Kernel`)
