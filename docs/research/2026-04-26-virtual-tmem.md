# Virtual TMEM

### Simulating the B200 "Tensor Memory Conveyor Belt" on GB10 Using the Register File and Texture Units

> Research note. No implementation. The goal is to figure out which pieces of this idea survive contact with the SM120 hardware before we burn engineering time.

**Date:** 2026-04-26
**Target hardware:** NVIDIA DGX Spark / GB10 (SM120/SM121), 128 GB unified LPDDR5X
**Question:** can we approximate the register-pressure relief and dual-port feel of B200 TMEM by (a) treating the register file as a circular streaming buffer and (b) routing weight loads through the Texture Mapping Unit so they land in a separate issue port from activations?

---

## TL;DR

Three findings, in the order they matter:

1. **Half of the proposal is already what CUTLASS does.** "Register-file as circular streaming buffer + `ldmatrix` shape-load" is a relabel of the standard CUTLASS multistage mainloop: `cp.async` → SMEM → `ldmatrix` → MMA-frag registers, double/triple-buffered. This is not a novel mechanism on GB10 — it is the *baseline* mechanism. The interesting question is whether our existing CuTe kernels are already this disciplined; if not, the win is in upgrading them, not in inventing a new buffering scheme.

2. **The Texture-unit "parallel highway" claim does not survive unified L1.** Since Volta the L1 data cache and texture cache are the same physical SRAM. SM120 almost certainly inherits this (one microbench gates it). That means `tex.*` and `ldg.*` do not get separate paths to DRAM — they share the same cache and the same memory channels. What they *do* get is **separate issue slots**: the TEX unit and the LSU dispatch independently inside the SM. So the achievable win is an IPC/issue-throughput win when a kernel is issue-bound, not a bandwidth win when it is DRAM-bound. Most NVFP4 decode mainloops on GB10 are DRAM-bound. Frame the project as "buy back issue throughput" rather than "double the pipes."

3. **The genuinely interesting empirical question is register-pressure relief.** B200 TMEM's killer feature is that **MMA accumulators no longer occupy architectural registers** — accumulators live in TMEM, not RF, freeing ~32-64 registers per thread for fusion epilogues, scratch, and operand staging. We cannot replicate this on SM120 because there is no off-register accumulator silicon. But: if our current CuTe attention/MLP kernels are register-bound by the *B-operand staging* (not the accumulator), and if we can move B-staging through the TEX cache instead of through registers, we may be able to recover occupancy. This is an `ncu --set full` audit on existing kernels, not new code. It either moves the needle or it doesn't.

Everything below is the evidence and the experiments to run.

---

## Part I — Reality check on the proposal

### 1.1 What TMEM actually is, and why GB10 doesn't have it

TMEM (Tensor Memory) is a datacenter Blackwell (SM100, B100/B200) feature: ~256 KB per SM of dedicated SRAM that holds MMA accumulators outside the architectural register file. Two consequences:

- **Accumulator state stops competing for registers.** A `wgmma`-style instruction can keep a 128×128 FP32 accumulator alive across the K-loop without burning ~32-64 registers per thread. Fusion epilogues that previously spilled now fit.
- **Async accumulation.** TMEM has a path from MMA → TMEM that is decoupled from RF, so the math pipeline can run further ahead of the register file's read/write traffic.

GB10/SM120 is **consumer Blackwell**. Same brand family, different silicon. There is no TMEM on SM120 — the public CUDA programming guide for SM120 lists it under "not supported." Verified by the absence of `cute::TmemAllocator` from any SM120 codepath in CUTLASS 4.x and by the fact that `wgmma` async-MMA does not assemble for `sm_120a` in current `nvcc`. (Both are one-line greps in CUTLASS / a `cuobjdump` of any compiled `sm_120a` kernel; flagged as Microbench M0 below.)

So whatever we build, we cannot recover the *accumulator-off-register* property. The B200 conveyor belt has two belts and we can only build one.

### 1.2 What "register-file rotation + `ldmatrix`" actually buys you on SM120

The proposal frames it as a TMEM substitute. It is not. It is the standard CUTLASS mainloop:

```
cp.async.ca.shared.global  [smem_stage_n], [gmem_ptr]      # producer
ldmatrix.sync.aligned.m8n8.x4  {ra0..ra3}, [smem_stage_m]  # shape-shift into RF
mma.sync.aligned.m16n8k16  {rd...}, {ra...}, {rb...}, {rc...}  # math
```

…with N stages of `smem_stage` rotating, and the math/load instructions issued back-to-back so the compiler can interleave them. CUTLASS 3.x exposes this as `MainloopSm80...` (and SM120 inherits the SM80 multistage mainloop because it has neither TMEM nor TMA). `ldmatrix` is exactly the "shape-aware funnel" the proposal describes — it has been there since Volta.

The interesting question is therefore not "should we build this?" but **"do our existing CuTe kernels already do this, and if not, why not?"** Two paths:

- **If yes (CuTe attention/MLP are already proper multistage)**: there is no register-rotation win to chase. Move on.
- **If no (something is buffering B in registers across K, or only single-staging)**: fix it inside the existing kernel. Do not call it Virtual TMEM. Call it "upgrade CuTe mainloop to N-stage."

`ncu --section LaunchStats --section SourceCounters` on the existing fused MLP and β-coop kernels will tell us in an afternoon. Listed as M4 below.

### 1.3 What the texture path does and does not do on SM120

**True on SM120 (subject to whitepaper confirmation in M1):**

- TEX unit and LSU have **separate instruction issue slots** inside the SM. A warp can issue `tex.*` and another warp can issue `ldg.*` in the same cycle without contending for the same dispatch port. This is real and is the only honest "parallel highway" left after unified L1.
- Texture address calculation (the `tex2D`/`tex1Dfetch` indexing) executes on the TEX unit's dedicated address hardware, not on integer ALUs. That is a small but real ALU-issue savings.
- The TEX cache *replacement policy* is tuned for 2D spatial locality, which can produce different hit-rate behavior than the LSU's L1 path even when both share physical SRAM.

**Almost certainly false on SM120 (gated by M1):**

- "Texture units have a separate path to DRAM." They do not on any unified-L1 architecture (Volta+). They share the same L1/L2/HBM hierarchy. Same DRAM channels.
- "Texture cache leaves L1 for activations." On unified-L1, there is one cache. They share it. What you can do is bias *eviction pressure* toward weights via TEX residency hints, but you cannot give activations exclusive use of L1 because there is no separate L1 to give them.

**Probably false in our specific kernels:**

- "Reclaim ~64 registers from B-buffer." Only true if our current kernel buffers B in architectural registers across the K-loop. CUTLASS-style multistage *does not*; B passes through SMEM and only briefly through MMA-frag registers. If our CuTe kernels do something different — e.g., hold a full B-tile in registers because the SMEM workspace was tight — the win exists. If they are already streaming B through SMEM, there are no registers to reclaim. M4 is the diagnostic.

**Definitely false:**

- "TEX results bypass architectural registers via the operand collector." MMA inputs are register-addressed in PTX. TEX results land in registers. The operand collector is a microarchitectural staging buffer, not an alternative ABI. You cannot route around the register file from PTX.

### 1.4 NVFP4 weights through textures — the format mismatch tax

`cudaTextureObject_t` supports `float`, `float2`, `float4`, `int4`, `uchar4`, etc. There is no native FP4 texture format. To bind 4-bit packed NVFP4 weights as a texture, the realistic path is:

- Bind the weight buffer as `texture<uint4>` (16 bytes / 32 packed FP4 values per fetch).
- Unpack in the shader: extract per-element 4-bit codes, dequant via a small LUT or arithmetic shift+sign-extend, scale by the per-block FP8 scale.
- Multiply by `weight_global_scale` per the load-time inversion convention (see memory: `feedback_nvfp4_dequant_convention.md`).

This works. But the "free format conversion" property of textures (the linear filter / format-cast hardware) does not apply to FP4 because the cast hardware doesn't know FP4. The unpack runs on math units. So in throughput terms, TEX-loaded NVFP4 should look identical to LDG-loaded NVFP4 *plus* whatever issue-port parallelism the TEX unit gives us. M3 measures this.

---

## Part II — The math

### 2.1 Bandwidth budget on GB10

| Thing | Number | Source |
|---|---|---|
| LPDDR5x peak | 273 GB/s | NVIDIA DGX Spark product page |
| L1/TEX cache (per SM, when hot) | hundreds of GB/s aggregate | runtime; see M2 |
| L2 (per SM, when hot) | tens of GB/s aggregate | runtime; see M2 |
| TEX issue rate | 1 tex/cycle/warp scheduler (typical) | needs SM120 whitepaper / M1 |
| LSU issue rate | 1 ldg/cycle/warp scheduler | needs SM120 whitepaper / M1 |

For an NVFP4 27B decode mainloop at batch-1, the steady-state mainloop is **DRAM-bound on weight reads**: ~13.5 GB of FP4 weights touched per token at one full forward, against 273 GB/s of bandwidth, meaning the floor is ~50 ms/token of pure weight read just to feed the math. We measure ~30-40 ms/token end-to-end on the hot path, so we are already operating well inside cache reuse for some weights.

**The relevant question for this proposal:** is GEMM mainloop performance on GB10 issue-bound or DRAM-bound? Three cases:

- **DRAM-bound.** TEX path provides zero bandwidth. Project dies.
- **Issue-bound on LSU.** TEX path moves weight issue off the LSU; achievable IPC rises. Project lives.
- **Compute-bound (Tensor-Core saturated).** TEX path provides nothing; the math units are already pinned. Project dies for a different reason.

The honest answer is "we don't know which case dominates per-shape, per-kernel." That makes this a measurement project, not a design project. M2 + M5 settle it.

### 2.2 What the issue-port win could be worth

If LSU is issue-bound (warp schedulers stall waiting on LSU issue slots) and TEX is idle, redirecting half the loads to TEX yields a theoretical doubling of load issue rate, capped by:

- The MMA pipeline's appetite for new operands (often the actual ceiling).
- L1/TEX cache bandwidth (shared physical SRAM).
- Address-generation throughput (TEX address calc is free; LSU is not).

A realistic ceiling is ~10-20% mainloop speedup *in the kernels where this applies*. Not a step-change. Worth chasing only if M5 shows a current LSU stall regime in the kernels we care about, and only if M4 shows we have a register-pressure pain point that TEX residency can relieve.

### 2.3 Where TMEM-on-B200 wins that we cannot replicate

For full honesty: B200 TMEM's biggest win is not buffering, it is **accumulator-off-register**. A 128×128 FP32 accumulator in TMEM frees ~64 registers per thread. That changes occupancy from 1 CTA/SM to 2 CTA/SM on tile shapes that matter, which doubles latency hiding. We cannot get this on SM120. Any proposal that claims to "simulate TMEM" without addressing this is selling the smaller half of the win. The right framing is: *this proposal is about reclaiming RF from operand staging, not from accumulators.*

---

## Part III — Microbenches to run before any code

The whole proposal collapses into "what does GB10 silicon actually do, and what do our existing kernels actually do." Six microbenches, each scoped to a single afternoon. Run in order; later ones are gated by earlier ones.

### M0 — Confirm SM120 lacks TMEM and `wgmma`

- **What:** `nvcc -arch=sm_120a -dryrun` a tiny `wgmma.async` kernel; expect rejection. Check `cuobjdump --dump-sass` of an existing CuTe kernel for any `wgmma`/`tcgen05`-family instructions. Grep CUTLASS 4.x for `sm_120a` use of `cute::TmemAllocator`.
- **Why:** Pin the baseline. If something here surprises us, the rest of the proposal changes.
- **Cost:** 30 min.
- **Decision:** confirms the framing in §1.1.

### M1 — Unified L1/TEX confirmation on SM120

- **What:** Microkernel that loads a 1 MB buffer twice, once via `ldg` and once via `tex.1d`, with the second load timed for cache hit. Track `l1tex__t_sectors_pipe_lsu_*`, `l1tex__t_sectors_pipe_tex_*`, and `l1tex__data_bank_conflicts_pipe_*` in `ncu`. If the TEX-after-LDG access hits at L1 latency (not L2), they share a cache.
- **Why:** Gates the entire "parallel highway" framing. If TEX has a separate cache, the proposal gets more interesting. If it shares (almost certainly), we know we're chasing issue-port parallelism only.
- **Cost:** 1-2 hours.
- **Decision:** if shared, the bandwidth claim is dead and the doc updates accordingly.

### M2 — Issue-port parallelism, raw

- **What:** Pure-load microkernel, no math. Two variants:
  - `LDG_only`: every warp issues `ldg.E.128` to fetch 128B/cycle.
  - `LDG||TEX`: half the warps issue `ldg.E.128`, half issue `tex.1d.v4.u32` to fetch the same byte volume from a different region.
  Measure achieved bytes/cycle via `dram__bytes_read.sum.per_second` and `smsp__inst_executed_pipe_lsu` / `..pipe_tex`. Sweep occupancy.
- **Why:** Establishes the ceiling. If `LDG||TEX` does not exceed `LDG_only` on this hardware, the issue-port story is also dead.
- **Cost:** 2-3 hours.
- **Decision:** quantifies the absolute upper bound on what TEX redirection can buy.

### M3 — NVFP4 unpack via TEX vs LDG

- **What:** Bind a packed FP4 weight buffer as `texture<uint4>` and as a global memory `uint4*`. Same dequant code, same scale path, same output layout. Measure per-warp BF16 output throughput (elements/cycle) and registers/thread for each variant.
- **Why:** Establishes whether the format-mismatch tax (TEX has no native FP4 cast) eats the issue-parallelism win. Also surfaces any unexpected TEX-cache thrashing on the packed-uint4 access pattern.
- **Cost:** 3-4 hours.
- **Decision:** confirms NVFP4 is a viable payload for the TEX path.

### M4 — Register-pressure audit on existing CuTe kernels

- **What:** `ncu --set full --section LaunchStats --section SourceCounters` on the live CuTe MLP, β-coop, and β-lite kernels at the canonical Qwen3.5-27B NVFP4 shapes. Pull regs/thread, achieved occupancy, register spills, and the per-source-line register usage map. Identify whether B-operand staging is in registers, SMEM, or both. Identify whether accumulator pressure or operand-staging pressure dominates.
- **Why:** Tells us whether there are *any* registers to reclaim and *where* they are being spent. Without this, "Virtual TMEM" is shooting in the dark.
- **Cost:** 1 day (audit + write-up).
- **Decision:** if accumulators dominate (likely), TEX-redirect cannot help. If B-staging dominates, TEX-redirect might.

### M5 — Kernel-classification: issue-bound vs DRAM-bound

- **What:** Roofline analysis of the existing CuTe attention and MLP kernels at decode shapes. `ncu --section MemoryWorkloadAnalysis --section SchedulerStats --section ComputeWorkloadAnalysis`. Look at warp-stall reasons: `stall_long_scoreboard` (DRAM), `stall_short_scoreboard` (cache/SMEM), `stall_dispatch` (issue-bound), `stall_membar`, etc.
- **Why:** This is the *real* gating microbench. If our kernels are uniformly DRAM-bound, the project dies regardless of what M0-M4 say. If they are issue-bound or short-scoreboard-bound on the LSU, the project has somewhere to go.
- **Cost:** 1 day (audit across 3-4 kernels at a few shapes).
- **Decision:** the headline kill-or-confirm. Run this *first* in practice; M0-M4 are setup for interpreting M5.

### M6 — End-to-end synth GEMM swap (only if M2 + M4 + M5 are all green)

- **What:** Take a small CUTLASS-style mainloop at one of our hot decode shapes. Hold tile shape, accumulator strategy, swizzle, and stages constant. Swap *only* the B-load: `cp.async → ldmatrix` baseline vs `tex.1d.v4.u32 → unpack → MMA-frag`. Measure TFLOPS, registers/thread, occupancy, and ncu stall breakdown.
- **Why:** The first real comparison of "Virtual TMEM" against the standard mainloop on the same shape. Only worth running if the prior microbenches haven't already killed the idea.
- **Cost:** 2-3 days.
- **Decision:** the headline number for the doc.

**Stopping rules:**

- If M5 shows uniform DRAM-bound: stop. The doc updates to "GB10 NVFP4 decode is DRAM-bound; Virtual TMEM provides nothing measurable. Time better spent on weight compression / streaming (see [silly-streaming](2026-04-19-silly-streaming.md))."
- If M2 shows `LDG||TEX` does not exceed `LDG_only` on issue throughput: stop. Same conclusion as above.
- If M4 shows accumulator pressure (not B-staging) is the register pain point: stop. We cannot reclaim accumulator registers without TMEM silicon.
- If all three are green: M6, then design discussion.

---

## Part IV — If the microbenches are all green, where would this actually win?

Speculative; only worth populating if M5 shows issue-bound regions.

- **NVFP4 prefill GEMM at small batch:** prefill mainloops with chunked-prefill at small tiles can be issue-bound rather than DRAM-bound because the K-tile is small relative to math throughput. A TEX-redirect on B might recover 5-15%.
- **MoE expert dispatch at low active-expert count:** when only 4-8 experts fire and each expert's GEMM is small, the kernel can be issue-bound on weight loading. Same TEX-redirect logic applies.
- **Fused QKV-projection stage of attention:** if the projection's K-tile is short (head_dim is small), the mainloop is short and issue-overhead heavy. TEX-redirect on the QKV weight might help.

If M5 shows our hot decode kernels are *not* issue-bound, none of these saves the doc. Better to redirect to:

- Weight compression / nvCOMP HW-DE prefetch ([silly-streaming](2026-04-19-silly-streaming.md)).
- Multi-stage SMEM expansion in CuTe kernels (a real, independent win regardless of TEX).

---

## Part V — Results

> Empty until microbenches run. Each entry is a single ncu trace + one-line conclusion.

### M0 — TMEM/wgmma absence on SM120

**Run:** 2026-04-26, host CUDA 13.2 (V13.2.78, build cuda_13.2.r13.2/compiler.37668154_0), nvcc + ptxas. Probes at `/tmp/m0_tmem_probe/`.

**Verdict:** Confirmed. SM120 (as exposed by CUDA 13.2 toolchain) has no compiler-accessible path to either TMEM or wgmma. The proposal's framing in §1.1 stands.

**Evidence:**

1. **`tcgen05.alloc` on `sm_120a`** — REJECTED:

    ```
    ptxas tcgen05_probe.ptx, line 26;
    error : Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'
    error : Feature '.cta_group::1' not supported on .target 'sm_120a'
    ```

    Same kernel on `sm_100a`: accepted (cubin produced silently). Control confirms `tcgen05` is genuinely SM100 datacenter Blackwell only.

2. **`wgmma.fence.sync.aligned` on `sm_120a`** (PTX `.version 8.7`) — REJECTED:

    ```
    error : Instruction 'wgmma.fence' not supported on .target 'sm_120a'
    error : Instruction 'wgmma.fence' cannot be compiled for architecture 'sm_120a'
    ```

    Same instruction on `sm_100a`: also REJECTED — wgmma is Hopper-only and SM100/Blackwell-datacenter replaced it with the `tcgen05.mma` family. So neither Blackwell variant exposes `wgmma`. On `sm_90a` (Hopper control): accepted silently.

3. **CUTLASS DSL family separation.** `cutlass/base_dsl/arch.py:117-138` — `is_family_of()` returns True only when `major.minor` match and the suffix is `a` or `f`. SM120 (major=12) and SM100 (major=10) are different families; SM100-only features do not propagate to SM120. SM100 ≠ SM120 in the DSL's own taxonomy, matching the ptxas evidence.

4. **The interesting curveball.** `cutlass/cute/arch/tmem.py:28-37` defines:

    ```python
    TMEM_MAX_ALLOC_COLUMNS_MAP = {"sm_120": 512, "sm_100": 512}
    TMEM_MIN_ALLOC_COLUMNS_MAP = {"sm_120": 32,  "sm_100": 32}
    ```

    The DSL has aspirational entries for `sm_120` TMEM column counts. The CUDA 13.2 toolchain *does not honor them* (per evidence #1). Three plausible interpretations, in decreasing order of plausibility:

    - DSL author included a future/anticipatory entry; silicon does not have TMEM.
    - SM121 (the GB10 variant) or `sm_120f` may expose TMEM in a future toolchain or stepping; CUDA 13.2 does not yet compile for it.
    - DSL bug / copy-paste leftover.

    **Not load-bearing for the proposal.** The toolchain we build with rejects `tcgen05.alloc` today; if NVIDIA enables it in CUDA 14.x and the silicon turns out to have TMEM, that *kills the proposal* in a different and better way (we would just use real TMEM). Either outcome is fine. Flagged for re-check on next CUDA toolkit bump.

5. **Flashinfer cubins under `~/.cache/uv/.../flashinfer_cubin/`** target `sm100f` (e.g. `Gemm_Bfloat16_E2m1E2m1_Fp32_..._sm100f.cubin`), never `sm120`. Indirect supporting evidence that TMEM-class GEMMs are an SM100-only build target in current upstream tooling.

**Cost:** ~30 min wall clock (probe-write + 6 ptxas runs + arch.py / tmem.py read). Probes left at `/tmp/m0_tmem_probe/` for re-run after any CUDA toolkit upgrade.

**Implication for the rest of the proposal:** §1.1 framing holds — SM120 silicon (per CUDA 13.2) cannot replicate B200 TMEM's accumulator-off-register property at the compiler level. The Virtual-TMEM proposal must therefore narrow to operand-staging RF relief (M4) plus issue-port parallelism (M2/M5), with no claim to recovering accumulator pressure.

### M1 — Unified L1/TEX cache

*[to fill in]*

### M2 — Issue-port parallelism

*[to fill in]*

### M3 — NVFP4 unpack via TEX

*[to fill in]*

### M4 — Register-pressure audit

*[to fill in]*

### M5 — Kernel classification (issue vs DRAM bound)

*[to fill in]*

### M6 — End-to-end GEMM swap

*[to fill in]*

### Verdict

*[to fill in: alive / pivot / dead, with one paragraph summary]*

---

## Sources to verify before relying on any specific number

- **NVIDIA SM120 architecture whitepaper.** Confirms unified L1/TEX, TEX issue rate, register file size per SM, occupancy ceilings. NVIDIA has not published a public deep-dive for SM120 as of this writing; expect to cross-reference the SM89 (Ada) whitepaper plus runtime probes. Listed as M0/M1.
- **CUTLASS 4.x SM120 mainloop.** Read `include/cutlass/gemm/collective/sm90_mma_tma_gmma_*.hpp` and `sm80_mma_multistage.hpp` to confirm what an "already-good" mainloop looks like, and audit our CuTe kernels against it. Pin a commit when citing.
- **CUDA Programming Guide §B (texture functions) and §K (architectural specifics).** For `cudaTextureObject_t` format support and TEX issue semantics.
- **`cuobjdump --dump-sass`** of existing CuTe MLP/β-coop/β-lite kernels at a current commit. Check for `wgmma`/`tcgen05` (expected absent), confirm `ldmatrix`/`mma.sync.aligned` baseline.
- **Closing thought.** The honest version of "Virtual TMEM on GB10" is "audit our existing CuTe mainloops to confirm they are CUTLASS-grade multistage, then see if a TEX-redirect on B-operand loads buys back any issue throughput in the kernels that turn out to be issue-bound rather than DRAM-bound." That is a much smaller and more sober project than the framing suggests, but it is one with a clean kill criterion and a small budget. Worth the afternoon to find out.
