# CuTe DSL Kernel Replacement — Design Spec (v2, post-audit)

> **Big picture:** This is a learning project — building a real SM120 attention kernel
> from scratch with an AI agent. Attention is ~4% of GPU time for this model (nsys
> confirmed), so the throughput gain is modest. The value is owning the full kernel
> stack, understanding CuTe DSL + inline PTX development on GB10, and building the
> skills to tackle the real bottlenecks (FP4 GEMM tuning, MTP spec decode) next.
> Have fun.

**Date:** 2026-04-11
**Status:** Design — awaiting implementation plan
**Parent spec:** `docs/superpowers/specs/2026-04-10-cute-paged-attention-design.md`
**Target hardware:** NVIDIA DGX Spark (GB10, SM120/SM121), 128 GB unified memory
**Model:** `natfii/Qwen3.5-27B-NVFP4-Opus-GB10` (GQA=6: 24 Q heads / 4 KV heads, head_dim=256)

---

## 1. Motivation

The CuTe paged attention backend is validated end-to-end with a PyTorch prototype:
coherent output across 28 layers, prefill + decode, GSM8K sanity confirmed. The
backend interface (`_backend.py`), disk cache (`disk_cache.py`), warmup stub
(`warmup.py`), and attribution docs are all shipped.

The PyTorch prototype in `kernel.py` uses `torch.einsum` and standard tensor ops to
simulate what the real kernel will do with hardware MMA instructions. It proves the
algorithm but runs at PyTorch dispatch overhead — not at SM120 native throughput.

This spec covers replacing the PyTorch internals of `kernel.py` with CuTe
JIT-compiled PTX kernels using BF16 m16n8k16 MMA instructions for both QK and PV
passes, following the b12x reference (lukealonso/b12x@c469c66) default FP8 KV path
and CUTLASS PR #3030 SM120 findings.

## 2. Scope

### In scope

- Replace PyTorch ops in `kernel.py` with CuTe JIT-compiled PTX kernel
- Two compiled variants: decode (cta_q=16) and prefill (cta_q=64)
- Inline PTX utility module (~15 wrappers for MMA, loads, dequant, softmax)
- Wire `warmup.py` to pre-compile both variants with dummy tensors
- Docker build with kernel compilation (lint first, build in tmux)
- GSM8K sanity gate
- ShareGPT throughput bench + nsys traces
- CUDA graph verification
- Fix b12x insights doc GQA ratio (6 → 4)

### Out of scope

- Changes to `_backend.py` (interface is stable)
- Changes to `disk_cache.py` (ported and working)
- New backend registration or platform changes
- Split-KV reduction kernel (deferred — single CTA handles full sequence for now)
- head_dim≠256 support (model uses head_dim=256; other dims not tested)
- Path A FP8 QK MMA (deferred optimization — start with Path B BF16 QK)

## 3. Kernel Architecture

### 3.1 Approach: CuTe JIT-Compiled PTX Kernel

One compiled kernel function per config (decode vs prefill). All three passes — QK
MMA, online softmax, PV MMA — execute within a single CTA. No inter-pass kernel
launch overhead.

The `@cute.kernel` decorator provides NVRTC compilation, argument marshalling, and
launch config. The compute core is `@cute.jit` helper functions wrapping inline PTX
for MMA instructions, memory loads, dequantization, and softmax primitives. This is
the same architecture b12x uses — not the high-level `cute.gemm()` / `make_tiled_mma()`
path, which does not support the interleaved QK→softmax→PV attention pattern.

The PyTorch prototype is moved to `tests/nvllm/attention/reference.py` (not kept in
the production file). Dead code (`qk_pass`, `pv_pass`, `online_softmax` helper
functions that were never called by the forward pass) is deleted.

### 3.2 MMA Path Decision: Path B (K Dequant to BF16)

**Chosen:** K FP8→BF16 dequant via `fp8x4_e4m3_to_bfloat2x2`, then BF16
`mma.sync.aligned.m16n8k16` for QK. This is b12x's default FP8 KV path.

**Rejected (for now):** Path A — Q BF16→FP8 cast, native FP8 `m16n8k32` MMA.
2x throughput but lossy Q quantization (~6.25% per-element error). Can be
revisited as an optimization if Path B quality is confirmed and more throughput
is needed.

**Rationale:** Path B preserves Q precision (no cast), matches the proven b12x
default, and requires the same inline PTX effort as Path A. Both K and V are
dequanted from FP8→BF16 using the same `fp8x4_e4m3_to_bfloat2x2` primitive.

### 3.3 Execution Flow Per CTA

```
1. Load Q tile via CpAsync (cp.async) into SMEM — row-major with
   128-byte permutation for bank-conflict avoidance
2. For each KV page in page_table[seq_idx]:
   a. Load K page via CpAsync (cp.async) into SMEM
   b. ldmatrix K from SMEM to registers
   c. Dequant K: FP8 → BF16 via fp8x4_e4m3_to_bfloat2x2
   d. QK MMA: BF16 m16n8k16 (Q BF16 native, K dequanted BF16)
      - Two m16n8k16 instructions per m16n16k16 output tile (b12x pattern)
      - num_mma_d_qk = head_dim / 16 = 16 iterations along K dimension
      - Output: FP32 scores, k_scale applied post-MMA in FP32
   e. Online softmax update (registers only):
      - Row max via warp shuffle (__shfl_xor_sync)
      - exp2-based rescaling (ex2.approx.ftz.f32)
      - Rescale running O accumulator when max increases
   f. Load V page via CpAsync into SMEM
   g. ldmatrix V from SMEM to registers
   h. Dequant V: FP8 → BF16 via fp8x4_e4m3_to_bfloat2x2
   i. PV MMA: BF16 m16n8k16
      - v_scale applied to P in FP32 before BF16 cast (b12x pattern)
      - P cast FP32 → BF16 for MMA operand
      - num_mma_d_vo = head_dim / 16 = 16 iterations along V dimension
      - Output: FP32 accumulator
3. Cross-warp reduction (decode only):
   - warps_kv=4 each produce partial O, m, d
   - Reduce via SMEM cta_sync buffers (b12x pattern)
4. Final softmax normalization: O /= row_sum
5. Write O (cast FP32 → BF16) to global memory
```

### 3.4 Tile Configurations

| Config | CTA tile Q | CTA tile KV | Stages | Threads | Warps Q | Warps KV | SMEM (est.) |
|--------|-----------|-------------|--------|---------|---------|----------|-------------|
| Decode | 16 | 64 | 1 | 128 | 1 | 4 | ~20 KB |
| Prefill | 64 | 64 | 1 | 128 | 4 | 1 | ~49 KB |

SM121 SMEM budget: 101 KB. Both configs fit with massive headroom (81 KB and 52 KB
respectively). Starting with num_stages=1 per b12x's finding that multi-staging does
not help with FP8 KV's already-halved memory footprint. Can add stages later if nsys
shows memory-bound stalls.

**Warp decomposition:**
- **Decode (warps_q=1, warps_kv=4):** Each warp processes different KV tile chunks
  in parallel. Requires cross-warp reduction of partial softmax states (O, m, d) via
  SMEM sync buffers. This is the b12x decode pattern.
- **Prefill (warps_q=4, warps_kv=1):** Each warp handles 16 Q rows independently,
  all sharing the same KV tile from SMEM. No cross-warp reduction needed (each warp
  has independent output rows).

### 3.5 SMEM Layout

Using `cute.struct.MemRange` / `cute.struct.Align` pattern from b12x (not ad-hoc
SmemAllocator):

- **Q buffer:** `(cta_tile_q, head_dim)` BF16, row-major with `_permuted_offset_128b`
  for bank-conflict-free `ldmatrix` access. No swizzle needed (CpAsync, not TMA).
- **K buffer:** `(cta_tile_kv, head_dim)` uint8 (FP8), row-major for CpAsync.
  Single stage (no multi-buffering).
- **V buffer:** `(cta_tile_kv, head_dim)` uint8 (FP8), row-major for CpAsync.
  Single stage.
- **Descale:** k_scale, v_scale as FP32 scalars (loaded once per CTA).
- **cta_sync (decode only):** `(warps_kv, cta_tile_q, head_dim)` FP32 for
  cross-warp partial O reduction, plus `(warps_kv, cta_tile_q)` FP32 for
  partial m and d. ~37 KB additional for decode.

**Decode SMEM total:** ~20 KB (Q+K+V+descale) + ~37 KB (cta_sync) = ~57 KB. Fits.
**Prefill SMEM total:** ~49 KB (Q+K+V+descale), no cta_sync. Fits.

### 3.6 MMA Instruction Mapping

| Pass | A dtype | B dtype | Accumulator | PTX Instruction | Wrapper |
|------|---------|---------|-------------|-----------------|---------|
| QK | BF16 | BF16 | FP32 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `bf16_mma_m16n16k16_f32` (issues 2x m16n8k16) |
| PV | BF16 | BF16 | FP32 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | same wrapper |

SM120 constraints: no tcgen05 MMA, no TMEM accumulators, registers only. Both QK
and PV use the same BF16 MMA instruction. The `bf16_mma_m16n16k16_f32` wrapper from
b12x issues two m16n8k16 PTX instructions to cover a 16x16 output tile per MMA group.

MMA iteration counts for head_dim=256:
- `num_mma_d_qk = 256 / 16 = 16` (K dimension iterations for QK)
- `num_mma_d_vo = 256 / 16 = 16` (V dimension iterations for PV)

### 3.7 FP8 Dequantization

Following b12x default FP8 KV path (lukealonso/b12x@c469c66):

- **K FP8 → BF16:** Inline during fragment load via `fp8x4_e4m3_to_bfloat2x2` +
  `frag_layout_swizzle_16b_to_8b` for register format conversion. Same for V.
- **k_scale:** Applied post-QK-MMA in FP32. The FP32 accumulator already contains
  the raw dot product; multiply by `scale * k_scale` in FP32 (free — no extra MMA).
- **v_scale:** Applied to P in FP32 before BF16 cast for PV MMA. This is ~2x more
  numerically stable than applying v_scale to V after dequant (linear algebra audit
  confirmed: one FP32 truncation error vs two BF16 truncation errors per term).

### 3.8 GQA Handling

Parametric — not hardcoded to any specific ratio.

- **Grid launch:** `(num_q_tiles, num_kv_heads, num_seqs)`
- **Within CTA:** `group_size = num_q_heads // num_kv_heads`
- Q head groups iterate within a single KV head's CTA, reusing the same K/V
  pages from SMEM
- Qwen3.5-27B: group_size=6 (24 Q / 4 KV). Any integer ratio supported.

### 3.9 Causal Masking

- **Prefill:** Per-token mask `q_pos >= kv_pos` applied to scores before softmax.
  Masked positions set to large negative for exp2 (`-2^15` or similar).
- **Decode:** No mask needed — all prior KV tokens are valid for a single query token.

### 3.10 Page Table Iteration

```
for page_idx in range(ceil(seq_len / block_size)):
    physical_page = page_table[seq_idx, page_idx]
    k_smem = cpasync_load(k_cache[physical_page])
    v_smem = cpasync_load(v_cache[physical_page])
    # Handle partial last page: mask out tokens beyond seq_len
```

Page size fixed at 64 tokens. KV cache layout: `(num_pages, page_size, num_kv_heads, head_dim)`.

### 3.11 Numerical Properties

End-to-end error vs FP32 reference (linear algebra audit):
- **Expected:** ~1-2% relative error per output element
- **Worst case (adversarial):** ~5-6%
- **Dominant source:** FP8 quantization of K (stored in KV cache), attenuated by softmax
- **Second source:** P FP32→BF16 truncation (0.39% per element)
- **Online softmax at seq_len=65536:** ~0.01% accumulated error, numerically safe
- **log2/exp2 vs ln/exp:** ~4e-7 additional error, negligible
- **Test tolerance:** ≥2% per element when comparing CuTe kernel vs PyTorch reference

## 4. File Structure

### 4.1 Modified Files

| File | Change |
|------|--------|
| `vllm/v1/attention/backends/cute_paged/kernel.py` | Replace with CuTe JIT kernel + PTX utilities + launcher |
| `vllm/v1/attention/backends/cute_paged/warmup.py` | Wire actual pre-compilation of decode + prefill |
| `vllm/v1/attention/backends/cute_paged/__init__.py` | Cache `__getattr__` results via `globals().update()` |
| `docs/kernel-insights/2026-04-11-b12x-paged-attention.md` | Fix GQA ratio: 6 → 4 |

### 4.2 New Files

| File | Purpose |
|------|---------|
| `tests/nvllm/attention/reference.py` | PyTorch reference paged attention (moved from kernel.py) |

### 4.3 Unchanged Files

| File | Reason |
|------|--------|
| `vllm/v1/attention/backends/cute_paged/_backend.py` | Interface is stable |
| `vllm/v1/attention/backends/cute_paged/disk_cache.py` | Ported and working |

### 4.4 Internal Structure of kernel.py After Replacement

```
1. Imports (CuTe DSL, CUTLASS, torch)
2. Inline PTX utilities (~15 @cute.jit wrappers):
   - ldmatrix_m8n8x4_b16          (SMEM → register load)
   - bf16_mma_m16n16k16_f32       (BF16 MMA, issues 2x m16n8k16)
   - fp8x4_e4m3_to_bfloat2x2     (FP8 K/V dequant to BF16)
   - frag_layout_swizzle_16b_to_8b (FP8 fragment reformat)
   - shfl_xor_sync                (warp shuffle for softmax reduction)
   - exp2_approx_ftz_f32          (fast exp2)
   - cp_async_load_128b           (async GMEM → SMEM 128-bit)
   - permuted_offset_128b         (bank-conflict-free SMEM addressing)
   - shared_ptr_to_u32            (SMEM pointer conversion for ldmatrix)
   - cp_async_commit_group / cp_async_wait_group (pipeline sync)
3. DecodeKernel class — class-based kernel with const_expr() for compile-time branching
   - @cute.kernel decorated method
   - Tile sizes, warp decomposition as instance attributes (Python constants at trace time)
4. PrefillKernel class — same pattern, different tile config
5. KernelConfig frozen dataclass — cta_q, cta_kv, head_dim, block_size, num_warps_q, num_warps_kv
6. DECODE_CONFIG / PREFILL_CONFIG constants
7. _compiled_kernels: dict[KernelConfig, compiled] — in-memory cache (lru_cache on frozen dataclass)
8. paged_attention_forward(...) — Public entry: selects config, launches compiled kernel
```

Class-based kernel pattern (like b12x) for `const_expr()` compile-time branching.
`KernelConfig` is a `@dataclass(frozen=True)` for hashability as cache key.

### 4.5 Warmup Pre-Compilation

`warmup.py` compiles exactly 2 kernel variants (not 8):

```python
WARMUP_CONFIGS = [
    DECODE_CONFIG,   # cta_q=16, cta_kv=64, head_dim=256, 1 stage
    PREFILL_CONFIG,  # cta_q=64, cta_kv=64, head_dim=256, 1 stage
]
```

For each config:
1. Apply disk cache patch
2. Instantiate kernel class (DecodeKernel or PrefillKernel) with config
3. Call `cute.compile(kernel_instance.forward, *dummy_tensors)` to trigger NVRTC
4. Disk cache captures and persists the `.o` file

**If warmup compilation fails, the Docker build must fail.** No silent fallback.

### 4.6 Compilation Error Handling

- `try/except` around CuTe DSL import — if CUTLASS not installed, log warning and
  `paged_attention_forward` falls back to the reference implementation (imported from
  test module). This supports dev environments without SM120.
- If NVRTC compilation fails at runtime (not caught by warmup), raise
  `RuntimeError("CuTe kernel compilation failed")` with the NVRTC error — do not
  silently fall back in production.

### 4.7 Thread Safety Fixes

- `disk_cache.py` `_PATCHED` check-and-set: add `threading.Lock` with double-checked
  locking pattern to prevent wrapper-of-a-wrapper on concurrent calls.
- `_compiled_kernels` dict access: protected by GIL for dict operations, but add a
  note that compilation itself may be called redundantly on first miss from concurrent
  threads (harmless, same as current disk cache behavior).

## 5. Milestone Plan

### Pre-build: Lint

Run `pre-commit run --all-files` on changed files before any docker build. Catch
lint failures early — saves 30+ minutes of wasted build time.

### Milestone 1: CuTe DSL Kernel + Docker Build

1. Opus agent writes the kernel in `kernel.py` (PTX utilities + class-based kernels + launcher)
2. Move PyTorch reference to `tests/nvllm/attention/reference.py`
3. Delete dead helper functions (qk_pass, pv_pass, online_softmax)
4. Wire `warmup.py` for real pre-compilation (2 configs, fail on error)
5. Fix `__init__.py` `__getattr__` caching
6. Fix b12x insights doc GQA ratio
7. Lint (`pre-commit run --all-files`)
8. User reviews all code
9. Docker build in tmux (`docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 .`)
10. Docker MCP for container ops if functional (skip without debugging if not)
11. Commit after clean build — full attribution standards

### Milestone 2: GSM8K Sanity Gate

1. Launch model with `scripts/run_qwen35_27b_cute_paged.sh`
2. GSM8K sanity via `/v1/completions` (not chat — avoids thinking mode)
3. Raw output saved to file
4. User reviews results
5. Commit with results

**Failure gate:** >5% drop vs baseline = correctness issue, fix before proceeding.

### Milestone 3: ShareGPT Bench + nsys Traces + CUDA Graphs

1. ShareGPT throughput bench via nvllm-perf
2. nsys profile: `--privileged` container, `--trace=cuda,nvtx`
3. Enable CUDA graphs (`cudagraph_mode:PIECEWISE`), verify no regression
4. Raw nsys output + `.nsys-rep` files saved to
   `benchmarks/nvllm/traces/cute_paged_attn/2026-04-11-dsl-kernel/`
5. User reviews all bench data + traces
6. Commit with full `summary.md`

**Failure gate:** If nsys shows regression vs FlashInfer, document honestly.

## 6. Subagent Rules

- All subagent reports must be **raw terminal output** or nsys profiles — never summaries
- Output saved to files on disk
- Docker builds run in tmux (subagents time out on 30-50 min SM120 compiles)
- Docker MCP used for container ops if functional; skipped without debugging if broken

## 7. Commit Standards

Every commit in this work must include:

- Pinned commit hashes for any external source reference
- Exact `file.py:L42-L80` permalink references (clickable in commit message)
- `Co-Authored-By:` trailer
- User reviews full diff before `git commit` runs

## 8. Key References

- **b12x paged attention:** lukealonso/b12x@c469c66
  - Insights doc: `docs/kernel-insights/2026-04-11-b12x-paged-attention.md`
- **CUTLASS PR #3030 (SM120 FMHA):**
  - Insights doc: `docs/kernel-insights/2026-04-10-cutlass-pr3030-sm120-fmha.md`
- **Parent spec:** `docs/superpowers/specs/2026-04-10-cute-paged-attention-design.md`
- **Parent plan:** `docs/superpowers/plans/2026-04-10-cute-paged-attention.md` (Tasks 4-7, 12)

## 9. Audit Trail

This spec was reviewed by 7 expert audit agents (2026-04-11). Key changes from v1:

| Change | Driven by |
|--------|-----------|
| Path B (BF16 QK) instead of Path A (FP8 QK) | CuTe DSL + CUTLASS audits |
| CpAsync for all loads (drop TMA for Q) | GPU Kernel + CuTe DSL + CUTLASS audits |
| num_stages=1 for FP8 KV | GPU Kernel + CUTLASS audits (b12x forces this) |
| Inline PTX core, not high-level CuTe DSL | CuTe DSL + CUTLASS audits |
| Class-based kernel pattern | CuTe DSL audit (const_expr requires it) |
| Cross-warp reduction for decode | GPU Kernel + Big-O audits |
| Move reference to test file, delete dead code | Linus audit |
| Fix head_dim=256 (model uses 256, not 128) | GPU Kernel + Linus audits |
| Frozen dataclass + in-memory lru_cache | Python audit |
| Thread-safe monkey-patch | Python audit |
| Test tolerance ≥2% | Linear algebra audit |
| Fix b12x insights GQA ratio | GPU Kernel audit |
| Warmup: 2 configs, fail on error | Linus + CuTe DSL audits |
