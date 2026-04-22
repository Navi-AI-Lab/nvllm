# D3 "Whole-decode uber-kernel" feasibility study

**Date:** 2026-04-22
**Author:** Natfii + Claude Opus 4.7 (rainy-day research spike)
**Context:** Phase E brainstorming. Considering D3 — one persistent CUDA kernel launch
that state-machines through all 64 decoder layers of Qwen3.5-27B in one decode step.
**Verdict:** **D3 strict is infeasible within reasonable project scope. Fall back to D2.5
(per-fusion-active-layer uber-kernel spanning the decoder-layer boundary). D3 is a
genuinely novel research direction worth documenting as future work.**

---

## TL;DR

Qwen3.5-27B is a hybrid: 16 of 64 decoder layers are standard softmax attention
(`full_attention`), 48 are Gated Delta Net (`linear_attention`) — Mamba-style SSM with a
recurrent state `[HV, V, K] = [32, 128, 128]` FP32 per sequence per layer (~2 MB/seq/layer,
~100 MB/seq across all 48 lin layers). They are interleaved (`full_attention_interval=4`,
pattern `lin, lin, lin, full, …`). Both layer types run a dense NVFP4 MLP afterward.

"D3 strict" is **one kernel launch = whole decode step**. To hit that ceiling we would
need to port to CuTe DSL:

1. `causal_conv1d_update` (small, tractable)
2. `fused_recurrent_gated_delta_rule_packed_decode` (the GDN recurrence — research-level)
3. `RMSNormGated` with the `z`-gate (tractable)
4. The pair of input projections `in_proj_qkvz` + `in_proj_ba` and the output `out_proj`
   (NVFP4 GEMV, leverages Phase D patterns)
5. Full persistent-grid state machine over `layer_idx ∈ [0, 64)` with per-layer-type
   dispatch, per-layer weight pointer arrays, and shared SMEM coordination.

Launch-count budget: current decode step does ~400 launches, D3 strict would do 1.
Saving is roughly 1-2 ms per step = 3-7% at a 30 ms step baseline. The win is real but
not transformative.

Complexity cost: ~3-5× the existing CuTe kernel surface in the fork, including one
kernel (GDN recurrent decode) nobody has publicly written in CuTe DSL on SM120 NVFP4.

**Recommendation:** D2.5 now (one kernel per fusion-active layer, crosses the
decoder-layer boundary — saves 32-48 launches per step, meaningful architectural win,
achievable in a session or two). D3 stays on the research shelf. If it ever happens, it's
its own multi-month arc and deserves a fresh spec.

---

## What Gated Delta Net actually is

Source: `vllm/model_executor/layers/mamba/gdn_linear_attn.py:212` — `GatedDeltaNetAttention`.

Mathematically, per token `t` and per value-head `hv`, GDN maintains a recurrent state
matrix `H ∈ ℝ^{V×K}` and updates it as:

```
g_t  = -exp(A_log[hv]) * softplus(a[t, hv] + dt_bias[hv])   # scalar, decay
β_t  = sigmoid(b[t, hv])                                     # scalar, gate
H_t  = H_{t-1} * exp(g_t)                                    # decay
v'_t = v_t - H_t @ k_t                                       # delta against state
v'_t *= β_t                                                  # gate delta
H_t  += v'_t ⊗ k_t                                           # outer-product update
o_t  = H_t @ q_t                                             # project query through state
```

Where (for Qwen3.5-27B defaults, `linear_num_key_heads=16`, `linear_num_value_heads=32`):
- `K = head_k_dim = 128`
- `V = head_v_dim = 128`
- `HV = 32` value heads (shared `num_v_heads // num_k_heads = 2` V-heads per K-head)
- `H = 16` key heads

State shape: `[HV, V, K] = [32, 128, 128]` FP32 per sequence per layer
= **2 MB / sequence / linear_attention layer**, 48 layers → **96 MB / sequence** total.

This is analogous to (and incompatible with) paged KV cache. Qwen3.5 uses vLLM's Mamba
cache infrastructure for these states (`self.kv_cache = [conv_state, ssm_state]` — two
buffers, not one).

### Shape contrast with full_attention

| | full_attention | linear_attention |
|---|---|---|
| state per token | grows (KV cache, `L × kv_heads × head_dim`) | fixed (`HV×V×K`) |
| compute per token | scales with sequence length (softmax over past) | constant (one matrix update) |
| parallelism | across tokens + heads (batched GEMM/flash) | across heads only; tokens serial within a sequence |
| state format | paged KV blocks | recurrent matrix + conv1d ring buffer |
| memory access | scattered (paged gather) | contiguous per-head |
| primary kernel | `DecodeKernel` (CuTe, ours) | `fused_recurrent_gated_delta_rule_packed_decode_kernel` (Triton, upstream FLA) |

These two compute patterns have **nothing in common at the CTA level**.

---

## Decode-path ops per layer type

Measured from the code paths, not from nsys — a real trace would likely show a few
more from fused-graph fragments. Call counts are per decoder layer per decode step.

### linear_attention layer (48 of these)

Walking `GatedDeltaNetAttention.forward_cuda` at
`vllm/model_executor/layers/mamba/gdn_linear_attn.py:508`:

| # | Op | Kernel |
|---|---|---|
| 1 | `input_layernorm(hidden_states, residual)` | fused_add_rms_norm (1 launch) |
| 2 | `in_proj_qkvz(hidden_states)` | NVFP4 GEMV `[hidden=5120] → [qkv+z_size]` |
| 3 | `in_proj_ba(hidden_states)` | NVFP4 GEMV `[hidden=5120] → [2·HV]` |
| 4 | `torch.zeros(core_attn_out)` | memset |
| 5 | `causal_conv1d_update(mixed_qkv, conv_state, ...)` | triton conv update |
| 6 | `fused_recurrent_gated_delta_rule_packed_decode(...)` | triton recurrent (the star) |
| 7 | `RMSNormGated(core_attn_out, z)` | custom norm with z-gate (1 launch) |
| 8 | `out_proj(core_attn_out)` | NVFP4 GEMV `[value_dim=4096] → [hidden=5120]` |
| 9 | `post_attention_layernorm(hidden_states, residual)` | fused_add_rms_norm |
| 10 | `mlp.gate_up_proj(hidden_states)` | NVFP4 GEMV (via Phase D or stock) |
| 11 | `mlp.act_fn` (SiluAndMul) | fused in D, 1 launch in stock |
| 12 | `mlp.down_proj(intermediate)` | NVFP4 GEMV (fused in D) |

→ **~7-10 kernel launches per linear_attention decode layer.**

The Phase D MLP fusion collapses 10-12 into a single launch. But steps 2-8 still stand —
GDN core has its own launch ladder.

### full_attention layer (16 of these, with Phase B+C+D active)

| # | Op | Kernel |
|---|---|---|
| 1 | `input_layernorm(hidden_states, residual)` | fused_add_rms_norm |
| 2 | `qkv_proj(hidden_states)` | NVFP4 GEMV |
| 3 | Attention + W_O + post_attn_RMSNorm + residual | **CuTe uber-kernel (Phases A+B+C)** — single launch |
| 4 | `mlp.gate_up + SiLU + down` | **Phase D fused kernel** — single launch |

→ **~3-4 kernel launches per full_attention decode layer** (fusion-active).

### Total per decode step

| | linear_attention | full_attention | Step total |
|---|---|---|---|
| layers | 48 | 16 | 64 |
| launches/layer | ~7-10 | ~3-4 | — |
| subtotal launches | ~360-480 | ~50-65 | **~410-545** |

Plus residual add + `self.norm` (final) + logits projection = a handful more.

Round number: **~400-500 kernel launches per decode step**, dominated by the 48
linear_attention layers which have no fork-side fusion.

---

## What D3 strict would have to do

One persistent kernel launch per decode step. Persistent grid over layers, per-CTA state
machine. Sketch:

```
__global__ void d3_uber(layer_meta[64], weights*, states*, hidden_states[B, 5120]):
    for layer_idx in 0..63:
        grid.sync()  // barrier between layers
        dispatch(layer_meta[layer_idx].type):
            case FULL_ATTENTION:
                CuTe attention+W_O+RMSNorm (A+B+C pattern)
                CuTe MLP gate_up+SwiGLU+down (D pattern)
                residual_add
                next-layer input_layernorm
            case LINEAR_ATTENTION:
                CuTe input projections (qkvz, ba)
                CuTe causal_conv1d_update
                CuTe GDN recurrent update  ← NOVEL
                CuTe RMSNormGated
                CuTe out_proj
                CuTe MLP gate_up+SwiGLU+down (D pattern)
                residual_add
                next-layer input_layernorm
```

Each CTA runs this state machine. The grid size has to be sized for the worst-case layer
type (likely the GDN recurrent's `(NV, B*HV) = (4, 16B)` grid dominates at small batch).

### Blockers, each with a cost estimate

#### B1: Port `fused_recurrent_gated_delta_rule_packed_decode` to CuTe DSL

Source: `vllm/model_executor/layers/fla/ops/fused_recurrent.py:256-335`.

The kernel itself is small — ~80 lines of Triton, compute pattern fits in one CTA per
`(v-tile, batch×head)` program. The math is mostly `tl.load` / `tl.sum` / outer-product
update. Porting the arithmetic to CuTe DSL should be straightforward.

**Hard parts:**
- **L2-norm + softplus + exp + sigmoid** inside the kernel — we have exp2/softplus patterns
  from Phase A/D (PTX helpers in attention kernel), but rsqrt / sigmoid are new.
- **Recurrent state `[HV, V, K] = [32, 128, 128]` per sequence** has to be loaded/stored
  — one slot per sequence in a global buffer. Whole state per token is 2 MB, way too big
  for SMEM. Work happens with per-V-tile (BV=32 rows) × full-K (=128) shape, so we load
  `[BV × K] = [32 × 128] × 4B = 16 KB` of state per CTA. Fits.
- **FP32 accumulation** everywhere — state is FP32. Fine, matches our Phase D pattern.
- **Per-sequence state indirection** via `ssm_state_indices` — paged-cache style. New
  pattern for our CuTe work but tractable (Phase B paged attention gather is similar).

Estimated effort: **1-2 weeks** of focused kernel work, including bit-exact verification
against the Triton reference. First-of-its-kind on SM120 NVFP4 stack.

#### B2: Port `causal_conv1d_update` to CuTe DSL

Source: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` (not read here, but
it's a short-kernel-window update — likely <50 lines of Triton for the update variant).

Conceptually a 1-D convolution against a ring buffer of the last `conv_kernel_size=4`
tokens. Small, local memory access. CuTe port is not hard.

Estimated effort: **2-3 days** including validation.

#### B3: Port `RMSNormGated` (with z-gate) to CuTe DSL

Source: `vllm/model_executor/layers/layernorm.py` — `RMSNormGated`. Takes `(core_attn_out,
z)` and does `norm_before_gate=True` variant: `RMSNorm(out) * SiLU(z)` (or similar; actual
math in the class body).

Already have RMSNorm primitives from Phase C. Adding a SiLU gate is straightforward.

Estimated effort: **1-2 days**.

#### B4: Two more NVFP4 GEMV shapes (in_proj_qkvz, in_proj_ba, out_proj)

| weight | input dim | output dim |
|---|---|---|
| in_proj_qkvz | 5120 | `2·K·H + 2·V·HV = 2·128·16 + 2·128·32 = 4096 + 8192 = 12288` |
| in_proj_ba | 5120 | `2·HV = 64` |
| out_proj | `V·HV = 4096` | 5120 |

All compatible with the Phase D FP4 GEMV pattern (skinny M=1). Would need to fit into
the winners table or a dispatched variant. **Probably the easiest piece.**

Estimated effort: **2-3 days** including sweep + winners-table entry.

#### B5: Persistent-grid state machine over 64 layers

Biggest architectural unknown. Key questions:

- **Grid size.** Attention kernel uses `grid = (num_q_tiles, num_kv_heads, num_seqs)`.
  MLP uses `grid = (slice_ctas, num_k_tiles, num_tokens)`. GDN recurrent uses `grid =
  (NV, B * HV)`. Unifying means sizing for the max-CTA shape (likely GDN at `4 × 16B`
  for small batch) and no-oping extra CTAs in other layer types. **Wasteful.**

- **Grid-wide sync between layers.** Requires `__grid_sync()` (cooperative-groups).
  Cooperative-launch grids have a hard upper bound on CTA count = `occupancy × num_SMs`.
  GB10 has 48 SMs; if each CTA uses moderate SMEM/regs we can fit ~96-192 CTAs in a
  cooperative grid. **That's tight for the widest-grid layers.** If the GDN recurrent's
  natural grid exceeds the cooperative-launch cap, D3 is architecturally blocked unless
  we restructure the recurrent decomposition.

- **Per-CTA state machine via `switch(layer_type)`.** Every CTA runs an `if/else` branch
  per layer. Warp divergence is fine at the CTA level but branch code size blows up the
  kernel — effectively we're compiling *both* linear and full paths into each CTA even
  though only one is used at a time. Register budget must fit the worst case of either
  branch. **Likely 2-3× registers per CTA vs specialized.**

- **Weight pointer arrays.** 64 layers × ~12 weight pointers each = ~768 Int64s to pass
  in. Fine as a tensor, but indexing adds latency per-layer. Some layers have different
  numbers of weights (full: ~6; linear: ~12 incl conv1d), so packing is awkward.

- **Shared SMEM reuse across layers.** Worth something — SMEM doesn't need to be
  reallocated between sub-kernels. But if GDN's SMEM footprint is bigger than attention's,
  we pay the GDN cost everywhere.

Estimated effort: **1 month+** of research + iteration. This is the part that nobody has
publicly done on an NVFP4 SM120 stack.

#### B6: CUDA-graph compatibility

Current Phase D works under PIECEWISE graphs (per `project_cuda_graphs_next` and
`project_full_graph_blocked`). FULL graphs are blocked on the CuTe backend for reasons
that were partially diagnosed but are not fully fixed.

D3 strict is essentially "whole decode = 1 kernel" which IS the FULL-graph endgame from
a launch-count perspective, but it requires getting the CuTe capture story right across
a persistent grid. **Adjacent to the FULL-graphs CuTe-bug on the open-issues list.**

### Summary

| Blocker | Effort | Novelty |
|---|---|---|
| B1: Port GDN recurrent to CuTe | 1-2 weeks | High (first-of-its-kind) |
| B2: Port causal_conv1d to CuTe | 2-3 days | Low |
| B3: Port RMSNormGated to CuTe | 1-2 days | Low |
| B4: 3 more NVFP4 GEMV shapes | 2-3 days | Low (reuses Phase D) |
| B5: Persistent grid state machine | 1 month+ | Very high |
| B6: CUDA-graph compat with persistent | 1-2 weeks | Medium |

**Total realistic timeline: 2-3 months of focused work for D3 strict.**

---

## Budget analysis

### Launches saved

Decode step baseline: **~400-500 launches** (see above).

D3 strict: **1 launch**.

Per-launch overhead on GB10 is typically 2-5 μs (queueing, driver, CUPTI). Call it 3 μs
average. Savings: **~1.2-1.5 ms per decode step**.

At 30 ms/step, that's **4-5% end-to-end**.

### Alternative: D2.5 (one kernel per fusion-active layer, crosses layer boundary)

D2.5 fuses:
- `input_layernorm(hs, res)` (of this layer)
- `qkv_proj → attn → W_O → post_attn_RMSNorm → residual` (Phases A+B+C, already fused)
- `gate_up → SwiGLU → down → residual_add` (Phase D extended with +residual)
- `next-layer input_layernorm` (pulled in as epilogue — same primitive whether next layer is linear or full)

Launches before: 16 full layers × ~3 launches = ~48 (within the fusion-active subset).
Launches after: 16 full layers × 1 launch = **16**. Savings: **32 launches × 3 μs ≈ 100 μs ≈ 0.3% per step**.

Smaller win, but:
- Achievable in 1-2 sessions
- Zero new CuTe kernels required (extends existing Phase A+B+C+D patterns)
- Clean architectural win (one kernel = one fusion-active decoder layer)
- Keeps the learning arc going without blowing up into a research project

### Alternative: D3-pragmatic (persistent across 16 full_attention layers only)

Fuse the 16 full_attention layers into **one persistent kernel**, release GPU between to
let linear_attention layers run as separate launches. This means:
- 48 linear_attention layers × ~7 launches = **~336 launches** (unchanged)
- 16 full_attention layers × 1 persistent kernel = **16 sub-regions in one launch**
- Plus fused_add_rmsnorm transitions between full and lin

Still N kernel launches per step (N = 48×7 + 1 + transitions ≈ 400). Doesn't save much
beyond D2.5 because linear_attention still dominates.

**D3-pragmatic is not a meaningful improvement over D2.5.** Skip it.

### Big-picture

|  | Launches/step | Time saved | Effort | Verdict |
|---|---|---|---|---|
| D1 | ~400 (no change) | 0 | ~1 session | baseline to beat |
| D2 | ~400 - 2 | ~6 μs | 1-2 sessions | too small to chase |
| D2.5 (recommended) | ~400 - 32 | ~100 μs (0.3%) | 1-2 sessions | **pragmatic** |
| D3-pragmatic (full-attn only) | ~400 - 50 | ~150 μs (0.5%) | ~1 month | skip |
| D3 strict | 1 | ~1.5 ms (5%) | **2-3 months** | aspirational |

D3 strict's 5% win is real but nobody would scope it without first doing the 1-2 session
D2.5 and measuring. And once D2.5 is shipped, the next performance lever is almost
certainly NOT D3 — it's either tensor-cores in Phase D (MMA atom upgrade, much bigger
per-kernel win) or widening Phase D's winners table to cover more NVFP4 shapes.

---

## Verdict

**D3 strict fails on engineering economics, not on physics.** The compute is doable, the
state machine works on paper, but the cost to ship (2-3 months including a first-of-kind
GDN CuTe port) is way out of proportion to the 5% perf win. The win also diminishes
every time Phase D gets its tiles tuned better (because per-kernel time goes down, launch
overhead becomes a bigger *fraction* but smaller *absolute*).

**D2.5 is the correct pragmatic target for Phase E.** It:
- Saves ~32 launches per decode step (~0.3%)
- Reuses every existing CuTe primitive and pattern
- Introduces exactly one architectural novelty: pulling next-layer's `input_layernorm` into
  the previous fusion-active kernel's epilogue, which works whether next layer is full or
  linear (both start with `input_layernorm(hs, res)` as a fused_add_rmsnorm primitive).
- Delivers a clean "one kernel = one fusion-active decoder layer" story — the spirit of
  the "Unreal material graph" idea applied within the achievable scope.

**D3 stays on the shelf as documented future work.** If ever pursued, the prerequisites
are:
1. A CuTe port of `fused_recurrent_gated_delta_rule_packed_decode` (standalone, validated)
2. A CuTe port of `causal_conv1d_update` (standalone, validated)
3. A persistent-grid pattern working for at least one simpler case (e.g., a persistent
   version of Phase D alone)
4. FULL CUDA graphs working for the CuTe backend (currently blocked per
   `project_full_graph_blocked`)

Each of those is its own project. Only after all four land would a full D3 attempt make
sense. Realistically that's a 6-12 month arc with a lot of "does anyone else have this
working?" checkpoints in the upstream ecosystem.

---

## What we learned (the actual rainy-day value)

1. **GDN has its own per-token state update** that has nothing in common with full
   attention's KV cache lookup. Fusing them into one persistent grid is a research
   problem, not an engineering task.

2. **48 of 64 layers are linear_attention** — a much bigger share than we'd assumed.
   Any "fuse full_attention layers together" approach barely dents the launch budget.

3. **Every decoder layer's MLP is the same Phase D target** — linear and full share
   the dense MLP. A D2.5-style cross-boundary fusion that pulls in the next layer's
   `input_layernorm` works for both layer types seamlessly (because both start with
   `input_layernorm(hs, res)` as a fused primitive).

4. **The big remaining Phase D perf lever isn't layer-count fusion; it's tensor
   cores.** Current Phase D is scalar-FP32-accum + warp-shuffle. Upgrading to an MMA
   atom is a whole separate arc and probably bigger than everything else.

5. **Upstream FLA and vLLM do most of the GDN work today**, with some fork-side
   shim (`fi_chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule_packed_decode`).
   Anything we port to CuTe has to bit-match those exactly — they're the ground truth.

---

## Recommendation for the Phase E brainstorm

1. Commit to **D2.5** as the Phase E design target. Start a fresh brainstorm for D2.5
   now that we know the shape (residual-add-in-epilogue + next-layer-input-layernorm
   pulled in).
2. File D3 as `docs/kernel-insights/2026-04-22-d3-feasibility.md` (this doc) and
   reference from the D2.5 spec under "Non-goals (deferred to D3)".
3. Revisit D3 only if Phase D's kernel time drops so much that launch overhead becomes
   the limiting factor (unlikely in the next 6 months).

---

## References

- `vllm/model_executor/layers/mamba/gdn_linear_attn.py` — `GatedDeltaNetAttention`
  class (1178 lines).
- `vllm/model_executor/layers/fla/ops/fused_recurrent.py:256-335` — the decode kernel
  that would need CuTe-porting for D3.
- `vllm/model_executor/models/qwen3_next.py:400-454` — decoder-layer dispatch between
  full and linear attention.
- `vllm/v1/attention/backends/cute_paged/mlp_kernel.py` — current Phase D MLP fusion
  (would extend with residual-add + next-layer-input-layernorm in D2.5).
- `vllm/nvllm/models/qwen3_5.py` — fork-owned decoder layer wrapping both paths, where
  the D2.5 kernel binding happens.
- `docs/superpowers/specs/2026-04-17-unreal-kernel-phase-d-mlp-fusion-design.md:502-509`
  — original "Phase E Preview" section; this doc supersedes it.
- `project_fused_path_nondeterminism.md` — correctness fix that unblocked Phase D
  shipping (per-CTA slot + fixed-order gather replacing non-det atomicAdd).
