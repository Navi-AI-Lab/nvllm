# CUDA Graph Enablement: Pipeline State Refactor

**Date:** 2026-04-14
**Target:** `FULL_AND_PIECEWISE` CUDA graph mode for CuTe paged attention decode kernel
**Model:** natfii/Qwen3.5-27B-NVFP4-Opus-GB10 (`Qwen3_5ForCausalLM` → `Qwen3NextAttention`)
**Hardware:** DGX Spark (GB10, SM120/SM121)

## Motivation

The CuTe uber-kernel (Phases A+B+C: attention + W_O GEMV + RMSNorm) currently runs with
`--enforce-eager` or `PIECEWISE` CUDA graphs — meaning the attention kernel always launches
eagerly. Enabling `FULL_AND_PIECEWISE` mode captures the entire model forward (including
attention) as a single CUDA graph for decode batches, eliminating all inter-op kernel launch
overhead. This is the biggest remaining decode latency win.

The backend already declares `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`, which is the
correct level for `FULL_AND_PIECEWISE` — full graphs for pure decode, piecewise for prefill.

### Secondary motivation: Qwen3.5 fusion wiring

The Phase B+C fusion code currently only exists in `qwen2.py` (`Qwen2Attention` /
`Qwen2DecoderLayer`). The Qwen3.5-27B model uses `Qwen3NextAttention` from `qwen3_next.py`,
which has no fusion wiring. This refactor fixes both problems — CUDA graph compatibility
and Qwen3.5 fusion — by moving all fusion ownership into `CutePagedAttentionImpl`.

## Design Philosophy: Think Like a Shader

The refactor follows GPU graphics pipeline conventions:

| Shader Concept | CUDA Graph Equivalent | Implementation |
|---|---|---|
| Compiled pipeline state | `CutePagedAttentionImpl` | Owns all I/O surfaces, kernel, config |
| Constant buffer (bound per-frame) | Persistent metadata tensors | `seq_lens`, `page_table` — updated in-place before `graph.replay()` |
| UAV / RWBuffer | Persistent output buffers | `wo_output`, `rmsnorm_output`, `residual_output` — fixed addresses |
| Texture binding | `bind_fusion_weights()` | Static weights bound once at init |
| Shader permutation | Always-fused variant | One permutation (A+B+C+gate), no unfused graph |
| Occupancy padding | `seq_len=1` sentinel | Padding CTAs run full code path, produce ignored results |

## Design Decisions

### 1. Single fused permutation

Always capture with all fusions enabled (attention + gate + W_O GEMV + RMSNorm). The
unfused path only exists for debugging, and `--enforce-eager` disables graphs entirely —
no need to capture a graph for the debug path.

### 2. Self-reset for wo_output zeroing

The W_O GEMV accumulates via `atomicAdd` from multiple CTAs. Instead of an external
`zero_()` call (extra kernel launch), each CTA zeros its own output rows before
accumulating. Gated by `block_idx.x == 0` — the first CTA per sequence writes zero + its
contribution, subsequent CTAs atomicAdd normally. Same philosophy as the Phase C arrival
counter self-reset.

### 3. seq_len=1 sentinel for padding CTAs

Following Triton's pattern: padding slots get `seq_len=1` (not 0). Every CTA exercises the
full code path during graph capture — one page load, one QK dot, one softmax, one PV, one
gate multiply, one W_O GEMV, one arrival counter tick, one RMSNorm. The graph records every
operation. No edge cases from empty loops or zero-length code paths.

During replay, padding CTAs produce garbage results in output slots that the model layer
ignores (it only reads `output[:num_actual_tokens]`).

### 4. Persistent I/O buffers owned by CutePagedAttentionImpl

The attention impl becomes a self-contained pipeline state object:

```
CutePagedAttentionImpl (the "pipeline state")
├── Static resources (bound once via bind_fusion_weights)
│   ├── wo_weight         : Tensor [N, K/2] uint8        — NVFP4 weight
│   ├── wo_scales         : Tensor [N, K_sf] fp8         — block scales
│   ├── wo_global_scale   : Tensor [1] fp32              — kernel reads via ld.global
│   ├── rmsnorm_gamma     : Tensor [hidden_dim] bf16     — layernorm weight
│   └── rmsnorm_eps       : float                         — constant
│
├── Persistent I/O buffers (allocated once to max_num_seqs, fixed addresses)
│   │   Content changes each forward; addresses never change (graph-safe).
│   │   Model layer writes dynamic inputs (residual, gate) into these buffers
│   │   before the attention call. Kernel reads/writes at captured addresses.
│   ├── wo_output         : Tensor [max_seqs, hidden_dim] fp32   — atomicAdd target
│   ├── rmsnorm_output    : Tensor [max_seqs, hidden_dim] bf16   — RMSNorm result
│   ├── residual_output   : Tensor [max_seqs, hidden_dim] bf16   — updated residual
│   ├── arrival_count     : Tensor [max_seqs] int32              — self-resetting
│   └── gate_buf          : Tensor [max_seqs, num_heads*head_dim] bf16  — output gate
│
└── Kernel (compiled once, replayed by graph)
    └── DecodeKernel._compiled
```

### 5. wo_global_scale as tensor pointer, not .item()

The kernel reads `wo_global_scale` via `ld.global.b32` from the tensor's stable address.
No Python-level `.item()` call (which causes a device-to-host sync incompatible with graph
capture). The host binds pointers; the kernel reads its own resources.

### 6. Output gate fusion for Qwen3NextAttention

`Qwen3NextAttention` has a sigmoid output gate between attention and W_O:

```
attn_output → sigmoid(gate) * attn_output → W_O GEMV → RMSNorm
```

The gate multiplication slots between the attention epilogue and Phase B:

1. Attention produces `attn_out[head][dim]` in registers
2. Load `gate[token, head*head_dim + dim]` from global memory
3. Apply `sigmoid(gate) * attn_out` element-wise
4. Phase B reads gated result, does W_O GEMV

**PTX for sigmoid:** `1 / (1 + exp2(-x * LOG2E))` — uses existing `exp2`, adds `rcp`
(reciprocal approximation). Three instructions: `fma`, `exp2`, `rcp`.

The `gate` tensor is computed by the model layer before the attention call (from the QKV
projection) and written into the impl's persistent `gate_buf`. Fixed address, graph-safe.

### 7. build_for_cudagraph_capture override

The metadata builder gets a `build_for_cudagraph_capture` override:

1. Calls `build()` to get normal metadata
2. Fills all `seq_lens` slots with `1` (fast capture, minimal KV iteration)
3. Forces `is_decode_only = True`

The `seq_lens` and `block_table` tensors already come from `CommonAttentionMetadata`'s
persistent buffers (managed by the model runner). No new allocation needed in the builder.

## Changes by File

### `vllm/v1/attention/backends/cute_paged/_backend.py`

- `CutePagedAttentionImpl.__init__`: Accept `vllm_config` to read `max_num_seqs`. Allocate
  persistent I/O buffers (`wo_output`, `rmsnorm_output`, `residual_output`, `arrival_count`,
  `gate_buf`).
- `CutePagedAttentionImpl.bind_fusion_weights(wo_weight, wo_scales, wo_global_scale,
  rmsnorm_gamma, rmsnorm_eps)`: Store static weight references. Called once from model layer
  init. Replaces the per-forward side-channel pattern.
- `CutePagedAttentionImpl.forward()`: Remove all `getattr(layer, '_wo_*')` calls. Use
  pre-bound weights and persistent buffers. Pass padded batch size for grid.z.
- `CutePagedMetadataBuilder.build_for_cudagraph_capture()`: Override — fill seq_lens with 1,
  force decode-only.

### `vllm/v1/attention/backends/cute_paged/kernel.py`

- `DecodeKernel.__call__`: Remove `torch.empty_like()` output allocation (use impl's buffer).
  Remove `wo_global_scale.item()` (pass tensor pointer). Grid.z = padded num_seqs. Assert
  query contiguity instead of `.contiguous()` copy. Fusion flags always `Int32(1)`.
- Kernel body: Add wo_output self-zero (CTA zeros its rows when `block_idx.x == 0` before
  atomicAdd). Add sigmoid gate multiply between attention epilogue and W_O GEMV entry.
  Add `rcp` PTX helper (`rcp.approx.f32`).
- `wo_global_scale`: Change from Python float argument to tensor pointer + `ld.global.b32`
  inside kernel.

### `vllm/model_executor/models/qwen3_next.py`

- `Qwen3NextAttention.__init__`: Call `self.attn.impl.bind_fusion_weights(...)` after
  weights are loaded (via `process_weights_after_loading` or lazy init on first forward).
- `Qwen3NextAttention.forward()`: Write `gate` tensor into impl's `gate_buf` before
  attention call. Read `wo_output` / `rmsnorm_output` / `residual_output` from impl after
  attention call instead of calling `self.o_proj()` separately.
- `Qwen3NextDecoderLayer.forward()`: Simplify residual stream — pass `residual` to impl
  via `set_residual()`, read updated residual from impl's `residual_output` buffer.

### `vllm/model_executor/models/qwen2.py`

- Remove all side-channel set/clear code (`self.attn._wo_weight = ...` / `= None` cycle).
- Remove per-forward `torch.zeros()` / `torch.empty()` allocation for fusion buffers.
- Remove `_arrival_count_buf` lazy growth.
- Wire `bind_fusion_weights()` same as qwen3_next.py (keep Qwen2 working for other models).

### `scripts/run_qwen35_27b_nvfp4-opus.sh` → rename

Rename to something cleaner that reflects Qwen3.5. Update `--compilation-config` default
from `PIECEWISE` to `FULL_AND_PIECEWISE`.

## Graph Capture / Replay Flow

### Capture (during model warmup)

```
Worker.compile_or_warm_up_model()
  → model_runner.capture_model()
    → For each batch_size in capture_sizes:
      1. Warmup: _dummy_run(size, mode=NONE, force_attention=True)
         - Triggers cute.compile JIT (cached for all future calls)
         - Exercises full code path including attention
      2. Capture: _dummy_run(size, mode=FULL, is_graph_capturing=True)
         - build_for_cudagraph_capture: seq_lens filled with 1
         - All CTAs run full path (attn → gate → W_O → RMSNorm)
         - CUDAGraphWrapper records the entire forward
```

### Replay (during inference)

```
execute_model(scheduler_output)
  → dispatch(): selects FULL mode + batch_descriptor for decode batch
  → build(): writes real seq_lens/block_table into persistent buffers
  → Impl: writes real residual/gate into persistent buffers
  → set_forward_context(mode=FULL, batch_descriptor=desc)
  → self.model(**inputs)
    → CUDAGraphWrapper detects FULL match → graph.replay()
      - Kernel reads updated seq_lens, page_table, gate, residual
        from same persistent addresses captured in the graph
      - Padding CTAs (seq_len=1) produce ignored results
```

## Validation Plan

1. **GSM8K sanity (eager):** Launch with `--enforce-eager`, verify 8/8 correct — confirms
   the fusion wiring to Qwen3NextAttention works before touching graphs.
2. **GSM8K sanity (PIECEWISE):** Switch to `PIECEWISE` mode — attention still eager but
   confirms the refactored forward path doesn't break piecewise compilation.
3. **GSM8K sanity (FULL_AND_PIECEWISE):** Enable full mode — the real test. Decode batches
   captured and replayed.
4. **Multi-turn conversation test:** Verify graph replay produces correct output across
   multiple turns (seq_lens growing, different batch sizes hitting different captured graphs).
5. **nsys trace:** Profile decode latency before/after to measure the actual speedup from
   eliminating inter-op launch overhead.

## Non-Goals

- Phase D+E (MLP fusion) — separate project, resumes after this ships
- Prefill graph capture — backend is `UNIFORM_SINGLE_TOKEN_DECODE`, prefill stays piecewise
- Speculative decode graph support — would require `UNIFORM_BATCH`, out of scope
- Qwen2 model testing — refactor keeps qwen2.py working but Qwen3.5-27B is the target
