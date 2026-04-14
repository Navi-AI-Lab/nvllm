# CUDA Graph Pipeline State Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `FULL_AND_PIECEWISE` CUDA graph mode for the CuTe paged attention decode kernel, and wire Phase B+C fusion into `Qwen3NextAttention` (the model class actually used by Qwen3.5-27B).

**Architecture:** The attention impl (`CutePagedAttentionImpl`) becomes a self-contained pipeline state object owning persistent I/O buffers and pre-bound fusion weights. The kernel dispatch removes all Python-level dynamic allocation and host-device syncs. The metadata builder provides a `build_for_cudagraph_capture` override. The model layer (`Qwen3NextAttention`/`Qwen3NextDecoderLayer`) calls `bind_fusion_weights()` once after weight loading and writes dynamic inputs (gate, residual) into persistent buffers each forward.

**Tech Stack:** CuTe Python DSL (CUTLASS 4.4.2), PTX inline assembly, PyTorch, vLLM V1 attention backend API

**Spec:** `docs/superpowers/specs/2026-04-14-cuda-graphs-pipeline-state-design.md`
**Blockers checklist:** 10 items cataloged in project memory `project_cudagraph_blockers.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `vllm/v1/attention/backends/cute_paged/_backend.py` | Modify | Builder override + impl persistent buffers + bind_fusion_weights + forward refactor |
| `vllm/v1/attention/backends/cute_paged/kernel.py` | Modify | Graph-safe dispatch + self-zero + gate fusion + rcp PTX + tensor pointer wo_global_scale |
| `vllm/model_executor/models/qwen3_next.py` | Modify | Fusion wiring in Qwen3NextAttention + Qwen3NextDecoderLayer |
| `scripts/serve-cute.sh` | Modify | Update compilation config default for validation |

---

### Task 1: Builder — `build_for_cudagraph_capture` override

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:305-367`

This is the simplest change. Follow the Triton backend pattern: call `self.build()`, then fill `seq_lens` with 1 to keep graph capture fast.

- [ ] **Step 1: Add the override method to `CutePagedMetadataBuilder`**

After the existing `build()` method (ends ~line 367), add:

```python
def build_for_cudagraph_capture(
    self, common_attn_metadata: CommonAttentionMetadata,
) -> CutePagedMetadata:
    """Override for CUDA graph capture.

    Fills seq_lens with 1 so every CTA exercises the full code path
    (one page load, one QK dot, etc.) during capture. Padding slots
    produce ignored results.
    """
    attn_metadata = self.build(0, common_attn_metadata)
    # All slots get seq_len=1: fast capture, full code path exercised
    attn_metadata.seq_lens.fill_(1)
    attn_metadata.is_decode_only = True
    return attn_metadata
```

- [ ] **Step 2: Verify compilation config support**

Confirm the builder already declares `UNIFORM_SINGLE_TOKEN_DECODE` at line 310-312 — this is the level needed for `FULL_AND_PIECEWISE`. No change needed, just verify:

```python
_cudagraph_support: ClassVar[AttentionCGSupport] = (
    AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
)
```

- [ ] **Step 3: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_backend.py
git commit -m "feat(cuda-graphs): add build_for_cudagraph_capture override to CuTe builder

Follows Triton backend pattern: build normal metadata then fill
seq_lens with 1 for fast capture. All CTAs exercise the full code
path during graph recording.

Blocker #10 from the CUDA graphs checklist."
```

---

### Task 2: Impl — persistent buffers, `bind_fusion_weights()`, forward refactor

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:149-273`

The impl becomes a pipeline state object. Fusion weights are bound once at init (not per-forward). Persistent I/O buffers replace dynamic allocation. The `getattr(layer, '_wo_*')` side-channel pattern is removed.

- [ ] **Step 1: Add `bind_fusion_weights()` method to `CutePagedAttentionImpl`**

After `__init__` (ends ~line 196), add the method. It stores static weight references AND allocates persistent I/O buffers:

```python
def bind_fusion_weights(
    self,
    wo_weight: torch.Tensor,
    wo_scales: torch.Tensor,
    wo_global_scale: torch.Tensor,
    rmsnorm_gamma: torch.Tensor,
    rmsnorm_eps: float,
    max_num_seqs: int,
) -> None:
    """Bind static fusion weights and allocate persistent I/O buffers.

    Called once from the model layer after weight loading. Replaces
    the per-forward side-channel set/clear pattern. All buffer
    addresses are stable — safe for CUDA graph capture and replay.

    Args:
        wo_weight: NVFP4 packed weights [N, K/2] uint8
        wo_scales: Per-block scales [N, K_sf] fp8
        wo_global_scale: Scalar scale [1] fp32 (kernel reads via ld.global)
        rmsnorm_gamma: LayerNorm weight [hidden_dim] bf16
        rmsnorm_eps: LayerNorm epsilon (e.g. 1e-6)
        max_num_seqs: Maximum batch size for buffer allocation
    """
    # Static weights (bound once, never change)
    self.wo_weight = wo_weight
    self.wo_scales = wo_scales
    self.wo_global_scale = wo_global_scale
    self.rmsnorm_gamma = rmsnorm_gamma
    self.rmsnorm_eps = rmsnorm_eps

    hidden_dim = rmsnorm_gamma.shape[0]
    device = wo_weight.device
    q_size = self.num_heads * self.head_size  # num_heads * head_dim

    # Persistent I/O buffers — fixed addresses for graph capture/replay.
    # Content changes each forward; addresses never change.
    self.wo_output = torch.zeros(
        max_num_seqs, hidden_dim, dtype=torch.float32, device=device)
    self.rmsnorm_output = torch.empty(
        max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device)
    self.residual_output = torch.empty(
        max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device)
    self.arrival_count = torch.zeros(
        max_num_seqs, dtype=torch.int32, device=device)
    self.gate_buf = torch.empty(
        max_num_seqs, q_size, dtype=torch.bfloat16, device=device)
    self.residual_buf = torch.empty(
        max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device)
    # NOTE: residual_buf is not in the spec's buffer diagram but is needed
    # for graph safety — the model layer copies the input residual here
    # before attention, ensuring a stable address for the kernel's Phase C.

    self._fusion_bound = True

    logger.info(
        "CuTe fusion bound: hidden_dim=%d, q_size=%d, max_seqs=%d, "
        "wo_weight=%s, rmsnorm_gamma=%s",
        hidden_dim, q_size, max_num_seqs,
        list(wo_weight.shape), list(rmsnorm_gamma.shape),
    )
```

- [ ] **Step 2: Initialize `_fusion_bound` flag in `__init__`**

At the end of `__init__` (after line 196), add:

```python
self._fusion_bound = False
```

- [ ] **Step 3: Refactor `forward()` to use pre-bound weights**

Replace the `getattr(layer, '_wo_*')` side-channel reads (lines 220-236) with reads from `self`:

```python
def forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: CutePagedMetadata,
    output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    assert output is not None, "Output tensor must be provided"

    if attn_metadata is None:
        return output.fill_(0)

    k_scale = getattr(layer, "_k_scale_float", 1.0)
    v_scale = getattr(layer, "_v_scale_float", 1.0)

    # Fusion weights: pre-bound via bind_fusion_weights() or None
    wo_weight = self.wo_weight if self._fusion_bound else None
    wo_scales = self.wo_scales if self._fusion_bound else None
    wo_global_scale = self.wo_global_scale if self._fusion_bound else None
    wo_output = self.wo_output if self._fusion_bound else None
    rmsnorm_gamma = self.rmsnorm_gamma if self._fusion_bound else None
    rmsnorm_residual = self.residual_buf if self._fusion_bound else None
    rmsnorm_output = self.rmsnorm_output if self._fusion_bound else None
    residual_output = self.residual_output if self._fusion_bound else None
    arrival_count = self.arrival_count if self._fusion_bound else None
    rmsnorm_eps = self.rmsnorm_eps if self._fusion_bound else None
    gate_buf = self.gate_buf if self._fusion_bound else None

    from vllm.v1.attention.backends.cute_paged.kernel import (
        paged_attention_forward,
    )

    num_actual_tokens = attn_metadata.num_actual_tokens

    # For graph-safe dispatch: padded batch size for grid.z,
    # and the persistent attention output buffer (avoids empty_like).
    num_seqs = len(attn_metadata.seq_lens)
    padded_num_seqs = num_seqs  # graph capture overrides via metadata

    result = paged_attention_forward(
        query=query[:num_actual_tokens],
        kv_cache=kv_cache,
        page_table=attn_metadata.block_table,
        seq_lens=attn_metadata.seq_lens,
        scale=self.scale,
        k_scale=k_scale,
        v_scale=v_scale,
        page_size=64,
        query_start_loc=attn_metadata.query_start_loc,
        wo_weight=wo_weight,
        wo_scales=wo_scales,
        wo_global_scale=wo_global_scale,
        wo_output=wo_output,
        rmsnorm_gamma=rmsnorm_gamma,
        rmsnorm_residual=rmsnorm_residual,
        rmsnorm_output=rmsnorm_output,
        residual_output=residual_output,
        arrival_count=arrival_count,
        rmsnorm_eps=rmsnorm_eps,
        gate_buf=gate_buf,
        padded_num_seqs=padded_num_seqs,
    )

    output[:num_actual_tokens].copy_(result)
    return output
```

**Key changes from current code:**
- Reads fusion state from `self.*` (pre-bound) instead of `getattr(layer, '_*')`
- Passes `gate_buf` to the kernel (new parameter)
- `_fusion_bound` flag controls whether fusion tensors or None are passed
- No per-forward side-channel set/clear needed

- [ ] **Step 4: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_backend.py
git commit -m "feat(cuda-graphs): add bind_fusion_weights and persistent buffers to impl

CutePagedAttentionImpl becomes a pipeline state object:
- bind_fusion_weights() stores static weights + allocates persistent
  I/O buffers with fixed addresses (graph-safe)
- forward() reads from self instead of per-forward side-channels
- gate_buf added for output gate fusion (Qwen3NextAttention)

Blockers #6, #7, #8 from the CUDA graphs checklist."
```

---

### Task 3: Kernel dispatch — graph-safe `DecodeKernel.__call__`

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/kernel.py:1774-1901`

Fix all 5 dispatch-level CUDA graph blockers in `DecodeKernel.__call__`.

- [ ] **Step 1: Fix `query.contiguous()` — assert instead of copy (Blocker #4)**

At line 1821, replace:
```python
q_flat = query.contiguous().view(-1)
```
with:
```python
assert query.is_contiguous(), (
    "CuTe decode kernel requires contiguous query tensor"
)
q_flat = query.view(-1)
```

- [ ] **Step 2: Fix `torch.empty_like(query)` — accept caller-provided output (Blocker #2)**

At line 1829, replace:
```python
output = torch.empty_like(query)
```
with:
```python
output = kwargs.get("output_buf", None)
if output is None:
    output = torch.empty_like(query)
```

When the impl passes a persistent output buffer, the kernel uses it. When called standalone (tests), it allocates as before.

- [ ] **Step 3: Fix `wo_global_scale.item()` — pass tensor pointer (Blocker #1)**

At line 1839, replace:
```python
wo_gs = float(wo_global_scale.item())
```
with:
```python
wo_gs_ptr = Int64(wo_global_scale.data_ptr())
```

This changes the kernel argument from a Python float to a pointer. The kernel will read the value via `ld.global.b32` (handled in Task 4).

Also update the else branch (line 1848):
```python
wo_gs_ptr = Int64(0)
```

Update `all_args` tuple (~line 1878): replace `wo_gs` with `wo_gs_ptr`.

- [ ] **Step 4: Fix dynamic grid — use padded num_seqs (Blocker #3)**

At line 1827, change:
```python
grid = (num_q_tiles, num_kv_heads, num_seqs)
```
to:
```python
# For graph capture: grid.z = padded batch size (from caller).
# Padding CTAs get seq_len=1 sentinel and produce ignored results.
padded_num_seqs = kwargs.get("padded_num_seqs", num_seqs)
grid = (num_q_tiles, num_kv_heads, padded_num_seqs)
```

- [ ] **Step 5: Fix Python branching — always-fused flag constants (Blocker #5)**

Replace the `if wo_weight is not None:` / `else:` branches (lines 1835-1851) with a single path. When fusion is active, pointers are real; when not, they're zero. The kernel guards on `wo_fused_flag` which is always `Int32(1)` when called with fusion, `Int32(0)` without:

```python
if wo_weight is not None:
    wo_weight_ptr = Int64(wo_weight.data_ptr())
    wo_scale_ptr = Int64(wo_scales.data_ptr())
    wo_output_ptr = Int64(wo_output.data_ptr())
    wo_gs_ptr = Int64(wo_global_scale.data_ptr())
    wo_K = num_q_heads * self.head_dim
    wo_nkt = Int32((wo_K // 16 + 3) // 4)
    wo_row_stride = Int32(wo_weight.shape[1])
    wo_fused_flag = Int32(1)
else:
    wo_weight_ptr = Int64(0)
    wo_scale_ptr = Int64(0)
    wo_output_ptr = Int64(0)
    wo_gs_ptr = Int64(0)
    wo_nkt = Int32(0)
    wo_row_stride = Int32(0)
    wo_fused_flag = Int32(0)
```

This looks the same as before but with `wo_gs` → `wo_gs_ptr`. The critical change is that when called from the graph-safe impl, `wo_weight is not None` is ALWAYS true (fusion always active), making this branch constant during graph capture. The `wo_fused_flag = Int32(1)` is a compile-time constant baked into the captured kernel.

- [ ] **Step 6: Add gate_buf pointer to kernel arguments**

After the Phase C args block (~line 1876), add:

```python
# Output gate: pointer to gate values for sigmoid fusion
gate_buf = kwargs.get("gate_buf", None)
if gate_buf is not None:
    gate_ptr = Int64(gate_buf.data_ptr())
    gate_fused_flag = Int32(1)
else:
    gate_ptr = Int64(0)
    gate_fused_flag = Int32(0)
```

Add `gate_ptr` and `gate_fused_flag` to the `all_args` tuple. Update `_jit_launch` and `_kernel` signatures to accept these two new parameters.

- [ ] **Step 7: Update `_jit_launch` and `_kernel` signatures**

Add `gate_ptr: Int64` and `gate_fused: Int32` parameters after `rmsnorm_fused: Int32` in both:

```python
@cute.jit
def _jit_launch(self, query, k_ptr: Int64, v_ptr: Int64,
                page_table, seq_lens, output,
                scale, k_scale, v_scale,
                num_q_heads, num_kv_heads,
                kv_page_stride: Int32,
                wo_weight_ptr: Int64, wo_scale_ptr: Int64,
                wo_output_ptr: Int64, wo_gs_ptr: Int64,  # Changed: was wo_global_scale float
                wo_num_k_tiles: Int32,
                wo_weight_row_stride: Int32,
                wo_fused: Int32,
                rmsnorm_gamma_ptr: Int64,
                rmsnorm_residual_ptr: Int64,
                rmsnorm_output_ptr: Int64,
                residual_output_ptr: Int64,
                arrival_count_ptr: Int64,
                rmsnorm_eps,
                hidden_dim: Int32,
                total_ctas_per_seq: Int32,
                rmsnorm_fused: Int32,
                gate_ptr: Int64,        # NEW
                gate_fused: Int32,      # NEW
                grid_x: Int32, grid_y: Int32, grid_z: Int32):
```

Apply the same signature change to `_kernel`.

**IMPORTANT:** This changes the `_jit_launch` and `_kernel` signatures. The `_compiled` cache (`self._compiled`) is keyed on the first compilation call. Since we're changing signatures, the cached kernel is invalidated — delete any `__pycache__` and `.cache/vllm_compile` artifacts before testing.

- [ ] **Step 8: Update `paged_attention_forward()` function signature**

The top-level `paged_attention_forward()` function (~line 2060+) needs `gate_buf`, `output_buf`, and `padded_num_seqs` added to its parameter list, then passed through to `DecodeKernel.__call__`:

```python
def paged_attention_forward(
    query, kv_cache, page_table, seq_lens, scale, k_scale, v_scale,
    page_size, query_start_loc,
    wo_weight=None, wo_scales=None, wo_global_scale=None, wo_output=None,
    rmsnorm_gamma=None, rmsnorm_residual=None, rmsnorm_output=None,
    residual_output=None, arrival_count=None, rmsnorm_eps=None,
    gate_buf=None,            # NEW
    output_buf=None,          # NEW
    padded_num_seqs=None,     # NEW
):
```

Pass these through in the `decode_kernel(...)` call:
```python
return decode_kernel(
    query=query, kv_cache=kv_cache, page_table=page_table,
    seq_lens=seq_lens, scale=scale, k_scale=k_scale, v_scale=v_scale,
    wo_weight=wo_weight, wo_scales=wo_scales,
    wo_global_scale=wo_global_scale, wo_output=wo_output,
    rmsnorm_gamma=rmsnorm_gamma, rmsnorm_residual=rmsnorm_residual,
    rmsnorm_output=rmsnorm_output, residual_output=residual_output,
    arrival_count=arrival_count, rmsnorm_eps=rmsnorm_eps,
    gate_buf=gate_buf,
    output_buf=output_buf,
    padded_num_seqs=padded_num_seqs,
)
```

- [ ] **Step 9: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/kernel.py
git commit -m "feat(cuda-graphs): graph-safe kernel dispatch — remove .item(), empty_like, .contiguous()

DecodeKernel.__call__ changes:
- query: assert contiguity instead of .contiguous() copy
- output: accept caller-provided persistent buffer
- wo_global_scale: pass tensor pointer, not .item() float
- grid.z: use padded_num_seqs for stable graph capture
- gate_ptr/gate_fused: new kernel args for output gate fusion

Blockers #1, #2, #3, #4, #5 from the CUDA graphs checklist."
```

---

### Task 4: Kernel body — self-zero, gate fusion, rcp PTX, wo_global_scale ld.global

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/kernel.py` (PTX helpers ~line 818, kernel body ~line 1008)

Three kernel body changes + one new PTX primitive.

- [ ] **Step 1: Add `_rcp_approx_f32` PTX helper**

After `_rsqrt_approx_f32` (~line 858), add:

```python
@dsl_user_op
def _rcp_approx_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """Hardware reciprocal approximation — rcp.approx.ftz.f32.

    Used for sigmoid: 1 / (1 + exp2(-x * LOG2E)).
    Single-instruction, sufficient precision for gating.
    """
    x_ir = Float32(x).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.f32(), [x_ir],
        "rcp.approx.ftz.f32 $0, $1;", "=f,f",
        has_side_effects=True, loc=loc, ip=ip)
    return Float32(result_ir)
```

- [ ] **Step 2: Add `_ld_global_b32_to_f32` PTX helper for wo_global_scale**

After the new `_rcp_approx_f32`, add a helper to load a 32-bit value from global memory and bitcast to float (for reading `wo_global_scale` tensor via pointer):

```python
@dsl_user_op
def _ld_global_b32_to_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """Load 32-bit from global memory, bitcast to FP32.

    Used for reading wo_global_scale from tensor pointer
    instead of Python .item() (graph-safe).
    """
    addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.f32(), [addr_ir],
        "ld.global.b32 $0, [$1];", "=f,l",
        has_side_effects=True, loc=loc, ip=ip)
    return Float32(result_ir)
```

Note: `_ld_global_f32` already exists at line 860 with the same PTX instruction `ld.global.f32`. These are equivalent — using `b32` (bitwise) vs `f32` (float) for the same 32-bit load. You can reuse `_ld_global_f32` directly instead of adding a new helper. The spec calls for `ld.global.b32` but `ld.global.f32` produces identical machine code. **Use `_ld_global_f32` (already exists) — do not add a duplicate.**

- [ ] **Step 3: Change wo_global_scale from Python float to kernel ld.global**

In the kernel body, find where `wo_global_scale` is used (it's the `wo_global_scale` parameter in the `_kernel` method, currently a Python float). Change the parameter type in the `_kernel` signature from float to `Int64` (pointer), then read the value inside the kernel:

In the kernel body, after the Phase B entry point (where `wo_global_scale` is first used), add:
```python
# Read wo_global_scale from tensor pointer (graph-safe, no .item())
wo_gs_val = _ld_global_f32(wo_gs_ptr)
```

Then replace all uses of `wo_global_scale` in the kernel body with `wo_gs_val`.

Find the Phase B dequant section where `wo_global_scale` is multiplied:
```python
# Current: uses wo_global_scale directly (was a Python float)
# Change to: uses wo_gs_val (loaded from tensor pointer)
```

**CRITICAL:** The `wo_global_scale` parameter was previously typed as a float in `_jit_launch`/`_kernel`. It is now `wo_gs_ptr: Int64`. Make sure the parameter name in the kernel signature matches what Task 3 Step 7 changed it to.

- [ ] **Step 4: Add wo_output self-zero (first CTA zeros its rows)**

In the kernel body, at the Phase B entry point (before the atomicAdd accumulation loop), add the self-zero logic. Find the section where Phase B starts writing to `wo_output` via `atomicAdd`:

```python
# Phase B self-zero: first CTA (block_idx.x == 0) writes zero
# to its wo_output rows before accumulating. Subsequent CTAs
# atomicAdd normally. Eliminates external zero_() call.
if cute.arch.block_idx.x == Int32(0):
    # Zero the wo_output row for this sequence
    wo_out_row_base = wo_output_ptr + Int64(
        seq_idx * hidden_dim * Int32(4))  # FP32 = 4 bytes
    # Each thread zeros (hidden_dim / 128) elements
    n_per_thr_zero = hidden_dim // Int32(128)
    zero_start = tid * n_per_thr_zero
    for _zi in cutlass.range_constexpr(40):  # 5120/128 = 40
        zero_idx = zero_start + Int32(_zi)
        _st_global_f32(
            wo_out_row_base + Int64(zero_idx * Int32(4)),
            Float32(0.0))
```

Wait — `range_constexpr(40)` assumes hidden_dim=5120. But the kernel already takes `hidden_dim` as a parameter. For the self-zero, we need a runtime loop since hidden_dim is dynamic:

```python
if cute.arch.block_idx.x == Int32(0):
    wo_out_row_base = wo_output_ptr + Int64(
        seq_idx * hidden_dim * Int32(4))
    n_per_thr_zero = hidden_dim // Int32(128)
    zero_start = tid * n_per_thr_zero
    _i_zero = Int32(0)
    while _i_zero < n_per_thr_zero:
        zero_idx = zero_start + _i_zero
        _st_global_f32(
            wo_out_row_base + Int64(zero_idx * Int32(4)),
            Float32(0.0))
        _i_zero = _i_zero + Int32(1)
```

**NOTE:** `range_constexpr(N)` with N>100 causes OOM in the CuTe DSL compiler (known from Phase C). Use a `while` loop for runtime-variable iteration counts. However, for Qwen3.5-27B `hidden_dim=5120 / 128 threads = 40 elements per thread`, and `range_constexpr(40)` is fine (it's under the ~100 limit). The constexpr version generates faster code. **Decision: use `range_constexpr(40)` since we only target Qwen3.5-27B (hidden_dim=5120).** Add a runtime assert `hidden_dim == 5120` in the Python wrapper.

Actually, looking at the Phase C code which already uses `range_constexpr(5)` for groups of 8 (5*8=40), the self-zero follows the same pattern. Use the group+inner constexpr approach:

```python
if cute.arch.block_idx.x == Int32(0):
    wo_out_row_base = wo_output_ptr + Int64(
        seq_idx * hidden_dim * Int32(4))
    my_start_z = tid * n_per_thr_c  # n_per_thr_c = hidden_dim // 128 = 40
    for _zg in cutlass.range_constexpr(5):
        base_z = my_start_z + Int32(_zg * 8)
        for _zi in cutlass.range_constexpr(8):
            idx_z = base_z + Int32(_zi)
            _st_global_f32(
                wo_out_row_base + Int64(idx_z * Int32(4)),
                Float32(0.0))
```

Place this BEFORE the existing Phase B atomicAdd loop. After the zero, add a `cute.arch.sync_threads()` to ensure all threads in the CTA have finished zeroing before any thread starts accumulating.

- [ ] **Step 5: Add sigmoid gate fusion between attention epilogue and Phase B**

Find the transition point between the attention epilogue (where `attn_out` values are in registers) and the Phase B entry. This is where the kernel writes attention output to SMEM/registers before Phase B reads it for the W_O GEMV.

The gate fusion slots in here:
1. Attention produces `attn_out[head][dim]` in registers (existing)
2. **NEW:** Load `gate[token, head*head_dim + dim]` from global memory
3. **NEW:** Apply `sigmoid(gate) * attn_out` element-wise
4. Phase B reads gated result (existing)

In the kernel body, after the attention output is computed (the warp reduction / cross-warp merge section produces final `o_accum` values), add gate fusion before Phase B reads the attention output:

```python
# Gate fusion: sigmoid(gate) * attn_output
# Slots between attention epilogue and Phase B W_O GEMV entry.
# gate_buf layout: [num_seqs, num_heads * head_dim] BF16
# Each thread owns its own slice of the attention output.
if gate_fused != Int32(0):
    # Compute gate element offset for this thread's output elements
    # q_head_idx = block_idx.x * cta_q + local_q_row
    # gate_base = gate_ptr + seq_idx * num_q_heads * head_dim * 2  (BF16)
    head_offset = q_head_idx * Int32(self.head_dim)

    # For each output element this thread holds (from attention epilogue),
    # load the corresponding gate value, apply sigmoid, multiply.
    # The attention output elements are in o_accum registers.
    # Gate elements are at gate_base + (head_offset + dim_offset) * 2 bytes

    gate_row_base = gate_ptr + Int64(
        seq_idx * num_q_heads * Int32(self.head_dim) * Int32(2))

    # Apply sigmoid(gate) * attn_out for each element
    # sigmoid(x) = 1 / (1 + exp2(-x * LOG2E))
    # LOG2E = 1.4426950408889634
    for _gi in cutlass.range_constexpr(self.num_mma_d):  # 16 for head_dim=256
        dim_off = Int32(_gi * 16) + lane_dim_offset
        gate_addr = gate_row_base + Int64(
            (head_offset + dim_off) * Int32(2))
        gate_f32 = _ld_global_b16_to_f32(gate_addr)
        # sigmoid: rcp(1 + exp2(-x * log2e))
        neg_x_log2e = Float32(0.0) - gate_f32 * Float32(1.4426950408889634)
        exp_val = cutlass.cute.exp2(neg_x_log2e)
        sigmoid_val = _rcp_approx_f32(Float32(1.0) + exp_val)
        # Multiply attention output by sigmoid(gate)
        # o_accum[_gi] *= sigmoid_val
```

**NOTE:** The exact register names for attention output (`o_accum`) and the loop structure depend on where in the kernel body the attention output is finalized. The implementer must:
1. Find where the attention output values are finalized in registers (after cross-warp merge)
2. Insert the gate multiply before those values flow into Phase B
3. Use the same register variable names for the attention output

The gate operates on the same head×dim grid as the attention output, so the indexing follows the existing attention output layout.

- [ ] **Step 6: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/kernel.py
git commit -m "feat(cuda-graphs): kernel self-zero, gate fusion, rcp PTX, tensor ptr wo_global_scale

Kernel body changes:
- rcp.approx.ftz.f32 PTX helper for sigmoid
- wo_output self-zero: CTA 0 zeros before atomicAdd (no external memset)
- sigmoid(gate) * attn_output between attention epilogue and Phase B
- wo_global_scale read via ld.global.f32 from tensor pointer

Blocker #9 (gate fusion), plus self-zero and tensor pointer."
```

---

### Task 5: Model layer — Qwen3NextAttention + Qwen3NextDecoderLayer fusion wiring

**Files:**
- Modify: `vllm/model_executor/models/qwen3_next.py:201-454`

Wire the fusion into the actual model class used by Qwen3.5-27B. This is where the Phase B+C fusion finally works on the correct model (previously only wired in qwen2.py which is the WRONG class).

- [ ] **Step 1: Add fusion binding to `Qwen3NextDecoderLayer.__init__`**

The decoder layer has access to all fusion weights:
- `self.self_attn.o_proj` — W_O projection (NVFP4 quantized)
- `self.post_attention_layernorm` — RMSNorm gamma/eps

After the existing `__init__` body (~line 395), add a deferred binding flag:

```python
# Fusion binding happens in _try_bind_fusion() after weights are loaded.
# We check on first forward because weights aren't available during __init__.
self._fusion_bound = False
```

- [ ] **Step 2: Add `_try_bind_fusion()` method to `Qwen3NextDecoderLayer`**

After `__init__`, add:

```python
def _try_bind_fusion(self) -> bool:
    """Attempt to bind CuTe fusion weights. Returns True if successful.

    Called lazily on first forward. Requires:
    1. This is a full_attention layer (not linear_attention)
    2. The attention backend is CutePagedAttentionImpl
    3. Weights have been loaded (o_proj has quantized weight attributes)
    """
    if self.layer_type != "full_attention":
        return False

    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )
    impl = self.self_attn.attn.impl
    if not isinstance(impl, CutePagedAttentionImpl):
        return False

    o_proj_linear = self.self_attn.o_proj
    # Access the quantized weight through the linear layer's quant_method
    # After process_weights_after_loading, NVFP4 layers have:
    #   .weight (packed uint8), .weight_scale (fp8),
    #   .weight_global_scale (fp32 scalar)
    quant_layer = o_proj_linear.linear_method
    if not hasattr(o_proj_linear, 'weight_global_scale'):
        logger.warning(
            "CuTe fusion: o_proj weights not loaded yet or not NVFP4, "
            "skipping fusion binding for layer %d", self.layer_idx)
        return False

    vllm_config = get_current_vllm_config()
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs

    impl.bind_fusion_weights(
        wo_weight=o_proj_linear.weight,
        wo_scales=o_proj_linear.weight_scale,
        wo_global_scale=o_proj_linear.weight_global_scale,
        rmsnorm_gamma=self.post_attention_layernorm.weight,
        rmsnorm_eps=self.post_attention_layernorm.variance_epsilon,
        max_num_seqs=max_num_seqs,
    )

    logger.info(
        "CuTe fusion bound for layer %d (full_attention)", self.layer_idx)
    return True
```

**NOTE:** The exact attribute names on the quantized linear layer (`weight`, `weight_scale`, `weight_global_scale`, `variance_epsilon`) must be verified by the implementer. The NVFP4 ModelOpt quantization layer stores weights with these names after `process_weights_after_loading()` runs (see `vllm/model_executor/layers/quantization/modelopt.py:1166-1197`). The RMSNorm eps may be `variance_epsilon` or `eps` depending on the norm class — check `GemmaRMSNorm` (aliased as `Qwen3NextRMSNorm`).

- [ ] **Step 3: Modify `Qwen3NextAttention.forward()` for fusion path**

Replace lines 280-315:

```python
def forward(
    self,
    positions: torch.Tensor,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    fusion_active: bool = False,
):
    qkv, _ = self.qkv_proj(hidden_states)

    if self.attn_output_gate:
        q_gate, k, v = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
        )
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig_shape, -1)
        gate = gate.reshape(*orig_shape, -1)
    else:
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate = None

    q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
        -1, self.num_heads * self.head_dim
    )
    k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
        -1, self.num_kv_heads * self.head_dim
    )

    q, k = self.rotary_emb(positions, q, k)

    if fusion_active and gate is not None:
        # Write gate to impl's persistent buffer for kernel fusion.
        # Kernel does: sigmoid(gate) * attn → W_O GEMV → RMSNorm
        num_tokens = hidden_states.shape[0]
        self.attn.impl.gate_buf[:num_tokens].copy_(gate[:num_tokens])

    attn_output = self.attn(q, k, v)

    if not fusion_active:
        # Unfused path: apply gate and o_proj in Python
        if self.attn_output_gate and gate is not None:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate
        output[:], _ = self.o_proj(attn_output)
    # When fusion_active, kernel already wrote to impl's persistent
    # buffers (wo_output, rmsnorm_output, residual_output).
    # Caller reads from those buffers directly.
```

- [ ] **Step 4: Modify `Qwen3NextDecoderLayer.forward()` for fusion path**

Replace lines 397-454:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor = None,
    **kwargs: object,
):
    # Lazy fusion binding on first forward (weights loaded by now)
    if not self._fusion_bound:
        self._fusion_bound = self._try_bind_fusion()

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual)

    fusion_active = (
        self._fusion_bound
        and self.layer_type == "full_attention"
    )

    if fusion_active:
        # Write residual to impl's persistent buffer for Phase C.
        # Kernel reads this for: new_residual = residual + wo_output
        num_tokens = hidden_states.shape[0]
        impl = self.self_attn.attn.impl
        impl.residual_buf[:num_tokens].copy_(residual[:num_tokens])

    self_attention_output = torch.empty_like(hidden_states)

    if self.layer_type == "linear_attention":
        self.linear_attn(
            hidden_states=hidden_states,
            output=self_attention_output,
        )
        hidden_states = self_attention_output
    elif self.layer_type == "full_attention":
        self.self_attn(
            hidden_states=hidden_states,
            output=self_attention_output,
            positions=positions,
            fusion_active=fusion_active,
        )
        if fusion_active:
            # Kernel produced: rmsnorm_output (hidden_states for MLP)
            # and residual_output (updated residual for next layer).
            # Skip post_attention_layernorm — kernel did it.
            hidden_states = impl.rmsnorm_output[:num_tokens]
            residual = impl.residual_output[:num_tokens]
        else:
            hidden_states = self_attention_output
    else:
        raise ValueError("Invalid layer_type")

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        else:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype) + 1
            )

    if not fusion_active:
        # Unfused path: apply post_attention_layernorm in Python
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
    # When fusion_active, Phase C already did residual add + RMSNorm

    hidden_states = self.mlp(hidden_states)

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        else:
            assert len(hidden_states.shape) == len(
                self.ffn_layer_scale.shape
            ), (
                f"shape must be the same {len(hidden_states.shape)}, "
                f"{len(self.ffn_layer_scale.shape)}"
            )
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype) + 1
            )

    return hidden_states, residual
```

**Key changes:**
- Lazy fusion binding on first forward (weights guaranteed loaded)
- `fusion_active` flag controls path: fused (kernel handles gate+W_O+RMSNorm) vs unfused (Python handles them)
- Residual written to impl's persistent buffer before attention
- After attention: read fused outputs from impl's persistent buffers
- Skip `post_attention_layernorm` when fusion active — kernel did it in Phase C

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/qwen3_next.py
git commit -m "feat(cuda-graphs): wire CuTe fusion into Qwen3NextAttention/DecoderLayer

Fusion wiring for the correct model class (Qwen3NextAttention, not
Qwen2Attention). Lazy bind_fusion_weights() on first forward:
- Gate written to impl.gate_buf before attention
- Residual written to impl.residual_buf before attention
- After attention: read rmsnorm_output and residual_output from impl
- Skip o_proj and post_attention_layernorm when fused

This is the fix for the 4-commit model class mixup."
```

---

### Task 6: Integration validation — eager → PIECEWISE → FULL_AND_PIECEWISE

**Files:**
- Modify: `scripts/serve-cute.sh` (compilation config)
- Use: `scripts/gsm8k_sanity.py` (existing test script)

Four-step validation sequence. Each step must pass before advancing to the next.

- [ ] **Step 1: GSM8K sanity — eager mode (fusion wiring test)**

Launch with `--enforce-eager` to test the fusion wiring without any graph capture:

```bash
./scripts/serve-cute.sh --debug
```

Wait for model to load, then run GSM8K:
```bash
python3 scripts/gsm8k_sanity.py --base-url http://localhost:8000/v1
```

**Expected:** 8/8 correct (100%). This confirms:
- bind_fusion_weights() successfully binds
- Gate fusion (sigmoid * attn_output) produces correct results
- W_O GEMV with self-zero produces correct results
- RMSNorm with tensor-pointer wo_global_scale produces correct results
- Residual stream through persistent buffers is correct

**If this fails:** The bug is in the fusion wiring or kernel body changes. Debug with print statements in the kernel dispatch (all Python-level code runs in eager mode). Check that gate values, wo_output, rmsnorm_output match the unfused path.

- [ ] **Step 2: GSM8K sanity — PIECEWISE mode**

Update `scripts/serve-cute.sh` to use PIECEWISE compilation (attention stays eager, rest of model compiled):

```bash
# In serve-cute.sh, change --compilation-config level to PIECEWISE
# or pass via env var:
VLLM_TORCH_COMPILE_LEVEL=1 ./scripts/serve-cute.sh
```

Run GSM8K again:
```bash
python3 scripts/gsm8k_sanity.py --base-url http://localhost:8000/v1
```

**Expected:** 8/8 correct. This confirms the refactored forward path doesn't break torch.compile's piecewise compilation.

- [ ] **Step 3: GSM8K sanity — FULL_AND_PIECEWISE mode (the real test)**

This is the target configuration. Update `scripts/serve-cute.sh` compilation config:

```bash
# serve-cute.sh should use:
# --compilation-config '{"level": 3}'  # FULL_AND_PIECEWISE
./scripts/serve-cute.sh
```

Run GSM8K:
```bash
python3 scripts/gsm8k_sanity.py --base-url http://localhost:8000/v1
```

**Expected:** 8/8 correct. This confirms:
- CUDA graph capture works (all persistent buffers at stable addresses)
- Graph replay produces correct decode output
- build_for_cudagraph_capture fills seq_lens=1 correctly
- Padding CTAs with seq_len=1 produce ignored results
- self-zero eliminates need for external wo_output.zero_()

**If this fails:** Common causes:
1. Tensor not at stable address — check that all kernel pointer args come from persistent buffers
2. Python branching during capture — ensure `_fusion_bound` is constant True for all captured layers
3. Missing sync — add `cute.arch.sync_threads()` after self-zero
4. wo_output not zeroed — check self-zero CTA guard condition

- [ ] **Step 4: Multi-turn conversation test**

Verify graph replay across multiple turns with growing seq_lens:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 64}'

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Write a haiku about GPU programming"}], "max_tokens": 128}'

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Explain PagedAttention in one paragraph"}], "max_tokens": 256}'
```

**Expected:** Coherent, correct responses for all three. Different batch sizes may hit different captured graph sizes.

- [ ] **Step 5: Update `scripts/serve-cute.sh` default compilation config**

Once validated, update the script to default to FULL_AND_PIECEWISE:

```bash
# Change the --compilation-config in serve-cute.sh from level 1 to level 3
# --compilation-config '{"level": 3}'
```

- [ ] **Step 6: nsys trace — measure decode latency improvement**

Profile decode with `--privileged` to get CUPTI injection. Compare against the most recent eager-mode trace.

```bash
# Inside container with --privileged:
nsys profile -t cuda,nvtx -o /workspace/traces/cuda_graphs_decode \
  --process-scope process-tree \
  python3 -m vllm.entrypoints.openai.api_server \
    --model natfii/Qwen3.5-27B-NVFP4-Opus-GB10 \
    --served-model-name default \
    --attention-backend CUTE_PAGED \
    --compilation-config '{"level": 3}' \
    ...
```

After warmup, send decode-heavy requests and capture the trace. Compare kernel launch overhead vs eager mode. Commit trace to `benchmarks/nvllm/traces/cute_paged_attn/YYYY-MM-DD-cuda-graphs/`.

- [ ] **Step 7: Commit script and trace**

```bash
git add scripts/serve-cute.sh
git commit -m "feat(cuda-graphs): enable FULL_AND_PIECEWISE for CuTe paged attention

Validated via GSM8K (8/8) across eager, PIECEWISE, and
FULL_AND_PIECEWISE modes. Multi-turn conversation test passed.

CUDA graph pipeline state refactor complete:
- Persistent buffers on CutePagedAttentionImpl
- bind_fusion_weights() replaces side-channel pattern
- Graph-safe kernel dispatch (no .item(), no dynamic alloc)
- Output gate fusion for Qwen3NextAttention
- Fusion wired into correct model class (qwen3_next.py)"
```

---

## Post-Ship (not part of this plan)

These items are documented in the spec's "Post-Ship Cleanup" section and execute
AFTER all 6 tasks above are validated:

1. **Seal `qwen2.py` fusion code as `.ref`** — the old Phase B+C wiring is dead code for Qwen3.5-27B.
   The Claude hook currently blocks all edits to `qwen2.py`; after sealing, relax the hook.
2. **Archive old unfused benchmark traces** to `historical/` subdirectory.
3. **Update project memory** — mark CUDA graphs as shipped.

---

## Appendix: Blocker Checklist Cross-Reference

| Blocker | Task | Step |
|---------|------|------|
| #1 — `wo_global_scale.item()` | Task 3 | Step 3 |
| #2 — `torch.empty_like(query)` | Task 3 | Step 2 |
| #3 — `grid.z = num_seqs` (exact) | Task 3 | Step 4 |
| #4 — `query.contiguous().view(-1)` | Task 3 | Step 1 |
| #5 — Python branching on `wo_weight is None` | Task 3 | Step 5 |
| #6 — Side-channel set/clear cycle | Task 2 | Step 3 |
| #7 — `_arrival_count_buf` lazy growth | Task 2 | Step 1 |
| #8 — `torch.zeros()` for wo_output each call | Task 2 | Step 1 |
| #9 — Output gate fusion | Task 4 | Step 5 |
| #10 — `build_for_cudagraph_capture` | Task 1 | Step 1 |

## Appendix: Implementer Notes

**CuTe DSL gotchas (from project memory):**
- `range_constexpr(N>100)` OOMs the compiler — use while loops for large N
- Multi-dim tensors must be `.view(-1)` before kernel launch (flat indexing)
- `inline_asm` takes positional IR values — no keyword args, single-line braces for PTX inside dynamic ifs
- CuTe DSL cannot read uint8 tensors directly — use `ld.global` via `data_ptr()`

**Weight attribute names (NVFP4 ModelOpt after process_weights_after_loading):**
- `layer.weight` — packed NVFP4 [N, K/2] uint8
- `layer.weight_scale` — per-block scales [N, K_sf]
- `layer.weight_global_scale` — scalar fp32 Parameter

**RMSNorm attribute (GemmaRMSNorm aliased as Qwen3NextRMSNorm):**
- Check for `self.weight` (gamma) and either `self.variance_epsilon` or `self.eps`

**Graph cache invalidation:**
- Changing kernel signatures invalidates the CuTe compilation cache
- Delete `/root/.cache/vllm/torch_compile_cache` inside the container after kernel changes
- Delete `__pycache__` directories under `vllm/v1/attention/backends/cute_paged/`

**Testing command:**
```bash
# GSM8K sanity (8 questions, fast)
python3 scripts/gsm8k_sanity.py --base-url http://localhost:8000/v1

# Manual single-query test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "prompt": "The capital of France is", "max_tokens": 16}'
```
