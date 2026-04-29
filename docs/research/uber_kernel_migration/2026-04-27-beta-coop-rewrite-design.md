# β-coop framework-output-buffer rewrite — design spec

**Date:** 2026-04-27
**Branch:** `feat/uber-kernel-migration` HEAD `7d429f1b7`
**Savepoint:** `feat/pre-beta-coop-rewrite-savepoint` at `7d429f1b7`
**Supersedes:** `docs/research/uber_kernel_migration/2026-04-27-beta-coop-framework-output-rewrite.md` (initial 510-line plan, rejected by spec-audit on 6 CRITICAL findings — kept for context)

## TL;DR

β-coop's outputs are currently consumed via Python attribute access (`impl.rmsnorm_output`, `impl.residual_output`, `impl.mlp_output`), which is invisible to torch.compile dynamo and causes (a) `cute_residual_mirror` to be DCE'd from the captured graph, and (b) the consume gate to specialize at trace time. Symptom: gibberish output (`这种现象 × 256`) under PIECEWISE+CUDA graphs.

The rewrite makes β-coop's outputs **framework-supplied tensors** that the layer pre-allocates and consumes as graph-tracked nodes. New thin-wrapper op `cute_beta_coop_run` mirrors `unified_attention_with_output` exactly. `_backend.forward` is refactored to write through caller-supplied output kwargs. The Python pipeline (post-attn-LN, MLP) for the fall-through path absorbs into `_backend.forward`. No more `cute_residual_mirror`, no more `cute_phase_e_dispatch`, no more `if _fusion_active` consume gate.

The pattern is verified production-grade in the same codebase (every attention backend uses it under PIECEWISE+CUDA graphs).

## Problem

Three stacked failures motivate the rewrite. All three were reproduced empirically on 2026-04-27:

1. **`cute_residual_mirror` DCE'd** — `vllm/v1/attention/backends/cute_paged/_mlp_op.py:266-311` (commit `7d429f1b7`). Despite `mutates_args=["residual_buf"]`, dynamo's reachability analysis sees no graph-level reader of `residual_buf` (downstream readers access it via `self.residual_buf` inside opaque op bodies — invisible). Verified empirically: 256-token decode emitted `0` `[RES_MIRROR_OP]` log lines despite β-coop firing.

2. **β-coop reads zeros at runtime** — `_backend.py:1271` (commit `7d429f1b7`) reads `self.residual_buf[:nat]` as `residual_in`. Because the mirror op was DCE'd, `residual_buf` stays at the CUDA-graph-allocator-zeroed value. β-coop computes `attn_out + 0` instead of `attn_out + (x_n + r_n)`. Cascade through 64 layers → Chinese-character spam.

3. **Consume-gate DCE under fullgraph** — the `if getattr(impl, "_fusion_active", False)` gate at `qwen3_5.py:480` (commit `7d429f1b7`) is specialized to "always-take-else" by dynamo at trace time. The 514b88c6f B-fix attempted to repair this with phantom-input ops; it produced coherent output under `PIECEWISE+cudagraph_mode=NONE` but gibberish under `PIECEWISE+cudagraph_mode=PIECEWISE` for unanalyzed reasons. Three pivots failed in that session — `.item()`, registry-lookup of `_phase_e_use_beta_coop`, registry-lookup of `_fusion_bound`. All three round-trip through Python state.

The phantom-input pattern fixes DCE but not the deeper issue: **β-coop's outputs are consumed via Python attribute access**, which Dynamo can't see. The fix is to stop using `impl.*` for runtime-varying tensors entirely.

Full post-mortem: `docs/research/uber_kernel_migration/2026-04-26-consume-gate-dce-and-graph-capture.md`. Empirical reproduction: `docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md`.

## Architectural answer (three decisions)

### Q1 — Per-step dispatch shape: match `unified_attention_with_output` as a PIECEWISE *splitting boundary*

**Conceptual reframe (corrected 2026-04-27 post-review):** `unified_attention_with_output` is in vLLM's `_attention_ops` list at `vllm/config/compilation.py:713-727` (commit `7d429f1b7`), which seeds the default `splitting_ops` at `compilation.py:1068` when user config doesn't override (our `scripts/serve-cute.sh` doesn't override — verified). Splitting ops are **PIECEWISE graph boundaries**: torch.compile splits the FX graph at these op calls. The op runs as **eager Python at runtime, between captured graph segments** — its body executes on every decode step, not once at capture.

This is materially different from a custom op captured *inside* a graph segment (which is the trap that bit `cute_residual_mirror` and the C2 diag — Python body runs once at capture, never at replay). For the new op to behave correctly, **`vllm::cute_beta_coop_run` MUST be added to `_attention_ops`** so torch.compile splits the graph at it.

With that registration:
- Pre-op graph segment captures + replays for `input_layernorm + qkv_proj + q_norm + k_norm + rotary + gate compute`.
- Op body runs **eager Python every step**: `get_attention_context(layer_name)` → reads real per-step `attn_metadata` → dispatches β-coop or fall-through.
- Post-op graph segment captures + replays for `consume + (skipped) post_attn_LN + decoder return`.

The op body's Python branch on `attn_metadata` resolves correctly per call — `attn_metadata` is freshly constructed every step from real `forward_context`, not frozen at capture. Per shape:

| Runtime shape | β-coop fits (`64·N ≤ 96`)? | Op body dispatch |
|---|---|---|
| `num_seqs=1` | ✓ | Calls `run_beta_coop_full(... → framework outputs)` |
| `num_seqs=2,4,8` | ✗ | Calls fall-through path (paged in fusion mode → β-lite Phase D MLP → framework outputs) |

This matches `unified_attention_with_output`'s production pattern exactly: the op is a runtime dispatch point between captured pieces, NOT a captured-time decision baked per shape. The reason production vLLM works under PIECEWISE+CUDA graphs is precisely this — every attention op runs eager at runtime via `get_attention_context`.

**`build_for_cudagraph_capture` at `_backend.py:1753-1767`** is still relevant for the *kernel-launch* side: when β-coop's CUDA ops execute, they capture into the post-op segment's recorded sequence (or into β-coop's own internal cooperative-launch). It's NOT what makes the op-body dispatch replay-safe; the splitting-op registration is.

### Q2 — Fall-through semantics: refactor `_backend.forward` + thin-wrapper op

Production vLLM uses thin-wrapper ops everywhere. `unified_attention_with_output` (`vllm/model_executor/layers/attention/attention.py:713-760`, commit `7d429f1b7`) is 5 lines that delegate to `self.impl.forward(...)`. Every backend (FA2/FA3, Triton, FlashInfer, MLA) inherits this pattern. PR #16756 established the precedent that "when a backend can fuse more, push the work down into impl.forward."

Our new op follows the same shape:

```python
def cute_beta_coop_run(query, key, value, residual, attn_input, gate,
                       output_rmsnorm, output_residual, output_mlp, layer_name):
    attn_metadata, layer, kv_cache, _ = get_attention_context(layer_name)
    layer.impl.forward(layer, query, key, value, kv_cache, attn_metadata,
                       residual=residual, attn_input=attn_input, gate=gate,
                       output_rmsnorm=output_rmsnorm,
                       output_residual=output_residual,
                       output_mlp=output_mlp)
```

`_backend.forward` (refactored) becomes the dispatcher. β-coop branch writes the three outputs directly via `run_beta_coop_full(... attn_output=output_rmsnorm, residual_output=output_residual, mlp_output=output_mlp)`. Fall-through branch absorbs the Python pipeline currently at `qwen3_5.py:482-507` (paged + post-attn-LN + Phase D MLP) and writes through the same three outputs.

No reentrancy: `_backend.forward` is no longer called from outside this op chain. The pre-existing `unified_attention_with_output` path is unchanged for non-fused layers and other backends.

### Q3 — Call site: inside `Qwen3_5Attention.forward`, trace-time branch on a stable post-weight-load flag

The branch predicate must be true ONLY when β-coop's framework-output path is fully wired. Naive use of `_fusion_bound` is too broad — `_fusion_bound` (attn fusion) and `_mlp_fusion_bound` (Phase D MLP fusion, gated by `CUTE_MLP_FUSION` env, default OFF per `_backend.py:699-704`) are independent. Skipping Python MLP whenever attn fusion is bound would skip the only valid MLP path when MLP fusion is off.

**Stricter predicate:** introduce a single flag `impl._beta_coop_framework_output_bound: bool`. Compute it at the **end of `_resolve_mlp_weights()`** (which runs after `_resolve_fusion_weights()` per the call sequence at `_backend.py:1681-1682`):

```python
self._beta_coop_framework_output_bound = (
    self._fusion_bound
    and self._mlp_fusion_bound
    and self._phase_e_coop_kernel is not None
)
```

This naming reflects what it gates: the *decoder-layer Python control flow* that pre-allocates framework outputs and routes to `cute_beta_coop_run`. Setting it after weight resolution avoids the trap of freezing it `False` before `_resolve_*_weights()` flips the underlying flags True at `_backend.py:657, 950`.

Call site:

```python
# In Qwen3_5Attention.forward, after qkv_proj + q/k_norm + rotary + gate compute:
if getattr(impl, "_beta_coop_framework_output_bound", False):
    # Caller (DecoderLayer) pre-allocates output_residual, output_mlp; passes via kwargs.
    # `output` is reused as output_rmsnorm.
    kv_cache_dummy_dep = unified_kv_cache_update(k, v, self.attn.layer_name)
    torch.ops.vllm.cute_beta_coop_run(
        q, k, v, residual, attn_input, gate,
        output, output_residual, output_mlp,
        self.attn.layer_name,
        kv_cache_dummy_dep=kv_cache_dummy_dep,
    )
else:
    self.attn(q, k, v)  # canonical unified_attention_with_output path (handles its own KV update + dummy_dep)
```

Notes:
- `self.attn.layer_name` (not `self.layer_name`) — the inner `Attention` instance owns `layer_name` per `vllm/model_executor/layers/attention/attention.py:280`.
- `unified_kv_cache_update + kv_cache_dummy_dep` is preserved before the op call because `CutePagedBackend.forward_includes_kv_cache_update = False` at `_backend.py:124` — KV update is external to the impl.
- `kv_cache_dummy_dep` is declared as a `Tensor | None = None` parameter on the new op; `mutates_args` does NOT list it; body does `del kv_cache_dummy_dep` exactly like `unified_attention_with_output` at `attention.py:726`.

Each layer's specialization is fixed at compile time on the stable post-weight-load flag.

`Qwen3_5Attention` is fork-specific code (`vllm/nvllm/models/qwen3_5.py`). Modifying it has no upstream review/merge cost.

## Components

### 1. `vllm/v1/attention/backends/cute_paged/_beta_coop_op.py` (NEW)

| | |
|---|---|
| Purpose | Single dispatch op that gives β-coop's outputs graph-tracked tensor identity |
| Op name | `torch.ops.vllm.cute_beta_coop_run` |
| Body | 5 lines: `get_attention_context(layer_name)` + delegate to `layer.impl.forward(...)` with output kwargs |
| `mutates_args` | `["output_rmsnorm", "output_residual", "output_mlp"]` (NOT `kv_cache_dummy_dep` — it's a phantom dep; body does `del kv_cache_dummy_dep`) |
| Counter hook | `_BETA_COOP_FIRE_COUNTER: dict[str, int]` incremented at top of body for empirical replay verification (mirrors `tests/compile/silly_attention.py:50` global counter) |
| Fake impl | Returns `None` |
| Splitting-op registration | Add `"vllm::cute_beta_coop_run"` to `vllm/config/compilation.py:_attention_ops` (the ClassVar list at L713) so torch.compile splits the FX graph at the call. Without this the op runs once at capture (in-graph trap from `feedback_op_body_capture_only`); with it, body runs eager every step (`feedback_splitting_op_runtime_dispatch`). |
| Side-effect import | Add `import vllm.v1.attention.backends.cute_paged._beta_coop_op  # noqa: F401` at the top of `vllm/nvllm/models/qwen3_5.py` (mirrors the existing `_mlp_op` import at `vllm/nvllm/layers/mlp.py:21`) so the op exists at torch.compile trace time. |

### 2. `vllm/v1/attention/backends/cute_paged/_backend.py:993+` (REFACTORED)

| | |
|---|---|
| Purpose | Backend dispatcher and workhorse: β-coop fast path OR fall-through, all writes go to caller-supplied tensors |
| New required kwargs | `residual: Tensor`, `attn_input: Tensor`, `gate: Tensor`, `output_rmsnorm: Tensor`, `output_residual: Tensor`, `output_mlp: Tensor` |
| β-coop branch | `run_beta_coop_full(... attn_output=output_rmsnorm, residual_output=output_residual, mlp_output=output_mlp)` — replaces the `self.rmsnorm_output / self.residual_output / self.mlp_output` writes at `_backend.py:1283-1305` (commit `7d429f1b7`) |
| Fall-through branch | (1) `paged_attention_forward(...)` in fusion mode (passes `gate_buf=gate, rmsnorm_output=output_rmsnorm, residual_output=output_residual`) — writes attn+gate+W_O+post-attn-LN through framework outputs; (2) β-lite Phase D MLP launch — see "β-lite side-channel elimination" below — writes `output_mlp`. |
| Retired internally | `self.rmsnorm_output / residual_output / mlp_output` *output* uses (the tensor allocations stay as scratch for now; cleanup in Phase 5) |

**β-lite side-channel elimination (rewiring at `_backend.py:1410-1450` — current β-lite kernel call site):**

The current β-lite call passes `self.rmsnorm_output` as input, `self.residual_buf` as `residual_post_ln`, `self.mlp_output` as output. This is the same hidden side-channel pattern we're escaping for β-coop. Rewire the launch to pass framework tensors:

```python
# In _backend.forward fall-through, after paged_attention_forward writes output_rmsnorm/output_residual:
self._mlp_kernel(
    output_rmsnorm[:nat],      # input — was self.rmsnorm_output
    self._mlp_gate_w, ...,      # weights stay on impl (constant post-attach)
    output_mlp[:nat],          # output — was self.mlp_output
    nat,
    residual_post_ln=output_residual[:nat],  # was self.residual_buf
    next_input_layernorm_gamma=_next_gamma,
    next_hidden_output=self.next_hidden_scratch[:nat],  # internal scratch — kept on impl, not consumed by Python
    ...
)
```

`self.next_hidden_scratch` stays as impl-internal scratch because it's intra-kernel state for Phase 4 (and Phase 4 is deleted per `project_own_the_stack` C1.5 — but the kernel arg is still required by the existing signature). It's never consumed by Python downstream, so DCE doesn't apply. Verify by inspecting captured FX after Phase 4 build that no `cute_residual_mirror` / `cute_phase_e_dispatch` calls survive.

### 3. `vllm/nvllm/models/qwen3_5.py:240+` and `:440+` (MODIFIED)

| | |
|---|---|
| `Qwen3_5Attention.forward` signature | Extend with three new optional kwargs: `output_residual: Tensor \| None = None, output_mlp: Tensor \| None = None` (existing `output: Tensor` is reused as `output_rmsnorm`). All three are mutated in place when fusion is bound. No return-value change — preserves the canonical "caller pre-allocates, callee writes through" contract. |
| `Qwen3_5Attention.forward` body | After QKV+q_norm+k_norm+rotary+gate (unchanged), branch on `getattr(impl, "_fusion_bound", False)`. If True: assert all three outputs were passed; call `cute_beta_coop_run(... output_rmsnorm=output, output_residual=output_residual, output_mlp=output_mlp)`. If False: assert `output_residual is None and output_mlp is None`; keep existing `self.attn(q, k, v)` path. |
| `Qwen3_5DecoderLayer.forward` | When fusion is bound, pre-allocate `output_residual` and `output_mlp` via `torch.empty_like(residual)` (the existing `self_attention_output` already serves as `output`/`output_rmsnorm`). Pass all three to `self.self_attn(...)`. After the call, assign `hidden_states = output_mlp` and `residual = output_residual`; skip the existing post-attn-LN + MLP blocks. |
| Retired locally | The `cute_residual_mirror(impl.gate_buf, gate)` call (gate is now a passed op arg). The `cute_residual_mirror(impl.residual_buf, residual)` call at line 460 (residual is a graph tensor passed directly to the op). The Python copy/consume block at lines 482-490 (no longer needed). |

## Data flow

**Key conceptual point:** because `vllm::cute_beta_coop_run` is registered in `_attention_ops` (Q1), torch.compile splits the FX graph at the op call. The op runs as **eager Python at runtime, between captured graph segments** — body executes on every decode step. This is structurally identical to `unified_attention_with_output` and unlike a custom op captured inside a graph (the trap from `feedback_op_body_capture_only`).

### Capture (PIECEWISE warmup, per shape)

For each captured shape `num_tokens ∈ {1, 2, 4, 8}`:
1. **Pre-op segment captured:** `input_layernorm + qkv_proj + q_norm + k_norm + rotary + gate compute` — these are normal torch ops, recorded into a CUDA graph for this shape.
2. **Op call boundary:** torch.compile records "call `vllm::cute_beta_coop_run`" as a graph break. No body execution yet.
3. **Post-op segment captured:** `consume + (skipped) post_attn_LN + decoder return` — also recorded into a CUDA graph for this shape.
4. (`build_for_cudagraph_capture` at `_backend.py:1753-1767` ensures the kernel-launch side sees real `attn_metadata` if any in-op capture happens during warmup — relevant only if β-coop's cooperative-launch internally records into a sub-graph.)

### Runtime decode step

For each runtime decode call at shape `num_tokens=N`:
1. Pre-op CUDA graph for shape N replays: input_LN + QKV + q/k_norm + rotary + gate.
2. **`cute_beta_coop_run` body runs eager Python:**
   - `get_attention_context(layer_name)` → fetches *real, fresh* per-step `attn_metadata` (not frozen from capture).
   - Delegates to `_backend.forward(...)` with framework output kwargs.
   - `_backend.forward` evaluates `_use_beta_coop` gate against current `attn_metadata.is_decode_only / seq_lens / num_seqs` — fresh decision per step.
   - β-coop fits → `run_beta_coop_full(...)` launches its kernels (cooperative launch may capture internally; that's β-coop's concern).
   - β-coop doesn't fit → `paged_attention_forward(...)` + β-lite MLP launch — kernels execute against framework output tensors directly.
3. Post-op CUDA graph for shape N replays: decoder layer continues with `hidden_states=output_mlp, residual=output_residual` (graph tensors).

**Why this fixes the gibberish:**
- β-coop reads `residual` directly from a graph-tracked op input (not `impl.residual_buf` filled by a DCE'd mirror op).
- The dispatch decision is a runtime Python evaluation against fresh `attn_metadata`, not a capture-time-frozen branch.
- Framework outputs are written through caller-owned tensors that the next captured graph segment reads directly — no Python-attribute round-trip anywhere.

## Error handling

Per `feedback_no_silent_fallbacks`:

1. **No `try/except` fallback** anywhere in the new op or `_backend.forward`. The fall-through branch is the complete-coverage path. If β-coop's *kernel* crashes (resource error, shape mismatch, cooperative-launch fitness violation despite gate), let the exception propagate.

2. **Strict dtype/shape preconditions at op entry** (`assert`, fail-loud):
   - All output/input tensors share `shape == [nat, hidden]`.
   - All are `torch.bfloat16`, contiguous, on the same device.
   - These should never trip in production — they catch shape-drift bugs during phase-by-phase rebuild.

3. **Retired `cute_residual_mirror` raises `NotImplementedError`** for one cycle (Phase 5 cleanup deletes it). If anything still calls it, fail loud.

## Testing & verification gates

Five phases, each with explicit pass criteria. Don't proceed until the gate passes.

### Phase 1 — Counter-pattern harness (no production code, no rebuild)

Create `tests/v1/cute_paged/test_beta_coop_op_skeleton.py` mirroring `tests/compile/silly_attention.py` + `tests/compile/fullgraph/test_simple.py:146-156` (commit `7d429f1b7`). Captures under `CUDAGraphMode.PIECEWISE` for sizes `[1, 2, 4, 8]`, replays N times per shape, asserts counter advances exactly N×4.

**Gate:** counter advances exactly N×4 per N replays per shape. If this fails, our entire architectural premise is wrong — STOP.

### Phase 2 — Op registration (rebuild #1)

Add `_beta_coop_op.py` with full skeleton; body initially raises `NotImplementedError("Phase 2 stub")`.

**Gate:** op visible at `torch.ops.vllm.cute_beta_coop_run`; fake impl returns `None`.

### Phase 3 — Refactor + layer rewrite, fall-through-forced (rebuild #2)

Refactor `_backend.forward` to take output kwargs. Modify `Qwen3_5Attention.forward` to call `cute_beta_coop_run` when `_fusion_bound`. Hard-wire `_use_beta_coop=False` inside `_backend.forward` for this phase only.

**Gate:** engine starts; coherent output on standard probe; GSM8K-50 ≥ 90% (per `feedback_post_quant_sanity`, `scripts/gsm8k_eval_50.py`).

### Phase 4 — β-coop dispatch enabled (rebuild #3) — moment of truth

Remove the Phase 3 force-fall-through. Live `_use_beta_coop` gate. Run with `CUTE_PHASE_E_FUSION=1 MAX_NUM_SEQS=1`.

**Gates (all required):**
- Engine starts; no `cudaErrorStreamCaptureInvalidated` at warmup.
- Coherent output (no `这种现象` loop) on 256-token completion at temperature=0.
- `_BETA_COOP_FIRE_COUNTER[layer_name]` ≈ 256 per fusion-bound layer after 256-token completion.
- GSM8K-50 ≥ 90%.
- nsys trace shows β-coop kernel actually launching at decode (not silent fall-through).

If gibberish persists at Phase 4: the suspected secondary cause from `2026-04-26-consume-gate-dce-and-graph-capture.md` (β-coop's cooperative-launch + atomic-counter spin-wait CUDA-graph replay quirks) is real. The architectural rewrite is *correct*; the kernel itself needs separate investigation.

### Phase 5 — Retire deprecated ops + commit (rebuild #4)

Delete `cute_residual_mirror`, `cute_phase_e_dispatch`, the Python copy/consume block at `qwen3_5.py:482-490`, the `self.rmsnorm_output / residual_output / mlp_output` allocations at `_backend.py:325-330`.

**Gates:**
- GSM8K-50 ≥ 90% (final).
- nsys trace committed under `benchmarks/nvllm/traces/beta_coop_framework_output/2026-04-27-shipped/baseline.nsys-rep` + `summary.md` per AGENTS.md §4.
- Memory updated: `project_phase_e_beta_math_bug` resolved; new `feedback_framework_output_buffer_pattern` written.
- Commit reviewed; user approves message per `feedback_commits`.

## References

- `docs/research/uber_kernel_migration/2026-04-26-consume-gate-dce-and-graph-capture.md` — full B-fix post-mortem (514b88c6f).
- `docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md` — C2 diagnostic ship + architectural limit (today).
- `docs/research/uber_kernel_migration/2026-04-27-beta-coop-framework-output-rewrite.md` — initial 510-line plan, rejected by spec-audit (kept for context).
- `tests/compile/silly_attention.py` (commit `7d429f1b7`) — reference op template.
- `tests/compile/fullgraph/test_simple.py:146-156` (commit `7d429f1b7`) — counter verification harness.
- `vllm/model_executor/layers/attention/attention.py:713-760` (commit `7d429f1b7`) — `unified_attention_with_output` blueprint.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1267-1310` (commit `7d429f1b7`) — current β-coop launch site (pre-rewrite).
- `vllm/v1/attention/backends/cute_paged/_backend.py:1753-1767` (commit `7d429f1b7`) — `build_for_cudagraph_capture` (per-shape attn_metadata).
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2685` (commit `7d429f1b7`) — `run_beta_coop_full` signature.
- Memory: `feedback_op_body_capture_only`, `feedback_mutates_args_not_dce_safe`, `feedback_opaque_op_not_enough`, `feedback_no_silent_fallbacks`, `feedback_post_quant_sanity`, `feedback_commits`, `project_phase_e_beta_math_bug`, `project_phase_e_phantom_speedup`.
