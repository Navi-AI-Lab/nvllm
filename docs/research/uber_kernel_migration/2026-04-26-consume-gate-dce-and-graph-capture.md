# Consume-gate DCE + graph-capture findings (2026-04-26)

Diagnostic baseline for the C2 follow-up architectural pass. References
the WIP commit `514b88c6f` (B-fix attempt, reverted in `3ffcf8740` on
`debug/beta-coop-residual-solo`) and its parent `5a0311ca3` (the
shippable C2 plumbing).

## TL;DR

The C2 migration's premise — β-coop replaces Python o_proj +
post_attention_layernorm — was structurally **unobservable to
torch.compile** under PIECEWISE compile. Two coupled DCE / specialisation
bugs in `vllm/nvllm/models/qwen3_5.py` caused the captured FX graph to
silently run the legacy Python pipeline and discard β-coop's outputs:

1. `cute_residual_mirror` was DCE-dropped despite
   `mutates_args=["residual_buf"]`. Dynamo's DCE removes ops whose
   mutations have no observable downstream reader **in the captured
   graph**; `impl.residual_buf` is read inside opaque op bodies via
   Python-attribute access, invisible to dynamo's reachability analysis.
   `mutates_args` alone is **not sufficient**.

2. The `if getattr(impl, "_fusion_active", False)` consume gate at
   `qwen3_5.py:466-476` was specialised to the else-branch by dynamo at
   trace time (`_fusion_active = False` is the impl `__init__` default;
   the per-step mutation happens inside the `unified_attention` opaque
   op where dynamo can't see). Captured graph: legacy Python o_proj +
   `post_attention_layernorm` ALWAYS ran; β-coop's `rmsnorm_output` /
   `residual_output` were never read.

   Dual-fire (paged + β-coop) happened to produce coherent output by
   accident: paged populated `output` with Phase A attn (via the
   framework op's `mutates_args`), Python o_proj computed `wo_out`,
   Python `post_attention_layernorm` reconstructed `residual_post_attn`.
   β-coop's outputs were entirely wasted compute.

   Solo (paged-skip) broke because nothing populated `output` with
   Phase A in solo mode → Python o_proj operated on uninitialised
   memory → gibberish.

The B-fix in `514b88c6f` proves both bugs are real and fixable under
`cudagraph_mode=NONE`. It does NOT yet survive `cudagraph_mode=PIECEWISE`
(production), suggesting at least one additional graph-capture issue
needs root-causing as part of the architectural pass.

## How the bug was confirmed

### Step 1 — captured FX graph inspection

```bash
docker exec nvllm find /root/.cache/vllm/torch_compile_cache \
  -name 'computation_graph.py' | head -1 \
  | xargs -I{} grep -oE 'torch\.ops\.vllm\.[a-z_]+' {} | sort -u
```

Output before B-fix:

```
torch.ops.vllm.cute_phase_e_dispatch
torch.ops.vllm.gdn_attention_core
torch.ops.vllm.unified_attention_with_output
torch.ops.vllm.unified_kv_cache_update
```

`cute_residual_mirror` is **absent**. Despite being called at
`qwen3_5.py:444` with `mutates_args=["residual_buf"]`. Same for
`gate_buf` mirror at `qwen3_5.py:264`.

### Step 2 — captured FX layer 3 segment shows Python pipeline

`/root/.cache/vllm/torch_compile_cache/<hash>/rank_0_0/backbone/computation_graph.py`,
layer 3 attention segment (submod_8) ran the legacy o_proj path:

```python
# qwen3_5.py:285 — applied even with _fusion_active=True at runtime
sigmoid: "bf16[s18, 6144]" = torch.sigmoid(gate_1)
mul: "bf16[s18, 6144]" = view * sigmoid
# scaled_fp4_quant.out + cutlass_scaled_fp4_mm = the o_proj
scaled_fp4_quant_out = torch.ops._C.scaled_fp4_quant.out(reshape, ...)
cutlass_scaled_fp4_mm = torch.ops._C.cutlass_scaled_fp4_mm(empty_1, empty, ...)
self_attention_output_3[slice(None, None, None)] = view_2

# qwen3_5.py:491-493 — fused-residual RMSNorm
add: "bf16[s18, 5120]" = self_attention_output_3 + x_32
# ... rsqrt, mul by gamma, .to(bf16)
to: "bf16[s18, 5120]" = mul_9.to(torch.bfloat16)

# Then cute_phase_e_dispatch consumes the post-LN output
cute_phase_e_dispatch = torch.ops.vllm.cute_phase_e_dispatch(
    to, empty_like, empty_like_1, add, 'language_model.model.layers.3.mlp')
```

Both the consume branch (`qwen3_5.py:466-476`) and the post_attn_LN gate
(`qwen3_5.py:490-496`) were dead-eliminated to favour the Python pipeline.

### Step 3 — solo result-matrix verification

| Mode                                       | Behaviour                          |
|--------------------------------------------|------------------------------------|
| EAGER + solo β-coop (no compile)           | COHERENT (no DCE, gates work)      |
| PIECEWISE + dual-fire (paged + β-coop)     | COHERENT (Python pipeline reconstructs from paged Phase A) |
| PIECEWISE + solo β-coop (paged gated off)  | GIBBERISH (nothing populates `output`) |

## What B-fix attempted (`514b88c6f`)

Three opaque ops to make the consume + post_attn_LN dispatch survive
torch.compile dead-elim:

- **`cute_residual_mirror`** (existing) — preserved across DCE by
  passing `residual_buf` and `gate_buf` as **phantom inputs** to
  `cute_attn_consume`, giving the mutations observable downstream
  readers.

- **`cute_attn_consume`** (new) — replaces the dead-eliminated consume
  branch. Always runs in the captured graph; dispatches at runtime via
  `_CUTE_ATTN_REGISTRY[layer_name]` lookup of `impl._fusion_bound`.

- **`cute_post_attn_ln_dispatch`** (new) — replaces the dead-eliminated
  post_attn_LN gate. Skips when fusion-bound (β-coop did Phase C);
  applies fused-residual RMSNorm in-place when not.

Captured FX after B-fix had all 4 ops:

```
torch.ops.vllm.cute_attn_consume
torch.ops.vllm.cute_phase_e_dispatch
torch.ops.vllm.cute_post_attn_ln_dispatch
torch.ops.vllm.cute_residual_mirror
torch.ops.vllm.gdn_attention_core
torch.ops.vllm.unified_attention_with_output
torch.ops.vllm.unified_kv_cache_update
```

`cute_residual_mirror` survived DCE thanks to the phantom dep.

## What broke under `cudagraph_mode=PIECEWISE`

PIECEWISE+NONE (B-fix v3): probe `"The capital of France is" → ` produced
`' Paris. Paris is a city in France, and it is also the capital of
France...'` — coherent.

PIECEWISE+graphs (B-fix v3): same probe produced `' Paris这种现象这种现象
这种现象...'` — first token correct (prefill), then a single-token
degenerate loop.

### Failed pivots in this session

- **v1**: tensor signal `_fusion_active_signal` + `int(signal.item())`
  inside the op body. Crashed at warmup with
  `cudaErrorStreamCaptureInvalidated`. **`.item()` causes a host-device
  sync that is incompatible with CUDA graph capture**.

- **v2**: registry-lookup of `impl._phase_e_use_beta_coop` (Python attr
  reset per-step at top of impl forward). Survived capture, gibberish
  at decode.

- **v3**: registry-lookup of `impl._fusion_bound` (set once at
  `attach_fusion`, stable across warmup + runtime). Same gibberish.

The graph-capture failure under `cudagraph_mode=PIECEWISE` was not
root-caused before the session ended.

## Suspected causes (for the architectural pass to investigate)

1. **vLLM V1 captures decode segments at warmup with shapes/state that
   diverge from runtime.** Python-attr reads inside opaque op bodies
   don't reliably reflect runtime state — what's True at warmup capture
   isn't necessarily what runs at replay. Even gating on `_fusion_bound`
   (intended to be capture-stable) didn't help, suggesting the issue
   is deeper than just the gate value.

2. **β-coop's cooperative-launch + atomic-counter spin-wait may have
   CUDA-graph replay quirks** independent of the consume gate. Captured
   cooperative kernels with stream-sync-aware barriers might not replay
   correctly across decode steps.

3. **PIECEWISE segment boundaries** — torch.compile may split the
   forward at op boundaries differently than expected. Each captured
   subgraph could have its own warmup-vs-runtime divergence.

4. **The Python o_proj path is still present in the captured graph
   alongside `cute_attn_consume`**. Even when consume copies β-coop's
   `rmsnorm_output` into `self_attention_output`, the Python o_proj
   already wrote a different value there earlier in the same forward.
   In solo, that earlier value is junk (no Phase A); but the order
   should be o_proj first, consume second, so consume should overwrite.
   Verify ordering at the kernel level under graph replay.

## Architectural answers (the C2 redesign should pick one)

- **Have β-coop write directly to the framework `output` parameter.**
  Removes the need for the Python pipeline entirely; consume becomes a
  no-op or is folded into the kernel. Requires β-coop to expose a
  bf16 attn-output buffer slot that the model framework can consume.

- **Use in-graph control flow** (`torch.cond` or `torch.where` on
  tensor signals) for the consume / post_attn_LN dispatch. Avoids
  `.item()` and Python-attr fragility entirely. Requires structuring
  the dispatch as data-dependent tensor ops rather than Python branches.

- **Capture multiple graphs per shape and dispatch externally.** vLLM
  V1 already captures separate graphs for prefill vs decode shapes;
  extend to capture separate fusion-active vs fusion-inactive variants.
  Heaviest engineering but cleanest semantics — each captured graph
  has stable behaviour at replay.

## What remains shippable on `feat/uber-kernel-migration`

Commit `5a0311ca3` (parent of the WIP) is correctness-positive:

- `cute_residual_mirror` opaque op (still DCE-dropped, but the call
  site is in place for the architectural fix to make observable).
- β-coop predicate hard-gate (no-silent-fallback).
- C2 attn-output-gate wiring through `phase_e_kernel.py`.
- Env-gated tensor dump harness for kernel-level diagnostics.

These don't change behaviour vs the prior dual-fire path (β-coop's
outputs are still discarded by dual-fire's reliance on Python o_proj +
post_attn_LN), but they're prerequisites for the architectural fix.

## How to reproduce the diagnostic

```bash
# 1. Launch with fusion enabled, PIECEWISE+graphs (default)
CUTE_PHASE_E_FUSION=1 ./scripts/serve-cute.sh

# 2. Wait for API_READY, send a probe
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","prompt":"The capital of France is","max_tokens":40,"temperature":0,"seed":42}'

# 3. Inspect the captured FX graph for vllm ops
docker exec nvllm find /root/.cache/vllm/torch_compile_cache \
  -name 'computation_graph.py' | head -1 \
  | xargs -I{} grep -oE 'torch\.ops\.vllm\.[a-z_]+' {} | sort -u

# 4. To reproduce the B-fix's PIECEWISE+NONE coherence:
#    Edit scripts/serve-cute.sh: cudagraph_mode "PIECEWISE" → "NONE"
#    Apply `git show 514b88c6f` to the working tree
#    Restart container, send probe — should be coherent
```

## References

- Commit `514b88c6f` — B-fix WIP (reverted in `3ffcf8740`).
- Commit `5a0311ca3` — shipping C2 plumbing (parent of this work).
- `memory:project_beta_coop_residual_solo_bug` — solo β-coop pickup notes.
- `memory:project_uber_kernel_migration` — C1, C1.5 status.
- `memory:feedback_pace_pressure` — don't let pace drive design;
  the architectural fix belongs on `feat/uber-kernel-migration`,
  not patched on a debug branch.
