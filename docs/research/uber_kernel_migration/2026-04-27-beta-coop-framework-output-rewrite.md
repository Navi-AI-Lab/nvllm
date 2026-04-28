# β-coop framework-output-buffer rewrite plan — 2026-04-27

**Branch:** `feat/uber-kernel-migration` HEAD `7d429f1b7`.
**Savepoint:** `feat/pre-beta-coop-rewrite-savepoint` at `7d429f1b7`.
**File backups:** `/tmp/c2_rewrite_backup/{_backend,_mlp_op,kernel,qwen3_5}.py.pre-rewrite`.

---

## TL;DR

β-coop's outputs are currently consumed via Python attribute access (`impl.rmsnorm_output`, `impl.residual_output`, `impl.mlp_output`), which is invisible to torch.compile dynamo. This forces a fragile mesh of opaque ops (`cute_residual_mirror`, `cute_phase_e_dispatch`, would-be `cute_attn_consume`, would-be `cute_post_attn_ln_dispatch`) to keep capture alive. Even with the phantom-input pattern from `514b88c6f`, the round-trip through Python state caused gibberish under PIECEWISE+graphs (only `PIECEWISE+cudagraph_mode=NONE` worked).

The rewrite makes β-coop's outputs **framework-supplied tensors** the layer pre-allocates. The new op `cute_beta_coop_run` mutates them in place (declared via `mutates_args`) and the layer consumes them as graph-tracked tensors. This eliminates the Python round-trip entirely and follows the canonical vLLM pattern (`unified_attention_with_output`).

**Production proof points already in tree:** every attention backend in vLLM uses this exact shape (FA2/FA3, Triton, FlashInfer, MLA), verified working under PIECEWISE+CUDA graphs.

---

## Why (architectural justification)

### What's broken today

Three independent failures stack on the current architecture:

1. **`cute_residual_mirror` DCE'd from captured graph** — `vllm/v1/attention/backends/cute_paged/_mlp_op.py:266-311` (commit `7d429f1b7`). Despite `mutates_args=["residual_buf"]`, dynamo's reachability analysis sees no graph-level reader of `residual_buf` (downstream readers access it via `self.residual_buf` inside opaque op bodies — invisible). Captured FX has 0 calls to the op. Verified empirically 2026-04-27: 256-token decode emitted 0 `[RES_MIRROR_OP]` log lines despite β-coop firing.

2. **β-coop reads zeros at runtime** — `_backend.py:1271` reads `self.residual_buf[:nat]` as `residual_in`. Because the mirror op was DCE'd, `residual_buf` stays at the CUDA-graph-allocator-zeroed value. β-coop computes `attn_out + 0` instead of `attn_out + (x_n + r_n)`. Layer-output residual stream is missing the input residual. Cascade through 64 layers → Chinese-character spam (`这种现象 × 256` reproduced today).

3. **Consume-gate DCE under fullgraph** — the `if getattr(impl, "_fusion_active", False)` gate at `qwen3_5.py:480` is specialized to "always-take-else" by dynamo at trace time (the attribute is False at `__init__`, mutated only inside `unified_attention` opaque op — invisible). See `docs/research/uber_kernel_migration/2026-04-26-consume-gate-dce-and-graph-capture.md`.

### Why phantom-input alone (`514b88c6f`) didn't fix it

The B-fix added phantom inputs to keep `cute_residual_mirror` alive AND introduced `cute_attn_consume` + `cute_post_attn_ln_dispatch` to replace the dead-eliminated Python branches. Captured FX after the B-fix had all 4 ops. Result matrix:

| Mode | Result |
|---|---|
| EAGER + solo β-coop | COHERENT (no compile, gates fire) |
| PIECEWISE + dual-fire | COHERENT *by accident* (Python pipeline reconstructs from paged Phase A — β-coop wasted) |
| PIECEWISE + `cudagraph_mode=NONE` + solo | **COHERENT** ✓ |
| PIECEWISE + `cudagraph_mode=PIECEWISE` + solo | **GIBBERISH** ✗ |

The phantom input fixed DCE. The graph-replay failure under `PIECEWISE+graphs` was never root-caused. Three pivots failed (.item() crashed capture, registry-lookup of `_phase_e_use_beta_coop`, registry-lookup of `_fusion_bound`). All three round-trip through Python state — capture-time state ≠ replay-time state. **The Python round-trip is the deeper problem.**

### Why framework-output-buffer fixes both layers at once

The pattern: caller pre-allocates output tensors; op mutates them in place; downstream consumes them as graph-tracked tensors. There is **no Python attribute** in the consumer path.

- DCE: mutated tensors are explicitly named in `mutates_args` AND have a graph-level reader (the layer's next op or return). Dynamo can't drop the op without losing observable behavior.
- Graph replay: outputs are baked into the captured CUDA op sequence. Each captured shape gets its own kernel-launch sequence, so per-shape dispatch (decode size 1 → β-coop, decode size 4 → legacy fall-through) lands the right kernels in the right graph.
- Per-step fidelity: nothing depends on Python state changing between capture and replay; weights/scales are constants set at attach time, residual stream is a graph tensor.

---

## Evidence (all three agents' findings, verified file-by-file)

### Pattern 1 — `unified_attention_with_output` (vLLM canonical)

**`vllm/model_executor/layers/attention/attention.py:713-760` (commit `7d429f1b7`):**

```python
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    del kv_cache_dummy_dep
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    self.impl.forward(self, query, key, value, kv_cache, attn_metadata,
                      output=output, output_scale=output_scale,
                      output_block_scale=output_block_scale)

direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_attention_with_output_fake,
)
```

Used by every attention backend in vLLM. Verified production under PIECEWISE+CUDA graphs.

### Pattern 2 — `kv_cache_dummy_dep` ordering preservation

**`vllm/model_executor/layers/attention/attention.py:457-494` (commit `7d429f1b7`):**

```python
kv_cache_dummy_dep = unified_kv_cache_update(key, value, self.layer_name)
unified_attention_with_output(query, key, value, output, self.layer_name,
                              kv_cache_dummy_dep=kv_cache_dummy_dep)
```

`unified_kv_cache_update` returns a 0-element tensor; that tensor is passed as a kwarg to `unified_attention_with_output` purely to give dynamo a data-dependency edge between the two ops. Inside the body: `del kv_cache_dummy_dep`. We will use this same shape if β-coop needs to depend on a precursor op (probably not — single op handles everything).

### Pattern 3 — counter-based replay verification

**`tests/compile/silly_attention.py:34-66` (commit `7d429f1b7`):**

```python
def silly_attention(q, k, v, out: torch.Tensor) -> None:
    global _global_counter
    _global_counter += 1
    out.copy_(q + k + v)

direct_register_custom_op(
    op_name="attention",
    op_func=silly_attention,
    mutates_args=["out"],
    fake_impl=silly_attention_fake,
    target_lib=silly_lib,
)
```

**`tests/compile/fullgraph/test_simple.py:146-156` (commit `7d429f1b7`):**

```python
input = torch.zeros(2).cuda()
reset_global_counter()
with set_forward_context(None, vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(num_tokens=2)):
    output = model(input)
assert get_global_counter() == 2
```

This is the cheapest empirical proof that an op actually fires under graph replay. We will replicate this for `cute_beta_coop_run` before doing any production wiring.

### Pattern 4 — `auto_functionalized` Inductor-pass fusion (reference, not blueprint)

**`vllm/compilation/passes/fusion/attn_quant_fusion.py:64-72` (commit `7d429f1b7`):**

```python
ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
at1 = auto_functionalized(ATTN_OP, query=q, key=k, value=v,
                           output=output_attn, layer_name=self._layer_name,
                           output_scale=None, output_block_scale=None,
                           kv_cache_dummy_dep=kv_cache_dummy_dep)
```

This is how Inductor passes rewrite the captured graph to fuse downstream quant into the attention op. We don't need an Inductor pass for the initial rewrite — just the explicit `mutates_args` declaration is sufficient. But the pattern is here if we later want to fuse e.g. NVFP4 quant of the residual stream.

### Pattern 5 — defensive `mutates_args` anchor (lying-but-safe)

**`vllm/v1/attention/backends/cute_paged/_c2_diag.py:236-240, 306` (commit `7d429f1b7`):** declares `mutates_args=["legacy_hidden"]` even though the op doesn't actually mutate it. The hidden_states tensor is read by downstream layers as a graph tensor, so this anchors the op in the graph. We will *not* use this pattern for the rewrite — the new op genuinely mutates its outputs, no lying needed.

### Anti-pattern — what we're moving away from

**`vllm/v1/attention/backends/cute_paged/_backend.py:1271` (commit `7d429f1b7`):**

```python
self._phase_e_coop_kernel.run_beta_coop_full(
    hidden_in=self.rmsnorm_output[:nat],
    residual_in=self.residual_buf[:nat],   # ← reads Python attr; DCE-fragile
    ...
    attn_output=self.rmsnorm_output[:nat],  # ← writes Python attr; not graph-visible
    mlp_output=self.mlp_output[:nat],       # ← writes Python attr; not graph-visible
    residual_output=self.residual_output[:nat],  # ← writes Python attr; not graph-visible
)
```

Every input and every output is a Python attribute on `impl`. Dynamo sees nothing here — the entire side-effect graph is invisible. The downstream consumer at `qwen3_5.py:482-485` does:

```python
self_attention_output[:nat].copy_(impl.rmsnorm_output[:nat])
residual[:nat].copy_(impl.residual_output[:nat])
```

These reads-via-attribute are why the consume gate DCE'd, why the residual mirror DCE'd, and why phantom inputs alone couldn't save the design. **The fix is to stop using `impl.*` for runtime-varying tensors entirely.**

---

## Target state (file-by-file)

### `vllm/v1/attention/backends/cute_paged/_beta_coop_op.py` (NEW)

New module containing the framework-output-buffer custom op. Skeleton:

```python
"""β-coop unified custom op (framework-output pattern).

Replaces:
  - cute_residual_mirror (residual stream copy at layer entry)
  - cute_phase_e_dispatch (β-lite consume gate)
  - The implicit Python-attribute round-trip at qwen3_5.py:482-485

The op's body runs at PIECEWISE-graph-capture time:
  1. Looks up impl via layer_name (vLLM canonical pattern).
  2. Reads attn_metadata from forward_context (size, is_decode_only).
  3. Decides at THIS shape's capture time: β-coop fires (1×size=1 fits coop cap)
     or fall-through (legacy paged + Python o_proj + post-attn-LN) fires.
  4. Records the decided kernel sequence into the captured graph.

Each captured shape gets the right kernel sequence baked in. At runtime, the
right captured graph replays for the runtime shape — no Python state needed.
"""

import torch
from vllm.utils.torch_utils import direct_register_custom_op

_BETA_COOP_FIRE_COUNTER: dict[str, int] = {}  # for verification harness


def cute_beta_coop_run(
    # Per-step graph tensors:
    query: torch.Tensor,           # [nat, num_q_heads, head_dim] BF16
    kv_cache: torch.Tensor,        # [pg, 2, ps, kv, hd] uint8 FP8
    residual: torch.Tensor,        # [nat, hidden] BF16 — pre-attn residual stream
    attn_input: torch.Tensor,      # [nat, hidden] BF16 — post-input-LN hidden
    gate_buf: torch.Tensor,        # [nat, q_size] BF16 — Qwen3.5 attn gate
    # Mutated (graph-tracked outputs):
    rmsnorm_out: torch.Tensor,     # [nat, hidden] BF16 — post-attn-LN'd, MLP-input shape
    residual_out: torch.Tensor,    # [nat, hidden] BF16 — post-attn residual stream
    mlp_out: torch.Tensor,         # [nat, hidden] BF16 — MLP output (next layer's hidden)
    # Layer identity:
    layer_name: str,
) -> None:
    """β-coop's full pipeline: reads residual, writes (rmsnorm_out, residual_out, mlp_out).

    Body runs at graph capture time. Dispatches to:
      - β-coop kernel if (is_decode_only AND coop-fitness AND fusion-bound)
      - legacy paged + Python pipeline otherwise

    Whichever path runs writes ALL THREE output tensors so the layer can
    consume them unconditionally.
    """
    from vllm.model_executor.layers.attention.attention import get_attention_context
    attn_metadata, attn_layer, kv_cache_, _ = get_attention_context(layer_name)
    impl = attn_layer.impl

    # Counter for empirical verification (test harness reads this).
    _BETA_COOP_FIRE_COUNTER[layer_name] = _BETA_COOP_FIRE_COUNTER.get(layer_name, 0) + 1

    nat = query.shape[0]
    is_decode_only = getattr(attn_metadata, "is_decode_only", False)
    num_seqs = len(attn_metadata.seq_lens) if attn_metadata is not None else 0

    will_fire_beta_coop = (
        impl._phase_e_env_pre_cached.enabled
        and is_decode_only
        and impl._mlp_fusion_bound
        and impl._phase_e_coop_kernel is not None
        and nat <= impl._fusion_max_num_seqs
        and (64 * num_seqs) <= impl._resident_cap
        and impl._phase_e_env_pre_cached.forced_path in ("coop", "auto")
    )

    if will_fire_beta_coop:
        # β-coop writes rmsnorm_out + residual_out + mlp_out directly.
        impl._phase_e_coop_kernel.run_beta_coop_full(
            # Inputs come from explicit op args (not impl.*):
            residual_in=residual,
            attn_input_bf16=attn_input,
            query=query,
            kv_cache=kv_cache,
            gate_buf=gate_buf,
            page_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            # Weights/scales come from impl (constants post-attach):
            wo_weight=impl.wo_weight,
            wo_scales=impl.wo_scales,
            wo_global_scale=impl.wo_global_scale,
            input_gamma=impl._phase_e_coop_input_gamma,
            post_attn_gamma=impl.rmsnorm_gamma,
            gate_w_fp4=impl._mlp_gate_w,
            gate_w_scale=impl._mlp_gate_s,
            up_w_fp4=impl._mlp_up_w,
            up_w_scale=impl._mlp_up_s,
            down_w_fp4=impl._mlp_down_w,
            down_w_scale=impl._mlp_down_s,
            # Outputs (framework-supplied):
            attn_output=rmsnorm_out,    # post-attn-LN'd (MLP input)
            residual_output=residual_out,
            mlp_output=mlp_out,
            # Scalars:
            scale=impl.scale,
            k_scale=impl.k_scale,
            v_scale=impl.v_scale,
            gate_up_global_scale=impl._mlp_gate_up_gs,
            down_global_scale=impl._mlp_down_gs,
            # hidden_in: dummy (β-coop's Phase 0 is a side-channel for QKV-fusion;
            # not consumed by this layer's attn path — pass an unused buffer).
            hidden_in=rmsnorm_out,  # placeholder; β-coop ignores this
        )
    else:
        # Fall-through: legacy paged_attention_forward + Python o_proj + post-attn-LN.
        # Implementation: reuses impl's existing legacy code path. Writes the same
        # three output tensors so the layer consumer doesn't branch.
        _fallthrough_legacy(
            impl, attn_metadata, query, kv_cache, residual, attn_input,
            gate_buf, rmsnorm_out, residual_out, mlp_out,
        )


def cute_beta_coop_run_fake(query, kv_cache, residual, attn_input, gate_buf,
                             rmsnorm_out, residual_out, mlp_out, layer_name):
    return None  # mutates_args declared; no return value


direct_register_custom_op(
    op_name="cute_beta_coop_run",
    op_func=cute_beta_coop_run,
    mutates_args=["rmsnorm_out", "residual_out", "mlp_out"],
    fake_impl=cute_beta_coop_run_fake,
)
```

### `vllm/nvllm/models/qwen3_5.py` — layer-side rewrite

Replace the current attention + consume + post-attn-LN + (β-lite consume MLP) sequence with a single op call. Pseudocode for the NEW state:

```python
# In the layer's full_attention path (around line 460 currently):
if self.layer_type == "full_attention" and impl is not None and impl._fusion_bound:
    # Pre-allocate framework outputs.
    rmsnorm_out  = torch.empty_like(residual)
    residual_out = torch.empty_like(residual)
    mlp_out      = torch.empty_like(residual)

    # Compute attn_input from input_layernorm result (already done at this point).
    # input_layernorm fused the residual add, so:
    #   hidden_states is post-input-LN'd (LN output)
    #   residual is post-add (x_n + r_n)
    attn_input = hidden_states

    # Pre-projected query (QKV proj still runs in Python — Phase 0 is a
    # future fusion target, not in scope here).
    query = self.self_attn.q_proj(attn_input).view(nat, num_q_heads, head_dim)

    torch.ops.vllm.cute_beta_coop_run(
        query=query,
        kv_cache=kv_cache,
        residual=residual,
        attn_input=attn_input,
        gate_buf=gate_buf_filled_externally,
        rmsnorm_out=rmsnorm_out,
        residual_out=residual_out,
        mlp_out=mlp_out,
        layer_name=self.layer_name,
    )

    # Layer outputs (graph-tracked tensors, no impl.* reads):
    hidden_states = mlp_out
    residual = residual_out
    # Skip post_attention_layernorm — β-coop already did it.
    # Skip MLP path — β-coop already did it.
```

Key consequences:
- No more `cute_residual_mirror` call.
- No more `if _fusion_active` consume gate.
- No more `if not _fusion_active: post_attention_layernorm(...)`.
- No more `cute_phase_e_dispatch` (β-lite MLP consume).
- No more `Qwen3_5MLP.forward` (in fusion path).

### `vllm/v1/attention/backends/cute_paged/_backend.py` — gate consolidation

The current `_will_fire_beta_coop_pre` gate at `_backend.py:1090-1099` and the parallel `_use_beta_coop` gate at `_backend.py:1223-1228` are duplicated logic. With the new op, β-coop is launched from `cute_beta_coop_run` body (not from `_backend.forward`). Two options:

- **Option B.1 (preferred):** Remove β-coop launch from `_backend.forward` entirely. `_backend.forward` only ever runs in the fall-through path now. Existing `_will_fire_beta_coop_pre` paged-skip wrapper at `_backend.py:1100-1103` (`result = None`) becomes unconditional in the new flow (the new op handles dispatch internally; if `cute_beta_coop_run` chose β-coop, paged never ran; if it chose fall-through, the new op called `_backend.forward` itself).
- **Option B.2 (transitional):** Keep `_backend.forward` as the fall-through implementer, called by `cute_beta_coop_run` body when `will_fire_beta_coop = False`. This is the lower-risk path and what the skeleton above shows.

We'll start with B.2 (transitional) and consolidate to B.1 once stable.

### `vllm/v1/attention/backends/cute_paged/_mlp_op.py` — retire

Mark `cute_residual_mirror` deprecated (keep behind a feature flag for one cycle in case we need to roll back). `cute_phase_e_dispatch` is retired wholesale (β-lite consume folds into `cute_beta_coop_run`'s fall-through path).

### `tests/v1/cute_paged/test_beta_coop_op.py` (NEW)

Counter-based verification mirroring `tests/compile/fullgraph/test_simple.py`:

```python
def test_cute_beta_coop_fires_per_replay():
    """Empirical proof the op body runs once per captured-shape replay."""
    # Build a model with the op in the call site.
    # Capture under PIECEWISE for sizes [1,2,4].
    # Reset counter.
    # Replay each size N times.
    # Assert _BETA_COOP_FIRE_COUNTER[layer_name] advances by exactly N per replay.
```

This is non-negotiable — without it we have no empirical evidence the rewrite escapes the trap that bit `514b88c6f`.

---

## Implementation phases

Each phase ends with a verification gate. Don't proceed to the next until the gate passes.

### Phase 1 — verification harness (no production wiring)

**Goal:** prove the framework-output-buffer pattern works for our shapes BEFORE touching production code.

1. Create `tests/v1/cute_paged/test_beta_coop_op_skeleton.py` that registers a `cute_beta_coop_skeleton` op (echo: `out.copy_(input)`), declares `mutates_args=["out"]`, and a counter-based test that captures under PIECEWISE for sizes `[1,2,4,8]` and asserts the counter advances on every replay.
2. Run inside the container: `.venv/bin/python -m pytest tests/v1/cute_paged/test_beta_coop_op_skeleton.py -v`.

**Gate:** counter advances exactly N per N replays for every shape. If it fails, our understanding is wrong — STOP, do not write any production code, re-investigate.

### Phase 2 — `cute_beta_coop_run` op body + fake impl (no callers)

**Goal:** the op exists, registers cleanly, fake impl runs without errors. No production code calls it yet.

1. Create `vllm/v1/attention/backends/cute_paged/_beta_coop_op.py` with the full skeleton from "Target state" above.
2. Body initially just calls the existing `_backend.forward` (legacy fall-through) — no β-coop dispatch yet.
3. Unit test: register the op, call it with dummy tensors, verify it doesn't crash.

**Gate:** `import` the new module from a Python REPL inside the container, the op shows up under `torch.ops.vllm.cute_beta_coop_run`. No registration errors.

### Phase 3 — layer-side rewrite, fall-through-only

**Goal:** rewire the layer to use the new op, but β-coop still gates off (forced fall-through). Production behavior should be IDENTICAL to PIECEWISE+graphs without `CUTE_PHASE_E_FUSION` (the existing default).

1. Modify `qwen3_5.py` to call `torch.ops.vllm.cute_beta_coop_run(...)` instead of the existing attention+consume+LN+MLP sequence.
2. New op body's `will_fire_beta_coop` always returns False (force fall-through) for this phase.
3. Build, launch, run a 256-token completion at temperature=0.

**Gate:** GSM8K-50 ≥ 90% (per `feedback_post_quant_sanity`). If output is wrong with fall-through-only, the layer-side rewrite has a bug — fix before adding β-coop dispatch.

### Phase 4 — β-coop dispatch enabled

**Goal:** flip β-coop on inside the new op body. This is the moment of truth.

1. Restore the `will_fire_beta_coop` gate logic (decode-only, coop-fitness, fusion-bound) inside `cute_beta_coop_run`'s body.
2. Build, launch with `CUTE_PHASE_E_FUSION=1 MAX_NUM_SEQS=1 bash scripts/serve-cute.sh`.
3. Run the 256-token completion test.

**Gate (must pass ALL):**
- Output is coherent (no `这种现象` loop).
- GSM8K-50 ≥ 90%.
- `_BETA_COOP_FIRE_COUNTER[layer_name]` shows ~256 fires per layer (counter test).
- nsys trace shows β-coop kernel actually launching (no silent fall-through).

If gibberish persists at this gate: the suspected secondary cause from the doc (β-coop's cooperative-launch + atomic-counter quirks under graph replay) is real, and is independent of this rewrite. We'd then need to investigate the β-coop kernel itself (cooperative launch + spin-wait barrier interactions with CUDA graph replay).

### Phase 5 — retire deprecated ops + commit

**Goal:** clean up dead code now that the new path is proven.

1. Delete `cute_residual_mirror` from `_mlp_op.py` (or leave a stub raising `NotImplementedError("retired by 2026-04-27 framework-output rewrite")`).
2. Delete `cute_phase_e_dispatch` (β-lite MLP consume) — folded into `cute_beta_coop_run` fall-through.
3. Squash to a single commit per `feedback_commits` confirmation.

**Gate:** clean rebuild, GSM8K-50 ≥ 90%, nsys trace committed under `benchmarks/nvllm/traces/beta_coop_framework_output/2026-04-27-shipped/`.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| β-coop's cooperative-launch + atomic-counter spin-wait has CUDA-graph replay quirks (suspect #2 from `2026-04-26-consume-gate-dce-and-graph-capture.md`) | Medium | Phase 4 verification gate explicitly tests this. If it fails, the rewrite is correct but β-coop kernel itself needs separate investigation. |
| β-coop expects `attn_input_bf16` etc. to be allocated with specific stride/dtype | Low | Match `torch.empty_like(residual)` shape, use BF16, contiguous. Cross-check against `phase_e_kernel.py:2685` arg docstrings. |
| Graph capture of multiple decode shapes results in different op-body decisions across shapes | Low (this is by design) | Each captured shape gets its own kernel sequence. Tested by Phase 4 gate. |
| Layer-name resolution inside the op fails (custom op vs `vllm::*` namespace) | Low | Pattern is known to work — `cute_residual_mirror` already does it. Plus `unified_attention_with_output` is the blueprint. |
| Removing `cute_residual_mirror` breaks paths we don't know about | Medium | Keep it as deprecated stub initially (Phase 5). Roll out the new path in Phase 4 first; only retire after confirmation. |
| QKV proj is still in Python → Phase 0 (input-LN) fusion is incomplete | Known scope-out | Phase 0 fusion is a separate effort. Current rewrite leaves QKV in Python — `attn_input` is the layer's existing post-input-LN hidden_states. |
| `_phase_e_env_pre_cached` doesn't exist as an attribute on impl | High (made up in skeleton) | Need to verify or add: `impl._phase_e_env_pre_cached = _phase_e_env_config()` set once at attach time. Or just call `_phase_e_env_config()` at op body time (it's fast and stable). |

---

## Rollback

If any phase gate fails and a fix isn't immediately obvious:

```bash
# Wholesale rollback to savepoint:
git checkout feat/uber-kernel-migration
git reset --hard feat/pre-beta-coop-rewrite-savepoint

# Or single-file recovery:
cp /tmp/c2_rewrite_backup/_backend.py.pre-rewrite vllm/v1/attention/backends/cute_paged/_backend.py
cp /tmp/c2_rewrite_backup/_mlp_op.py.pre-rewrite vllm/v1/attention/backends/cute_paged/_mlp_op.py
cp /tmp/c2_rewrite_backup/kernel.py.pre-rewrite vllm/v1/attention/backends/cute_paged/kernel.py
cp /tmp/c2_rewrite_backup/qwen3_5.py.pre-rewrite vllm/nvllm/models/qwen3_5.py
```

Per `feedback_commits`, do not commit until all phase gates pass + user confirms commit message.

---

## Verification checklist (full run-through before declaring done)

- [ ] Phase 1 counter-based skeleton test: counter advances exactly N per N replays for sizes [1,2,4,8].
- [ ] Phase 2 op registers without errors; visible at `torch.ops.vllm.cute_beta_coop_run`.
- [ ] Phase 3 fall-through-only run produces coherent output; GSM8K-50 ≥ 90%.
- [ ] Phase 4 β-coop-active run produces coherent output; GSM8K-50 ≥ 90%.
- [ ] Phase 4 `_BETA_COOP_FIRE_COUNTER` shows expected per-layer fire counts after a 256-token completion.
- [ ] Phase 4 nsys trace shows β-coop kernel launches at decode (not silent fall-through).
- [ ] Phase 4 captured FX graph (if dumpable) shows `cute_beta_coop_run` calls and NO `cute_residual_mirror` / `cute_phase_e_dispatch` calls.
- [ ] Phase 5 deprecated ops removed; clean rebuild green.
- [ ] Phase 5 nsys trace committed under `benchmarks/nvllm/traces/beta_coop_framework_output/2026-04-27-shipped/`.
- [ ] Memory updated: `project_phase_e_beta_math_bug` resolved; new `feedback_framework_output_buffer_pattern` written.

---

## References

- `docs/research/uber_kernel_migration/2026-04-26-consume-gate-dce-and-graph-capture.md` — full B-fix post-mortem.
- `docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md` — C2 diagnostic ship + architectural limit.
- `tests/compile/silly_attention.py` (commit `7d429f1b7`) — reference op template.
- `tests/compile/fullgraph/test_simple.py` (commit `7d429f1b7`) — counter verification harness.
- `vllm/model_executor/layers/attention/attention.py:713-760` (commit `7d429f1b7`) — production blueprint.
- `vllm/compilation/passes/fusion/attn_quant_fusion.py:64-72` (commit `7d429f1b7`) — auto_functionalized integration reference.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2685` (commit `7d429f1b7`) — `run_beta_coop_full` signature.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1267-1310` (commit `7d429f1b7`) — current launch site (pre-rewrite).
- Memory: `feedback_op_body_capture_only`, `feedback_mutates_args_not_dce_safe`, `feedback_opaque_op_not_enough`, `feedback_dynamo_disable_fullgraph`, `project_phase_e_beta_math_bug`, `project_phase_e_phantom_speedup`.
