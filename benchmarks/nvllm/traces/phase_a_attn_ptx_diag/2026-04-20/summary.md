# Phase A Attention PTX-Diff — summary

**Date:** 2026-04-20
**Kernel:** `vllm.v1.attention.backends.cute_paged.kernel.DecodeKernel._jit_launch`
**Dims:** Qwen3.5-27B decode (`num_q_heads=24, num_kv_heads=4, head_dim=256, page_size=64`)
**Harness:** `docs/research/phase_a_attn_ptx_diag/harness.py`
**Driver:** `docs/research/phase_a_attn_ptx_diag/run_diagnostic.sh`

## Headline finding

**The emitted MLIR, PTX, and CUBIN are all byte-identical between
`nvllm:gb10-phaseD2e` and `nvllm:gb10-phaseA`.** No differences at any
level of lowering.

| Artifact | D2e md5 | Phase A md5 | Same? |
|---|---|---|---|
| MLIR  | `5205c767d8c224464fee811e7e833391` | `5205c767d8c224464fee811e7e833391` | yes |
| PTX   | `c47a6f67ca733ecf5ddc3ac7e0deb9ca` | `c47a6f67ca733ecf5ddc3ac7e0deb9ca` | yes |
| CUBIN | `8a918e5417b98d776f0f0e8f98f44372` | `8a918e5417b98d776f0f0e8f98f44372` | yes |

Sizes match exactly (PTX 2239208 B, CUBIN 1036192 B, MLIR 6278530 B).
Normalized PTX diff (`diff/ptx_side_by_side.txt`) reports 0 changed
lines across 68520 PTX lines per side.

This is a stronger result than the MLP diagnostic
(commit `396c3bbcf`), which found PTX+CUBIN byte-identical but MLIR
drifting ~357 bytes (cosmetic `Constexpr`-vs-`Int32` signature
representation). Attention MLIR is identical too.

## What this means for the Phase A Q2 hypothesis

The working hypothesis entering this session was: "MLP PTX was
falsified as the source of the Phase A Q2 math break; the likely
remaining suspect is the attention kernel, via CuTe DSL's package-
level source-hash cache key reacting to edits in sibling files
(`_tile_presets.py`, `_backend.py`, `_mlp_op.py`, `mlp_kernel.py`)."

**The attention-kernel hypothesis is falsified at the machine-code
level.** The compiled attention kernel that runs during serving is
bit-identical across the two images.

Concretely, the package-level source-hash concern — raised in
`memory:feedback_cute_source_sensitivity` — either (a) did not fire
for this pair of images at the compile point we exercise, or (b)
fired and forced a recompile but the deterministic MLIR->PTX lowering
produced identical output. Either way, executing code is the same.

Known source drift between the two images (file-level git diff):

```
vllm/v1/attention/backends/cute_paged/_backend.py    158 +/-
vllm/v1/attention/backends/cute_paged/_mlp_op.py     148 +/-
vllm/v1/attention/backends/cute_paged/mlp_kernel.py  101 +/-
```

`kernel.py` itself was not modified between the two images. The
attention kernel function body and its imports are identical, which
is why the MLIR signature is identical.

## Remaining suspects for the Phase A Q2 math break

With both MLP and attention PTX falsified, the break must come from
non-kernel code (or from a path not exercised by the compile-trigger
harness). Candidates in rough order of prior likelihood:

1. **`_backend.py` dispatch glue / `_mlp_op.py` opaque op wrapper.**
   158 + 148 lines changed between images. This is the layer that
   feeds tensors, scales, and flags into `paged_attention_forward` /
   the MLP kernel. Incorrect scale handling, wrong pointer
   arithmetic, or a misordered fusion argument would silently produce
   bad math without touching PTX.
2. **Python-side NVFP4 quant/dequant path.** Weight loading, global
   scale application, or weight format transforms in `_backend.py`
   could corrupt values before they reach either kernel.
3. **CUDA graph capture / replay.** If the Phase A image captures
   graphs differently (e.g. capture order, stream choice, graph
   input-tensor liveness), replay could produce wrong math even when
   the individual kernels are byte-identical.
4. **Invocation-path specialization not covered by this harness.**
   Our harness specializes decode at Qwen3.5-27B dims with no
   fusion. Prefill and mixed batches take different code paths
   inside the package. If Q2 was measured during prefill or with
   different `padded_num_seqs`, those paths are untested by this
   diagnostic. Fusion flags themselves are runtime `Int32`
   (`kernel.py:1523, 1551, 1707`), so fused vs unfused compiles
   produce the same PTX — that branch is ruled out.

## Environment

- CUTLASS DSL version: `4.4.2` (both images)
- `CUTE_DSL_NO_CACHE=1` — disk cache disabled, forcing fresh compile
- `CUTE_DSL_KEEP_PTX=1`, `CUTE_DSL_KEEP_IR=1`, `CUTE_DSL_KEEP_CUBIN=1`

Image git HEADs:
- D2e: `dc4bc7d6e887b2aac0f813f1bcdbd5389fc68979` (branch
  `feat/unreal-kernel-phase-d`)
- Phase A: `316be8c1b44b60956739912f320535729ff0af28` (same branch)

The "Phase A" image was built from working-tree state at that commit
with uncommitted Constexpr-refactor edits applied; see MLP
diagnostic (`benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/summary.md`)
for the audit trail of that state.

## How to reproduce

```bash
# From repo root:
mkdir -p benchmarks/nvllm/traces/phase_a_attn_ptx_diag/2026-04-20/{d2e,phaseA,diff}

bash docs/research/phase_a_attn_ptx_diag/run_diagnostic.sh \
  nvllm:gb10-phaseD2e \
  "$PWD/benchmarks/nvllm/traces/phase_a_attn_ptx_diag/2026-04-20/d2e"

bash docs/research/phase_a_attn_ptx_diag/run_diagnostic.sh \
  nvllm:gb10-phaseA \
  "$PWD/benchmarks/nvllm/traces/phase_a_attn_ptx_diag/2026-04-20/phaseA"

md5sum \
  benchmarks/nvllm/traces/phase_a_attn_ptx_diag/2026-04-20/d2e/*.ptx \
  benchmarks/nvllm/traces/phase_a_attn_ptx_diag/2026-04-20/phaseA/*.ptx

.venv/bin/python docs/research/phase_a_ptx_diag/diff_ptx.py \
  benchmarks/nvllm/traces/phase_a_attn_ptx_diag/2026-04-20/d2e/*.ptx \
  benchmarks/nvllm/traces/phase_a_attn_ptx_diag/2026-04-20/phaseA/*.ptx \
  benchmarks/nvllm/traces/phase_a_attn_ptx_diag/2026-04-20/diff/ptx_side_by_side.txt
```

## Next step

Shift focus from "find the compile-time PTX difference" to "find the
runtime value difference." Candidates for the next principled
diagnostic:

- **`_backend.py` / `_mlp_op.py` live-Python reference-math harness.**
  Run the fused attention -> W_O -> RMSNorm -> gate pipeline under
  both images on a short Qwen3.5-27B decode batch with fixed weights
  and fixed inputs; compare intermediate tensors (QK-scores, softmax,
  attention-out, W_O-out, RMSNorm-out) element-wise. First tensor
  that diverges names the layer.
- **CUDA-graph capture diff.** Capture one decode step under both
  images with torch's `_CUDAGraph.debug_dump()` (or equivalent) and
  diff the recorded stream op sequence.
