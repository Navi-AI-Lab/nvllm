# Phase A MLP math harness — summary

**Date:** 2026-04-20
**Target:** `Phase_D_MLP_Kernel.__call__` at Qwen3.5-27B dims
(hidden=5120, interm=17408, prefill-legacy preset 256/640/8).
**Harness:** `docs/research/phase_a_mlp_math_harness/harness.py`
**Driver:** `docs/research/phase_a_mlp_math_harness/run_diagnostic.sh`
**Diff:**   `docs/research/phase_a_mlp_math_harness/diff_outputs.py`

## Headline finding

**Phase_D_MLP_Kernel.__call__ produces bit-identical output across
nvllm:gb10-phaseD2e and nvllm:gb10-phaseA on all four test cases,
including the previously untested nat=8 decode-batch path.**

| Case | md5 (D2e) | md5 (Phase A) | Same? |
|---|---|---|---|
| `zero_nat1`         | `1276481102f218c981e0324180bafd9f` | `1276481102f218c981e0324180bafd9f` | yes |
| `seed_nat1`         | `d6d9ed37b4f4a3954c9f88ebaea74100` | `d6d9ed37b4f4a3954c9f88ebaea74100` | yes |
| `seed_nat8`         | `507ed14f2a11de96296b3e44e990f925` | `507ed14f2a11de96296b3e44e990f925` | yes |
| `seed_nat1_repeat`  | `d6d9ed37b4f4a3954c9f88ebaea74100` | `d6d9ed37b4f4a3954c9f88ebaea74100` | yes |

NaN bit patterns (seed cases overflow the FP4 accumulator at random
u8 weights — all ~40960 elements NaN) match exactly across images.
nat=8 bit-identical between images — the Constexpr refactor's eager-
compile specialization at nat=1 does NOT prevent correct runtime
behavior at nat=8.

## Combined with today's PTX-diag results

| Diagnostic | What's bit-identical? |
|---|---|
| MLP PTX (`phase_a_ptx_diag`, commit `396c3bbcf`)     | PTX + CUBIN byte-identical; MLIR drifts cosmetically ~357 B |
| Attention PTX (`phase_a_attn_ptx_diag`, commit `1f008b4fe`) | PTX + CUBIN + MLIR all byte-identical |
| MLP `__call__` (this harness)                         | Output tensor bit-identical across 4 cases including nat=8 |

Nothing between the `Phase_D_MLP_Kernel.__call__` boundary and the
compiled kernel machine code differs in behavior.

## Full vllm/ tree diff between images

```
vllm/nvllm/layers/mlp.py           — docstring cleanup only
vllm/v1/attention/backends/cute_paged/_backend.py
    — tile-preset resolver plumbing only
vllm/v1/attention/backends/cute_paged/_mlp_op.py
    — docstring cleanup only
vllm/v1/attention/backends/cute_paged/mlp_kernel.py
    — Constexpr refactor (now proven behaviorally neutral)
vllm/v1/attention/backends/cute_paged/_tile_presets.py (new)
    — preset registry sibling module

vllm/_C.abi3.so                    — stripped, still differs (~194 MB)
vllm/_C_stable_libtorch.abi3.so    — stripped, still differs (~115 MB)
vllm/cumem_allocator.abi3.so       — differs (tiny)
vllm/_moe_C.abi3.so                — differs (~81 MB)
vllm/vllm_flash_attn/_vllm_fa{2,3}_C.abi3.so  — differ
vllm/_version.py                   — git HEAD string
```

No Python source in the vllm tree has a semantic change that hasn't
been ruled out by PTX-diag + this harness. The .so files differ after
`strip`, suggesting different build artifacts from the same source
(build timestamps, paths, or rodata ordering), not intentional code
changes — but this is unconfirmed.

## Open question (for next session)

Does the Phase A Q2 break actually still reproduce? The evidence
chain above rules out every kernel-level difference. The break
was observed on 2026-04-20 morning during Phase A gate-2; we have
not re-verified it since. Possible scenarios:

1. **Break still reproduces.** Then the remaining suspects are:
   - `.so` differences (need `objdump --disassemble` compare of
     `_C.abi3.so` text sections to confirm code is same)
   - Model-load / CUDA-graph-capture timing under serving that our
     harness doesn't exercise
   - vLLM dispatch at a level outside the files we diffed

2. **Break does NOT reproduce on a fresh serve.** Then Phase A's
   original gate-2 failure was a transient image build artifact
   (stale Docker cache layer, ABI mismatch, wrong cutlass version
   that matches 4.4.2 now) — not a code bug. Fix: rebuild.

## How to reproduce this diagnostic

```bash
mkdir -p benchmarks/nvllm/traces/phase_a_mlp_math_harness/2026-04-20/{d2e,phaseA,diff}

bash docs/research/phase_a_mlp_math_harness/run_diagnostic.sh \
    nvllm:gb10-phaseD2e \
    "$PWD/benchmarks/nvllm/traces/phase_a_mlp_math_harness/2026-04-20/d2e"

bash docs/research/phase_a_mlp_math_harness/run_diagnostic.sh \
    nvllm:gb10-phaseA \
    "$PWD/benchmarks/nvllm/traces/phase_a_mlp_math_harness/2026-04-20/phaseA"

.venv/bin/python docs/research/phase_a_mlp_math_harness/diff_outputs.py \
    benchmarks/nvllm/traces/phase_a_mlp_math_harness/2026-04-20/d2e \
    benchmarks/nvllm/traces/phase_a_mlp_math_harness/2026-04-20/phaseA \
    benchmarks/nvllm/traces/phase_a_mlp_math_harness/2026-04-20/diff/summary.md
```

## Environment

- Both images: torch `2.12.0.dev20260402+cu132`, cutlass-dsl `4.4.2`,
  CUDA device `NVIDIA GB10`
- Image git HEADs:
  - D2e: `dc4bc7d6e` (branch `feat/unreal-kernel-phase-d`)
  - Phase A: `316be8c1b` (same branch) + uncommitted Constexpr
    refactor working-tree edits
