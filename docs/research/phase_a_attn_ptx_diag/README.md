# Phase A Attention PTX-Diff Diagnostic

Harness and tooling for capturing the PTX emitted by the CuTe paged
`DecodeKernel` under the shipped D2e image and the rolled-back Phase A
Constexpr-refactor image, then producing a categorized side-by-side
diff of the two.

**Motivation.** Commit `396c3bbcfa` (the MLP PTX-diff) falsified the
hypothesis that Phase A's source edits perturbed the MLP-kernel PTX —
the two images emit byte-identical MLP PTX/CUBIN. The Q2 math break
in Phase A must therefore come from a sibling kernel in the same
`cute_paged/` package. The attention kernel in `kernel.py` is the
natural next suspect: it lives in the same package, so CuTe DSL's
package-level source-hash cache key would also see any edit to
`_tile_presets.py` / `_backend.py` / `_mlp_op.py`.

## Contents

- `harness.py`         — standalone PTX-capture harness for the decode
                         kernel. Instantiates via
                         `_get_compiled_kernel(DECODE_CONFIG)` and
                         invokes `kernel(**kwargs)` with zero tensors
                         at Qwen3.5-27B attention dims
                         (`num_q_heads=24, num_kv_heads=4,
                         head_dim=256, page_size=64`).
- `run_diagnostic.sh`  — docker driver; takes `<image:tag> <host_outdir>`.

## Diff tool

Reuses `docs/research/phase_a_ptx_diag/diff_ptx.py` (from the MLP
diagnostic) to normalize + unified-diff two PTX files and tag
changes by instruction family.

## Why in docs/research instead of benchmarks/

`benchmarks/` is evidence-only per `memory:feedback_benchmarks_evidence_only`.
Pre-run scripts and harnesses live under `docs/research/`; the PTX
dumps and `summary.md` go under
`benchmarks/nvllm/traces/phase_a_attn_ptx_diag/`.
