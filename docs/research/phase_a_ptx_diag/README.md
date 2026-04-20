# Phase A PTX-Diff Diagnostic

Harness and tooling for capturing the PTX emitted by `Phase_D_MLP_Kernel`
under the shipped D2e image and the rolled-back Phase A Constexpr-refactor
image, then producing a categorized side-by-side diff of the two.

**Headline result (2026-04-20):** The emitted PTX is **byte-identical**
between the two images at Qwen3.5-27B production dims with the
`prefill-legacy` preset. The hypothesis that Phase A's source edits
perturbed the MLP-kernel PTX is falsified at this level. See
`benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/summary.md` for
the full evidence and the hypotheses for where the actual Phase A Q2
math break comes from (likely the sibling attention kernel via
package-level DSL cache-key drift).

## Contents

- `harness.py`         — standalone PTX-capture harness (production dims,
                         prefill-legacy preset). Imports
                         `Phase_D_MLP_Kernel` from the container's
                         installed `vllm` package, so the baked-in
                         kernel source determines the emitted PTX.
- `run_diagnostic.sh`  — docker driver; takes `<image:tag> <host_outdir>`.
- `diff_ptx.py`        — normalize + unified-diff two PTX files, tag
                         each changed line by instruction-family
                         category (REGALLOC, FP4_CONVERT, SMEM, etc.).

## Why in docs/research instead of benchmarks/

`benchmarks/` is evidence-only per `memory:feedback_benchmarks_evidence_only`.
Pre-run scripts and harnesses live under `docs/research/`; the PTX dumps
and summary.md go under `benchmarks/nvllm/traces/phase_a_ptx_diag/`.
