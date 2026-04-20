# Phase A MLP math harness

Direct-`__call__` harness for comparing `Phase_D_MLP_Kernel` output
between `nvllm:gb10-phaseD2e` and `nvllm:gb10-phaseA` on deterministic
seeded inputs.

## Motivation

Two PTX-diagnostics landed 2026-04-20:
- MLP kernel (`docs/research/phase_a_ptx_diag/`, commit `396c3bbcf`):
  MLP PTX+CUBIN byte-identical between the two images.
- Attention `DecodeKernel` (`docs/research/phase_a_attn_ptx_diag/`,
  commit `1f008b4fe`): PTX+CUBIN+MLIR byte-identical.

Both CuTe kernels compile to bit-identical machine code. The Phase A
Q2 math break must come from non-kernel code — Python glue,
marshalling, CUDA graph capture, or a runtime path the PTX-diag
harnesses do not exercise (e.g. `nat > 1`, real weight values).

This harness closes that gap by invoking `Phase_D_MLP_Kernel.__call__`
directly with deterministic seeded tensors and saving outputs. If D2e
and Phase A diverge on any case, the break is inside the
`Phase_D_MLP_Kernel` call boundary; if they all match, the break lives
elsewhere (the `_mlp_op.py` opaque-op wrapper, the attention path, or
torch/CUDA-graph glue).

## Contents

- `harness.py`          — runs the kernel on 4 cases and saves `.pt`
                          outputs under `/workdir/out/`:
  1. `zero_nat1`  — all-zero weights + input (matches PTX-diag path)
  2. `seed_nat1`  — seeded FP4 weights + seeded BF16 input, `nat=1`
  3. `seed_nat8`  — same seeded weights, `nat=8` (decode batch)
  4. `seed_nat1_repeat` — repeat case 2; determinism check across
                          consecutive calls
- `run_diagnostic.sh`   — docker driver, `<image:tag> <host_outdir>`
- `diff_outputs.py`     — elementwise compare two output dirs, emit
                          `summary.md` with fingerprints + localization

## Usage

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

## Why in docs/research/

`benchmarks/` is evidence-only per `memory:feedback_benchmarks_evidence_only`.
Pre-run harnesses live under `docs/research/`; the `.pt` dumps and
`diff/summary.md` go under `benchmarks/nvllm/traces/`.
