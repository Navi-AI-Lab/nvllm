# Phase A PTX-Diff Diagnostic — 2026-04-20

## TL;DR

**The PTX is byte-identical.** The D2e-shipped fused MLP kernel and the
rolled-back Phase A `cute.compile+Constexpr` fused MLP kernel, when
compiled at Qwen3.5-27B production dims with the `prefill-legacy` preset,
emit the exact same PTX and CUBIN. This falsifies the hypothesis in
`memory:project_phase_a_bootstrap_drift` and `memory:feedback_cute_source_sensitivity`
that the D3a / Constexpr source edits "perturbed the compiled PTX
enough to break FP4 decode numerics." At the fused MLP kernel, there
is no PTX drift.

**The Phase A Q2 math break therefore cannot be coming from the fused
MLP kernel.** It must be coming from elsewhere in the serving path —
most likely a sibling kernel in the same CuTe DSL package
(`kernel.py` attention) whose cache key depends on package-level source
state, OR from a Torch/Inductor / CUDA-graph interaction triggered by
the D3a/Constexpr import-time changes.

Follow-up attention-kernel PTX diagnostic is the next principled step.

## Context

Phase A attempted the `cute.compile() + cutlass.Constexpr[int]` γ-scope
refactor on branch `feat/unreal-kernel-phase-d` on 2026-04-20. The
refactor built cleanly (gate 0 Jupyter pre-flight 4/4, gate 1 Docker
build OK, gate 3 perf +0.88% vs D2e within ±5%), but gate 2 caught a
Q2 math break on Qwen3.5-27B (raw output `"50/5.  12/12."`,
D3a-class broken). The image `nvllm:gb10-phaseA` was rolled back per
spec §Risks; `nvllm:gb10-phaseD2e` remains the shipped baseline.

The original diagnostic hypothesis: the refactor's source edits
produced a different MLIR/PTX than D2e, landing the kernel in a
math-fatal equivalence class. This diagnostic captures PTX from both
images to fingerprint the drift.

The actual result: there is no PTX drift at the MLP kernel.

## Methodology

- **Harness:** `docs/research/phase_a_ptx_diag/harness.py` constructs
  `Phase_D_MLP_Kernel` at Qwen3.5-27B production dims (hidden=5120,
  interm=17408) with the `prefill-legacy` tile preset (tile_s=256,
  tile_k=640, slice_ctas=8) and invokes once with zero-filled tensors.
- **Driver:** `docs/research/phase_a_ptx_diag/run_diagnostic.sh` runs
  the harness inside each image with `CUTE_DSL_KEEP_PTX=1`,
  `CUTE_DSL_KEEP_IR=1`, `CUTE_DSL_KEEP_CUBIN=1`,
  `CUTE_DSL_DUMP_DIR=/workdir/ir_dump`, `CUTE_DSL_NO_CACHE=1`.
- **Images:**
  - `nvllm:gb10-phaseD2e` — image id
    `sha256:d78822420ab8`, baked `.git/HEAD=dc4bc7d6e` (Phase D1 commit;
    image is older than the `2ffda1fb8` D2e-fix commit — see
    "Commit-label caveat" below).
  - `nvllm:gb10-phaseA` — image id `sha256:6e7151ae0ab9`, baked
    `.git/HEAD=316be8c1b`. Contains D3a inline tile registry
    + `_tile_presets.py` sibling module + Constexpr γ-scope refactor
    on top.
- **DSL version:** `nvidia-cutlass-dsl==4.4.2` in both images.
- **Normalizer + diff:** `docs/research/phase_a_ptx_diag/diff_ptx.py`
  normalizes kernel-name suffixes + `.loc` directives before
  unified-diffing and tags each changed line by PTX-instruction-family
  category.

## Artifacts

- Raw D2e dump:       `d2e/` — 3 DSL artifacts + env.txt + harness log
- Raw Phase A dump:   `phaseA/` — 3 DSL artifacts + env.txt + harness log
- Diff output:        `diff/ptx_side_by_side.txt` (5-line header, 0 hunks)

## Raw result

### Identity hashes

| File         | D2e                              | Phase A                          | Match? |
|--------------|----------------------------------|----------------------------------|--------|
| `*.sm_121a.ptx`    | `71b8d466021cbf1263b58e28512e4c05` | `71b8d466021cbf1263b58e28512e4c05` | **YES — byte-identical** |
| `*.sm_121a.cubin`  | `f7ee06da3b618bd8180d143dc57271dd` | `f7ee06da3b618bd8180d143dc57271dd` | **YES — byte-identical** |
| `*.mlir`           | `5048fda6c1e99ec2e71e88a305238b0f` | `2263dd5541735b83c5bac24ddd31a77d` | NO (1275 vs 1282 lines) |

### Size drift

| File        | D2e bytes | Phase A bytes | Δ  |
|-------------|-----------|---------------|-----|
| PTX         | 59080     | 59080         | 0  |
| CUBIN       | 47408     | 47408         | 0  |
| MLIR        | 93323     | 93680         | +357 |

### diff_ptx.py category counts (PTX)

```
# d2e lines:     1905
# phaseA lines:  1905
# category counts (changed lines, D2e->Phase A):
(empty — zero changed lines)
```

### Source-code differences (for the record)

mlp_kernel.py md5 D2e image: `d8630cab2bf39723dbe2aed2e1fbe42e`
mlp_kernel.py md5 Phase A image: `b7686547840f859ed981cf7673681404`
Phase A image has 11 `Constexpr|cutlass.Int32.*tile` matches in mlp_kernel.py;
D2e image has 0.

`_backend.py` differs by the tile-preset-resolver wiring (adds
`_tile_presets.resolve_tile_preset_from_env()` call + passes
tile_s/tile_k/slice_ctas as explicit kwargs when constructing
`Phase_D_MLP_Kernel`).

Phase A image has a sibling file `_tile_presets.py` that does not exist
in the D2e image.

## Commit-label caveat

The `nvllm:gb10-phaseD2e` image's baked `.git/HEAD` is `dc4bc7d6e`
(Phase D1 — "ship gate fails"), which predates the `2ffda1fb8` D2e
weight_global_scale fix commit. This image has historically produced
correct GSM8K 8/8 in all tests, so either:
- the image was built from D1 HEAD with uncommitted D2e-fix changes on
  top of the working tree, or
- the D2e fix was a no-op at the compiled-kernel level and the pre-fix
  image happens to be numerically equivalent.

Either interpretation is consistent with the PTX-identity finding:
the shipped-correct kernel and the Phase A-broken kernel produce the
same PTX, so the math break must come from outside the MLP kernel.

## What this falsifies

The following memory claims are overturned by this evidence:

- **`memory:project_phase_a_bootstrap_drift` (pre-2026-04-20-afternoon):**
  "Changing the kernel signature from Int32 to Constexpr[int]
  necessarily produces a different MLIR (and PTX) than D2e, putting
  the kernel in a new equivalence class. This class happened to be
  math-fatal on FP4 for Qwen3.5-27B Q2."
  — The MLIR does differ (~357 bytes), but it lowers to identical PTX.
  So the "new equivalence class" hypothesis at the MLP kernel is wrong.

- **`memory:feedback_cute_source_sensitivity`:**
  "Any runtime-Python code addition to the same module that hosts a
  `@cute.kernel`-decorated function can perturb the compiled PTX
  enough to break FP4 decode numerics" — where "the compiled PTX"
  implicitly meant the modified kernel's PTX.
  — The observed evidence is consistent with a weaker claim: source
  additions to a DSL package can perturb PTX for *some* kernel in
  that package, but not necessarily the one whose source was edited.
  The MLP kernel survived source additions without PTX drift; the
  math break came from somewhere else in the package or the
  surrounding runtime.

These entries should be updated to reflect the PTX-level evidence.

## Hypotheses for the real drift source

Ranked by prior likelihood:

1. **CuTe paged attention kernel** (`vllm/v1/attention/backends/cute_paged/kernel.py`).
   Same DSL package as `mlp_kernel.py`; CuTe DSL's JIT cache-key may
   include package-level source state, so adding `_tile_presets.py`
   and editing `mlp_kernel.py` could shift the attention kernel's
   cache key into a different PTX equivalence class — even though
   the attention-kernel source is byte-identical between images.
   **Testable:** extend this diagnostic with an attention-only harness.

2. **Torch / Inductor compilation path divergence.** Import-time
   side effects from the Constexpr refactor (eager `cute.compile()`
   at `__init__` vs lazy on first `__call__`) could change the
   torch-compile graph partitioning or fusion-pass decisions for
   the non-CuTe ops surrounding the fused MLP.

3. **Other CuTe DSL kernel in the package** (RMSNorm-level helpers,
   epilogue kernels, etc.) — same mechanism as (1), different target.

4. **CUDA-graph capture state** — Phase A's eager `cute.compile` in
   `__init__` runs strictly before graph capture begins; D2e's lazy
   compile runs during first forward pass. If graph capture interacts
   with the DSL compile's side effects (JIT cache writes, scratch
   buffers, etc.), Phase A and D2e could see different captured state.

## Next step

Build a parallel attention-kernel PTX-diff diagnostic to test
hypothesis (1). The harness structure from this task is directly
reusable — only the kernel-construct-and-call shape needs to change.
`diff_ptx.py` works as-is.

## How to reproduce

```bash
# 1. Confirm both images exist locally.
docker image inspect nvllm:gb10-phaseD2e > /dev/null
docker image inspect nvllm:gb10-phaseA   > /dev/null

# 2. Run the harness under each image.
./docs/research/phase_a_ptx_diag/run_diagnostic.sh \
    nvllm:gb10-phaseD2e /tmp/ptx_dump_d2e
./docs/research/phase_a_ptx_diag/run_diagnostic.sh \
    nvllm:gb10-phaseA /tmp/ptx_dump_phaseA

# 3. Move dumps to the evidence location.
mkdir -p benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/{d2e,phaseA}
cp -r /tmp/ptx_dump_d2e/*     benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/d2e/
cp -r /tmp/ptx_dump_phaseA/*  benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/phaseA/

# 4. Confirm identity.
md5sum benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/d2e/*.ptx \
       benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/phaseA/*.ptx

# 5. Run categorized diff (will produce empty hunks).
D2E_PTX=$(ls benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/d2e/*.ptx | head -1)
A_PTX=$(ls benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/phaseA/*.ptx | head -1)
python3 docs/research/phase_a_ptx_diag/diff_ptx.py \
    "$D2E_PTX" "$A_PTX" \
    benchmarks/nvllm/traces/phase_a_ptx_diag/2026-04-20/diff/ptx_side_by_side.txt
```

## References

- Rollback memory: `project_phase_a_bootstrap_drift`
- Source-hash sensitivity memory: `feedback_cute_source_sensitivity`
  (needs update — see "What this falsifies" above)
- Phase D branch status: `project_phase_d_inflight`
- Spec (Phase A design, rolled-back):
  `docs/superpowers/specs/2026-04-20-phase-d-cute-compile-constexpr-design.md`
- Plan (Phase A execution, rolled-back):
  `docs/superpowers/plans/2026-04-20-phase-d-cute-compile-constexpr.md`
- Plan (this diagnostic):
  `docs/superpowers/plans/2026-04-20-phase-a-ptx-diff-diagnostic.md`
- Gate-2 Phase A failure evidence:
  `benchmarks/nvllm/traces/cute_paged_mlp_fusion/2026-04-20-phase-A-constexpr/prefill-legacy/`
  (same `nvllm:gb10-phaseA` image — matched by baked
  `.git/HEAD=316be8c1b44b60956739912f320535729ff0af28`)

## Pinned source references

For each code reference in this doc, the authoritative line-anchor is
given below. Line numbers are pinned to a specific commit hash so
they remain resolvable even if the tree moves.

**Committed code (branch `feat/unreal-kernel-phase-d`):**

- `Phase_D_MLP_Kernel.__init__` signature:
  `vllm/v1/attention/backends/cute_paged/mlp_kernel.py:L301-L307` (commit `316be8c1b`)
- `Phase_D_MLP_Kernel.__call__` signature:
  `vllm/v1/attention/backends/cute_paged/mlp_kernel.py:L397-L417` (commit `316be8c1b`)
- Lazy `cute.compile` site (first-call JIT):
  `vllm/v1/attention/backends/cute_paged/mlp_kernel.py:L483` (commit `316be8c1b`)
- D3a inline `_TILE_PRESETS` registry + `_resolve_tile_preset`:
  `vllm/v1/attention/backends/cute_paged/mlp_kernel.py:L76-L99` (commit `316be8c1b`)

**Historical D3a commits (when the inline registry + assert extensions landed):**

- `600717948` — `feat(cute-paged): Phase D3a — tile-preset registry + resolver`
- `175ee2b85` — `feat(cute-paged): Phase D3a — wire __init__ through tile-preset resolver`
- `adad1b0f2` — `feat(cute-paged): Phase D3a — preset-name context in kernel-construct asserts`

**Historical D2e commit (shipped MLP fusion baseline):**

- `2ffda1fb8` — `fix(cute-paged): Phase D2e — apply NVFP4 weight_global_scale in MLP kernel`

**Working-tree-only state at Phase A image build (NOT committed):**

The Phase A image (`nvllm:gb10-phaseA`, id `sha256:6e7151ae0ab9`) was
built from working-tree state on branch `feat/unreal-kernel-phase-d`
at HEAD `316be8c1b` plus the following uncommitted additions:

- `vllm/v1/attention/backends/cute_paged/_tile_presets.py` — new
  sibling module (Path B workaround to avoid inline registry perturbing
  `mlp_kernel.py`). Image-source md5 is recorded as the file md5 in
  the image; run `md5sum` of the in-image file to verify a future
  rebuild matches.
- `vllm/v1/attention/backends/cute_paged/_backend.py` — modified to
  call `resolve_tile_preset_from_env()` and pass `tile_s/tile_k/slice_ctas`
  explicitly to `Phase_D_MLP_Kernel.__init__`. Authoritative anchor is
  the md5 of `_backend.py` inside the image (see Source-code
  differences section above).
- `vllm/v1/attention/backends/cute_paged/mlp_kernel.py` — Constexpr
  γ-scope refactor (11 `Constexpr|cutlass.Int32.*tile` sites). Image
  md5 `b7686547840f859ed981cf7673681404` (vs D2e image md5
  `d8630cab2bf39723dbe2aed2e1fbe42e`).

**DSL env-var source** (in the `nvidia-cutlass-dsl==4.4.2` package,
not in this repo):

- `cutlass/base_dsl/env_manager.py` — `EnvironmentVarManager.__init__`
  reads `{prefix}_KEEP_PTX`, `{prefix}_KEEP_IR`, `{prefix}_KEEP_CUBIN`,
  `{prefix}_DUMP_DIR`, `{prefix}_NO_CACHE`. The concrete `CuTeDSL`
  subclass sets `prefix="CUTE_DSL"` (in `cutlass/cutlass_dsl/cutlass.py`
  class `CuTeDSL`).
