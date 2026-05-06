# NVLLM Design Tenets

## Preamble

Read this before adding or materially changing any nvllm kernel,
graph-captured op, CUDA graph replay path, NVFP4 path, or numerics-sensitive
fused op. This guide records invariants learned from debug arcs. It is not a
general debugging checklist and does not replace `AGENTS.md`, `CLAUDE.md`, or
the source-specific attribution rules in `docs/kernel-insights/`.

If a requested change conflicts with a tenet here, refuse the conflicting
part and explain why.

## How To Update This File

- Add a tenet only when a debug session was burned on it and at least one
  committed receipt exists: a research directory, kernel-insights doc, trace
  summary, code commit, or committed source file.
- Private auto-memory files are not receipts. They can explain history, but
  the canonical evidence must live in the repo.
- IDs are stable. Add new IDs in insertion order within each section and never
  renumber. Supersede old tenets with `Status: superseded by <ID>` and a short
  note instead of deleting them.
- The failure taxonomy is append-only within each arc.
- Keep this file under the domain-guide budget in
  `docs/contributing/editing-agent-instructions.md`.

## Graph Replay Correctness

### G-1
**Title:** Cooperative grid-wide barriers require cooperative launch.
**Rule:** Any kernel using a global atomic counter plus spin-wait,
cooperative-groups grid sync, or another cross-CTA rendezvous MUST launch
cooperatively and prove its grid fits the resident CTA cap.
**Why:** Non-cooperative launch does not guarantee CTA co-residency. A
non-resident CTA cannot reach the rendezvous, so resident CTAs can spin forever.
**How to apply:** Search the kernel for global counters, spin loops, grid sync,
or cross-CTA barriers. Verify `cooperative=True` at launch and check
`CTA_count <= resident_CTAs_per_SM * SM_count`.
**First observed / receipts:** 2026-04-23,
`docs/research/phase-e-task17-beta-coop-smoke-2026-04-23/`.

### G-2
**Title:** Captured dispatch must not mutate runner attributes.
**Rule:** Code near `dispatch_cudagraph` MUST NOT mutate `self` attributes or
other runner-owned state as part of a captured dispatch probe.
**Why:** State mutation in the capture/replay hot path perturbed FULL replay
coherence during the FULL graph spike.
**How to apply:** Grep changed dispatch paths for `setattr(self, ...)`,
`self.<name> =`, counters, and logging side effects. Move probes outside the
hot path or make them one-shot summaries outside capture.
**First observed / receipts:** 2026-04-30,
`docs/research/2026-04-29-full-graph-spike/`.

### G-3
**Title:** Captured ops must not synchronize to host.
**Rule:** A graph-captured op MUST NOT call `.item()`, `.cpu()`, blocking
copies, or any other host-device synchronization.
**Why:** Host synchronization inside capture invalidates the CUDA stream
capture.
**How to apply:** Grep captured-op bodies and helpers for sync APIs before
testing replay. Replace runtime scalar inspection with tensor-side checks or
post-run summaries.
**First observed / receipts:** 2026-04-29,
`docs/research/2026-04-29-full-graph-spike/`.

### G-4
**Title:** Mutating custom ops need a downstream graph reader.
**Rule:** A custom op declaring `mutates_args` MUST feed at least one value read
by the captured graph after the op runs.
**Why:** If no downstream graph node observes the mutation, graph compilation
can dead-code-eliminate the op even though the Python intent was side-effectful.
**How to apply:** Inspect the captured FX graph or replay artifact and verify
the op exists and a later graph node consumes the mutated buffer.
**First observed / receipts:** 2026-04-26,
`docs/research/2026-04-29-full-graph-spike/`.

### G-5
**Title:** Custom-op Python bodies run at capture, not every replay.
**Rule:** A custom op body MUST NOT be used for per-replay instrumentation or
per-token Python side effects under full graph capture.
**Why:** The Python body executes while tracing/capturing; replay runs the
captured work, not the Python body.
**How to apply:** Put replay-time diagnostics in graph-safe tensors, kernel
outputs, or post-run checks instead of Python logs inside the op body.
**First observed / receipts:** 2026-04-29,
`docs/research/2026-04-29-full-graph-spike/`.

### G-6
**Title:** Splitting ops create PIECEWISE boundaries intentionally.
**Rule:** Adding an op to `_attention_ops` or another splitting-op registry
MUST be intentional and documented because it creates a PIECEWISE boundary.
**Why:** A splitting op runs eagerly between captured graph pieces; accidental
registration changes graph structure and can hide replay bugs.
**How to apply:** In PR review, check new custom-op registrations against the
compile config and explain why each splitting boundary is required.
**First observed / receipts:** 2026-04-30,
`docs/research/2026-04-29-full-graph-spike/`.

### G-7
**Title:** Manual KV updates must enter through a registered dispatcher.
**Rule:** KV updates outside canonical `Attention.forward` MUST call a
registered `torch.ops.*` dispatcher, not a raw Python helper, when they need to
survive opaque-attention CUDA graph capture.
**Why:** A raw Python call can disappear during capture or DCE; the registered
dispatcher gives the compiler a graph-visible op with explicit mutation
metadata.
**How to apply:** Verify the update path uses `direct_register_custom_op` (or
an existing registered op), has `mutates_args` and `fake_impl`, and appears in
the captured graph.
**First observed / receipts:** 2026-04-30,
`docs/research/2026-04-29-full-graph-spike/`.

### G-8
**Title:** Fullgraph custom ops require explicit fake implementations.
**Rule:** vLLM fullgraph custom ops MUST be registered with
`direct_register_custom_op` and an explicit `fake_impl`; `@disable` and
`allow_in_graph` are not sufficient substitutes.
**Why:** The op-registration repro showed the direct registration path works
under `torch.compile(fullgraph=True)` while decorator-based attempts did not.
**How to apply:** Add a small repro or unit probe for new graph ops before
starting a Docker rebuild.
**First observed / receipts:** 2026-04-25,
`docs/research/phase_f1_opaque_gate/op_registration_repro.py`.

### G-9
**Title:** Opaque op alone is not enough.
**Rule:** Collapsing work behind one opaque op is necessary but not sufficient;
outer Python gates, duplicate launch sites, and side-effect launches must also
collapse to one graph-visible path.
**Why:** Replay coherence still failed when the op was opaque but Python-side
control flow could dual-fire or pick a different launch site.
**How to apply:** Grep the feature flag and launch helper names. Prove there is
one runtime launch site for each graph-captured path.
**First observed / receipts:** 2026-04-25,
`docs/research/phase_f1_opaque_gate/`.

## Layer And Kernel Contracts

### L-1
**Title:** Decoder layers return the fused-residual contract.
**Rule:** A decoder layer returns `(hidden=mlp_out,
residual=residual_post_attn)`, never a pre-summed `residual_final`.
**Why:** The next layer's input RMSNorm fuses `hidden + residual` itself.
Returning a pre-summed residual double-counts at layer N+1.
**How to apply:** Trace every fused-kernel return value by buffer stage before
replacing the Python layer. Confirm against `_forward_static_with_residual`.
**First observed / receipts:** `vllm/nvllm/layers/layernorm.py`.

### L-2
**Title:** NVFP4 reference dequant divides by the loader scale.
**Rule:** NVFP4 reference dequant MUST divide by `weight_global_scale` because
the vLLM loader inverts the global scale at load time.
**Why:** Multiplying by the already-inverted scale makes per-site math look
plausible while end-to-end output degrades.
**How to apply:** When writing references, name the value as loader-visible
scale and test against the live loaded tensors.
**First observed / receipts:** `docs/research/phase_e2_beta_math/`.

### L-3
**Title:** MMA A fragments are row-interleaved.
**Rule:** Tests for PTX A operand fragments MUST use non-uniform values that
catch row-interleaved layout mistakes.
**Why:** Uniform fixtures let the wrong fragment layout pass.
**How to apply:** Seed each row and lane differently in MMA layout tests and
compare against a Python reference.
**First observed / receipts:** `docs/research/phase_a_attn_ptx_diag/`.

### L-4
**Title:** `lm_head` stays BF16 unless quality evidence says otherwise.
**Rule:** Do not quantize `lm_head` to FP4 by default; any change must carry
model-quality evidence for the target model and serving path.
**Why:** Feasibility work found the implementation risk and quality risk out of
proportion to the expected gain.
**How to apply:** Keep `lm_head` BF16 in new quant paths unless the PR includes
a committed quality gate and the rollback path.
**First observed / receipts:**
`docs/kernel-insights/2026-04-22-d3-feasibility.md`.

### L-5
**Title:** Fused-kernel buffers are named by stage.
**Rule:** Fused-kernel inputs and outputs MUST be named by buffer stage, not
only by semantic role.
**Why:** Bugs hid behind names like `residual` when the kernel actually needed
`residual_buf`, `residual_output`, or another concrete stage.
**How to apply:** In new fused kernels, document the producer and consumer for
each hidden/residual buffer at the call site.
**First observed / receipts:** `docs/research/phase_e2_beta_math/`.

## Numerics And Quantization

### N-1
**Title:** Verify kernels against references, not decoded vibes.
**Rule:** Kernel correctness MUST be checked with `CUTE_DEBUG_FUSION` or an
equivalent tensor-level harness against a Python reference; Q2 text on
distilled models is insufficient.
**Why:** Distilled models can produce plausible text while sub-ULP reorder drift
flips knife-edge argmax decisions.
**How to apply:** Add a deterministic tensor-level diff before relying on
end-to-end text quality.
**First observed / receipts:** `docs/research/phase_a_mlp_math_harness/`.

### N-2
**Title:** No silent partial-coverage fallback.
**Rule:** A kernel gate MUST NOT silently fall back to a partial-coverage kernel.
It must tighten the gate, fall through to a complete-coverage path, or fail
loudly.
**Why:** Silent fallback made correctness look model-dependent instead of
surfacing an unsupported shape.
**How to apply:** Review every feature gate for unsupported shapes and ensure
the selected fallback covers the entire required operation.
**First observed / receipts:** `docs/research/uber_kernel_migration/`.

### N-3
**Title:** Explicit `None` does not trigger `kwargs.get` defaults.
**Rule:** CuTe kernel kwargs MUST normalize explicit `None` with
`if x is None: x = default`; do not rely on `kwargs.get(k, default)`.
**Why:** A present key with value `None` bypasses the default and reaches the
kernel path.
**How to apply:** Grep changed launch wrappers for `kwargs.get` and audit any
argument whose caller can pass `None`.
**First observed / receipts:** `docs/research/uber_kernel_migration/`.

## Review Gates

### D-1
**Title:** Verify the runtime consumer before relying on config.
**Rule:** Before relying on an env var, model class, or model config dimension,
MUST grep for the runtime consumer or read the live model `config.json`.
**Why:** Build-time exposure and docs do not prove the serve-time path honors
the value.
**How to apply:** Include the consumer path or config field in the PR notes for
new graph/kernel gates.
**First observed / receipts:** `docs/research/2026-04-29-full-graph-spike/`.

### D-2
**Title:** Empty env vars are not unset.
**Rule:** Serve-time env reads that treat empty as unset MUST use
`os.getenv(name) or default`, not only `os.getenv(name, default)`.
**Why:** `os.getenv(name, default)` returns `""` when the variable exists but is
empty.
**How to apply:** Grep changed env reads and decide whether empty string is a
valid value. If not, normalize with `or default`.
**First observed / receipts:** `docs/research/2026-04-29-full-graph-spike/`.

## Failure Taxonomy

| Arc | Failure | Symptom | Prevented by | Receipt |
| --- | ------- | ------- | ------------ | ------- |
| NVFP4 bringup | Reference dequant used the loader scale in the wrong direction | Per-site math plausible, end-to-end output degraded | L-2 | `docs/research/phase_e2_beta_math/` |
| NVFP4 bringup | Distilled model text masked tensor-level drift | Q2 output looked coherent while argmax changed | N-1 | `docs/research/phase_a_mlp_math_harness/` |
| NVFP4 bringup | `lm_head` FP4 was scoped despite quality risk | Small projected gain with high correctness risk | L-4 | `docs/kernel-insights/2026-04-22-d3-feasibility.md` |
| FULL-graph blocker | `mutates_args` custom op was DCE'd | Captured graph omitted the intended side-effect op | G-4 | `docs/research/2026-04-29-full-graph-spike/` |
| FULL-graph blocker | `.item()` ran inside an opaque-op body | CUDA stream capture invalidated | G-3 | `docs/research/2026-04-29-full-graph-spike/` |
| FULL-graph blocker | Opaque op left outer gates and launch sites split | Replay coherence failed despite an opaque op | G-9 | `docs/research/phase_f1_opaque_gate/` |
| FULL-graph blocker | `self` mutation near `dispatch_cudagraph` perturbed replay | FULL replay became non-deterministic | G-2 | `docs/research/2026-04-29-full-graph-spike/` |
| beta-coop bringup | Non-cooperative launch used a global atomic barrier | Decode hung | G-1 | `docs/research/phase-e-task17-beta-coop-smoke-2026-04-23/` |
| beta-coop bringup | Fused layer returned `residual_final` as residual | Next layer double-counted input residual | L-1 | `vllm/nvllm/layers/layernorm.py` |

## Cross-References

- `AGENTS.md` section 1: contribution policy.
- `AGENTS.md` section 3: kernel source attribution.
- `AGENTS.md` section 4: performance evidence standard.
- `docs/contributing/editing-agent-instructions.md`: budget and guide rules.
- `docs/kernel-insights/`: source and design receipts.
- `docs/research/`: debug-arc receipts.
- `m0at/rvllm` v3 SPEC.md at
  `3f7969bd768db1bcb03955978d1138ebdd9a6229`: design-pattern reference only,
  not a code dependency.
