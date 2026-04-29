# C2 Diagnostic — β-coop vs Legacy under PIECEWISE+graphs

**Date:** 2026-04-26
**Branch base:** `feat/uber-kernel-migration` HEAD `90b06d5df`
**Diagnostic branch:** `diag/c2-beta-coop-vs-legacy` (to be created)
**Author session ID:** brainstorming session 2026-04-26 PM
**Status:** spec — implementation in next session

---

## TL;DR

This spec defines an env-gated diagnostic probe that answers one binary question:

> Under `cudagraph_mode=PIECEWISE` in dual-fire mode, do β-coop's outputs match the legacy path's outputs at every full-attention layer of Qwen3.5-27B-NVFP4?

The answer determines the next-session design for the C2 uber-kernel migration:

| Probe result | Diagnosis | Next design step |
|---|---|---|
| Match within tolerance | β-coop kernel is graph-replay-correct | Design the consume-gate op pattern; gibberish was an op-pattern issue under graphs, not a kernel issue |
| Diverge | β-coop kernel itself is graph-replay-broken | Investigate cooperative-launch + atomic-counter spin under graph replay before any consume-gate work |

Plus a 5-minute sanity rung (rung 0) that rules out NVFP4+graphs+paged-alone as a confound, and a stashed companion harness (`CUTE_C2_DIAG_EAGER`) for if the primary probe is inconclusive.

---

## Background

### How we got here

`feat/uber-kernel-migration` HEAD `90b06d5df` ships C1 + C1.5 + C2 plumbing. C2 is correctness-positive in dual-fire under PIECEWISE+graphs but solo β-coop produces gibberish under PIECEWISE+graphs.

Last session (2026-04-26 PM) diagnosed two coupled bugs at the FX-graph level (full findings in `docs/research/uber_kernel_migration/2026-04-26-consume-gate-dce-and-graph-capture.md`):

1. `cute_residual_mirror` is DCE'd despite `mutates_args=["residual_buf"]` because no graph op reads `impl.residual_buf` as an explicit tensor input.
2. The `if getattr(impl, "_fusion_active", False)` consume gate at `qwen3_5.py:466-476` specialises to the else-branch at trace time. Captured FX always runs the legacy Python o_proj + post_attn_LN.

The B-fix attempt at commit `514b88c6f` (reverted) introduced backend opaque ops (`cute_attn_consume`, `cute_post_attn_ln_dispatch`) modeled on `cute_phase_e_dispatch` and made both fixes work under `cudagraph_mode=PIECEWISE` *with cudagraph_mode=NONE* — but produced gibberish under `cudagraph_mode=PIECEWISE` *with graph capture enabled*. Three internal pivots within the B-fix all reproduced the same graph-mode gibberish.

### Why we don't know if it's a kernel bug or an op-pattern bug

Under PIECEWISE+graphs in **dual-fire**, β-coop fires but its outputs are unobserved by the captured graph. The legacy path reconstructs the answer. So we have no signal as to whether β-coop's outputs would have been correct.

Under PIECEWISE+graphs in **B-fix solo**, β-coop's outputs ARE the only producers, and we get gibberish. But that could be:
- (kernel) β-coop's outputs are wrong under graph replay (would be wrong in dual-fire too, but unobserved).
- (op-pattern) β-coop's outputs are correct under graph replay, but the op-pattern that consumes them via opaque ops + phantom deps + registry-lookup interacts badly with PIECEWISE segment boundaries.

Both hypotheses fit the evidence. The next design step (consume-gate redesign vs kernel investigation) depends on which is true.

### Upstream check (per CLAUDE.md §1)

Searched `vllm-project/vllm` issues for `NVFP4 cuda graph`, `nvfp4 cudagraph`, `llm-compressor graph capture`, `NVFP4 piecewise compilation`. Three SM120/121 + NVFP4 + CUDA-graph issues exist:

- [#35659](https://github.com/vllm-project/vllm/issues/35659) — `cudaErrorIllegalAddress` on FlashInfer-CUTLASS NVFP4 MoE under sustained load. **Different GEMM stack** (FlashInfer-CUTLASS vs our own SM120 CUTLASS), different attention backend, different SM tier.
- [#38208](https://github.com/vllm-project/vllm/issues/38208) — CUDA Illegal Instruction during graph capture with Nemotron Hybrid Mamba-2 NVFP4 on sm_121, using **Marlin GEMM + Triton attention + Mamba-2 mixer**. Not our kernel stack at all.
- [#37060](https://github.com/vllm-project/vllm/issues/37060) — sm110, different SM family.

**None match our stack** (our own SM120 CUTLASS NVFP4 GEMM + CuTe paged attention + β-coop). The upstream NVFP4+graph fragility is in adjacent stacks; symptom class is crash/illegal-instruction, not silent gibberish. Combined with our own dual-fire path working correctly under PIECEWISE+graphs on the same NVFP4 weights, "our gibberish is not a generic NVFP4+graphs platform issue" is the strongest available conclusion without a direct probe.

### Container baseline at design time

Snapshot taken 2026-04-26 16:05 before this spec was written:

```
container: nvllm  up 40m  image=nvllm:gb10
model:     ig1/Qwen3.5-27B-NVFP4
attn:      CUTE_PAGED
cudagraph: PIECEWISE
flashinfer-autotune: false
kv-cache:  fp8_e4m3
max-num-seqs: 4
β-coop kernel compiled at 16:03:53; last completion 16:05:06 with 200 OK
```

Confirms PIECEWISE+graphs+dual-fire+β-coop is healthy in production right now. Container stopped after snapshot to free unified memory before next-session implementation.

---

## Decision context

### Decisions taken in this brainstorming session

| Question | Choice | Reason |
|---|---|---|
| Where in the architectural option space | **Option 1 (β-coop writes to graph-observable sinks)** | Option 1 is the only choice that moves toward "kernel owns the layer's data path; Python is plumbing." |
| Within Option 1 | **(a) Backend-only opaque ops** | Same shape as `cute_phase_e_dispatch` which works in production; doesn't change framework op signature |
| Strategy for de-risking PIECEWISE+graphs failure | **(i) Diagnose first, design second** | The B-fix was already shape (a) and broke under graphs; we need to know if the kernel or the op-pattern is at fault before redesigning |
| Probe shape | **(P2) Comparison + parallel stashed (c) eager-replay harness** | (P1) suffices for the primary signal; stashed (c) is no-rebuild Python and worth having available |

### Decisions deferred

- The actual consume-gate op redesign (any of the three architectural answers in the migration findings doc). Deferred until probe results inform direction.
- Whether to delete the diagnostic after use or keep it as an env-gated harness per `feedback_keep_debug_harnesses`. Decide after results.

---

## Architecture (Section 1)

### Scope (in)

- One comparison call site at `qwen3_5.py:466-476`, env-gated `CUTE_C2_DIAG=1`.
- One companion eager-replay harness at `cute_paged/_c2_eager_replay.py`, env-gated `CUTE_C2_DIAG_EAGER=1`. Off by default.
- Per-layer + per-step logging to stderr; on-divergence dump to `/tmp/c2_diag/`.
- Tolerances: `atol=1e-2`, `rtol=1e-2`. Report L∞, median(|rel|), and which-layer-first-diverges.
- **Sanity rung 0** (5-min probe before main diagnostic): disable β-coop launch entirely (`CUTE_FUSION_DISABLE=1` or whatever env is wired in the cute_paged backend — verify at probe-wiring time). If gibberish → bigger problem than C2, halt and root-cause that first. If correct → rules out NVFP4+graphs as a confound.

### Scope (out)

- No new opaque ops, no kernel changes, no framework-op signature changes.
- No production code paths touched when `CUTE_C2_DIAG` unset.
- No multi-rank / TP behavior — single-GPU only.
- No automatic regression test harness — this is a one-shot diagnostic, not CI.
- No FULL-graph mode testing per `project_full_graph_blocked.md`.

### Branch policy

New branch off `feat/uber-kernel-migration` HEAD `90b06d5df`. Branch name `diag/c2-beta-coop-vs-legacy`. Single commit when probe is wired; one or more commits as we use it. **Do NOT merge to `feat/uber-kernel-migration` until results inform the next design** — keeps the migration branch's HEAD clean as a known-state ref.

### Success criterion for the probe itself (not the diagnosis)

The probe is correctly wired when running serve-cute under default config + `CUTE_C2_DIAG=1` produces:
1. No *unintended* crash, hang, or OOM during graph capture or replay. (A divergence-triggered `RuntimeError` after a dump is the *intended* outcome on a real divergence; that's not a crash.)
2. Stderr lines for every full-attn layer at every step, format `[C2_DIAG] L=N step=M  hidden L∞=X.XXe-Y rel_med=X.XXe-Y  residual L∞=X.XXe-Y rel_med=X.XXe-Y  OK`.
3. On first divergence above tolerance: dump file at `/tmp/c2_diag/`, RuntimeError raised with layer+step in message, container logs unambiguous.

---

## Components (Section 2)

### `vllm/nvllm/models/qwen3_5.py` — comparison call site (additive edit)

Single block inserted between the consume-gate (~lines 466-476) and the post-attn-LN gate (~lines 490-496), guarded by `if os.getenv("CUTE_C2_DIAG") == "1"`. Conceptual shape:

```python
if os.getenv("CUTE_C2_DIAG") == "1" and impl is not None and \
   getattr(impl, "_fusion_bound", False) and self.layer_type == "full_attention":
    from vllm.v1.attention.backends.cute_paged import _c2_diag
    _c2_diag.compare_and_log(
        layer_idx=self.layer_idx,
        step_idx=_c2_diag.next_step_idx(),
        nat=nat,
        legacy_hidden=hidden_states,
        legacy_residual=residual,
        beta_rmsnorm_output=impl.rmsnorm_output,
        beta_residual_output=impl.residual_output,
    )
```

Reads only graph-output tensors and impl buffers it doesn't mutate; no DCE concern. Lives outside any opaque op, runs in eager Python that frames the captured-graph segments.

### `vllm/v1/attention/backends/cute_paged/_c2_diag.py` — primary probe (new file)

Pure Python module. Public API:

- `compare_and_log(layer_idx, step_idx, nat, legacy_hidden, legacy_residual, beta_rmsnorm_output, beta_residual_output)` — computes L∞ + median(rel) on `[:nat]` slices, logs one line per call, dumps + raises on first divergence above tolerance read from `CUTE_C2_DIAG_TOL_ATOL` (default `1e-2`) and `CUTE_C2_DIAG_TOL_RTOL` (default `1e-2`).
- `_dump_on_divergence(...)` — writes `torch.save` bundle to `${CUTE_C2_DIAG_DUMP_DIR:-/tmp/c2_diag}/layer{N}_step{S}.pt`. One-shot: first divergence ends the run.
- `next_step_idx()` — module-local counter that increments each call from layer 0; used only for log readability.
- `assert_no_flashinfer_autotune()` — startup safeguard called once when module imports under `CUTE_C2_DIAG=1`. Reads compilation config from `vllm.config`, raises if `enable_flashinfer_autotune=true` (host-reboot risk per `feedback_flashinfer_autotune_sm120`).

### `vllm/v1/attention/backends/cute_paged/_c2_eager_replay.py` — stashed companion harness (new file, off by default)

Independent of `_c2_diag.py`. When `CUTE_C2_DIAG_EAGER=1` is set, exposes:

- `EagerReplayHook` — class with:
  - `snapshot(impl, hidden_pre_qkv, residual, layer_inputs)` — captures β-coop's inputs at the consume-gate site (host-mirrored to pinned tensors so they survive subsequent graph activity).
  - `replay_and_compare(impl, captured_inputs)` — calls `impl.kernel.run_beta_coop_full(...)` (`vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2685`) on the same inputs in eager mode, compares outputs against the graph-replayed `impl.rmsnorm_output / residual_output`.

Wired into `qwen3_5.py`'s gated block via a separate `os.getenv("CUTE_C2_DIAG_EAGER") == "1"` check. `CUTE_C2_DIAG=1` and `CUTE_C2_DIAG_EAGER=1` are independent flags; either can run alone or both together.

### Env variables

| Var | Default | Effect |
|---|---|---|
| `CUTE_C2_DIAG` | unset | When `=1`: compare β-coop outputs vs legacy outputs at consume-gate site, log per-layer, dump + raise on divergence. |
| `CUTE_C2_DIAG_EAGER` | unset | When `=1`: also run β-coop in eager on the same inputs and compare graph-replay vs eager outputs. |
| `CUTE_C2_DIAG_TOL_ATOL` | `1e-2` | Override absolute tolerance. |
| `CUTE_C2_DIAG_TOL_RTOL` | `1e-2` | Override relative tolerance. |
| `CUTE_C2_DIAG_DUMP_DIR` | `/tmp/c2_diag` | Override dump directory. |
| `CUTE_C2_DIAG_INJECT_NOISE` | unset | Self-test: when set to a float, adds that constant to `impl.rmsnorm_output[:nat]` immediately before comparison; used to verify probe halts on real divergence. |
| `CUTE_FUSION_DISABLE` (existing, name to verify) | unset | Sanity rung 0: disable β-coop launch entirely; tests paged-only under PIECEWISE+graphs. |

The `CUTE_FUSION_DISABLE` env var name is TBD — verify at probe-wiring time by greping the cute_paged backend for existing fusion-disable knobs. If none exists, add one (small additive change).

### What is NOT touched

- No edits to `_mlp_op.py` (existing custom ops).
- No edits to `phase_e_kernel.py` (β-coop launcher).
- No edits to `attention.py` (framework op).
- No new opaque ops, no kernel changes, no graph-mode code paths.

---

## Data flow (Section 3)

### Per-step lifecycle (under `CUTE_C2_DIAG=1`, dual-fire, PIECEWISE+graphs)

```
Step boundary (decode of one token, batch of nat sequences)
│
├─ Graph segment N starts (captured once at warmup, replayed)
│  │
│  ├─ Layer 0: input_LN
│  ├─ Layer 0: cute_residual_mirror(impl.residual_buf, residual)
│  ├─ Layer 0: self_attn  ──► unified_attention_with_output (paged)
│  │                          β-coop kernel also fires inside same op
│  ├─ Layer 0: post_attention_layernorm (Python, runs always in dual-fire)
│  ├─ [DIAG block, eager Python, between graph ops]
│  ├─ Layer 0: cute_phase_e_dispatch (MLP)
│  ├─ Layer 0: residual + LN for next layer entry
│  └─ ...layers 1..47 same shape (DIAG block fires only on full_attention layers)
│
├─ Graph segment N ends
│
└─ Eager Python boundary (no diag work)
```

The diag block sits inside per-layer eager Python. qwen3_5.py's decoder forward IS eager Python — only specific computation paths (the actual attn/MLP ops) are graph segments. Layer-by-layer iteration is Python, so the diag insertion point runs naturally between captured segments without affecting capture.

### Trigger condition

```python
diag_active = (
    os.getenv("CUTE_C2_DIAG") == "1"
    and impl is not None
    and getattr(impl, "_fusion_bound", False)        # β-coop attached this layer
    and self.layer_type == "full_attention"          # not a linear-attn layer
    and nat > 0                                      # decode has tokens this step
)
```

Per Qwen3.5-27B's stride-4 pattern: 16 full-attn layers across 48 total. We get 16 comparison points per step.

### Tensor pairs compared

| # | Tensor pair | Slice | Producer A (legacy) | Producer B (β-coop) |
|---|---|---|---|---|
| 1 | `hidden_states` vs `impl.rmsnorm_output` | `[:nat]` | Python `post_attention_layernorm` output | Phase C output (post-attn RMSNorm) |
| 2 | `residual` vs `impl.residual_output` | `[:nat]` | Python `residual_post_attn` (legacy LN's residual return) | Phase C residual_output |

Tensors are BF16, shape `[nat, 5120]`.

### Per-call output (stderr line)

```
[C2_DIAG] step=42 L=3 nat=2  hidden  L∞=1.23e-04 rel_med=2.45e-05  residual  L∞=8.91e-05 rel_med=1.10e-05  OK
```

`OK` if both pairs within tolerance. `DIVERGED` if either exceeds.

### On first divergence

```python
torch.save({
    "commit": <git rev-parse HEAD>,
    "cudagraph_mode": <runtime read>,
    "step_idx": ..., "layer_idx": ..., "nat": ...,
    "atol": ..., "rtol": ...,
    "legacy_hidden": hidden_states[:nat].clone(),
    "legacy_residual": residual[:nat].clone(),
    "beta_rmsnorm_output": impl.rmsnorm_output[:nat].clone(),
    "beta_residual_output": impl.residual_output[:nat].clone(),
    "beta_inputs": {
        "hidden_in": <captured at layer entry>,
        "residual_in": <captured at layer entry>,
        "input_gamma": impl.input_gamma.clone(),
        "post_attn_gamma": impl.post_attn_gamma.clone(),
        # ... full set matching run_beta_coop_full's signature
    },
}, f"/tmp/c2_diag/layer{L}_step{S}.pt")
raise RuntimeError(f"[C2_DIAG] diverged: layer={L} step={S} ...")
```

The `beta_inputs` capture is the (b)-style dump that lets us replay offline — including with the stashed (c) eager-replay harness if we wire it later.

### Why first divergence ends the run

- After divergence, β-coop's downstream effects on next layers' inputs are corrupted by definition; further per-layer comparisons lose meaning.
- One dump file is enough for offline forensics.
- Failing loud per `feedback_no_silent_fallbacks`.

### Sanity rung 0 lifecycle

```
CUTE_FUSION_DISABLE=1 (verify name at wiring time) + cudagraph_mode=PIECEWISE
└─ run for one prompt (~256 tokens)
   ├─ if coherent output → rung 0 PASS, proceed to (b)
   └─ if gibberish → rung 0 FAIL, halt, root-cause paged-alone+NVFP4+graphs first
```

---

## Error handling (Section 4)

### Failure modes

| Condition | Action | Rationale |
|---|---|---|
| Divergence above tolerance | Dump bundle, `RuntimeError` with layer/step | The whole point of the probe |
| `impl is None` while `_fusion_bound` was checked | `RuntimeError("[C2_DIAG] impl missing on full-attn layer")` | Gating assumptions are wrong; can't trust other comparisons |
| `impl.rmsnorm_output` or `impl.residual_output` is the wrong shape | `RuntimeError("[C2_DIAG] β-coop buffer shape mismatch: expected [nat,H], got ...")` | β-coop didn't fire as expected; comparing would be garbage-vs-garbage |
| `nat == 0` | Skip silently | Empty-decode step — no data to compare, not an error |
| `CUTE_C2_DIAG=1` set but `_fusion_bound=False` for ALL layers across the first step | Log warning once: `[C2_DIAG] WARNING: no full-attn layer reports _fusion_bound — is dual-fire enabled?` | User set the env var but β-coop isn't attached; probe will produce no output |
| Dump-directory write fails | Log error, still raise the RuntimeError on divergence | Don't lose the failure signal even if forensics dump fails |
| `torch.save` itself fails after divergence | Log error + the comparison stats; raise RuntimeError | Tensor stats from comparison are still in stderr — partial forensics |
| `CUTE_C2_DIAG_EAGER=1` set but no β-coop launcher reachable | `RuntimeError("[C2_DIAG_EAGER] cannot import run_beta_coop_full")` | Stashed harness is opt-in; if you opt in and it can't run, fail loud |
| `CUTE_C2_DIAG_EAGER=1` causes graph capture to fail | Catch `cudaErrorStreamCaptureInvalidated`, log explicitly with hint to disable EAGER mode, re-raise | Per `feedback_item_breaks_cuda_graphs`: host-device sync inside opaque op breaks graphs |

### What the probe explicitly does NOT do

- **No try/except around the comparison itself.** A bare `assert` or numerical-compare bug must not be swallowed (per `feedback_bare_assert_hides_bugs`).
- **No silent fallback to a degraded comparison** (per `feedback_no_silent_fallbacks`).
- **No retry loop on transient errors.** First failure halts.
- **No `try/except` around `get_forward_context()` lookups** — exact pattern `_mlp_op.py:247-296` warns about. Probe reads only state passed in as arguments.

### Probe non-perturbation guarantees

- **No `.item()` calls** anywhere in the comparison code path. Stats computed via tensor ops; scalar-tensor → Python float happens *only inside f-string formatting* of the log line, which runs after the captured graph segment for that layer has already executed (per `feedback_item_breaks_cuda_graphs`).
- **No `.cpu()` / `.to('cpu')` on graph tensors during capture window.** Dump uses `.clone()` to keep tensors on device until after divergence; clone happens on the eager Python side, not inside any graph op.
- **No allocator-pressure spikes from clones.** Clones happen at most once per full diag run (only on first divergence).
- **No reordering of layer compute.** The comparison reads tensors that are already produced by the time the eager Python reaches the diag block.

### Probe failure-isolation policy

If the probe itself crashes, the user must be able to disambiguate "probe broke" from "model broke under graphs." All probe-internal exceptions wrap with a `[C2_DIAG]` prefix and the original exception chained via `raise X from e`.

### Host-safety bounds (per user direction)

The probe must never push the SoC into a state that requires a Spark reboot. Inherits serve-cute's existing safety knobs (commit `2b21f3450` baked `enable_flashinfer_autotune=false`).

| Bound | Mechanism |
|---|---|
| No OOM on unified memory | One dump bundle ever per run; clone size ≤ ~10 MB total. Negligible against 128 GB. |
| No infinite loop / hang | Single comparison per layer per step; first divergence raises; no retries. |
| No driver-crashing kernel state | Probe never launches new kernels (b-only path). The stashed (c) harness is the only place that re-launches β-coop, and only at `CUTE_C2_DIAG_EAGER=1`. |
| No segfault on engine exit | RuntimeError propagates through engine's normal exception path; vLLM V1's EngineCore handles unhandled exceptions cleanly. |
| No /tmp disk-fill | `/tmp/c2_diag/` capped at one dump file per run (~10 MB). User can `rm -rf /tmp/c2_diag` between runs. |
| No flashinfer autotune triggered | `_c2_diag.assert_no_flashinfer_autotune()` runs at module import; raises if enabled. |
| No graph-capture cudaError → driver wedge | Probe code path is read-only graph-output tensors + eager-Python ops outside the captured segment. Stashed (c) is the only graph-capture risk; on `cudaErrorStreamCaptureInvalidated`, fall through to engine error handler. |

**Bottom line:** loud RuntimeError + tensor dump + clear log line; never anything that requires a power-cycle.

---

## Testing (Section 5)

### Pre-flight: probe wires correctly when set

After implementation, before launching real diagnostic runs:

1. **No-op check (probe disabled):** Run serve-cute with `CUTE_C2_DIAG` *unset*. Confirm:
   - Output coherent on standard probe (`"What is the capital of France?"` → `"Paris..."`).
   - No `[C2_DIAG]` log lines appear.
   - Spot-check 5-10 nominal completions for sanity (full GSM8K-50 gate per `feedback_post_quant_sanity` is overkill for a code-gated probe; reserve it for after the diagnostic run if results are clean and we want to confirm no behavioral drift).

2. **Active check (probe enabled):** Run serve-cute with `CUTE_C2_DIAG=1`. Confirm:
   - At least one `[C2_DIAG] step=... L=... OK|DIVERGED` log line per full-attn layer per step.
   - Log lines stop appearing on linear-attn layers.
   - On a 256-token decode, expect ~`255 × 16 ≈ 4080` log lines.

3. **Rung 0 separately:** Run with `CUTE_FUSION_DISABLE=1` (verify env name) and `CUTE_C2_DIAG` *unset*. Confirm:
   - Output coherent (paged-only baseline).
   - No `[C2_DIAG]` lines.

### Self-consistency: forced-divergence injection

Before trusting "no divergence" as a real answer, prove the probe *can* detect divergence:

- Set `CUTE_C2_DIAG_INJECT_NOISE=1.0`. The probe adds 1.0 to `impl.rmsnorm_output[:nat]` immediately before comparison.
- With injection on: confirm probe halts on layer 0 step 0, dumps the bundle, raises RuntimeError, container logs the divergence message.
- With injection off: confirm probe runs without halt (in dual-fire — outputs match within tolerance by construction).

### Result interpretation gate

Before declaring "(b) probe shows match" or "(b) probe shows divergence" as basis for next-session design work:

1. **Match interpretation:** ≥1000 log lines without a `DIVERGED`. Run on TWO prompts (one short, one long ≥256 tokens). Both must pass. Conclusion: "β-coop kernel is graph-replay-correct in dual-fire under PIECEWISE+graphs; the C2 design problem is in the consume-gate op pattern."

2. **Divergence interpretation:** First-divergence dump bundle present at `/tmp/c2_diag/`. Re-run with same prompt + same env to confirm reproducibility. Compare divergence layer/step across runs:
   - Same: deterministic — β-coop kernel has a graph-replay bug.
   - Different: non-deterministic — possibly cooperative-launch / atomic-counter-spin issue per `feedback_cooperative_grid_barrier`.

### Stashed (c) harness verification (only if (b) inconclusive)

- Enable `CUTE_C2_DIAG_EAGER=1`. Confirm harness runs without graph-capture errors.
- Compare graph-replay outputs vs eager-replay outputs of β-coop on the same inputs. If they differ on the same input: β-coop's graph-replay has a state-dependent issue.

### What we don't test

- **No CI / pytest integration.** Probe is meant to be deleted (or stashed) once the C2 redesign is done.
- **No multi-GPU / TP testing.** Single-GPU only.
- **No FULL-graph mode testing.** Per `project_full_graph_blocked.md`.

---

## Open questions / risks

1. **`CUTE_FUSION_DISABLE` env name:** TBD. Verify at probe-wiring time. If not present, add as small additive change.
2. **Step counter source:** `_get_step_idx()` reads model's forward counter if available, else module-local counter. Imperfect under linear-attn-only steps, but only used for log readability; not load-bearing.
3. **Tolerance choice:** `atol=1e-2`, `rtol=1e-2` is calibrated for BF16 unit-roundoff on RMSNorm outputs. May need to widen if β-coop's FP32 reduction-order differs more than expected from the legacy path. If tolerance has to widen above `1e-1`, that itself is a divergence signal worth investigating.
4. **`_fusion_bound` semantic:** Current code uses `_fusion_active` (per-step, mutated inside opaque ops) and `_fusion_bound` (one-shot at attach_fusion). The probe uses `_fusion_bound` because it's stable. Verify at probe-wiring time that `_fusion_bound` is set on every full-attn layer's impl — not just some — when β-coop is attached.

---

## How to pick up next session

1. Branch off `feat/uber-kernel-migration` HEAD `90b06d5df`. Branch name `diag/c2-beta-coop-vs-legacy`.
2. Read this spec.
3. Read `docs/research/uber_kernel_migration/2026-04-26-consume-gate-dce-and-graph-capture.md` for diagnostic baseline.
4. Reference `git show 514b88c6f` for the B-fix code if needed (NOT a starting point; reference only).
5. Implement the three files (`qwen3_5.py` edit, `_c2_diag.py` new, `_c2_eager_replay.py` new) per Section 2.
6. Run rung 0 first.
7. Run pre-flight self-tests (disabled, enabled, injection).
8. Run main diagnostic.
9. Interpret per Section 5; record results in `docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md`.

---

## References

- Memory: `project_uber_kernel_migration`, `project_beta_coop_residual_solo_bug`, `feedback_mutates_args_not_dce_safe`, `feedback_item_breaks_cuda_graphs`, `feedback_cooperative_grid_barrier`, `feedback_no_silent_fallbacks`, `feedback_bare_assert_hides_bugs`, `feedback_keep_debug_harnesses`, `feedback_flashinfer_autotune_sm120`, `feedback_post_quant_sanity`, `feedback_correct_model`, `project_full_graph_blocked`, `project_num_seqs_2_target`.
- Findings doc: `docs/research/uber_kernel_migration/2026-04-26-consume-gate-dce-and-graph-capture.md`.
- B-fix reference: commit `514b88c6f` (reverted in `3ffcf8740`).
- Relevant call sites verified at design time:
  - `vllm/nvllm/models/qwen3_5.py:466-476` — consume gate
  - `vllm/v1/attention/backends/cute_paged/_mlp_op.py:176,229,306` — existing custom op registrations
  - `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2685` — `run_beta_coop_full`
  - `vllm/model_executor/layers/attention/attention.py:713-760` — `unified_attention_with_output`
- Upstream issues searched (none match our stack): vllm-project/vllm #35659, #38208, #37060, #29852, #29715, #39625.
