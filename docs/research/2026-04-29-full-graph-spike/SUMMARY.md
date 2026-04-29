# FULL_AND_PIECEWISE + CuTe (β-coop) — n=1 Spike — Result

**Date:** 2026-04-29
**Branch:** `feat/full-and-piecewise-spike`
**Base commit:** `631ddcc62` (`docs(release): nvllm-v0.3.0`)

## Verdict

**PARTIAL — C0 PASS, C1 BLOCKED, C2/C3 NOT RUN.**

The spike's instrumentation, harness, fail-fast guard, and config-gated post-condition assert are installed and synced into the runtime container. The fallback flag and the `_assert_spike_invariant` post-condition were exercised under live serve and behaved correctly (warning fired at `_backend.py:141`; assert silent on all 16 layers under spike config). The `CUTE_FULL_GRAPH_PROBE` log was NOT exercised — C1 blocked before `/v1/models` readiness, so no decode call ever reached the FULL-dispatch branch where the probe lives. The flag is verified inert on PIECEWISE+β-coop (8/8 GSM8K sanity, C0 PASS).

C1 BLOCKED on a previously-undocumented compile-time wall: `cute.compile(self._jit_launch_phase_0_to_4, …)` at `phase_e_kernel.py:3007` runs for >95 minutes on first FULL_AND_PIECEWISE start at 101% CPU with no log progress and no completion. This is upstream of any FULL graph capture or replay — we cannot get to dispatch, so cannot verify `batch_desc.cg_mode=FULL`. C2 and C3 are gate-sequential and were not run.

This branch ships the spike's **code surface and harness** (verified live via C0) plus an **honest BLOCKED report** for C1 with a documented compile-time root cause and four follow-up paths. It does NOT prove that `serve-cute-full.sh n=1` is safe to recommend — the original spike goal — and per spec §5 ("we don't ship a quiet downgrade") that fact is not papered over.

Per `feedback_no_silent_fallbacks` and the spec's risk table, this is a hard-stop on the spike branch as written. The α-vs-β fork in spec §5 was conditional on observable C2/C3 evidence; we did not get to that fork.

## Gate-by-gate

| Gate | Verdict | Evidence |
|---|---|---|
| C0 | PASS | `evidence/2026-04-29-1155/c0_summary.md` (8/8 GSM8K sanity; **β-coop ON via `CUTE_PHASE_E_FUSION=1`** — confirmed by 16× "CuTe Phase E β-coop kernel attached" lines in `c0_docker_logs.txt`; `CUTE_PHASE_E_FALLBACK_RAISE=1` warning fired at `_backend.py:141`; sync-host-edits delivered cp into `/app/nvllm/vllm/...`) |
| C1 | **BLOCKED** | `evidence/2026-04-29-1215/c1_summary.md` (95+ min in `cute.compile` of β-coop full, no progress, killed at wall) |
| C2 (replay coherence) | NOT RUN | sequential gate; C1 prerequisite not met |
| C2 (single-token) | NOT RUN | sequential gate |
| C3 | NOT RUN | sequential gate |
| C4 | DEFERRED | spec; out of scope this branch |

## What changed in code

- `vllm/v1/attention/backends/cute_paged/_backend.py` — cached `_PHASE_E_FALLBACK_RAISE` flag with import-time warning; β-coop except-handler gated on the flag (default behavior unchanged); config-gated `_assert_spike_invariant` post-condition called from `process_weights_after_loading` (covers all early-return paths in both fusion-weight resolvers in one place; outside spike config it is a no-op).
- `vllm/v1/worker/gpu/model_runner.py` — env-gated `CUTE_FULL_GRAPH_PROBE` one-shot log of `batch_desc.cg_mode` immediately before the FULL dispatch branch (first 32 calls). `import os` added to module imports.
- `scripts/serve-cute-full.sh` — tightened to n=1 spike profile: `MAX_NUM_SEQS=1`, `cudagraph_capture_sizes=[1]`, `-e CUTE_PHASE_E_FUSION=1 CUTE_PHASE_E_FALLBACK_RAISE=1 CUTE_FULL_GRAPH_PROBE=1`. Comment header points at the spec.
- `scripts/serve-cute.sh` — forwards `CUTE_PHASE_E_FALLBACK_RAISE` (`${VAR:-0}` pattern) so C0 baseline can exercise the flag on the prod-shape PIECEWISE serve.

No persistent-buffer refactor (α workspace strategy held; spec §2.1 not exercised because we never got to C2).

## Harness

- `docs/research/2026-04-29-full-graph-spike/`
  - `README.md` — gate sequence + pass criteria.
  - `_sync_host_edits.sh` — bind-mount-replacement: `docker cp` host edits into the freshly-launched container BEFORE Python imports the CuTe backend, so the new code is live without a second model load. Replaces an earlier two-load design.
  - `c0_piecewise_baseline.sh` — PIECEWISE + flag-inertness 8/8 GSM8K sanity. PASSED.
  - `c1_replay_proof.sh` — FULL_AND_PIECEWISE batch-1 dispatch proof. BLOCKED at `cute.compile`.
  - `c2_replay_coherence.py` + `c2_single_token_determinism.py` — external arms; not exercised.
  - `c3_gsm8k_parity.sh` — PIECEWISE-vs-FULL answer-level parity on GSM8K-50; not exercised.

The 1800s wait ceiling in C1/C3 was the largest practical wait we could hold; the actual β-coop full compile was still running past it with no completion.

## C1 BLOCKED — observed blocker and recurrence factor

**Observed blocker:** the first call to `cute.compile(self._jit_launch_phase_0_to_4, …)` at `phase_e_kernel.py:3007` did not complete within the 95-min wait window on this run. EngineCore was alive at 101% CPU (single core), no Python traceback, no CUDA error, no log output between the "Compiling PhaseE_Beta_Kernel β-coop full (first call for this config)…" line and the manual kill. We do NOT know the actual upper bound — it may complete given more time, or it may be stuck. We did not have py-spy available to identify the slow function inside CuTe DSL codegen (image does not include it). So this is "cold compile exceeds practical session budget on Spark," not a proven bug — and not a proven hang either.

**Recurrence factor (NOT the root cause of slowness):** even if the first cold compile DOES complete given more time, the next FULL container will pay the same cost from scratch. `_PHASE_E_COOP_FULL_COMPILE_CACHE` (in `phase_e_kernel.py`) is an in-memory dict — no disk persistence across container restarts. The runtime container does NOT export `B12X_CUTE_COMPILE_DISK_CACHE=1` / `B12X_CUTE_COMPILE_CACHE_DIR=…` (Dockerfile sets them only at image-build warmup, where the warmup itself is `|| true` and may have no-op'd). This explains why every fresh FULL container would re-enter the cold compile, but it does NOT explain why the first compile takes >95 minutes. Those are two separate questions; the recurrence factor is the one with a clear fix (mount + propagate the env), the slowness itself is the one that still needs investigation.

Four follow-up paths (not done this branch — all are spec-out-of-scope):

1. **Bind-mount a host kernel-cache dir into the FULL container** and propagate `B12X_CUTE_COMPILE_DISK_CACHE=1` to runtime. Doesn't fix the *first* compile, but bounds the cost to once-per-CuTe-version-bump.
2. **Pre-compute β-coop full cache offline** by triggering `_compiled_phase_coop_full` for the n=1 config in a separate one-shot script, persist to disk, mount on every FULL launch.
3. **py-spy on EngineCore during the compile** to identify the slow function. py-spy is not installed in the current image.
4. **Decompose the fused phase-0-through-4 kernel** into smaller compile units. β-coop full's design maximizes runtime efficiency at the cost of compile time; a 2-or-3-piece split likely halves cold compile.

`feedback_constexpr_oom` ("range_constexpr(N>100) OOMs compiler; use runtime while loops") hints at a similar threshold inside CuTe DSL — β-coop full may have a borderline-large constexpr loop that doesn't OOM but spends huge time in codegen.

## Out of scope, not yet done

- β-lite under FULL (n>1) — C4, separate branch.
- Spec decode under FULL.
- Repo default flip from PIECEWISE to FULL_AND_PIECEWISE.
- Performance benchmarking.
- The four follow-up paths above (kernel-cache persistence, offline pre-compile, py-spy diag, kernel decomposition).

## Key memories validated by this run

- `feedback_docker_bindmount` — confirmed: docker cp into `/app/nvllm/vllm/...` (the editable-install path) lands BEFORE Python imports if done within ~1s of `docker run -d`.
- `feedback_rebuild_guard` — confirmed: Python-only edits do not require docker build; the sync flow is correct.
- `feedback_no_silent_fallbacks` — applied: C1 reported as BLOCKED, not silently passed.
- `feedback_pace_pressure` — applied: did not relax the gate criterion to make C1 "pass."
- `feedback_correct_model` — confirmed: `ig1/Qwen3.5-27B-NVFP4` is the right model for kernel debugging.
- `project_full_graph_blocked` — adds new finding: FULL+CuTe is "open, not broken" remains accurate. The new wall is **compile time**, not capture/replay correctness, which is upstream of the questions C1 was meant to answer.
