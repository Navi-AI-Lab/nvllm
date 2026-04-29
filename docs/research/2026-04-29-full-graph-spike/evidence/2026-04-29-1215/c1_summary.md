# C1 — FULL dispatch proof — **BLOCKED**

- Date: 2026-04-29 (UTC 16:15 → 17:54)
- Wait window: 99 minutes from `Compiling PhaseE_Beta_Kernel β-coop full (first call for this config)…` (16:19:01 UTC) until manual kill at 17:54 UTC.
- Process state at kill: `pid 183 VLLM::EngineCore Rl 101% CPU 3.3% mem ELAPSED 01:37:56` — process alive and busy, no CUDA error, no Python traceback, no log progress.
- HTTP `/v1/models` was never `200`.

## What blocks C1 (and therefore C2, C3)

`vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:3007` calls `cute.compile(self._jit_launch_phase_0_to_4, …)` on the FIRST FULL_AND_PIECEWISE start. The CuTe DSL compile of the fused phase-0-through-4 β-coop full kernel runs for >95 min on a cold cache without emitting an interior progress line and without completing within the spike's wait window.

`_PHASE_E_COOP_FULL_COMPILE_CACHE` (in `phase_e_kernel.py`) is an in-memory dict: it does not persist across container restarts. The runtime container does NOT export `B12X_CUTE_COMPILE_DISK_CACHE=1` / `B12X_CUTE_COMPILE_CACHE_DIR=…` (these are only set at image-build time for the warmup at `Dockerfile.gb10:132-135`, where the warmup itself is `|| true` and may have no-op'd). So every fresh FULL container pays this cold-compile cost from scratch.

The container's `nvidia-smi` showed 26.8 GiB GPU memory in use and steady CPU 101% on a single core throughout the 95-min window — consistent with single-threaded LLVM/PTX codegen inside CuTe DSL. Memory was not exhausted (3.3% RSS).

## Evidence

- `c1_serve_launch.txt` — serve-cute-full.sh launch log (n=1 spike profile, all three CUTE_* env vars present).
- `c1_sync_host_edits.txt` — confirms edits cp'd into `/app/nvllm/vllm/...` 1s after `docker run -d`, before vLLM's worker imports.
- `c1_docker_logs_blocked_at_compile.txt` (182 lines) — full log; final line is `Compiling PhaseE_Beta_Kernel β-coop full (first call for this config)…`. No further output.
- `c1_docker_logs_timeout.txt` (~tail-100) — captured by the script when its 1800s wait ceiling expired.

The expected `c1_response.json`, `c1_probe.log`, `c1_passes_full.txt` do NOT exist — the curl was never reached because `/v1/models` never returned 200.

## Why this counts as BLOCKED, not FAIL

- The flag (`CUTE_PHASE_E_FALLBACK_RAISE=1`) DID fire its import-time warning correctly — see `_backend.py:141` line in the log. Our edits are live.
- The β-coop kernels attached for all 16 CuTe-fused layers (`_backend.py:874` × 16). The `_assert_spike_invariant` post-condition did NOT raise — meaning `_beta_coop_framework_output_bound=True` was satisfied for all layers under the spike config. Both spike-side asserts are passing.
- The block is upstream of where we are testing: in CuTe DSL's first-call compile of β-coop full kernel, before any FULL graph is captured or replayed. We cannot prove `batch_desc.cg_mode=FULL` reached `gpu/model_runner.py:L1069` because the model never finished readiness.
- This corresponds to spec §5's "Both α and C2/C3 fail within branch budget → β-coop's existing structure is not graph-friendly → Hard-stop". `feedback_no_silent_fallbacks` applies: do not silently downgrade by claiming a partial pass.

## What this does NOT prove

- Whether `cute.compile(…)` would eventually complete given enough wall time.
- Whether FULL_AND_PIECEWISE+CuTe+β-coop replays correctly. The whole spike target.
- Whether the spike's `_PHASE_E_FALLBACK_RAISE` would intercept a real β-coop failure under FULL.
- Whether C2/C3 outcomes are ambiguous — cannot get there.

## What this DOES prove (concrete)

- The harness, sync-flow, and serve script tightenings are operationally correct.
- C0 PASS shows `CUTE_PHASE_E_FALLBACK_RAISE=1` is inert on the established PIECEWISE path. (Independent of C1.)
- The `CUTE_FULL_GRAPH_PROBE` log line is in place and would fire if the graph reached dispatch. (Independent of C1's outcome — the probe code is shipped and verified to be present in the running container via `c1_sync_host_edits.txt`.)
- The `_assert_spike_invariant` at `_backend.py:_assert_spike_invariant` (post-`process_weights_after_loading`) does not falsely fire under spike config — silent pass on all 16 layers.

## Suggested follow-up paths (NOT done in this branch)

1. **Bind-mount a host dir as `/opt/vllm/kernel_cache`** (or a CuTe-compile-persist dir) and propagate `B12X_CUTE_COMPILE_DISK_CACHE=1` to the runtime container in `scripts/serve-cute-full.sh`. The Dockerfile sets this only for image-build warmup; runtime needs it too.
2. **Pre-compute β-coop full cache offline.** Run a script that triggers `_compiled_phase_coop_full` cache fill for the n=1 config, persist to disk, then mount on every FULL launch.
3. **Investigate why `cute.compile` is so slow** on this fused phase-0-through-4 kernel. Memory `feedback_constexpr_oom` (range_constexpr(N>100) OOMs compiler) hints at a similar threshold; β-coop full may have a borderline-large constexpr loop that doesn't OOM but spends huge time in codegen. py-spy would help — not installed in current image.
4. **Decompose the fused kernel.** β-coop full is one big fused phase-0-through-4 unit; this maximizes runtime efficiency at the cost of compile time. Splitting into two compile units may halve compile.

These are exploratory; none of them is on the spec's α-or-β decision tree (the spec assumed compile time was bounded). The compile-time wall is a NEW finding.

## Net for the spike

- C0: PASS (independent — flag inertness on PIECEWISE)
- C1: BLOCKED (cute.compile of β-coop full does not finish within 95 min on Spark cold cache)
- C2: NOT RUN (gates are sequential)
- C3: NOT RUN
- C4: deferred (spec)

Following spec §5: report the blocker honestly, do not claim partial PASS, do not silently downgrade. The serve-cute-full.sh n=1 profile is NOT yet validated for batch-1 serving.
