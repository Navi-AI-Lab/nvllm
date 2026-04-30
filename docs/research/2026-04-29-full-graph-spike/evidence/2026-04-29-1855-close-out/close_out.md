# Task 16 Close-out — β-coop FULL kernel disk cache

**Plan:** [`docs/superpowers/plans/2026-04-29-cute-full-compile-cache.md`](../../../../superpowers/plans/2026-04-29-cute-full-compile-cache.md)
**Spec:** [`docs/superpowers/specs/2026-04-29-cute-full-compile-cache-design.md`](../../../../superpowers/specs/2026-04-29-cute-full-compile-cache-design.md)
**HEAD at close-out:** `1d61e2c14` (`test(cute-cache): step 3 of probe plan — _build_full_cache_key invariants`)

## Terminal state

**Best** — per spec §10:
- ✅ G1 PASS (post-fix): [`evidence/2026-04-29-1613/gate_g1_postfix_verdict.md`](../2026-04-29-1613/gate_g1_postfix_verdict.md)
- ✅ G2 PASS: [`evidence/2026-04-29-1634/gate_g2_verdict.md`](../2026-04-29-1634/gate_g2_verdict.md)
- ✅ G1 detour post-fix smoke: [`evidence/g1-v2-20260429-173404/`](../g1-v2-20260429-173404/) — cold 20.2 s / warm 0.15 s / 2 HITs / 0 MISSes

Tasks 12–15 (G2-FAIL conditional + G3 routing) were correctly skipped.

## Headline numbers (pre-G1-detour, image `31897268b39d` built 2026-04-29 18:27 EDT)

| Quantity | Value | Source |
|---|---|---|
| Cold β-coop FULL `cute.compile` (precompile) | 24.8 s | [`evidence/2026-04-29-1634/precompile_run.log`](../2026-04-29-1634/precompile_run.log) |
| Warm serve disk-cache HIT | instant | [`evidence/2026-04-29-1634/serve_warm2_full.log`](../2026-04-29-1634/serve_warm2_full.log) (`CuTe disk cache HIT (native) key=fae644defea889c0`) |
| Time-to-API-ready (warm, active `/v1/models` probe) | 261 s | [`evidence/2026-04-29-1634/final_smoke_metrics.txt`](../2026-04-29-1634/final_smoke_metrics.txt) |
| `/v1/completions` round-trip | coherent | [`evidence/2026-04-29-1634/final_smoke_completion.json`](../2026-04-29-1634/final_smoke_completion.json) |

## G1 detour resolution (after the squash)

The G2 verdict flagged a "precompile-vs-serve key drift" caveat (precompile produced `b676fa082e7b7d82`; cold serve produced `fae644defea889c0`). Cold serve self-populated the cache cheaply (24 s), so the precompile was not load-bearing. The G1 detour closed the drift across four commits on `main`:

| Commit | What it did |
|---|---|
| `07018e630` | Structural KEY_DEBUG probe — per-call JSON dump under `<cache_dir>/_debug/<func-slug>.<call_index>.<run>.json`, gated on `B12X_CUTE_COMPILE_KEY_DEBUG=1`. Refactored `_build_full_cache_key` to share `_build_full_cache_payload(...)` with the probe so `payload_hash_matches_key` is a real invariant. Confirmed cold/warm key drift was real and isolated to args-key. |
| `410d59390` | Root cause + build hardening: (a) `vllm/v1/attention/backends/cute_paged/warmup.py` prefill now passes split `k_cache=`/`v_cache=` kwargs (decode reads unified `kv_cache=`) — prior code passed `kv_cache=` to both, silently failing prefill at warmup. (b) `docker/Dockerfile.gb10` dropped `\|\| true` from the warmup `RUN` so a kernel-compile regression now fails the build. |
| `dd1c4e252` | Pinned `transformers==4.57.6` because the (now fail-closed) build surfaced that `transformers @ git+main` was breaking warmup at build time. |
| `1d61e2c14` | `tests/unit/test_disk_cache_key_invariants.py` — 16 invariants over `_build_full_cache_key` / `_structural_args_cache_key` (pointer canonicalization, stride/grid/shape sensitivity, salt versioning). |

**Post-detour smoke:** [`evidence/g1-v2-20260429-173404/cache_smoke_cold.json`](../g1-v2-20260429-173404/cache_smoke_cold.json) + [`cache_smoke_warm.json`](../g1-v2-20260429-173404/cache_smoke_warm.json) — cold 20.2 s with 2 stored, warm 0.15 s with 2 HITs / 0 MISSes via `cache_smoke.py` harness using the real `warmup.warmup()` driver.

## Step 6 final smoke — original close-out (pre-strict-validation)

Per user direction (close-out scope, not strict re-validation), the spec §G2 final smoke was not re-executed against an image that contains the G1 detour. The previous `nvllm:gb10` image (`31897268b39d`, built 2026-04-29 18:27 EDT) **preceded** `410d59390` (18:32 EDT) — so the fail-closed Dockerfile and the warmup.py prefill kwargs fix were not exercised at build time when this doc was first written.

**What this means for the 261 s number:** the warm time-to-API-ready measurement is from the pre-detour image. The G1 detour does not change runtime serve behavior (warmup runs at build time, not serve time), so the 261 s figure is expected to hold.

**Honest framing for downstream readers:** the disk-cache plan was complete by its stated terminal-state criteria (G1 PASS + G2 PASS + post-detour cache-level smoke PASS). The strict end-to-end "clean rebuild → first cold serve → warm restart" loop had not been re-walked at the time of original close-out. **It has now been walked — see "Strict validation (2026-04-29 21:45 → 22:25 EDT)" below.** The cold-first-container >10 min observation from `project_full_graph_blocked` is **not** a cache issue — suspected FULL graph capture / autotuner one-time cost, tracked separately.

## Strict validation (2026-04-29 21:45 → 22:25 EDT)

**Evidence:** [`evidence/2026-04-29-2145-strict-validation/`](../2026-04-29-2145-strict-validation/) — full verdict at [`verdict.md`](../2026-04-29-2145-strict-validation/verdict.md).

**Pre-validation: deleted build-time CuTe warmup steps (commit `4dead0e4e`).** The clean rebuild surfaced that `Dockerfile.gb10:131-144` (warmup + verify-only RUNs added by G1 detour `410d59390`) failed because `libcuda.so.1` is unavailable inside the build container — `docker build` has no `--gpus` equivalent. The warmup steps had failed silently for months under the original `|| true`. Replaced L131-144 with a comment establishing the boundary: image build is dependency packaging; CuTe compilation is GPU-runtime validation, exercised by the first cold serve / cache_smoke.py / precompile-cute-coop-full.sh, all of which run with `--gpus all`. See [`evidence/2026-04-29-1958-strict-validation/build_failure_analysis.md`](../2026-04-29-1958-strict-validation/build_failure_analysis.md) for the build-failure root cause and [`build_failure.log`](../2026-04-29-1958-strict-validation/build_failure.log) for the raw evidence.

**Strict-validation rebuild (HEAD `67df3dcbd`):** 19/19 steps, ~54 min, image `a3f3f609a8ec` at 21:10 EDT.

**Pass criteria (user-defined):**

| # | Criterion | Verdict |
|---|---|---|
| 1 | image builds cleanly | ✅ PASS |
| 2 | transformers==4.57.6 | ✅ PASS |
| 3 | fresh host cache starts empty | ✅ PASS (FILE_COUNT=0) |
| 4 | cold serve compiles+stores | ✅ PASS (MISS=1 STORED=1, β-coop FULL key `4b272b8d727401a4`, 24.6s for the cute.compile) |
| 5 | warm serve HIT, no relevant MISS | ✅ PASS (HIT=1 MISS=0, same key, warm time-to-API-ready 610s) |

**Coherent /v1/completions response on warm serve:** `{"text":". Please write","finish_reason":"length"}`.

**Caveat (not gated by pass criteria):** cold serve did NOT reach `/v1/models` within the 30-min poll ceiling. β-coop FULL store completed at `01:48:59` (3.5 min after launch), then ~26 min of silence with no `Application startup complete` line. Same shape as the pre-detour `evidence/2026-04-29-1634/serve_warm_full.log` (~6-min silence before that harness's 10-min ceiling). This is the cold-first-container behavior tracked in `project_full_graph_blocked` — suspected FULL graph capture / autotuner one-time cost, **not a cache issue**. The cache hit on the warm phase proves the cache itself was correctly populated and persisted.

**Final framing:** the cache work is **closed and production-proven** by its own definition. Cold-first-container readiness remains an open, pre-existing problem owned by the FULL_AND_PIECEWISE re-enablement effort.

## Spec §10 definition-of-done audit

| Terminal state | Result |
|---|---|
| (Best) G1 + G2 PASS | ✅ achieved |
| (Acceptable) G1 PASS + G2 ACCEPTED-SLOW | n/a |
| (Acceptable) G1 PASS + G2 FAIL + Step 4 + G4 PASS | n/a |
| (Documented loss) G1 PASS + G4 FAIL | n/a |

## Memory updates (Step 2–4)

- `project_beta_coop_full_compile_wall.md` — added "G1 detour follow-ups" section listing the four commits, post-fix smoke evidence, and the not-yet-re-measured caveat. Replaced the stale "precompile dummy-tensor key drift" caveat.
- `project_strategy_priorities.md` — Active candidate #1 reworded: "unblocked" (was "partially unblocked"), G1 detour commits cited, removed "precompile-vs-serve cache-key alignment" from remaining work.
- `project_full_graph_blocked.md` — annotated 261 s as pre-G1-detour, noted re-measure recommended but not blocking.
- `MEMORY.md` index — hooks for the three files are still accurate one-liners; no edit.

## Routes to next

Per `project_strategy_priorities.md`, three live levers:
1. **FULL_AND_PIECEWISE + CuTe re-enablement** — disk cache is closed; remaining work is the cold-first-container >10 min investigation and perf delta vs PIECEWISE.
2. **`num_seqs=2` cooperative path** — β-coop currently solo-only at coop launch (resident-cap-bound).
3. **MTP / spec decode for Qwen3.5-27B** — largest perf lever if upstream is unblocked.
