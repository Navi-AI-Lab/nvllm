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

## Step 6 final smoke — NOT re-run on a clean post-detour build

Per user direction (close-out scope, not strict re-validation), the spec §G2 final smoke was not re-executed against an image that contains the G1 detour. The current `nvllm:gb10` image (`31897268b39d`, built 2026-04-29 18:27 EDT) **precedes** `410d59390` (18:32 EDT) — so the fail-closed Dockerfile and the warmup.py prefill kwargs fix are not yet exercised at build time.

**What this means for the 261 s number:** the warm time-to-API-ready measurement is from the pre-detour image. The G1 detour does not change runtime serve behavior (warmup runs at build time, not serve time), so the 261 s figure is expected to hold. The load-bearing test of the G1 detour is whether the next clean rebuild itself succeeds (fail-closed Dockerfile) — that build hasn't run yet.

**Honest framing for downstream readers:** the disk-cache plan is complete by its stated terminal-state criteria (G1 PASS + G2 PASS + post-detour cache-level smoke PASS). The strict end-to-end "clean rebuild → first cold serve → warm restart" loop has not been re-walked since the G1 detour. The cold-first-container >10 min observation from `project_full_graph_blocked` is **not** a cache issue — suspected FULL graph capture / autotuner one-time cost, tracked separately.

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
