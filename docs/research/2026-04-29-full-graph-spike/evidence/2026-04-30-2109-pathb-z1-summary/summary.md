# Path B Z1 — controlled cache-pin causality test summary

## Verdict

**Cache artifact/directory identity is the dominant load-bearing variable for the FULL+β-coop replay-coherence FAIL** and explains 9 of 10 observed Z1 outcomes. The discriminator found in X-trial log analysis (62 MB vs 73 MB inductor-compiled artifact) is not merely correlated; locking the artifact identity moved the PASS rate to 5/5 with the good cache and to 1/5 with the bad cache. The bad.5 PASS (with sha256 verified unchanged across the trial) means we have NOT closed the root cause — there is residual non-determinism beyond the artifact.

Per the user's verdict framework: this lands at **"5/5 PASS + cache_reused on all 5 → causality basically closed"** for the good-cache direction, with the bad-cache control showing strong (4/5) FAIL bias but not 100% FAIL.

**Scope note:** The Z1 mount in `c2_full_layer_bisect.sh` pins all of `/root/.cache/vllm` (modelinfos + torch_compile_cache including AOT model + cache_key_factors.json + computation_graph.py), not just the AOT model file. See manifest below for what actually differed between the two locked snapshots.

## Cache directory manifest — only the AOT model differed

The mount controlled the whole `/root/.cache/vllm` tree, but only one file differs between the two locked snapshots — the AOT-compiled model. Both snapshots contain four files:

| File | GOOD sha256 | BAD sha256 | Differs? |
|---|---|---|---|
| `modelinfos/vllm-...-Qwen3_5ForConditionalGeneration.json` (760 B) | `75d82e16...` | `75d82e16...` | no |
| `torch_compile_cache/b690a46483/rank_0_0/backbone/computation_graph.py` (1.76 MB) | `7b5d064a...` | `7b5d064a...` | no |
| `torch_compile_cache/b690a46483/rank_0_0/backbone/cache_key_factors.json` (8.3 KB) | `896f9f8e...` | `896f9f8e...` | no |
| `torch_compile_cache/torch_aot_compile/9a5549f23a17.../rank_0_0/model` | `651e00bd...` (70 MB disk) | `af68c498...` (81 MB disk) | **yes** |

So the mount's nominal scope (whole vllm cache dir) is broader than the actual differentiator (the AOT model file). The verdict "artifact identity is the dominant variable" specifically means the AOT model bytes; the upstream cache-key shared logic (`cache_key_factors.json`, `computation_graph.py`) was identical across both snapshots, ruling those out as the discriminator.

| Label | AOT model disk size | "collected artifacts" log size | sha256 |
|---|---|---|---|
| GOOD | 73,413,669 B (~70 MB) | 62,142,045 B (~62.1 MB) | `651e00bd5997bacd9a062da66e6c9a078ed3c4469c27c715d8b025041a2a8264` |
| BAD | 84,885,730 B (~81 MB) | 73,614,088 B (~73.6 MB) | `af68c498c6ee45b60165d584a870f2f072068153a7c76d9592fc0097efe63c80` |

The "log-reported" size is what `torch.compile` prints as "collected artifacts: N entries, M artifacts, X bytes total" at compile time — the X-trial discriminator. It's smaller than the on-disk size because the on-disk file includes additional metadata beyond the collected-artifacts measure. Both snapshots came from the same input graph (same cache_key_factors.json, same computation_graph.py); inductor produced two distinct output binaries — that's the non-determinism.

## Trial results

### 5 trials, locked GOOD cache

| Trial | unique | same_pass | cross_pass | overall | cache_reused | sha256 unchanged | t2ready |
|---|---|---|---|---|---|---|---|
| Z1.good.1 | 1 | True | True | **PASS** | yes | yes | 120s |
| Z1.good.2 | 1 | True | True | **PASS** | yes | yes | 120s |
| Z1.good.3 | 1 | True | True | **PASS** | yes | yes | 120s |
| Z1.good.4 | 1 | True | True | **PASS** | yes | yes | 120s |
| Z1.good.5 | 1 | True | True | **PASS** | yes | yes | 120s |

**5/5 PASS.** Cache reuse confirmed by zero "saved AOT compiled function" lines, sha256 unchanged on every trial, and 120s boot times (vs ~200s cold-compile boots).

### 5 trials, locked BAD cache

| Trial | unique | same_pass | cross_pass | overall | cache_reused | sha256 unchanged | t2ready |
|---|---|---|---|---|---|---|---|
| Z1.bad.1 | 2 | False | True | **FAIL** | yes | yes | 120s |
| Z1.bad.2 | 5 | False | False | **FAIL** | yes | yes | 120s |
| Z1.bad.3 | 4 | False | False | **FAIL** | yes | yes | 120s |
| Z1.bad.4 | 4 | False | False | **FAIL** | yes | yes | 130s |
| Z1.bad.5 | 1 | True | True | **PASS** | yes | yes | 130s |

**4/5 FAIL.** Cache reuse also confirmed on every trial (sha256 unchanged, no saved AOT lines).

The bad.5 PASS is real but does not undermine the verdict — it suggests a smaller secondary source of non-determinism (CUDA scheduler ordering / atomic-add winner ordering / similar low-magnitude FP-perturbation source) that occasionally lets the bad artifact still produce a coherent result.

## Comparison to baseline

| Configuration | Sample | PASS rate |
|---|---|---|
| X-trials (audit-OFF, cold cache, mixed artifact) | 5 | 60% (3/5) |
| Z1.good (locked good cache) | 5 | **100% (5/5)** |
| Z1.bad (locked bad cache) | 5 | **20% (1/5)** |
| Pre-Z1 historical (v1, v2, X.1-X.5) | 7 audit-OFF | 43% (3/7) |

Locking the artifact moves the PASS rate from ~60% baseline to either 100% (good) or 20% (bad). The 5x ratio between cache labels makes the artifact identity the load-bearing variable.

## Implications for Z

**Production fix shape: persist a known-good torch.compile AOT cache across container starts, with fail-closed handling and a probe-off validation gate.**

Concrete next steps (NOT executed in this experiment, deferred to a separate Z-design):

1. **Mount a known-good cache fail-closed.** Add a host mount of `/root/.cache/vllm` (or a tighter scope, `torch_compile_cache/torch_aot_compile/`) in `scripts/serve-cute*.sh`. The mount alone is NOT sufficient; the production wrapper must:
   - Verify the expected AOT model file is present at the expected path before launch (refuse to start if missing or empty).
   - Verify the AOT model sha256 matches the blessed value (refuse to start on mismatch).
   - Mount the blessed cache **read-only** after bootstrap, so the running container cannot mutate the artifact (prevents an in-flight bad-compile from corrupting the canonical cache).
   - Keep the bootstrap cache RW only during the bless step; switch to RO for production serves.

2. **Bootstrap-and-validate workflow.** A one-shot script that:
   - Boots the model with diagnostics OFF (no `CUTE_FULL_GRAPH_PROBE`, no `CUTE_WO_RESET_LOG`, no `CUTE_DISPATCH_AUDIT`) — see the probe-off validation note below.
   - Runs lower-8 c2_replay_coherence as the gate.
   - Accepts the resulting cache only if `unique=1, cross_prompt=independent`.
   - Records the AOT model path + sha256 + size as the "blessed manifest".

3. **Probe-off validation gate (REQUIRED before declaring a cache production-ready).** All Z1 trials, including the locked-good 5/5 PASS, used `CUTE_WO_RESET_LOG=1` (carryover from X) and `CUTE_FULL_GRAPH_PROBE=1` (hardcoded in `c2_full_layer_bisect.sh:86`). The probe state is NOT zero, and a hot-path probe is the kind of perturbation that has flipped past results (see the Step 1 audit-ON c2 PASS vs Gate 1 audit-OFF c2 FAIL anomaly). Production blessing must validate locked-good with all CUTE_* probes set to 0 to confirm the artifact-identity finding survives in the diagnostic-clean configuration.

4. **Per-config cache scope.** This experiment used Qwen3.5-27B-NVFP4 + lower-8 β-coop + FULL_AND_PIECEWISE + this branch's torch/vLLM build. Each combination has its own cache key — the workflow needs an explicit manifest table mapping (model, backend, layer set, cudagraph mode, torch/vLLM build sha) → blessed cache path + AOT model sha256, and a stale-detection step that re-blesses on any of those changing.

5. **Report upstream when we have a reduced repro.** torch.compile / inductor producing two distinct binaries (different sha256s) for the same input graph (identical `cache_key_factors.json` + `computation_graph.py`) is the underlying issue. Our locking is a workaround; the long-term fix is upstream.

## What this experiment did NOT prove

- **The bad.5 PASS.** A smaller source of non-determinism remains — possibly CUDA scheduler ordering, atomic-add winner ordering, or a similar low-magnitude FP-perturbation source — that occasionally let the bad artifact still produce a coherent result with sha256 verified unchanged across the trial. We have NOT identified what that is. Root cause is therefore NOT closed; **this is dominant evidence for upstream torch.compile/inductor non-determinism, not a fully closed root-cause analysis.**
- **Probe-on confound.** Z1 trials ran with `CUTE_WO_RESET_LOG=1` and `CUTE_FULL_GRAPH_PROBE=1` (per item 3 above). The artifact-identity verdict is strong, but a probe-off re-validation is required before declaring it production-ready.
- **The X.1-X.5 artifact sha256s are NOT preserved** — those containers were torn down before this experiment. We can only correlate by log-reported size, not by binary identity. The Z1 artifacts (good and bad) are preserved on disk for further analysis.
- **The 70 MB / 81 MB disk-vs-log size discrepancy** isn't fully explained. Disk size includes more than the "collected artifacts" measure; could include compile metadata. The discriminator that matters is the log-reported "collected artifacts" bytes; the on-disk file includes additional bytes beyond that measure.

## Status

- v2 closeout's "stochastic FAIL inside FULL+β-coop replay/kernel behavior" framing is **largely superseded**: most of the variance is explained by which inductor artifact got compiled. The bad.5 surprise PASS keeps the door open for a smaller secondary source.
- Path B is closed for hypothesis directions we ruled out: dispatch, Python decisions, metadata layout, and static allocation are not the cause (Step 1 + Y/Y2 + Z1 evidence chain).
- v3 (mlp_partial_fp32 host-reset) is **NOT the right fix.** The right next move is the cache-lock + bootstrap-and-validate workflow, with the probe-off and fail-closed requirements above.
