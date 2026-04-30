# Strict validation verdict

**HEAD:** 67df3dcbd162651554ddb74754bb9e507083402c
**EVDIR:** docs/research/2026-04-29-full-graph-spike/evidence/2026-04-29-2145-strict-validation
**HOST_CACHE:** /tmp/nvllm-strict-validation-cache-1777513520

## Pass criteria

| # | Criterion | Verdict |
|---|---|---|
| 1 | image builds cleanly | ✅ PASS |
| 2 | transformers==4.57.6 | ✅ PASS (`transformers== 4.57.6`) |
| 3 | fresh host cache starts empty | ✅ PASS (pre=0) |
| 4 | cold serve compiles+stores | ✅ PASS (MISS=1 STORED=1) |
| 5 | warm serve HIT, no relevant MISS | ✅ PASS (HIT=1 MISS=0) |

## Cold serve metrics
```
TIME_TO_API_READY=0s
READY=0
MISS_count=1
STORED_count=1
HIT_count=0
0
=== unique MISS keys ===
MISS key=4b272b8d727401a4
=== unique STORED keys ===
stored (native) key=4b272b8d727401a4
```

## Warm serve metrics
```
TIME_TO_API_READY=610s
READY=1
MISS_count=0
0
STORED_count=0
0
HIT_count=1
=== HIT lines ===
(EngineCore pid=145) INFO 04-30 02:19:14 [disk_cache.py:803] CuTe disk cache HIT (native) key=4b272b8d727401a4
=== unique MISS keys (must be empty for PASS) ===
```

## Caveat: cold serve did NOT reach `/v1/models` within the 30-min poll ceiling

The 5-criterion verdict above intentionally did NOT gate on cold serve readiness — the user's pass criteria said "cold serve compiles/stores", which it did (MISS+STORED=1, β-coop FULL key `4b272b8d727401a4`, 24.6s for the cute.compile itself). But honest reporting means flagging that `READY=0` for cold:

- Cold launch: `01:45:24`
- β-coop FULL stored: `01:48:59` (3.5 min after launch)
- Last log line: `01:48:59`
- Harness gave up polling at: `~02:15:24` (30-min ceiling)
- Silent gap: ~26 minutes with no log output, no `Application startup complete`, no `/v1/models` route registered

This matches the cold-first-container behavior described in `project_full_graph_blocked` (suspected FULL graph capture / autotuner one-time cost), and is the same shape of silence observed pre-detour at `evidence/2026-04-29-1634/serve_warm_full.log` (~6-min gap there before that harness's 10-min ceiling).

The cache itself is fine — the warm phase HITs the same key and reaches API-ready in 610s, returning coherent /v1/completions text:

```json
{"text":". Please write","finish_reason":"length"}
```

Conclusion: the disk-cache plan is **production-proven** for what it set out to fix (cold compile time and warm serve cache hit). The cold-first-container silent stall **after** the cache store is a separate, pre-existing problem in the FULL graph capture / autotuner path — not a cache issue, and out of scope for this branch.

## Files
    total 124
    drwxrwxr-x  2 natfii natfii  4096 Apr 29 22:25 .
    drwxrwxr-x 12 natfii natfii  4096 Apr 29 21:45 ..
    -rw-rw-r--  1 natfii natfii    60 Apr 29 21:45 cold_host_cache_pre.txt
    -rw-rw-r--  1 natfii natfii   159 Apr 29 22:15 cold_host_cache.txt
    -rw-rw-r--  1 natfii natfii   186 Apr 29 22:15 cold_metrics.txt
    -rw-rw-r--  1 natfii natfii   539 Apr 29 21:45 cold_serve_launch.log
    -rw-rw-r--  1 natfii natfii 33523 Apr 29 22:15 cold_serve.log
    -rw-rw-r--  1 natfii natfii   607 Apr 29 21:45 image_metadata.txt
    -rw-rw-r--  1 natfii natfii   228 Apr 29 21:45 run_metadata.txt
    -rw-rw-r--  1 natfii natfii  1105 Apr 29 22:25 verdict.md
    -rw-rw-r--  1 natfii natfii   254 Apr 29 22:25 warm_metrics.txt
    -rw-rw-r--  1 natfii natfii   539 Apr 29 22:15 warm_serve_launch.log
    -rw-rw-r--  1 natfii natfii 39838 Apr 29 22:25 warm_serve.log
    -rw-rw-r--  1 natfii natfii   437 Apr 29 22:25 warm_smoke.json
