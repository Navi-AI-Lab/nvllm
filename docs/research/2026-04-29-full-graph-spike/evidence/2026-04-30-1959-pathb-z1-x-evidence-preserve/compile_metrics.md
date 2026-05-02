# Path B Z1 — X-trial compile metrics preserved

**Purpose:** Preserve the compile-side evidence from the X.1-X.5 audit-OFF reproducibility
trials before running the Z1 controlled cache-pin causality test. The X-trial containers
were torn down between/after each run (no host cache mount), so the actual binary AOT
artifacts are gone. Only what we logged remains.

**Source trials:** `evidence/2026-04-30-1910-pathb-x-trial-1/` … `evidence/2026-04-30-1947-pathb-x-trial-5/`

**Branch:** `feat/cute-beta-coop-persistent-buffers`
**git SHA at time of trials:** `d36abf7713dfaaf8c5beb4dd7ee2c0099428e93a` (X.1-X.3),
`68c6ab944a288d81b25da309c42e75f675431341` (X.4-X.5).

## Summary table

| Trial | compile time | artifact size (B) | artifact size (MB) | cache key (hex) | unique | same / cross / overall |
|---|---|---|---|---|---|---|
| X.1 | 83.40 s | 62,118,662 | 62.12 | `9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721` | 1 | true / true / **true** |
| X.2 | 101.57 s | 73,393,077 | 73.39 | `9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721` | 3 | false / false / **false** |
| X.3 | 83.04 s | 62,136,258 | 62.14 | `9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721` | 1 | true / true / **true** |
| X.4 | 84.71 s | 62,179,567 | 62.18 | `9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721` | 1 | true / true / **true** |
| X.5 | 101.19 s | 73,336,693 | 73.34 | `9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721` | 4 | false / true / **false** |

**Discriminator:** PASS trials cluster at ~83-85 s compile / ~62.1-62.2 MB artifact
(within 60 KB of each other across X.1, X.3, X.4). FAIL trials cluster at
~101 s compile / ~73.3-73.4 MB artifact (within 60 KB of each other across X.2, X.5).
Cache key is identical in every trial (`9a5549f23a…`) — torch's AOT cache lookup
key is the same input, but two distinct compiled binaries land non-deterministically.

## Per-trial artifact path

All five trials saved to:

```
/root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721/rank_0_0/model
```

(inside the per-container ephemeral cache, which is destroyed when the container is removed.)

## Per-trial relevant log lines

### X.1 (PASS)

```
(EngineCore pid=221) INFO 04-30 23:05:04 [backends.py:390] Compiling a graph for compile range (1, 65536) takes 83.40 s
(EngineCore pid=221) INFO 04-30 23:05:08 [backends.py:895] collected artifacts: 65 entries, 21 artifacts, 62118662 bytes total
(EngineCore pid=221) INFO 04-30 23:05:08 [decorators.py:655] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721/rank_0_0/model
(EngineCore pid=221) INFO 04-30 23:05:08 [monitor.py:48] torch.compile took 92.60 s in total
```

### X.2 (FAIL, unique=3)

```
(EngineCore pid=221) INFO 04-30 23:13:09 [backends.py:390] Compiling a graph for compile range (1, 65536) takes 101.57 s
(EngineCore pid=221) INFO 04-30 23:13:13 [backends.py:895] collected artifacts: 65 entries, 21 artifacts, 73393077 bytes total
(EngineCore pid=221) INFO 04-30 23:13:13 [decorators.py:655] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721/rank_0_0/model
(EngineCore pid=221) INFO 04-30 23:13:13 [monitor.py:48] torch.compile took 111.04 s in total
```

### X.3 (PASS)

```
(EngineCore pid=220) INFO 04-30 23:20:45 [backends.py:390] Compiling a graph for compile range (1, 65536) takes 83.04 s
(EngineCore pid=220) INFO 04-30 23:20:49 [backends.py:895] collected artifacts: 65 entries, 21 artifacts, 62136258 bytes total
(EngineCore pid=220) INFO 04-30 23:20:49 [decorators.py:655] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721/rank_0_0/model
(EngineCore pid=220) INFO 04-30 23:20:49 [monitor.py:48] torch.compile took 92.16 s in total
```

### X.4 (PASS)

```
(EngineCore pid=220) INFO 04-30 23:34:34 [backends.py:390] Compiling a graph for compile range (1, 65536) takes 84.71 s
(EngineCore pid=220) INFO 04-30 23:34:38 [backends.py:895] collected artifacts: 65 entries, 21 artifacts, 62179567 bytes total
(EngineCore pid=220) INFO 04-30 23:34:39 [decorators.py:655] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721/rank_0_0/model
(EngineCore pid=220) INFO 04-30 23:34:39 [monitor.py:48] torch.compile took 93.97 s in total
```

### X.5 (FAIL, unique=4)

```
(EngineCore pid=221) INFO 04-30 23:42:24 [backends.py:390] Compiling a graph for compile range (1, 65536) takes 101.19 s
(EngineCore pid=221) INFO 04-30 23:42:29 [backends.py:895] collected artifacts: 65 entries, 21 artifacts, 73336693 bytes total
(EngineCore pid=221) INFO 04-30 23:42:29 [decorators.py:655] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721/rank_0_0/model
(EngineCore pid=221) INFO 04-30 23:42:29 [monitor.py:48] torch.compile took 110.52 s in total
```

## Evidence gap (explicit)

**sha256 of the actual binary AOT artifacts is NOT available.**
The X-trial containers were spun up via `c2_full_layer_bisect.sh` without any host
mount of `/root/.cache/vllm`, so the per-container scratch cache (and its
`rank_0_0/model` binary) was destroyed at `docker rm -f nvllm`. Cache key
(`9a5549f23a…`) was identical across all 5 trials, but artifact size separated
cleanly into two clusters (62 MB vs 73 MB), so we know there are at least two
distinct compiled binaries — we just don't have the bytes to hash.

The Z1 experiment (this dispatch) closes that gap by mounting the cache to a
host directory, capturing the sha256+size+key of a known-good (62 MB) artifact,
and then running 5 trials with that locked artifact reused.

## Reproduction

The X-trial protocol was:

```
docker rm -f nvllm 2>/dev/null
CUTE_WO_RESET_LOG=1 bash docs/research/2026-04-29-full-graph-spike/c2_full_layer_bisect.sh '3,7,11,15,19,23,27,31'
# wait for /v1/models
.venv/bin/python docs/research/2026-04-29-full-graph-spike/c2_replay_coherence.py
docker rm -f nvllm
```

Z1 adds `PATHB_Z1_VLLM_CACHE_HOST_DIR=<host-dir>` before the bisect-script call.
