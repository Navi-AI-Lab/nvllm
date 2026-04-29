# Gate G1 verdict — CuTe DSL disk-cache plumbing

Reference: `docs/superpowers/specs/2026-04-29-cute-full-compile-cache-design.md` §G1

## Run metadata

- Run id: `smoke-1777492816`
- Branch: `feat/cute-full-compile-cache`
- HEAD before this task: `962100655` (Task 3)
- Container: `nvllm:gb10` (built ~6h prior; image-baked vllm at `/app/nvllm/vllm`,
  bind-mount `/workspace/vllm` from host source — see deviation note below)
- Cache root: `/opt/vllm/kernel_cache/smoke-1777492816` (host: `/tmp/nvllm-cute-cache/smoke-1777492816`)
- Date: 2026-04-29

## Cold JSON (verbatim)

```json
{"phase": "cold", "run_id": "smoke-1777492816", "ok": true, "elapsed_s": 21.203, "n_compiled": 2, "files_before": 0, "files_after": 2, "new_files": ["6d/6d2a120c65a30728ef2b350fe7ddc33e322b27ccbe9452519f15b39afa934390.o", "97/977267e3e7ff886e6bafc0c4956efd98f7aa85b1db7bc7d19c2dd1371f006aa4.o"], "miss_count": 2, "hit_count": 0}
```

## Warm JSON (verbatim)

```json
{"phase": "warm", "run_id": "smoke-1777492816", "ok": false, "elapsed_s": 20.203, "n_compiled": 2, "files_before": 2, "files_after": 3, "new_files": ["92/9210947419f81b2117c37d9c4b1884374883cc18fbad1542555c744c3bec4791.o"], "miss_count": 1, "hit_count": 1, "err": "warm wallclock 20.2s > 5s; cache likely missed"}
```

## Verdict: **FAIL**

Gate G1 pass criteria (from plan):

| Criterion | Cold | Warm | Result |
|---|---|---|---|
| `ok: true` | true | false | FAIL (warm) |
| `miss_count >= 1` (cold) | 2 | — | PASS |
| `hit_count >= 1` (warm) | — | 1 | PASS |
| `miss_count == 0` (warm) | — | 1 | FAIL |
| `elapsed_s < 5.0` (warm) | — | 20.203 | FAIL |

### Root cause

**The decode kernel's cache key is non-deterministic across processes.**
The prefill key is stable (`977267e3e7ff886e...` cold == warm → HIT). The
decode key drifts (`6d2a120c65a30728...` cold ≠ `9210947419f81b21...` warm
→ MISS, full recompile, second on-disk artifact written).

Evidence trail (from raw stderr captured in `cache_smoke_cold.raw.log`
and `cache_smoke_warm.raw.log`):

Cold:
```
[disk_cache.py:440] CuTe disk cache MISS key=6d2a120c65a30728 - compiling   # decode
[disk_cache.py:456] CuTe disk cache stored (native) key=6d2a120c65a30728
[disk_cache.py:440] CuTe disk cache MISS key=977267e3e7ff886e - compiling   # prefill
[disk_cache.py:456] CuTe disk cache stored (native) key=977267e3e7ff886e
```

Warm (after container restart, same on-disk cache_dir):
```
[disk_cache.py:440] CuTe disk cache MISS key=9210947419f81b21 - compiling   # decode (NEW key)
[disk_cache.py:456] CuTe disk cache stored (native) key=9210947419f81b21
[disk_cache.py:429] CuTe disk cache HIT (native) key=977267e3e7ff886e        # prefill (stable)
```

So the plumbing — `apply_disk_cache_patch`, on-disk write, on-disk read,
HIT/MISS log emission — all works. The two-process cache reuse fails
because `_build_full_cache_key(self, func, args, kwargs)` produces a
different digest for the decode kernel between two cold-process invocations
even when the kernel-config and dummy-tensor shapes are byte-identical.

Plausible salts to investigate in `disk_cache._build_full_cache_key` and
its callees: data_ptr() values (would change every process), torch
tensor IDs, function `__qualname__` with module-load-counter suffix,
process-local hash seed (`PYTHONHASHSEED` not set), or a `random.uuid` /
timestamp leaked into the key.

Note: prefill cache key being stable but decode being unstable suggests
the salt source is in code that runs only on the decode path
(`DecodeKernel`-specific), or is a function-source hash that includes
something like a closure over a process-local object.

### Routing per plan

This matches the plan's anticipated G1-FAIL → Task 6 (b12x-native
serialize wrapper for cache-key stability) branch. Task 5 should adopt
the FAIL verdict and Task 6 should fix the decode key drift.

## Deviations from plan

1. **`warmup.warmup()` not used as the compile driver** — was the plan's
   primary instruction. While invoking it, we surfaced a stale-API bug:
   `warmup.warmup` calls the prefill kernel with `kv_cache=` (unified
   tensor), but the prefill kernel signature was updated to expect split
   `k_cache=`+`v_cache=` (zero-copy stride addressing in commit 748c9695c
   "perf(kernel): eliminate .contiguous() KV cache copies"). The build
   currently runs warmup under `|| true` in `docker/Dockerfile.gb10:135`
   so this regression went silent. Cache_smoke now drives both kernels
   directly via `_get_compiled_kernel(config)` + correct kwargs, which
   exercises the same `cute.compile()` path warmup uses but with kwargs
   the kernel actually accepts. Documented in cache_smoke.py module
   docstring.

2. **`sys.path.insert(0, '/workspace')` shim** — when running the script
   via `docker exec /opt/venv/bin/python <script>`, Python prepends the
   script directory to `sys.path` instead of the cwd. The image's
   editable-install path-hook then maps `vllm` → `/app/nvllm/vllm`
   (the build-time-baked source from BEFORE Task 2). This caused the
   first runs to load pre-Task-2 disk_cache.py with NO HIT/MISS logs
   (visible in the runtime log as `[disk_cache.py:458]` instead of the
   expected `[disk_cache.py:477]` for the "enabled" line). The shim
   forces the bind-mounted `/workspace/vllm` (post-Task-2 source) to
   take precedence. **Without an image rebuild, the runtime serve path
   has the same problem** — the apply_disk_cache_patch wired in Task 2
   inside `_backend.py` may also be loading from /app/nvllm at serve
   time. This must be re-verified after image rebuild.

3. **Capture handler attached AFTER `import vllm`** — vllm's logger
   init clears handlers on its named loggers, silently dropping any
   handler attached before the import. Initial runs showed
   `miss_count: 0` despite MISS lines appearing in stderr. Attaching
   the `_LineCapture` handler after `import vllm` recovered the
   captured lines.

## Reproducibility

```bash
# Inside repo root, from a fresh nvllm:gb10 container:
docker run -d --name nvllm --gpus all --ipc=host --entrypoint /bin/bash \
  -v "$PWD:/workspace" -w /workspace \
  -v /tmp/nvllm-cute-cache:/opt/vllm/kernel_cache \
  -e B12X_CUTE_COMPILE_CACHE_DIR_ROOT=/opt/vllm/kernel_cache \
  nvllm:gb10 -c 'sleep 1200'

RUN_ID="smoke-$(date +%s)"
docker exec nvllm /opt/venv/bin/python \
  docs/research/2026-04-29-full-graph-spike/cache_smoke.py \
  --phase=cold --run-id="$RUN_ID"

docker rm -f nvllm
docker run -d --name nvllm ... # same args as above
docker exec nvllm /opt/venv/bin/python \
  docs/research/2026-04-29-full-graph-spike/cache_smoke.py \
  --phase=warm --run-id="$RUN_ID"
```

## Files

- `cache_smoke_cold.json` — 1-line JSON, parseable by jq
- `cache_smoke_cold.raw.log` — full stderr+stdout from cold run
- `cache_smoke_warm.json` — 1-line JSON
- `cache_smoke_warm.raw.log` — full stderr+stdout from warm run
- `gate_g1_verdict.md` — this file
