# Gate G1 — postfix verdict (PASS)

Reference: `docs/superpowers/specs/2026-04-29-cute-full-compile-cache-design.md` §G1
Prior FAIL verdict: `../2026-04-29-1551/gate_g1_verdict.md`

## Run metadata

- Run id (pass): `smoke-postfix-v2-1777493912`
- Run ids (KEY_DEBUG diagnosis prior to v2): `smoke-postfix-1777493626`, `smoke-postfix-debug-1777493718`
- Branch: `feat/cute-full-compile-cache`
- Container: `nvllm:gb10` (image-baked vllm at `/app/nvllm/vllm`,
  bind-mount `/workspace/vllm` from host source — same as 2026-04-29-1551)
- Cache root: `/opt/vllm/kernel_cache/smoke-postfix-v2-1777493912`
  (host: `/tmp/nvllm-cute-cache/smoke-postfix-v2-1777493912`)
- Date: 2026-04-29

## Cold JSON (verbatim)

```json
{"phase": "cold", "run_id": "smoke-postfix-v2-1777493912", "ok": true, "elapsed_s": 21.084, "n_compiled": 2, "files_before": 0, "files_after": 2, "new_files": ["be/bea46b694461b1f32d1f5459add5db8f8ae4013e8ee30af31c1fcbe5a34c8fcd.o", "ea/ea57a0c6edf419a13be8da433ae7ee27c8ab6f521500dd8c4dee9c832ddcc3cb.o"], "miss_count": 2, "hit_count": 0}
```

## Warm JSON (verbatim)

```json
{"phase": "warm", "run_id": "smoke-postfix-v2-1777493912", "ok": true, "elapsed_s": 0.083, "n_compiled": 2, "files_before": 2, "files_after": 2, "new_files": [], "miss_count": 0, "hit_count": 2}
```

## Verdict: **PASS**

Gate G1 pass criteria (from plan):

| Criterion | Cold | Warm | Result |
|---|---|---|---|
| `ok: true` | true | true | PASS |
| `miss_count >= 1` (cold) | 2 | — | PASS |
| `hit_count >= 2` (warm) | — | 2 | PASS |
| `miss_count == 0` (warm) | — | 0 | PASS |
| `elapsed_s < 5.0` (warm) | — | 0.083 | PASS |

Both kernels HIT on warm restart:

```
[disk_cache.py:509] CuTe disk cache HIT (native) key=ea57a0c6edf419a1   # decode
[disk_cache.py:509] CuTe disk cache HIT (native) key=bea46b694461b1f3   # prefill
```

Decode key now stable across processes (was `6d2a120c…` cold ≠ `9210947419…`
warm in 2026-04-29-1551, then `2812bf560f7d90ff` cold ≠ `69eb0d702ea39b8a`
warm at the first v1-name-binding attempt — see "Diagnosis" below).

## Root cause and fix

The decode kernel passes `Int64(kv_cache.data_ptr())` and other pointer
Int64 scalars positionally into `cute.compile`. The original
`_structural_cache_key` hashed those via the generic-object branch
(`vars(value).items()`), which serialized the embedded address —
fresh container = fresh address = different key.

### v1 attempt (insufficient): name-based binding

Initial fix bound positional args to the JIT function's signature via
`inspect.signature(inspect.unwrap(func))` and treated names ending in
`_ptr` as type-only. This *failed* because cute.jit-decorated bound
methods (e.g. `DecodeKernel._jit_launch`) report `self` as the first
positional parameter, shifting every name by one position.

KEY_DEBUG output captured under `B12X_CUTE_COMPILE_KEY_DEBUG=1` proved
this empirically (cold/warm differed only in the `value` of an
`Int64`-typed entry the binder labeled `query`):

- cold: `('query', ('object', 'cutlass.base_dsl.typing', 'Int64', (('value', 269833727918080),)))`
- warm: `('query', ('object', 'cutlass.base_dsl.typing', 'Int64', (('value', 272910388314112),)))`

Files: `cold_with_keydbg.json`, `warm_with_keydbg.json` (full per-call
KEY_DEBUG dumps).

### v2 (passing): value-type-based canonicalization

Recognize cutlass `Int64` *by value type*, not by parameter name:

- Convention in this codebase: `Int64` = runtime pointer; `Int32` =
  shape/flag. Verified by reading every Int64/Int32 use site in
  `vllm/v1/attention/backends/cute_paged/kernel.py` (decode kernel
  `all_args`, prefill kernel call site).
- New helper `_is_runtime_pointer_value(v)` returns True iff
  `type(v).__module__ == "cutlass.base_dsl.typing"` and
  `type(v).__qualname__ == "Int64"`.
- `_structural_args_cache_key(func, args, kwargs)` walks args + kwargs
  and replaces any cutlass-Int64 with `("runtime_ptr", module, qualname)`,
  hashing all other values via the existing `_structural_cache_key`.
- Salt bumped `v1` → `v2_ptr_canonical` to force-invalidate any
  pre-existing entries computed under the buggy v1 key.

`B12X_CUTE_COMPILE_KEY_DEBUG=1` instrumentation kept (off by default).

## How to reproduce

```bash
EVDIR=docs/research/2026-04-29-full-graph-spike/evidence/2026-04-29-1613
RUN_ID=smoke-postfix-v2-1777493912

docker rm -f nvllm 2>/dev/null
docker run -d --name nvllm \
  --gpus all --ipc=host --entrypoint /bin/bash \
  -v "$PWD:/workspace" -w /workspace \
  -v /tmp/nvllm-cute-cache:/opt/vllm/kernel_cache \
  -e B12X_CUTE_COMPILE_CACHE_DIR_ROOT=/opt/vllm/kernel_cache \
  nvllm:gb10 -c 'sleep 1200'
docker exec nvllm /workspace/.venv/bin/python \
  docs/research/2026-04-29-full-graph-spike/cache_smoke.py \
  --phase=cold --run-id="$RUN_ID"

docker rm -f nvllm
docker run -d --name nvllm \
  --gpus all --ipc=host --entrypoint /bin/bash \
  -v "$PWD:/workspace" -w /workspace \
  -v /tmp/nvllm-cute-cache:/opt/vllm/kernel_cache \
  -e B12X_CUTE_COMPILE_CACHE_DIR_ROOT=/opt/vllm/kernel_cache \
  nvllm:gb10 -c 'sleep 1200'
docker exec nvllm /workspace/.venv/bin/python \
  docs/research/2026-04-29-full-graph-spike/cache_smoke.py \
  --phase=warm --run-id="$RUN_ID"
docker rm -f nvllm
```

## Files in this evidence dir

- `cache_smoke_cold.json`, `cache_smoke_cold.raw.log` — v2 cold pass
- `cache_smoke_warm.json`, `cache_smoke_warm.raw.log` — v2 warm pass
- `cold_with_keydbg.json`, `cold_key_debug.log` — KEY_DEBUG dumps that
  caught the v1 name-binding misalignment
- `warm_with_keydbg.json`, `warm_key_debug.log` — same, warm phase
- `gate_g1_postfix_verdict.md` (this file)
