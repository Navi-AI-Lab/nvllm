# Kernel Insights: b12x CuTe Attention & Disk Cache

**Date:** 2026-04-10
**Source:** [lukealonso/b12x](https://github.com/lukealonso/b12x)
**Pinned commit:** [`c469c66`](https://github.com/lukealonso/b12x/tree/c469c6637f6251adefc282956f5392e559ea915d)
**License:** Apache-2.0

---

## What Was Borrowed

### 1. CuTe DSL Compilation Disk Cache

**Source file:** [`b12x/cute/runtime_patches.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/runtime_patches.py)

| Piece | Source lines | Our file |
|---|---|---|
| Cache key building (SHA256 of function + toolchain + args) | [L342-L359](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/runtime_patches.py#L342-L359) | `disk_cache.py:_build_full_cache_key` |
| Structural cache key (recursive hashable repr of args) | [L169-L313](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/runtime_patches.py#L169-L313) | `disk_cache.py:_structural_cache_key` |
| Native load via ExternalBinaryModule | [L367-L377](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/runtime_patches.py#L367-L377) | `disk_cache.py:_load_native` |
| Native store via dump_to_object + atomic replace | [L380-L391](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/runtime_patches.py#L380-L391) | `disk_cache.py:_store_native` |
| CompileCallable monkey-patch + warning suppression | [L394-L422](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/runtime_patches.py#L394-L422) | `disk_cache.py:apply_disk_cache_patch` |
| Package tree fingerprinting | [L94-L99](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/runtime_patches.py#L94-L99) | `disk_cache.py:_package_fingerprint` |
| Runtime toolchain key (Python + torch + CUTLASS versions) | [L108-L135](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/runtime_patches.py#L108-L135) | `disk_cache.py:_runtime_toolchain_key` |

**Why:** CuTe DSL's built-in `@functools.cache` is in-memory only — compiled
kernels die with the process. Without disk persistence, pre-compiling during
`docker build` is useless because the compiled objects don't survive into the
runtime container. b12x solved this with a monkey-patch on `CompileCallable._compile`
that intercepts compilation, builds a deterministic cache key, and persists the
compiled `.o` file to disk.

**How it was adapted:**
- Package fingerprint root changed from `b12x` to `cute_paged`
- Cache key version prefix changed from `b12x_cute_compile_cache_v2` to `nvllm_cute_compile_cache_v1`
- Added serialization-based fallback for objects that don't support `dump_to_object`
- Added simple public API (`build_cache_key`, `store_to_disk`, `load_from_disk`) for testing
- Environment variable names kept compatible (`B12X_CUTE_COMPILE_CACHE_DIR`)

### 2. FP8 Dequantization Pattern (Deferred)

**Source file:** `b12x/attention/paged/forward_paged.py`

The `fp8x4_e4m3_to_bfloat2x2` dequantization and descale-on-P pattern
(applying `v_scale` to P in FP32 before BF16 cast) is handled by the
`.contiguous()` copy path in the current implementation. Direct FP8 dequant
in the CuTe DSL kernel is deferred — it will be revisited if/when the
`.contiguous()` copy overhead is eliminated via strided addressing.

---

## Verification Checklist

- [x] All permalink URLs resolve (tested 2026-04-10)
- [x] License compatibility confirmed (Apache-2.0 to Apache-2.0)
- [x] Per-piece links provided for all borrowed code
- [x] README acknowledgment added
