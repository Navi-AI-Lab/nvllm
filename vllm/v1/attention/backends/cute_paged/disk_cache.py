"""CuTe DSL compilation disk cache.

Ported from lukealonso/b12x@c469c66 b12x/cute/runtime_patches.py.
Caches compiled CuTe DSL kernels to disk so subsequent launches skip
the NVRTC compilation step entirely. Used during docker build for
zero cold-start serving, and at runtime for iterative development.

The public API (build_cache_key, store_to_disk, load_from_disk) uses
pickle for general objects. The monkey-patch path
(apply_disk_cache_patch) uses CUTLASS-native ExternalBinaryModule
when available, falling back to pickle.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import logging
import os
import pickle
import sys
import threading
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False
_PATCH_LOCK = threading.Lock()
_CUTE_PAGED_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Public API — simple cache key / store / load for testing and direct use
# ---------------------------------------------------------------------------

def build_cache_key(source: str, args: dict, toolchain_version: str) -> str:
    """Build a deterministic SHA256 cache key from source, args, and toolchain.

    Args:
        source: Kernel source code or function source string.
        args: Dictionary of compilation arguments (shapes, dtypes, etc.).
        toolchain_version: CUTLASS/toolchain version string.

    Returns:
        64-character hex SHA256 digest.
    """
    payload = repr(("nvllm_cute_cache_v1", source, args, toolchain_version))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def store_to_disk(cache_dir: str, key: str, obj: object) -> None:
    """Atomically store a compiled object to disk via pickle.

    Writes to a temporary file first, then atomically replaces the
    target path. This prevents corrupted reads from concurrent access.

    Security note: only caches objects we compiled ourselves — the cache
    directory is local and not exposed to untrusted input.
    """
    subdir = os.path.join(cache_dir, key[:2])
    os.makedirs(subdir, exist_ok=True)
    target = os.path.join(subdir, key)
    tmp = target + ".tmp"
    try:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
    except Exception:
        # Clean up tmp on failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_from_disk(cache_dir: str, key: str) -> object | None:
    """Load a cached compiled object from disk.

    Returns None if the file is missing, corrupted, or unpickling fails.

    Security note: only loads from our own local cache directory — not
    exposed to untrusted input.
    """
    path = os.path.join(cache_dir, key[:2], key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.loads(f.read())  # noqa: S301
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full compilation cache key — used by the monkey-patch
# Ported from b12x's _build_compile_disk_cache_key and helpers.
# ---------------------------------------------------------------------------

def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


@lru_cache(maxsize=1)
def _runtime_toolchain_key() -> tuple[object, ...]:
    """Capture versions of Python, PyTorch, CUTLASS, and CUDA bindings."""
    torch_version = _distribution_version("torch")
    torch_cuda_version = ""
    try:
        import torch
        if not torch_version:
            torch_version = getattr(torch, "__version__", "")
        torch_cuda_version = getattr(torch.version, "cuda", "") or ""
    except Exception:
        pass

    cutlass_version = _distribution_version("nvidia-cutlass-dsl")
    if not cutlass_version:
        cutlass_version = _distribution_version("cutlass")
    if not cutlass_version:
        try:
            import cutlass
            cutlass_version = getattr(cutlass, "__version__", "")
        except Exception:
            cutlass_version = ""

    return (
        ("python", sys.implementation.name, sys.version_info[:3]),
        ("torch", torch_version),
        ("torch_cuda", torch_cuda_version),
        ("cutlass_dsl", cutlass_version),
        ("cuda_bindings", _distribution_version("cuda-bindings")),
    )


def _compile_environment_key() -> tuple[tuple[str, str], ...]:
    """Capture env vars that affect NVRTC compilation."""
    compile_env_vars = (
        "CC", "CXX", "CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH",
        "CUDACXX", "CUTE_DSL_ARCH", "NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS",
    )
    return tuple((name, os.environ.get(name, "")) for name in compile_env_vars)


def _iter_fingerprint_files(root: Path) -> list[Path]:
    """List all source files under root, excluding __pycache__ and .pyc."""
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    files.sort()
    return files


def _tree_state(root: Path) -> tuple[tuple[str, int, int], ...]:
    entries = []
    for path in _iter_fingerprint_files(root):
        stat = path.stat()
        entries.append((
            str(path.relative_to(root)), stat.st_mtime_ns, stat.st_size,
        ))
    return tuple(entries)


@lru_cache(maxsize=8)
def _tree_fingerprint_cached(
    root_str: str, state: tuple[tuple[str, int, int], ...],
) -> str:
    root = Path(root_str)
    digest = hashlib.sha256()
    for rel_path, _mtime_ns, _size in state:
        path = root / rel_path
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> str:
    return _tree_fingerprint_cached(str(root), _tree_state(root))


def _package_fingerprint() -> str:
    """SHA256 of all source files in the cute_paged package."""
    return _tree_fingerprint(_CUTE_PAGED_ROOT)


def _function_fingerprint(func: Any) -> tuple[str, str, str]:
    """Fingerprint a callable via source or bytecode."""
    func = inspect.unwrap(func)
    module = getattr(func, "__module__", "")
    qualname = getattr(
        func, "__qualname__", getattr(func, "__name__", type(func).__qualname__),
    )
    # For our own package, use the tree fingerprint
    if module.startswith("vllm.v1.attention.backends.cute_paged"):
        return module, qualname, f"cute_paged:{_package_fingerprint()}"
    try:
        source = inspect.getsource(func)
        payload = source.encode("utf-8")
    except (OSError, TypeError):
        code = getattr(func, "__code__", None)
        if code is None:
            payload = repr(func).encode("utf-8")
        else:
            payload = repr((
                code.co_code, code.co_consts, code.co_names,
                code.co_varnames, code.co_argcount, code.co_kwonlyargcount,
            )).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return module, qualname, digest


def _structural_cache_key(value: Any, visited: set[int] | None = None) -> Any:
    """Recursively build a hashable structural representation of a value.

    Handles dicts, lists, tuples, sets, functions, CUTLASS runtime tensors,
    and arbitrary objects with __dict__. Detects cycles.
    """
    if visited is None:
        visited = set()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Path):
        return ("path", str(value))
    if inspect.isfunction(value) or inspect.ismethod(value):
        return _function_fingerprint(value)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, dict):
        return tuple(sorted(
            (_structural_cache_key(k, visited), _structural_cache_key(v, visited))
            for k, v in value.items()
        ))
    if isinstance(value, (tuple, list)):
        return tuple(_structural_cache_key(v, visited) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_structural_cache_key(v, visited) for v in value))

    # CUTLASS runtime tensor types
    type_name = type(value).__name__
    type_module = type(value).__module__
    if type_name == "CUstream" and type_module.startswith("cuda.bindings"):
        return ("cuda_stream",)
    if type_module == "cutlass.cute.runtime" and type_name == "_Tensor":
        dtype = getattr(value, "_dtype", getattr(value, "element_type", None))
        shape = tuple(value.shape)
        stride = tuple(value.stride)
        memspace = getattr(value, "memspace", getattr(value, "_memspace", None))
        return ("runtime_tensor", dtype, shape, stride, memspace)

    # __cache_key__ protocol
    cache_key_attr = getattr(value, "__cache_key__", None)
    if cache_key_attr is not None:
        return (
            "cache_key", type_module, type_name,
            _structural_cache_key(cache_key_attr, visited),
        )

    # Generic object with __dict__
    object_id = id(value)
    if object_id in visited:
        return ("cycle", type_module, type_name)
    if hasattr(value, "__dict__"):
        visited.add(object_id)
        try:
            return (
                "object", type_module, type_name,
                tuple(sorted(
                    (k, _structural_cache_key(v, visited))
                    for k, v in vars(value).items()
                )),
            )
        finally:
            visited.remove(object_id)

    return ("repr", type_module, type_name, repr(value))


def _compile_options_cache_key(compile_callable: Any) -> tuple[str, ...]:
    """Extract serialized compile options from a CompileCallable."""
    compile_options = getattr(compile_callable, "_compile_options", None)
    if compile_options is None:
        return ()
    options = getattr(compile_options, "options", {})
    serialized = []
    for option in options.values():
        value = option.serialize()
        if value:
            serialized.append(value)
    return tuple(serialized)


def _build_full_cache_key(
    compile_callable: Any,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """Build the full cache key for a CUTLASS compilation.

    Includes function fingerprint, package state, toolchain versions,
    compile arguments, compile options, and environment variables.
    """
    payload = (
        "nvllm_cute_compile_cache_v1",
        _function_fingerprint(func),
        _package_fingerprint(),
        _runtime_toolchain_key(),
        _structural_cache_key(args),
        _structural_cache_key(kwargs),
        _compile_options_cache_key(compile_callable),
        _compile_environment_key(),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CUTLASS-native store/load — uses ExternalBinaryModule when available
# ---------------------------------------------------------------------------

def _cache_prefix(cache_key: str) -> str:
    return f"nvllm_cute_{cache_key}"


def _native_cache_path(cache_dir: str, cache_key: str) -> Path:
    return Path(cache_dir) / cache_key[:2] / f"{cache_key}.o"


def _load_native(cache_dir: str, cache_key: str) -> Any | None:
    """Try loading via CUTLASS ExternalBinaryModule (compiled .o files)."""
    object_path = _native_cache_path(cache_dir, cache_key)
    if not object_path.exists():
        return None
    try:
        from cutlass.base_dsl.export.external_binary_module import (
            ExternalBinaryModule,
        )
        module = ExternalBinaryModule(str(object_path))
        return getattr(module, _cache_prefix(cache_key))
    except Exception:
        return None


def _store_native(cache_dir: str, cache_key: str, compiled: Any) -> bool:
    """Try storing via CUTLASS dump_to_object. Returns True on success."""
    if not hasattr(compiled, "dump_to_object"):
        return False
    try:
        object_path = _native_cache_path(cache_dir, cache_key)
        object_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = object_path.with_suffix(".tmp")
        object_bytes = compiled.dump_to_object(_cache_prefix(cache_key))
        with open(tmp_path, "wb") as f:
            f.write(object_bytes)
        os.replace(tmp_path, object_path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Monkey-patch entry point
# ---------------------------------------------------------------------------

_COMPILE_ONLY_CACHE_WARNING = "Cache is disabled as user wants to compile only."


def apply_disk_cache_patch(cache_dir: str, enabled: bool = True) -> None:
    """Apply the CuTe DSL disk cache monkey-patch.

    Wraps ``CompileCallable._compile`` to check the disk cache before
    running NVRTC compilation. Also suppresses the "compile only" warning.

    Safe to call when CUTLASS is not installed — catches ImportError
    and logs a warning.

    Args:
        cache_dir: Root directory for cached compiled objects.
        enabled: If False, returns immediately without patching.
    """
    global _PATCHED
    if not enabled:
        return
    if _PATCHED:
        return
    with _PATCH_LOCK:
        if _PATCHED:
            logger.debug("CuTe DSL disk cache patch already applied, skipping.")
            return
        try:
            from cutlass.base_dsl.compiler import CompileCallable
            from cutlass.base_dsl.dsl import BaseDSL
        except (AttributeError, ImportError) as e:
            logger.warning(
                "CuTe DSL disk cache patch failed (CUTLASS not available): %s. "
                "Falling back to in-memory cache (first request will be slow).",
                e,
            )
            return

        original_compile = CompileCallable._compile
        original_print_warning = BaseDSL.print_warning
        original_print_warning_once = BaseDSL.print_warning_once

        @wraps(original_compile)
        def _cached_compile(self, func, *args, **kwargs):
            key = _build_full_cache_key(self, func, args, kwargs)
            # Try native CUTLASS format first, then pickle fallback
            cached = _load_native(cache_dir, key)
            if cached is not None:
                return cached
            cached = load_from_disk(cache_dir, key)
            if cached is not None:
                return cached
            result = original_compile(self, func, *args, **kwargs)
            # Try native store, fall back to pickle
            if not _store_native(cache_dir, key, result):
                try:
                    store_to_disk(cache_dir, key, result)
                except Exception:
                    pass
            return result

        @wraps(original_print_warning)
        def _patched_print_warning(self, message):
            if message == _COMPILE_ONLY_CACHE_WARNING:
                return None
            return original_print_warning(self, message)

        @wraps(original_print_warning_once)
        def _patched_print_warning_once(self, message):
            if message == _COMPILE_ONLY_CACHE_WARNING:
                return None
            return original_print_warning_once(self, message)

        CompileCallable._compile = _cached_compile
        BaseDSL.print_warning = _patched_print_warning
        BaseDSL.print_warning_once = _patched_print_warning_once
        _PATCHED = True
        logger.info("CuTe DSL disk cache enabled: %s", cache_dir)
