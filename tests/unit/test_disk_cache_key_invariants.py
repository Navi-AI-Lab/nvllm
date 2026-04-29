"""Compile-cache key + KEY_DEBUG-probe structural invariants.

Companion to ``test_disk_cache_pointer_canonicalization.py``. That file
locks in the pointer-canonicalization rule (Int64 args don't perturb
the key). This file locks in the broader invariants the
``_build_full_cache_payload`` / ``_write_key_debug_payload`` layer
relies on:

  - the refactor: ``_build_full_cache_key(...)`` produces
    ``sha256(repr(_build_full_cache_payload(...)).encode())``. If a
    future change splits the two paths, the dump's
    ``payload_hash_matches_key`` invariant becomes decorative.

  - the probe: KEY_DEBUG=0 is silent; KEY_DEBUG=1 writes one JSON per
    call under ``<cache_dir>/_debug/<func-slug>.<call_index>.<run>.json``;
    ``payload_hash_matches_key == True`` always.

  - the call_index counter: per-(func_slug, run) sequence, independent
    across run tags (so cold and warm counters don't interleave even
    in single-process invocation).

  - kwargs key-order independence: ``_structural_args_cache_key``
    sorts kwargs, so insertion order must not change the key.

  - ``_is_runtime_pointer_value`` at the leaf level, so the
    canonicalization gate is testable without going through the whole
    key builder.

  - synthetic CUTLASS-runtime ``_Tensor`` shape/stride/dtype
    sensitivity (changing the shape DOES change the key).

All tests use synthetic fakes — no real CuTe compile, no real cutlass
import. The fakes set ``__module__`` / ``__qualname__`` to mimic the
real CUTLASS classes so the value-type-based canonicalization fires.
"""

from __future__ import annotations

import hashlib
import json


# ---------------------------------------------------------------------------
# Fake helpers (mirror the convention from test_disk_cache_pointer_canon).
# ---------------------------------------------------------------------------

def _make_fake_int64_class():
    class Int64:
        def __init__(self, value: int):
            self.value = value

        def __repr__(self):
            return f"Int64({self.value:#x})"

    Int64.__module__ = "cutlass.base_dsl.typing"
    Int64.__qualname__ = "Int64"
    return Int64


def _make_fake_int32_class():
    class Int32:
        def __init__(self, value: int):
            self.value = value

        def __repr__(self):
            return f"Int32({self.value})"

    Int32.__module__ = "cutlass.base_dsl.typing"
    Int32.__qualname__ = "Int32"
    return Int32


def _make_fake_runtime_tensor_class():
    """Mirror cutlass.cute.runtime._Tensor's shape/stride/dtype layout."""

    class _Tensor:
        def __init__(self, dtype, shape, stride, memspace="gmem"):
            self._dtype = dtype
            self.shape = tuple(shape)
            self.stride = tuple(stride)
            self.memspace = memspace

    _Tensor.__module__ = "cutlass.cute.runtime"
    _Tensor.__qualname__ = "_Tensor"
    return _Tensor


class _FakeCompileCallable:
    _compile_options = None


# ---------------------------------------------------------------------------
# Refactor invariant: _build_full_cache_key == sha256(repr(payload))
# ---------------------------------------------------------------------------

def test_build_full_cache_key_hashes_repr_of_payload():
    """The refactor's contract: the key is exactly
    ``sha256(repr(_build_full_cache_payload(...)).encode("utf-8")).hexdigest()``.

    If a future change splits these (e.g. switches to a different
    serialization or sorts the payload differently), the
    ``payload_hash_matches_key`` field in KEY_DEBUG dumps becomes
    decorative. This test catches that.
    """
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    Int64 = _make_fake_int64_class()
    Int32 = _make_fake_int32_class()

    def _fake_jit(*positional, **kw):
        return None

    args = (Int64(0xDEAD), Int32(32), Int32(8))
    kwargs = {"page_size": Int32(16), "scale": 1.0}

    payload = dc._build_full_cache_payload(_FakeCompileCallable(), _fake_jit, args, kwargs)
    expected = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    actual = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, args, kwargs)

    assert actual == expected


def test_build_full_cache_payload_tuple_shape():
    """Lock in the tuple slot layout the probe depends on."""
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    def _fake_jit(*a, **k):
        return None

    payload = dc._build_full_cache_payload(_FakeCompileCallable(), _fake_jit, (), {})
    assert isinstance(payload, tuple)
    assert len(payload) == 7
    assert payload[0] == dc._CACHE_KEY_VERSION
    # Slots 1..6 must be the same components the probe reads at
    # _write_key_debug_payload's component_hashes step:
    #   1=func_fp, 2=pkg_fp, 3=toolchain, 4=args_key, 5=opts, 6=env
    assert isinstance(payload[1], tuple)  # func fingerprint
    assert isinstance(payload[2], str)    # package fingerprint (hex digest)
    assert isinstance(payload[3], tuple)  # toolchain
    # args_key starts with ('ptr_canonical_v1', ...) per
    # _structural_args_cache_key.
    assert isinstance(payload[4], tuple)
    assert payload[4][0] == "ptr_canonical_v1"
    assert isinstance(payload[5], tuple)  # compile options
    assert isinstance(payload[6], tuple)  # compile env


# ---------------------------------------------------------------------------
# Probe gating + filesystem layout
# ---------------------------------------------------------------------------

def test_probe_disabled_writes_nothing(tmp_path, monkeypatch):
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    monkeypatch.delenv("B12X_CUTE_COMPILE_KEY_DEBUG", raising=False)
    monkeypatch.delenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", raising=False)

    def _fake_jit(*a, **k):
        return None

    key = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, (), {})
    result = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, (), {},
    )
    assert result is None
    assert not (tmp_path / "_debug").exists()


def test_probe_enabled_writes_expected_path(tmp_path, monkeypatch):
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG", "1")
    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "cold")
    # Keep counters fresh per test so call_index assertions are stable.
    dc._CALL_INDEX_COUNTERS.clear()

    def _fake_jit(*a, **k):
        return None
    _fake_jit.__qualname__ = "test_probe_enabled_writes_expected_path._fake_jit"

    args = ()
    kwargs = {}
    key = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, args, kwargs)
    out = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, args, kwargs,
    )

    assert out is not None
    assert out.parent == tmp_path / "_debug"
    assert out.name.endswith(".cold.json")
    # Filename pattern: <func-slug>.<call_index:03d>.<run>.json
    parts = out.name.rsplit(".", 3)
    assert len(parts) == 4
    func_slug, call_index_str, run_str, ext = parts
    assert ext == "json"
    assert run_str == "cold"
    assert call_index_str == "000"
    assert func_slug.endswith("_fake_jit")


def test_probe_payload_hash_matches_key(tmp_path, monkeypatch):
    """The whole point of refactoring _build_full_cache_payload: the
    probe's hash invariant must hold every time."""
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG", "1")
    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "cold")
    dc._CALL_INDEX_COUNTERS.clear()

    Int64 = _make_fake_int64_class()
    Int32 = _make_fake_int32_class()

    def _fake_jit(*a, **k):
        return None

    args = (Int64(0x100), Int32(32))
    kwargs = {"page_size": Int32(16)}

    key = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, args, kwargs)
    out = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, args, kwargs,
    )

    record = json.loads(out.read_text())
    assert record["key"] == key
    assert record["payload_hash"] == key
    assert record["payload_hash_matches_key"] is True


def test_probe_run_label_sanitized(tmp_path, monkeypatch):
    """Path-traversal characters in the run label must not escape
    the _debug/ directory.

    The real safety property is: the resolved final path stays under
    ``<cache_dir>/_debug/``. Dots in the filename are fine (they are
    legal filename chars and the run-label sanitizer correctly keeps
    them); slashes are not, which is what the sanitizer strips.
    """
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG", "1")
    # Slash + dot-dot — slashes must become underscores; dots are fine.
    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "../oops/cold")
    dc._CALL_INDEX_COUNTERS.clear()

    def _fake_jit(*a, **k):
        return None

    key = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, (), {})
    out = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, (), {},
    )

    assert out is not None
    # The resolved path must stay under <cache_dir>/_debug/. This is
    # the real path-traversal check; a sanitizer that misses '/' would
    # let the file escape, but '..' as filename text is harmless.
    debug_root = (tmp_path / "_debug").resolve()
    assert out.resolve().is_relative_to(debug_root)
    # Filename itself contains no path separator after sanitization.
    assert "/" not in out.name
    assert "\\" not in out.name
    assert "\x00" not in out.name


# ---------------------------------------------------------------------------
# call_index counter behavior
# ---------------------------------------------------------------------------

def test_call_index_increments_per_func_run(tmp_path, monkeypatch):
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG", "1")
    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "cold")
    dc._CALL_INDEX_COUNTERS.clear()

    def _fake_jit(*a, **k):
        return None
    _fake_jit.__qualname__ = "increments_per_func_run.kernel"

    key = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, (), {})
    p0 = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, (), {},
    )
    p1 = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, (), {},
    )
    p2 = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, (), {},
    )

    assert p0.name.endswith(".000.cold.json")
    assert p1.name.endswith(".001.cold.json")
    assert p2.name.endswith(".002.cold.json")


def test_call_index_independent_across_runs(tmp_path, monkeypatch):
    """cold and warm run-tags must have independent counters even
    inside a single process — otherwise pairing by (func_slug,
    call_index) breaks."""
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG", "1")
    dc._CALL_INDEX_COUNTERS.clear()

    def _fake_jit(*a, **k):
        return None
    _fake_jit.__qualname__ = "independent_across_runs.kernel"

    key = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, (), {})

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "cold")
    p_cold = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, (), {},
    )

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "warm")
    p_warm = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, (), {},
    )

    assert p_cold.name.endswith(".000.cold.json")
    assert p_warm.name.endswith(".000.warm.json")


def test_call_index_independent_across_func_slugs(tmp_path, monkeypatch):
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG", "1")
    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "cold")
    dc._CALL_INDEX_COUNTERS.clear()

    def _fake_jit_a(*a, **k):
        return None
    _fake_jit_a.__qualname__ = "slug_independence.KernelA"

    def _fake_jit_b(*a, **k):
        return None
    _fake_jit_b.__qualname__ = "slug_independence.KernelB"

    key_a = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit_a, (), {})
    key_b = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit_b, (), {})

    pa = dc._write_key_debug_payload(
        str(tmp_path), key_a, _FakeCompileCallable(), _fake_jit_a, (), {},
    )
    pb = dc._write_key_debug_payload(
        str(tmp_path), key_b, _FakeCompileCallable(), _fake_jit_b, (), {},
    )

    # Both should be call_index=0 because they have different slugs.
    assert pa.name.endswith(".000.cold.json")
    assert pb.name.endswith(".000.cold.json")
    assert pa.name != pb.name  # different slugs


# ---------------------------------------------------------------------------
# kwargs ordering invariant
# ---------------------------------------------------------------------------

def test_kwargs_insertion_order_does_not_change_key():
    """``_structural_args_cache_key`` sorts kwargs, so dict-insertion
    order must not affect the key."""
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    Int32 = _make_fake_int32_class()

    def _fake_jit(*a, **k):
        return None

    a = {"page_size": Int32(16), "num_q_heads": Int32(32), "scale": 1.0}
    b = {"scale": 1.0, "num_q_heads": Int32(32), "page_size": Int32(16)}

    k_a = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, (), a)
    k_b = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, (), b)
    assert k_a == k_b


# ---------------------------------------------------------------------------
# Synthetic CUTLASS _Tensor shape/stride/dtype sensitivity
# ---------------------------------------------------------------------------

def test_runtime_tensor_shape_change_changes_key():
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    _Tensor = _make_fake_runtime_tensor_class()

    def _fake_jit(*a, **k):
        return None

    base = (_Tensor("bf16", (1, 32, 256), (8192, 256, 1)),)
    altered = (_Tensor("bf16", (2, 32, 256), (8192, 256, 1)),)  # different shape

    k_base = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, base, {})
    k_altered = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, altered, {})
    assert k_base != k_altered


def test_runtime_tensor_stride_change_changes_key():
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    _Tensor = _make_fake_runtime_tensor_class()

    def _fake_jit(*a, **k):
        return None

    base = (_Tensor("bf16", (1, 32, 256), (8192, 256, 1)),)
    altered = (_Tensor("bf16", (1, 32, 256), (4096, 128, 1)),)  # different stride

    k_base = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, base, {})
    k_altered = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, altered, {})
    assert k_base != k_altered


def test_runtime_tensor_dtype_change_changes_key():
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    _Tensor = _make_fake_runtime_tensor_class()

    def _fake_jit(*a, **k):
        return None

    base = (_Tensor("bf16", (1, 32, 256), (8192, 256, 1)),)
    altered = (_Tensor("fp16", (1, 32, 256), (8192, 256, 1)),)  # different dtype

    k_base = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, base, {})
    k_altered = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, altered, {})
    assert k_base != k_altered


# ---------------------------------------------------------------------------
# _is_runtime_pointer_value leaf-level checks
# ---------------------------------------------------------------------------

def test_is_runtime_pointer_value_classifies_int64_only():
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    Int64 = _make_fake_int64_class()
    Int32 = _make_fake_int32_class()

    assert dc._is_runtime_pointer_value(Int64(0x1234)) is True
    # Int32 is a real shape scalar — must NOT be canonicalized away.
    assert dc._is_runtime_pointer_value(Int32(32)) is False
    # Plain Python ints are not from cutlass.base_dsl.typing.
    assert dc._is_runtime_pointer_value(42) is False
    assert dc._is_runtime_pointer_value(0xDEADBEEF) is False
    # Strings, floats, None.
    assert dc._is_runtime_pointer_value("k_ptr") is False
    assert dc._is_runtime_pointer_value(1.0) is False
    assert dc._is_runtime_pointer_value(None) is False


# ---------------------------------------------------------------------------
# Component-hash determinism (cold-vs-warm proxy)
# ---------------------------------------------------------------------------

def test_component_hashes_deterministic_within_process(tmp_path, monkeypatch):
    """Two probe dumps for the same call must produce identical
    component_hashes — cold vs warm equivalence proxy without needing
    a real container restart."""
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG", "1")
    dc._CALL_INDEX_COUNTERS.clear()

    Int64 = _make_fake_int64_class()
    Int32 = _make_fake_int32_class()

    def _fake_jit(*a, **k):
        return None
    _fake_jit.__qualname__ = "component_hashes_determinism.kernel"

    args = (Int64(0xAAA), Int32(32))
    kwargs = {"scale": 1.0}

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "cold")
    key = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, args, kwargs)
    p_cold = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, args, kwargs,
    )

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "warm")
    p_warm = dc._write_key_debug_payload(
        str(tmp_path), key, _FakeCompileCallable(), _fake_jit, args, kwargs,
    )

    cold = json.loads(p_cold.read_text())
    warm = json.loads(p_warm.read_text())

    assert cold["key"] == warm["key"]
    assert cold["component_hashes"] == warm["component_hashes"]


def test_component_hashes_identify_changed_component(tmp_path, monkeypatch):
    """Changing a single arg must perturb ONLY component_hashes['args'],
    not the other top-level component slots."""
    from vllm.v1.attention.backends.cute_paged import disk_cache as dc

    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG", "1")
    monkeypatch.setenv("B12X_CUTE_COMPILE_KEY_DEBUG_RUN", "cold")
    dc._CALL_INDEX_COUNTERS.clear()

    Int32 = _make_fake_int32_class()

    def _fake_jit(*a, **k):
        return None
    _fake_jit.__qualname__ = "identify_changed_component.kernel"

    args_a = (Int32(32),)
    args_b = (Int32(64),)  # only the args differ

    key_a = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, args_a, {})
    p_a = dc._write_key_debug_payload(
        str(tmp_path), key_a, _FakeCompileCallable(), _fake_jit, args_a, {},
    )
    key_b = dc._build_full_cache_key(_FakeCompileCallable(), _fake_jit, args_b, {})
    p_b = dc._write_key_debug_payload(
        str(tmp_path), key_b, _FakeCompileCallable(), _fake_jit, args_b, {},
    )

    a = json.loads(p_a.read_text())["component_hashes"]
    b = json.loads(p_b.read_text())["component_hashes"]

    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"args"}, (
        f"only component_hashes['args'] should differ; got {differing}"
    )
