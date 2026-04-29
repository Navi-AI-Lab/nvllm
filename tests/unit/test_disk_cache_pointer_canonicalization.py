"""Pointer-typed args must not perturb the compile cache key.

Reproduces Gate G1's failure mode: decode kernel passes
Int64(kv_cache.data_ptr()) into cute.compile, and across processes the
pointer value differs — but the generated PTX shouldn't, so the key
must be stable.

Why value-type-based, not parameter-name-based:
inspect.signature on cute.jit-decorated bound methods reports `self`
as the first positional parameter, shifting every other name by one
position. Verified empirically with B12X_CUTE_COMPILE_KEY_DEBUG=1
during Gate G1 retry: cold and warm KEY_DEBUG dumps differed only
in the `value` of an Int64-typed entry the binder labeled `query`,
which was actually `k_ptr` shifted by `self` consuming `q_flat`.

Convention in this codebase: cutlass `Int64` is *always* a runtime
pointer (data_ptr() output); `Int32` is *always* a shape or flag.
We canonicalize by VALUE TYPE accordingly.

This test uses fake module/qualname stand-ins so we don't need the
real cutlass package to be importable in the unit-test environment.
"""

from __future__ import annotations


def _make_fake_int64_class():
    """Build a class whose module/qualname mimic cutlass.base_dsl.typing.Int64."""

    class Int64:
        def __init__(self, value: int):
            self.value = value

        def __repr__(self):
            return f"Int64({self.value:#x})"

    Int64.__module__ = "cutlass.base_dsl.typing"
    Int64.__qualname__ = "Int64"
    return Int64


def _make_fake_int32_class():
    """Real shape scalar — must stay in the key."""

    class Int32:
        def __init__(self, value: int):
            self.value = value

        def __repr__(self):
            return f"Int32({self.value})"

    Int32.__module__ = "cutlass.base_dsl.typing"
    Int32.__qualname__ = "Int32"
    return Int32


def _shape_args(Int32):
    """Common Int32 shape args reused by both calls."""
    return (
        Int32(32),  # num_q_heads
        Int32(8),   # num_kv_heads
        Int32(4096),  # kv_page_stride
        Int32(8),   # grid_x
        Int32(1),   # grid_y
        Int32(1),   # grid_z
    )


def test_pointer_args_do_not_change_cache_key():
    from vllm.v1.attention.backends.cute_paged import disk_cache

    Int64 = _make_fake_int64_class()
    Int32 = _make_fake_int32_class()

    def _fake_jit(*positional):
        return None

    shapes = _shape_args(Int32)

    # Different pointer addresses, identical shape args.
    args_a = (
        Int64(0x70AAAA0000),  # k_ptr
        Int64(0x70AAAA1000),  # v_ptr
        Int64(0x70AAAA2000),  # wo_weight_ptr
        Int64(0x70AAAA3000),  # gate_ptr
        *shapes,
    )
    args_b = (
        Int64(0x70BBBB1111),
        Int64(0x70BBBB2222),
        Int64(0x70BBBB3333),
        Int64(0x70BBBB4444),
        *shapes,
    )

    class _FakeCompileCallable:
        _compile_options = None

    cc = _FakeCompileCallable()
    key_a = disk_cache._build_full_cache_key(cc, _fake_jit, args_a, {})
    key_b = disk_cache._build_full_cache_key(cc, _fake_jit, args_b, {})

    assert key_a == key_b, (
        f"pointer addresses leaked into compile key: "
        f"key_a={key_a[:16]} key_b={key_b[:16]}"
    )


def test_real_shape_arg_change_does_change_key():
    from vllm.v1.attention.backends.cute_paged import disk_cache

    Int64 = _make_fake_int64_class()
    Int32 = _make_fake_int32_class()

    def _fake_jit(*positional):
        return None

    base = (
        Int64(0x10000),
        Int64(0x20000),
        Int32(32),  # num_q_heads
        Int32(8),   # num_kv_heads
    )
    altered = (
        Int64(0x10000),
        Int64(0x20000),
        Int32(16),  # num_q_heads CHANGED
        Int32(8),
    )

    class _FakeCompileCallable:
        _compile_options = None

    cc = _FakeCompileCallable()
    k1 = disk_cache._build_full_cache_key(cc, _fake_jit, base, {})
    k2 = disk_cache._build_full_cache_key(cc, _fake_jit, altered, {})
    assert k1 != k2, (
        "real shape arg num_q_heads change must change the cache key"
    )


def test_kwargs_pointer_canonicalization():
    """Same idea but exercising the kwargs path too."""
    from vllm.v1.attention.backends.cute_paged import disk_cache

    Int64 = _make_fake_int64_class()
    Int32 = _make_fake_int32_class()

    def _fake_jit(*positional, **kw):
        return None

    common_kwargs = {"num_q_heads": Int32(32), "kv_page_stride": Int32(4096)}
    a = {**common_kwargs, "k_ptr": Int64(0x100), "v_ptr": Int64(0x200)}
    b = {**common_kwargs, "k_ptr": Int64(0x999), "v_ptr": Int64(0xAAA)}

    class _FakeCompileCallable:
        _compile_options = None

    cc = _FakeCompileCallable()
    k1 = disk_cache._build_full_cache_key(cc, _fake_jit, (), a)
    k2 = disk_cache._build_full_cache_key(cc, _fake_jit, (), b)
    assert k1 == k2, (
        "pointer-typed kwargs must not affect the cache key"
    )
