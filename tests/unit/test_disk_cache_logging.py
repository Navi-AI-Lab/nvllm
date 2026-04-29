"""HIT/MISS log lines on the disk-cache path.

Tests `_cached_compile` (the wrapped CompileCallable._compile after
apply_disk_cache_patch) emits 'CuTe disk cache HIT' on cache hit and
'CuTe disk cache MISS' on miss. These lines are the source of truth
Gate G2 uses to assert cache hit at serve time — absence of the
'Compiling PhaseE_Beta_Kernel β-coop full' line is NOT sufficient
because that logger.info fires before _compile_coop_full delegates to
cute.compile, regardless of cache state.
"""

from __future__ import annotations

import logging


# Module-level so the disk-fallback store path can serialize an instance.
# (Locally-defined classes can't be re-imported by the loader.)
class _SerializableFakeResult:
    pass


def test_miss_then_hit_emit_distinct_log_lines(tmp_path, monkeypatch, caplog):
    # vLLM configures the parent `vllm` logger with propagate=False (see
    # vllm/logger.py:67), so caplog's root-attached handler can't see
    # records from descendants. Import the module first (so vLLM's
    # dictConfig finishes installing its loggers) and only THEN attach
    # caplog.handler to the disk_cache logger directly.
    from vllm.v1.attention.backends.cute_paged import disk_cache

    target_logger = logging.getLogger(
        "vllm.v1.attention.backends.cute_paged.disk_cache"
    )
    target_logger.addHandler(caplog.handler)
    target_logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO)

    # Reset the patched flag so apply_disk_cache_patch installs fresh.
    disk_cache._PATCHED = False

    # Stub CompileCallable + BaseDSL so the test runs without CUTLASS.
    # Real coverage of the patch path is exercised by the cache_smoke
    # diagnostic in Task 4 — this unit test only proves the log lines
    # fire from the right branches in _cached_compile.

    class FakeBaseDSL:
        @staticmethod
        def print_warning(*a, **kw):
            pass

        @staticmethod
        def print_warning_once(*a, **kw):
            pass

    compile_call_count = {"n": 0}

    class FakeCompileCallable:
        @staticmethod
        def _compile(self_obj, func, *args, **kwargs):
            compile_call_count["n"] += 1
            return _SerializableFakeResult()

    monkeypatch.setattr(
        "cutlass.base_dsl.compiler.CompileCallable",
        FakeCompileCallable,
        raising=False,
    )
    monkeypatch.setattr(
        "cutlass.base_dsl.dsl.BaseDSL", FakeBaseDSL, raising=False,
    )

    disk_cache.apply_disk_cache_patch(cache_dir=str(tmp_path))

    # Trigger one MISS (no on-disk artifact yet) and one HIT (after store).
    # The wrapped function expects (self, func, *args, **kwargs); we pass
    # placeholders the cache-key builder accepts.

    from cutlass.base_dsl.compiler import CompileCallable  # noqa: F401 — patched

    # First call: MISS (disk empty). Stores result.
    # Second call: HIT (loads from disk).
    # Cache-key inputs identical across calls.

    def _dummy(): ...
    cc = type("X", (), {"__class__": object})()
    CompileCallable._compile(cc, _dummy)
    CompileCallable._compile(cc, _dummy)

    msgs = [r.message for r in caplog.records if "CuTe disk cache" in r.message]
    miss_lines = [m for m in msgs if "MISS" in m]
    hit_lines = [m for m in msgs if "HIT" in m]
    assert len(miss_lines) >= 1, f"expected at least one MISS line, got: {msgs}"
    assert len(hit_lines) >= 1, f"expected at least one HIT line, got: {msgs}"
