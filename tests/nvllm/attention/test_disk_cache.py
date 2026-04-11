"""Tests for CuTe DSL compilation disk cache."""
import hashlib
import os
import tempfile

import pytest


class TestDiskCacheKey:
    """Cache key generation produces stable, unique keys."""

    def test_same_inputs_same_key(self):
        from vllm.v1.attention.backends.cute_paged.disk_cache import (
            build_cache_key,
        )
        key1 = build_cache_key("def f(): pass", {"shape": (4, 128)}, "4.4.0")
        key2 = build_cache_key("def f(): pass", {"shape": (4, 128)}, "4.4.0")
        assert key1 == key2

    def test_different_source_different_key(self):
        from vllm.v1.attention.backends.cute_paged.disk_cache import (
            build_cache_key,
        )
        key1 = build_cache_key("def f(): pass", {"shape": (4, 128)}, "4.4.0")
        key2 = build_cache_key("def g(): pass", {"shape": (4, 128)}, "4.4.0")
        assert key1 != key2

    def test_different_args_different_key(self):
        from vllm.v1.attention.backends.cute_paged.disk_cache import (
            build_cache_key,
        )
        key1 = build_cache_key("def f(): pass", {"shape": (4, 128)}, "4.4.0")
        key2 = build_cache_key("def f(): pass", {"shape": (8, 128)}, "4.4.0")
        assert key1 != key2

    def test_different_toolchain_different_key(self):
        from vllm.v1.attention.backends.cute_paged.disk_cache import (
            build_cache_key,
        )
        key1 = build_cache_key("def f(): pass", {"shape": (4, 128)}, "4.4.0")
        key2 = build_cache_key("def f(): pass", {"shape": (4, 128)}, "4.5.0")
        assert key1 != key2


class TestDiskCacheStorage:
    """Cache stores and loads compiled objects from disk."""

    def test_store_and_load(self):
        from vllm.v1.attention.backends.cute_paged.disk_cache import (
            load_from_disk,
            store_to_disk,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            key = "abc123"
            obj = {"compiled": True, "data": b"fake_cubin"}
            store_to_disk(tmpdir, key, obj)
            loaded = load_from_disk(tmpdir, key)
            assert loaded == obj

    def test_load_missing_returns_none(self):
        from vllm.v1.attention.backends.cute_paged.disk_cache import (
            load_from_disk,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_from_disk(tmpdir, "nonexistent") is None

    def test_corrupted_file_returns_none(self):
        from vllm.v1.attention.backends.cute_paged.disk_cache import (
            load_from_disk,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write garbage to the expected path
            key = "abc123"
            subdir = os.path.join(tmpdir, key[:2])
            os.makedirs(subdir, exist_ok=True)
            path = os.path.join(subdir, key)
            with open(path, "wb") as f:
                f.write(b"corrupted garbage data")
            assert load_from_disk(tmpdir, key) is None


class TestDiskCachePatch:
    """The monkey-patch applies safely and falls back on failure."""

    def test_patch_applies_without_cutlass(self):
        """When CUTLASS is not importable, patch logs warning and returns."""
        from vllm.v1.attention.backends.cute_paged.disk_cache import (
            apply_disk_cache_patch,
        )
        # Should not raise even if cutlass is not installed
        # (it catches ImportError internally)
        apply_disk_cache_patch(cache_dir="/tmp/nonexistent", enabled=False)

    def test_cache_key_is_sha256(self):
        from vllm.v1.attention.backends.cute_paged.disk_cache import (
            build_cache_key,
        )
        key = build_cache_key("def f(): pass", {"shape": (4,)}, "4.4.0")
        assert len(key) == 64  # SHA256 hex digest length
        int(key, 16)  # Must be valid hex
