# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the blessed-cache import-time gate.

Covers:
  - manifest absent / malformed / wrong-shape
  - file SHA + size verification (hit, drift, missing, zero-byte)
  - mount mode verification (declared :ro but writable -> refuse)
  - schema v1 + v2 (mounts[] + flat files[] with mount_id)
  - unsafe_dev_trials refusal
  - config_hash crosscheck
  - cute_required gate (v1 refused, v2 with no cute_native_object refused)
  - strict-mode tripwire on StandaloneCompiledArtifacts.insert

Does not exercise the in-engine AOT-load-miss raise (decorators.py:514) —
that path requires real torch.compile state. Covered by the bless flow
end-to-end test.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from vllm.v1.attention.backends.cute_paged import blessed_cache_gate as gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_file(p: Path, data: bytes) -> dict[str, object]:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {
        "relative_path": str(p.name),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _v1_manifest(tmp_path: Path) -> tuple[Path, dict]:
    cache_root = tmp_path / "vllm_cache"
    cache_root.mkdir()
    files = []
    for role, name in [
        ("aot_model", "torch_aot_compile/abc/rank_0_0/model"),
        ("computation_graph", "torch_compile_cache/x/rank_0_0/backbone/computation_graph.py"),
        ("cache_key_factors", "torch_compile_cache/x/rank_0_0/backbone/cache_key_factors.json"),
    ]:
        full = cache_root / name
        entry = _write_file(full, f"role={role}".encode() * 4)
        entry["relative_path"] = name
        entry["role"] = role
        files.append(entry)
    m = {
        "schema_version": 1,
        "config_hash": "deadbeef" * 8,
        "blessed_image_id": "sha256:dummy",
        "validation": {"unsafe_dev_trials": False},
        "mount": {
            "host_path": str(cache_root),
            "container_path": str(cache_root),
            "mode": "ro",
        },
        "files": files,
    }
    path = tmp_path / "manifest_v1.json"
    path.write_text(json.dumps(m))
    return path, m


def _v2_manifest(
    tmp_path: Path, include_cute: bool = True
) -> tuple[Path, dict]:
    vllm_cache = tmp_path / "vllm_cache"
    cute_cache = tmp_path / "cute_kernel_cache"
    vllm_cache.mkdir()
    cute_cache.mkdir()
    files = []
    for role, name in [
        ("aot_model", "torch_aot_compile/abc/rank_0_0/model"),
        ("computation_graph", "torch_compile_cache/x/rank_0_0/backbone/computation_graph.py"),
        ("cache_key_factors", "torch_compile_cache/x/rank_0_0/backbone/cache_key_factors.json"),
    ]:
        full = vllm_cache / name
        entry = _write_file(full, f"role={role}".encode() * 4)
        entry["relative_path"] = name
        entry["role"] = role
        entry["mount_id"] = "vllm_cache"
        files.append(entry)
    if include_cute:
        full = cute_cache / "kernel_abc.o"
        entry = _write_file(full, b"\x00\x01\x02fake cute .o" * 64)
        entry["relative_path"] = "kernel_abc.o"
        entry["role"] = "cute_native_object"
        entry["mount_id"] = "cute_kernel_cache"
        files.append(entry)
    m = {
        "schema_version": 2,
        "config_hash": "cafef00d" * 8,
        "blessed_image_id": "sha256:dummy",
        "validation": {"unsafe_dev_trials": False},
        "mounts": [
            {
                "id": "vllm_cache",
                "host_path": str(vllm_cache),
                "container_path": str(vllm_cache),
                "mode": "ro",
            },
            {
                "id": "cute_kernel_cache",
                "host_path": str(cute_cache),
                "container_path": str(cute_cache),
                "mode": "ro",
            },
        ],
        "files": files,
    }
    path = tmp_path / "manifest_v2.json"
    path.write_text(json.dumps(m))
    return path, m


# ---------------------------------------------------------------------------
# verify_manifest_or_refuse
# ---------------------------------------------------------------------------


class TestVerifyV1:
    def test_happy_path(self, tmp_path):
        # v1 mount path is the underlying tempdir which is writable —
        # use mode!=ro so the writability probe doesn't trip.
        path, m = _v1_manifest(tmp_path)
        m["mount"]["mode"] = "rw"
        path.write_text(json.dumps(m))
        manifest = gate.verify_manifest_or_refuse(str(path))
        assert manifest["schema_version"] == 1

    def test_missing_manifest_refuses(self, tmp_path):
        with pytest.raises(gate.BlessedCacheGateError, match="not found"):
            gate.verify_manifest_or_refuse(str(tmp_path / "nope.json"))

    def test_malformed_json_refuses(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json {")
        with pytest.raises(gate.BlessedCacheGateError, match="unreadable or malformed"):
            gate.verify_manifest_or_refuse(str(p))

    def test_sha_mismatch_refuses(self, tmp_path):
        path, m = _v1_manifest(tmp_path)
        m["mount"]["mode"] = "rw"
        m["files"][0]["sha256"] = "00" * 32
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="sha256 mismatch"):
            gate.verify_manifest_or_refuse(str(path))

    def test_size_mismatch_refuses(self, tmp_path):
        path, m = _v1_manifest(tmp_path)
        m["mount"]["mode"] = "rw"
        m["files"][0]["size_bytes"] = 1
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="size mismatch"):
            gate.verify_manifest_or_refuse(str(path))

    def test_missing_file_refuses(self, tmp_path):
        path, m = _v1_manifest(tmp_path)
        m["mount"]["mode"] = "rw"
        # delete the actual file but keep the manifest entry
        (Path(m["mount"]["container_path"]) /
         m["files"][0]["relative_path"]).unlink()
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="missing"):
            gate.verify_manifest_or_refuse(str(path))

    def test_zero_byte_refuses(self, tmp_path):
        path, m = _v1_manifest(tmp_path)
        m["mount"]["mode"] = "rw"
        target = Path(m["mount"]["container_path"]) / m["files"][0]["relative_path"]
        target.write_bytes(b"")
        # Update sha+size so we hit the zero-byte branch (not the size branch)
        m["files"][0]["sha256"] = hashlib.sha256(b"").hexdigest()
        m["files"][0]["size_bytes"] = 0
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="zero-byte"):
            gate.verify_manifest_or_refuse(str(path))

    def test_unsafe_dev_trials_refuses(self, tmp_path):
        path, m = _v1_manifest(tmp_path)
        m["mount"]["mode"] = "rw"
        m["validation"]["unsafe_dev_trials"] = True
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="unsafe_dev_trials"):
            gate.verify_manifest_or_refuse(str(path))

    def test_config_hash_crosscheck(self, tmp_path):
        path, m = _v1_manifest(tmp_path)
        m["mount"]["mode"] = "rw"
        path.write_text(json.dumps(m))
        # match: PASS
        gate.verify_manifest_or_refuse(str(path), expected_config_hash="deadbeef" * 8)
        # mismatch: REFUSE
        with pytest.raises(gate.BlessedCacheGateError, match="config_hash mismatch"):
            gate.verify_manifest_or_refuse(str(path), expected_config_hash="cafe" * 16)

    def test_ro_declared_but_writable_refuses(self, tmp_path):
        # mode=ro + the underlying tempdir is writable -> the gate must refuse
        path, _ = _v1_manifest(tmp_path)
        with pytest.raises(gate.BlessedCacheGateError, match="declared mode=ro but"):
            gate.verify_manifest_or_refuse(str(path))

    def test_missing_required_role_refuses(self, tmp_path):
        path, m = _v1_manifest(tmp_path)
        m["mount"]["mode"] = "rw"
        m["files"] = [f for f in m["files"] if f["role"] != "aot_model"]
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="missing required roles"):
            gate.verify_manifest_or_refuse(str(path))

    def test_cute_required_refuses_v1(self, tmp_path):
        path, m = _v1_manifest(tmp_path)
        m["mount"]["mode"] = "rw"
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="schema_version=1"):
            gate.verify_manifest_or_refuse(str(path), cute_required=True)


class TestVerifyV2:
    def test_happy_path_with_cute(self, tmp_path):
        path, m = _v2_manifest(tmp_path, include_cute=True)
        for mount in m["mounts"]:
            mount["mode"] = "rw"
        path.write_text(json.dumps(m))
        manifest = gate.verify_manifest_or_refuse(str(path), cute_required=True)
        assert manifest["schema_version"] == 2

    def test_v2_without_cute_refused_when_required(self, tmp_path):
        path, m = _v2_manifest(tmp_path, include_cute=False)
        for mount in m["mounts"]:
            mount["mode"] = "rw"
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="no files with role"):
            gate.verify_manifest_or_refuse(str(path), cute_required=True)

    def test_v2_unknown_mount_id_refuses(self, tmp_path):
        path, m = _v2_manifest(tmp_path)
        for mount in m["mounts"]:
            mount["mode"] = "rw"
        m["files"][0]["mount_id"] = "ghost_cache"
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="unknown mount_id"):
            gate.verify_manifest_or_refuse(str(path))

    def test_v2_duplicate_mount_id_refuses(self, tmp_path):
        path, m = _v2_manifest(tmp_path)
        for mount in m["mounts"]:
            mount["mode"] = "rw"
        m["mounts"].append(dict(m["mounts"][0]))  # duplicate id
        path.write_text(json.dumps(m))
        with pytest.raises(gate.BlessedCacheGateError, match="duplicate mounts"):
            gate.verify_manifest_or_refuse(str(path))


class TestUnsupportedSchema:
    def test_v0_refused(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"schema_version": 0}))
        with pytest.raises(gate.BlessedCacheGateError, match="unsupported manifest schema_version"):
            gate.verify_manifest_or_refuse(str(p))

    def test_v999_refused(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"schema_version": 999}))
        with pytest.raises(gate.BlessedCacheGateError, match="unsupported manifest schema_version"):
            gate.verify_manifest_or_refuse(str(p))


# ---------------------------------------------------------------------------
# apply_blessed_cache_gate (env-driven)
# ---------------------------------------------------------------------------


class TestApplyGateEnv:
    def test_disabled_is_noop(self, monkeypatch):
        monkeypatch.delenv(gate.ENV_GATE_ENABLED, raising=False)
        gate.apply_blessed_cache_gate()  # must not raise

    def test_enabled_without_manifest_path_refuses(self, monkeypatch):
        monkeypatch.setenv(gate.ENV_GATE_ENABLED, "1")
        monkeypatch.delenv(gate.ENV_MANIFEST_PATH, raising=False)
        with pytest.raises(gate.BlessedCacheGateError, match="unset"):
            gate.apply_blessed_cache_gate()

    def test_enabled_with_empty_manifest_path_refuses(self, monkeypatch):
        monkeypatch.setenv(gate.ENV_GATE_ENABLED, "1")
        monkeypatch.setenv(gate.ENV_MANIFEST_PATH, "   ")
        with pytest.raises(gate.BlessedCacheGateError, match="unset"):
            gate.apply_blessed_cache_gate()

    def test_enabled_e2e_v2(self, monkeypatch, tmp_path):
        path, m = _v2_manifest(tmp_path)
        for mount in m["mounts"]:
            mount["mode"] = "rw"
        path.write_text(json.dumps(m))
        monkeypatch.setenv(gate.ENV_GATE_ENABLED, "1")
        monkeypatch.setenv(gate.ENV_MANIFEST_PATH, str(path))
        monkeypatch.setenv(gate.ENV_CONFIG_HASH, "cafef00d" * 8)
        monkeypatch.setenv(gate.ENV_CUTE_REQUIRED, "1")
        # strict=0 in this test (tripwire test is below)
        gate.apply_blessed_cache_gate()  # must not raise


# ---------------------------------------------------------------------------
# Strict tripwire
# ---------------------------------------------------------------------------


class TestStrictTripwire:
    def setup_method(self):
        # Ensure clean state between tests (caching module-level patch)
        from vllm.compilation import caching as caching_mod
        # Capture original_insert for restoration
        self._caching_mod = caching_mod
        self._orig_insert = caching_mod.StandaloneCompiledArtifacts.insert
        gate._STRICT_TRIPWIRE_INSTALLED = False

    def teardown_method(self):
        self._caching_mod.StandaloneCompiledArtifacts.insert = self._orig_insert
        gate._STRICT_TRIPWIRE_INSTALLED = False

    def test_insert_raises_after_install(self):
        from vllm.compilation.caching import StandaloneCompiledArtifacts

        store = StandaloneCompiledArtifacts()
        # Pre-install: insert works.
        store.insert("submod_0", "1", b"baseline-bytes")

        gate.install_strict_tripwires()
        with pytest.raises(gate.BlessedCacheGateError, match="[Cc]old compile under blessed-cache strict"):
            store.insert("submod_1", "1", b"new-bytes")

    def test_install_is_idempotent(self):
        gate.install_strict_tripwires()
        gate.install_strict_tripwires()  # must not double-wrap
        from vllm.compilation.caching import StandaloneCompiledArtifacts
        store = StandaloneCompiledArtifacts()
        # The error message should still match the strict-tripwire one,
        # not be double-wrapped.
        with pytest.raises(gate.BlessedCacheGateError) as ei:
            store.insert("submod_X", "1", b"abc")
        # Only one frame in the chain from the gate
        assert "old compile under blessed-cache strict" in str(ei.value).lower()
