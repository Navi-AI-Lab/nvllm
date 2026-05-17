"""Unit tests for schema-v2 helpers in scripts/bless_cute_full_cache.py.

Covers:
  - copy_cute_kernel_cache_into_staging: deep-copies all files + preserves
    nested structure + handles empty source
  - enumerate_cute_kernel_cache_files: produces sorted, hashed entries with
    correct role + mount_id
  - accept(): writes schema v2 with mounts[] + flat files[] (each file
    carries mount_id)

Does NOT exercise docker / GPU. Phase 1 / Phase 2 docker integration is
covered by the existing bless flow (one full bless cycle = end-to-end test).

Run: .venv/bin/python -m pytest tests/scripts/test_bless_schema_v2.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bless_cute_full_cache import (  # noqa: E402
    BlessConfig,
    MOUNT_CONTAINER_PATH,
    MOUNT_ID_CUTE_CACHE,
    MOUNT_ID_VLLM_CACHE,
    STAGING_CUTE_SUBDIR,
    TrialResult,
    accept,
    copy_cute_kernel_cache_into_staging,
    enumerate_cute_kernel_cache_files,
)


# ---------------------------------------------------------------------------
# copy_cute_kernel_cache_into_staging
# ---------------------------------------------------------------------------


class TestCopyCute:
    def test_flat_files_copied(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.o").write_bytes(b"obj-a" * 10)
        (src / "b.o").write_bytes(b"obj-b" * 10)
        staging = tmp_path / "staging"
        staging.mkdir()

        dest = copy_cute_kernel_cache_into_staging(src, staging)

        assert dest == staging / STAGING_CUTE_SUBDIR
        assert dest.is_dir()
        assert (dest / "a.o").read_bytes() == b"obj-a" * 10
        assert (dest / "b.o").read_bytes() == b"obj-b" * 10

    def test_nested_files_copied_preserving_structure(self, tmp_path):
        src = tmp_path / "src"
        (src / "ns" / "deep").mkdir(parents=True)
        (src / "ns" / "deep" / "kernel.o").write_bytes(b"deep" * 8)
        (src / "ns" / "kernel.bin").write_bytes(b"bin" * 8)
        staging = tmp_path / "staging"
        staging.mkdir()

        dest = copy_cute_kernel_cache_into_staging(src, staging)

        assert (dest / "ns" / "deep" / "kernel.o").read_bytes() == b"deep" * 8
        assert (dest / "ns" / "kernel.bin").read_bytes() == b"bin" * 8

    def test_empty_source_yields_empty_dest_dir(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        staging = tmp_path / "staging"
        staging.mkdir()

        dest = copy_cute_kernel_cache_into_staging(src, staging)

        assert dest.is_dir()
        assert list(dest.iterdir()) == []

    def test_missing_source_yields_empty_dest_dir(self, tmp_path):
        # bless caller should fail-closed when source doesn't exist; the
        # helper itself returns an empty dest dir so callers can detect.
        src = tmp_path / "nope"
        staging = tmp_path / "staging"
        staging.mkdir()

        dest = copy_cute_kernel_cache_into_staging(src, staging)

        assert dest.is_dir()
        assert list(dest.iterdir()) == []

    def test_overwrites_existing_dest(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "new.o").write_bytes(b"new")
        staging = tmp_path / "staging"
        (staging / STAGING_CUTE_SUBDIR / "stale.o").parent.mkdir(parents=True)
        (staging / STAGING_CUTE_SUBDIR / "stale.o").write_bytes(b"stale")

        dest = copy_cute_kernel_cache_into_staging(src, staging)

        assert (dest / "new.o").exists()
        assert not (dest / "stale.o").exists(), (
            "copy_cute_kernel_cache_into_staging must wipe the dest dir before "
            "copy so stale cache files from prior bless don't survive."
        )


# ---------------------------------------------------------------------------
# enumerate_cute_kernel_cache_files
# ---------------------------------------------------------------------------


class TestEnumerateCute:
    def test_entries_sorted_and_complete(self, tmp_path):
        cd = tmp_path / "cute_kernel_cache"
        (cd / "z.o").parent.mkdir(parents=True, exist_ok=True)
        (cd / "z.o").write_bytes(b"z-bytes" * 4)
        (cd / "a.o").write_bytes(b"a-bytes" * 4)

        entries = enumerate_cute_kernel_cache_files(cd)

        assert [e["relative_path"] for e in entries] == ["a.o", "z.o"]
        for e in entries:
            assert e["role"] == "cute_native_object"
            assert e["mount_id"] == MOUNT_ID_CUTE_CACHE
            assert len(e["sha256"]) == 64
            assert e["size_bytes"] > 0

    def test_empty_returns_empty(self, tmp_path):
        cd = tmp_path / "cute_kernel_cache"
        cd.mkdir()
        assert enumerate_cute_kernel_cache_files(cd) == []

    def test_missing_dir_returns_empty(self, tmp_path):
        assert enumerate_cute_kernel_cache_files(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# accept() — manifest schema v2
# ---------------------------------------------------------------------------


def _make_trial(n: int = 1) -> TrialResult:
    return TrialResult(
        trial_n=n, c2_pass=True, cache_reused=True,
        aot_sha256_post="deadbeef" * 8,
        c2_json={}, log_paths={},
    )


def _stage_v2_inputs(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    """Build a staging dir with realistic file roles. Returns
    (staging_dir, resolved_paths, cute_files)."""
    staging = tmp_path / "staging"
    files_to_create = {
        "aot_model": "torch_compile_cache/torch_aot_compile/abc/rank_0_0/model",
        "computation_graph":
            "torch_compile_cache/abc/rank_0_0/backbone/computation_graph.py",
        "cache_key_factors":
            "torch_compile_cache/abc/rank_0_0/backbone/cache_key_factors.json",
        "model_info": "modelinfos/foo.json",
    }
    resolved_paths = {}
    for role, rel in files_to_create.items():
        p = staging / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"role={role}".encode() * 4)
        resolved_paths[role] = rel
    # Stage CuTe cache subdir
    cute_dir = staging / STAGING_CUTE_SUBDIR
    cute_dir.mkdir()
    (cute_dir / "kernel.o").write_bytes(b"kernel-bytes" * 16)
    cute_files = enumerate_cute_kernel_cache_files(cute_dir)
    return staging, resolved_paths, cute_files


class TestAcceptSchemaV2:
    def test_manifest_is_v2_with_mounts_array(self, tmp_path):
        staging, resolved, cute_files = _stage_v2_inputs(tmp_path)
        cfg = BlessConfig(
            config_hash="cafebabe" * 8,
            image_id="sha256:imagex",
            hf_revision="r" * 40,
            rebless=False,
            k_trials=5,
            unsafe_dev_trials=False,
        )
        launch_config = {
            "model_id": "ig1/X",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cute_phase_e_layers": "0,1,2,3,4,5,6,7",
        }

        manifest_path = accept(
            staging_dir=staging,
            blessed_root=tmp_path / "blessed",
            manifest_root=tmp_path / "manifests",
            cfg=cfg,
            resolved_paths=resolved,
            cute_files=cute_files,
            trial_results=[_make_trial(i) for i in range(1, 6)],
            launch_config=launch_config,
        )

        m = json.loads(manifest_path.read_text())
        assert m["schema_version"] == 2
        assert "mounts" in m
        assert "mount" not in m, "v2 must not retain v1 single-mount object"
        ids = {x["id"] for x in m["mounts"]}
        assert ids == {MOUNT_ID_VLLM_CACHE, MOUNT_ID_CUTE_CACHE}

        # Mount container_paths match the contract serve script relies on.
        by_id = {x["id"]: x for x in m["mounts"]}
        assert by_id[MOUNT_ID_VLLM_CACHE]["container_path"] == \
            MOUNT_CONTAINER_PATH[MOUNT_ID_VLLM_CACHE]
        assert by_id[MOUNT_ID_CUTE_CACHE]["container_path"] == \
            MOUNT_CONTAINER_PATH[MOUNT_ID_CUTE_CACHE]
        assert all(x["mode"] == "ro" for x in m["mounts"])

        # Every files[] entry carries mount_id.
        for entry in m["files"]:
            assert entry["mount_id"] in ids, (
                f"file {entry['relative_path']!r} has mount_id="
                f"{entry.get('mount_id')!r} not in mounts[]"
            )

        # At least one CuTe native object was recorded.
        cute_entries = [f for f in m["files"]
                        if f["role"] == "cute_native_object"]
        assert len(cute_entries) >= 1

    def test_cute_files_default_empty_still_writes_v2(self, tmp_path):
        # If a caller forgets cute_files (None), accept() must still emit v2
        # but with no cute_native_object entries — the in-engine gate's
        # cute_required check is what makes that fatal at serve time.
        staging, resolved, _ = _stage_v2_inputs(tmp_path)
        cfg = BlessConfig(
            config_hash="0" * 64, image_id="sha256:i", hf_revision="r" * 40,
            rebless=False, k_trials=5, unsafe_dev_trials=False,
        )
        launch_config = {
            "model_id": "ig1/X", "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cute_phase_e_layers": "0,1,2,3,4,5,6,7",
        }
        manifest_path = accept(
            staging_dir=staging,
            blessed_root=tmp_path / "blessed",
            manifest_root=tmp_path / "manifests",
            cfg=cfg,
            resolved_paths=resolved,
            cute_files=None,
            trial_results=[_make_trial()],
            launch_config=launch_config,
        )
        m = json.loads(manifest_path.read_text())
        assert m["schema_version"] == 2
        assert not any(f["role"] == "cute_native_object" for f in m["files"])
