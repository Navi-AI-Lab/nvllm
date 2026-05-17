# SPDX-License-Identifier: Apache-2.0
"""nvllm blessed-cache import-time gate (Python authoritative verify).

The shell preflight in `scripts/serve-cute-full.sh` validates the manifest
*on the host* before `docker run`. This module duplicates that validation
*inside the container* so we catch:
  - bad mount paths (manifest's container_path missing or wrong)
  - RO/RW mode mismatches (manifest says ro but bind-mounted rw)
  - drift introduced after the host-side preflight (shouldn't happen with
    :ro mounts, but the verify is cheap and the diagnostic is precise)

Two layers:
  * verify_manifest_or_refuse() — file sha + size + mount sanity.
  * install_strict_tripwires()  — monkey-patch StandaloneCompiledArtifacts
                                   .insert to raise on any new artifact
                                   collection (cold compile = hard fail).
A complementary tripwire is added directly in
``vllm/compilation/decorators.py`` at the AOT-load attempt site (so a miss
fails *before* the cold compile path runs at all). See
``apply_blessed_cache_gate`` for the env-var contract.

Env-var contract:
  NVLLM_BLESSED_CACHE_GATE          "1" enables this module (set by serve script)
  NVLLM_BLESSED_CACHE_MANIFEST      container-side path to manifest JSON
  NVLLM_BLESSED_CACHE_CONFIG_HASH   optional crosscheck against manifest.config_hash
  NVLLM_BLESSED_CACHE_STRICT        "1" => install_strict_tripwires (NOT during
                                    bless Phase 1 bootstrap; Phase 2 + prod only)
  NVLLM_BLESSED_CUTE_CACHE_REQUIRED "1" => schema_version<2 manifests refused
                                    (i.e. CuTe .o cache must be in manifest)
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


ENV_GATE_ENABLED = "NVLLM_BLESSED_CACHE_GATE"
ENV_MANIFEST_PATH = "NVLLM_BLESSED_CACHE_MANIFEST"
ENV_CONFIG_HASH = "NVLLM_BLESSED_CACHE_CONFIG_HASH"
ENV_STRICT = "NVLLM_BLESSED_CACHE_STRICT"
ENV_CUTE_REQUIRED = "NVLLM_BLESSED_CUTE_CACHE_REQUIRED"

REQUIRED_ROLES_BASE = {"aot_model", "computation_graph", "cache_key_factors"}
ROLE_CUTE_NATIVE_OBJECT = "cute_native_object"

_STRICT_TRIPWIRE_INSTALLED = False


class BlessedCacheGateError(RuntimeError):
    """Fail-closed signal from the gate. Distinct type so the engine
    boot trace points at the gate, not at a generic RuntimeError."""


# ---------------------------------------------------------------------------
# Manifest verify
# ---------------------------------------------------------------------------


def _read_manifest(manifest_path: str) -> dict[str, Any]:
    if not os.path.isfile(manifest_path):
        raise BlessedCacheGateError(
            f"manifest not found at container path {manifest_path!r}. "
            f"Set {ENV_MANIFEST_PATH} and mount the manifest file (RO) into the "
            "container."
        )
    try:
        return json.loads(Path(manifest_path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise BlessedCacheGateError(
            f"manifest at {manifest_path!r} is unreadable or malformed: {e}"
        ) from e


def _is_writable_mount(path: str) -> bool:
    """True if a tempfile can be created+removed under ``path``.

    More authoritative than ``os.access(W_OK)`` (which only checks DAC
    permissions, not mount flags). EROFS surfaces as PermissionError or
    OSError depending on filesystem.
    """
    try:
        with tempfile.NamedTemporaryFile(
            dir=path, prefix=".nvllm_ro_probe_", delete=True
        ):
            return True
    except OSError:
        return False


def _build_mount_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return ``{mount_id: {container_path, host_path, mode}}``.

    Supports both v1 (single ``mount`` object, implicit ``vllm_cache`` id)
    and v2 (``mounts[]`` array). Validates each container_path exists and
    matches the declared mode.
    """
    schema_version = manifest.get("schema_version")
    lookup: dict[str, dict[str, str]] = {}
    if schema_version == 1:
        m = manifest.get("mount")
        if not isinstance(m, dict):
            raise BlessedCacheGateError("v1 manifest missing required 'mount' object")
        lookup["vllm_cache"] = {
            "container_path": m["container_path"],
            "host_path": m.get("host_path", ""),
            "mode": m.get("mode", "ro"),
        }
    elif schema_version == 2:
        mounts = manifest.get("mounts")
        if not isinstance(mounts, list) or not mounts:
            raise BlessedCacheGateError("v2 manifest 'mounts' must be a non-empty list")
        for m in mounts:
            mid = m.get("id")
            if not mid:
                raise BlessedCacheGateError(f"mounts[] entry missing 'id': {m!r}")
            if mid in lookup:
                raise BlessedCacheGateError(f"duplicate mounts[] id: {mid!r}")
            lookup[mid] = {
                "container_path": m["container_path"],
                "host_path": m.get("host_path", ""),
                "mode": m.get("mode", "ro"),
            }
    else:
        raise BlessedCacheGateError(
            f"unsupported manifest schema_version={schema_version!r}; "
            "expected 1 or 2"
        )
    for mid, m in lookup.items():
        cp = m["container_path"]
        if not os.path.isdir(cp):
            raise BlessedCacheGateError(
                f"mount {mid!r}: container_path {cp!r} is not a directory "
                "(missing -v bind-mount?)"
            )
        if m["mode"] == "ro" and _is_writable_mount(cp):
            raise BlessedCacheGateError(
                f"mount {mid!r}: declared mode=ro but {cp!r} is writable "
                "from inside the container (mount it with :ro)"
            )
    return lookup


def _verify_file(full_path: str, expected_sha: str, expected_size: int) -> None:
    if not os.path.isfile(full_path):
        raise BlessedCacheGateError(f"blessed file missing: {full_path}")
    actual_size = os.path.getsize(full_path)
    if actual_size == 0:
        raise BlessedCacheGateError(f"blessed file is zero-byte: {full_path}")
    if actual_size != expected_size:
        raise BlessedCacheGateError(
            f"size mismatch for {full_path}: "
            f"expected {expected_size}, got {actual_size}"
        )
    h = hashlib.sha256()
    with open(full_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual_sha = h.hexdigest()
    if actual_sha != expected_sha:
        raise BlessedCacheGateError(
            f"sha256 mismatch for {full_path}:\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {actual_sha}"
        )


def verify_manifest_or_refuse(
    manifest_path: str,
    expected_config_hash: str | None = None,
    cute_required: bool = False,
) -> dict[str, Any]:
    """Verify a blessed-cache manifest from inside the container.

    Args:
      manifest_path: container-side path to the manifest JSON.
      expected_config_hash: if non-None, manifest.config_hash must equal this.
      cute_required: if True, manifest.schema_version must be >= 2 AND
                     manifest must include at least one cute_native_object
                     file (i.e. the CuTe .o cache is part of the bless).

    Returns the parsed manifest dict on success; raises BlessedCacheGateError
    on any failure.
    """
    manifest = _read_manifest(manifest_path)

    if expected_config_hash:
        got = manifest.get("config_hash")
        if got != expected_config_hash:
            raise BlessedCacheGateError(
                f"config_hash mismatch: manifest={got!r} expected={expected_config_hash!r}. "
                "The serve script computed a different config_hash than the "
                "manifest claims — env/config drift."
            )

    if manifest.get("validation", {}).get("unsafe_dev_trials"):
        raise BlessedCacheGateError(
            f"manifest {manifest_path} carries validation.unsafe_dev_trials=true. "
            "Production refuses dev-bless manifests; re-run "
            "scripts/bless-cute-full-cache.sh --rebless."
        )

    schema_version = manifest.get("schema_version")
    if cute_required and schema_version is not None and schema_version < 2:
        raise BlessedCacheGateError(
            f"{ENV_CUTE_REQUIRED}=1 but manifest schema_version={schema_version} "
            "does not include CuTe .o cache entries. Re-bless with schema v2."
        )

    mount_lookup = _build_mount_lookup(manifest)

    files = manifest.get("files") or []
    if not files:
        raise BlessedCacheGateError(
            f"manifest {manifest_path} has no files[] entries"
        )

    seen_roles: set[str] = set()
    for entry in files:
        rel = entry["relative_path"]
        role = entry.get("role", "")
        seen_roles.add(role)
        mount_id = entry.get("mount_id", "vllm_cache")
        if mount_id not in mount_lookup:
            raise BlessedCacheGateError(
                f"file {rel}: unknown mount_id={mount_id!r} "
                f"(known: {sorted(mount_lookup)})"
            )
        full = os.path.join(mount_lookup[mount_id]["container_path"], rel)
        _verify_file(full, expected_sha=entry["sha256"],
                     expected_size=entry["size_bytes"])

    missing_base = REQUIRED_ROLES_BASE - seen_roles
    if missing_base:
        raise BlessedCacheGateError(
            f"manifest {manifest_path} missing required roles: "
            f"{sorted(missing_base)}"
        )
    if cute_required and ROLE_CUTE_NATIVE_OBJECT not in seen_roles:
        raise BlessedCacheGateError(
            f"{ENV_CUTE_REQUIRED}=1 but manifest has no files with "
            f"role={ROLE_CUTE_NATIVE_OBJECT!r}"
        )

    logger.info(
        "[blessed-cache gate] verify PASS: schema_v=%s files=%d mounts=%s",
        schema_version, len(files), sorted(mount_lookup),
    )
    return manifest


# ---------------------------------------------------------------------------
# Strict-mode tripwire
# ---------------------------------------------------------------------------


def install_strict_tripwires() -> None:
    """Monkey-patch StandaloneCompiledArtifacts.insert to raise on any new
    artifact insertion. Catches cold-compile attempts that bypass the
    AOT-load-miss guard in vllm/compilation/decorators.py.

    Idempotent — re-applying is a no-op (the gate may be invoked twice if
    the cute_paged backend module is imported more than once).
    """
    global _STRICT_TRIPWIRE_INSTALLED
    if _STRICT_TRIPWIRE_INSTALLED:
        return
    from vllm.compilation import caching as caching_mod

    original_insert = caching_mod.StandaloneCompiledArtifacts.insert

    @functools.wraps(original_insert)
    def _strict_insert(
        self: Any, submod_name: str, shape: str, entry: bytes
    ) -> None:
        sha = hashlib.sha256(entry).hexdigest()[:16]
        raise BlessedCacheGateError(
            f"{ENV_STRICT}=1: attempted to insert a new standalone compile "
            f"artifact submod={submod_name!r} shape={shape!r} "
            f"bytes={len(entry)} sha256={sha}. "
            "Cold compile under blessed-cache strict mode is fail-closed. "
            "Likely cause: AOT load missed and torch.compile fell through. "
            "Re-bless or investigate AOT cache contents."
        )

    caching_mod.StandaloneCompiledArtifacts.insert = _strict_insert
    _STRICT_TRIPWIRE_INSTALLED = True
    logger.info("[blessed-cache gate] strict tripwire installed on "
                "StandaloneCompiledArtifacts.insert")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_blessed_cache_gate() -> None:
    """Module-init entry point. Reads env, verifies manifest, optionally
    installs strict tripwires. Idempotent. Raises BlessedCacheGateError on
    any failure (engine refuses to start)."""
    if os.environ.get(ENV_GATE_ENABLED, "0") != "1":
        return

    manifest_path = os.environ.get(ENV_MANIFEST_PATH, "").strip()
    if not manifest_path:
        raise BlessedCacheGateError(
            f"{ENV_GATE_ENABLED}=1 but {ENV_MANIFEST_PATH} is unset/empty. "
            "Set the container-side path to the blessed manifest JSON."
        )

    expected_hash = (os.environ.get(ENV_CONFIG_HASH, "").strip() or None)
    strict = os.environ.get(ENV_STRICT, "0") == "1"
    # v1 manifests cannot prove the CuTe .o cache is pinned. Refuse them
    # whenever the runtime promises a strict / CuTe-on path:
    #   - NVLLM_BLESSED_CACHE_STRICT=1: production cold-compile-is-fatal path
    #   - NVLLM_BLESSED_CUTE_CACHE_REQUIRED=1: operator-asserted requirement
    #   - CUTE_PHASE_E_FUSION=1: β-coop Phase-E uber-kernel uses CuTe .o cache
    cute_required = (
        strict
        or os.environ.get(ENV_CUTE_REQUIRED, "0") == "1"
        or os.environ.get("CUTE_PHASE_E_FUSION", "0") == "1"
    )

    manifest = verify_manifest_or_refuse(
        manifest_path=manifest_path,
        expected_config_hash=expected_hash,
        cute_required=cute_required,
    )

    if strict:
        install_strict_tripwires()

    logger.info(
        "[blessed-cache gate] PASS manifest=%s config_hash=%s strict=%s "
        "cute_required=%s",
        manifest_path,
        manifest.get("config_hash", "?")[:7] + "…",
        strict,
        cute_required,
    )
