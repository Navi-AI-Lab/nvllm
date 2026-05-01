"""Unit tests for scripts/bless_cute_full_cache.py orchestrator.

Run: .venv/bin/python -m pytest tests/scripts/test_bless_orchestrator.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = REPO_ROOT / "scripts/bless_cute_full_cache.py"


def _run_orch(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(REPO_ROOT / ".venv/bin/python"), str(ORCHESTRATOR), *args],
        capture_output=True,
        text=True,
    )


class TestCLI:
    def test_help_lists_required_args(self):
        r = _run_orch("--help")
        assert r.returncode == 0, r.stderr
        for arg in ("--config-hash", "--image-id", "--hf-revision",
                    "--rebless", "--unsafe-trials"):
            assert arg in r.stdout, f"{arg} missing from --help"

    def test_missing_required_args_exit_nonzero(self):
        r = _run_orch()  # no args
        assert r.returncode != 0
        assert "--config-hash" in r.stderr or "required" in r.stderr.lower()


class TestDataclasses:
    """Import-only sanity for the public dataclasses."""

    def test_trial_result_importable(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            from bless_cute_full_cache import TrialResult  # noqa
            tr = TrialResult(
                trial_n=1, c2_pass=True, cache_reused=True,
                aot_sha256_post="abc", c2_json={}, log_paths={},
            )
            assert tr.trial_n == 1
        finally:
            sys.path.pop(0)
