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


class TestPhase1:
    def setup_method(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

    def teardown_method(self):
        sys.path.pop(0)

    def test_build_phase1_docker_args(self, tmp_path):
        from bless_cute_full_cache import build_phase1_docker_args
        staging = tmp_path / "staging"
        staging.mkdir()
        args = build_phase1_docker_args(
            container_name="nvllm",
            image="nvllm:gb10",
            hf_cache=Path("/home/u/.cache/huggingface"),
            flashinfer_cache=Path("/home/u/.cache/flashinfer"),
            cute_compile_host_cache=Path("/tmp/nvllm-cute-cache"),
            staging_dir=staging,
            model_id="ig1/Qwen3.5-27B-NVFP4",
            kv_cache_dtype="fp8_e4m3",
            attention_backend="CUTE_PAGED",
            max_model_len=16384,
            max_num_seqs=1,
            max_num_batched_tokens=65536,
            cute_phase_e_layers="0,1,2,3,4,5,6,7",
        )
        # Should be the rw mount, no :ro suffix:
        assert f"{staging}:/root/.cache/vllm" in " ".join(args)
        assert ":ro" not in " ".join(args)
        # Probes off:
        assert "CUTE_FULL_GRAPH_PROBE=0" in " ".join(args)
        assert "CUTE_WO_RESET_LOG=0" in " ".join(args)
        assert "CUTE_DISPATCH_AUDIT=0" in " ".join(args)
        # β-coop on:
        assert "CUTE_PHASE_E_FUSION=1" in " ".join(args)
        assert "CUTE_PHASE_E_FALLBACK_RAISE=1" in " ".join(args)
        # Layer set:
        assert "CUTE_PHASE_E_LAYERS=0,1,2,3,4,5,6,7" in " ".join(args)
        # Cudagraph mode:
        assert "FULL_AND_PIECEWISE" in " ".join(args)

    def test_expected_files_returns_4_paths(self):
        from bless_cute_full_cache import expected_cache_files
        files = expected_cache_files()
        roles = {f["role"] for f in files}
        assert roles == {"aot_model", "computation_graph", "cache_key_factors",
                         "model_info"}
        assert len(files) == 4


class TestPhase2:
    def setup_method(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

    def teardown_method(self):
        sys.path.pop(0)

    def test_build_phase2_docker_args_uses_ro(self, tmp_path):
        from bless_cute_full_cache import build_phase2_docker_args
        staging = tmp_path / "staging"
        staging.mkdir()
        args = build_phase2_docker_args(
            container_name="nvllm", image="nvllm:gb10",
            hf_cache=Path("/h/.cache/hf"),
            flashinfer_cache=Path("/h/.cache/fi"),
            cute_compile_host_cache=Path("/tmp/cc"),
            staging_dir=staging, model_id="ig1/Qwen3.5-27B-NVFP4",
            kv_cache_dtype="fp8_e4m3", attention_backend="CUTE_PAGED",
            max_model_len=16384, max_num_seqs=1, max_num_batched_tokens=65536,
            cute_phase_e_layers="0,1,2,3,4,5,6,7",
        )
        joined = " ".join(args)
        assert f"{staging}:/root/.cache/vllm:ro" in joined

    def test_classify_cache_reuse_pass(self):
        from bless_cute_full_cache import classify_cache_reuse
        path = "/root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a17/rank_0_0/model"
        log = (
            f"Directly load AOT compilation from path {path}\n"
            "model loaded\n"
            "ready"
        )
        ok, reasons = classify_cache_reuse(
            container_log=log,
            sha_pre="abc",
            sha_post="abc",
            expected_aot_path=path,
        )
        assert ok is True
        assert reasons == []

    def test_classify_cache_reuse_fail_no_load_marker(self):
        from bless_cute_full_cache import classify_cache_reuse
        path = "/root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a17/rank_0_0/model"
        ok, reasons = classify_cache_reuse(
            container_log="some other text",
            sha_pre="abc", sha_post="abc",
            expected_aot_path=path,
        )
        assert ok is False
        assert any("AOT load marker absent" in r for r in reasons)

    def test_classify_cache_reuse_fail_marker_path_mismatch(self):
        from bless_cute_full_cache import classify_cache_reuse
        # Marker present but pointing at a different artifact path.
        log = (
            "Directly load AOT compilation from path "
            "/root/.cache/vllm/torch_compile_cache/torch_aot_compile/DIFFERENT/rank_0_0/model"
        )
        path = "/root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a17/rank_0_0/model"
        ok, reasons = classify_cache_reuse(
            container_log=log,
            sha_pre="abc", sha_post="abc",
            expected_aot_path=path,
        )
        assert ok is False
        assert any("path mismatch" in r for r in reasons)

    def test_classify_cache_reuse_fail_saved_aot_present(self):
        from bless_cute_full_cache import classify_cache_reuse
        path = "/root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a17/rank_0_0/model"
        log = (
            f"Directly load AOT compilation from path {path}\n"
            "saved AOT compiled function to /path"
        )
        ok, reasons = classify_cache_reuse(
            container_log=log,
            sha_pre="abc", sha_post="abc",
            expected_aot_path=path,
        )
        assert ok is False
        assert any("saved AOT" in r for r in reasons)

    def test_classify_cache_reuse_fail_sha_drift(self):
        from bless_cute_full_cache import classify_cache_reuse
        path = "/root/.cache/vllm/torch_compile_cache/torch_aot_compile/9a5549f23a17/rank_0_0/model"
        log = f"Directly load AOT compilation from path {path}"
        ok, reasons = classify_cache_reuse(
            container_log=log,
            sha_pre="abc", sha_post="def",
            expected_aot_path=path,
        )
        assert ok is False
        assert any("sha" in r.lower() for r in reasons)

    def test_parse_c2_json_pass(self, tmp_path):
        from bless_cute_full_cache import parse_c2_json
        p = tmp_path / "c2.json"
        p.write_text(json.dumps({
            "same_prompt_pass": True, "cross_prompt_pass": True,
            "same_prompt_unique_count": 1, "overall_pass": True,
        }))
        c2_pass, summary = parse_c2_json(p)
        assert c2_pass is True
        assert summary["same_prompt_unique_count"] == 1

    def test_parse_c2_json_fail_unique_gt_1(self, tmp_path):
        from bless_cute_full_cache import parse_c2_json
        p = tmp_path / "c2.json"
        p.write_text(json.dumps({
            "same_prompt_pass": False, "cross_prompt_pass": True,
            "same_prompt_unique_count": 3, "overall_pass": False,
        }))
        c2_pass, _ = parse_c2_json(p)
        assert c2_pass is False
