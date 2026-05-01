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


class TestAcceptReject:
    def setup_method(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

    def teardown_method(self):
        sys.path.pop(0)

    def _make_staging(self, tmp_path: Path) -> tuple[Path, dict[str, str]]:
        staging = tmp_path / "staging" / "abcd"
        for rel in (
            "torch_compile_cache/torch_aot_compile/9a55/rank_0_0/model",
            "torch_compile_cache/b690/rank_0_0/backbone/computation_graph.py",
            "torch_compile_cache/b690/rank_0_0/backbone/cache_key_factors.json",
            "modelinfos/vllm-X.json",
        ):
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            (staging / rel).write_text("x")
        resolved = {
            "aot_model": "torch_compile_cache/torch_aot_compile/9a55/rank_0_0/model",
            "computation_graph": "torch_compile_cache/b690/rank_0_0/backbone/computation_graph.py",
            "cache_key_factors": "torch_compile_cache/b690/rank_0_0/backbone/cache_key_factors.json",
            "model_info": "modelinfos/vllm-X.json",
        }
        return staging, resolved

    def _bless_config(self):
        from bless_cute_full_cache import BlessConfig
        return BlessConfig(
            config_hash="a" * 64, image_id="sha256:img",
            hf_revision="b" * 40, rebless=False, k_trials=5,
            unsafe_dev_trials=False,
        )

    def test_accept_writes_manifest_and_moves_cache(self, tmp_path):
        from bless_cute_full_cache import accept, TrialResult
        staging, resolved = self._make_staging(tmp_path)
        blessed_root = tmp_path / "blessed"
        manifest_root = tmp_path / "manifests"
        blessed_root.mkdir(); manifest_root.mkdir()
        cfg = self._bless_config()
        results = [
            TrialResult(trial_n=i, c2_pass=True, cache_reused=True,
                         aot_sha256_post="sha", c2_json={}, log_paths={})
            for i in range(1, 6)
        ]
        manifest_path = accept(
            staging_dir=staging,
            blessed_root=blessed_root,
            manifest_root=manifest_root,
            cfg=cfg,
            resolved_paths=resolved,
            trial_results=results,
            launch_config={
                "model_id": "ig1/Qwen3.5-27B-NVFP4",
                "kv_cache_dtype": "fp8_e4m3",
                "attention_backend": "CUTE_PAGED",
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": [1],
                "max_num_seqs": 1, "max_model_len": 16384,
                "max_num_batched_tokens": 65536,
                "cute_phase_e_fusion": 1,
                "cute_phase_e_layers": "0,1,2,3,4,5,6,7",
                "cute_phase_e_fallback_raise": 1,
                "cute_full_graph_probe": 0, "cute_wo_reset_log": 0,
                "cute_dispatch_audit": 0,
                "cute_mlp_fusion": 1, "cute_attn_fusion": 1,
            },
        )
        assert manifest_path.exists()
        m = json.loads(manifest_path.read_text())
        assert m["config_hash"] == cfg.config_hash
        assert m["validation"]["trials"] == 5
        assert m["validation"]["trials_passed"] == 5
        assert m["validation"]["unsafe_dev_trials"] is False
        assert len(m["files"]) == 4
        # Atomic move: staging gone, blessed populated.
        assert not staging.exists()
        assert (blessed_root / cfg.config_hash).exists()

    def test_reject_moves_staging_to_evidence(self, tmp_path):
        from bless_cute_full_cache import reject, TrialResult
        staging, _ = self._make_staging(tmp_path)
        evidence_root = tmp_path / "evidence"
        evidence_root.mkdir()
        results = [
            TrialResult(trial_n=1, c2_pass=False, cache_reused=True,
                         aot_sha256_post="sha", c2_json={}, log_paths={}),
            TrialResult(trial_n=2, c2_pass=True, cache_reused=False,
                         aot_sha256_post="sha", c2_json={}, log_paths={}),
        ]
        evidence_dir = reject(
            staging_dir=staging,
            evidence_root=evidence_root,
            cfg=self._bless_config(),
            trial_results=results,
        )
        assert not staging.exists()
        assert evidence_dir.exists()
        assert (evidence_dir / "bless_failure.json").exists()


class TestReadmePopulator:
    def test_regenerate_readme_table_with_one_manifest(self, tmp_path):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            from bless_cute_full_cache import regenerate_readme_table
            (tmp_path / "README.md").write_text(
                "before\n<!-- BEGIN AUTO-GENERATED TABLE -->\nold\n"
                "<!-- END AUTO-GENERATED TABLE -->\nafter\n"
            )
            (tmp_path / "m.json").write_text(json.dumps({
                "config_hash": "abc",
                "blessed_at": "2026-05-01T00:00:00Z",
                "blessed_image_id": "sha256:d3ddffea3c1234567890abcdef",
                "config": {"model_id": "ig1/M",
                           "cudagraph_mode": "FULL_AND_PIECEWISE",
                           "cute_phase_e_layers": "0,1,2,3,4,5,6,7"},
                "validation": {"unsafe_dev_trials": False},
            }))
            regenerate_readme_table(tmp_path)
            out = (tmp_path / "README.md").read_text()
            assert "ig1/M" in out
            assert "FULL_AND_PIECEWISE" in out
            assert "old" not in out
            assert "before" in out and "after" in out
        finally:
            sys.path.pop(0)
