# docs/research/gemm_sweep/test_gen_winners_header.py
"""Unit tests for gen_winners_header.py — pins the 12 expected (shape,
bucket, idx) triples from the spec
(docs/superpowers/specs/2026-04-21-gemm-winners-table-design.md §3 table).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEP_DIR = REPO_ROOT / "benchmarks/nvllm/traces/gemm_sweep_sm120/2026-04-21-qwen35-27b"
SHORTLIST_HEADER = REPO_ROOT / "csrc/libtorch_stable/quantization/fp4/nvfp4_shortlist_configs.hpp"
WINNERS_JSON = SWEEP_DIR / "winners.json"
WINNERS_HEADER = REPO_ROOT / "csrc/libtorch_stable/quantization/fp4/nvfp4_winners_table.hpp"
SCRIPT = REPO_ROOT / "docs/research/gemm_sweep/gen_winners_header.py"

# Expected winners from spec §3 (rank-1 of shortlist.json top-3 per bucket).
# shape -> bucket -> (cfg_name, idx_in_shortlist_configs_hpp)
EXPECTED = {
    "qkv_proj": {
        "16-32":   ("Cfg_128x256x128_Auto_Pers",       6),
        "64-128":  ("smoke_M256",                      11),
        "192-256": ("Cfg_128x128x128_TmaWSPing_Pers",  1),
    },
    "o_proj": {
        "16-32":   ("Cfg_128x256x128_Auto_Pers",       6),
        "64-128":  ("Cfg_128x128x256_TmaWSPing_Pers",  5),
        "192-256": ("Cfg_256x128x128_TmaWSCoop_Pers",  10),
    },
    "gate_up_proj": {
        "16-32":   ("Cfg_128x128x256_Auto_Pers",       2),
        "64-128":  ("Cfg_128x128x256_TmaWSCoop_Pers",  3),
        "192-256": ("Cfg_128x128x256_Auto_Pers",       2),
    },
    "down_proj": {
        "16-32":   ("Cfg_128x128x256_Auto_Pers",       2),
        "64-128":  ("Cfg_128x128x256_Auto_Pers",       2),
        "192-256": ("Cfg_128x128x256_Auto_Pers",       2),
    },
}
# Qwen3.5-27B shapes from config.json: (N, K)
EXPECTED_NK = {
    "qkv_proj":     (8192,  5120),
    "o_proj":       (5120,  6144),
    "gate_up_proj": (34816, 5120),
    "down_proj":    (5120,  17408),
}


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False, capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_script_exists():
    assert SCRIPT.exists(), f"Codegen script missing: {SCRIPT}"


def test_emit_produces_winners_json(tmp_path):
    winners_out = tmp_path / "winners.json"
    header_out = tmp_path / "nvfp4_winners_table.hpp"
    proc = _run([
        "--sweep-dir", str(SWEEP_DIR),
        "--shortlist-header", str(SHORTLIST_HEADER),
        "--model-tag", "qwen35_27b",
        "--winners-out", str(winners_out),
        "--header-out", str(header_out),
    ])
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert winners_out.exists()
    data = json.loads(winners_out.read_text())
    assert data["model_tag"] == "qwen35_27b"
    for shape, buckets in EXPECTED.items():
        n, k = EXPECTED_NK[shape]
        assert data["by_shape"][shape]["N"] == n
        assert data["by_shape"][shape]["K"] == k
        for bucket, (cfg_name, idx) in buckets.items():
            entry = data["by_shape"][shape][bucket]
            assert entry["cfg"] == cfg_name, f"{shape} {bucket}: expected {cfg_name}, got {entry['cfg']}"
            assert entry["idx"] == idx,      f"{shape} {bucket}: expected idx {idx}, got {entry['idx']}"


def test_emit_produces_header_with_all_12_idx(tmp_path):
    winners_out = tmp_path / "winners.json"
    header_out = tmp_path / "nvfp4_winners_table.hpp"
    proc = _run([
        "--sweep-dir", str(SWEEP_DIR),
        "--shortlist-header", str(SHORTLIST_HEADER),
        "--model-tag", "qwen35_27b",
        "--winners-out", str(winners_out),
        "--header-out", str(header_out),
    ])
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    text = header_out.read_text()
    assert "namespace nvllm::fp4" in text
    assert "struct ShapeWinners" in text
    assert "lookup_m_mid_winner" in text
    # Row for each shape must appear with correct (N, K) and correct idx triple.
    for shape, buckets in EXPECTED.items():
        n, k = EXPECTED_NK[shape]
        i1 = buckets["16-32"][1]
        i2 = buckets["64-128"][1]
        i3 = buckets["192-256"][1]
        # Match in a whitespace-tolerant way.
        needle = f"{{ {n:>6}, {k:>6}, /*16-32*/ {i1:>2}, /*64-128*/ {i2:>2}, /*192-256*/ {i3:>2}"
        assert needle.replace(" ", "") in text.replace(" ", ""), (
            f"Expected row for {shape} not found. Looking for: {needle}"
        )


def test_check_mode_matches_committed_header():
    """--check mode must exit 0 if the committed header matches what codegen
    produces, non-zero otherwise. Only passes once the committed header is
    regenerated (Task 2 commits both)."""
    if not WINNERS_HEADER.exists():
        pytest.skip("committed nvfp4_winners_table.hpp not created yet (pre-Task 2)")
    proc = _run([
        "--sweep-dir", str(SWEEP_DIR),
        "--shortlist-header", str(SHORTLIST_HEADER),
        "--model-tag", "qwen35_27b",
        "--check",
    ])
    assert proc.returncode == 0, f"Committed header is stale. stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    # Paranoid second gate: the committed winners.json should also be up-to-date
    # (regenerated every time the header is).
    assert WINNERS_JSON.exists(), f"winners.json missing: {WINNERS_JSON}"
    data = json.loads(WINNERS_JSON.read_text())
    assert data["model_tag"] == "qwen35_27b"
    for shape in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        assert shape in data["by_shape"], f"winners.json missing shape: {shape}"
