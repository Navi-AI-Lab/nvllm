"""
Out-of-engine precompile for the β-coop FULL kernel.

Builds dummy tensors of the shapes/dtypes/strides run_beta_coop_full
expects for max_num_seqs=1, then calls
PhaseE_Beta_Kernel.run_beta_coop_full(..., compile_only=True). The
compile_only kwarg short-circuits the launch but still drives the
_compile_coop_full path that populates the disk cache.

Spec: docs/superpowers/specs/2026-04-29-cute-full-compile-cache-design.md §5 Step 2.1.

Run inside a one-shot nvllm container — see scripts/precompile-cute-coop-full.sh.

Exit codes:
  0 — compile completed
  1 — env / config / image precondition failed
  2 — run_beta_coop_full raised at compile_only=True (shape mismatch)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompile-cute-coop-full")


# --- Bind-mount-aware import ----------------------------------------------
# nvllm:gb10 was built with `pip install -e /app/nvllm`. When the precompile
# container bind-mounts the host source tree at /workspace, /workspace/vllm
# may contain post-build edits. Insert /workspace at the head of sys.path so
# `import vllm` picks up the edited tree, not the build-time-baked /app/nvllm.
_WORKSPACE = Path("/workspace")
if _WORKSPACE.exists() and (_WORKSPACE / "vllm" / "__init__.py").exists():
    sys.path.insert(0, str(_WORKSPACE))


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        logger.error("missing required env: %s", name)
        sys.exit(1)
    return val


def _resolve_config_dims(model_id: str) -> dict[str, int]:
    """Read config.json directly so we don't depend on a transformers version
    that knows the model's architecture name. text_config wins over top-level
    when present (Qwen3.5 places the dims there)."""
    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snap = snapshot_download(
        repo_id=model_id,
        cache_dir=os.path.join(cache_dir, "hub")
        if not cache_dir.endswith("hub") else cache_dir,
        allow_patterns=["config.json"],
        local_files_only=True,
    )
    with open(Path(snap) / "config.json") as f:
        cfg = json.load(f)

    # text_config takes precedence (Qwen3.5 puts dims there); fall back to
    # top-level for non-multimodal checkpoints.
    tc = cfg.get("text_config", cfg)

    out = {}
    for k in ("hidden_size", "intermediate_size", "num_attention_heads",
              "num_key_value_heads", "head_dim", "num_hidden_layers",
              "rms_norm_eps"):
        v = tc.get(k)
        if v is None:
            v = cfg.get(k)
        out[k] = v
    if out["head_dim"] is None and out["hidden_size"] and out["num_attention_heads"]:
        out["head_dim"] = out["hidden_size"] // out["num_attention_heads"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=os.environ.get("HF_MODEL", "ig1/Qwen3.5-27B-NVFP4"),
    )
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--page-size", type=int, default=16,
        help="Pages per slot — must match serve config (16 default).",
    )
    parser.add_argument(
        "--num-pages", type=int, default=4,
        help="Dummy KV cache page count — only the layout matters.",
    )
    args = parser.parse_args()

    # Heavy imports happen here (after argparse) so --help works on host venvs
    # that lack a CUDA-enabled torch or vllm.
    import torch  # noqa: F401  (used below)

    _require_env("B12X_CUTE_COMPILE_DISK_CACHE")
    cache_dir = _require_env("B12X_CUTE_COMPILE_CACHE_DIR")
    logger.info("disk cache dir: %s", cache_dir)

    # Apply the disk-cache patch BEFORE any cute.compile path can fire.
    # Without this, run_beta_coop_full's _compile_coop_full will compile
    # but never persist.
    from vllm.v1.attention.backends.cute_paged.disk_cache import (
        apply_disk_cache_patch,
    )
    apply_disk_cache_patch(cache_dir=cache_dir)

    cfg = _resolve_config_dims(args.model)
    logger.info(
        "model dims: hidden=%d intermediate=%d heads=%d kv_heads=%d "
        "head_dim=%d layers=%d eps=%s",
        cfg["hidden_size"], cfg["intermediate_size"],
        cfg["num_attention_heads"], cfg["num_key_value_heads"],
        cfg["head_dim"], cfg["num_hidden_layers"], cfg["rms_norm_eps"],
    )

    from vllm.v1.attention.backends.cute_paged import phase_e_kernel

    # Construct the kernel exactly as _backend.py does at serve time —
    # tile_s/tile_k/slice_ctas left None so CUTE_MLP_TILE env preset
    # resolves identically across precompile and serving.
    kernel = phase_e_kernel.PhaseE_Beta_Kernel(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_attn_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        rms_eps=cfg.get("rms_norm_eps") or 1e-6,
    )

    # ------------------------------------------------------------------
    # Build dummy tensors mirroring _backend.py:1517-1560 EXACTLY.
    # NVFP4 packs 2 vals per byte → packed_dim = K // 2.
    # NVFP4 blockscale group=16 → sf_dim = K // 16.
    # ------------------------------------------------------------------
    nat = args.max_num_seqs
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    H = cfg["num_attention_heads"]
    Hk = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    page_size = args.page_size
    num_pages = args.num_pages
    max_pages = num_pages

    bf = torch.bfloat16
    dev = "cuda"

    # Phase 0/1 inputs
    hidden_in = torch.zeros(nat, hidden, dtype=bf, device=dev)
    residual_in = torch.zeros(nat, hidden, dtype=bf, device=dev)
    input_gamma = torch.ones(hidden, dtype=bf, device=dev)
    post_attn_gamma = torch.ones(hidden, dtype=bf, device=dev)
    attn_input_bf16 = torch.zeros(nat, hidden, dtype=bf, device=dev)

    # Attn (Phase 1)
    query = torch.zeros(nat, H, hd, dtype=bf, device=dev)
    # KV cache layout: [num_pages, 2, page_size, num_kv_heads, head_dim] uint8
    kv_cache = torch.zeros(
        num_pages, 2, page_size, Hk, hd, dtype=torch.uint8, device=dev,
    )
    page_table = torch.zeros(nat, max_pages, dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([page_size] * nat, dtype=torch.int32, device=dev)

    # W_O (NVFP4)
    K_packed = hidden // 2   # bytes
    K_sf = hidden // 16      # blockscale rows
    wo_weight = torch.zeros(hidden, K_packed, dtype=torch.uint8, device=dev)
    wo_scales = torch.zeros(
        hidden, K_sf, dtype=torch.float8_e4m3fn, device=dev,
    )
    wo_global_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
    attn_output = torch.zeros(nat, hidden, dtype=bf, device=dev)

    # MLP gate/up: input dim = hidden, output dim = intermediate
    gate_w_fp4 = torch.zeros(inter, K_packed, dtype=torch.uint8, device=dev)
    gate_w_scale = torch.zeros(inter, K_sf, dtype=torch.uint8, device=dev)
    up_w_fp4 = torch.zeros(inter, K_packed, dtype=torch.uint8, device=dev)
    up_w_scale = torch.zeros(inter, K_sf, dtype=torch.uint8, device=dev)

    # MLP down: input dim = intermediate, output dim = hidden
    inter_packed = inter // 2
    inter_sf = inter // 16
    down_w_fp4 = torch.zeros(hidden, inter_packed, dtype=torch.uint8, device=dev)
    down_w_scale = torch.zeros(hidden, inter_sf, dtype=torch.uint8, device=dev)
    mlp_output = torch.zeros(nat, hidden, dtype=bf, device=dev)

    # Caller-supplied output buffers
    residual_output = torch.empty(nat, hidden, dtype=bf, device=dev)
    # gate_buf: BF16, [nat, num_attn_heads * head_dim], contiguous
    gate_buf = torch.zeros(nat, H * hd, dtype=bf, device=dev)

    t0 = time.monotonic()
    logger.info("starting compile_only=True (heartbeat fires every 5min)…")
    try:
        kernel.run_beta_coop_full(
            hidden_in=hidden_in,
            residual_in=residual_in,
            input_gamma=input_gamma,
            post_attn_gamma=post_attn_gamma,
            attn_input_bf16=attn_input_bf16,
            query=query,
            kv_cache=kv_cache,
            page_table=page_table,
            seq_lens=seq_lens,
            wo_weight=wo_weight,
            wo_scales=wo_scales,
            wo_global_scale=wo_global_scale,
            attn_output=attn_output,
            gate_w_fp4=gate_w_fp4,
            gate_w_scale=gate_w_scale,
            up_w_fp4=up_w_fp4,
            up_w_scale=up_w_scale,
            down_w_fp4=down_w_fp4,
            down_w_scale=down_w_scale,
            mlp_output=mlp_output,
            residual_output=residual_output,
            gate_buf=gate_buf,
            compile_only=True,
        )
    except (TypeError, AssertionError, RuntimeError) as e:
        logger.error("run_beta_coop_full(compile_only=True) failed: %s", e)
        return 2

    elapsed = time.monotonic() - t0
    logger.info(
        "compile completed in %.1fs (%.1fmin)", elapsed, elapsed / 60.0,
    )
    print(f"PRECOMPILE_OK elapsed_s={elapsed:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
