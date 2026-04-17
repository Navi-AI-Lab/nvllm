# Own-the-stack Phase B Tier-3 evidence — 2026-04-17

**Refactor branch:** `feat/own-the-stack-phase-b`
**Model:** `natfii/Qwen3.5-27B-NVFP4-Opus-GB10`
**Image:** `nvllm:gb10-ots`
**Graph mode:** PIECEWISE CUDA graphs
**Baseline commit (fusion-ship):** `37cceaa6c`

## GSM8K result

**8/8 (100%) — matches baseline.**

Two runs, both 8/8:
- Initial run without `CUTE_DEBUG_FUSION` — 8/8
- Re-run with `CUTE_DEBUG_FUSION=1` for evidence capture — 8/8

## Fusion engagement

`decode_log.txt` — first 3 full-attention layers × 2 decode steps × phase B/C:

- **Phase B (W_O GEMV):** kernel `absmax` / `mean` vs Python-dequant reference → `close=True` on every line.
- **Phase C (residual + RMSNorm):** kernel `hidden` and `residual` outputs vs reference → `close_h=True close_r=True` on every line.

Across the entire GSM8K run, 1920 decode steps engaged fusion; zero lines with `close=False` or `close_h=False` or `close_r=False`.

Startup logged the new API firing for all 16 full-attention layers:

```
INFO [_backend.py:302] CuTe fusion attached: layer=model.layers.3 max_num_seqs=4 hidden_dim=5120 q_size=6144 attn_output_gate=True
(... 16 such lines, layer indices 3 7 11 ... 63 ...)
INFO [_backend.py:355] CuTe fusion resolved: layer=model.layers.3 wo_weight=[...] rmsnorm_gamma=[...]
(... 16 such lines ...)
```

Confirms `CutePagedAttentionImpl.attach_fusion(parent_layer)` + `_resolve_fusion_weights()` replaced the old `_fusion_bind_callback` / `bind_fusion_weights` pair with no behavioral change.

## Tier-1 jupyter tests (host-side)

All 5 pass: `notebooks/nvllm/fusion_bind_tests.ipynb`.

1. NVFP4 happy-path
2. BF16 skip-path (CLAUDE.md debug protocol step 2 regression gate)
3. Double-resolve rebinds to fresh tensor identity (C1 + C2)
4. Buffer pointer stability across attach calls (H3)
5. Per-forward gate boundary `num_actual_tokens > max_num_seqs` (A3)

## How to reproduce

```bash
cd /home/natfii/docker/nvllm
git checkout feat/own-the-stack-phase-b
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10-ots .
NVLLM_IMAGE=nvllm:gb10-ots CUTE_DEBUG_FUSION=1 ./scripts/serve-cute.sh
.venv/bin/python scripts/gsm8k_sanity.py
```

## Rollback

Single-commit refactor. `git revert HEAD` returns to `37cceaa6c`.
Docker image snapshot `nvllm:gb10-preshim-20260417` tags the pre-refactor build.
