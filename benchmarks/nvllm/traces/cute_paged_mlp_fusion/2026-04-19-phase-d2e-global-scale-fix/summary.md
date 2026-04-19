# Phase D2e — NVFP4 weight_global_scale fix

## Verdict: **SHIP**

Full fused stack (attention fusion + MLP fusion, `CUTE_ATTN_FUSION=1`
`CUTE_MLP_FUSION=1`) produces coherent 27B output.

- **GSM8K changed/ (full stack)**: 7/8, verdict PASS.
  Q2's "WRONG" is a GSM8K extraction artifact — the model wrote
  `600/60 = 10` in its reasoning and the simple extractor grabbed the
  first number (`600`) instead of the computed answer (`10`). True
  mathematical correctness is 8/8.
- **GSM8K ../2026-04-19-phase-d2e-stable-output-buf/changed/ (isolation,
  attn fusion off)**: 8/8, verdict PASS. Pure MLP-fusion path;
  no ambiguity.

Compared to pre-fix D2d (same kernel, no `weight_global_scale` passed):
0/8 gibberish (bitwise identical across multiple D2 variants, see
`../2026-04-18-phase-d2-op-body-move/`).

## Root cause

The Phase D MLP kernel dequantized NVFP4 weights as
`fp4_nibble × per_block_scale`. The complete NVFP4 dequant formula is
`fp4_nibble × per_block_scale × weight_global_scale`, where
`weight_global_scale = 1 / input_global_scale` from quantization time
(see `compressed_tensors_w4a4_nvfp4.py::process_weights_after_loading`
— the factor vLLM's standard NVFP4 GEMM applies as `alpha` post-matmul).

The kernel never took `weight_global_scale` as an argument; the backend
never bound it. Kernel output was therefore scaled by
`prod(1/weight_global_scale)` — roughly one order of magnitude too
large per matmul, compounding through silu.

## Why D1 "passed" 8/8 despite the bug

D1's `Qwen3_5MLP.forward` branched on `impl._mlp_fusion_active`, a
Python-level boolean. torch.compile traced this at warmup (prefill
shape, fusion disabled → `False`) and specialized the compiled graph to
the `False` branch permanently — Dynamo does not retrace attribute
reads on non-Tensor objects. At decode replay, the branch still ran
`False`, so `impl.mlp_output` (written by the buggy kernel) was never
read. The traced graph always ran the unfused gate_up/down GEMMs. The
kernel fired wastefully as an invisible side effect. GSM8K 8/8 came
from the unfused path.

D2 moved the kernel launch into an opaque custom op, which made
Inductor treat its output as a live data edge — and the
always-broken kernel's output finally reached the model's forward
pass. Result: gibberish. The D2 rewrite didn't break the kernel; it
removed the architectural accident that was hiding the kernel's
pre-existing bug.

## The fix

Kernel (`mlp_kernel.py`): add `gate_up_global_scale` +
`down_global_scale` scalar kwargs to `__call__`, thread them through
`_jit_launch` and `_kernel`, multiply into dequant at the three
weight-load sites (gate, up, down — both Path A production and Path B
fallback). Gate and up share one scale because both live in one
`MergedColumnParallelLinear`.

Backend (`_backend.py::_resolve_mlp_weights`): bind
`self._mlp_gate_up_gs` and `self._mlp_down_gs` from
`gate_up.weight_global_scale` / `down.weight_global_scale`. `.item()`
sync happens once at attach — subsequent forwards pass Python floats.

Op body (`_mlp_op.py`): pass the cached scales via kwargs.

## Context

- **Image**: `nvllm:gb10-phaseD2e`.
- **Model**: `natfii/Qwen3.5-27B-NVFP4-Opus-GB10`.
- **Config**: `max-model-len=65536`, `max-num-seqs=4`,
  `kv-cache-dtype=fp8_e4m3`, `cudagraph_mode=PIECEWISE`,
  `--language-model-only`, `--gpu-memory-utilization 0.80`.
- **Workload**: 4 concurrent × 128 tok, temperature=0, ignore_eos=true.

## Raw numbers (full stack)

| Metric | Value |
|---|---|
| Self CUDA time total | 182.926 s |
| Phase D MLP kernel | 139.356 s / 2032 calls / 68.581 ms avg |
| Attention fused kernel | 32.012 s / 2032 calls / 15.754 ms avg |
| GemmUniversal (unfused non-MLP GEMMs) | 6.645 s / 34544 calls |
| gdn_attention_core (linear-attn layers) | 191.568 ms / 6096 calls |

34544 GemmUniversal calls vs. D1 baseline's 38608 = 4064 fewer =
`16 fused layers × 2 GEMMs (gate_up + down) × 127 steps`. Dual-firing
remains eliminated — no regression on the D2 architectural win.

## Kernel perf is still tuning-unready

MLP kernel avg is 68 ms for 4-token decode. Kernel is tiled for
prefill-sized work (`tile_s=256`, `tile_k=640`, `slice_ctas=8`); most
CTAs are idle on small decode batches. Fused-path decode is ~3× slower
per question than unfused (`8.7 s/Q` fused vs `~2.5 s/Q` unfused from
D2d baseline). **Correctness is landed; perf is Phase D3 scope.**

## Live-Python kernel correctness repro

The bug was isolated in `/tmp/phase_d2e/kernel_repro.py` before the
Docker rebuild cycle (per the `feedback_debug_math_live.md` memory
entry). The repro quantizes random BF16 weights via
`scaled_fp4_quant`, runs `Phase_D_MLP_Kernel`, and diffs against a
PyTorch reference (`silu(x@W_gate^T) * (x@W_up^T) @ W_down^T`).

- Pre-fix: kernel `std=21.0` vs ref `std=3.8` — 5.5× scale mismatch.
- Forcing reference to also omit global scale: `std=21.0` vs `20.97`
  → matches the kernel's bug exactly, confirming global-scale as the
  single failure point.
- Post-fix: kernel `std=3.79` vs ref `std=3.80` — matches within
  NVFP4 quantization precision (abs-diff p99 = 0.87, ≈ 23% of ref
  std, consistent with 3-matmul FP4 error compounding).

## How to reproduce

```bash
cd /home/natfii/docker/nvllm
# Full stack (attention fusion + MLP fusion)
CUTE_ATTN_FUSION=1 ./scripts/phase_d2e_trace_capture.sh changed
# Isolation (MLP fusion only, attn fusion off — D2d repro method)
CUTE_ATTN_FUSION=0 ./scripts/phase_d2e_trace_capture.sh changed
# Baseline for either mode (MLP fusion off)
CUTE_ATTN_FUSION=<1|0> ./scripts/phase_d2e_trace_capture.sh baseline

# Kernel-math correctness (no Docker, ~30 s)
.venv/bin/python /tmp/phase_d2e/kernel_repro.py
```

## Artifacts in this directory

```
changed/
  decode_log.txt               # vLLM server log
  profiler_out_0.txt           # kernel-level profile
  rank0.*.pt.trace.json.gz     # torch profiler trace
  gsm8k_changed.{json,log}     # 7/8 PASS (1 extraction artifact)
  workload_{1..4}.json         # raw completion responses
```
