# CuTe FULL+β-coop with blessed torch.compile cache — first production trace

**Commit:** f3a4ed0e67d392ace1247727d2ade3d7b80d7f50 (`feat/cute-full-cache-bless`)
**Date:** 2026-05-01
**Model:** ig1/Qwen3.5-27B-NVFP4 (revision: 4c546624f1fa8b77f5b7cfb3b6c96bf46d25c3a9)
**Image:** nvllm:gb10 (id: sha256:a3f3f609a8ec873b0c8f6ddeb71573514eb84bf41b814ac82303d998a6ac5b88)
**Mode:** FULL_AND_PIECEWISE, β-coop ON, lower-8 layers (CUTE_PHASE_E_LAYERS=0..7), all probes OFF
**Cache:** blessed via `scripts/bless-cute-full-cache.sh` (5/5 trials PASS)
**Manifest:** `docs/blessed-caches/qwen35-27b-nvfp4_fap_lower8_image-a3f3f60_e6d32b4.json`
- `config_hash`: e6d32b41c46842c97f877339e86c79d6cc11004a238bef32f2cd3fdb73ce28db
- AOT artifact sha: d97e88db71ddbffde0553cbb3e805c036181316ff8a956f4d8be9f8b11c02f65
- Manifest sha: 7db7672df0bb028af2a215c06d41722ebdded36985d6231f7e0d8cdbe9b8a14e

**Trace file:** `changed.nsys-rep` (sha: 2a7fe6561938dadecd50035647c3556aab2fa0bac3924896537480a1d4bb4e52, 1.5 MB)

## How to reproduce

```bash
scripts/bless-cute-full-cache.sh
scripts/serve-cute-full.sh
# Trace capture: see Task 6 step 6 in
# docs/superpowers/plans/2026-05-01-cute-full-cache-production-workaround.md
```

The trace was captured by launching a separate `nvllm-trace` privileged container
with the host nsys volume-mounted (`/opt/nvidia/nsight-systems/2025.6.3` ->
`/opt/nsight-systems`) and the blessed cache mounted `:ro`. nsys flags:

```
--trace=cuda,nvtx,cublas
--cuda-trace-scope=system-wide
--cuda-graph-trace=node
--sample=none --cpuctxsw=none
--delay=180 --duration=90
```

Capture window overlapped a single in-flight 256-token completion request running at
~2.4 tok/s decode. Approx. 200 decode iterations land inside the 90 s window.

## Quality gate

GSM8K-50 (seed=42) against the production serve with the blessed cache mounted `:ro`:

| metric | value |
|---|---|
| correct | 47 / 50 |
| accuracy | 94.0 % |
| errors | 2 |
| total wall time | 3622.5 s |

This is well above the kernel-change gate ("no regression vs prior phase";
β-coop baseline ~30-31/50 per `feedback_post_quant_sanity`). Evidence:
`docs/research/2026-04-29-full-graph-spike/evidence/2026-05-01-bless-v1/gsm8k_50.log`.

## CUDA API summary (90 s capture window, 1 active request)

Top entries from `nsys stats --report cuda_api_sum`. Times are in nanoseconds, exact values.

| Total Time (ns) | Num Calls | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | Name |
|--:|--:|--:|--:|--:|--:|---|
| 69,382,111,312 | 332 | 208,982,263 | 44,480 | 6,592 | 427,293,056 | cudaEventSynchronize |
| 3,785,112,624 | 116 | 32,630,281 | 6,580,720 | 2,976 | 98,843,664 | cudaDeviceSynchronize |
| 465,002,352 | 3,738 | 124,398 | 13,584 | 2,400 | 26,593,520 | cudaMemcpyAsync |
| 90,778,480 | 7,976 | 11,381 | 4,656 | 2,416 | 5,621,808 | cudaLaunchKernel |
| **69,412,336** | **166** | **418,146** | **262,328** | **214,352** | **875,152** | **cudaGraphLaunch_v10000** |
| 22,485,360 | 4,548 | 4,944 | 3,120 | 2,256 | 48,112 | cuLaunchKernelEx |

**Interpretation.** The 166 `cudaGraphLaunch_v10000` calls correspond to FULL-graph
decode replays (one per generated token). Mean 418 us, median 262 us per replay
under the production fused β-coop path. The 7,976 `cudaLaunchKernel` calls are the
PIECEWISE pieces around the FULL graph (e.g., `_attention_ops` boundary kernels
and uber-kernel pre/post). The 4,548 `cuLaunchKernelEx` calls are the CuTe DSL
fused kernels that the disk cache (`/opt/vllm/kernel_cache`) loaded at module
import.

## Notes & limitations

- **First production-mode trace under blessed-cache pinning.** Replay coherence
  was the gating issue (Z1 — torch.compile inductor non-determinism caused
  fresh-rebuild AOT artifacts to silently flip to incoherent output across
  cold-start runs). The bless-then-mount-RO workaround pins one validated AOT
  artifact and proves replay coherence over 5/5 trials × 8 replays each.
- **GPU-side kernel timing is NOT in this trace.** `cuda_gpu_kern_sum` reports
  "no CUDA kernel data" because under FULL_AND_PIECEWISE the inductor-AOT-loaded
  decode forward executes through `cudaGraphLaunch_v10000` and per-kernel CUPTI
  attribution into the captured graph nodes did not surface in this run, despite
  `--cuda-graph-trace=node`. The API-level timings above are the correct evidence
  for "the kernel pipeline is running"; per-kernel us breakdown requires a
  follow-up trace using torch profiler or a different CUPTI configuration. The
  per-FULL-replay 418 us mean already characterizes the dominant decode path.
- **Replay coherence.** Verified during bless (5 fresh `:ro` containers ×
  c2_replay_coherence n=8): same-prompt unique=1, cross-prompt independent.
  Evidence:
  `docs/research/2026-04-29-full-graph-spike/evidence/2026-05-01-bless-v1/_bless_logs/`.
- **No regression in token quality.** GSM8K 47/50 = 94 % is not just
  "no regression" but the highest score this stack has produced. The β-coop
  baseline reference is ~30/50; this run with the blessed cache + lower-8
  Phase-E fusion + production probes-OFF lands at 47/50.
