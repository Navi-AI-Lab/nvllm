# Silly Streaming microbenches

> ## 🪦 RIP Silly Stream — 2026-04-19
>
> **Verdict: not feasible on GB10.** The gatekeeper (#1) killed it fast. Two independent failures compound:
>
> 1. **Blackwell HW-DE is not exposed on GB10.** nvCOMP 5.2 (`nvidia-nvcomp-cu13==5.2.0`) with `decompress_backend=NVCOMP_DECOMPRESS_BACKEND_HARDWARE` raises `RuntimeError code=12` at decode time. The HW Decompression Engine ships on datacenter Blackwell (B100/B200), not on GB10 / SM120.
> 2. **SW fallback is too slow to paper over it.** CUDA-backend Deflate on 1 GiB: **3.7 GB/s on uniform-random**, 12-13 GB/s on trivially compressible data. Real NVFP4 weights would sit near the random column (~5-8 GB/s best case).
>
> Plugging the SW number back into #4 pushes the latency penalty from 8× to ~10-12× — worse than the default assumptions suggested.
>
> **Long live silly stream** — the idea is shelved, not forgotten. If a future GB-class edge part ever exposes HW-DE, rerun `#1` first. Everything below is preserved as the evidence trail of the attempt.
>
> — run notes: throwaway venv at `/tmp/silly_streaming_venv`; scripts in this dir still reference the dead `kvikio.nvcomp_codec.NvCompBatchCodec` API (removed in kvikio 26.04). Use `nvidia.nvcomp` directly if reviving.

---

Four cheap experiments that kill-or-validate the "GDS + nvCOMP + Blackwell
Decompress Engine for JIT MoE expert streaming on GB10" idea before any
loader code gets written.

Full motivation, bandwidth math, target-model table, and prior art:
[`2026-04-19-silly-streaming.md`](../2026-04-19-silly-streaming.md).

## Scripts

| # | Script | Question it answers | Deps |
|---|---|---|---|
| 1 | `01_nvcomp_deflate_bench.py` | Is the Blackwell HW-DE active on SM120 (GB10)? | kvikio-cu13, cupy-cuda13x |
| 2 | `02_cufile_vs_mmap_bench.py` | Does cuFile beat mmap on unified memory? | kvikio-cu13, cupy-cuda13x |
| 3 | `03_nvfp4_compression_ratio_survey.py` | What ratio does Deflate get on real NVFP4 weights? | safetensors, torch |
| 4 | `04_moe_streaming_overlap_sim.py` | Does per-layer stream+compute overlap pencil out? | stdlib only |

## Install

```bash
uv pip install kvikio-cu13 cupy-cuda13x safetensors torch
```

## Run (not yet — these exist for the next session)

```bash
# #1 — gatekeeper: HW-DE alive on GB10?
.venv/bin/python docs/research/silly_streaming/01_nvcomp_deflate_bench.py \
    --size-gb 1 --iterations 5 --codec Deflate

# #2 — cuFile vs mmap (prepare the test file first)
dd if=/dev/urandom of=/tmp/silly_streaming_4g.bin bs=1M count=4096
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
.venv/bin/python docs/research/silly_streaming/02_cufile_vs_mmap_bench.py \
    --path /tmp/silly_streaming_4g.bin --size-gb 4

# #3 — real-checkpoint compression ratios
.venv/bin/python docs/research/silly_streaming/03_nvfp4_compression_ratio_survey.py \
    --model ~/models/Qwen3.5-27B-NVFP4-Opus-GB10

# #4 — does the math overlap with compute? (stdlib only, no hardware)
for m in kimi-k2.5 glm-5.1 deepseek-v4 qwen3.6-35b-a3b; do
    .venv/bin/python docs/research/silly_streaming/04_moe_streaming_overlap_sim.py --model "$m"
done
```

## Expected outcomes

- **#1 ≥ 50 GB/s Deflate decomp with low SM utilization** → HW-DE is live on GB10; the feasibility story stands. If we see ~15-25 GB/s with SMs pegged, we're on SW fallback and the overlap budget tightens but the project is still plausible (rerun #4 with `--hw-de-gbs 20` to re-verdict).
- **#2 cuFile within 10-15% of mmap** → unified-memory hypothesis confirmed; pick whichever API is cleaner. A big cuFile win would be surprising and worth investigating before trusting either number.
- **#3 overall ratio < 1.15x on NVFP4** → single-stream Deflate is load-bearing in the weak sense; consider splitting mantissa/scale streams if the per-class ratios differ by >2x.
- **#4 verdict "feasible" or "tight"** for at least one of Kimi K2.5 / GLM-5.1 / DeepSeek V4 → worth prototyping a loader.

## Total cost to kill-or-validate

~3-4 hours wall time, no Docker rebuilds, no kernel changes.
