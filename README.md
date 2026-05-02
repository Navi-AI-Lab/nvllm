<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <img width="441" height="707" alt="a1" src="https://github.com/user-attachments/assets/6fdaa971-95bc-4652-97d7-fa160fa2954c" />

</p>

<h3 align="center">
vLLM Fork with focus on GB10 Homelabs
</h3>

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

If you use vLLM for your research, please cite their [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Quick Start

> **Recommended:** Build from source. The prebuilt images on GHCR and Docker Hub lag behind `main` — custom kernel work (CuTe attention, stream-K GEMM) ships here first and images are only rebuilt periodically.

### Prebuilt image (convenience)

```bash
docker pull ghcr.io/navi-ai-lab/nvllm:latest
```

Also on Docker Hub: `docker.io/naviailab/nvllm:latest`

### Build from source (recommended)

**Required flags:** `--gpus all --ipc=host --network host` (vLLM needs shared memory and GPU access).

**Cache mounts** (recommended — avoid re-downloads and JIT recompilation on restart):
- `~/.cache/huggingface` — model weights
- `~/.cache/flashinfer` — FlashInfer JIT kernels
- `~/.cache/vllm_compile` → `/root/.cache/vllm/torch_compile_cache` — CUDA graph cache

**For gated models** (e.g., Gemma 4): pass `-e HF_TOKEN=hf_...` or mount a token file.

### Prerequisites
- NVIDIA DGX Spark (GB10) or GH200
- Docker with NVIDIA Container Toolkit
- Hugging Face account with model access
- `huggingface-cli` on host (`pip install huggingface-hub`)

### 1. Clone and build
```bash
git clone https://github.com/Navi-AI-Lab/nvllm.git
cd nvllm
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 .
```

Already cloned? Pull the latest first:
```bash
cd nvllm && git pull
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 .
```

### 2. Serve a model
```bash
./scripts/serve.sh
```

First run downloads the model automatically (~18 GB).
API available at `http://localhost:8000/v1`.

All models are served as `default` — use `"model": "default"` in API calls:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Serve Scripts

| Script | Model | Status | Context |
|--------|-------|--------|---------|
| `serve.sh` | [Qwen3.5-27B-NVFP4-Opus-GB10](https://huggingface.co/natfii/Qwen3.5-27B-NVFP4-Opus-GB10) | Active (default) | 64K |
| `serve-cute.sh` | [Qwen3.5-27B with CuTe Paged Attention](https://huggingface.co/ig1/Qwen3.5-35B-A3B-NVFP4) | Active (kernel dev) | 64K |
| `serve-nemotron.sh` | Nemotron-3-Super-120B-A12B-NVFP4 | Not Ready | 128K |
| `serve-gemma4.sh` | Gemma 4 31B IT NVFP4 | Degraded (see script) | 32K |

### Flags

| Flag | Effect |
|------|--------|
| `--tq` | TurboQuant KV cache — more context capacity, ~25% lower throughput (serve.sh only) |
| `--debug` | Eager mode, no CUDA graphs (for debugging) |

### Roadmap

**Now — Qwen3.5-27B kernel work**
- CuTe DSL paged attention uber-kernel (fused attention + W_O GEMV + RMSNorm)
- CUDA graph support (FULL_AND_PIECEWISE mode)
- End-to-end fusion validation through Qwen3NextAttention

**Next — expand model support**
- Gemma 4 31B IT — blocked on vLLM PR #38891 (per-layer attention backend for mixed head_dim)
- Devstral 2 Large — NVFP4 quantization and serve script

### SM120 Stream-K Decode Optimization

This fork includes a custom CUTLASS FP4 GEMM kernel with **stream-K scheduling** for small-M decode (M≤16). Stream-K distributes K-dimension work across SMs, improving utilization when the batch size is too small to fill all SMs with standard tile scheduling.

Based on CUTLASS's own `sm120_bs_gemm_nvf4_nvf4_f32_f32_stream_k` test kernel, adapted for vLLM's dispatch:
- **Tile:** 128×128×256 (K doubled from default 128)
- **Schedule:** `KernelTmaWarpSpecializedCooperative`
- **Tile scheduler:** `StreamKScheduler`

Benchmarked on Qwen3.5-27B-NVFP4 (rate=8, max-num-seqs=4):

| Metric | Baseline | Stream-K | Delta |
|--------|----------|----------|-------|
| Output tok/s | 40.0 | 44.9 | **+12.2%** |
| TPOT p50 | 89.2 ms | 80.0 ms | **-10.2%** |
| TPOT p99 | 91.7 ms | 82.7 ms | **-9.8%** |

> **Warning:** Large models (>75 GB) that leave minimal memory headroom on the GB10's 128 GB unified memory may crash during CUDA graph capture with the stream-K kernel. Use `--debug` (eager mode) to test first, or use a smaller model.

### CuTe Paged Attention Backend (Prototype)

Custom paged attention backend using CuTe Python DSL, targeting SM120/SM121 FP8 MMA instructions. Registered as `CUTE_PAGED` in vLLM's attention backend registry.

**Status:** Backend interface validated end-to-end. PyTorch prototype serves live inference. CuTe DSL kernel replacement in progress.

Launch with: `./scripts/serve-cute.sh --debug`

## Acknowledgments

- **[b12x](https://github.com/lukealonso/b12x)** by Luke Alonso — CuTe DSL paged attention with FP8 KV inline dequant, TMA plane loading, and split-KV merge. Reference implementation for the CuTe paged attention backend. Pinned at [`c469c66`](https://github.com/lukealonso/b12x/tree/c469c6637f6251adefc282956f5392e559ea915d).
  - [`docs/kernel-insights/2026-04-10-b12x-cute-attention.md`](docs/kernel-insights/2026-04-10-b12x-cute-attention.md) — CuTe attention & disk cache patterns
  - [`docs/kernel-insights/2026-04-11-b12x-paged-attention.md`](docs/kernel-insights/2026-04-11-b12x-paged-attention.md) — Full paged attention kernel architecture (1165 lines, 59 pinned permalinks)
  - [`docs/kernel-insights/2026-04-17-b12x-mlp-fusion.md`](docs/kernel-insights/2026-04-17-b12x-mlp-fusion.md) — Per-slice MLP fusion structure and UE4M3 blockscale encoding referenced from [b12x @ c469c66](https://github.com/lukealonso/b12x/tree/c469c6637f6251adefc282956f5392e559ea915d) for the Phase D fused MLP decode kernel.
- **[CUTLASS PR #3030](https://github.com/NVIDIA/cutlass/pull/3030)** by blake-snc (Second Nature Computing) — SM120 Flash Attention v2 reference for fused multi-head attention on Blackwell.
  - [`docs/kernel-insights/2026-04-10-cutlass-pr3030-sm120-fmha.md`](docs/kernel-insights/2026-04-10-cutlass-pr3030-sm120-fmha.md) — SM120 FMHA patterns and tile configs
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** by NVIDIA — CuTe Python DSL for SM120 kernel development. The FP4 decode GEMM kernel with stream-K scheduling is adapted from CUTLASS test kernels.
- **[Simon Veitner's CuTe DSL / NVFP4 blog](https://veitner.bearblog.dev/blog/)** — Reference reading for NVFP4 GEMV K-parallel reduction patterns identified as the next β-coop optimization direction (see [phaseE-tax bench](benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-phaseE-tax-3leg/summary.md)).
- **[vLLM](https://github.com/vllm-project/vllm)** — The upstream project this fork is based on.
