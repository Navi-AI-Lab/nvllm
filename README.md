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

## Quick Start (Prebuilt Image)

Pull the prebuilt image (~15-25 GB) and run a model:

```bash
docker pull ghcr.io/navi-ai-lab/nvllm:latest

docker run --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  --entrypoint bash \
  ghcr.io/navi-ai-lab/nvllm:latest \
  scripts/serve_qwen35.sh
```

**Required flags:** `--gpus all --ipc=host --network host` (vLLM needs shared memory and GPU access).

**Cache mounts** (recommended — avoid re-downloads and JIT recompilation on restart):
- `~/.cache/huggingface` — model weights
- `~/.cache/flashinfer` — FlashInfer JIT kernels
- `~/.cache/vllm_compile` → `/root/.cache/vllm/torch_compile_cache` — CUDA graph cache

**For gated models** (e.g., Gemma 4): pass `-e HF_TOKEN=hf_...` or mount a token file.

### Available Serve Scripts

| Script | Model | Context |
|--------|-------|---------|
| `serve_qwen35.sh` | Qwen3.5-122B-A10B-NVFP4 (MoE) | 32K |
| `serve_qwen35_27b.sh` | Qwen3.5-27B-NVFP4-Opus (dense) | 64K |
| `serve_nemotron.sh` | Nemotron-3-Super-120B-A12B-NVFP4 | 128K |
| `serve_gemma4.sh` | Gemma 4 31B IT NVFP4 (local quant) | 32K |
| `serve_qwen35_agents.sh` | Qwen3.5-122B-A10B-NVFP4 (agents) | 64K |
| `serve_qwen3_coder_next.sh` | Qwen3-Coder-Next-NVFP4 | 128K |

Also available on Docker Hub: `docker.io/naviailab/nvllm:latest`

## Quick Start

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

### 2. Run a model
```
./scripts/run_qwen35.sh
```

First run downloads the model automatically (~25 GB).
API available at `http://localhost:8000/v1`.

All models are served as `default` — use `"model": "default"` in API calls:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Available Models

Benchmarks will eventually be dated and version pinned once repo is stable. For now ignore tok/s here.

| Script | Model | Active Params | Speed | Context |
|--------|-------|---------------|-------|---------|
| `run_qwen35_27b_nvfp4-opus.sh` | [Qwen3.5-27B-NVFP4-Opus-GB10](https://huggingface.co/natfii/Qwen3.5-27B-NVFP4-Opus-GB10) | 27B | ~45 tok/s | 64K |
| `run_nemotron.sh` | Nemotron-3-Super-120B | 12B | TBD | 16K (128K w/ --tq) |
| `run_qwen35.sh` | Qwen3.5-122B-A10B | 10B | TBD | 32K |
| `run_qwen3_coder_next.sh` | Qwen3-Coder-Next | 3B | TBD | 128K |
| `run_gemma4.sh` | Gemma 4 31B IT | 31B | TBD | 32K |

### Baseline Config

All launch scripts use a standard baseline for consistent benchmarking:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max-num-seqs` | 4 | Fixed across all scripts — baseline for multi-user serving comparisons |
| `kv-cache-dtype` | auto (FP8) | FP8 KV cache — best throughput. Use `--tq` for TurboQuant (more context, ~25% slower) |
| `gpu-memory-utilization` | varies | Tuned per model to only what's needed for the target context length |

Benchmarks are always run with `max-num-seqs=4` and FP8 KV so results are comparable across models and optimizations.

### Flags

| Flag | Effect |
|------|--------|
| `--tq` | TurboQuant KV cache — more context capacity, ~25% lower throughput |
| `--debug` | Eager mode, no CUDA graphs (for debugging) |

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
