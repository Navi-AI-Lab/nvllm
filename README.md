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

### Available Models

| Script | Model | Active Params | Speed | Context |
|--------|-------|---------------|-------|---------|
| `run_nemotron.sh` | Nemotron-3-Super-120B | 12B | ~20 tok/s | 16K (128K w/ --tq) |
| `run_qwen35.sh` | Qwen3.5-122B-A10B | 10B | ~25 tok/s | 32K |
| `run_qwen3_coder_next.sh` | Qwen3-Coder-Next | 3B | ~34 tok/s | 128K |
| `run_gemma4.sh` | Gemma 4 31B IT | 31B | ~7 tok/s | 32K |

### Flags

| Flag | Effect |
|------|--------|
| `--tq` | TurboQuant 3.5-bit KV cache (saves memory, enables longer context) |
| `--debug` | Eager mode, no CUDA graphs (for debugging) |
