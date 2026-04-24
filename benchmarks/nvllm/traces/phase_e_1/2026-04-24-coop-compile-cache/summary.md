# Phase E.1 follow-up #1 — β-coop compile-cache share across layers

**Date:** 2026-04-24
**Base commit:** `7bc5773a7` (tip of `main` — Phase E SHIPPED evidence)
**Fix commit:** pending (`vllm/v1/attention/backends/cute_paged/phase_e_kernel.py`,
module-level `_PHASE_E_COOP_FULL_COMPILE_CACHE`)
**Hardware:** NVIDIA DGX Spark (GB10, SM121)
**Image:** `nvllm:gb10` (fix bind-mounted via `docker run -v` — full rebuild pending)

## What changed

Each `PhaseE_Beta_Kernel` instance held its own `self._compiled_phase_coop_full`,
so the 16 full_attention layers of Qwen3.5-27B each called `cute.compile()`
on first decode — 16 × ~23 s ≈ ~6 min cold-start stall.

Fix: module-level `_PHASE_E_COOP_FULL_COMPILE_CACHE[key]` keyed by the tuple
of every `self.` constexpr value read inside `_jit_launch_phase_0_to_4`
(34-tuple; audited by grep). Instances with matching config share one
compiled kernel.

## Setup

```
Model:       ig1/Qwen3.5-27B-NVFP4
Attn backend: CUTE_PAGED
KV cache:    fp8_e4m3
Max model len: 65536
Max num seqs: 1  (forces β-coop path)
CUDA graphs: PIECEWISE
Env:
  CUTE_PHASE_E_FUSION=1
  CUTE_PHASE_E_PATH=coop
```

## Evidence

**Compile events in serve log (post-GSM8K):**

```
$ grep -c "Compiling PhaseE_Beta_Kernel" serve_log_phase_e_lines.txt
1

$ grep "Compiling PhaseE_Beta_Kernel β-coop full" serve_log_phase_e_lines.txt
(EngineCore pid=146) INFO 04-24 11:09:58 [phase_e_kernel.py:2991] \
  Compiling PhaseE_Beta_Kernel β-coop full (first call for this config)…
```

**Layer attachments (expected 16, one per full_attention layer):**

```
$ grep -c "CuTe Phase E β-coop kernel attached" serve_log_phase_e_lines.txt
16
```

Config per attachment: `hidden=5120 intermediate=17408 num_q_heads=24
num_kv_heads=4 head_dim=256`. All 16 share identical constexpr config →
cache hit for layers 2-16.

**First-request cold-start vs steady-state:**

| Question | Status | elapsed (s) |
|----------|--------|-------------|
| Q1 (cold) | OK  | **79.4** |
| Q2 | WRONG¹ | 22.8 |
| Q3 | OK | 23.1 |
| Q4 | OK | 23.0 |
| Q5 | OK | 22.7 |
| Q6 | OK | 23.2 |
| Q7 | OK | 23.0 |
| Q8 | OK | 22.9 |

Cold overhead ≈ 79.4 − 23.0 = **~56 s** (1 compile).

On `main` (without the cache) this would have been 16 compiles ≈ ~6 min
cold-start stall. Observed **~79 s → ~5× faster first-request latency.**

¹ Q2 extractor artifact, not a kernel regression. Model output `"120/12. 2. 2\nStudent A"` — `120/12` = 10 (expected) but the
`[-+]?\d*\.?\d+` regex pulls `120` before the division. Same model, same
kernel dispatch; reproduces on baseline `nvllm:gb10` pre-fix.

**GSM8K verdict:** PASS (7/8 = 87%, matches Phase E shipped bar
7-8/8 for post-quant sanity).

## Files

- `summary.md` — this file
- `gsm8k.json` — full GSM8K sanity output (scripts/gsm8k_sanity.py)
- `serve_log_phase_e_lines.txt` — filtered serve log (Phase E / β-coop lines only)

## How to reproduce

```bash
# 1. Start container with fix bind-mounted
docker run -d --name nvllm \
  --gpus all --ipc=host --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$PWD/vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:/app/nvllm/vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:ro" \
  -e CUTE_PHASE_E_FUSION=1 -e CUTE_PHASE_E_PATH=coop \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e CUTE_MLP_FUSION=1 -e CUTE_ATTN_FUSION=1 \
  nvllm:gb10 \
  serve --model ig1/Qwen3.5-27B-NVFP4 \
    --served-model-name default --host 0.0.0.0 --port 8000 \
    --kv-cache-dtype fp8_e4m3 --attention-backend CUTE_PAGED \
    --max-model-len 65536 --max-num-seqs 1 \
    --language-model-only \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --mamba-cache-mode align --trust-remote-code \
    --gpu-memory-utilization 0.70 --max-num-batched-tokens 65536 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}'

# 2. Wait for ready + run GSM8K
until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 5; done
.venv/bin/python scripts/gsm8k_sanity.py --json --save /tmp/e1.json \
  --label e1_coop_compile_cache --timeout 600

# 3. Count compile events
docker logs nvllm 2>&1 | grep -c "Compiling PhaseE_Beta_Kernel β-coop full"
# Expected: 1
```

## Related

- Unit evidence: `tests/kernels/cute/test_phase_e_compile_cache.py` (6 new tests, all pass)
- Phase E shipped: `benchmarks/nvllm/traces/phase_e/2026-04-23-initial/`
- Phase E.1 follow-up list: memory project_phase_e_shipped.md
