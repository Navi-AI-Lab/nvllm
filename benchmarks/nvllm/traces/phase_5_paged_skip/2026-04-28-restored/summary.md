# Phase 5 — paged-skip optimization restored

## Verdict: ship — paged kernel does not fire during decode under route-on β-coop

- **Correctness**: ✅ Paris 256-tok coherent; GSM8K sanity 8/8 PASS (12 s/Q vs Phase 4 16 s/Q, ~25% latency drop).
- **Paged-skip in effect**: ✅ — `paged_attention_forward` does **not** appear in the kernel summary at all. β-coop fires solo on decode (only its own write to framework `output_rmsnorm` / `output_residual` / `output_mlp`).
- **β-coop coverage**: 512 launches over 32 generated tokens × 16 fusion-bound full-attention layers = exact (32 × 16 = 512). All 16 stride-4 layer indices appear (3, 7, 11, …, 63) in the per-layer NVTX rows.
- **No fallback warnings** in serve.log.
- **No `cudaErrorStreamCaptureInvalidated`**.

The Phase 4 commit (`0185f84a0`, this trace's image) had `_skip_paged = _use_beta_coop and not _framework_output_route` — which kept paged firing every full-attn layer when the route was active for writer-invariant safety. Phase 5 (uncommitted at trace time, file diff in `_backend.py`) drops the `and not _framework_output_route` clause so paged is skipped whenever β-coop will fire, with explicit paged replay in the β-coop except handler when β-coop raises (writer-invariant preserved without the every-step paged cost).

## Context

- **Commit at trace**: `0185f84a0` (Phase 4 head) + uncommitted Phase 5 working tree (one file: `vllm/v1/attention/backends/cute_paged/_backend.py`).
- **Image**: `nvllm:gb10` SHA `f626ca7c63e4`
- **Model**: `ig1/Qwen3.5-27B-NVFP4` (non-distilled)
- **Hardware**: NVIDIA DGX Spark (GB10, SM120/121), 128 GB unified
- **Backend**: `CUTE_PAGED`, `fp8_e4m3` KV cache, PIECEWISE CUDA graphs
- **Phase E flags**: `CUTE_PHASE_E_FUSION=1`, `CUTE_PHASE_E_PATH=auto` (β-coop selects when num_seqs * 64 ≤ resident_cap=96)
- **Workload**: 1 sequence × 32 generated tokens, `temperature=0` warm-up + `temperature=0` timed.
- **Profiler**: vLLM `--profiler-config '{"profiler":"torch", ..., "active_iterations":200}'`, `/start_profile` → completion → `/stop_profile`.

## Profiling methodology

Per `feedback_vllm_profiling`, nsys CUPTI cannot trace vLLM V1's spawned `EngineCore` child process; we use vLLM's torch profiler (`/start_profile` / `/stop_profile`). `record_shapes=false` to avoid CUPTI buffer OOM (per Phase E 2026-04-23 retry pattern). `with_stack=true`. Raw `*.pt.trace.json.gz` committed alongside; canonical kernel-time table is `profile_kernels.txt`.

## Kernel-duration table (from profile_kernels.txt — no rounding)

Top kernels by Self CUDA time over the 32-token timed window:

| Kernel | Self CUDA | % | # of Calls | CUDA time avg |
|---|---|---|---|---|
| `vllm::cute_beta_coop_run` (splitting op outer) | 20.258s | 85.11% | 512 | 40.455 ms |
| `kernel_cutlass__kernel_phase_0_to_4_vllmv1attentionb…` (β-coop fused) | 20.258s | 85.11% | 496 | 40.843 ms |
| `internal::gemvx…` (lm_head GEMV-style) | 1.812s | 7.61% | 4496 | 403.122 µs |
| `cutlass::Kernel2 GemmUniversal…` (NVFP4 GEMM) | 1.174s | 4.93% | 3632 | 323.178 µs |
| `aten::mm` | 501.036ms | 2.11% | 176 | 2.847 ms |
| `aten::copy_` | 438.654ms | 1.84% | 1251 | 350.685 µs |
| `cudaStreamIsCapturing` | 285.677ms | 1.20% | 2525 | 113.139 µs |
| `_C::cutlass_scaled_fp4_mm` | 48.582ms | 0.20% | 160 | 303.640 µs |
| `cudaGraphLaunch` | 41.584ms | 0.17% | 2015 | 20.637 µs |
| `vllm::gdn_attention_core` (linear attn outer) | 31.228ms | 0.13% | 1536 | 22.497 µs |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 26.820ms | 0.11% | 1488 | 18.024 µs |
| `_C_cache_ops::reshape_and_cache_flash` | 1.583ms | 0.01% | 512 | 3.092 µs |

**Notably absent: `paged_attention_forward` and any `cute_kernel_phase_*` paged variant.** That's the smoking gun for paged-skip — they are in the source code path but never called during this 32-token decode window because `_skip_paged = _use_beta_coop = True` for every full-attn layer.

### Per-layer β-coop NVTX rows (sample — full list in profile_kernels.txt)

All 16 stride-4 fusion-bound layer indices appear with ~31 calls each (matches 32-token timed window − 1 sampling offset):

| Layer | Self CUDA | # calls | CUDA time avg |
|---|---|---|---|
| `PhaseE_Beta.coop.layers.3.self_attn.attn` | 1.247s | 31 | 40.231 ms |
| `PhaseE_Beta.coop.layers.7.self_attn.attn` | 1.273s | 31 | 41.048 ms |
| `PhaseE_Beta.coop.layers.11.self_attn.attn` | 1.264s | 31 | 40.769 ms |
| `PhaseE_Beta.coop.layers.15.self_attn.attn` | 1.273s | 31 | 41.058 ms |
| `PhaseE_Beta.coop.layers.19.self_attn.attn` | 1.257s | 31 | 40.536 ms |
| `PhaseE_Beta.coop.layers.23.self_attn.attn` | 1.281s | 31 | 41.309 ms |
| … | … | … | … |
| `PhaseE_Beta.coop.layers.63.self_attn.attn` | 1.270s | 31 | 40.966 ms |

Per-layer β-coop CUDA time spread is 40.23 – 41.40 ms — consistent across layers (max-min < 3%).

## Smoking-gun analysis

In Phase 4 (proven by sanity 8/8 with paged firing every layer), the writer-invariant guarantee was paid for with paged work that β-coop immediately overwrote. With 16 full-attn layers × 32 tokens, that would have been **512 paged_attention_forward calls** in this trace window (~16 × per-call paged time). They are gone in Phase 5: the kernel name does not appear at all in the kernel-time table.

The except-path replay logic exists (in `_backend.py` β-coop except handler) but did not fire during this run — the engine logs show no β-coop exceptions and no fallback warnings, consistent with the kernel being stable for `num_seqs=1` cooperative-launch fitness on this hardware.

## Reproduction

```bash
# 1. Engine with profiler enabled (Phase 5 working tree on top of 0185f84a0)
docker rm -f nvllm 2>/dev/null
docker run -d --name nvllm --gpus all --ipc=host --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v /tmp/profiles:/tmp/profiles \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_PHASE_E_FUSION=1 -e CUTE_PHASE_E_PATH=auto \
  -e CUTE_MLP_FUSION=1 -e CUTE_ATTN_FUSION=1 \
  nvllm:gb10 serve \
  --model ig1/Qwen3.5-27B-NVFP4 --served-model-name default \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype fp8_e4m3 --attention-backend CUTE_PAGED \
  --max-model-len 65536 --max-num-seqs 1 \
  --language-model-only --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --mamba-cache-mode align --trust-remote-code \
  --gpu-memory-utilization 0.70 --max-num-batched-tokens 65536 \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":true,"active_iterations":200,"torch_profiler_record_shapes":false,"torch_profiler_with_stack":true}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'

# 2. Wait until /v1/models responds, then warm-up
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"The capital of France is ","max_tokens":16,"temperature":0.0}' >/dev/null

# 3. Profile a 32-token decode
curl -X POST http://localhost:8000/start_profile
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"The capital of France is ","max_tokens":32,"temperature":0.0}' >/dev/null
curl -X POST http://localhost:8000/stop_profile

# 4. Trace lands in /tmp/profiles/
ls /tmp/profiles/
```

## Files in this bundle

- `summary.md` — this document
- `profile_kernels.txt` — vLLM-emitted kernel summary (raw stdout from torch profiler `key_averages` print)
- `rank0.pt.trace.json.gz` — full Chrome trace from EngineCore (~21 MB compressed)
- `async_llm.pt.trace.json.gz` — async_llm process trace (~414 KB)
- `serve.log` — serve container log

## Cross-references

- Phase 4 commit (paged double-fired): `0185f84a0`
- Phase 5 commit (paged-skip restored): `<this commit>`
- Phase E shipped (per-layer β-coop microbenchmark): `benchmarks/nvllm/traces/phase_e/2026-04-23-initial/summary.md`
