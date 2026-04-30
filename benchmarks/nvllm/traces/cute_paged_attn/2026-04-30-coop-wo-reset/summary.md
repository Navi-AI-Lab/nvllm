# v2 — Gate 1 nsys trace (β-coop wo_output captured reset)

- **Commit hash**: 52770bb8ac4d8ee3b7f05e5e1f57f1c8334f1b02
- **Model**: ig1/Qwen3.5-27B-NVFP4
- **Config**: cudagraph_mode=FULL_AND_PIECEWISE, β-coop on layers
  3,7,11,15,19,23,27,31, kv_cache=fp8_e4m3, max_model_len=16384,
  max_num_seqs=1, capture_sizes=[1].
- **Trace file**: `changed.nsys-rep` (82.9 MB)
- **Gate 1 verdict (separate evidence dir)**: FAIL — see
  `docs/research/2026-04-29-full-graph-spike/evidence/2026-04-30-1805/c2_replay_coherence.md`

## Activity counts (nsys export → sqlite)

- CUPTI_ACTIVITY_KIND_MEMSET rows: 166
- CUPTI_ACTIVITY_KIND_KERNEL rows: 2946
- Distinct globalPid in the trace: 1 (API server pid=124)
- Rows with `graphNodeId IS NOT NULL`: 0 (memset and kernel both)
- nsys cuda_api_sum cudaMemsetAsync host calls: 208 (0.0% of total)

## Honest limitation — EngineCore subprocess not captured

Per project memory `feedback_vllm_profiling`: vLLM V1 spawns model
work into an `EngineCore` subprocess (pid=327 in the captured run,
visible in `server.log`). nsys with default settings followed the
nsys-launcher → API-server PID tree but did **not** cross into the
EngineCore worker. As a result the trace contains **only host-side
sampling and post-processing kernels** from the API server PID; the
β-coop kernels and the captured `cudaMemsetAsync` for `wo_output`
buffers (which run inside the EngineCore worker) are absent.

Concrete signals in this trace that confirm this:
- 0 rows have `graphNodeId IS NOT NULL` → no captured-graph events
  from this PID; replay never crossed the recorded tree.
- The 166 memsets are `bytes=4` scalar fills (sampling helpers),
  not the expected `nat * 4 * hidden * 4 = 81920 B` reset pattern.
- Top kernels are `at::native::vectorized_elementwise_kernel<...
  FillFunctor>` and `cub::DeviceSelectSweepKernel` (sampling
  pipeline), not `PhaseE_Beta.coop` / paged-attn entry kernels.

## What is preserved as evidence

- `changed.nsys-rep` — the raw trace, committed for forensic
  inspection in nsys-ui.
- `gpu_activity_ordering.txt` — first 200 GPU activities ordered by
  start time.
- `server.log` — full server stdout/stderr, including EngineCore
  pid=327 mentions confirming where the model work actually runs.

## Follow-up needed for graph-node ordering proof

Capturing EngineCore kernels requires either nsys's child-tree
follow option or vLLM's torch profiler API hooks (per
`feedback_vllm_profiling`). That promotion is out of scope for this
Phase 6 dispatch but should be tracked as the next nsys arc.

## How to reproduce this trace

See `docs/superpowers/plans/2026-04-30-beta-coop-persistent-buffers-v2-plan.md`
Task 11. Briefly:

```bash
docker run -d --name nvllm --gpus all --ipc=host --network host --privileged \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/flashinfer:/root/.cache/flashinfer \
  -v /tmp/nvllm-cute-cache:/opt/vllm/kernel_cache \
  -v $(pwd)/benchmarks/nvllm/traces/cute_paged_attn/2026-04-30-coop-wo-reset:/traces \
  -v /opt/nvidia/nsight-systems/2025.6.3:/opt/nvidia/nsight-systems/2025.6.3:ro \
  -e CUTE_PHASE_E_FUSION=1 \
  -e CUTE_PHASE_E_LAYERS='3,7,11,15,19,23,27,31' \
  -e CUTE_PHASE_E_FALLBACK_RAISE=1 \
  -e CUTE_FULL_GRAPH_PROBE=1 \
  ... [rest of nvllm:gb10 env per plan]
  --entrypoint /bin/bash nvllm:gb10 -c 'sleep 7200'

bash docs/research/2026-04-29-full-graph-spike/_sync_host_edits.sh

docker exec -d nvllm bash -lc '
  /opt/nvidia/nsight-systems/2025.6.3/bin/nsys profile --trace=cuda,nvtx \
    --output=/traces/changed --force-overwrite=true \
    /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
      --model ig1/Qwen3.5-27B-NVFP4 ... \
      --compilation-config '"'"'{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1]}'"'"' \
      > /traces/server.log 2>&1
'
# Wait /v1/models, send 4 decode requests, pkill -INT vllm.entrypoints to flush.
```
