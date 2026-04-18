# Phase D1 — custom-op wrap for fused MLP, trace summary

## Verdict: ship gate FAILS — dual-firing persists despite the custom-op wrap

- **Correctness**: ✅ GSM8K 8/8 on both modes.
- **No-regression gate** (≤ 2%): ❌ **2.66× slower** with fusion on (16.7 s/question vs 6.3 s/question).
- **Dual-firing eliminated**: ❌ — `gate_up_proj` + `down_proj` NVFP4 GEMMs still fire in the 16 fused full_attention layers, in addition to the Phase D fused kernel.
- **Architecture**: custom op registers and is visible to Inductor (appears in a generated Triton fusion kernel name), but the compiled graph still contains the unfused-math path.

**Conclusion**: Phase D1 did NOT resolve the torch.compile dead-branch issue. The root cause is different from what the Phase D1 hypothesis anticipated. Do not cap Phase D. Scope Phase D2 for deeper investigation.

## Context

- **Commit**: `f0cdaa8f2` + uncommitted Phase D1 working tree (custom op `_mlp_op.py` + updated `attach_mlp_fusion` + updated `Qwen3_5MLP.forward` + updated decoder call site).
- **Image**: `nvllm:gb10-phaseD1`
- **Model**: `natfii/Qwen3.5-27B-NVFP4-Opus-GB10`
- **Config**: `max-model-len=65536`, `max-num-seqs=4`, `kv-cache-dtype=fp8_e4m3`, CUDA graphs `PIECEWISE`, `--language-model-only`, `--gpu-memory-utilization 0.80`.
- **Workload**: 4 concurrent × 128 tok, `temperature=0`, `ignore_eos=true`.

## Raw numbers

| Metric | Baseline (`CUTE_MLP_FUSION=0`) | Changed (`CUTE_MLP_FUSION=1`) | Delta |
|---|---|---|---|
| GSM8K | 8/8, 6.1–6.6 s/Q | 8/8, 16.5–17.0 s/Q | +160 % slower |
| Self CUDA time total | 43.783 s | 247.346 s | +464 % |
| Attention CuTe kernel | 34.344 s / 2032 / 16.901 ms avg | 34.496 s / 2032 / 16.977 ms avg | ≈ 0 |
| Phase D fused MLP CuTe kernel | — | **203.348 s / 2032 / 100.073 ms avg** | **new** |
| General NVFP4 GEMM (GemmUniversal) | 7.785 s / 38608 | 7.828 s / 38608 | ≈ 0 |
| `vllm::silu_mul_cvt_fp16_to_fp4` | 12.972 ms / 8128 | 9.736 ms / 6096 | −25 % |
| `triton_red_fused__to_copy_add_cute_mlp_forward_mean_...` | — | 4.217 ms / 2032 | new (Inductor fused cute_mlp_forward with surrounding RMSNorm) |

Delta breakdown:
- Attention kernel time: unchanged (≈ 34 s both modes).
- Phase D fused MLP kernel time (new in changed): +203.3 s.
- Every other kernel's count and time: nearly unchanged.
- **Changed minus baseline ≈ Phase D kernel time** (247 − 44 ≈ 203 s).

That last line is the smoking gun: the changed run does everything the baseline run does, **plus** the Phase D fused kernel work. gate_up/down_proj GEMMs were not eliminated.

## Dual-firing evidence

Fusion attached successfully for all 16 full_attention layers (every 4th: 3, 7, 11, …, 63). Log from `changed/decode_log.txt`:

```
CuTe MLP fusion attached: layer=model.layers.3  hidden=5120 interm=17408 num_k_tiles=8 tile_s=256 tile_k=640 slice_ctas=8 op_key=model.layers.3.mlp
CuTe MLP fusion attached: layer=model.layers.7  ... op_key=model.layers.7.mlp
...
CuTe MLP fusion attached: layer=model.layers.63 ... op_key=model.layers.63.mlp
```

Custom op registration and attach-side wiring both succeeded. `_cute_layer_name` is set on all 16 MLP modules. `_CUTE_MLP_REGISTRY` has all 16 entries.

Yet:

- General NVFP4 GEMM kernel call count is **identical** between baseline (38608) and changed (38608). These calls include the full_attention layers' `qkv_proj`, `o_proj`, **`gate_up_proj`**, and **`down_proj`**. If the custom op had prevented MLP dual-firing, we'd expect 16 × 2 × 127 = 4064 fewer MLP GEMM calls in changed.
- The Phase D fused MLP kernel also fires 2032 times (16 layers × 127 steps) at 100 ms/call → 203.3 s total.
- Both paths coexist in the compiled graph.

## Hypotheses for why the custom-op wrap did not eliminate dual-firing

The Phase D1 hypothesis was: replacing the Python-level `if _mlp_fusion_active` branch with `if _cute_layer_name is None` + an opaque `torch.ops.vllm.cute_mlp_forward` call would prevent Dynamo from dead-branching the fused path. Evidence shows this didn't happen. Possible mechanisms:

1. **`_cute_layer_name` check is still a Python-level attribute gate.** The forward has `if layer_name is None:` still baked in. If Dynamo traced this branch when `_cute_layer_name` was None (e.g., the first compile pass happened before `attach_mlp_fusion` ran, or for a different model instance), the specialized graph locks in the unfused body.
2. **Inductor inlined/decomposed cute_mlp_forward.** The Triton kernel name `triton_red_fused__to_copy_add_cute_mlp_forward_mean_...` shows Inductor fused *around* cute_mlp_forward, which implies it saw the op at graph-level. But its decomposition may have included the fallback branch as well (the fallback calls `_mlp_gate_up_proj`, `_mlp_act_fn`, `_mlp_down_proj` — all of which are themselves registered ops that Inductor can trace).
3. **The pre-D1 attention-side side-effect Phase D kernel launch is still in place** (`_backend.py::forward` launches `self._mlp_kernel(...)` unconditionally when fusion is bound + active). If the compiled graph also emits the unfused GEMMs, we get both.

(1) and (2) are mutually reinforcing. Real fix almost certainly needs to:
- Eliminate the `if layer_name is None:` Python branch from `Qwen3_5MLP.forward` entirely (make it truly unconditional), OR
- Register cute_mlp_forward with explicit non-composable semantics so Inductor can't decompose it, OR
- Stop launching the Phase D kernel as a side effect from attention — move the kernel launch *into* the cute_mlp_forward op body so there's a single place the kernel fires.

The third approach is architecturally cleanest and worth pursuing in Phase D2.

## Kernel performance (secondary concern)

Even if dual-firing were eliminated, the Phase D kernel would still need to beat the unfused sequence:

- Unfused MLP per decode step (16 layers × 4 tokens): negligible — subsumed in baseline's ~10 s non-attention time.
- Phase D kernel per decode step (16 layers × 4 tokens): 100 ms per layer launch × 16 = 1.6 s per step. With 127 steps → 203 s total.

A 100 ms kernel for a 4-token × 5120-hidden × 17408-intermediate decode MLP is **2-3 orders of magnitude** too slow. Expected for SM121: microseconds for the MLP at decode size. Likely causes:

- `tile_s=256`, `tile_k=640`, `slice_ctas=8` — the kernel is tiled for prefill-sized work, not 4-token decode. Most CTAs are idle or double-scanning.
- No SMEM pipelining (single-buffer per original plan §Phase 8). Decoder loads dominate the kernel time.

Phase D2 kernel-tuning scope (separate from the dual-firing fix):
- Adapt tile sizes to actual decode batch (`tile_s ≤ num_actual_tokens`).
- Add ping/pong SMEM pipelining (originally Phase 8 Task 25).
- Profile per-CTA occupancy and L2 hit rate.

## Ship gate

| Gate | Target | Baseline | Changed | Status |
|---|---|---|---|---|
| GSM8K correctness | 8/8 | 8/8 | 8/8 | ✅ |
| Single firing (dual-firing eliminated) | gate_up/down GEMMs drop for fused layers | 38608 calls | 38608 calls | ❌ |
| Wall-clock regression | ≤ 2 % | — | +164 % (GSM8K avg time) / +464 % (CUDA time) | ❌ |
| Trace pair + summary committed | baseline + changed + summary.md | | | ✅ |

**Phase D stays open**. Phase D1 commit is a partial architectural improvement (custom op lands, registry works, Inductor recognizes the op) but the dual-firing problem is unresolved and the fused kernel itself is undertuned.

## How to reproduce

```bash
cd /home/natfii/docker/nvllm
# Baseline (fusion disabled)
./scripts/phase_d_trace_capture.sh baseline
# Changed (fusion enabled)
./scripts/phase_d_trace_capture.sh changed
```

Script is self-contained. Uses `nvllm:gb10-phaseD1` (override via `NVLLM_IMAGE=`).

## Artifacts in this directory

```
baseline/
  decode_log.txt                    # full vLLM server log for CUTE_MLP_FUSION=0 run
  profiler_out_0.txt                # kernel-level profile (Self CUDA time total: 43.783 s)
  rank0.*.pt.trace.json.gz          # pytorch profiler trace, full event tree
  gsm8k_baseline.{json,log}         # 8/8 pass
  workload_{1..4}.json              # raw completion responses
changed/
  decode_log.txt
  profiler_out_0.txt                # Self CUDA time total: 247.346 s
  rank0.*.pt.trace.json.gz
  gsm8k_changed.{json,log}          # 8/8 pass
  workload_{1..4}.json
```

## Recommended next steps (Phase D2 scope)

1. **Fix dual-firing first** — architectural, before any perf work.
    - Move Phase D kernel launch from `_backend.py::forward` (attention-side side effect) into `cute_mlp_forward` op body. One kernel, one caller.
    - Make `Qwen3_5MLP.forward` truly unconditional for the fusion-attached case: avoid the `if layer_name is None:` Python branch altogether by setting a sentinel layer_name ("" or similar) that the op body treats as "skip".
    - Dump the compiled FX graph (`TORCH_LOGS="+inductor,output_code"`) to confirm the unfused path is *actually* absent before re-running traces.
2. **Re-run trace pair** once dual-firing is gone. Expected: changed total ≈ 34 s attention + 1–5 s Phase D (if kernel is reasonably tuned) = ~40 s. If it's still > 50 s, move to kernel tuning.
3. **Kernel tuning** (original Phase 8 Task 25 scope) — only if (1) + (2) clear and we're still regressing.
