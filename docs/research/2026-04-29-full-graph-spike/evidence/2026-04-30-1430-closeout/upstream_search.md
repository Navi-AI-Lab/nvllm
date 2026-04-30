# Upstream search 2026-04-30 — FULL graph + β-coop nondet

## vllm-project/vllm

### Direct analogs (HIGH RELEVANCE — same failure mode as ours)

- #35175 MERGED — [Bugfix] Restore CUDA graph persistent buffers for FP8 FlashMLA decode — https://github.com/vllm-project/vllm/pull/35175
  why: NEAR-PERFECT ANALOG. FP8 path called `get_mla_metadata_dense_fp8()` allocating fresh raw tensors every call; under FULL_AND_PIECEWISE the graph captured tensor addresses, replay read stale metadata from originally-captured addresses → "garbled output that starts normal then degenerates after ~50 tokens." Cumulative divergence pattern matches our "first 8 chars stable, decode tokens 2+ diverge." Fix = copy fresh allocations into pre-allocated persistent CUDA-graph buffers. Pattern to replicate: pre-allocate ALL β-coop scratch/counters/partials in `__init__`, copy-in before the cooperative launch, never alloc inside the captured op.

- #37363 OPEN — fix(compilation): fix piecewise CUDA graph bugs with splitting_ops — https://github.com/vllm-project/vllm/issues/37363 (PR #37361)
  why: DIRECT MECHANISM MATCH. "When a `splitting_op` allocates new tensors, the next piece's CUDA graph replays with stale addresses → silent data corruption." β-coop is registered as a splitting_op via direct_register_custom_op and mutates `output_rmsnorm/output_residual/output_mlp`. PR #37361 saves input tensor refs at capture time (`input_buffers`) and copies new data into them before replay. Suggests our PIECEWISE-pass case may share the same root cause; FULL has same surface area.

- #36042 MERGED — Fix CUDA graph decode capture crash in AITER FlashAttention — https://github.com/vllm-project/vllm/pull/36042
  why: "`unified_attention` is not CUDA-graph-capture-safe because it performs dynamic memory allocations (`torch.empty`) and runtime kernel selection inside the wrapper. This causes `Memory access fault by GPU node-X on address (nil)` at 91% of decode FULL graph capture." Same lesson: NO alloc/branch inside a captured op. Already in our notes (`feedback_no_self_mut_in_cudagraph_dispatch`). Verifies upstream consensus that runtime branching kills FULL capture.

- #40969 OPEN — [Bug]: DeepSeek-V4-Flash hangs after ~6 requests with cudagraph_mode=FULL_AND_PIECEWISE + chunked prefill on SM 12.x (GB10) — https://github.com/vllm-project/vllm/issues/40969
  why: SAME HARDWARE (GB10/SM12.1), SAME cudagraph_mode (FULL_AND_PIECEWISE), SAME failure shape (works initially then hangs/diverges). Reporter notes `PIECEWISE` workaround is stable while `FULL_AND_PIECEWISE` hangs at request ~6-7 with 100% SM but zero token output — exactly matches our intermittent 40-min hang at "first FULL probe fired." Suspect = "metadata builder produces inconsistent state between capture and replay" or "dispatcher fails to detect mixed batches and replays a wrong-shape FULL graph."

### Adjacent / supporting evidence

- #41331 OPEN — [Bug]: Garbled Output in DeepSeek-V4 with CUDA Graph Enabled Under Concurrent Identical Input Requests — https://github.com/vllm-project/vllm/issues/41331
  why: "When `cudagraph_mode` is set to `FULL_DECODE_ONLY`, some requests produce garbled output, while single-request inference works fine." Concurrency-conditional FULL graph corruption — shows there's a class of bugs where FULL replay state is shared/aliased across requests when it shouldn't be. Our 8-replay nondet test would catch the same class.

- #35659 OPEN — [Bug]: cudaErrorIllegalAddress under sustained parallel load with CUDA Graphs on Blackwell SM120 (NVFP4 MoE) — https://github.com/vllm-project/vllm/issues/35659
  why: SM120 + CUDA graphs + sustained replay → memory corruption "accumulates over time during CUDA graph replays." Our cumulative layer-count threshold (1 layer PASS, 8 layers FAIL) tracks the same accumulation pattern.

- #22945 OPEN — [Feature][CUDAGraph]: Audit CUDAGraph support in attention backends — https://github.com/vllm-project/vllm/issues/22945
  why: Tracking issue for `_cudagraph_support` levels (UNIFORM_SINGLE_TOKEN_DECODE / UNIFORM_BATCH / ALWAYS). Reference for what level our CuTe paged-attn backend should declare. Our β-coop currently uses splitting_op routing which bypasses this gate.

- #40742 OPEN — [Bug]: CUDA graph capture crashes during startup due to Inductor autotuning torch.cuda.synchronize() inside graph capture (FULL_DECODE_ONLY + MLA + FP8) when PDL is enabled — https://github.com/vllm-project/vllm/issues/40742
  why: Crash happens because something inside the captured region calls `torch.cuda.synchronize()`. Reinforces our `feedback_item_breaks_cuda_graphs` rule: any host-device sync in the op body is fatal. Worth auditing β-coop body for hidden syncs (check `.item()`, `.to('cpu')`, `tensor.numpy()`).

- #38123 MERGED — [compile] Allow strings in custom ops without regressing compilation times — https://github.com/vllm-project/vllm/pull/38123
  why: torch.compile + custom-op interaction has known compile-time regressions; potentially relevant if our 40-min hang is `cute.compile` re-firing under capture (cache-miss path). Side note, low priority.

- #34880 OPEN — [Spec Decode][CUDA Graphs] Enables Eagle drafter support for FULL CUDA Graph mode — https://github.com/vllm-project/vllm/pull/34880
  why: Working example of adding a new module to FULL graph capture flow. Reference for what compile/dispatch hooks must be implemented.

- #41285 OPEN — [Model Runner v2] Fix v2 compile counter `num_gpu_runner_capture_triggers` and `num_cudagraph_captured` — https://github.com/vllm-project/vllm/pull/41285
  why: Counter-based cudagraph capture diagnostics — relevant if we want to instrument our flaky 40-min hang ("did capture even start? did it finish?").

- #26678 CLOSED — [Bug]: use_inductor_partition + splitting_ops results in AssertionError — https://github.com/vllm-project/vllm/issues/26678
  why: PyTorch issue: `torch.library.custom_op` with `mutates_args=("output",)` does NOT correctly set `origin_node` in `torch._inductor/graph.py:1865-1881`, breaking inductor graph_partition. We use `mutates_args=["output_rmsnorm","output_residual","output_mlp"]` — exactly the same construct. Smoking gun for "mutates_args + graph capture is fragile."

## pytorch/pytorch

(Searches for "torch.cuda.graph custom_op", "cudagraph functionalize mutates_args", "torch.cuda.graph stale buffer" all returned 0 results. The cross-reference in vLLM #26678 above points at `torch/_inductor/graph.py:1865-1881` `origin_node` bug as the underlying torch-side defect for `mutates_args` ops in graph partition. cc'd reviewers @ProExpertProg @zou3519 @BoyuanFeng.)

## NVIDIA/cutlass + cute-dsl

(Both `gh search issues --repo NVIDIA/cute-dsl ...` calls returned "resources do not exist or you do not have permission" — cute-dsl is not a public GitHub repo or it's hosted under a different name. NVIDIA/cutlass searches for "cooperative cuda graph" / "cute.compile cuda graph" returned 0 hits — no upstream-known issue at the cutlass/cute layer. Our problem is most likely above the cutlass layer, in vllm dispatch / capture-time alloc, not inside the kernel.)

## Notable patterns / quotes

- "graph captures tensor addresses during recording. On replay, freshly-allocated tensors live at different addresses, so the kernel reads stale metadata from the originally-captured addresses, producing garbled output that starts normal then degenerates after ~50 tokens." (#35175 — VERBATIM analog of our "first 8 chars stable, decode tokens 2+ diverge")
- "When a `splitting_op` allocates new tensors (e.g. via `torch.bmm`), the next piece's CUDA graph replays with stale addresses → silent data corruption." (#37363)
- "`unified_attention` is not CUDA-graph-capture-safe because it performs dynamic memory allocations (`torch.empty`) and runtime kernel selection inside the wrapper." (#36042)
- "metadata builder produces inconsistent state between capture and replay" — V4 indexer top-k buffers (#40969)
- "memory corruption accumulates over time during CUDA graph replays" (#35659) — cumulative threshold matches our 1-layer-OK, 8-layer-FAIL pattern
- "vllm.unified_attention_with_output, which has an inplace-mutation, which results in [graph.py L1865-1881] not properly setting `origin_node`" (#26678) — confirms `mutates_args` + graph partition is fragile in pytorch core

## Cherry-pick candidates (merged, applicable)

1. **#35175 MERGED** — pattern: pre-allocate persistent buffers in `__init__`, copy-in fresh metadata before launch. Direct template for β-coop scratch/counters/partials. **STRONGEST CANDIDATE — cherry-pick the pattern, not the file.**
2. **#36042 MERGED** — pattern: remove all dynamic alloc / runtime branching from the captured op body. Audit β-coop wrapper for any `torch.empty`, `.contiguous()`, conditional kernel selection.
3. **#37361 CLOSED** (open issue #37363) — pattern: at capture time, save input refs in `input_buffers`, copy new data in before each replay. Applicable if our root cause is splitting_op input-address aliasing (not yet confirmed).

## Top hypothesis after upstream review

Our symptom = #35175 in CuTe form. β-coop body almost certainly performs an alloc (or holds a reference to a tensor the graph allocator pool can recycle) inside the captured region, OR returns a reference to a workspace tensor whose backing memory the graph pool reuses across replays. Cumulative-with-layer-count failure (1L PASS, 8L FAIL) matches "stale metadata accumulates as more layers reuse the same recycled buffer." First-token-stable / decode-tokens-2+-diverge matches "first replay reads fresh-from-warmup state, second replay onward reads stale graph-pool addresses."

Action: audit β-coop op body line-by-line for (a) any `torch.empty/zeros/ones`, (b) any tensor returned/aliased that didn't come from `mutates_args` outputs, (c) any workspace tensor allocated in the wrapper rather than registered as a persistent buffer at module init.
