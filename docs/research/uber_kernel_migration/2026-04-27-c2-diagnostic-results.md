# C2 Diagnostic Results — 2026-04-27

**Branch:** `diag/c2-beta-coop-vs-legacy` off `feat/uber-kernel-migration` HEAD `788697bff`.
**Plan:** [`2026-04-26-c2-diagnostic-plan.md`](2026-04-26-c2-diagnostic-plan.md).
**Spec:** [`2026-04-26-c2-diagnostic-spec.md`](2026-04-26-c2-diagnostic-spec.md).

---

## TL;DR

The C2 diagnostic was built to compare β-coop's outputs against the legacy post-attn-LN outputs in dual-fire under PIECEWISE+graphs and halt on first divergence. Plumbing now works end-to-end (six rebuilds total resolved four architectural blockers), but the diagnostic has a **fundamental architectural limit** that the spec did not anticipate: under PIECEWISE+graphs, the custom-op's Python body executes only once during graph capture (where it correctly skips to avoid `cudaErrorStreamCaptureInvalidated`), and never during steady-state decode replay. As a result, the diag cannot observe β-coop's actual decode-time outputs. **Decision: accept this limit, use `CUTE_DUMP_TENSORS=1` for offline forensics, and proceed to fix the residual_in plumbing bug from `project_phase_e_beta_math_bug` directly.**

The 256-token completion **did** confirm β-coop is broken at decode time: output was `这种现象` × 256 (Chinese-character spam). Already-known bug, hypothesis intact.

---

## Verdict distribution captured

| Configuration | `[C2_DIAG]` lines | Outcome |
|---|---|---|
| `CUTE_C2_DIAG=1` (default `MAX_NUM_SEQS=4`, `CUTE_PHASE_E_FUSION=0`) | 1 | Halt at L=3 step=0 nat=4: legacy hidden L∞=3.12, residual L∞=30.0, rel_med=1.0. **β buffers all-zero** — β-coop never fired (FUSION=0). |
| `CUTE_PHASE_E_FUSION=1 CUTE_C2_DIAG=1` (`MAX_NUM_SEQS=4`) | 1 | Same halt. β-coop attached but `64*num_seqs=256 > resident_cap=96` — cooperative-fitness gate failed at every capture size ≥ 2. |
| `MAX_NUM_SEQS=1 CUTE_PHASE_E_FUSION=1 CUTE_C2_DIAG=1` | 1 | Same halt at nat=1. β buffers still all-zero — β-coop didn't fire during the profiling/prefill pass before the diag fired. |
| `MAX_NUM_SEQS=1 CUTE_PHASE_E_FUSION=1 CUTE_C2_DIAG=1` (with `_phase_e_consumed` gate added) | 0 during warmup, 0 during 256-token decode | Engine came up clean. Decode produced `这种现象` × 256 (gibberish). Diag never logged a single comparison despite β-coop firing on every decode token. |

Final-config dump bundle (`/tmp/c2_diag/layer3_step0.pt`, captured under run #2 with the dump-dir patch):
- `legacy_hidden`: shape `(1, 5120)`, real values, range `[-3.12, 2.64]`.
- `legacy_residual`: shape `(1, 5120)`, real values, range `[-6.44, 30.0]`.
- `beta_rmsnorm_output`: **all-zero** — β-coop did not write.
- `beta_residual_output`: **all-zero** — β-coop did not write.

---

## Architectural limit: why the diag silently fails under PIECEWISE+graphs

Under vLLM V1's `VLLM_COMPILE` mode with `cudagraph_mode=PIECEWISE`, each segment between attention boundaries is captured as a CUDA graph. A `direct_register_custom_op` (the only viable Dynamo-compatible registration per `feedback_dynamo_disable_fullgraph`) is treated by Inductor as an opaque op whose Python body runs **once at graph-capture time** to record the CUDA ops it launches; subsequent invocations replay only the recorded CUDA op sequence — the Python body does not re-run.

Our `_cute_c2_diag_compare_impl` body has two unavoidable runtime checks:

1. **Capture-skip** (`torch.cuda.is_current_stream_capturing() → return`): required because `compare_and_log` calls `.item()` on `L∞` / `rel_med` to format log lines and decide halt-vs-continue. Host-device sync inside an op body during graph capture raises `cudaErrorStreamCaptureInvalidated`. See `feedback_item_breaks_cuda_graphs`.
2. **Prefill-skip** (`nat > beta.shape[0] → return`): impl buffers are sized at `max_num_seqs` (decode-only); during prefill warmup `nat` can reach `max_model_len=65536`. Subtract on mismatched shapes raises.

Both guards are correct in isolation. Combined with capture-time-only Python execution they leave **zero windows** in which the op can actually fire on a real β-coop decode step:

| Phase | Stream-capturing? | β-coop fires? | Diag op body runs? | Useful comparison? |
|---|---|---|---|---|
| Profiling (eager prefill) | No | No (prefill, `is_decode_only=False`) | Yes — but prefill-skip triggers | No |
| Eager warmup decode iter | No | Sometimes (gate-dependent) | Yes — fires anyway | Only if β-coop fitness gate also passed |
| Graph-capture warmup | **Yes** | Yes (size-dependent) | Yes — but capture-skip triggers | No |
| Steady-state decode replay | No | Yes | **No — body doesn't re-run** | **No** |

The eager-warmup-decode window is the only window where the diag has any chance of firing with real β data — and it requires the diag's gate to match β-coop's own multi-clause fire gate exactly. Mismatch → spurious DIVERGED. Match (via `_phase_e_consumed`) → trace-time DCE under fullgraph → zero firings.

---

## Failed gate-tightening attempt

Adding `getattr(impl, "_phase_e_consumed", False)` to the diag's outer Python `if` was attempted (see commit history pre-squash). Goal: skip the diag when β-coop didn't actually write the buffers this step. Result: under `torch.compile` fullgraph, `_phase_e_consumed` is False at the trace-time evaluation of the `if`, so Dynamo dead-codes the entire `if` body — the `torch.ops.vllm.cute_c2_diag_compare` call never enters the graph at all. The 256-token decode emitted **zero** `[C2_DIAG]` lines despite β-coop being demonstrably broken. The gate change was reverted.

This is the same trap as the consume-gate DCE problem (`feedback_opaque_op_not_enough`): Python `if` on impl-attributes is not safe under fullgraph compile when the attribute can change at runtime — the *value at trace time* is what gets baked.

---

## Plumbing wins (kept across the squash)

Despite the architectural limit, six concrete blockers were resolved and four lessons were saved:

1. **vLLM EngineCore env stripping** — `feedback_vllm_enginecore_env_strip`. Most `docker -e` vars don't reach pid 146; serve-cute writes `/tmp/c2_diag/ENV`, qwen3_5.py prelude sources at module import.
2. **`os.getenv(name, default)` set-but-empty trap** — `feedback_getenv_empty_string`. Use `os.getenv(name) or default` everywhere.
3. **`@torch._dynamo.disable` and `allow_in_graph` both rejected under fullgraph** — `feedback_dynamo_disable_fullgraph`. Only `direct_register_custom_op` with explicit `fake_impl` works.
4. **Custom op DCE'd if no graph op reads mutated tensor** — `feedback_mutates_args_not_dce_safe` (already documented). Diag op uses `mutates_args=()` and side-effects via Python-level logging + halt.
5. **Image bake required, not bind-mount** — `feedback_no_shortcuts`. Bind-mounting `vllm/` shadows `_C.abi3.so`. Surgical mount of single new files works but is fragile.
6. **Capture-skip + impl-buffer prefill-skip** runtime guards — both correct, both required. Documented in op impl docstring.

These are reusable for any future "diff a fused kernel against its reference path under graphs" diagnostic.

---

## Reproduction

The diagnostic itself remains in the repo, gated on `CUTE_C2_DIAG=1`. Production behavior unchanged when unset.

To reproduce the warmup verdict:

```bash
docker rm -f nvllm 2>/dev/null
rm -f /tmp/c2_diag/*.pt
CUTE_C2_DIAG=1 bash scripts/serve-cute.sh
docker logs nvllm 2>&1 | grep '\[C2_DIAG\]'  # 1 line, halt
```

To reproduce the silent-during-decode behavior:

```bash
# (Requires the _phase_e_consumed gate — see git history pre-squash for the variant.)
MAX_NUM_SEQS=1 CUTE_PHASE_E_FUSION=1 CUTE_C2_DIAG=1 bash scripts/serve-cute.sh
# wait for /v1/models
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"Tell me a story.","max_tokens":256,"temperature":0.0}'
docker logs nvllm 2>&1 | grep -c '\[C2_DIAG\]'  # 0
```

For real β-coop forensics, prefer:

```bash
CUTE_DUMP_TENSORS=1 CUTE_PHASE_E_FUSION=1 bash scripts/serve-cute.sh
# Sends a prompt; first 3 decode steps × 16 layers dump to /tmp/nvllm-dumps/
# Compare offline: residual_in vs residual_out vs reference Python path.
```

---

## Next action

Skip restructuring the C2 diag. Instead use `project_phase_e_beta_math_bug`'s existing hypothesis (β-coop's `residual_in` arg points at the post-Phase-C residual stream instead of pre-Phase-C) and trace the wiring in `_backend.py:1267-1310` directly. Validate the fix by sending a 256-token completion and confirming coherent output; capture an `nsys` trace for AGENTS.md §4 evidence.
