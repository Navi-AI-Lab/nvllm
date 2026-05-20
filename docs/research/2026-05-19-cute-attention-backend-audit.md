# CuTe Attention Backend Audit - 2026-05-19

Scope: static review of the current `cute_paged` attention backend, adjacent
Qwen3.5 integration, serve scripts, README surface, tests, and relevant local
fork commit history. This is a cleanup and risk audit, not a benchmark report.

Evidence standard: performance statements below are source-level complexity
claims unless they cite a committed trace or evidence commit. No new nsys trace
was captured for this audit.

## Commit-History Anchors

These commits are useful context for deciding what is intentional history and
what is now removable clutter.

- `53831213c` (2026-04-11), `feat: CuTe paged attention v1 - prototype
  validated end-to-end`: introduced the backend as a first-class attention
  backend with `accept_output_buffer=True`.
- `26bda34d6` (2026-04-13), `feat(kernel): CuTe DSL paged attention decode
  kernel - first working implementation`: first working FP8 decode kernel.
- `748c9695c` (2026-04-13), `perf(kernel): eliminate .contiguous() KV cache
  copies - zero-copy stride addressing`: explicitly removed whole-cache K/V
  copies from the decode path after nsys showed them dominating runtime.
- `12c1d61b4` (2026-04-13), `feat(kernel+backend): wire W_O fusion params
  through __call__, paged_attention_forward, and backend`: introduced the W_O
  fusion parameter path.
- `f5fce0ddf` (2026-04-13), `fix: hidden_dim 3584->5120`: fixed the W_O GEMV
  tiling for Qwen3.5-27B hidden size 5120. This is relevant to the current
  hard-coded hidden-size finding.
- `f97219029` (2026-04-14), `feat(cuda-graphs): graph-safe kernel dispatch -
  remove .item(), empty_like, .contiguous()`: says `DecodeKernel.__call__`
  accepted a caller-provided persistent output buffer. The current public
  wrapper regressed to `output_buf=None`.
- `5fa6bccc5` (2026-04-22), `feat(cute): attach_next_input_layernorm API +
  Phase E workspace`: introduced the cross-layer input-layernorm attachment and
  Phase 4 epilogue workspace.
- `44b9a980e` (2026-04-23), `feat(cute): PhaseE_Beta_Kernel beta-coop unified
  kernel (phases 0->4)`: introduced the unified cooperative kernel with phases
  0 through 4 and self-reset counters.
- `54da780f3` (2026-04-25), `refactor(cute): C1.5 - delete Phase 4 + F.1
  layer-LN bake plumbing`: deleted Phase 4 from the production beta-coop body
  and intentionally left some old methods commented. This is the root of most
  Phase 4 comment archaeology.
- `d3ddffe4a` (2026-04-29), `feat(cute-full-cache): beta-coop FULL kernel
  disk-cache - Token-1 in minutes, not hours`: introduced the runtime CuTe disk
  cache and notes follow-ons around key drift and stale warmup prefill kwargs.
- `fcbdef8da` (2026-04-30), `feat(cute beta-coop): captured wo_output reset op
  (v2 patch)`: added the graph-captured `wo_output[:nat]` reset custom op.
- `13b2337d9` (2026-05-04), `wo_split: restrict CUTE_WO_SPLIT to evidenced set
  {1, 2, 4, 8}`: makes split values a production-evidence surface, not just a
  kernel capability surface.
- `c85f8da03` (2026-05-04), `wo_split: implement K-parallel W_O GEMV body
  (gated on wo_split>1)`: moved W_O to a split-aware K-parallel body.
- `69c530082` (2026-05-04), `wo_split: remove dead total_ctas_per_seq_attn
  kernel arg (merge-prep)`: good precedent for deleting dead Phase E plumbing
  once the evidence says it is stale.
- `fb2fe9c06` (2026-05-17), `feat(qwen3.6): driver update - scripts cleanup +
  fusion_max_tokens spec-decode fix + recipe + tokenizer patch (#20)`: states
  `_fusion_max_num_seqs` is deliberately misnamed and now semantically means
  max fusion tokens.

## Phase 1 - Architecture And Readability

### Finding 1.1 - Large stale rollback blocks clutter `_backend.py`

Severity: Medium. Win type: readability, stale-risk removal.

Current source has large disabled blocks and rollback copies. These do not
dominate every active path, but they do materially raise the cost of reading
the 2400-line backend implementation:

- `vllm/v1/attention/backends/cute_paged/_backend.py:377-390`: disabled
  skip-flag fields and inert defaults.
- `vllm/v1/attention/backends/cute_paged/_backend.py:563-667`: fully commented
  `attach_next_input_layernorm` and `attach_input_layernorm`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1158-1191`: old
  `bind_fusion_weights` implementation.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1279-1286`: disabled Phase
  D2 reset comment.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1423-1437`: old
  `_will_fire_beta_coop_pre` predicate.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1535-1583`: old Phase 3
  runtime predicate block.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1587-1599`: old Phase 4
  next-layer LN locals.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1997-2048`: old
  attention-side MLP launch block.

Commit context: `54da780f3` explicitly says several methods were commented out
"per feedback_comment_not_delete" and that a later C4 would fully remove them.
The current file still carries that C1.5 archaeology almost a month later.

Recommendation: move rollback rationale to this doc family or a short design
note, then delete commented-out implementations. Keep only current invariants
near the code. The operational behavior should be recoverable from commit
history, not from 100+ line commented copies in the hot file.

### Finding 1.2 - `_fusion_max_num_seqs` now means max fusion tokens

Severity: Medium. Win type: naming and mental-model cleanup.

Current source:

- `vllm/v1/attention/backends/cute_paged/_backend.py:490-502` derives
  `fusion_max_tokens = max_num_seqs * decode_query_len`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:496-497` says
  `_fusion_max_num_seqs` is kept misnamed for blast-radius reasons.
- `vllm/v1/attention/backends/cute_paged/_backend.py:534` stores
  `self._fusion_max_num_seqs = fusion_max_tokens`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1274-1277` and
  `vllm/v1/attention/backends/cute_paged/_mlp_op.py:80-89` use the field as an
  active-token capacity.

Commit context: `fb2fe9c06` explains the MTP/spec-decode bug and states the
field name was kept only for minimal blast radius.

Recommendation: finish the rename to `_fusion_max_tokens`. If there is concern
about local out-of-tree references, keep a temporary alias for one release
cycle and make the canonical name match the current invariant.

### Finding 1.3 - Phase E top-of-file docs still describe deleted Phase 4

Severity: Medium. Win type: onboarding and correctness of first-read docs.

Current source:

- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:4-14` says the file
  contains Phase 4 epsilon epilogue tasks.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:188` still says the
  beta-coop path fuses residual plus next-norm.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2742-2783` still
  has a Task 16 header describing Phases 0 + 1 + 3 + 4 and secondary-barrier
  Phase 4 mechanics.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2785-2872` documents
  that Phase 4 inputs were deleted and the beta-coop full kernel ends at Phase
  3.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:5775-5780` repeats
  that Phase 4 was deleted and the kernel now ends at the Phase 3 MLP write.
- `vllm/v1/attention/backends/cute_paged/_backend.py:916` still says real work
  happens in Phases 1-4.

Commit context: `44b9a980e` introduced phases 0->4; `54da780f3` deleted Phase
4 from the production body.

Recommendation: update the module docstring and class overview so the first
screen states the current production path: Phase 0 input-LN placeholder, Phase
1 attention/W_O, Phase 2 barrier, Phase 3 MLP, no production Phase 4.

### Finding 1.4 - `phase_e_kernel.py` is a mixed production/debug/history file

Severity: Low/Medium. Win type: navigability and ownership boundaries.

Current source concentrates production kernel launch, test-backed debug
launchers, compile cache/heartbeat, PTX helpers, stale historical comments,
and region timing:

- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:140-185` contains
  heartbeat and process-wide compile-cache state.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:384-612` and later
  blocks include standalone/debug phase kernels.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2742` onward is the
  production beta-coop full path.

Recommendation: do not do a risky mechanical split while kernel source is
source-sensitive. Some debug entry points still have tests, so the first move
should be labeling rather than deletion: add a compact file map near the top or
move clearly debug/test launchers after a "debug-only" divider. If a future
refactor touches this file anyway, extract non-production helpers behind stable
import names.

### Finding 1.5 - Production fallback imports test modules

Severity: Medium. Win type: package boundary cleanup.

Current source:

- `vllm/v1/attention/backends/cute_paged/kernel.py:2316-2324`: when CuTe is not
  available, fallback imports `tests.nvllm.attention.reference`.
- `vllm/v1/attention/backends/cute_paged/kernel.py:2336-2344`: non-decode path
  imports the same test reference.

Recommendation: production package code should not import from `tests`. Either
move the reference fallback into a production module with explicit limitations,
or fail closed with a clear error if this backend is decode-only in supported
serving modes.

### Finding 1.6 - README and serve-script defaults disagree on beta-coop default

Severity: Medium. Win type: operator clarity.

Current source:

- `README.md:159-161` says the beta-coop fused kernel is the default and points
  users at `./scripts/serve-qwen35.sh`.
- `scripts/serve-qwen35.sh:116-130` defaults `CUTE_MLP_FUSION=1`,
  `CUTE_ATTN_FUSION=1`, but `CUTE_PHASE_E_FUSION=0`.
- `scripts/serve-qwen35-full.sh:87` and `scripts/serve-qwen35-full.sh:240`
  default Phase E on for the full/blessed-cache path.
- `scripts/serve-qwen36.sh:17-18` explicitly says all CuTe fusions default off
  for Qwen3.6, and `scripts/serve-qwen36.sh:193` exports
  `CUTE_PHASE_E_FUSION=0` by default.

Commit context: `66c8f6bae` turned CuTe kernel fusion off after
non-deterministic reductions, `9e93c1c9f` re-enabled defaults after the
deterministic reduction fix, and `fb2fe9c06` renamed scripts and documented the
new Qwen3.6 driver state. The current README sentence appears too broad for
the renamed script matrix.

Recommendation: rewrite the README status line as a matrix:
`serve-qwen35.sh` = PIECEWISE decode with `CUTE_MLP_FUSION=1`,
`CUTE_ATTN_FUSION=1`, `CUTE_PHASE_E_FUSION=0`, `CUTE_WO_SPLIT=1`;
`serve-qwen35-full.sh` = FULL_AND_PIECEWISE blessed path with Phase E on for
lower-8 by default; `serve-qwen36.sh` = MTP bring-up with MLP/attention/Phase E
fusions off and `CUTE_WO_SPLIT=1`. README wording around Qwen3.6 should say
`CUTE_WO_SPLIT=8` is an evidenced opt-in, not the script default.

### Finding 1.7 - Phase E env-config comment contradicts import-time snapshot

Severity: Low/Medium. Win type: operator/debug clarity.

Current source:

- `vllm/v1/attention/backends/cute_paged/_backend.py:147` says
  `_phase_e_env_config()` should be called once per forward.
- `vllm/v1/attention/backends/cute_paged/_backend.py:173-178` snapshots
  `_PHASE_E_ENV` at module import time.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1356` reads the import-time
  snapshot during forward.

Recommendation: update the helper docstring to match the current import-time
contract, or move back to per-forward parsing if dynamic env mutation is
intended. Today the code and comment teach different operational behavior.

## Phase 2 - Runtime Complexity And Big-O Taxes

### Finding 2.1 - Legacy paged W_O is 5120-specialized; beta-coop is under-guarded

Severity: High if the backend is expected to support non-Qwen3.5/3.6 27B
hidden sizes; Low if the path is intentionally model-specialized.

Current source, legacy paged path:

- `vllm/v1/attention/backends/cute_paged/kernel.py:1647-1648` hard-codes
  `hd_wo = Int32(5120)` and `n_per_thr = Int32(40)`.
- `vllm/v1/attention/backends/cute_paged/kernel.py:1796-1799` says
  `range_constexpr(5)` assumes `hidden_dim/128 = 40` and Qwen3.5-27B.
- `vllm/v1/attention/backends/cute_paged/kernel.py:1825-1827` later derives
  RMSNorm dimensions from runtime `hidden_dim`, so not all fused stages share
  the same specialization assumption.

Current source, beta-coop path:

- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:217-221` only
  asserts `hidden_size % 128 == 0`.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:4345-4354` derives
  W_O output groups from `self.hidden_size // self.num_threads // 8`.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:4572-4574` uses the
  same 8-element group structure in the gather path.

That means beta-coop is not hard-coded to 5120, but the current guard appears
weaker than the loop shape: hidden sizes divisible by 128 but not by 1024 can
leave an 8-element-group tail uncovered.

Commit context: `f5fce0ddf` fixed the same area from hidden 3584 to 5120.

Recommendation: assert the actual supported hidden-size invariant before
enabling each fused path, or compile-specialize the W_O loop structure from
hidden size with explicit tail handling. Silent partial coverage would be worse
than a fail-closed gate.

### Finding 2.2 - Public decode wrapper ignores the caller output buffer

Severity: Medium. Win type: per-layer allocation/copy removal.

Current source:

- `vllm/v1/attention/backends/cute_paged/_backend.py:1193-1212` receives the
  framework `output` tensor.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1443-1478` calls
  `_run_paged()` without passing `output`.
- `vllm/v1/attention/backends/cute_paged/kernel.py:2037-2041` allocates
  `torch.empty_like(query)` if `output_buf` is `None`.
- `vllm/v1/attention/backends/cute_paged/kernel.py:2370` passes
  `output_buf=None  # Not used yet`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:2062-2063` copies the
  result back into `output`.

Commit context: `53831213c` introduced `accept_output_buffer=True`; `f97219029`
explicitly says `DecodeKernel.__call__` accepted a caller-provided persistent
buffer for graph-safe dispatch.

Recommendation: plumb the output tensor through `paged_attention_forward` and
`DecodeKernel.__call__`, then remove the raw-attention allocation. The final
copy removal applies to the non-framework route; the framework-output route
already writes through the provided output/residual/MLP buffers.

### Finding 2.3 - Prefill falls back through PyTorch reference plus cache copies

Severity: High if prefill on this backend is expected to be performant;
otherwise Medium as an operator-surprise issue.

Current source:

- `vllm/v1/attention/backends/cute_paged/kernel.py:2186-2202` has an empty
  `PrefillKernel._kernel` body.
- `vllm/v1/attention/backends/cute_paged/kernel.py:2336-2344` handles
  non-decode by importing the test reference and passing
  `kv_cache[:, 0].contiguous()` and `kv_cache[:, 1].contiguous()`.
- `vllm/v1/attention/backends/cute_paged/kernel.py:65` still defines
  `PREFILL_CONFIG`, but every non-decode batch returns through the reference
  fallback before `DECODE_CONFIG` is selected.

Commit context: `748c9695c` removed whole-cache `.contiguous()` calls from the
decode path because they copied the K and V caches every forward. The prefill
fallback now carries a similar shape of cost.

Recommendation: make the supported state explicit. If this is a decode-only
backend in production, fail closed or route prefill to a known backend. If
prefill is supported, move it off the test reference and avoid full cache
copies.

### Finding 2.4 - Legacy attention fusion zeros full capacity, not active rows

Severity: Medium. Win type: obvious O(capacity) reset reduction.

Current source:

- `vllm/v1/attention/backends/cute_paged/_backend.py:1319-1321` zeros
  `self.wo_output` and `self.arrival_count` whenever attention fusion runs.
- `vllm/v1/attention/backends/cute_paged/_backend.py:417-420` allocates
  `wo_output` as `[max_num_seqs, total_ctas_per_seq, hidden_dim]`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1274-1277` already knows
  `num_actual_tokens` and buffer fit.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1319-1321` runs before
  `_skip_paged = _use_beta_coop` is known, so beta-coop can pay the legacy
  reset even though it later uses a separate beta-coop W_O buffer.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1639-1641` resets the
  beta-coop W_O buffer with the active-slice custom op.
- `vllm/v1/attention/backends/cute_paged/kernel.py:1971-1975` self-resets
  `arrival_count` in the legacy kernel, making the steady-state host reset
  suspicious outside exception recovery.

Positive contrast: `vllm/v1/attention/backends/cute_paged/_mlp_op.py:114-115`
zeros only active MLP slices.

Commit context: `fcbdef8da` added a custom op that zeros
`_phase_e_coop_wo_output[:nat]`, showing the repo already has a preferred
captured-slice reset pattern.

Recommendation: compute the beta-coop skip decision before legacy resets, zero
only `[:num_actual_tokens]` if graph-capture shape rules allow it, and either
delete the steady-state `arrival_count.zero_()` or document why the kernel
self-reset is insufficient. Keep exception recovery resets separate.

### Finding 2.5 - beta-coop Phase 0 does real work for an ignored side-channel

Severity: Medium. Win type: remove dead default-path GPU work.

Current source:

- `vllm/nvllm/models/qwen3_5.py:512-521` computes input layernorm before
  attention.
- `vllm/nvllm/models/qwen3_5.py:352-358` passes that `hidden_states` value as
  `attn_input` to `cute_beta_coop_run`.
- `vllm/v1/attention/backends/cute_paged/_beta_coop_op.py:92-109` forwards
  `attn_input` into `CutePagedAttentionImpl.forward`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:908-918` still documents a
  dummy Phase 0 scratch/gamma setup.
- `vllm/v1/attention/backends/cute_paged/_backend.py:935-942` allocates
  `_phase_e_coop_attn_input_scratch` and `_phase_e_coop_input_gamma`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1663-1670` passes
  `_attn_output_buf` as a placeholder and dummy gamma/scratch.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:3610-3679` Phase 0
  performs full RMSNorm-style hidden/residual work and writes
  `attn_input_bf16`.

Commit context: `44b9a980e` added Phase 0 as part of phases 0->4; `54da780f3`
removed the later cross-layer LN bake, but Phase 0 still remains as a
placeholder in current production construction.

Recommendation: either wire the available `attn_input` side-channel into
something useful, or compile a production specialization that skips Phase 0.
Keep the debug/full phase variant only where it is still tested.

### Finding 2.6 - Dormant pre-W_O arrival signal still runs at default split

Severity: Low/Medium. Win type: small default-path synchronization cleanup.

Current source:

- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:261-275` defaults
  `CUTE_WO_SPLIT=1`.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2844-2849`
  documents `pre_wo_arrival_count` as dormant at split 1.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:3155-3157` resets
  `pre_wo_arrival_count` every launch.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:4187-4204` still
  performs producer fence/sync/atomic work even when `wo_split == 1`.
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:4212-4217` has an
  empty consumer mask at split 1.

Commit context: `c85f8da03` introduced the K-parallel W_O body gated on
`wo_split>1`; `13b2337d9` keeps default split at 1 and restricts evidenced
values.

Recommendation: gate producer signaling and host reset behind compile-time
`wo_split > 1`, or document why the default path needs the no-op signal.

### Finding 2.7 - Disk-cache key over-scans and duplicates package fingerprint

Severity: Medium for cold compile/key construction; Low for steady state.

Current source:

- `vllm/v1/attention/backends/cute_paged/disk_cache.py:155-177` walks and
  stats the package tree to compute tree state.
- `vllm/v1/attention/backends/cute_paged/disk_cache.py:180-196` caches content
  hashes by `(root, state)`, but `_tree_state(root)` still runs for every call.
- `vllm/v1/attention/backends/cute_paged/disk_cache.py:204-213` returns
  `cute_paged:<package_fingerprint>` for cute_paged functions.
- `vllm/v1/attention/backends/cute_paged/disk_cache.py:396-404` also includes
  `_package_fingerprint()` in the full payload.

Commit context: `d3ddffe4a` says pointer canonicalization and explicit cache
key behavior were important for G1 cold/warm key stability. That makes the
cache key worth keeping simple and inspectable.

Recommendation: include the package fingerprint once, not twice. Consider
function-specific source hashing for kernels where package-wide invalidation is
too broad, or memoize tree state during one process if safe. The current shape
appears to do two package tree-state scans per cute_paged key build, although
content bytes are cached after the first fingerprint.

### Finding 2.8 - Framework-output route may make mirror copies redundant

Severity: Low/Medium. Win type: avoid graph-captured side-effect copies when
direct tensors are already passed.

Current source:

- `vllm/nvllm/models/qwen3_5.py:282-296` mirrors gate into `impl.gate_buf`.
- `vllm/nvllm/models/qwen3_5.py:524-547` mirrors residual into
  `impl.residual_buf`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1628-1635` uses direct
  `residual[:nat]` and `gate[:nat]` tensors when `_framework_output_route` is
  active.

Recommendation: keep the opaque mirror ops for the legacy/non-framework route,
but investigate skipping them when the framework-output route is statically
bound. This should be validated carefully because the mirror ops were added to
preserve side effects under CUDA graph capture.

### Positive 2.A - Active-slice reset pattern exists and should be reused

`vllm/v1/attention/backends/cute_paged/_wo_output_reset_op.py:81-135` validates a
3D CUDA FP32 buffer and uses `cudaMemsetAsync` for `wo_output[:nat]`. This is a
good pattern for captured reset work: explicit preconditions, active slice,
current torch stream, and no Python-side full-slab memset.

## Phase 3 - Tests, Docs, And Operational Surface

### Finding 3.1 - Beta-lite integration test is a contradictory source-contract test

Severity: Medium. Win type: false-confidence reduction.

Current source:

- `tests/kernels/cute/test_phase_e_beta_lite_integration.py:1-7` starts with a
  beta-lite end-to-end claim, then admits it is source-level placeholder
  coverage.
- `tests/kernels/cute/test_phase_e_beta_lite_integration.py:24-33` checks for
  source substrings like `emit_epilogue` and `_phase_e_env_config`.
- `tests/kernels/cute/test_phase_e_beta_lite_integration.py:27` can pass just
  because `emit_epilogue` appears in source, even though
  `vllm/v1/attention/backends/cute_paged/_backend.py:1919` currently passes
  `emit_epilogue=False`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:173-178` snapshots
  `_PHASE_E_ENV` at import time, so the presence of `_phase_e_env_config` in
  the file does not prove runtime dispatch behavior.

Recommendation: replace this with a small behavior test around an exposed
dispatch predicate/helper, or rename it honestly as a source-contract test and
make it assert the current `emit_epilogue=False` state.

### Finding 3.2 - `attach_next_input_layernorm` tests are stale against current code

Severity: Medium/High. Win type: delete broken historical test surface.

Current source:

- `tests/kernels/cute/test_phase_e_backend_api.py:1-9` describes testing
  `attach_next_input_layernorm`.
- `tests/kernels/cute/test_phase_e_backend_api.py:34-110` calls
  `impl.attach_next_input_layernorm(...)`, but the implementation is commented
  out in `_backend.py`.
- `tests/kernels/cute/test_phase_e_backend_api.py:132-157` expects the retired
  method to raise on a free-memory kill switch.
- `vllm/nvllm/models/qwen3_5.py:832-839` says the cross-layer binding loops and
  attach methods were deleted/commented in C1.5.

Commit context: `5fa6bccc5` introduced this API; `54da780f3` deleted the
production path and left methods commented. These tests belong to the old
phase unless intentionally xfailed with that context.

Recommendation: remove or xfail only the attach-method tests, then keep or
separate still-valid env parser/resident-cap tests. Add current coverage around
the actual framework-output route and unconditional input-layernorm regime.

### Finding 3.3 - Model-binding test still expects deleted cross-layer binding

Severity: Medium. Win type: stale-doc/test cleanup.

Current source:

- `tests/kernels/cute/test_phase_e_model_binding.py:23-31` expects
  `attach_next_input_layernorm` in the model source.
- `tests/kernels/cute/test_phase_e_model_binding.py:52-73` expects the old
  "Phase E cross-layer binding" shape.
- `vllm/nvllm/models/qwen3_5.py:512-521` now runs input layernorm
  unconditionally at layer entry.
- `vllm/nvllm/models/qwen3_5.py:298-365` uses the beta-coop framework-output
  route instead of next-layer LN attachment.
- `vllm/nvllm/models/qwen3_5.py:717-756` uses opaque
  `cute_phase_e_dispatch` for MLP/runtime consume.

Recommendation: rewrite the model-binding test around current invariants:
always-run input-LN at layer entry, framework output route emits
`cute_beta_coop_run`, and MLP consume goes through `cute_phase_e_dispatch`.

### Finding 3.4 - Dispatch predicate test copies old rules and contradicts current gate

Severity: Medium. Win type: prevent regression tests from encoding false rules.

Current source:

- `vllm/v1/attention/backends/cute_paged/_backend.py:1396-1405` says the
  cooperative resident-cap gate is hard even in forced-coop mode.
- `tests/kernels/cute/test_phase_e_dispatch.py:16-17` imports the real env
  parser helper; that part of the test is not the problem.
- `tests/kernels/cute/test_phase_e_dispatch.py:77-101` has its own copied
  `_choose_path` helper.
- `tests/kernels/cute/test_phase_e_dispatch.py:107-110` expects forced coop to
  ignore cap for `total_ctas=2048`.

Recommendation: expose a real dispatch-decision helper in backend code and
test that. If this must remain a source-level test, update it to match the
hard cap rule.

### Finding 3.5 - Additional stale Phase E source tests now fail or assert old buffers

Severity: Medium. Win type: stale-test cleanup.

Current source:

- `tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py:23-27` expects the
  literal source string `residual_in=self.residual_buf`.
- `tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py:45-48` expects the
  literal source string `residual_post_ln=self.residual_buf`.
- `vllm/v1/attention/backends/cute_paged/_backend.py:1628-1635` and
  `vllm/v1/attention/backends/cute_paged/_backend.py:1885` now route through
  locals rather than those exact source strings.
- `tests/kernels/cute/test_phase_f1_opaque_gate.py:60` expects
  `cute_phase_e_dispatch` to consume `next_hidden_scratch`.
- `vllm/v1/attention/backends/cute_paged/_mlp_op.py:207-210` currently consumes
  `impl.mlp_output` and `impl.residual_output`.
- `tests/kernels/cute/test_phase_f1_opaque_gate.py:99-100` calls deleted
  `cute_phase_e_skip_input_layernorm`, while
  `vllm/v1/attention/backends/cute_paged/_mlp_op.py:237-244` documents that op
  as deleted in C1.5.

Recommendation: either delete these stale source tests or rewrite them around
the current buffer contract. The source-string form is especially fragile now
that correctness depends on framework-route locals, not direct `self.*` names.

### Finding 3.6 - Profiling span tests are acceptable but should stay scoped

Severity: Low.

Current source:

- `tests/kernels/cute/test_phase_e_record_function_spans.py:31-34` reads source
  with `inspect`.
- `tests/kernels/cute/test_phase_e_record_function_spans.py:61-81` uses a
  source-slice heuristic.
- `tests/kernels/cute/test_phase_e_record_function_spans.py:84-138` checks
  strings/regexes.

Recommendation: source tests are reasonable for literal profiler label
contracts. Keep them clearly named as source-contract tests and avoid using
them as evidence for behavioral dispatch correctness.

### Finding 3.7 - README/script/profile matrix should cite history-backed status

Severity: Medium.

This is the docs side of Finding 1.6. Current README status should mention:

- `serve-qwen35.sh` defaults Phase E off.
- `serve-qwen35-full.sh` defaults Phase E on for the blessed/full path.
- `serve-qwen36.sh` defaults Phase E off during Qwen3.6 bring-up.
- `CUTE_WO_SPLIT=8` is evidenced but opt-in (`README.md:118`,
  `README.md:159-161`, `scripts/serve-qwen35.sh:130`).
- `README.md:125` says `CUTE_WO_SPLIT=8` carries over for Qwen3.6, while
  `scripts/serve-qwen36.sh:116` and `scripts/serve-qwen36.sh:198` default split
  to 1. `scripts/serve-qwen36.sh:18` also calls split 1 the
  "production-blessed K-parallel decode path", which is misleading because
  split 1 is the non-K-parallel default.
- `scripts/profile_cute_paged.sh:115-130` launches with `python3`, enables
  `--enable-prefix-caching`, and does not pass the same CuTe env matrix as the
  serve scripts.
- `scripts/serve-qwen35.sh:70` and `scripts/serve-qwen36.sh:103` explicitly
  avoid prefix caching because it corrupts SSM state.

Commit context: `fb2fe9c06` already contains the script rename/default story
and the Qwen3.6 MTP regression evidence. The README and profiling script should
reuse that precise status instead of a broad "beta-coop is default" sentence
or a profiler-only launch shape.

## Suggested Cleanup Order

1. Fix docs/tests that are plainly stale: README/script/profile matrix, Phase E
   docstrings/comments, `attach_next_input_layernorm` tests, model-binding
   tests, dispatch predicate test, `test_phase_f1_opaque_gate.py`, and
   `test_uber_kernel_buffer_contracts.py`.
2. Rename `_fusion_max_num_seqs` to `_fusion_max_tokens` with a temporary alias
   if needed.
3. Delete large commented rollback blocks after preserving any still-useful
   rationale in a short design note.
4. Plumb caller output buffer through `paged_attention_forward`.
5. Add fail-closed hidden-size assertions for model-specialized fused paths.
6. Remove or gate default-path no-op work: Phase 0 placeholder and split-1
   pre-W_O arrival signal.
7. Simplify disk-cache key fingerprinting once the behavioral cleanup is done.

## Fresh-Eye Reviews

Phase 1 review: completed. Adjustments integrated: softened the rollback-block
wording, added wider Phase 4/docstring staleness, added the import-time
`_PHASE_E_ENV` mismatch, and clarified the script/default matrix.

Phase 2 review: completed. Adjustments integrated: scoped the hard-coded 5120
finding to legacy paged W_O, added beta-coop hidden-size guard risk, limited
the output-copy claim to the non-framework route, corrected the Phase 0
side-channel wording, and added reset/mirror-copy cleanup candidates.

Phase 3 review: completed. Adjustments integrated: tightened source-test
wording, added the `emit_epilogue=False` false-positive case, added stale
buffer-contract and F.1 opaque-gate tests, and expanded the operational matrix
to include `profile_cute_paged.sh`.
