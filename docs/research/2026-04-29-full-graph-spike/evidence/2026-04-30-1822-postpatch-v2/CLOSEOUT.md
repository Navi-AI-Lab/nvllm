# β-coop Persistent Buffers Patch v2 — Closeout (2026-04-30-1822)

## Verdict

**Patch v2 does not close the FULL+β-coop blocker.** The
`cudaMemsetAsync` for `_phase_e_coop_wo_output[:nat]` is issued
during FULL-graph capture on the current stream (expected to be
captured as a graph memset node, but per-replay graph-node
ordering was not independently proven due to the nsys
child-process limitation) and the op fires on every attached
layer (8 unique stable data_ptrs in the `[CUTE_WO_RESET]` runtime
log). Gate 1 still FAILs with `unique=4` same-prompt, cross-prompt
dependent. The "stale content at stable address" hypothesis from
the v1 closeout — i.e. that zeroing `wo_output` before each
cooperative launch would close the bug — is **insufficient**.
Some other source of divergence remains.

**Production recommendation: PIECEWISE + β-coop remains the
supported path** (v0.3.0 status quo). PIECEWISE C0 GSM8K-sanity 8/8
and C2 replay coherence (unique=1, cross-indep) both PASS — no
PIECEWISE regression from v2. FULL+β-coop blocker remains OPEN; v2
is **not sufficient to enable FULL; PIECEWISE-clean
infrastructure/refactor remains shippable** (persistent buffers +
captured reset op are ready for any further v3 attempt).

## Evidence table

Code commit: `e24c819a3` on branch
`feat/cute-beta-coop-persistent-buffers`.

| Configuration | Result | Evidence |
|---|---|---|
| Op-level functional CUDA smoke | PASS | `../2026-04-30-1736-v2-op-smoke/op_smoke.log` |
| C0 PIECEWISE+β-coop GSM8K-sanity (8/8) | PASS | `../2026-04-30-1742/c0_summary.md` |
| C2 PIECEWISE+β-coop replay coherence | PASS (unique=1, cross indep) | `../2026-04-30-1752/c2_replay_coherence.md` |
| **C2 FULL+β-coop lower-8 (Gate 1)** | **FAIL (unique=4, cross dep)** | `../2026-04-30-1805/c2_replay_coherence.md` |
| `[CUTE_WO_RESET]` capture-side log | 8 unique lines (one per attached layer) | `../2026-04-30-1805/cute_wo_reset_log.txt` |
| C2 FULL+β-coop all-16 (Gate 2) | not run (Gate 1 FAIL) | — |
| nsys trace (host-side limitation) | DONE_WITH_CONCERNS | `<repo-root>/benchmarks/nvllm/traces/cute_paged_attn/2026-04-30-coop-wo-reset/` |

### Comparison with prior baselines

| Patch state | Same-prompt unique (8 replays) | Cross-prompt | Note |
|---|---|---|---|
| Pre-v1 (no patch) | 3 | dep | `../2026-04-30-1311/` (FULL lower-8) |
| v1 (`1cc51ab95`) | 2 | dep | `../2026-04-30-1548/` (post-patch v1) |
| **v2 (this commit)** | **4** | **dep** | `../2026-04-30-1805/` |

**Important framing of the v1 → v2 unique-count change.** v2's
unique=4 is numerically larger than v1's unique=2, but **this is
not load-bearing evidence that v2 made things worse**. Both samples
are 8 trials drawn from the same "broken FULL-graph regime" where
the model alternates between equally-likely token paths at the
divergence point. v1 happened to land on 2 of N possible variants
in its 8 trials, v2 on 4. To call v2 "worse" would require a
controlled multi-seed test we did not run. What we **can**
conclude: v2 did not bring unique below 1, so v2 did not fix the
bug.

Worth observing in the v2 same-prompt outputs: divergence is
mostly in **punctuation/spacing tokens** (`Paris.\n` vs `Paris.\n\n`
vs `Paris\n\n`) plus one Italy/Tokyo answer swap on the 6th trial.
The model's content tokens are largely stable; the bug surfaces in
formatting tokens. This pattern is consistent with the prior
observation that the model is "alternating between equally-likely
tokens at the divergence point in a pattern correlated with prior
replay state" (v1 closeout).

## What v2 changed

Phase 1 commit `fcbdef8da` (with M1+M2 follow-up `af7036150`):

- **New file** `vllm/v1/attention/backends/cute_paged/_wo_output_reset_op.py`
  (~149 lines): `direct_register_custom_op` named
  `cute_paged_reset_wo_output` wrapping a ctypes-bound
  `cudaMemsetAsync` from `libcudart.so`. Lazy 3-candidate library
  loading (`find_library` → `libcudart.so.12` → `libcudart.so`).
  Op-body preconditions (`is_cuda`, `dtype==float32`, `dim()==3`,
  `is_contiguous()`, `0<=nat<=shape[0]`). `nat==0` early return.
  Env-gated capture-side log via `CUTE_WO_RESET_LOG=1`.
- **Side-effect import** at `vllm/nvllm/models/qwen3_5.py:42`
  (mirrors existing `_beta_coop_op` registration site at :41).
- **Callsite** at `vllm/v1/attention/backends/cute_paged/_backend.py:1540-1545`
  (eager body of the existing `cute_beta_coop_run` splitting
  boundary; FX-graph topology unchanged from v1).

Phase 2 commit `688d22918` (sync infrastructure):

- `_sync_host_edits.sh` extended with docker-cp + sentinel checks
  for the two new files (per `feedback_rebuild_guard` — no
  60-min `nvllm:gb10` rebuild needed).
- `c2_full_layer_bisect.sh` forwards `CUTE_WO_RESET_LOG` into the
  container (per `feedback_vllm_enginecore_env_strip`).

## Why v2 failed despite the op firing correctly

The `[CUTE_WO_RESET]` runtime log shows the op fires exactly 8
times during FULL-graph capture, one per attached β-coop layer in
the lower-8 set, each on a stable persistent data_ptr at strided
offsets (e.g. `0x3047c9e00`, `0x304b7c800`, …) with
`nat=1, shape=(1, 4, 5120)`. Each such call is issued during
FULL-graph capture on the current stream and is expected to be
captured as a graph memset node before the corresponding β-coop
kernel launch, though per-replay graph-node ordering could not be
independently proven (nsys child-process limitation, see below).
So the patch is mechanically wired as designed; the limitation is
that we have not closed the loop on the captured-node ordering
proof.

That the patch is mechanically correct **and** the bug is
unfixed implies one of:

1. **`wo_output` residue is not the load-bearing source of
   divergence.** Some other persistent workspace tensor accumulates
   stale content across replays. The most likely candidate per the
   v1 closeout's escalation menu: `_phase_e_coop_mlp_partial_fp32`,
   which uses an in-kernel CTA-local reset (Phase 3.2.5) that may
   not preempt downstream readers under FULL replay the way a
   host-captured memset would.
2. **The divergence source is not workspace residue at all.** Could
   be a Python-side decision in the β-coop op body that freezes at
   capture time but should re-evaluate per replay. Could be a
   downgrade-mode quirk in `gpu_model_runner.py`'s
   `dispatch_cudagraph` that quietly turns FULL into a hybrid
   PIECEWISE on a per-step basis. Could be related to upstream
   #40969 (OPEN as of 2026-04-30; last upstream activity
   2026-04-28T11:06:08Z — recheck before relying on this) — same
   hardware, same cudagraph_mode, same intermittent flake pattern.
3. **Reset ordering may interact poorly with Phase 0 inputs.** The
   reset zeros `wo_output[:nat]` immediately before the cooperative
   launch. If any Phase 0 data flow assumes `wo_output` carries
   prior-replay content as a side-channel, our reset removes that
   assumption. (Reading `_backend.py:1543-1602` makes Phase 0
   inputs look explicit, so this is unlikely — but it is a
   re-checkable hypothesis.)

The 8-line reset log is the strongest direct evidence available;
the nsys trace cannot disambiguate further because of the known
EngineCore subprocess limitation (see nsys summary).

## nsys trace — why it's incomplete

Per `feedback_vllm_profiling`, vLLM V1 spawns model work into an
EngineCore subprocess. nsys with default settings followed only the
nsys-launcher → API-server PID tree, not the EngineCore worker.
The trace at
`benchmarks/nvllm/traces/cute_paged_attn/2026-04-30-coop-wo-reset/changed.nsys-rep`
contains 0 graph-captured events (`graphNodeId IS NOT NULL` count
= 0), 166 sampling-side memset rows (4-byte fills, not the
expected 81920 B `wo_output` reset pattern), and zero β-coop kernel
names. Per AGENTS.md §4, the trace is preserved for forensic
inspection in nsys-ui. For per-replay graph-node ordering proof,
a follow-up arc using nsys child-tree-follow or vLLM's torch
profiler API would be required.

The reset-log evidence (8 unique data_ptrs, captured at
`(EngineCore pid=216)`) is more direct proof that the op fires
correctly under FULL-graph capture than the nsys trace would
have been. nsys would only have added the per-replay ordering
proof, which is moot when the op fires but the bug is unfixed.

## Status

- Branch `feat/cute-beta-coop-persistent-buffers` HEAD: `e24c819a3`
- Patch v1 + v2 are **not sufficient to enable FULL;
  PIECEWISE-clean infrastructure/refactor remains shippable**.
  PIECEWISE production path is intact; the persistent-buffer +
  captured-reset scaffolding is ready for any further v3 attempt.
- FULL+β-coop blocker (`project_full_graph_blocked.md`): still
  OPEN. v2 hypothesis insufficient.
- Production path: PIECEWISE + β-coop (v0.3.0 status quo).

## Recommendation (for human review on return)

Three reasonable next-step paths, listed without preference:

**A. Escalate to v3 — host-captured reset for `mlp_partial_fp32`.**
The v1 closeout's pre-emptive escalation candidate. Replace the
in-kernel CTA-local reset (Phase 3.2.5) with a host-captured
`cudaMemsetAsync` graph node, mirroring v2's pattern but on
`_phase_e_coop_mlp_partial_fp32[:nat]`. Cheap to implement
(reuses v2's `_wo_output_reset_op.py` op pattern; note that
`mlp_partial_fp32`'s buffer shape differs from `wo_output`'s, so
only the op pattern carries over, not the literal byte-count or
slice math). Risk: same "hypothesis is wrong" failure mode if
mlp_partial isn't the load bearer either.

**B. Re-evaluate the diagnosis.** v2's "wired correctly but
unfixed" result is a strong signal that the workspace-residue
hypothesis may be off entirely. Worth checking before another
patch attempt:
- Are there Python-side decisions in the β-coop op body
  (`_beta_coop_op.py`) that freeze at capture but should
  re-evaluate per replay?
- Is `dispatch_cudagraph` quietly downgrading FULL to a hybrid
  mode on per-step basis under specific batch shapes?
- Does upstream #40969 (OPEN as of 2026-04-30; last upstream
  activity 2026-04-28T11:06:08Z — recheck before relying on
  this; same hardware) provide new diagnostic clues?

**C. Defer to upstream.** #40969 is OPEN as of 2026-04-30 (last
upstream activity 2026-04-28T11:06:08Z — recheck before relying
on this) and matches our hardware + cudagraph_mode + intermittent
flake pattern. May get fixed by upstream first; our spike effort
could be redirected to other priorities (FULL_AND_PIECEWISE
re-enablement is one of three candidates per
`project_strategy_priorities`).

Per the user's pre-departure instruction: do **not** auto-escalate
to v3. v2 closeout stops here. Human review decides path A/B/C.

## Followup investigation candidates (carried over from v1)

1. **mlp_partial_fp32 host-captured reset** — the next workspace
   in line, still in scope per the v1 closeout's escalation menu.
2. **Track #35175 pattern variants** — upstream's fix copies fresh
   data into persistent buffers per call; our case has no fresh
   data to copy (the buffers are pure scratch); the right shape
   is "captured zero, not captured copy_in." v2 implemented
   exactly that for `wo_output`; either the pattern needs to
   extend to other workspaces, or the analog doesn't apply
   directly here.
3. **Track #40969** — OPEN upstream as of 2026-04-30 (last
   upstream activity 2026-04-28T11:06:08Z; recheck before relying
   on this). Same hardware (GB10/SM12.1), same
   `cudagraph_mode FULL_AND_PIECEWISE`. Watch for an upstream fix
   that may inform our approach.
