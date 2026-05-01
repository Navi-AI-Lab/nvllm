# β-coop Persistent Buffers Patch v2 — Closeout (2026-04-30-1822)

> **2026-05-01 ADDENDUM:** the original v2 closeout below was written
> when v2's Gate 1 had just FAILed and we hadn't yet decided what
> Path B (re-evaluate diagnosis) would surface. Path B has now run.
> See `## Path B Update — 2026-05-01` at the end of this doc for the
> updated verdict. Short version: Z1 cache-pin causality test moves
> the v2 closeout's "stochastic FAIL" reading to "FAIL is dominated
> by which torch.compile inductor binary got compiled," and
> orphans v3 (mlp_partial_fp32 host-reset) as the wrong patch shape.

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
  (at the time of this section's writing; current HEAD is
  `b2677abd3` after Path B + Z1 work — see addendum below).
- Patch v1 + v2 are **not sufficient to enable FULL;
  PIECEWISE-clean infrastructure/refactor remains shippable**.
  PIECEWISE production path is intact; the persistent-buffer +
  captured-reset scaffolding is ready for any further v3 attempt.
- FULL+β-coop blocker (`project_full_graph_blocked.md`): still
  OPEN. v2 hypothesis insufficient.
- Production path: PIECEWISE + β-coop (v0.3.0 status quo).

> 2026-05-01: status above stands, but Path B work has reframed
> what "v3" looks like — v3-as-mlp_partial-host-reset is now
> orphaned. See addendum.

## Recommendation (historical — superseded by Path B addendum)

> **2026-05-01:** Paths A and B below have been resolved by the
> Path B addendum at the end of this doc. Path A (mlp_partial_fp32
> host-reset) is **orphaned** because Z1 showed the FAIL is
> dominated by torch.compile artifact identity, not workspace
> residue. Path B (re-evaluate diagnosis) is **DONE** — the
> diagnosis was wrong. Path C (defer to upstream) remains
> applicable as a long-term direction.

**A. ~~Escalate to v3 — host-captured reset for `mlp_partial_fp32`~~** (orphaned 2026-05-01).
The v1 closeout's pre-emptive escalation candidate. Originally
proposed to replace the in-kernel CTA-local reset (Phase 3.2.5)
with a host-captured `cudaMemsetAsync` graph node mirroring v2's
pattern but on `_phase_e_coop_mlp_partial_fp32[:nat]`. **Now
orphaned**: Path B Step 2 (code inspection) showed the in-kernel
reset is mechanically correct under FULL replay; Z1 showed the
FAIL is upstream-of-our-code (torch.compile inductor
non-determinism producing two distinct binaries for the same
cache key). A workspace reset cannot fix a compile-output bug.

**B. ~~Re-evaluate the diagnosis~~** (done 2026-05-01).
This is what Path B did. Result: workspace-residue hypothesis was
off; the dominant variable is which inductor artifact got
compiled. See Path B addendum.

**C. Defer to upstream.** #40969 is OPEN as of 2026-04-30 (last
upstream activity 2026-04-28T11:06:08Z — recheck before relying
on this) and matches our hardware + cudagraph_mode + intermittent
flake pattern. The torch.compile inductor non-determinism that
Z1 surfaced is a separate upstream concern. Both still applicable
as long-term tracks.

## Followup investigation candidates (updated 2026-05-01 post-Path B)

1. ~~**mlp_partial_fp32 host-captured reset**~~ — **orphaned**
   2026-05-01. Path B Step 2 showed the in-kernel CTA-local reset
   is mechanically correct; Z1 showed the FAIL is upstream of any
   workspace reset. Do not pursue.
2. **Track #35175 pattern variants** — upstream's fix copies fresh
   data into persistent buffers per call; our case has no fresh
   data to copy (the buffers are pure scratch); the right shape
   is "captured zero, not captured copy_in." v2 implemented
   exactly that for `wo_output`; the workspace-side of this is
   complete. The pattern that DOES apply post-Path B: persisting
   the inductor AOT cache across container starts (Z1 finding).
3. **Track #40969** — OPEN upstream as of 2026-04-30 (last
   upstream activity 2026-04-28T11:06:08Z; recheck before relying
   on this). Same hardware (GB10/SM12.1), same
   `cudagraph_mode FULL_AND_PIECEWISE`. Watch for an upstream fix.
4. **Track torch.compile / inductor non-determinism** (NEW
   2026-05-01). Z1 found that inductor produces two distinct
   binaries (sha256s differ) for the same AOT cache key on the
   same input graph. This is the dominant cause of FULL+β-coop
   FAIL. Workaround: pin a known-good cache. Long-term: report
   upstream when we have a reduced repro.

## Path B Update — 2026-05-01

After this closeout was written, Path B (re-evaluate diagnosis,
per the user's pre-departure choice) ran. Five sub-steps:

### Step 1 — dispatch_cudagraph audit (`evidence/2026-04-30-1848-pathb-step1-dispatch-audit/`)

Per-call probe at the post-DP-re-dispatch point in
`gpu_model_runner.py`. 89 of 91 active dispatches during the
c2_replay_coherence pattern were `cg_mode=FULL` with
`uniform_decode=True`; the 2 PIECEWISE rows were during FULL
graph capture for non-uniform shapes, not steady decode.

**Hybrid-dispatch hypothesis ruled out for steady decode.**

Sidebar from this run: c2 produced `unique=1 PASS` (vs Gate 1's
`unique=4 FAIL`, same code path mod the new probe). That flip is
the first hint that the bug is not deterministic at this code
state — either the audit perturbed something or the bug is
genuinely stochastic. Step X (below) tested this.

### Step 2 — code inspection (Y, then Y2)

**Y:** `_beta_coop_op.py` body is short and clean — delegates to
`impl.forward`, no host syncs, no `.item()`. The actual decisions
live in `_backend.py:1488-1604` and `phase_e_kernel.py`. Walked
through the five hoisted workspace buffers and their reset
mechanisms:
- `wo_output`: v2's captured `cudaMemsetAsync` (firing correctly,
  8 unique data_ptrs in capture log).
- `mlp_partial_fp32`: in-kernel CTA-local reset at Phase 3.2.5,
  `_st_global_f32(0.0)` per disjoint slice, `cute.arch.sync_threads()`
  before Phase 3.3 atomic_add. Mechanically correct.
- `mlp_arrival_count`, `grid_barrier_i32`, `phase1_arrival_count`:
  small-counter `.zero_()` (graph-safe per the in-code narrowing
  comment at `phase_e_kernel.py:3025-3035`).

**Y verdict: all five workspace resets are correct.**
v3-as-mlp_partial-host-reset would be redundant.

**Y2:** Greped `_backend.py:1488-1604`, `_beta_coop_op.py`, and
`phase_e_kernel.py` launch wrapper for hidden per-call allocations
(`torch.empty`, `torch.zeros`, `.zero_(`, `.clone()`,
`.contiguous()`, `.to(`). One latent fallback at
`phase_e_kernel.py:2868`: `residual_output = torch.empty(...)`
when `residual_output is None`. The β-coop callsite always passes
`residual_output=_residual_output_buf` so the fallback never
fires in our path — but it's a defensive risk if any caller change
ever leaves it None. Flagged for a future defensive raise; **not
patched in this work** to keep the code under test stable for X
and Z1.

**Y2 verdict: clean for the actual β-coop path.**

### Step X — audit-OFF reproducibility (`evidence/2026-04-30-{1910,1918,1926}-pathb-x-trial-{1,2,3}/` and `2026-04-30-{1939,1947}-pathb-x-trial-{4,5}/`)

5 audit-OFF Gate 1 trials at HEAD `d36abf771`, `CUTE_WO_RESET_LOG=1`,
`CUTE_DISPATCH_AUDIT=0`, fresh container per trial.

| Trial | unique | overall |
|---|---|---|
| X.1 | 1 | PASS |
| X.2 | 3 | FAIL |
| X.3 | 1 | PASS |
| X.4 | 1 | PASS |
| X.5 | 4 | FAIL |

**2/5 FAIL → real stochastic bug.** Per the user's verdict
framework: any v3 acceptance criteria must be statistical (5/5
PASS or 9/10 PASS audit-OFF), not single-run.

### Step A — log-diff PASS vs FAIL

Compared `docker_logs_full.txt` across the 3 PASS and 2 FAIL
X-trials (timestamps stripped). Identical category counts (warn,
error, capture events, wo_reset count, first-FULL probe). One
strong, perfectly-clustered discriminator:

| Trial | Status | Compile time | "collected artifacts" log size |
|---|---|---|---|
| X.1 | PASS | 83.40s | 62,118,662 B |
| X.2 | FAIL | 101.57s | 73,393,077 B |
| X.3 | PASS | 83.04s | 62,136,258 B |
| X.4 | PASS | 84.71s | 62,179,567 B |
| X.5 | FAIL | 101.19s | 73,336,693 B |

**Same torch AOT cache key
(`9a5549f23a178e35a9a3e9b4bed7adf1d137d22f3fc06ef8048d589e5d625721`)
across all trials, but inductor produced two distinct compiled
artifacts.** PASS bucket: ~83-85s, ~62.1 MB. FAIL bucket: ~101s,
~73.3 MB. 4-of-4 PASS in 62 MB, 2-of-2 FAIL in 73 MB. The 6th
data point (Step 1 audit-ON PASS) also lands in the 62 MB bucket
— 5-of-5 PASS in 62 MB.

Hypothesis: torch.compile / inductor non-determinism produces two
binaries for the same input graph; the "larger" variant has
slightly different FP-reduction or kernel-selection behavior that
flips knife-edge formatting tokens (`Paris.\n` vs `Paris.\n\n`).

### Step Z1 — controlled cache-pin causality test (`evidence/2026-04-30-2109-pathb-z1-summary/summary.md` + 10 trial dirs)

Mounted `/root/.cache/vllm` to a host directory via a new
`PATHB_Z1_VLLM_CACHE_HOST_DIR` env hook in
`c2_full_layer_bisect.sh` (committed as `f002ee43a`). NO backend
.py changes.

Bootstrapped two cache snapshots:
- GOOD: log-reported 62.1 MB, disk 70 MB, sha256 `651e00bd...`
- BAD: log-reported 73.6 MB, disk 81 MB, sha256 `af68c498...`

Both at the same compile cache key. Distinct sha256s confirm
inductor non-determinism (same input, different output binary).

Ran 5 fresh-container trials with each cache locked.

| Cache | trials | PASS | FAIL | cache_reused | sha256 unchanged |
|---|---|---|---|---|---|
| GOOD | 5 | **5** | 0 | 5/5 | 5/5 |
| BAD | 5 | 1 | **4** | 5/5 | 5/5 |

5x PASS-rate ratio between locked-good and locked-bad. Cache
reuse confirmed by zero "saved AOT compiled function" lines, fast
~120s boot times (vs ~200s cold-compile), and pre-vs-post
sha256 identity.

**Z1 verdict: cache artifact/directory identity is the dominant
load-bearing variable for the FULL+β-coop FAIL** and explains
9/10 observed Z1 outcomes. Per the user's verdict framework:
5/5 PASS + cache_reused on all 5 → causality basically closed
for the good-cache direction. **NOT a fully closed root cause** —
the bad.5 PASS leaves a smaller (~20%) source of non-determinism
unaccounted for. This is dominant evidence for upstream
torch.compile/inductor non-determinism, not a complete RCA.

**Manifest scope note.** The Z1 mount in `c2_full_layer_bisect.sh`
pinned all of `/root/.cache/vllm`, but only one file actually
differed between the two locked snapshots: the AOT-compiled model
at `torch_compile_cache/torch_aot_compile/9a5549f2.../rank_0_0/model`.
The other three files (`modelinfos/...json`, `computation_graph.py`,
`cache_key_factors.json`) were byte-identical between good and bad
snapshots. The mount's nominal scope was broader than the
differentiator. See the Z1 summary doc's manifest section for full
sha256s.

The bad.5 surprise PASS (with sha256 verified unchanged across
the trial) suggests a smaller secondary source of
non-determinism — possibly CUDA scheduler ordering, atomic-add
winner ordering, or a similar low-magnitude perturbation source —
that occasionally lets the bad artifact still produce a coherent
result. Flagged as residual risk for any production deployment.

### Updated production fix shape (replaces v3-as-mlp_partial-reset)

**Persist a known-good torch.compile AOT cache across container
starts, with fail-closed handling and a probe-off validation
gate.** This is the next production change. NOT executed in Path
B's evidence work — deferred for a follow-up doc/code change.

Concrete next moves (deferred):

1. **Mount fail-closed in production serve scripts**
   (`scripts/serve-cute*.sh`). Currently only the spike
   `c2_full_layer_bisect.sh` has the env hook (`f002ee43a`). The
   production wrapper must NOT rely on a bare RW mount: it must
   verify expected AOT model path + sha256 before launch, refuse
   to start on missing/empty/mismatched cache, and switch to RO
   mount after bootstrap so a misbehaving in-flight compile
   cannot corrupt the canonical artifact.
2. **Bootstrap-and-validate workflow.** A one-shot script that
   boots with all CUTE_* probes set to 0, runs lower-8
   c2_replay_coherence, only blesses the cache if `unique=1` and
   `cross_prompt=independent`, and records the AOT model path +
   sha256 + size as the blessed manifest. Or seed from a
   verified-good build artifact.
3. **Probe-off validation gate (REQUIRED).** Z1 trials ran with
   `CUTE_WO_RESET_LOG=1` (X carryover) and
   `CUTE_FULL_GRAPH_PROBE=1` (hardcoded in
   `c2_full_layer_bisect.sh:86`). Probe state is NOT zero, and
   hot-path probes have flipped past results. Production blessing
   must validate locked-good with all CUTE_* probes off to confirm
   the artifact-identity finding survives in the diagnostic-clean
   configuration.
4. **Per-config cache scope.** This experiment used
   Qwen3.5-27B-NVFP4 + lower-8 β-coop + FULL_AND_PIECEWISE + the
   torch/vLLM build on this branch. Each combination has its own
   cache key — the workflow needs an explicit (model, backend,
   layer set, cudagraph mode, torch/vLLM build sha) → blessed
   cache + sha256 manifest, plus a stale-detection step.
5. **Track upstream torch.compile / inductor non-determinism.**
   Same-input-graph-different-output-binary is the underlying
   issue. We don't have a reduced repro yet; if/when we do, a bug
   report to torch is the long-term remediation path.

### Status (post-Path B, 2026-05-01)

- Branch HEAD: `b2677abd3` (Path B + Z1 evidence committed).
- v1 + v2 patches: shippable as PIECEWISE-clean infra. The
  v2 captured wo_output reset op continues to fire correctly
  under FULL — it's just not load-bearing for FULL-coherence.
- FULL+β-coop blocker: still **OPEN for production use**. Z1
  shows we *can* produce a 100% PASS configuration by pinning
  the right artifact, but we have not yet built the
  cache-lock/bootstrap workflow that would make this safe in
  production.
- Production path: PIECEWISE + β-coop (v0.3.0 status quo) — no
  change.
- v3 (mlp_partial_fp32 host-reset): **orphaned**. Do not pursue.

### Residual risks

1. **bad.5 PASS** — there's a smaller (~20%) source of
   non-determinism that even a locked-bad artifact can occasionally
   evade. Not characterized. If the locked-good production fix
   isn't 100% reliable in long-running deployment, this is the
   next thing to investigate.
2. **Per-config cache invalidation** — model upgrades, layer-set
   changes, cudagraph mode changes, or torch/vLLM version bumps
   all invalidate the cache. The bootstrap workflow needs a
   stale-detection/re-bless step.
3. **Latent `residual_output = torch.empty(...)` fallback** at
   `phase_e_kernel.py:2868`. Not firing on the β-coop path today,
   but should be hardened to a defensive raise. Deferred.
