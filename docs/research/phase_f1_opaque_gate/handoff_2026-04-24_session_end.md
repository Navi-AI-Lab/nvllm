# Phase E.2 + F.1 — End-of-session handoff (2026-04-24)

**Session goal:** Execute the 19-task plan at
`docs/superpowers/plans/2026-04-24-phase-e2-f1-beta-correctness-opaque-gate.md` —
Phase E.2 (β kernel math fix: raw γ → (1+γ)) + Phase F.1 (replace dead-branching
Python gates with opaque custom ops so β output gets consumed under PIECEWISE).

**Where we landed:** Tasks 1–14 + 15a (β-lite GSM8K 8/8) **shipped**. Tasks
15b–19 **blocked** by an upstream-class crash that capped the session.

---

## 1. What's done — 8 commits

| # | SHA | Summary |
|---|---|---|
| 1 | `98551dba6` | E.2 #1 — β-lite ε epilogue uses (1+γ); new test against `Qwen3_5RMSNorm` |
| 2 | `c2a6d8766` | E.2 #2 — β-coop all 7 phases + standalone DecodeKernel + 2 bad refs to (1+γ) |
| 3 | `c3b643eca` | docs — E.2 reference-diff harness README |
| 4 | `729c5e733` | F.1 Layer 1 — Python op-registration repro |
| 5 | `9cf3bcf91` | F.1 — `cute_phase_e_dispatch` + `cute_phase_e_skip_input_layernorm` ops |
| 6 | `9f39b86ef` | F.1 — wire opaque ops into Qwen3_5 decoder |
| 7 | `325f4bb9a` | F.1 fix — split attach loop so `input_layernorm` always attaches |
| 8 | `437d20971` | docker — bump flashinfer-python pin 0.6.3 → 0.6.7 (didn't fix the wedge; see §3) |

Test green count: **14 unit tests** (β-lite + β-coop math) + **5 unit tests**
(F.1 ops) + **8/8 GSM8K** under autotune-disabled config = 27/27.

## 2. Plan status

- ✅ **Tasks 1–14**: math fixes + opaque ops + decoder wiring all committed.
- ✅ **Task 15a**: β-lite Docker rebuild + GSM8K 8/8 PASS — *with* the
  workaround `--kernel-config '{"enable_flashinfer_autotune":false}'`. All 8
  questions correct (Q1–Q8: 72/10/5/42/624/35/48/16). Server stayed alive
  for the full ~42 min benchmark. ~0.86 tok/s — slow because autotune was off,
  but math + wiring are end-to-end correct.
- ⏸️ **Task 15b**: β-coop GSM8K — NOT RUN. Need this once the wedge is fixed
  for performance to be meaningful.
- ⏸️ **Tasks 16–19**: nsys kernel-count trace, layer-by-layer numerical
  equivalence, evidence bundle, memory updates — all blocked on a serving
  setup that's both fast AND stable.

## 3. The crash — what we know

### Signature

Every container start with `--cudagraph_mode PIECEWISE`, regardless of CuTe
fusion env vars, hits this exact pattern:

```
INFO ... [gpu_model_runner.py:5962] Estimated CUDA graph memory: -X.XX GiB total   ← NEGATIVE (canary)
INFO ... [gpu_worker.py:436] Available KV cache memory: ~40 GiB
INFO ... [kv_cache_utils.py:1431] GPU KV cache size: ~340K tokens
INFO ... [_backend.py:1531] CutePagedMetadataBuilder: block_size=64, layers=16
INFO - autotuner.py:446 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
[silence — process disappears, container exits 255 with OOMKilled=false; host kernel-panics shortly after]
```

3+ host hard-reboots in this session.

### What's been ruled out

- ❌ **Phase F.1 wiring** — confirmed by all-fusion-OFF bisect run
  (`CUTE_MLP_FUSION=0 CUTE_PHASE_E_FUSION=0 CUTE_ATTN_FUSION=0`): same wedge.
- ❌ **flashinfer 0.6.3 vs 0.6.7** — confirmed by 0.6.7 rebuild (commit
  `437d20971`, image `debb0fa1dd29`): same wedge. Flashinfer issue #2884
  hint to bump 0.6.3 → 0.6.7+ does NOT fix our specific failure mode.

### The only known workaround

`--kernel-config '{"enable_flashinfer_autotune": false}'` — server boots,
serves correctly, but ~0.86 tok/s for 27B NVFP4 on GB10 (vs the 30+ tok/s
shipping target). Saved as memory `feedback_flashinfer_autotune_sm120.md`.

### The "Why did 2026-04-23 work?" mystery

Per `memory:project_phase_e_shipped`, β-lite served fine yesterday. **No
known change between yesterday and the first crash today** in the area
that should affect this:
- The Phase E.2 kernel math changes are FP32-arithmetic-only — should not
  change memory layout.
- The Phase F.1 changes add a few ops + `torch.empty_like()` calls that
  happen OUTSIDE the captured graph (eager Python wrapper around the op).
- The image rebuild today picked the same flashinfer 0.6.3 wheel via
  cached pip layers (verified by `gh api` showing 0.6.3 was the only pin
  in the Dockerfile until commit `437d20971`).

Possibilities the next investigator should consider:
1. Yesterday's working image is still on disk — different SHA. If we tag
   it and run a SAME-image diff vs today's, we can see what really changed.
2. yesterday's serve script may have used different env vars or
   `--max-num-seqs` — check `scripts/local/serve-qwen35-*.sh` git log.
3. flashinfer JIT cache state at `~/.cache/flashinfer/0.6.3/121a/` may
   have been mutated (timestamps show `cached_ops/fp4_gemm_cutlass_sm120/`
   was touched 2026-04-24 15:03 — DURING our crashed runs).

## 4. Diagnostic signal: `Estimated CUDA graph memory: -1.72 GiB total`

This appears BEFORE flashinfer autotune in every crashed run (-1.66, -1.97,
-2.36, -1.72 across 4 runs). NEGATIVE estimate means vLLM thinks CUDA graph
capture freed memory rather than consumed it. Almost certainly an accounting
bug in `gpu_model_runner.py:5962` — the source location to inspect:

```bash
docker run --rm --entrypoint sed nvllm:gb10 -n '5950,5975p' /app/nvllm/vllm/v1/worker/gpu_model_runner.py
```

If that estimate is bogus, vLLM's downstream KV-cache size decision is
wrong, and flashinfer's workspace alloc tips the balance into kernel OOM.
Fixing the estimator might be the real fix.

## 5. Hypotheses ranked for the next session

1. **vLLM CUDA-graph memory accounting regression** — the negative-estimate
   canary points here. Search vLLM `git log` for changes to
   `gpu_model_runner.py:profile_cudagraph_memory` or similar in the last
   ~2 weeks. Yesterday's working build vs today's may differ here even
   though our local source didn't change (cached COPY layer invalidation
   could pull a different upstream main).
2. **Flashinfer JIT cache corruption** — `cached_ops/` timestamps from
   2026-04-24 13:14 / 15:03 / 15:04 (during the failed runs) are
   suspicious. Try: `mv ~/.cache/flashinfer ~/.cache/flashinfer.bak` and
   rerun without persisting the host volume mount.
3. **Phase E.2 #2 (`commit c2a6d8766`) widened the kernel size** — the
   `(Float32(1.0) + gamma_f32)` change could (in CuTe DSL) generate
   slightly more code per kernel; if compiled-kernel SMEM grew enough to
   reduce the resident-cap probe in `_backend.py:_probe_resident_cap`,
   memory accounting could go off. Bisect: `git revert c2a6d8766` and
   rebuild — if the wedge goes away, Phase E.2 is the trigger and we have
   a much narrower problem.
4. **Concurrent flashinfer autotune + CuTe DSL JIT competing for VRAM** —
   try `CUTE_PHASE_E_FUSION=0 CUTE_MLP_FUSION=0 CUTE_ATTN_FUSION=0` AND
   `--kernel-config '{"enable_flashinfer_autotune":true}'` to separate
   "is it CuTe?" from "is it flashinfer?". Already proved CuTe-off + autotune-on
   crashes, but we never tried CuTe-off + autotune-off baseline as the negative
   control.

## 6. Concrete next-session checklist

1. **Capture the working baseline first.** Find yesterday's image:
   ```bash
   docker images | grep nvllm  # look for older SHAs
   ```
   If still on disk, tag it `nvllm:gb10-2026-04-23-known-good` and run
   the SAME serve cmd. If THAT crashes too, the bug is in HW/driver/host
   state, not our build. If it works, diff the two images.
2. **If yesterday's image is gone:** bisect on `git revert` of our 8
   commits, one at a time, rebuilding each. Start with `c2a6d8766`
   (Phase E.2 #2 — the largest blast radius).
3. **Check upstream vLLM's recent gpu_model_runner.py changes** —
   the `Estimated CUDA graph memory` line was added/changed recently per
   the search results showing `# 39863 EngineCore dies silently` is
   dated 2026-04-15, very close to when this started.
4. **Try the JIT-cache-clean run** — fastest cheap test:
   `mv ~/.cache/flashinfer ~/.cache/flashinfer.bak`, rerun with full F.1
   + autotune ENABLED. If it works, JIT cache corruption is the cause
   and the workaround is to clear that cache after every code change
   (and probably warrants a dedicated invalidation strategy).
5. **Don't run more PIECEWISE tests until a hypothesis fires.** Each
   crashed run risks a host hard-reboot. Use the autotune-OFF setup
   for any CORRECTNESS work in the meantime.

## 7. Files / state references

- **Plan:** `docs/superpowers/plans/2026-04-24-phase-e2-f1-beta-correctness-opaque-gate.md`
- **Spec:** `docs/superpowers/specs/2026-04-24-phase-f1-opaque-gate-refactor-design.md`
- **E.2 audit (Batch B scope expansion):** `docs/research/phase_e2_beta_math/batch_b_audit_2026-04-24.md`
- **F.1 op-registration repro:** `docs/research/phase_f1_opaque_gate/op_registration_repro.py`
- **Op tests:** `tests/kernels/cute/test_phase_f1_opaque_gate.py` (5 tests)
- **Math tests:** `tests/kernels/cute/test_phase_e2_beta_math.py` (3 tests)
- **Existing math tests (still green):** `tests/kernels/cute/test_phase_e_epsilon_epilogue.py` (11 tests)
- **Working autotune-disabled image:** `nvllm:gb10` SHA `debb0fa1dd29` (built 17:43)
- **Workaround memory entry:** `~/.claude/projects/-home-natfii-docker-nvllm/memory/feedback_flashinfer_autotune_sm120.md`

## 8. What NOT to do

- Don't keep retrying live PIECEWISE runs hoping the wedge goes away —
  3 host reboots already, this is a deterministic upstream-class bug.
- Don't claim the upstream Flashinfer #2884 fix is the one we need —
  this session proved 0.6.7 did NOT fix it for us.
- Don't blame F.1 — the all-fusion-OFF bisect ruled that out.
- Don't ship the autotune-disabled config to production — 0.86 tok/s is
  not a viable serving setup; it's a diagnostic-only workaround.

## 9. Hand-off summary in one paragraph

The Phase E.2 + F.1 code itself is **correct and tested**: 14 unit tests
green, 5 op tests green, GSM8K 8/8 with the autotune-disabled workaround.
The blocker is an **upstream/environmental wedge** — vLLM reports a
nonsensical "Estimated CUDA graph memory: -1.72 GiB" right before
flashinfer autotune starts, and the autotuner subsequently OOMs the kernel
hard enough to take the host down. This bug is independent of our F.1
code (proven by all-fusion-OFF bisect) and is NOT fixed by upgrading
flashinfer 0.6.3 → 0.6.7 (proven by commit `437d20971` rebuild). Best
next step is to find yesterday's working image SHA and bisect what
actually changed at the system level vs git, OR clean the flashinfer JIT
cache and retest, OR investigate the negative-memory-estimate bug in
`gpu_model_runner.py:5962` directly.
