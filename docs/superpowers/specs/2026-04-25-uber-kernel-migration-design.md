# Uber-kernel migration — completing the β-coop substitution

**Date:** 2026-04-25
**Branch (target):** `feat/uber-kernel-migration` (to be created off `main`)
**Target:** Decode-path full-attention layer for Qwen3.5-27B on SM120 (DGX Spark, GB10).
**Scope:** Restructures decode dispatch so β-coop is the single full-attn-decode kernel. Prefill out of scope.

---

## TL;DR

The β-coop kernel was meant to **replace** `paged_attention_forward` for full-attn decode, but the substitution was never finished and a second cross-layer state mistake sat undiagnosed for weeks. Two distinct bugs:

1. **Buffer aliasing**: `paged_attention_forward` runs first and populates `self.residual_output` with the post-Phase-C residual. β-coop then reads that buffer as `residual_in` and re-runs Phase C, double-counting `wo_out` per layer. Cascade through 16 fused full-attn layers → gibberish + ~15 ms/layer regression.
2. **Stale layer-LN bake**: β-coop's Phase 4 ε epilogue bakes layer N+1's input_LN into `next_hidden_scratch`. In Qwen3.5's stride-4 layer pattern, layer N+1 is **always linear-attn** — and linear-attn layers don't honor the F.1 skip-op, so they re-apply input_LN over the pre-baked output, corrupting the residual stream. The bake has never worked correctly for this model.

This migration fixes both at the structural level. β-coop becomes the only full-attn-decode kernel — **single cooperative-launch path, no sibling kernels, no fallback**. `paged_attention_forward` retires from the decode path (kept only for prefill). β-lite deletes entirely; its function is absorbed by β-coop after a Phase 1 SMEM shrink lifts the cooperative-launch resident_cap to cover num_seqs ≤ 4. Anything beyond cap raises and refuses to serve, per Q5=A. **Phase 4 deletes entirely** (per fresh-eyes spec audit Finding 1: keeping Phase 4's mlp_out add into residual_output causes layer N+1's `input_layernorm` to double-count mlp_out via standard fused-residual semantics). β-coop's output is `(mlp_output, residual_output=residual_post_attn)`; layer N+1 computes the sum itself. **The Phase 4 ε bake is gone** — every layer (full or linear) runs `input_layernorm` at its own entry, matching every surveyed hybrid SSM/attention model. The F.1 skip-op + cross-layer flag plumbing all delete.

Correctness gate: `gsm8k_eval_50.py seed=42` ≥ 90% under PIECEWISE CUDA graphs.
Perf gate: sustained tok/s with β-coop ON ≥ current production baseline (β-OFF).

---

## Background

### How the bug was found

2026-04-25 PM session traced the post-F.1 gibberish (`project_phase_e_phantom_speedup`, `project_phase_e_beta_math_bug`) past the math layer. The `(1+γ)` math fixes shipped 2026-04-24 morning across all 9 sites — verified live — yet β-coop ON still produced gibberish. Past-commit review (`9f39b86ef` F.1 wiring, `c2a6d8766` math fix, `b73af0d23` compile cache) surfaced that the regular `paged_attention_forward` uber-kernel runs at `_backend.py:1034` whenever `use_fusion=True`, and β-coop reads `residual_in=self.residual_output[:nat]` at `_backend.py:1175`. The regular path's Phase C populated `self.residual_output` with `attn_out + h + r`; β-coop then computed `new_res = wo_out + (attn_out + h + r) = 2·attn_out + h + r`, double-counting one attention pass per fused layer.

The math at every site was correct. The buffer-stage plumbing was wrong.

### Why this is structural, not surface

A one-line fix (`residual_in=self.residual_buf[:nat]`) recovers correctness but leaves the +15 ms/layer regression in place — both kernels still compute Phase A+B+C every step. The duplication is the disease; the buffer aliasing is the symptom. β-coop was conceived as the unified Phase-0-through-4 uber-kernel (`project_uber_kernel`); `paged_attention_forward` is the previous-generation kernel that should have retired when β-coop landed but didn't.

This migration completes that retirement.

---

## Decisions

| # | Question | Choice | Rationale |
|---|---|---|---|
| Q1 | Migration scope | **Maximal** — β-coop is THE full-attn-decode kernel | "Dream box" framing; no parallel paths |
| Q2 | Prefill | **Stays separate** — `paged_attention_forward` keeps prefill | Prefill is a different compute pattern; specializing β-coop for both bloats it |
| Q3 | β-lite role | **Kill β-lite entirely. β-coop cooperative-only with Phase 1 SMEM shrink to lift resident_cap. Hard-cap num_seqs ≤ 4; beyond cap raises.** (revised 2026-04-25 PM after spec audit Finding 2 + user direction "no sibling kernels.") | Two-launch-mode "C" was unsound: β-coop's atomic-counter spin-wait grid barrier requires CTA co-residency, which only cooperative launch guarantees. Non-cooperative launch deadlocks the existing barrier. User picked the consolidation path: shrink Phase 1 SMEM (~45 KB → ~17 KB via packed-FP8 K/V ping-pong + smaller tiles + Q packing), lift resident_cap to ~288 (= 6 CTAs/SM × 48 SMs), cover num_seqs=1-4 cooperatively. **The num_seqs ≤ 4 cap matches the shipping config — `scripts/serve.sh:52` defaults `MAX_NUM_SEQS=4`** — so the hard-cap is not a production regression, it's the ceiling we already serve at. num_seqs > 4 raises (per Q5=A no-fallback). |
| Q4 | Phase 0 / Phase 4 boundary | **Drop the bake — per-layer input_LN at layer entry** (revised 2026-04-25 PM after self-review + brainstorm agent) | Bake corrupts residual stream for every full→linear boundary in stride-4 (16/16 fused layers affected). Cited "300-500 µs/token saving" was illusory. Pattern matches every surveyed hybrid model (Jamba, Zamba2, Qwen3-Next, Megatron hybrid). See `q4_brainstorm_layer_LN_2026-04-25.md`. |
| Q5 | `paged_attention_forward` retirement | **Retire entirely for decode**, no fallback | Silent fallbacks mask regressions (the failure mode that hid this bug for weeks). Reliability lives in CI, not at runtime. |

---

## Architecture

**Single decode path, cooperative-only.** β-coop (`PhaseE_Beta_Kernel.run_beta_coop_full` in `phase_e_kernel.py`) is the only full-attn-decode kernel. Cooperative launch only — the existing atomic-counter spin-wait grid barrier requires CTA co-residency, which non-cooperative launch doesn't guarantee (per spec audit Finding 2).

**SMEM shrink to lift resident_cap.** Phase 1 SMEM shrinks from ~45 KB → target ~17 KB via packed-FP8 K/V ping-pong, smaller Q tile, and async double-buffering. Resident_cap rises from 96 → ~288 (= 6 CTAs/SM × 48 SMs). Covers `num_seqs ≤ 4` cooperatively (256 CTAs ≤ 288). The shrink work lands in this branch; without it, the migration only covers num_seqs=1.

**Hard cap at num_seqs=4.** Matches `scripts/serve.sh:52` default `MAX_NUM_SEQS=4`. Anything beyond `64 × num_seqs > resident_cap` raises a clear error at dispatch time and refuses to serve (per Q5=A no-fallback). The cap is checked in `_backend.forward()` before the launch; it's a hard precondition, not a runtime degradation.

**`paged_attention_forward` retires for decode.** Its file (`vllm/v1/attention/backends/cute_paged/kernel.py`) stays in tree and continues to handle prefill calls. The decode dispatch in `_backend.forward()` routes 100% to β-coop.

**Per-layer self-contained semantics, no cross-layer state.** Every layer (full-attn or linear-attn) runs `self.input_layernorm(hidden, residual)` at its own entry. β-coop's output is `(mlp_output, residual_output where residual_output = residual_post_attn)` — Phase 4 deletes entirely (per audit Finding 1: any in-place mutation of residual_output to add mlp_out causes layer N+1's input_LN to double-count). Layer N+1's `input_layernorm` computes the residual+mlp sum itself, matching the unfused flow exactly. The F.1 `cute_phase_e_skip_input_layernorm` op deletes entirely; `cute_phase_e_dispatch` simplifies (no skip-flag plumbing, no LN-baked buffer). Cross-layer state machinery (`_phase_e_skip_next_ln`, `_input_layernorm_module`, `_emit_next_layernorm`, `next_hidden_scratch`, `attach_input_layernorm`, `attach_next_input_layernorm`) all delete. Matches Qwen3-Next upstream pattern.

**β-lite deletes entirely.** Function absorbed by β-coop after the SMEM shrink lifts the cap. The β-lite dispatch block, the `_phase_e_use_beta_lite` flag, the try/except β-lite fallback, and the `Phase_D_MLP_Kernel` invocation paths used by β-lite all go away from the decode path. (`Phase_D_MLP_Kernel` itself stays in tree if reachable from prefill / non-CuTe paths; in this branch we don't audit those.)

**Linear-attention layers unchanged.** β-coop is full_attn-only; linear_attn keeps its existing path.

**No fallback.** β-coop launch failure (compile error, runtime fault, dispatch mismatch, num_seqs > cap) raises and kills the request. Reliability lives in CI/tests, not in silent runtime safety nets.

---

## Components

### Modified

**`vllm/v1/attention/backends/cute_paged/phase_e_kernel.py`**
- Single `@cute.kernel` body for Phases 0-3 (Phase 4 deletes); one `_jit_launch_phase_0_to_4` renamed/reduced to `_jit_launch_phase_0_to_3`.
- Cooperative launch only; existing grid barrier stays as-is.
- Reads `residual_in` from a caller-supplied buffer (no longer aliases the legacy kernel's output).
- **Phase 4 deletes entirely** — Phase 1C already produces `residual_post_attn` in `residual_output`. The kernel returns at end of Phase 3. No LN bake, no in-place add of mlp_out. β-coop output is `(mlp_output, residual_output=residual_post_attn)`.
- **SMEM shrink** in Phase 1: packed-FP8 K/V ping-pong with `cp.async.cg`, halved Q SMEM via FP8 storage + dequant on read, smaller `tile_s` if needed. Target ~17 KB Phase 1 SMEM. Drops `next_input_layernorm_gamma`, `next_hidden_output`, `emit_next_layernorm` parameters from `run_beta_coop_full`.

**`vllm/v1/attention/backends/cute_paged/_backend.py`**
- Drops the `paged_attention_forward()` call for decode-only paths.
- Keeps it for prefill (`is_decode_only == False`).
- Hard cap: `assert 64 × num_seqs ≤ resident_cap` before β-coop launch; raise `RuntimeError` with clear message if num_seqs exceeds cap.
- β-coop launch reads `residual_in=self.residual_buf[:nat]` (was `self.residual_output[:nat]` — the bug).
- Sets `self._phase_e_consumed = True` after launch; raises on launch failure (no β-lite fallback chain).
- **Deletes**: `attach_input_layernorm`, `attach_next_input_layernorm`, `_emit_next_layernorm`, `_input_layernorm_module`, `_next_input_layernorm_module`, `_phase_e_skip_next_ln`, `next_hidden_scratch` allocation, `_phase_e_use_beta_lite` flag, β-lite dispatch block, `_use_beta_lite` cascade.
- **Redefines** `_fusion_active`: post-migration, "fusion active" means "β-coop will fire this step" (not "the legacy uber-kernel populated buffers"). Update the load-bearing `and use_fusion` comment at `_backend.py:1108-1110` (per audit Finding 5).

**`vllm/v1/attention/backends/cute_paged/_mlp_op.py`**
- `cute_phase_e_dispatch` simplifies: consume branch reads `(mlp_output, residual_output=residual_post_attn)` from β-coop output. Doesn't set `_phase_e_skip_next_ln`. `hidden_out` reads from `mlp_output`, not `next_hidden_scratch`.
- **`cute_phase_e_skip_input_layernorm` op deletes entirely.**

**`vllm/nvllm/models/qwen3_5.py`**
- Residual mirror at line 460 (already correct).
- Layer forward input_LN gate (lines 421-441) collapses: every layer (full-attn or linear-attn) runs `self.input_layernorm(hidden, residual)` unconditionally. F.1 skip-op call site deletes.
- `Qwen3_5Model.__init__` post-hook: drops `attach_input_layernorm` loop (lines 636-647) and `attach_next_input_layernorm` loop (lines 656-676). Phase E binding becomes unnecessary.
- Layer-0 special case (`residual is None`) unchanged.

### Unchanged

**`vllm/v1/attention/backends/cute_paged/kernel.py`** — `paged_attention_forward` and `DecodeKernel` stay; reachable only from prefill after migration.

**`vllm/v1/attention/backends/cute_paged/mlp_kernel.py`** — `Phase_D_MLP_Kernel` and `cute_mlp_forward` op stay; reachable only from non-decode paths (prefill, debug).

### Deleted

- `_phase_e_use_beta_lite` flag and its if-cascade in `_backend.forward`.
- The β-lite dispatch block.
- The `paged_attention_forward` invocation from the decode-only branch.
- The `try/except → β-lite` fallback ladder.
- **F.1 layer-LN bake plumbing**: `cute_phase_e_skip_input_layernorm` op, `attach_input_layernorm`, `attach_next_input_layernorm`, `_phase_e_skip_next_ln` flag, `_input_layernorm_module` field, `_next_input_layernorm_module` field, `_emit_next_layernorm` flag, `next_hidden_scratch` buffer.
- **Phase 4 ε epilogue entirely** — the standalone `run_phase_4_only` kernel and the corresponding `if emit_next_ln == Int32(1):` block at `phase_e_kernel.py:4631-4652` and the in-place mlp_out add at `phase_e_kernel.py:~4655` (the `else` branch that writes residual_final without LN bake — also gone since Phase 4 doesn't fire).

---

## Data flow

### Per-layer flow (one full-attn decode call, layer N+1)

```
qwen3_5.py:
  inputs: (hidden_states, residual) from layer N
          = (mlp_out_N, residual_final_N) when prev fired (full or linear)
          = (hidden_initial, None)        when layer 0

  step 1: input_layernorm — UNCONDITIONAL for ALL layer types
            if residual is None:           # layer 0
              residual = hidden_states
              hidden_states = self.input_layernorm(hidden_states)
            else:
              hidden_states, residual = self.input_layernorm(hidden_states, residual)

  step 2: residual mirror
            impl.residual_buf[:nat].copy_(residual[:nat])
            ↑ POST-input-LN residual = the buffer β-coop reads as residual_in

  step 3: attention forward → _backend.forward()
            β-coop launch:  residual_in = self.residual_buf[:nat]
                            (was self.residual_output, the buffer-aliasing bug)

  step 4: F.1 dispatch op  cute_phase_e_dispatch
            consumes β output:
              hidden_out  ← self.mlp_output         (raw mlp_out)
              residual_out ← self.residual_output   (= residual_post_attn = attn+h+r)
                                                    NOT residual_final
                                                    (Phase 4 deleted; layer N+1's
                                                     input_layernorm computes the
                                                     residual+mlp sum itself)

  outputs: (hidden_out, residual_out) → layer N+2
           layer N+2 runs its own input_layernorm at entry — combines hidden+residual
           into residual_final there, exactly as the unfused flow does
```

### β-coop internal data flow

```
                       ┌──────────────────────────────┐
                       │ β-coop @ layer N+1           │
  residual_buf  ──────►│ Phase 0: dummy (side-channel │
  (post-input-LN)      │          for future QKV-     │
                       │          fusion; not consumed)│
                       │                              │
                       │ Phase 1A: attn(query, K, V)  │
  query, kv_cache ────►│ Phase 1B: W_O matmul         │──► (internal: wo_output)
                       │ Phase 1C: residual_buf +     │──► attn_output (LN-applied)
                       │           wo_output + LN     │──► residual_output (= attn+h+r)
                       │                              │
                       │ Phase 2: grid barrier        │
                       │   cooperative=True (REQUIRED) │
                       │   atomic counter + spin-wait │
                       │                              │
                       │ Phase 3: MLP D               │──► mlp_output
                       │   (gate, up, down GEMV)      │
                       │                              │
                       │ Phase 4: DELETED             │
                       │   (no in-place mlp_out add;  │
                       │    no LN bake; layer N+1's   │
                       │    input_layernorm does the  │
                       │    residual + mlp sum)       │
                       └──────────────────────────────┘
                       β-coop output: (mlp_output, residual_output=residual_post_attn)
```

### Buffer contracts (after migration)

| Buffer | Role | Written by | Read by |
|---|---|---|---|
| `residual_buf` | Post-input-LN residual (β-coop INPUT) | `qwen3_5.py:460` mirror | β-coop Phase 1C |
| `residual_output` | residual_post_attn = attn+h+r (β-coop OUTPUT) | β-coop Phase 1C only | F.1 dispatch op → next layer's input_LN |
| `mlp_output` | Phase 3 output (β-coop OUTPUT) | β-coop Phase 3 | F.1 dispatch op → next layer |
| `attn_output` | Phase 1C output, intra-kernel scratch | β-coop Phase 1C | β-coop Phase 3 |

Pre-migration, `paged_attention_forward` Phase C also wrote to `residual_output`. Post-migration, β-coop's Phase 1C is the sole writer (Phase 4 is deleted; no in-place mutation). **`next_hidden_scratch` is deleted entirely.** F.1 dispatch op consumes `(mlp_output, residual_output=residual_post_attn)` and returns those to the model forward; layer N+1's `input_layernorm` does the residual+mlp sum.

### Layer-0 and last-layer edges (uniform)

- **Layer 0** (`residual is None`): `qwen3_5.py` runs `input_layernorm(hidden)` (no residual) → `residual = hidden_initial`. Then β-coop runs normally. No special-casing.
- **Last layer** (idx 63): same as any other fused layer. β-coop returns `(mlp_output, residual_output)`. The model's `Qwen3_5Model.forward` then runs the final `self.norm` over the residual stream as it does today; no per-layer bake to gate on. `_emit_next_layernorm` flag is gone.

### Math sanity

For β-coop's Phase 1C: `residual_output = residual_buf + wo_out = (h+r) + attn_out = residual_post_attn`. ✓

β-coop returns `(mlp_output, residual_output=residual_post_attn)`.

For layer N+1's `input_layernorm(mlp_output, residual_post_attn)` per `_forward_static_with_residual`:
```
combined = mlp_output + residual_post_attn = mlp_out + attn_out + h + r = residual_final
residual_new = combined  (the new residual_final)
x_new = LN(combined) · (1+γ_{N+1})
```
Matches the standard unfused transformer flow exactly.

**Pre-migration bug history:**
- Buffer aliasing: β-coop's Phase 1C did `residual_post_attn = self.residual_output + wo_out`, where `self.residual_output` was already-written by `paged_attention_forward` Phase C → `2·attn_out + h + r`. Fixed by C1 (read `residual_buf` instead).
- Phase 4 double-add (audit Finding 1): β-coop's Phase 4 did `residual_final = residual_post_attn + mlp_out` and returned that as residual; layer N+1's input_LN did `combined = mlp_out + residual_final = 2·mlp_out + ...`. Fixed by C1.5 (delete Phase 4).
- F.1 layer-LN bake (Q4 self-review): β-coop's Phase 4 baked `LN(residual_final)·(1+γ_{N+1})` into next_hidden_scratch, but layer N+1 was always linear-attn (Qwen3.5 stride-4) which re-applied input_LN, corrupting. Fixed by C1.5 (delete the bake + skip-op + plumbing).

---

## Error handling & invariants

### Failure-mode policy: fail loud

Per Q5=A, no silent fallback.

| Failure mode | Detection | Action |
|---|---|---|
| β-coop compile fails (first call) | `cute.compile()` raises | Raise; fix in code |
| Cooperative launch can't fit (occupancy mis-predicted) | `cudaLaunchCooperativeKernel` returns error | Raise; dispatch heuristic was wrong, fix it |
| Buffer pointer aliasing | Caught by L2 buffer-contracts test | CI failure pre-merge |
| Resident-cap drifts mid-decode (long context grows) | dispatch re-checks per-call | Falls through to non-cooperative on next call |
| Math drift vs Python reference | `CUTE_DEBUG_FUSION=1` reference-diff in CI | CI failure |

### Invariants

**I1. `residual_buf` is the post-input-LN residual at β-coop launch.** Single writer (`qwen3_5.py:460`), single reader (β-coop's `residual_in_ptr`).

**I2. `residual_output` is exclusively β-coop's output.** Pre-migration, `paged_attention_forward` was also a writer (the bug). Post-migration, β-coop's Phase 1C and Phase 4 are the only writers.

**I3. `_phase_e_consumed` is always cleared after dispatch (both branches).**

**I4. Every layer runs `input_layernorm` at its own entry, regardless of layer type.** No cross-layer LN-bake. Verified by reading any single layer's forward() in isolation.

(Pre-revision I3 "cross-impl flag plumbing" and I5 "stride-4 boundary state" both delete with the bake — there's no flag and no state to leak.)

### Fragile points

1. **Phase 2 grid barrier under `cooperative=False`** — load-bearing technical assumption. If the CuTe DSL grid_sync primitive doesn't compile under non-cooperative launch, the kernel needs a real branch with two implementations. **1-day spike before C3** to confirm.
2. **Resident-cap probe is SMEM-conservative.** `floor(smem_per_sm / smem_bytes) × num_sms` = `floor(102400 / 45568) × 48` = 96. Future SMEM shrink lowers `smem_bytes`, raising cap. Per-call check at line 1135 must stay in sync with launch-mode selector.

---

## Test strategy

### Test pyramid

**L0 — Phase unit tests** *(extend existing)*
- `run_phase_0_only`, `run_phase_01_only`, `run_phase_3_only`, `run_phase_4_only` — synthetic inputs, compare against Python reference.
- **NEW**: `cooperative=True` vs `cooperative=False` parity at every phase test (Q3=C regression catch).

**L1 — β-coop full kernel test** *(extend existing)*
- `run_beta_coop_full` end-to-end with synthetic inputs.
- **NEW**: composed-reference diff. Run β-coop, then run Python reference of the full Phase 0→4 chain (input_LN → attn → W_O → post_attn_LN → MLP → next_input_LN), `torch.allclose` in BF16.

**L2 — Backend integration test** *(NEW — most important)*
- Mount real `CutePagedAttentionImpl`, real attn metadata, real model weights (small toy 2-layer model).
- Fire `_backend.forward()` end-to-end.
- Assert: β-coop's `residual_in` was sourced from `self.residual_buf`. β-coop's outputs landed in `self.residual_output` (= residual_post_attn) and `self.mlp_output`. F.1 dispatch op consumed those (no `next_hidden_scratch` post-C1.5).
- Compare layer-N output (post-dispatch) against Python reference chain.
- **Had this existed pre-F.1, it would have caught the gibberish bug immediately.** This is the single most important test addition.

**L3 — Multi-layer flow test** *(NEW, simplified after Q4=B revision)*
- 4-layer toy: linear, linear, linear, FULL, linear, linear, linear, FULL.
- Two consecutive `Qwen3_5Model.forward()` calls (simulates two decode steps).
- Assert per layer: every layer (full or linear) runs `input_layernorm` at entry. β-coop's Phase 4 output is `residual_final`, not LN-baked. Compare end-to-end output against unfused reference.
- Verifies I4 — no cross-layer state to verify.

**L4 — End-to-end correctness gate**
- `scripts/gsm8k_eval_50.py seed=42` ≥ 90% with `CUTE_PHASE_E_FUSION=1`.
- Required for branch merge.

**L5 — End-to-end perf gate**
- Sustained tok/s under PIECEWISE with β-coop ON ≥ current production baseline (β-OFF).
- nsys trace committed under `benchmarks/nvllm/traces/uber_kernel_migration/<date>/` with `summary.md`.
- Required for branch merge.

### Per-commit gates

| Commit | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| C1: residual_buf bandaid (β-coop AND β-lite) + L2 test | ✓ | ✓ | new | — | ✓ | (deferred) |
| C1.5: delete Phase 4 + delete F.1 layer-LN bake plumbing + per-layer input_LN | ✓ | ✓ | ✓ | ✓ | ✓ | (deferred) |
| C2: drop paged_attention_forward call from decode + redefine `_fusion_active` | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| C3: SMEM shrink Phase 1 + delete β-lite + hard-cap num_seqs=4 | ✓ | new | ✓ | ✓ | ✓ | ✓ |
| C4: cleanup orphaned buffers / flags / dead imports | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

C1 lands the L2 integration test as part of the bandaid so structural commits can rely on it. C1.5 deletes the bake plumbing AND Phase 4 (the in-place mlp_out add per audit Finding 1) — large delete, no addition; ships per-layer input_LN by collapsing the gate. C2 retires paged_attention_forward from decode and redefines the `_fusion_active` invariant (per audit Finding 5). **C3 is the SMEM shrink + β-lite delete + hard-cap commit** — biggest change, gated on a fresh L1 (β-coop with shrunken Phase 1 SMEM still produces correct math at all phase tests). C4 only happens after L4+L5 pass on C3.

**C1.5 detail:**
- Delete `cute_phase_e_skip_input_layernorm` op (`_mlp_op.py:239-301`).
- Delete `attach_input_layernorm` (`_backend.py:507-519`) and `attach_next_input_layernorm` (`_backend.py:433-505`); drop `next_hidden_scratch` allocation (`:481-484`).
- Delete `_emit_next_layernorm`, `_phase_e_skip_next_ln`, `_input_layernorm_module`, `_next_input_layernorm_module` fields from impl init.
- **Delete Phase 4 entirely** — both the `if emit_next_ln == Int32(1):` LN-bake branch AND the `else:` raw-residual branch at `phase_e_kernel.py:4631-4674`; the kernel returns at end of Phase 3.
- Drop `next_input_layernorm_gamma`, `next_hidden_output`, `emit_next_layernorm` args from `run_beta_coop_full` (`phase_e_kernel.py:2685-2732`).
- Drop `attach_input_layernorm` (lines 636-647) + `attach_next_input_layernorm` (lines 656-676) loops from `Qwen3_5Model.__init__`.
- Collapse layer forward input_LN gate (`qwen3_5.py:421-441`) to unconditional `self.input_layernorm(hidden, residual)` for the non-first-layer branch (drop the `_mlp_layer_name` check entirely).
- Update `cute_phase_e_dispatch` consume branch: `hidden_out` reads from `mlp_output`, `residual_out` reads from `residual_output` (= residual_post_attn from Phase 1C, NOT residual_final).

**C3 detail (SMEM shrink + β-lite delete):**
- Phase 1 SMEM shrink: pack K/V to FP8 storage with `cp.async.cg` ping-pong double-buffer (~halves K + V SMEM); pack Q to FP8 storage with dequant-on-read (~halves Q SMEM); evaluate smaller `tile_s` if needed. Target ~17 KB Phase 1 SMEM.
- Re-probe `resident_cap` after shrink; expect ~288.
- Add `assert 64 × num_seqs ≤ resident_cap` precondition with clear `RuntimeError` message at β-coop dispatch.
- Delete β-lite invocation block from `_backend.forward` (`_backend.py:1222-~1290`).
- Delete `_phase_e_use_beta_lite` flag and the if-cascade gating it.
- Verify L0 phase tests still pass with shrunken Phase 1.

### Out of scope

- **FULL CUDA graph mode** — platform-blocked per `project_full_graph_blocked`. PIECEWISE only.
- **Prefill behavior** — stays on `paged_attention_forward`.
- **Linear-attention layers** — untouched.
- **Eager mode** — broken on SM120 per `project_eager_baseline_broken`; not a regression target.

---

## Open risks

1. **Phase 1 SMEM shrink to ~17 KB is the load-bearing technical assumption.** Without it, β-coop only fits num_seqs=1 cooperatively and the migration scopes to single-seq only — leaving num_seqs=2-4 (production range per `serve.sh:52`) unsupported. Realistic intermediate target ~32 KB unlocks num_seqs=2; reaching ~17 KB for num_seqs=4 needs aggressive packed-FP8 K/V + Q packing + possibly smaller `tile_s`. Risk: shrink hits a correctness or perf wall before reaching target. Mitigation: ladder the shrink — first commit to ~32 KB (num_seqs=2), measure, then push for ~17 KB (num_seqs=4); if ~17 KB blocks, ship at num_seqs=2 cap and document as known limit.

2. **L5 perf gate not yet measured.** External review estimates +15 ms/layer recovery from removing the paged_attention_forward double-fire (C2's payoff). C1.5's per-layer input_LN adds back ~7 µs/layer of input_LN compute, mostly absorbed by the bake's removal. C3's SMEM shrink may have its own perf signal (cp.async overlap can speed up Phase 1 even at constant SMEM). Expected net: comfortably positive vs current β-OFF baseline; unverified until C3 lands.

3. **Resident-cap heuristic sensitivity.** `_probe_resident_cap` uses SMEM-only fallback. Real `cuOccupancyMaxActiveBlocksPerMultiprocessor` runs only after first compile. If the SMEM-only fallback diverges from real occupancy post-shrink, the hard-cap precondition might fire too aggressively or too loosely. Test in L2 + L3 with edge num_seqs (1, 4, 5) to pin down behavior.

4. **β-lite was latent-broken pre-migration** (audit Finding 6). β-lite reads `self.residual_output` at `_backend.py:1268` — same buffer-aliasing bug as β-coop. Any GSM8K evidence collected with β-lite ON in the past should not be trusted as ground truth. C1's bandaid fixes both β-coop AND β-lite to read `residual_buf` so β-lite isn't math-broken between C1 and C3-when-it-deletes.

(Pre-revision risks "grid_sync compile spike" and "cross-impl flag propagation" are gone — Q3 reverted to cooperative-only, and C1.5 deletes the flag entirely.)

---

## References

### Memory
- `project_uber_kernel_migration` — this design's home memory; brainstorm log
- `project_phase_e_beta_math_bug` (revised 2026-04-25 PM) — root cause, buffer-stage trace
- `project_phase_e_phantom_speedup` (revised 2026-04-25 PM) — destructive-era analysis
- `project_uber_kernel` — Unreal Kernel vision, Phase A/B/C/D history
- `project_fusion_debug_plan` — revised plan replacing item B with this migration

### External evidence
- `docs/research/uber_kernel_migration/external_review_A_vs_C_2026-04-25.md` — research-subagent report on FA3 / FlashInfer / TRT-LLM XQA / CUTLASS dispatch patterns (Q3 = β-lite collapse decision)
- `docs/research/uber_kernel_migration/q4_brainstorm_layer_LN_2026-04-25.md` — research-subagent report verifying Q4=A bake corruption + recommending Q4=B against Jamba / Zamba2 / Qwen3-Next / Megatron hybrid prior art

### Related specs
- `docs/superpowers/specs/2026-04-23-phase-e-d25-design.md` — β-coop kernel design
- `docs/superpowers/specs/2026-04-14-unreal-kernel-phase-de-mlp-research.md` — MLP fusion background

### Code anchors (HEAD of `fix/phase-d-mlp-decode-tile-preset`)
- `vllm/v1/attention/backends/cute_paged/_backend.py:1034` — `paged_attention_forward` call site (to be deleted for decode)
- `vllm/v1/attention/backends/cute_paged/_backend.py:1175` — β-coop `residual_in` source (to be changed `residual_output` → `residual_buf`)
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2685` — `run_beta_coop_full` entry
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:3849, 3901, 3945` — Phase 1C residual-add sites
- `vllm/v1/attention/backends/cute_paged/_mlp_op.py:192-218` — `cute_phase_e_dispatch_impl`
- `vllm/v1/attention/backends/cute_paged/_mlp_op.py:248-283` — `cute_phase_e_skip_input_layernorm_impl`
- `vllm/nvllm/models/qwen3_5.py:380-450` — layer forward including F.1 op call sites
- `vllm/nvllm/models/qwen3_5.py:460` — residual mirror

---

## Definition of done

- All 5 commits (C1, C1.5, C2, C3, C4) land on `feat/uber-kernel-migration` branch.
- Each commit independently passes its applicable test gates per the per-commit gate table.
- L4 (`gsm8k_eval_50 seed=42`) ≥ 90% on C3 with `CUTE_PHASE_E_FUSION=1` AND `MAX_NUM_SEQS=4`.
- L5 (sustained tok/s with β-ON ≥ β-OFF baseline) on C3, evidence committed under `benchmarks/nvllm/traces/uber_kernel_migration/`.
- Production serve config flips from `CUTE_PHASE_E_FUSION=0` → `=1` in C4.
- `project_uber_kernel_migration` memory updated to "MIGRATION SHIPPED" with commit hashes and trace links.
- PR opened on `Navi-AI-Lab/nvllm` only (per `feedback_never_touch_upstream_vllm`).
- **Hard-cap behavior verified**: launching with `num_seqs=5` raises a clear error message and refuses to serve (no silent degradation).
