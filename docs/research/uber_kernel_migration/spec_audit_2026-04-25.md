# Spec audit — uber-kernel migration revised 2026-04-25

**Auditor:** fresh-eyes opus subagent
**HEAD audited:** `ef53c792aa328a6a19966fb23897f3929d7eb45c` (`fix/phase-d-mlp-decode-tile-preset` merged into main)
**Spec:** `docs/superpowers/specs/2026-04-25-uber-kernel-migration-design.md`

## Summary

**Confidence: LOW — the spec needs a math-flow revision before implementation.** The buffer-aliasing diagnosis and the Q4 bake-corruption analysis are correct and well-supported by live code. However, the post-Q4-revision math contains a new, structurally identical bug: Phase 4 still adds `mlp_out` into `residual_output` to produce `residual_final`, then layer N+1's `input_layernorm` adds `mlp_out` AGAIN via its standard residual-accumulation semantics — double-counting MLP. This is the same class of mistake the self-review just caught (load-bearing claim that doesn't survive contact with live code), and it would corrupt every full-attn-decode layer just as Q4=A did. There is also a separate HIGH-severity issue with Q3=C: β-coop's existing grid-barrier deadlocks under non-cooperative launch, which kills the `num_seqs ≥ 2` path the design promises. Two MED issues with test coverage and one MED with `_phase_e_active` invariants. Recommend revise spec, do not write plans yet.

## Severity-ranked findings

### Finding 1: Phase 4 + per-layer input_LN double-counts `mlp_out` (mirror of Q4=A)

- **Severity:** HIGH
- **Where:** Spec §Data flow lines 174-178, 199-201; §Per-layer flow lines 132-134, 144-148; §Buffer contracts line 186 (residual_output row).
- **Issue:** The new design has β-coop Phase 4 produce `residual_final = residual_post_attn + mlp_out` and write it to `residual_output`. Layer N+1's entry then runs `self.input_layernorm(hidden_states=mlp_out, residual=residual_final)`, which Qwen3_5RMSNorm fuses as `combined = hidden_states + residual` followed by RMSNorm. The `combined` value is therefore `mlp_out + (mlp_out + residual_post_attn) = 2·mlp_out + residual_post_attn`. The MLP contribution gets added twice on every full-attn-decode layer.
- **Evidence:** Qwen3_5RMSNorm forward semantics at `vllm/nvllm/layers/layernorm.py:58-80 (commit ef53c792a)`:
  ```python
  x = (x.float() + residual.float() if orig_dtype == torch.float16 else x + residual)
  residual = x  # NEW RESIDUAL = hidden_states + old residual
  x = x.float()
  variance = x.pow(2).mean(dim=-1, keepdim=True)
  x = x * torch.rsqrt(variance + variance_epsilon)
  x = x * (1.0 + weight.float())
  return x, residual
  ```
  The standard Qwen residual-stream protocol is: layer outputs `(hidden_out=mlp_out, residual_out=residual_post_attn)`, and layer N+1's `input_layernorm` does the residual-add to produce `residual_final`. The current legacy unfused branch at `vllm/nvllm/models/qwen3_5.py:553 (commit ef53c792a)` honors that — `hidden_states = self.mlp(hidden_states)` returns mlp_out unchanged and `residual` flows through untouched.

  In the existing β-coop+bake design, the skip-op compensates: dispatch op returns `(baked_LN(residual_final), residual_final)` AND sets `_phase_e_skip_next_ln=True`, so layer N+1's input_LN is bypassed — there's no double-add. The skip is what makes the bake math work.

  The revised spec deletes the bake AND the skip-op, but keeps Phase 4's `residual_post + mlp_out` add into `residual_output`. There's nothing left to compensate for it. Mathematical proof:
  - Standard flow (correct): layer-N output = `(mlp_out, residual_post_attn)`. Layer-N+1 input_LN: `combined = mlp_out + residual_post_attn = residual_final`.
  - Spec proposed flow (incorrect): layer-N output = `(mlp_out, residual_final)`. Layer-N+1 input_LN: `combined = mlp_out + residual_final = mlp_out + (mlp_out + residual_post_attn) = 2·mlp_out + residual_post_attn`.
- **Suggested fix:** **Drop Phase 4 entirely (or reduce it to a no-op).** Phase 1C already produces `residual_post_attn` in `residual_output`. β-coop output should be `(mlp_output, residual_output)` where residual_output = residual_post_attn, NOT residual_final. Let layer N+1's `input_layernorm` do the residual+mlp accumulation as in the standard flow. Update §Data flow and §Math sanity to reflect this. This is a structural change to the migration; the spec must be revised before C1.5 or C2 land.

### Finding 2: Q3=C non-cooperative launch deadlocks under the existing grid barrier

- **Severity:** HIGH
- **Where:** Spec §Architecture lines 56-61; §Open risks #1 (line 307); §Per-commit gates row C3 (line 281).
- **Issue:** The spec proposes that β-coop's body works under `cooperative=False` for `num_seqs ≥ 2` cases by using "atomic-counter spin-wait on a global counter (the existing `grid_barrier_i32` pattern)." That spin-wait already exists in the kernel and **requires all CTAs to be co-resident on the SMs** to avoid deadlock. Cooperative launch guarantees co-residency; non-cooperative does not — the driver may schedule subsets of CTAs sequentially. For `64·num_seqs > resident_cap (=96 today)`, non-cooperative launch will park some CTAs while resident CTAs spin-wait for them, hanging forever.
- **Evidence:** Existing barrier implementation at `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:3979-3994 (commit ef53c792a)`:
  ```python
  if tid == Int32(0):
      _atomic_add_u32(grid_barrier_ptr + Int64(seq_idx * Int32(4)), Int32(1))
  arrived = Int32(0)
  while arrived < total_ctas_per_seq_grid:
      arrived = _ld_volatile_u32(grid_barrier_ptr + Int64(seq_idx * Int32(4)))
  ```
  Hard launch hardcoded as `cooperative=True` at `phase_e_kernel.py:3130 (commit ef53c792a)`. The kernel docstring at `phase_e_kernel.py:3071-3076` explicitly states: "Cooperative launch (cooperative=True) is REQUIRED because the kernel uses a grid-wide barrier that spans all CTAs of one seq."

  Resident-cap calculation at `_backend.py:489-494 (commit ef53c792a)` reports `resident_cap = floor(102400/45568) × 48 = 96`, and `64 × num_seqs = 128 > 96` for num_seqs=2. The hermes/interactive `num_seqs=2` case (memory `project_num_seqs_2_target`) is the steady-state target — exactly the case Q3=C claims to handle via non-cooperative.

  The spec's framing in Risk #1 — "if the CuTe DSL grid_sync primitive doesn't compile under non-cooperative" — misidentifies the issue. The kernel does NOT use `cute.arch.grid_sync()`; it uses an atomic-counter spin-wait that is independent of cooperative semantics for compilation but completely dependent on cooperative semantics for **forward progress** at the GPU.
- **Suggested fix:** Either (a) keep β-lite as a separate two-launch kernel for `num_seqs > 1` (no in-kernel barrier needed; matches every surveyed FA3/FlashInfer/TRT-LLM dispatch pattern), or (b) commit to cooperatively launching at num_seqs=2+ by aggressively shrinking SMEM (the same future-work item the spec flags as out-of-scope). The "same body, two launch modes" framing as written is unsound and Risk #1 understates the severity (one-day spike will fail). Update §Architecture to make β-lite NOT a launch mode but a sibling kernel, OR explicitly scope β-coop to num_seqs=1 only and continue dispatching to legacy paths for higher batches.

### Finding 3: C1.5's L4 gate doesn't actually validate C1.5

- **Severity:** MED
- **Where:** Spec §Per-commit gates row C1.5 (line 279); §Test pyramid L4 (line 266).
- **Issue:** L4 is `gsm8k_eval_50.py seed=42 ≥ 90% with CUTE_PHASE_E_FUSION=1`. With β-coop forced ON post-C1, the pre-C1.5 (still bake-baking, but with C1's `residual_in=residual_buf` fix) state would either pass or fail L4 — and post-C1.5 (no bake, per-layer input_LN) is a different math regime. Per Finding 1, post-C1.5 math is **wrong** (double-counts mlp_out), so C1.5 will fail L4 catastrophically. But if Finding 1 is fixed by dropping Phase 4's mlp_out add, then post-C1.5 math becomes the same as pre-C1.5's underlying intent (just without the bake/skip pair). The L4 gate alone can't distinguish "bake removed, layer-LN inserted, math accidentally still wrong" from "all clean." L3 (multi-layer reference diff) is the right test for this — but the gate table has L3 as `—` (not required) for C1.5.
- **Evidence:** Spec table line 279 `C1.5: ... | ✓ | ✓ | ✓ | — | ✓ | (deferred)` — L3 dash. L3 description (line 261-263) explicitly verifies "every layer (full or linear) runs `input_layernorm` at entry. β-coop's Phase 4 output is `residual_final`, not LN-baked. Compare end-to-end output against unfused reference." That's exactly the test that catches Finding 1.
- **Suggested fix:** Move L3 to ✓ for C1.5. C1.5 is the commit that introduces the per-layer-LN regime; it must be gated by the test that verifies the regime. Lifting L3 from `—` to ✓ is also required for catching residual-stream math regressions (Finding 1 class). Without L3 at C1.5, the bug ships.

### Finding 4: L2 test description references buffers C1.5 deletes

- **Severity:** MED
- **Where:** Spec §Test pyramid L2 (line 255).
- **Issue:** L2 asserts "β-coop's outputs landed in `self.residual_output` and `self.next_hidden_scratch`, F.1 dispatch op consumed them." But C1.5 deletes `next_hidden_scratch` (line 114, 290). Post-C1.5, the L2 assertion is unsatisfiable — the test would have to be updated to assert outputs land in `residual_output` and `mlp_output`, and dispatch consumes those.
- **Evidence:** Spec line 90 deletes `next_hidden_scratch` allocation. Spec line 190: "`next_hidden_scratch` is deleted entirely." Spec line 294: "`hidden_out` reads from `mlp_output` (raw mlp_out), not `next_hidden_scratch`." But L2 description was not updated to match.
- **Suggested fix:** Update L2 description to reflect post-C1.5 buffer set. Specify version of L2 for C1 (asserts old buffers) vs version for C1.5+ (asserts new buffers). Or add a third assertion form for the transition.

### Finding 5: `use_fusion` invariant (`_phase_e_active and use_fusion`) becomes meaningless after C2

- **Severity:** MED
- **Where:** `_backend.py:1107-1119 (commit ef53c792a)`. Spec §Architecture line 63 (`paged_attention_forward` retires for decode); not addressed in spec.
- **Issue:** The current code has a load-bearing comment at `_backend.py:1108-1110`: "β-lite reads `self.residual_output` below, which is only populated by the attention uber-kernel when `use_fusion=True`. Keep `use_fusion` in this AND; removing it would silently feed stale residual data from the previous step into the ε epilogue." Post-C2, `paged_attention_forward` no longer runs for decode. β-coop becomes the SOLE writer of `residual_output`. The `and use_fusion` clause is no longer the right invariant; the right invariant is "β-coop will populate residual_output itself this step" — which is implied by `_phase_e_active` already. The spec's deletion list (line 110-115) doesn't mention this AND clause, but `_phase_e_active`'s correctness depends on whether `use_fusion` still implies "attn fusion ran first."
- **Evidence:** `_backend.py:980 (commit ef53c792a)`: `self._fusion_active = self._fusion_bound and is_decode_only and fits_buffer`. After C2 deletes `paged_attention_forward` from the decode path, `_fusion_active=True` no longer means "the attn uber-kernel populated buffers." It just means "fusion is bound." The semantics drift.
- **Suggested fix:** Spec C2 should explicitly redefine `_fusion_active` (or rename it) post-migration, since "use_fusion" used to gate "did the legacy uber-kernel run." Add a deletion-list entry for the `and use_fusion` clause OR document that β-coop now becomes the populating writer and the AND becomes harmless.

### Finding 6: β-lite has the same buffer-aliasing bug, not flagged

- **Severity:** MED
- **Where:** `_backend.py:1268 (commit ef53c792a)`. Spec §Background lines 28-29.
- **Issue:** β-lite reads `residual_post_ln=self.residual_output[:nat]` at `_backend.py:1268` — same pattern as β-coop's pre-fix `residual_in=self.residual_output[:nat]`. β-lite ALSO depends on `paged_attention_forward` having written `(h+r) + attn_out` to `residual_output` first. So β-lite has identical double-counting if you trace through it. Spec frames this as "β-coop bug" — but β-lite was on the same buggy path the entire time. This isn't a finding that breaks the migration, but it suggests pre-migration β-lite was never math-correct either, which means any "β-lite ON" historical evidence is suspect.
- **Evidence:**
  - `_backend.py:1268`: `residual_post_ln=self.residual_output[:nat]` — β-lite reads.
  - `kernel.py:1929-1931 (commit ef53c792a)`: `paged_attention_forward` writes `new_res = res_f32 + wo_f32` to `resout_base`. Same writer, same buffer, same alias.
- **Suggested fix:** Either acknowledge in §Background that β-lite was also broken (no behavioral change for the migration since β-lite collapses anyway), or reframe the bug as "both β kernels alias the legacy uber-kernel's residual_output." Pre-migration GSM8K evidence under β-lite ON should not be trusted as ground truth for math correctness.

### Finding 7: Spec line numbers drift slightly from live code

- **Severity:** LOW (nitpick)
- **Where:** Spec §Code anchors (lines 334-342); §C1.5 detail (lines 287-294).
- **Issue:** Several cited line ranges are off by 1-3 lines vs live code at HEAD `ef53c792a`:
  - Spec says `attach_input_layernorm` at `_backend.py:433-528` → actual is `attach_next_input_layernorm` at L433-505 plus `attach_input_layernorm` at L507-519. Spec lumped both into one range and rounded up.
  - Spec says skip-op at `_mlp_op.py:240-303` → actual is L239-301.
  - Spec says `attach_input_layernorm` loop at `qwen3_5.py:633-647` → actual is L636-647 (lines 633-635 are import + setup, not the loop itself).
  - Spec says skip-op call site at `qwen3_5.py:380-450` → actual call site at L431-435 (the surrounding 380-450 range is the layer forward).
  None of these are bugs in design, just citation drift.
- **Evidence:** `_backend.py:433`, `_mlp_op.py:239`, `qwen3_5.py:431` (commit ef53c792a) all spot-checked.
- **Suggested fix:** Tighten line citations during plan-writing. Per `feedback_pinned_code_refs`, all anchors should be `file:Lstart-Lend (commit hash)` form when they ship. Not a blocker.

### Finding 8: Phase 4 "memcpy" wording is contradictory

- **Severity:** LOW
- **Where:** Spec §Components line 82 ("memcpy of `residual_final → caller buffer`"); §Data flow lines 174-178; §Buffer contracts line 190.
- **Issue:** Components says Phase 4 "collapses to a memcpy of residual_final → caller buffer." Data flow shows Phase 4 doing `residual_output (in-place mutated to + mlp_out)`. Buffer contracts says "`next_hidden_scratch` is deleted entirely — Phase 4 no longer bakes; dispatch op reads `mlp_output` directly as the next-layer hidden." All three are subtly different stories. After Finding 1 is resolved, Phase 4 should NOT add mlp_out at all, AND there's no separate memcpy needed (residual_post_attn already lives in residual_output from Phase 1C). Phase 4 should likely be **deleted entirely**.
- **Evidence:** N/A (internal spec consistency).
- **Suggested fix:** After fixing Finding 1, rewrite §Components / §Data flow / §Buffer contracts to be consistent: "Phase 4 deletes; β-coop output is (mlp_output, residual_output=residual_post_attn); layer N+1's input_layernorm does the residual+mlp accumulation."

## Load-bearing-claim check

| Claim | Pass/fail | Citation |
|---|---|---|
| `paged_attention_forward` call site at `_backend.py:1034` | PASS | Verified at `_backend.py:1034 (commit ef53c792a)` |
| β-coop `residual_in` source at `_backend.py:1175` (alias bug) | PASS | Verified at `_backend.py:1175 (commit ef53c792a)`: `residual_in=self.residual_output[:nat]` |
| `paged_attention_forward` Phase C writes `new_res = residual + wo_out` to residual_output | PASS | Verified at `kernel.py:1929-1931 (commit ef53c792a)` |
| `run_beta_coop_full` at `phase_e_kernel.py:2685` | PASS | Verified at `phase_e_kernel.py:2685 (commit ef53c792a)` |
| Phase 4 ε epilogue bake branch at `phase_e_kernel.py:4631-4652` | PASS | Verified at L4631-4652 (commit ef53c792a); `if emit_next_ln == Int32(1):` block |
| `cute_phase_e_dispatch_impl` at `_mlp_op.py:192-218` | PASS | Verified exact match (commit ef53c792a) |
| `cute_phase_e_skip_input_layernorm_impl` at `_mlp_op.py:248-283` | PASS | Verified exact match (commit ef53c792a) |
| residual mirror at `qwen3_5.py:460` | PASS | Verified: `impl.residual_buf[:nat].copy_(residual[:nat])` (commit ef53c792a) |
| `attach_input_layernorm` + `attach_next_input_layernorm` loops at `qwen3_5.py:633-676` | PASS (line drift) | Actual L636-676 (commit ef53c792a); spec rounded |
| Linear-attn layer N+1 doesn't honor F.1 skip-op | PASS | `_cute_layer_name` set only on full-attn `mlp_module` at `_backend.py:832 (commit ef53c792a)`; linear-attn falls through `attach_mlp_fusion` at `qwen3_5.py:383 (commit ef53c792a)` (only `Qwen3_5MLP` instance triggers it) |
| Qwen3.5-27B layer pattern is stride-4 (3 lin, 1 full) | PASS | `config.json` text_config.layer_types: `['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', ...]` repeats; 64 layers total |
| Cooperative launch hardcoded `cooperative=True` | PASS | `phase_e_kernel.py:3130 (commit ef53c792a)` |
| **β-coop output `(mlp_out, residual_final)` → layer N+1 `input_layernorm` → no double-count** | **FAIL** | See Finding 1 |
| **Non-cooperative launch with existing barrier won't deadlock** | **FAIL** | See Finding 2 |

## Coverage gaps in test pyramid

1. **L3 should be required for C1.5, not deferred.** L3 is the only test that compares end-to-end multi-layer output against a Python reference chain. Finding 1's bug (residual-stream double-add) is exactly what L3 catches and L4 (gsm8k 50-Q) catches only weakly (a 2·mlp_out residual can still produce SOME tokens correctly on a strong base model). Move L3 from `—` to ✓ for C1.5.

2. **L2 needs a `cooperative=False` path test.** L0 lists "cooperative=True vs cooperative=False parity at every phase test" but L2 doesn't. After C3 there's a real risk that the non-cooperative launch hangs (Finding 2) — L2 should fire β-coop with num_seqs=2 and a wall-clock-budget timeout to catch deadlock as a hard failure. Without that, L4/L5 will hang the entire eval suite, masking the bug as "test infra problem."

3. **No test verifies the `use_fusion` AND clause invariant after C2** (Finding 5). After C2 deletes the legacy launch, a regression test should assert that `_phase_e_active` correctly gates β-coop and that no stale data path remains.

4. **L4 alone is insufficient for residual-stream math regressions.** A 2·mlp_out residual leaks slowly: early layers degrade gradually, and on stride-4 only every 4th layer compounds the error. GSM8K is a coarse signal — at 50 questions, threshold 90% can mask a 5-10% silent regression. Recommend adding a per-layer L1-norm comparison test (β-coop output vs reference) that fails on >1% drift.

## Recommendation

**Revise spec.** Findings 1 and 2 are HIGH severity and structural — they require non-trivial design changes (drop Phase 4's mlp_out add; reconcile β-coop's grid barrier with non-cooperative launch). Findings 3 and 4 are MED and require gate-table updates. Finding 5 needs a one-line acknowledgment in C2's deletion list.

Specifically before writing plans:
- Resolve Finding 1: confirm the residual-stream protocol (β-coop output should be `residual_post_attn`, not `residual_final`; Phase 4 deletes or no-ops).
- Resolve Finding 2: either keep β-lite as a sibling kernel (not a launch mode of β-coop) for num_seqs ≥ 2, OR explicitly scope β-coop to num_seqs=1 with legacy fallback for higher batches. Don't claim Q3=C is achievable until the grid-barrier-vs-non-cooperative-launch question has a concrete answer (the existing atomic spin-wait will not work).
- Update gate table: L3 ✓ for C1.5; L2 description for C1.5+ buffer set.

This is the same shape of audit catch as the Q4=A self-review found: the spec's narrative is internally consistent but doesn't survive contact with `_forward_static_with_residual` semantics or the kernel's barrier requirements. One more iteration of fresh eyes + brainstorm is warranted before plan-writing.
