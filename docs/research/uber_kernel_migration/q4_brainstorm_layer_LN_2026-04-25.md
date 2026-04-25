# Q4 brainstorm — layer-boundary input_LN handling

**Date:** 2026-04-25
**Repo state:** branch `fix/phase-d-mlp-decode-tile-preset` @ `ef53c792aa328a6a19966fb23897f3929d7eb45c`
**Spec under review:** `docs/superpowers/specs/2026-04-25-uber-kernel-migration-design.md`

## TL;DR

The spec self-review is **correct**: Q4=A's Phase 4 ε bake corrupts the residual stream for every fused full-attn → linear-attn boundary in Qwen3.5-27B's stride-4 layout. The bake fires for layer N+1's input_LN where N+1 is **always** linear-attn, but only full-attn layers' decoder forward honors the F.1 skip-op — so layer N+1 (linear-attn) double-applies input_LN to a pre-baked tensor and folds the LN'd payload into its own residual. Since this affects 16/16 fused full-attn boundaries, the "300-500 µs/token saving" is illusory: β-coop has never produced correct math for this model. **Recommendation: Option B — drop the Phase 4 bake, run input_LN at the start of each layer (Qwen3Next upstream pattern).** Option A-revised collapses to a no-op for stride-4 and only earns its complexity if a future model has consecutive full-attn layers. Option D is a major restructure with no upside over B.

## Verification of the claimed bake-corruption issue

All four claims hold. Citations against `ef53c792a`:

**Claim 1 — Qwen3.5-27B stride-4 pattern.**
Confirmed via `~/.cache/huggingface/hub/models--natfii--Qwen3.5-27B-NVFP4-Opus-GB10/snapshots/.../config.json`:

```
num_layers: 64
layer_types[0:8]: ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
                   'linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
layer_types[60:64]: ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
full_attn_idxs:    [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, ...]   total full: 16
```

For every full-attn layer N at index `4k+3`, layer N+1 at `4k+4` is linear-attn (or, for k=15, no next layer at all).

**Claim 2 — `attach_mlp_fusion` runs only on full-attn layers.**
`vllm/nvllm/models/qwen3_5.py:370-388` (commit `ef53c792a`):

```python
if self.layer_type == "full_attention":
    try:
        ...
        impl = self.self_attn.attn.impl
        if isinstance(impl, CutePagedAttentionImpl):
            impl.attach_fusion(self)
            if isinstance(self.mlp, Qwen3_5MLP):
                impl.attach_mlp_fusion(self.mlp, layer_name=f"{self.prefix}.mlp")
```

`attach_mlp_fusion` at `_backend.py:832` (`mlp_module._cute_layer_name = layer_name`) is the only writer of `_cute_layer_name`. Linear-attn layers' MLPs never get the attribute.

**Claim 3 — F.1 skip-op gate fires only on full-attn layers.**
`vllm/nvllm/models/qwen3_5.py:421-441` (commit `ef53c792a`):

```python
else:
    _mlp_layer_name = getattr(self.mlp, "_cute_layer_name", None)
    if _mlp_layer_name is not None:
        ...
        torch.ops.vllm.cute_phase_e_skip_input_layernorm(
            hidden_states, residual, out_x, out_residual, _mlp_layer_name,
        )
        ...
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

Linear-attn layers fall through to the unconditional `self.input_layernorm(hidden_states, residual)` branch. There is **no** mechanism for a linear-attn layer to honor a pre-baked input_LN — it always runs RMSNorm with-residual.

**Claim 4 — `attach_next_input_layernorm` blindly binds layer N+1.**
`vllm/nvllm/models/qwen3_5.py:670-676` (commit `ef53c792a`):

```python
next_norm = (
    getattr(self.layers[idx + 1], 'input_layernorm', None)
    if idx + 1 < num_layers
    else None
)
impl.attach_next_input_layernorm(next_norm)
```

No layer-type check. For full-attn at idx 3, `next_norm = self.layers[4].input_layernorm` — a linear-attn module. `_emit_next_layernorm = True` (`_backend.py:452`).

**Phase 4 ε epilogue math.**
`phase_e_kernel.py:2611-2631` (commit `ef53c792a`) writes:

```python
out_f32 = normed_round * (Float32(1.0) + gamma_f32)   # qwen RMSNorm (1+γ)
_st_global_bf16_from_f32(next_hidden_base + ..., out_f32)
```

So `next_hidden_scratch[i] = RMSNorm(residual_final) · (1 + γ_{linear_attn_N+1})`.

**Math of the corruption.**
At layer N+1 entry (linear-attn), `hidden_states = next_hidden_scratch` (LN-baked) and `residual = residual_final_N`. Linear-attn falls through to `self.input_layernorm(x, r)`, which `vllm/nvllm/layers/layernorm.py:60-80` implements as:

```
x_new = x + r;  residual_new = x_new;  x_new = norm(x_new) * (1+γ)
```

Result: `residual_new = LN(rf)·(1+γ_{N+1}) + rf`  (corrupted residual stream — LN'd payload folded back in)
        `x_new = norm(LN(rf)·(1+γ_{N+1}) + rf) · (1+γ_{N+1})`  (double-LN applied to the wrong sum)

This is broken for every k in 0..15 (all 16 full-attn layers). Bug is structural, not transient.

**Linear-attn pre-bake-aware path: none exists.** No skip-op call, no flag check, no conditional input_LN branch in `Qwen3_5DecoderLayer.forward` for `layer_type == "linear_attention"`. Confirmed by reading `qwen3_5.py:415-494`.

## Option B — per-layer Phase 0 input_LN (run input_LN at start of each layer)

**Scope.** Phase 0 is **already fully implemented** in the kernel — not a stub. β-coop's `phase_0_to_4` unified kernel includes Phase 0 (`phase_e_kernel.py:2648-2660`, gated to `bx==0 && by==0`, single CTA per seq, writes `attn_input_bf16` as side-channel). `run_phase_0_only` (`phase_e_kernel.py:287`) and `run_phase_01_only` (`:353`) are tested entry points. Phase 0 currently does NOT produce the canonical input_LN consumed by Phase 1 in the live β-coop launch — Phase 1's query is pre-projected upstream of β-coop, so Phase 0's output is currently just a side-channel for "future QKV-fusion" (per kernel comment at `:2649-2651` and the `_backend.py:1172-1174` "dummy — output side-channel" note).

**What changes for Option B:**
1. Delete `attach_next_input_layernorm` and the `next_hidden_scratch` allocation/wiring (`_backend.py:433-505`, `:481-484`, `:1197-1198`, `:1204`).
2. Delete the F.1 skip-op call site at `qwen3_5.py:431-436` and `attach_input_layernorm` (`_backend.py:507-528`); leave the `else` branch at `:438-441` as the unconditional input_LN call for ALL layer types.
3. Delete `_cute_phase_e_skip_input_layernorm` op (`_mlp_op.py:240-303`).
4. Phase 4 ε epilogue at `phase_e_kernel.py:2611-2639` collapses to "memcpy residual_final → caller buffer". `emit_next_layernorm` flag and `_phase_e_skip_next_ln` flag delete entirely.
5. β-coop launch in `_backend.py:1146-1208` drops `next_input_layernorm_gamma`, `next_hidden_output`, `emit_next_layernorm` args; `_phase_e_consumed` consume path in `cute_phase_e_dispatch` (`_mlp_op.py:207-213`) reads from a residual_final buffer (β-coop output) directly, no LN-bake involved.
6. The dispatch op's `hidden_out` becomes `residual_final` and the next layer's `input_layernorm(hidden, residual)` runs unconditionally on entry. Matches Qwen3Next upstream pattern.

**Pros:**
- Single source of truth for input_LN — runs in one place per layer, like every other transformer.
- Matches upstream Qwen3Next pattern (`vllm/model_executor/models/qwen3_next.py` HEAD: input_LN unconditional at layer entry, before the linear/full branch).
- Matches Jamba (`huggingface/transformers@2dba8e0495 src/.../jamba/modeling_jamba.py:803-860`) and Zamba2 (same commit, `zamba2/modeling_zamba2.py`): both apply input_LN at the start of each layer regardless of attn vs SSM.
- Eliminates 4 cross-impl flags, 1 module ref, 1 pre-allocated scratch buffer (`next_hidden_scratch`).
- Removes a class of bugs the team has already hit twice (`feedback_opaque_op_not_enough`, `project_phase_e_phantom_speedup`).
- Roughly halves the spec's "Open risks" section: risk 2 (cross-impl flag propagation) and the "I3/I5 invariants" disappear.

**Cons:**
- Loses the *advertised* "300-500 µs/token saving" — but the saving was never real for this model. RMSNorm is a single-CTA fused kernel; modern vLLM ships it as one launch per layer at ~5-10 µs each. 64 layers × ~7 µs ≈ 450 µs/token of input_LN cost on the critical path, and it's already there in the β-OFF baseline.
- Slight regression vs a hypothetical *correct* bake on a future model with consecutive full-attn layers. None of the team's current targets (Qwen3.5, Qwen3-Next, Gemma 4) have that pattern.

**Phase 0 disposition:** Two reasonable framings.
- (a) Treat Phase 0 as the kernel's input_LN and call β-coop with `hidden_in = pre-LN` so Phase 0 produces the canonical post-LN tensor for downstream Phase 1 — but Phase 1 currently consumes a pre-projected `query`, so this requires also folding the query projection into the kernel. **Out of scope** for this migration.
- (b) Keep input_LN as a separate `Qwen3_5RMSNorm.forward` call at the layer's Python entry (same as today's linear-attn fall-through path). Phase 0 stays as side-channel/future-fusion. **Recommended** — minimal change, ships on the migration's existing C-series commits.

## Option A-revised — smart bake (only when next is also full-attn)

**For Qwen3.5-27B specifically:** stride-4 means there's never a full→full boundary except possibly the last layer pair. Index 63 is full-attn (verified above), `idx+1 == num_layers` returns `None` in `attach_next_input_layernorm`, so `_emit_next_layernorm = False` already. **Net result for Qwen3.5: smart bake never bakes. Identical observable behavior to Option B.**

**For other models:**
- Qwen3-Next (parent): same stride pattern (3:1 linear:full), same outcome.
- Gemma 4 31B: blocked on PR #38891; no fused decode kernel yet, so the question doesn't bite.
- Hypothetical models with consecutive full-attn layers (e.g., a "full_full_full_full" stride or end-of-network full-attn cluster): smart bake retains a real saving.

**Pros:**
- Future-proofs for non-stride-4 layouts at modest scope cost (one `if config.layer_types[idx+1] == "full_attention":` check at `qwen3_5.py:670` and one in the kernel's `emit_next_layernorm` gate).

**Cons:**
- Keeps every flag/buffer/cross-impl-state mechanism Option B deletes. Earns its complexity zero times on the team's current model lineup.
- The "single source of truth" framing the migration champions explicitly fights against retaining conditional baking.
- Bait for the next bug: every future kernel modifier has to remember the bake-only-when-next-full-attn invariant.

**Verdict:** Reject. If a future model needs the bake, reintroduce it then with that model's evidence in hand. Premature optimization.

## Option D — extend skip-op to linear-attn

**What linear-attn would need to do.** The skip-op currently lives on the MLP module via `_cute_layer_name`. To extend: linear-attn's MLP (also `Qwen3_5MLP` in dense, `Qwen3NextSparseMoeBlock` in MoE) would need an attached `_cute_layer_name` and an associated impl on which `_phase_e_skip_next_ln`, `_input_layernorm_module`, and the registry entry live. Linear-attn layers don't have a `CutePagedAttentionImpl` (there's no full-attn impl), so a parallel "linear-attn fusion holder" object would have to be invented just to hang flags off.

**Scope.**
- New attach pathway on linear-attn layers — meaningful refactor, since attach is currently gated by `isinstance(impl, CutePagedAttentionImpl)` (`qwen3_5.py:377`).
- Skip-op call site at `qwen3_5.py:421-441` already works for any layer type that exposes `_mlp_layer_name`; that part is small.
- The dispatch-op equivalent on linear-attn (`qwen3_5.py:541-554`) gets messier: linear-attn doesn't run β at all, so its "dispatch" is a pass-through that just resets `_phase_e_skip_next_ln`. Doable but adds another opaque op.
- Cross-layer flag still flows full-attn-N → linear-attn-N+1 → linear-attn-N+2 → linear-attn-N+3 → full-attn-N+4. Each linear-attn layer must clear-and-pass the flag correctly. New fragile invariant.

**Pros:**
- Preserves the "save next-layer input_LN" optimization end-to-end.

**Cons:**
- Major surface-area increase for a saving that, even charitable accounting, is ~7 µs/layer × 16 layers ≈ 110 µs/token. Linear-attn layers don't have β; they pay the full input_LN cost regardless of bake — so the bake only saves on full-attn entries, not all 64.
- Three new attach calls, one new opaque op, one new fake registry, three new flag-flow invariants. Failure modes multiply.
- Distracting from the migration's actual goal (retiring `paged_attention_forward` from decode).

**Verdict:** Reject. Investment-to-return ratio is wrong, and it inflates the very state-machine complexity the migration is trying to compress.

## Prior art — hybrid SSM/attention models

Pinned commits below. None of the surveyed implementations bake next-layer input_LN into the previous layer's epilogue.

| Model | Repo @ commit | Pattern |
|---|---|---|
| **Jamba** | `huggingface/transformers@2dba8e0495974930af02274d75bd182d22cc1686` `src/transformers/models/jamba/modeling_jamba.py:803-860` | `JambaAttentionDecoderLayer` and `JambaMambaDecoderLayer` each open with `residual = hidden_states; hidden_states = self.input_layernorm(hidden_states)` then run their respective branch. Pre-norm, per-layer, uniform across attn/SSM. No fusion. |
| **Zamba2** | same commit, `zamba2/modeling_zamba2.py` | Both `Zamba2MambaDecoderLayer` and `Zamba2AttentionDecoderLayer` apply `self.input_layernorm` at entry. No kernel-level cross-layer LN fusion (mamba-ssm/causal-conv1d kernels handle SSM internals only). |
| **Qwen3-Next** (vLLM upstream) | `vllm-project/vllm@main` `vllm/model_executor/models/qwen3_next.py` | Layer forward applies `self.input_layernorm(hidden_states, residual)` BEFORE the `linear_attention`/`full_attention` branch — uniform, unconditional. This is the upstream parent of Qwen3.5 and the reference for "what stock vLLM expects". |
| **Mamba-2** | `state-spaces/mamba@7438488222dc44eb9146e4f39c0764a8f651ede6` `mamba_ssm/modules/...` | Pure-SSM (no attn). RMSNorm before SSM block, residual after. No relevant prior art for boundary handling. |
| **Hymba** (NVIDIA) | publicly released as `nvidia/Hymba-1.5B-Base` on HF; reference impl ships per-layer pre-norm in the published modeling code, no cross-layer fusion. | (Not directly load-bearing — same pattern as Jamba.) |
| **Megatron-LM hybrid** | `NVIDIA/Megatron-LM@15e07a2ddf6f7398022de29e84742b6e94b4d4c0` `megatron/core/models/mamba` | NVIDIA's production Mamba/hybrid stack. Per-layer norm at entry; TE-fused attention does NOT bake next-layer norm. |

**Common shape:** every hybrid model surveyed runs input_LN at the start of each decoder layer regardless of layer type. None of them attempt cross-layer LN baking, even with custom CUDA kernels available. The bake idea — "save 300-500 µs by skipping next-layer RMSNorm" — appears to be unique to this fork's β-coop kernel and has no production precedent.

## Recommendation

**Adopt Option B.** Delete the Phase 4 ε bake, the F.1 skip-op, and the `next_hidden_scratch` plumbing. Run input_LN at the start of each layer for both branches (the linear-attn fall-through path becomes the unified path). This:

1. Fixes a confirmed correctness bug that has been silently corrupting the residual stream for every full-attn → linear-attn boundary in Qwen3.5-27B.
2. Aligns with every surveyed hybrid model's pattern (Jamba, Zamba2, Qwen3-Next, Megatron hybrid).
3. Honors the migration spec's stated goal of single-source-of-truth and self-contained per-layer semantics.
4. Removes risks 2 ("cross-impl flag propagation") and the I3/I5 invariants from the spec's "Open risks" section.
5. Simplifies β-coop's kernel signature: drop `next_input_layernorm_gamma`, `next_hidden_output`, `emit_next_layernorm`. Phase 4 collapses to `mlp_out + residual_post_attn → residual_output`, a memcpy of `residual_final` to the next-layer hidden buffer.

**Suggested commit ordering** (extends the spec's C1-C4):
- **C1.5** (between C1 and C2): drop the bake. Same commit deletes `attach_next_input_layernorm`, `_emit_next_layernorm`, `next_hidden_scratch`, the F.1 skip-op, `_phase_e_skip_next_ln`, and Phase 4's emit branch. Adds an unconditional `input_layernorm` call at the start of every layer (already present in the linear-attn fall-through path; simply collapse the if/else at `qwen3_5.py:421-441`).
- L4 gate becomes the canary: gsm8k_eval_50 ≥ 90% under PIECEWISE proves the bake-removal didn't regress beyond the baseline (and likely *recovers* quality because the bake was active and broken).
- L5 perf gate: re-measure tok/s. Expectation — neutral to slightly slower than the (broken) bake-on configuration, but neutral vs the β-OFF baseline that's currently in production. The migration's primary perf win comes from killing the duplicate-firing of `paged_attention_forward`, not from the bake.

**Acknowledged uncertainty:**
- The "300-500 µs/token saving" was estimated, never measured against a *correct* bake. If a future correctness fix recovers it, deleting the machinery now means rebuilding it later. I judge this acceptable given (a) zero current models have full→full boundaries, (b) the prior-art survey shows no production system attempts the bake, (c) the spec's reliability axiom says "reliability lives in CI/tests, not silent runtime safety nets" — the same axiom argues against retaining state-machine machinery that's never exercised.
- Phase 0's eventual role (canonical input_LN inside the kernel, eliminating the Python-side RMSNorm call) remains future work tied to query-projection fusion. Out of scope for this migration. Option B leaves Phase 0 as a tested side-channel ready for that future fold-in.
