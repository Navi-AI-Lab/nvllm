# Upstream Cherry-Pick: NVFP4 + Gemma 4 31B + Tool-Use

**Date:** 2026-04-09
**Goal:** Cherry-pick 11 upstream vLLM PRs to get Gemma 4 31B NVFP4 serving with tool-use on DGX Spark (GB10/SM121), while aligning our NVFP4 stack with upstream's new `NvFp4LinearKernel` structure.

## Context

Our fork is based on upstream/main from ~April 5 (commit `9a528260ef`), 119 commits behind. We have 46 local commits, with only 1 modified kernel file (`nvfp4_scaled_mm_sm120_kernels.cu` — stream-K decode GEMM). The upstream NVFP4 codebase is being restructured and Gemma 4 support landed in v0.19.0 with follow-up fixes still flowing in.

**Target model:**
- `RedHatAI/gemma-4-31B-it-NVFP4` — RedHat's NVFP4 quantization

## Approach: Surgical Cherry-Pick

Cherry-pick individual PRs in dependency order. Chosen over rebase or merge to minimize blast radius, allow per-PR verification, and keep bisectability.

## Cherry-Pick Inventory

### Already in fork (no action)
- #38826 — Core Gemma 4 architecture
- #38872 — Gemma 4 cleanup
- #38847 — Gemma4ToolParser fix
- #38832 — NVFP4+MTP crash fix for Qwen3.5

### Phase 1: NVFP4 Bugfixes (independent, clean cherry-picks)

| Order | SHA | PR | Title | Conflict Risk |
|-------|-----|-----|-------|---------------|
| 1 | `731055548` | #37502 | Fix Marlin NVFP4 rescaling | NONE |
| 2 | `9c81f35b1` | #38819 | Re-enable FA4 as default MLA prefill backend | LOW |

### Phase 2: NVFP4 Refactor (sequential dependency)

| Order | SHA | PR | Title | Conflict Risk |
|-------|-----|-----|-------|---------------|
| 3 | `2800706f0` | #39129 | Move NVFP4 GEMM into NvFp4LinearKernel | **MEDIUM** |
| 4 | `201813724` | #39322 | Batch-invariant NVFP4 linear | LOW |

**#39129 is the only PR with real conflict risk.** It touches `modelopt.py` which has a 223-line divergence between our fork and upstream. The 7 new files in `vllm/model_executor/kernels/linear/nvfp4/` will apply cleanly. Our stream-K kernel patch (`csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`) is completely untouched.

#39322 depends on #39129. Must land after.

### Phase 3: NVFP4 MoE Backend

| Order | SHA | PR | Title | Conflict Risk |
|-------|-----|-----|-------|---------------|
| 5 | `e8ebbdde8` | #38251 | FlashInfer CuteDSL batched experts for NVFP4 MoE | LOW |

### Verification Gate 1
- Docker build (`docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 .`)
- Serve Qwen3.5-27B NVFP4 — confirm no regression
- GSM8K sanity check on Qwen3.5

### Phase 4: Gemma 4 Follow-ups

| Order | SHA | PR | Title | Conflict Risk |
|-------|-----|-----|-------|---------------|
| 6 | `47e605092` | #38879 | Fast prefill / YOCO KV-sharing for Gemma 4 | LOW |
| 7 | `3aecdf08b` | #39045 | Quantized MoE weight loading | LOW |

### Verification Gate 2
- Serve `RedHatAI/gemma-4-31B-it-NVFP4` with `--enforce-eager`
- If OK, remove `--enforce-eager` and test with CUDA graphs
- GSM8K sanity check
- Quick chat test for coherent output

### Phase 5: Tool-Use / Reasoning Fixes

| Order | SHA | PR | Title | Conflict Risk |
|-------|-----|-----|-------|---------------|
| 8 | `d734445fc` | #38909 | Fix streaming HTML duplication after tool calls | LOW |
| 9 | `f53fa26e0` | #38992 | Fix invalid JSON in streaming tool calls | LOW |
| 10 | `8477fe427` | #39027 | adjust_request for reasoning parser + Gemma 4 | LOW |
| 11 | `13151a4df` | #39114 | Fix streaming tool call corruption (bools/numbers) | LOW |

### Verification Gate 3
- Send tool-use request to Gemma 4 and verify valid streaming JSON
- Test reasoning parser output

## Serving Configuration

### Memory Budget (Gemma 4 31B NVFP4 on GB10)

| Component | Estimate |
|-----------|----------|
| NVFP4 weights | ~15.5 GB |
| FP8 KV cache (8K ctx, 4 seqs) | ~3 GB |
| Activations + overhead | ~15-18 GB |
| **Total** | **~35-37 GB** |
| **Headroom** | **~83 GB** |

### Launch Command

```bash
vllm serve RedHatAI/gemma-4-31B-it-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager  # first run only
```

### Debugging Output

Capture logs for debugging if output quality is wrong or model fails to load:
```bash
# Run with verbose logging + output to file
VLLM_LOGGING_LEVEL=DEBUG docker logs -f nvllm 2>&1 | tee /tmp/gemma4-debug.log

# Quick curl test with full response for sanity check
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"RedHatAI/gemma-4-31B-it-NVFP4","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":16}' | python3 -m json.tool
```

If output is gibberish or degraded, follow the debugging protocol in CLAUDE.md (check upstream issues first, verify BF16 base, isolate quant vs CUDA graph).

### Known Limitation

Gemma 4 has mixed head_dim (256 for sliding-window layers, 512 for full-attention layers). Without PR #38891 (per-layer attention backend selection, still open upstream), all layers fall back to TRITON_ATTN. This causes a significant decode penalty. Once #38891 merges, cherry-pick it for ~5x decode improvement on 83% of layers.

## Watch List (Open PRs)

| PR | Title | Why We Want It |
|----|-------|----------------|
| #38891 | Per-layer attention backend selection | Big decode speedup for Gemma 4's mixed head_dim |
| #39067 | Fix MoE top_k lookup | Needed if we try Gemma 4 26B-A4B (MoE) later |
| #39256 | NVFP4 per-expert weight loading for Gemma 4 MoE | Needed for NVFP4 quant of 26B-A4B |

## Conflict Resolution Guide (PR #39129)

The dry-run cherry-pick of `2800706f0` produces 4 conflicting files. All 7 new files under `kernels/linear/nvfp4/` apply cleanly. Because our fork has **zero local modifications** to the NVFP4 dispatch paths (all divergence is "fork behind upstream"), every conflict can be resolved by taking the upstream side.

**Resolution command:**
```bash
git cherry-pick --no-commit 2800706f0
git checkout --theirs -- \
  vllm/model_executor/layers/quantization/modelopt.py \
  vllm/model_executor/kernels/linear/__init__.py \
  vllm/model_executor/layers/quantization/utils/nvfp4_utils.py \
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py
git add -u
git commit
```

**Post-resolution verification:**
- `git diff HEAD upstream/main -- <file>` should show zero diff for all 4 files
- Confirm `kernels/linear/__init__.py` imports `MMLinearKernel` and `MMLinearLayerConfig` from `.base`
- Confirm `compressed_tensors_w4a4_nvfp4.py` has `logger = init_logger(__name__)` after imports (may need manual add)

### Conflict Details

| File | Regions | Root Cause | Resolution |
|------|---------|------------|------------|
| `modelopt.py` | 3 | Old nvfp4_utils imports, `self.backend` → `self.kernel`, `apply_nvfp4_linear` → `self.kernel.apply_weights` | Take theirs |
| `kernels/linear/__init__.py` | 2 | Missing `_POSSIBLE_NVFP4_KERNELS` registry + broadened TypeVar bounds + `__all__` entries | Take theirs |
| `nvfp4_utils.py` | 1 (whole file) | PR deletes old backend enum/dispatch; our fork has older version of same code | Take theirs |
| `compressed_tensors_w4a4_nvfp4.py` | 2 | Old nvfp4_utils imports + `apply_nvfp4_linear` call | Take theirs + add logger |

## Risk Summary

- **1 conflict point** — PR #39129 has 4 conflicting files, all resolvable with `checkout --theirs`
- **10 other PRs** — LOW/NONE conflict risk, new files or non-overlapping regions
- **Stream-K kernel patch** — completely isolated, untouched by all 11 PRs
- **Rollback** — each phase can be reverted independently via `git revert`
