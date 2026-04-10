# Upstream Cherry-Pick: NVFP4 + Gemma 4 31B + Tool-Use — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cherry-pick 11 upstream vLLM PRs to enable Gemma 4 31B NVFP4 serving with tool-use on DGX Spark.

**Architecture:** Surgical cherry-pick in 5 phases with verification gates between each. All conflicts in PR #39129 are resolved by taking upstream's side (`checkout --theirs`). Existing run/serve scripts are updated to support the RedHat HF checkpoint.

**Tech Stack:** git cherry-pick, Docker (Dockerfile.gb10), vLLM, curl for smoke tests

**Spec:** `docs/superpowers/specs/2026-04-09-upstream-cherry-pick-nvfp4-gemma4-design.md`

---

### Task 1: Phase 1 — Cherry-pick NVFP4 bugfixes

**Files:** Git-managed (upstream commits applied to working tree)

- [ ] **Step 1: Cherry-pick #37502 — Fix Marlin NVFP4 rescaling**

```bash
git cherry-pick 731055548
```

Expected: Clean apply, no conflicts.

- [ ] **Step 2: Cherry-pick #38819 — Re-enable FA4 as default MLA prefill**

```bash
git cherry-pick 9c81f35b1
```

Expected: Clean apply, no conflicts. Touches `vllm/config/attention.py`.

- [ ] **Step 3: Verify both commits landed**

```bash
git log --oneline -3
```

Expected: Two new commits on top of HEAD, both with upstream PR titles.

---

### Task 2: Phase 2 — Cherry-pick NVFP4 refactor (#39129 with conflict resolution)

**Files:**
- Modify: `vllm/model_executor/layers/quantization/modelopt.py`
- Modify: `vllm/model_executor/kernels/linear/__init__.py`
- Modify: `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`
- Modify: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`
- Create: `vllm/model_executor/kernels/linear/nvfp4/__init__.py`
- Create: `vllm/model_executor/kernels/linear/nvfp4/base.py`
- Create: `vllm/model_executor/kernels/linear/nvfp4/cutlass.py`
- Create: `vllm/model_executor/kernels/linear/nvfp4/emulation.py`
- Create: `vllm/model_executor/kernels/linear/nvfp4/fbgemm.py`
- Create: `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py`
- Create: `vllm/model_executor/kernels/linear/nvfp4/marlin.py`

- [ ] **Step 1: Cherry-pick #39129 with conflict resolution**

All conflicts are "fork behind upstream" — no local modifications to NVFP4 dispatch. Take theirs for all 4 conflicting files.

```bash
git cherry-pick --no-commit 2800706f0
git checkout --theirs -- \
  vllm/model_executor/layers/quantization/modelopt.py \
  vllm/model_executor/kernels/linear/__init__.py \
  vllm/model_executor/layers/quantization/utils/nvfp4_utils.py \
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py
git add -u
git add vllm/model_executor/kernels/linear/nvfp4/
```

- [ ] **Step 2: Verify conflict resolution**

Check that the 4 resolved files now match upstream exactly:

```bash
for f in \
  vllm/model_executor/layers/quantization/modelopt.py \
  vllm/model_executor/kernels/linear/__init__.py \
  vllm/model_executor/layers/quantization/utils/nvfp4_utils.py \
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py; do
  echo "=== $f ==="
  git diff --staged upstream/main -- "$f" | head -5 || echo "MATCHES UPSTREAM"
done
```

Expected: Each file shows no diff or minimal diff (only if upstream has post-#39129 changes).

- [ ] **Step 3: Check for missing logger in compressed_tensors_w4a4_nvfp4.py**

```bash
grep -n 'init_logger' vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py
```

If missing, add `from vllm.logger import init_logger` and `logger = init_logger(__name__)` after the imports. The `checkout --theirs` should include this since it takes the upstream version of the file.

- [ ] **Step 4: Check kernels/linear/__init__.py has MMLinearKernel imports**

```bash
grep -n 'MMLinearKernel\|MMLinearLayerConfig' vllm/model_executor/kernels/linear/__init__.py | head -5
```

Expected: Imports from `.base` are present. If not, add manually:
```python
from vllm.model_executor.kernels.linear.base import (
    MMLinearKernel,
    MMLinearLayerConfig,
)
```

- [ ] **Step 5: Commit the resolved cherry-pick**

```bash
git commit -m "upstream: [Refactor] Move NVFP4 GEMM management into NvFp4LinearKernel (#39129)

Cherry-picked from upstream vllm-project/vllm@2800706f0.
Conflicts resolved by taking upstream side (no local NVFP4 dispatch modifications).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Cherry-pick #39322 — Batch-invariant NVFP4 linear**

```bash
git cherry-pick 201813724
```

Expected: Clean apply — depends on #39129 which is now in our tree.

---

### Task 3: Phase 3 — Cherry-pick NVFP4 MoE backend

- [ ] **Step 1: Cherry-pick #38251 — FlashInfer CuteDSL batched experts for NVFP4 MoE**

```bash
git cherry-pick e8ebbdde8
```

Expected: Clean apply. Mostly new files for CuteDSL MoE backend.

---

### Task 4: Verification Gate 1 — Build and Qwen3.5 regression test

- [ ] **Step 1: Docker build**

```bash
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/nvllm-build-phase3.log
```

Expected: Successful build. Watch for CUDA extension compile failures or OOM. Full output — do not pipe through `tail`.

- [ ] **Step 2: Serve Qwen3.5-27B NVFP4 and verify no regression**

Launch the existing Qwen3.5 serve script (or manually):

```bash
# Quick sanity — does it start and respond?
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":16}' | python3 -m json.tool
```

Expected: Coherent response ("Four" or "4"). If gibberish, stop — the NVFP4 refactor broke something.

- [ ] **Step 3: Check logs for errors**

```bash
docker logs nvllm 2>&1 | grep -iE 'error|exception|traceback|illegal|nan' | head -20
```

Expected: No fatal errors. Warnings about attention backend fallback are OK.

---

### Task 5: Phase 4 — Cherry-pick Gemma 4 follow-ups

- [ ] **Step 1: Cherry-pick #38879 — Fast prefill / YOCO KV-sharing for Gemma 4**

```bash
git cherry-pick 47e605092
```

Expected: Clean apply. Touches `vllm/model_executor/models/gemma4.py`.

- [ ] **Step 2: Cherry-pick #39045 — Quantized MoE weight loading**

```bash
git cherry-pick 3aecdf08b
```

Expected: Clean apply. Touches MoE weight loading path.

---

### Task 6: Update run script for RedHat HF checkpoint

**Files:**
- Modify: `scripts/run_gemma4.sh`

- [ ] **Step 1: Update run_gemma4.sh to support HF model ID**

The current script requires a local model path at `$HOME/.cache/huggingface/hub/gemma-4-31B-it-NVFP4`. Update it to accept an HF model ID and let vLLM pull it directly.

Change the `MODEL` default and the pre-flight check in `scripts/run_gemma4.sh`:

Replace:
```bash
MODEL="${GEMMA4_MODEL_PATH:-$HOME/.cache/huggingface/hub/gemma-4-31B-it-NVFP4}"
```
With:
```bash
MODEL="${GEMMA4_MODEL_PATH:-RedHatAI/gemma-4-31B-it-NVFP4}"
```

Replace the pre-flight check block (lines 35-44):
```bash
# Gemma4 must be quantized locally — no HF pull
if [ ! -d "$MODEL" ]; then
  echo "ERROR: NVFP4 checkpoint not found at: $MODEL" >&2
  echo "" >&2
  echo "Quantize it first:" >&2
  echo "  ./scripts/quantize_gemma4.sh" >&2
  echo "" >&2
  echo "Or point to an existing checkpoint:" >&2
  echo "  GEMMA4_MODEL_PATH=/path/to/gemma-4-31B-it-NVFP4 $0" >&2
  exit 1
fi
```
With:
```bash
# If MODEL is a local path, check it exists. If it looks like an HF ID, let vLLM pull it.
if [[ "$MODEL" != */* ]] && [ ! -d "$MODEL" ]; then
  echo "ERROR: Local model path not found: $MODEL" >&2
  echo "  Set GEMMA4_MODEL_PATH to a local path or HF model ID" >&2
  exit 1
elif [[ "$MODEL" == */* ]] && [ -d "$MODEL" ]; then
  # Local path with a slash — mount it
  MOUNT_ARGS="-v $MODEL:/model"
  MODEL="/model"
elif [[ "$MODEL" == */* ]]; then
  # HF model ID — vLLM pulls directly, mount HF cache for persistence
  MOUNT_ARGS=""
fi
```

Update the `docker run` command: replace the hardcoded `-v "$MODEL:/model"` mount with `$MOUNT_ARGS` and use `$MODEL` directly as the `--model` argument (it will be either `/model` for local paths or the HF ID for remote).

Update the header comment:
```bash
# Dense vision-language model with ngram speculative decoding.
# Supports both local checkpoints and HF model IDs (e.g., RedHatAI/gemma-4-31B-it-NVFP4).
```

Update the echo block to show the actual model value:
```bash
echo "  Model:       $MODEL"
```

- [ ] **Step 2: Similarly update serve_gemma4.sh**

In `scripts/serve_gemma4.sh`, change `--model /model` to accept an environment variable:

Replace:
```bash
  --model /model \
```
With:
```bash
  --model "${GEMMA4_MODEL:-/model}" \
```

Update the header:
```bash
# Supports both local mounts (-v /path:/model) and HF model IDs
# (set GEMMA4_MODEL=RedHatAI/gemma-4-31B-it-NVFP4).
```

---

### Task 7: Verification Gate 2 — Serve Gemma 4 31B NVFP4

- [ ] **Step 1: Rebuild Docker image**

```bash
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/nvllm-build-gemma4.log
```

- [ ] **Step 2: Launch Gemma 4 in debug mode (enforce-eager)**

```bash
./scripts/run_gemma4.sh --debug
```

Expected: Container starts, model loads from HF. Watch logs:

```bash
docker logs -f nvllm 2>&1 | tee /tmp/gemma4-debug.log
```

Wait for `Uvicorn running on http://0.0.0.0:8000` in the logs.

- [ ] **Step 3: Smoke test — basic chat**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":16}' | python3 -m json.tool
```

Expected: Coherent response. If gibberish, check `/tmp/gemma4-debug.log` for errors and follow CLAUDE.md debugging protocol.

- [ ] **Step 4: Smoke test — slightly longer output**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Explain the Pythagorean theorem in 3 sentences."}],"max_tokens":128}' | python3 -m json.tool
```

Expected: Coherent, multi-sentence response.

- [ ] **Step 5: Check logs for errors**

```bash
docker logs nvllm 2>&1 | grep -iE 'error|exception|traceback|illegal|nan' | head -20
```

Expected: No fatal errors.

- [ ] **Step 6: If debug mode passed, restart without --enforce-eager**

```bash
docker stop nvllm && docker rm nvllm
./scripts/run_gemma4.sh
```

Repeat the smoke tests from Steps 3-5. If CUDA graphs cause issues, note them and continue with `--enforce-eager` for now.

---

### Task 8: Phase 5 — Cherry-pick tool-use / reasoning fixes

- [ ] **Step 1: Cherry-pick #38909 — Fix streaming HTML duplication after tool calls**

```bash
git cherry-pick d734445fc
```

- [ ] **Step 2: Cherry-pick #38992 — Fix invalid JSON in streaming tool calls**

```bash
git cherry-pick f53fa26e0
```

- [ ] **Step 3: Cherry-pick #39027 — adjust_request for reasoning parser + Gemma 4 fixes**

```bash
git cherry-pick 8477fe427
```

- [ ] **Step 4: Cherry-pick #39114 — Fix streaming tool call corruption (bools/numbers)**

```bash
git cherry-pick 13151a4df
```

- [ ] **Step 5: Verify all 4 commits landed**

```bash
git log --oneline -5
```

Expected: 4 new commits with upstream PR titles.

---

### Task 9: Verification Gate 3 — Tool-use test

- [ ] **Step 1: Rebuild and relaunch**

```bash
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/nvllm-build-tooluse.log
docker stop nvllm && docker rm nvllm
./scripts/run_gemma4.sh
```

Wait for startup.

- [ ] **Step 2: Test tool-use request**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }],
    "max_tokens": 256
  }' | python3 -m json.tool
```

Expected: Response contains a `tool_calls` array with a valid `get_weather` function call and `{"city": "San Francisco"}` arguments as valid JSON.

- [ ] **Step 3: Test streaming tool-use**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }],
    "stream": true,
    "max_tokens": 256
  }'
```

Expected: SSE stream with `delta.tool_calls` chunks. No duplicated HTML, no corrupted JSON.

- [ ] **Step 4: Log final state**

```bash
echo "=== Cherry-pick complete ==="
git log --oneline $(git merge-base HEAD upstream/main)..HEAD | head -20
echo ""
echo "=== Upstream delta remaining ==="
git rev-list --count $(git merge-base HEAD upstream/main)..upstream/main
```

---

### Task 10: Commit script updates and spec

- [ ] **Step 1: Stage and commit script changes**

```bash
git add scripts/run_gemma4.sh scripts/serve_gemma4.sh
git commit -m "feat: update Gemma 4 scripts for RedHat HF checkpoint support

Scripts now accept HF model IDs (e.g., RedHatAI/gemma-4-31B-it-NVFP4)
in addition to local paths. GEMMA4_MODEL_PATH controls the model source.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Stage and commit spec + plan docs**

```bash
git add docs/superpowers/specs/2026-04-09-upstream-cherry-pick-nvfp4-gemma4-design.md
git add docs/superpowers/plans/2026-04-09-upstream-cherry-pick-nvfp4-gemma4.md
git commit -m "docs: add upstream cherry-pick spec and plan for NVFP4 + Gemma 4

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
