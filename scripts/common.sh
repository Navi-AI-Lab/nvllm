#!/bin/bash
# nvllm -- Common helper functions for run scripts
#
# Source this file from run scripts:
#   source "$(dirname "$0")/common.sh"

NVLLM_IMAGE="${NVLLM_IMAGE:-nvllm:gb10}"

# ---------------------------------------------------------------------------
# nvllm_check_image
#   Verify the nvllm:gb10 Docker image exists locally.
# ---------------------------------------------------------------------------
nvllm_check_image() {
  local image="${1:-$NVLLM_IMAGE}"
  if ! docker image inspect "$image" &>/dev/null; then
    echo "ERROR: Docker image '$image' not found." >&2
    echo "" >&2
    echo "Build it first (from the repo root):" >&2
    echo "  cd $(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || echo /home/natfii/docker/nvllm)" >&2
    echo "  docker build -t nvllm:gb10 -f docker/Dockerfile.gb10 ." >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# nvllm_check_hf_auth
#   Verify Hugging Face authentication via HF_TOKEN env var or CLI login.
# ---------------------------------------------------------------------------
nvllm_check_hf_auth() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return 0
  fi
  if command -v huggingface-cli &>/dev/null && huggingface-cli whoami &>/dev/null; then
    return 0
  fi
  echo "ERROR: Hugging Face authentication not found." >&2
  echo "" >&2
  echo "Set up authentication with one of:" >&2
  echo "  export HF_TOKEN=hf_..." >&2
  echo "  huggingface-cli login" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# nvllm_ensure_model MODEL_ID
#   Check if a model is cached in ~/.cache/huggingface/hub.
#   If not, download it with huggingface-cli.
# ---------------------------------------------------------------------------
nvllm_ensure_model() {
  local model_id="$1"
  local hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
  local cache_dir="$hf_home/hub/models--${model_id/\//--}"

  if [[ -d "$cache_dir" ]]; then
    echo "Model found in cache: $model_id"
    return 0
  fi

  echo "Model '$model_id' not found in cache — downloading..."
  nvllm_check_hf_auth
  if ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface-cli not found. Install with: pip install huggingface_hub[cli]" >&2
    exit 1
  fi
  huggingface-cli download "$model_id"
}

# ---------------------------------------------------------------------------
# nvllm_cleanup_container NAME
#   Remove an existing container (force, ignore errors).
# ---------------------------------------------------------------------------
nvllm_cleanup_container() {
  local name="$1"
  docker rm -f "$name" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# nvllm_check_port PORT
#   Warn if the port is already in use by another Docker container.
# ---------------------------------------------------------------------------
nvllm_check_port() {
  local port="$1"
  local user
  user=$(docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null \
    | grep -E ":${port}->" | awk '{print $1}' | head -1 || true)
  if [ -n "$user" ]; then
    echo "WARNING: Port $port is already in use by container '$user'." >&2
    echo "         Stop it first or choose a different port." >&2
  fi
}

# ---------------------------------------------------------------------------
# nvllm_check_free_mem MIN_GB
#   Abort if host MemAvailable is below MIN_GB. Guards against OOM during
#   kernel dev (CUDA graph capture + nsys + ptxas overhead can eat 5+ GiB
#   after load). Uses /proc/meminfo MemAvailable — counts cached-but-
#   reclaimable memory as free. GB10 unified memory: CPU free == GPU free.
# ---------------------------------------------------------------------------
nvllm_check_free_mem() {
  local min_gb="${1:-90}"
  local free_gb
  free_gb=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
  if [ "$free_gb" -lt "$min_gb" ]; then
    echo "ERROR: only ${free_gb} GiB host memory available; need >= ${min_gb} GiB" >&2
    echo "       for Phase E kernel dev (model + KV cache + graph capture + scratch)." >&2
    echo "       Stop stale containers: docker ps; docker stop <name>" >&2
    echo "       Or export NVLLM_SKIP_MEM_CHECK=1 to bypass (risky)." >&2
    if [ "${NVLLM_SKIP_MEM_CHECK:-0}" != "1" ]; then
      exit 1
    fi
    echo "       Bypassing because NVLLM_SKIP_MEM_CHECK=1." >&2
  else
    echo "Host memory: ${free_gb} GiB available (threshold ${min_gb} GiB) — OK"
  fi
}

# ---------------------------------------------------------------------------
# nvllm_compute_blessed_config_hash <18 positional args>
#   Emits a deterministic sha256 hex string identifying a blessed-cache
#   configuration. Argument order is part of the contract — changing the
#   order or set of args breaks all existing manifests.
#
# Args (in order):
#   1  blessed_image_id              (e.g. "sha256:d3ddffea3c...")
#   2  model_id                      (e.g. "ig1/Qwen3.5-27B-NVFP4")
#   3  model_revision_resolved       (full HF commit sha)
#   4  kv_cache_dtype                (e.g. "fp8_e4m3")
#   5  attention_backend             (e.g. "CUTE_PAGED")
#   6  cudagraph_mode                (e.g. "FULL_AND_PIECEWISE")
#   7  cudagraph_capture_sizes_json  (e.g. "[1]")
#   8  max_num_seqs
#   9  max_model_len
#   10 max_num_batched_tokens
#   11 cute_phase_e_fusion           (0 or 1)
#   12 cute_phase_e_layers           (csv, e.g. "0,1,2,3,4,5,6,7")
#   13 cute_phase_e_fallback_raise   (0 or 1)
#   14 cute_full_graph_probe         (0 or 1)
#   15 cute_wo_reset_log             (0 or 1)
#   16 cute_dispatch_audit           (0 or 1)
#   17 cute_mlp_fusion               (0 or 1)
#   18 cute_attn_fusion              (0 or 1)
# ---------------------------------------------------------------------------
nvllm_compute_blessed_config_hash() {
  if [ "$#" -ne 18 ]; then
    echo "ERROR: nvllm_compute_blessed_config_hash requires 18 args, got $#" >&2
    return 1
  fi
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq not found. Install jq (apt-get install jq)." >&2
    return 1
  fi
  local canonical
  canonical=$(jq -cnS \
    --arg image_id "$1" \
    --arg model_id "$2" \
    --arg model_revision_resolved "$3" \
    --arg kv_cache_dtype "$4" \
    --arg attention_backend "$5" \
    --arg cudagraph_mode "$6" \
    --argjson cudagraph_capture_sizes "$7" \
    --argjson max_num_seqs "$8" \
    --argjson max_model_len "$9" \
    --argjson max_num_batched_tokens "${10}" \
    --argjson cute_phase_e_fusion "${11}" \
    --arg cute_phase_e_layers "${12}" \
    --argjson cute_phase_e_fallback_raise "${13}" \
    --argjson cute_full_graph_probe "${14}" \
    --argjson cute_wo_reset_log "${15}" \
    --argjson cute_dispatch_audit "${16}" \
    --argjson cute_mlp_fusion "${17}" \
    --argjson cute_attn_fusion "${18}" \
    '{blessed_image_id: $image_id,
      config: {
        model_id: $model_id,
        model_revision_resolved: $model_revision_resolved,
        kv_cache_dtype: $kv_cache_dtype,
        attention_backend: $attention_backend,
        cudagraph_mode: $cudagraph_mode,
        cudagraph_capture_sizes: $cudagraph_capture_sizes,
        max_num_seqs: $max_num_seqs,
        max_model_len: $max_model_len,
        max_num_batched_tokens: $max_num_batched_tokens,
        cute_phase_e_fusion: $cute_phase_e_fusion,
        cute_phase_e_layers: $cute_phase_e_layers,
        cute_phase_e_fallback_raise: $cute_phase_e_fallback_raise,
        cute_full_graph_probe: $cute_full_graph_probe,
        cute_wo_reset_log: $cute_wo_reset_log,
        cute_dispatch_audit: $cute_dispatch_audit,
        cute_mlp_fusion: $cute_mlp_fusion,
        cute_attn_fusion: $cute_attn_fusion
      }}')
  printf '%s' "$canonical" | sha256sum | awk '{print $1}'
}

# ---------------------------------------------------------------------------
# nvllm_resolve_blessed_manifest <config_hash>
#   Glob $NVLLM_BLESSED_MANIFEST_DIR/*.json (default: docs/blessed-caches/),
#   excluding _archive/. Find every manifest whose .config_hash equals input.
#   Return 0 with path on stdout if exactly one match.
#   Return 1 if zero matches ("no manifest").
#   Return 2 if 2+ matches ("corruption: duplicate config_hash").
# ---------------------------------------------------------------------------
nvllm_resolve_blessed_manifest() {
  local needle="$1"
  local dir="${NVLLM_BLESSED_MANIFEST_DIR:-}"
  if [ -z "$dir" ]; then
    local repo_root
    repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null) || {
      echo "ERROR: NVLLM_BLESSED_MANIFEST_DIR unset and not in a git repo." >&2
      return 1
    }
    dir="$repo_root/docs/blessed-caches"
  fi
  if [ ! -d "$dir" ]; then
    return 1  # no dir means no manifests
  fi
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq not found." >&2
    return 1
  fi
  local matches=()
  shopt -s nullglob
  local f hash
  for f in "$dir"/*.json; do
    hash=$(jq -r '.config_hash // empty' "$f" 2>/dev/null) || continue
    if [ "$hash" = "$needle" ]; then
      matches+=("$f")
    fi
  done
  shopt -u nullglob
  case "${#matches[@]}" in
    0) return 1 ;;
    1) printf '%s\n' "${matches[0]}"; return 0 ;;
    *) echo "ERROR: duplicate config_hash $needle in:" >&2
       printf '  %s\n' "${matches[@]}" >&2
       return 2 ;;
  esac
}

# ---------------------------------------------------------------------------
# nvllm_verify_blessed_cache <manifest_path>
#   Read manifest's mount.host_path and files[]. For each file: must exist,
#   non-empty, size match, sha256 match. Return 0 on full pass; return 1 on
#   any drift with diagnostic to stderr.
# ---------------------------------------------------------------------------
nvllm_verify_blessed_cache() {
  local manifest="$1"
  if [ ! -f "$manifest" ]; then
    echo "ERROR: manifest not found: $manifest" >&2
    return 1
  fi
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq not found." >&2
    return 1
  fi
  local host_path
  host_path=$(jq -r '.mount.host_path' "$manifest")
  host_path="${host_path/#\~/$HOME}"
  if [ ! -d "$host_path" ]; then
    echo "ERROR: blessed cache host_path missing: $host_path" >&2
    return 1
  fi

  local n
  n=$(jq -r '.files | length' "$manifest")
  if [ "$n" -lt 1 ]; then
    echo "ERROR: manifest has no files[] entries: $manifest" >&2
    return 1
  fi

  local i=0 rel expected_sha expected_size full actual_sha actual_size
  while [ "$i" -lt "$n" ]; do
    rel=$(jq -r ".files[$i].relative_path" "$manifest")
    expected_sha=$(jq -r ".files[$i].sha256" "$manifest")
    expected_size=$(jq -r ".files[$i].size_bytes" "$manifest")
    full="$host_path/$rel"
    if [ ! -f "$full" ]; then
      echo "ERROR: blessed-cache file missing: $full" >&2
      return 1
    fi
    actual_size=$(stat -c '%s' "$full")
    if [ "$actual_size" -eq 0 ]; then
      echo "ERROR: blessed-cache file is zero-byte: $full" >&2
      return 1
    fi
    if [ "$actual_size" != "$expected_size" ]; then
      echo "ERROR: size mismatch for $rel: expected $expected_size, got $actual_size" >&2
      return 1
    fi
    actual_sha=$(sha256sum "$full" | awk '{print $1}')
    if [ "$actual_sha" != "$expected_sha" ]; then
      echo "ERROR: sha256 mismatch for $rel:" >&2
      echo "  expected: $expected_sha" >&2
      echo "  actual:   $actual_sha" >&2
      return 1
    fi
    i=$((i+1))
  done
  return 0
}
