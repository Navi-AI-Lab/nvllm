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
