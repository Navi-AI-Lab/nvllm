#!/bin/bash
# Plain-bash test runner for scripts/common.sh blessed-cache helpers.
# Run: tests/scripts/test_blessed_cache_helpers.sh
# Pattern: each test is a function; main() calls them and tracks pass/fail.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
source "$REPO_ROOT/scripts/common.sh"

PASS=0
FAIL=0
FAIL_NAMES=()

assert_eq() {
  local desc="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    PASS=$((PASS+1))
    echo "  PASS: $desc"
  else
    FAIL=$((FAIL+1))
    FAIL_NAMES+=("$desc")
    echo "  FAIL: $desc"
    echo "    expected: $expected"
    echo "    actual:   $actual"
  fi
}

assert_neq() {
  local desc="$1" a="$2" b="$3"
  if [ "$a" != "$b" ]; then
    PASS=$((PASS+1))
    echo "  PASS: $desc"
  else
    FAIL=$((FAIL+1))
    FAIL_NAMES+=("$desc")
    echo "  FAIL: $desc"
    echo "    expected $a != $b but they're equal"
  fi
}

assert_exit_code() {
  local desc="$1" expected="$2"; shift 2
  local actual=0
  "$@" >/dev/null 2>&1 || actual=$?
  if [ "$expected" = "$actual" ]; then
    PASS=$((PASS+1))
    echo "  PASS: $desc"
  else
    FAIL=$((FAIL+1))
    FAIL_NAMES+=("$desc")
    echo "  FAIL: $desc (expected exit $expected, got $actual)"
  fi
}

# ---- compute_blessed_config_hash tests ----

test_compute_hash_deterministic() {
  echo "[test] compute_hash_deterministic"
  local h1 h2
  h1=$(nvllm_compute_blessed_config_hash \
    "sha256:abc" "ig1/M" "rev1" "fp8_e4m3" "CUTE_PAGED" \
    "FULL_AND_PIECEWISE" "[1]" 1 16384 65536 \
    1 "0,1,2,3,4,5,6,7" 1 0 0 0 1 1)
  h2=$(nvllm_compute_blessed_config_hash \
    "sha256:abc" "ig1/M" "rev1" "fp8_e4m3" "CUTE_PAGED" \
    "FULL_AND_PIECEWISE" "[1]" 1 16384 65536 \
    1 "0,1,2,3,4,5,6,7" 1 0 0 0 1 1)
  assert_eq "same args -> same hash" "$h1" "$h2"
}

test_compute_hash_changes_on_image_id() {
  echo "[test] compute_hash_changes_on_image_id"
  local h1 h2
  h1=$(nvllm_compute_blessed_config_hash \
    "sha256:aaa" "ig1/M" "rev1" "fp8_e4m3" "CUTE_PAGED" \
    "FULL_AND_PIECEWISE" "[1]" 1 16384 65536 \
    1 "0,1,2,3,4,5,6,7" 1 0 0 0 1 1)
  h2=$(nvllm_compute_blessed_config_hash \
    "sha256:bbb" "ig1/M" "rev1" "fp8_e4m3" "CUTE_PAGED" \
    "FULL_AND_PIECEWISE" "[1]" 1 16384 65536 \
    1 "0,1,2,3,4,5,6,7" 1 0 0 0 1 1)
  assert_neq "image_id change -> different hash" "$h1" "$h2"
}

test_compute_hash_changes_on_probe_state() {
  echo "[test] compute_hash_changes_on_probe_state"
  local h_off h_on
  h_off=$(nvllm_compute_blessed_config_hash \
    "sha256:abc" "ig1/M" "rev1" "fp8_e4m3" "CUTE_PAGED" \
    "FULL_AND_PIECEWISE" "[1]" 1 16384 65536 \
    1 "0,1,2,3,4,5,6,7" 1 0 0 0 1 1)
  h_on=$(nvllm_compute_blessed_config_hash \
    "sha256:abc" "ig1/M" "rev1" "fp8_e4m3" "CUTE_PAGED" \
    "FULL_AND_PIECEWISE" "[1]" 1 16384 65536 \
    1 "0,1,2,3,4,5,6,7" 1 1 0 0 1 1)  # cute_full_graph_probe=1
  assert_neq "probe state change -> different hash" "$h_off" "$h_on"
}

test_compute_hash_emits_64_hex_chars() {
  echo "[test] compute_hash_emits_64_hex_chars"
  local h
  h=$(nvllm_compute_blessed_config_hash \
    "sha256:abc" "ig1/M" "rev1" "fp8_e4m3" "CUTE_PAGED" \
    "FULL_AND_PIECEWISE" "[1]" 1 16384 65536 \
    1 "0,1,2,3,4,5,6,7" 1 0 0 0 1 1)
  if [[ "$h" =~ ^[0-9a-f]{64}$ ]]; then
    PASS=$((PASS+1))
    echo "  PASS: hash is 64 hex chars: $h"
  else
    FAIL=$((FAIL+1))
    echo "  FAIL: hash is not 64 hex chars: $h"
    FAIL_NAMES+=("hash format")
  fi
}

# ---- main runner ----

main() {
  echo "=== blessed-cache helpers test suite ==="
  test_compute_hash_deterministic
  test_compute_hash_changes_on_image_id
  test_compute_hash_changes_on_probe_state
  test_compute_hash_emits_64_hex_chars

  echo ""
  echo "=== Summary: $PASS passed, $FAIL failed ==="
  if [ "$FAIL" -gt 0 ]; then
    echo "Failed tests:"
    for n in "${FAIL_NAMES[@]}"; do echo "  - $n"; done
    exit 1
  fi
}

main "$@"
