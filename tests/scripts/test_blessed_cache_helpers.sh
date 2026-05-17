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

test_compute_hash_rejects_wrong_arg_count() {
  echo "[test] compute_hash_rejects_wrong_arg_count"
  assert_exit_code "17 args -> return 1" 1 \
    nvllm_compute_blessed_config_hash 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
}

test_compute_hash_anchor_value() {
  echo "[test] compute_hash_anchor_value"
  # Anchor: any change to the jq filter body or arg order silently flips this
  # hash, so the test is the canonical-shape contract. Update only when the
  # 18-input contract is intentionally bumped (and bump all manifests).
  local anchor_hash="d427dc2be6147ace5097388d731c67d9fb482697a2b50607714d9951da525311"
  local h
  h=$(nvllm_compute_blessed_config_hash \
    "sha256:0000000000000000000000000000000000000000000000000000000000000000" \
    "test/anchor-model" \
    "0000000000000000000000000000000000000000" \
    "fp8_e4m3" "CUTE_PAGED" "FULL_AND_PIECEWISE" "[1]" \
    1 16384 65536 \
    1 "0,1,2,3,4,5,6,7" 1 0 0 0 1 1)
  assert_eq "anchor hash unchanged" "$anchor_hash" "$h"
}

# ---- resolve_blessed_manifest tests ----

setup_manifest_fixture() {
  local dir="$1"
  rm -rf "$dir"
  mkdir -p "$dir"
  cat > "$dir/cfg-aaa_aaaaaaa.json" <<'EOF'
{"schema_version":1,"config_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","files":[]}
EOF
  cat > "$dir/cfg-bbb_bbbbbbb.json" <<'EOF'
{"schema_version":1,"config_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","files":[]}
EOF
  mkdir -p "$dir/_archive"
  cat > "$dir/_archive/old-aaa.json" <<'EOF'
{"schema_version":1,"config_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","files":[]}
EOF
}

test_resolve_manifest_found() {
  echo "[test] resolve_manifest_found"
  local dir
  dir=$(mktemp -d)
  setup_manifest_fixture "$dir"
  local result
  result=$(NVLLM_BLESSED_MANIFEST_DIR="$dir" \
    nvllm_resolve_blessed_manifest "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
  assert_eq "found by config_hash" "$dir/cfg-aaa_aaaaaaa.json" "$result"
  rm -rf "$dir"
}

test_resolve_manifest_no_match_exits_1() {
  echo "[test] resolve_manifest_no_match_exits_1"
  local dir actual=0
  dir=$(mktemp -d)
  setup_manifest_fixture "$dir"
  # Subshell so the env var binds to the function call (env <var> <func> does
  # not work because shell functions are not external commands).
  (NVLLM_BLESSED_MANIFEST_DIR="$dir" \
    nvllm_resolve_blessed_manifest "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff") \
    >/dev/null 2>&1 || actual=$?
  if [ "$actual" = 1 ]; then
    PASS=$((PASS+1)); echo "  PASS: exit 1 on no match"
  else
    FAIL=$((FAIL+1)); FAIL_NAMES+=("resolve_manifest no_match")
    echo "  FAIL: expected 1, got $actual"
  fi
  rm -rf "$dir"
}

test_resolve_manifest_duplicate_exits_2() {
  echo "[test] resolve_manifest_duplicate_exits_2"
  local dir actual=0
  dir=$(mktemp -d)
  setup_manifest_fixture "$dir"
  cat > "$dir/cfg-aaa_dup.json" <<'EOF'
{"schema_version":1,"config_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","files":[]}
EOF
  (NVLLM_BLESSED_MANIFEST_DIR="$dir" \
    nvllm_resolve_blessed_manifest "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") \
    >/dev/null 2>&1 || actual=$?
  if [ "$actual" = 2 ]; then
    PASS=$((PASS+1)); echo "  PASS: exit 2 on duplicate"
  else
    FAIL=$((FAIL+1)); FAIL_NAMES+=("resolve_manifest duplicate")
    echo "  FAIL: expected 2, got $actual"
  fi
  rm -rf "$dir"
}

test_resolve_manifest_ignores_archive() {
  echo "[test] resolve_manifest_ignores_archive"
  local dir
  dir=$(mktemp -d)
  setup_manifest_fixture "$dir"
  # _archive/old-aaa.json has the same config_hash; should not be counted.
  local result
  result=$(NVLLM_BLESSED_MANIFEST_DIR="$dir" \
    nvllm_resolve_blessed_manifest "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
  assert_eq "ignores _archive/" "$dir/cfg-aaa_aaaaaaa.json" "$result"
  rm -rf "$dir"
}

test_resolve_manifest_warns_on_missing_config_hash() {
  echo "[test] resolve_manifest_warns_on_missing_config_hash"
  local dir actual=0
  dir=$(mktemp -d)
  setup_manifest_fixture "$dir"
  cat > "$dir/cfg-broken.json" <<'EOF'
{"schema_version":1,"files":[]}
EOF
  local stderr_capture
  stderr_capture=$(NVLLM_BLESSED_MANIFEST_DIR="$dir" \
    nvllm_resolve_blessed_manifest "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff" 2>&1 1>/dev/null) \
    || actual=$?
  if [ "$actual" = 1 ] && echo "$stderr_capture" | grep -q "missing/empty .config_hash"; then
    PASS=$((PASS+1)); echo "  PASS: warns on missing .config_hash, returns 1"
  else
    FAIL=$((FAIL+1)); FAIL_NAMES+=("resolve_manifest missing_config_hash")
    echo "  FAIL: actual=$actual stderr=$stderr_capture"
  fi
  rm -rf "$dir"
}

test_resolve_manifest_no_env_no_repo_returns_1() {
  echo "[test] resolve_manifest_no_env_no_repo_returns_1"
  local actual=0 stderr_capture
  # Run from /tmp with both env var unset and outside any git repo.
  # Use a subshell to clear env var; cd to a non-repo dir.
  stderr_capture=$(cd /tmp && unset NVLLM_BLESSED_MANIFEST_DIR && \
    nvllm_resolve_blessed_manifest "deadbeef" 2>&1 1>/dev/null) \
    || actual=$?
  # Note: BASH_SOURCE points to scripts/common.sh in the repo, so the
  # git-rev-parse-from-script-dir fallback DOES find the repo. To force the
  # error path, we'd need to source common.sh from outside the repo, which
  # is environment-dependent. Instead just verify the function returns
  # cleanly (return 1, no crash) when env unset and the repo IS available
  # (manifest dir would be docs/blessed-caches/, hash deadbeef won't match).
  if [ "$actual" = 1 ]; then
    PASS=$((PASS+1)); echo "  PASS: returns 1 when manifest dir defaulted and hash absent"
  else
    FAIL=$((FAIL+1)); FAIL_NAMES+=("resolve_manifest no_env path")
    echo "  FAIL: actual=$actual"
  fi
}

# ---- verify_blessed_cache tests ----

setup_cache_fixture() {
  local cache_dir="$1"
  rm -rf "$cache_dir"
  mkdir -p "$cache_dir/sub"
  printf 'hello' > "$cache_dir/sub/model"  # exact 5 bytes
}

test_verify_cache_pass() {
  echo "[test] verify_cache_pass"
  local cache_dir manifest_dir manifest sha
  cache_dir=$(mktemp -d)
  manifest_dir=$(mktemp -d)
  setup_cache_fixture "$cache_dir"
  sha=$(sha256sum "$cache_dir/sub/model" | awk '{print $1}')
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{"config_hash":"x","mount":{"host_path":"$cache_dir"},"files":[{"relative_path":"sub/model","sha256":"$sha","size_bytes":5,"role":"aot_model"}]}
EOF
  assert_exit_code "verify passes for matching cache" 0 \
    nvllm_verify_blessed_cache "$manifest"
  rm -rf "$cache_dir" "$manifest_dir"
}

test_verify_cache_fail_on_size() {
  echo "[test] verify_cache_fail_on_size"
  local cache_dir manifest_dir manifest sha
  cache_dir=$(mktemp -d); manifest_dir=$(mktemp -d)
  setup_cache_fixture "$cache_dir"
  sha=$(sha256sum "$cache_dir/sub/model" | awk '{print $1}')
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{"config_hash":"x","mount":{"host_path":"$cache_dir"},"files":[{"relative_path":"sub/model","sha256":"$sha","size_bytes":99,"role":"aot_model"}]}
EOF
  assert_exit_code "verify fails on size mismatch" 1 \
    nvllm_verify_blessed_cache "$manifest"
  rm -rf "$cache_dir" "$manifest_dir"
}

test_verify_cache_fail_on_sha() {
  echo "[test] verify_cache_fail_on_sha"
  local cache_dir manifest_dir manifest
  cache_dir=$(mktemp -d); manifest_dir=$(mktemp -d)
  setup_cache_fixture "$cache_dir"
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{"config_hash":"x","mount":{"host_path":"$cache_dir"},"files":[{"relative_path":"sub/model","sha256":"deadbeef","size_bytes":5,"role":"aot_model"}]}
EOF
  assert_exit_code "verify fails on sha mismatch" 1 \
    nvllm_verify_blessed_cache "$manifest"
  rm -rf "$cache_dir" "$manifest_dir"
}

test_verify_cache_fail_on_missing() {
  echo "[test] verify_cache_fail_on_missing"
  local cache_dir manifest_dir manifest
  cache_dir=$(mktemp -d); manifest_dir=$(mktemp -d)
  setup_cache_fixture "$cache_dir"
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{"config_hash":"x","mount":{"host_path":"$cache_dir"},"files":[{"relative_path":"missing.bin","sha256":"x","size_bytes":1,"role":"aot_model"}]}
EOF
  assert_exit_code "verify fails on missing file" 1 \
    nvllm_verify_blessed_cache "$manifest"
  rm -rf "$cache_dir" "$manifest_dir"
}

test_verify_cache_fail_on_zero_byte() {
  echo "[test] verify_cache_fail_on_zero_byte"
  local cache_dir manifest_dir manifest
  cache_dir=$(mktemp -d); manifest_dir=$(mktemp -d)
  mkdir -p "$cache_dir/sub"
  : > "$cache_dir/sub/model"  # 0-byte file
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{"config_hash":"x","mount":{"host_path":"$cache_dir"},"files":[{"relative_path":"sub/model","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size_bytes":0,"role":"aot_model"}]}
EOF
  # Even though sha and size match a true empty file, our verify rejects 0-byte explicitly.
  assert_exit_code "verify rejects zero-byte file" 1 \
    nvllm_verify_blessed_cache "$manifest"
  rm -rf "$cache_dir" "$manifest_dir"
}

test_verify_cache_fail_on_empty_files_array() {
  echo "[test] verify_cache_fail_on_empty_files_array"
  local cache_dir manifest_dir manifest
  cache_dir=$(mktemp -d); manifest_dir=$(mktemp -d)
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{"config_hash":"x","mount":{"host_path":"$cache_dir"},"files":[]}
EOF
  assert_exit_code "verify rejects empty files[] array" 1 \
    nvllm_verify_blessed_cache "$manifest"
  rm -rf "$cache_dir" "$manifest_dir"
}

# ---- schema v2 (mounts[] + flat files[] with mount_id) tests ----

test_verify_v2_pass() {
  echo "[test] verify_v2_pass"
  local vllm_root cute_root manifest_dir manifest
  local sha_aot sha_ko
  vllm_root=$(mktemp -d); cute_root=$(mktemp -d); manifest_dir=$(mktemp -d)
  mkdir -p "$vllm_root/sub"
  printf 'aot-bytes-padding' > "$vllm_root/sub/model"
  printf 'kernel-object-padding' > "$cute_root/k.o"
  sha_aot=$(sha256sum "$vllm_root/sub/model" | awk '{print $1}')
  sha_ko=$(sha256sum "$cute_root/k.o" | awk '{print $1}')
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{
  "schema_version": 2,
  "config_hash": "x",
  "mounts": [
    {"id":"vllm_cache","host_path":"$vllm_root","container_path":"/root/.cache/vllm","mode":"ro"},
    {"id":"cute_kernel_cache","host_path":"$cute_root","container_path":"/opt/vllm/kernel_cache","mode":"ro"}
  ],
  "files": [
    {"relative_path":"sub/model","sha256":"$sha_aot","size_bytes":$(stat -c '%s' "$vllm_root/sub/model"),"role":"aot_model","mount_id":"vllm_cache"},
    {"relative_path":"k.o","sha256":"$sha_ko","size_bytes":$(stat -c '%s' "$cute_root/k.o"),"role":"cute_native_object","mount_id":"cute_kernel_cache"}
  ]
}
EOF
  assert_exit_code "v2 verify passes" 0 nvllm_verify_blessed_cache "$manifest"
  rm -rf "$vllm_root" "$cute_root" "$manifest_dir"
}

test_verify_v2_fail_on_cute_sha_drift() {
  echo "[test] verify_v2_fail_on_cute_sha_drift"
  local vllm_root cute_root manifest_dir manifest sha_aot
  vllm_root=$(mktemp -d); cute_root=$(mktemp -d); manifest_dir=$(mktemp -d)
  mkdir -p "$vllm_root/sub"
  printf 'aot' > "$vllm_root/sub/model"
  printf 'kernel-obj' > "$cute_root/k.o"
  sha_aot=$(sha256sum "$vllm_root/sub/model" | awk '{print $1}')
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{
  "schema_version": 2,
  "config_hash": "x",
  "mounts": [
    {"id":"vllm_cache","host_path":"$vllm_root","container_path":"/root/.cache/vllm","mode":"ro"},
    {"id":"cute_kernel_cache","host_path":"$cute_root","container_path":"/opt/vllm/kernel_cache","mode":"ro"}
  ],
  "files": [
    {"relative_path":"sub/model","sha256":"$sha_aot","size_bytes":3,"role":"aot_model","mount_id":"vllm_cache"},
    {"relative_path":"k.o","sha256":"00deadbeef00","size_bytes":10,"role":"cute_native_object","mount_id":"cute_kernel_cache"}
  ]
}
EOF
  assert_exit_code "v2 verify fails on cute sha drift" 1 nvllm_verify_blessed_cache "$manifest"
  rm -rf "$vllm_root" "$cute_root" "$manifest_dir"
}

test_verify_v2_fail_on_unknown_mount_id() {
  echo "[test] verify_v2_fail_on_unknown_mount_id"
  local vllm_root manifest_dir manifest sha
  vllm_root=$(mktemp -d); manifest_dir=$(mktemp -d)
  mkdir -p "$vllm_root/sub"
  printf 'aot' > "$vllm_root/sub/model"
  sha=$(sha256sum "$vllm_root/sub/model" | awk '{print $1}')
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{
  "schema_version": 2,
  "config_hash": "x",
  "mounts": [
    {"id":"vllm_cache","host_path":"$vllm_root","container_path":"/root/.cache/vllm","mode":"ro"}
  ],
  "files": [
    {"relative_path":"sub/model","sha256":"$sha","size_bytes":3,"role":"aot_model","mount_id":"ghost_cache"}
  ]
}
EOF
  assert_exit_code "v2 verify fails on unknown mount_id" 1 nvllm_verify_blessed_cache "$manifest"
  rm -rf "$vllm_root" "$manifest_dir"
}

test_verify_v2_fail_on_missing_mount_host_path() {
  echo "[test] verify_v2_fail_on_missing_mount_host_path"
  local manifest_dir manifest
  manifest_dir=$(mktemp -d)
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{
  "schema_version": 2,
  "config_hash": "x",
  "mounts": [
    {"id":"vllm_cache","host_path":"/nonexistent/path/that/cannot/exist","container_path":"/root/.cache/vllm","mode":"ro"}
  ],
  "files": [
    {"relative_path":"sub/model","sha256":"x","size_bytes":1,"role":"aot_model","mount_id":"vllm_cache"}
  ]
}
EOF
  assert_exit_code "v2 verify fails on missing mount host_path" 1 nvllm_verify_blessed_cache "$manifest"
  rm -rf "$manifest_dir"
}

test_verify_unsupported_schema_rejected() {
  echo "[test] verify_unsupported_schema_rejected"
  local cache_dir manifest_dir manifest
  cache_dir=$(mktemp -d); manifest_dir=$(mktemp -d)
  manifest="$manifest_dir/m.json"
  cat > "$manifest" <<EOF
{"schema_version":999,"config_hash":"x","mounts":[],"files":[]}
EOF
  assert_exit_code "verify rejects schema_version=999" 1 \
    nvllm_verify_blessed_cache "$manifest"
  rm -rf "$cache_dir" "$manifest_dir"
}

# ---- refuse_* tests ----

test_refuse_no_manifest_exits_1_with_hint() {
  echo "[test] refuse_no_manifest_exits_1_with_hint"
  local out
  out=$(nvllm_refuse_no_manifest "abc123" 2>&1; echo "exit=$?")
  echo "$out" | grep -q "abc123" \
    && echo "$out" | grep -q "bless-cute-full-cache.sh" \
    && echo "$out" | grep -q "exit=1" \
    && { PASS=$((PASS+1)); echo "  PASS: refuse_no_manifest message + exit=1"; } \
    || { FAIL=$((FAIL+1)); FAIL_NAMES+=("refuse_no_manifest"); echo "  FAIL: refuse_no_manifest"; echo "$out"; }
}

test_refuse_cache_drift_exits_1_with_diagnostic() {
  echo "[test] refuse_cache_drift_exits_1_with_diagnostic"
  local out
  out=$(nvllm_refuse_cache_drift "abc123" "/tmp/m.json" 2>&1; echo "exit=$?")
  echo "$out" | grep -q "DRIFT DETECTED" \
    && echo "$out" | grep -q "exit=1" \
    && { PASS=$((PASS+1)); echo "  PASS: refuse_cache_drift"; } \
    || { FAIL=$((FAIL+1)); FAIL_NAMES+=("refuse_cache_drift"); echo "  FAIL: refuse_cache_drift"; echo "$out"; }
}

test_refuse_unsafe_dev_manifest_exits_1() {
  echo "[test] refuse_unsafe_dev_manifest_exits_1"
  local out
  out=$(nvllm_refuse_unsafe_dev_manifest "/tmp/m.json" 2>&1; echo "exit=$?")
  echo "$out" | grep -q "unsafe_dev_trials" \
    && echo "$out" | grep -q "exit=1" \
    && { PASS=$((PASS+1)); echo "  PASS: refuse_unsafe_dev_manifest"; } \
    || { FAIL=$((FAIL+1)); FAIL_NAMES+=("refuse_unsafe_dev_manifest"); echo "  FAIL"; echo "$out"; }
}

# ---- resolve_hf_revision tests (offline-only sanity) ----

test_resolve_hf_revision_function_exists() {
  echo "[test] resolve_hf_revision_function_exists"
  if declare -F nvllm_resolve_hf_revision >/dev/null; then
    PASS=$((PASS+1)); echo "  PASS: function defined"
  else
    FAIL=$((FAIL+1)); FAIL_NAMES+=("resolve_hf_revision missing"); echo "  FAIL"
  fi
}

# ---- refuse_if_container_exists tests ----

test_refuse_if_container_exists_returns_0_when_absent() {
  echo "[test] refuse_if_container_exists_returns_0_when_absent"
  # Use an unlikely name to guarantee absence.
  local fake_name="nvllm-test-$$-$(date +%s)"
  if nvllm_refuse_if_container_exists "$fake_name" >/dev/null 2>&1; then
    PASS=$((PASS+1)); echo "  PASS: returned 0 for absent container"
  else
    FAIL=$((FAIL+1)); FAIL_NAMES+=("refuse_if_container_exists absent")
    echo "  FAIL: returned non-zero for absent container"
  fi
}

# ---- main runner ----

main() {
  echo "=== blessed-cache helpers test suite ==="
  test_compute_hash_deterministic
  test_compute_hash_changes_on_image_id
  test_compute_hash_changes_on_probe_state
  test_compute_hash_emits_64_hex_chars
  test_compute_hash_rejects_wrong_arg_count
  test_compute_hash_anchor_value
  test_resolve_manifest_found
  test_resolve_manifest_no_match_exits_1
  test_resolve_manifest_duplicate_exits_2
  test_resolve_manifest_ignores_archive
  test_resolve_manifest_warns_on_missing_config_hash
  test_resolve_manifest_no_env_no_repo_returns_1
  test_verify_cache_pass
  test_verify_cache_fail_on_size
  test_verify_cache_fail_on_sha
  test_verify_cache_fail_on_missing
  test_verify_cache_fail_on_zero_byte
  test_verify_cache_fail_on_empty_files_array
  test_verify_v2_pass
  test_verify_v2_fail_on_cute_sha_drift
  test_verify_v2_fail_on_unknown_mount_id
  test_verify_v2_fail_on_missing_mount_host_path
  test_verify_unsupported_schema_rejected
  test_refuse_no_manifest_exits_1_with_hint
  test_refuse_cache_drift_exits_1_with_diagnostic
  test_refuse_unsafe_dev_manifest_exits_1
  test_resolve_hf_revision_function_exists
  test_refuse_if_container_exists_returns_0_when_absent

  echo ""
  echo "=== Summary: $PASS passed, $FAIL failed ==="
  if [ "$FAIL" -gt 0 ]; then
    echo "Failed tests:"
    for n in "${FAIL_NAMES[@]}"; do echo "  - $n"; done
    exit 1
  fi
}

main "$@"
