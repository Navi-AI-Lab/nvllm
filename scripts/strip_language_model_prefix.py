#!/usr/bin/env python3
"""Strip the 'language_model.' prefix from all weight keys in a safetensors checkpoint.

Uses memory-mapped access so the full 17+ GB file is not loaded into RAM at once.
Writes to a temporary file first, then atomically replaces the original.
"""

import os
import sys
import tempfile
import time

from safetensors import safe_open
from safetensors.torch import save_file

CHECKPOINT = "/home/natfii/.cache/huggingface/hub/Qwen3.5-27B-NVFP4/model.safetensors"
# Checkpoint keys: model.language_model.layers.X... -> model.layers.X...
OLD_PREFIX = "model.language_model."
NEW_PREFIX = "model."


def main():
    if not os.path.isfile(CHECKPOINT):
        print(f"ERROR: file not found: {CHECKPOINT}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening {CHECKPOINT} (memory-mapped) ...")
    f = safe_open(CHECKPOINT, framework="pt", device="cpu")
    old_keys = sorted(f.keys())
    total = len(old_keys)
    prefixed = sum(1 for k in old_keys if k.startswith(OLD_PREFIX))

    print(f"Total tensors : {total}")
    print(f"With prefix   : {prefixed}")
    print(f"Without prefix: {total - prefixed}")
    print()

    # Show a sample of keys before renaming
    print("=== Sample keys BEFORE rename ===")
    for k in old_keys[:10]:
        print(f"  {k}")
    if total > 10:
        print(f"  ... ({total - 10} more)")
    print()

    if prefixed == 0:
        print("Nothing to do -- no keys have the prefix.")
        sys.exit(0)

    # Build the renamed tensor dict.
    # safe_open.get_tensor() returns a memory-mapped tensor, so peak RSS stays
    # modest (each tensor is mapped on demand and can be paged out).
    print("Building renamed tensor dictionary ...")
    new_tensors = {}
    for i, key in enumerate(old_keys, 1):
        new_key = NEW_PREFIX + key[len(OLD_PREFIX):] if key.startswith(OLD_PREFIX) else key
        if new_key in new_tensors:
            print(f"ERROR: collision after stripping prefix: '{key}' -> '{new_key}' "
                  f"(already mapped)", file=sys.stderr)
            sys.exit(1)
        new_tensors[new_key] = f.get_tensor(key)
        if i % 100 == 0 or i == total:
            print(f"  loaded {i}/{total} tensors", end="\r")
    print()

    # Preserve any metadata
    metadata = f.metadata()

    # Write to a temp file in the same directory (same filesystem -> atomic rename)
    out_dir = os.path.dirname(CHECKPOINT)
    print(f"Writing new checkpoint to temp file in {out_dir} ...")
    t0 = time.time()

    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".safetensors.tmp")
    os.close(fd)
    try:
        save_file(new_tensors, tmp_path, metadata=metadata)
        elapsed = time.time() - t0
        tmp_size = os.path.getsize(tmp_path)
        print(f"Wrote {tmp_size / 1e9:.2f} GB in {elapsed:.1f}s")

        # Atomic replace
        os.replace(tmp_path, CHECKPOINT)
        print(f"Replaced original file.")
    except BaseException:
        # Clean up on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    # Verify by re-reading
    print()
    print("=== Verifying result ===")
    v = safe_open(CHECKPOINT, framework="pt", device="cpu")
    vkeys = sorted(v.keys())
    print(f"Total tensors: {len(vkeys)}")
    still_prefixed = sum(1 for k in vkeys if k.startswith(OLD_PREFIX))
    print(f"Still prefixed: {still_prefixed}")
    print()
    print("Sample keys AFTER rename:")
    for k in vkeys[:10]:
        print(f"  {k}")
    if len(vkeys) > 10:
        print(f"  ... ({len(vkeys) - 10} more)")

    if still_prefixed == 0:
        print()
        print("SUCCESS: all 'model.language_model.' prefixes remapped to 'model.'.")
    else:
        print()
        print(f"WARNING: {still_prefixed} keys still have the prefix!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
