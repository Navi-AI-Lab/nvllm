#!/usr/bin/env python3
"""Dump QK score s0 per-thread from production kernel.

Reads debug output at head 0, dims 240-271 which contain s0 values
for warp 0's 32 lanes. If s0 varies by group (lane//4), the QK MMA
output is group-dependent, proving the root cause.
"""
import torch, sys
logging_level = "WARNING"
__import__("logging").basicConfig(level=logging_level)

device = "cuda"
num_q_heads, num_kv_heads, head_dim = 24, 4, 256
page_size, seq_len = 64, 6
scale = 1.0 / (head_dim ** 0.5)

query = torch.zeros(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
query[0, :, 0] = 1.0

kv_shape = (2, page_size, num_kv_heads, head_dim)
k_cache = torch.zeros(kv_shape, dtype=torch.uint8, device=device)
fp8 = {1: 0x38, 2: 0x40, 3: 0x42, 4: 0x44, 5: 0x45, 6: 0x46}
for t in range(seq_len):
    for h in range(num_kv_heads):
        k_cache[0, t, h, 0] = fp8[t + 1]

v_cache = torch.zeros(kv_shape, dtype=torch.uint8, device=device)
for t in range(seq_len):
    v_cache[0, t, :, t] = 0x38

page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device=device)

from vllm.v1.attention.backends.cute_paged.kernel import (
    _get_compiled_kernel, DECODE_CONFIG, _CUTE_AVAILABLE,
)
if not _CUTE_AVAILABLE:
    print("CUTLASS not available"); sys.exit(1)

kernel = _get_compiled_kernel(DECODE_CONFIG)
print("Compiling...")
out = kernel(
    query=query, k_cache=k_cache.contiguous(),
    v_cache=v_cache.contiguous(), page_table=page_table,
    seq_lens=seq_lens_t, scale=scale, k_scale=1.0, v_scale=1.0,
    page_size=page_size,
)
c = out.float()

# Read s0 debug dump from HEAD 6, dims 0-31
s0_vals = c[0, 6, :32].cpu()
print(f"\ns0 per-lane (head 6, dims 0-31):")
print(f"  Raw: {s0_vals.tolist()}")

print(f"\ns0 by (group, sub) — should be CONSTANT across groups:")
for g in range(8):
    for s in range(4):
        lane = g * 4 + s
        val = s0_vals[lane].item()
        if abs(val) > 1e-6:
            print(f"  lane {lane:2d} (g={g}, s={s}): s0={val:.6f}")

# Check if s0 varies by group for sub=0
sub0_vals = [s0_vals[g * 4].item() for g in range(8)]
unique = len(set(f"{v:.4f}" for v in sub0_vals if abs(v) > 1e-6))
print(f"\ns0 at sub=0 across groups: {[f'{v:.4f}' for v in sub0_vals]}")
if unique > 1:
    print("CONFIRMED: s0 varies by group → QK MMA output is group-dependent!")
    print("Root cause: QK D-fragment mapping or QK score computation is wrong")
else:
    print("s0 is constant across groups → QK is correct, bug is elsewhere")

# Also show the sync_o dump (head 0, dims 0-15)
print(f"\nsync_o dump (head 0, dims 0-15): {c[0, 0, :16].tolist()}")
print(f"sync_o dump (head 1, dims 0-15): {c[0, 1, :16].tolist()}")
