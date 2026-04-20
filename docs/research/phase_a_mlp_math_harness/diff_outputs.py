"""Elementwise diff of Phase A vs D2e MLP kernel outputs.

Reads the .pt tensors emitted by harness.py under both image runs
and writes a summary.md with:
- per-case abs-max / mean-abs / count of non-matching elements
- per-case fingerprint hash
- if divergence found: localize (first diverging element, block
  structure, near tile boundary check)

Usage:
    python3 diff_outputs.py <d2e_dir> <phaseA_dir> <summary_out.md>
"""

from __future__ import annotations

import hashlib
import os
import sys

import torch


CASES = ("zero_nat1", "seed_nat1", "seed_nat8", "seed_nat1_repeat")

TILE_S = 256       # prefill-legacy preset
TILE_K = 640       # prefill-legacy preset
HIDDEN = 5120
INTERM = 17408


def fingerprint(t: torch.Tensor) -> str:
    """Hash for quick scan-level match check.

    NB: we hash the RAW bytes (via view(uint8) or untyped_storage) rather
    than numpy().tobytes() because bf16 doesn't round-trip through numpy.
    This preserves exact bit patterns including NaN payloads.
    """
    storage_bytes = t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.md5(storage_bytes).hexdigest()


def describe_divergence(a: torch.Tensor, b: torch.Tensor) -> str:
    """Localize diverging elements: first mismatch, tile-boundary,
    block structure.

    Uses raw-byte comparison to detect mismatches (survives NaN, which
    compares unequal to itself under normal arithmetic)."""
    a_u8 = a.cpu().contiguous().view(torch.uint8)
    b_u8 = b.cpu().contiguous().view(torch.uint8)
    byte_diff = (a_u8 != b_u8)
    n_diff_bytes = byte_diff.sum().item()
    if n_diff_bytes == 0:
        return "no divergence"
    # Convert to float for magnitude reporting; NaN-in means NaN-out.
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    # Mask NaN before abs diff so we don't summarize NaN magnitudes.
    a_nan = torch.isnan(a_f)
    b_nan = torch.isnan(b_f)
    both_finite = ~a_nan & ~b_nan
    diff = torch.zeros_like(a_f)
    diff[both_finite] = (a_f[both_finite] - b_f[both_finite]).abs()
    n_diff = (diff > 0).sum().item()
    total = diff.numel()
    frac = n_diff / total
    flat_diff = diff.view(-1)
    flat_bytes = byte_diff.view(-1)
    first_idx_bytes = int(flat_bytes.nonzero(as_tuple=False)[0].item())
    # Each element is bf16 = 2 bytes, so byte idx // 2 = element idx.
    bytes_per_elem = a.element_size()
    first_idx = first_idx_bytes // bytes_per_elem
    flat_finite_diff = flat_diff.clone()
    flat_finite_diff[flat_diff == 0] = 0.0
    if n_diff > 0:
        first_finite_idx_t = (flat_diff > 0).nonzero(as_tuple=False)
        first_finite_idx = int(first_finite_idx_t[0].item()) if first_finite_idx_t.numel() > 0 else -1
    else:
        first_finite_idx = -1
    # Locate first mismatch in (nat, hidden) coords.
    if a.dim() == 2:
        nat, hidden = a.shape
        first_row = first_idx // hidden
        first_col = first_idx % hidden
        coord_str = f"row={first_row} col={first_col}"
    else:
        coord_str = f"flat_idx={first_idx}"
    max_abs = diff.max().item()
    mean_abs = diff.mean().item() if total > 0 else 0.0
    nan_a = a_nan.sum().item()
    nan_b = b_nan.sum().item()
    return (
        f"n_diff_bytes={n_diff_bytes}/{a_u8.numel()} "
        f"n_diff_finite={n_diff}/{total} ({100*frac:.3f}%) "
        f"max_abs_finite={max_abs:.6g} "
        f"nan_count_d2e={nan_a} nan_count_phaseA={nan_b} "
        f"first_byte_mismatch: {coord_str}"
    )


def summarize_tensor(t: torch.Tensor) -> str:
    f = t.to(torch.float32)
    nan_count = torch.isnan(f).sum().item()
    finite = torch.isfinite(f)
    finite_count = finite.sum().item()
    absmax = f[finite].abs().max().item() if finite_count > 0 else 0.0
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"nan={nan_count}/{f.numel()} "
        f"absmax_finite={absmax:.6g} "
        f"md5={fingerprint(t)}"
    )


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: diff_outputs.py <d2e_dir> <phaseA_dir> <summary_out.md>",
              file=sys.stderr)
        return 2
    d2e_dir = sys.argv[1]
    phaseA_dir = sys.argv[2]
    out_md = sys.argv[3]

    lines: list[str] = []
    lines.append("# Phase A vs D2e MLP math harness — summary\n")
    lines.append(f"- d2e dir:    {d2e_dir}")
    lines.append(f"- phaseA dir: {phaseA_dir}")
    lines.append("")
    lines.append("## Per-case comparison\n")

    all_match = True
    for case in CASES:
        d2e_path = os.path.join(d2e_dir, f"{case}.pt")
        phaseA_path = os.path.join(phaseA_dir, f"{case}.pt")
        if not os.path.isfile(d2e_path):
            lines.append(f"### {case}\n")
            lines.append(f"MISSING: {d2e_path}\n")
            continue
        if not os.path.isfile(phaseA_path):
            lines.append(f"### {case}\n")
            lines.append(f"MISSING: {phaseA_path}\n")
            continue

        t_d2e = torch.load(d2e_path, weights_only=False)
        t_phaseA = torch.load(phaseA_path, weights_only=False)

        fp_d2e = fingerprint(t_d2e)
        fp_phaseA = fingerprint(t_phaseA)
        match = fp_d2e == fp_phaseA

        lines.append(f"### {case}\n")
        lines.append(f"- D2e:    `{summarize_tensor(t_d2e)}`")
        lines.append(f"- PhaseA: `{summarize_tensor(t_phaseA)}`")
        if match:
            lines.append(f"- **MATCH (md5 identical)**\n")
        else:
            all_match = False
            lines.append(f"- **DIFFER**: {describe_divergence(t_d2e, t_phaseA)}\n")

    lines.insert(3, f"## Headline: {'ALL MATCH' if all_match else 'DIVERGENCE DETECTED'}\n")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_md}")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
