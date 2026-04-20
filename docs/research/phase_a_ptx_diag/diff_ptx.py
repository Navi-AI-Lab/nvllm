"""Normalize + side-by-side diff two CuTe DSL PTX dumps and tag hunks.

Usage:
    python3 diff_ptx.py <d2e_ptx> <phaseA_ptx> <output_diff.txt>

Normalizations applied before diffing:
  - rename kernel-name hash suffix to <KNAME>
  - zero-out .loc directives (line numbers drift with source)
  - zero-out .extern function name hashes

Hunks are tagged by first-matching category heuristic:
  REGALLOC     - lines containing .reg or %r/%f/%p/%rd register decls
  BRANCH       - setp., @p, bra, ret
  FP4_CONVERT  - cvt.rn.*, cvt.rp.*, cvt.f32.f4*, fp4, e2m1, ue4m3
  SMEM         - ld.shared, st.shared, .shared
  GMEM         - ld.global, st.global, atom.global
  ARITH        - fma., mul., add., mad., fp16/bf16/fp32
  MISC         - everything else

Category counts are printed at the top of the output.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


KERNEL_NAME_RE = re.compile(r"(cute_dsl_kernel|Phase_D_MLP_Kernel|_jit_launch)[_a-zA-Z0-9]*")
LOC_RE = re.compile(r"\.loc\s+\d+\s+\d+\s+\d+")
EXTERN_RE = re.compile(r"\.extern\s+\.func\s+[_a-zA-Z0-9]+")

CATEGORIES = [
    ("FP4_CONVERT", re.compile(r"\bcvt\.|\bfp4\b|\be2m1\b|\bue4m3\b", re.IGNORECASE)),
    ("BRANCH",      re.compile(r"\bsetp\.|\bbra\b|\bret\b|\@%p\d")),
    ("SMEM",        re.compile(r"\.shared\b|\bld\.shared\b|\bst\.shared\b")),
    ("GMEM",        re.compile(r"\bld\.global\b|\bst\.global\b|\batom\.global\b")),
    ("REGALLOC",    re.compile(r"\.reg\s+|%r\d+|%f\d+|%p\d+|%rd\d+")),
    ("ARITH",       re.compile(r"\bfma\.|\bmul\.|\badd\.|\bmad\.|\.f16\b|\.bf16\b|\.f32\b")),
]


def normalize(text: str) -> str:
    text = KERNEL_NAME_RE.sub("<KNAME>", text)
    text = LOC_RE.sub(".loc 0 0 0", text)
    text = EXTERN_RE.sub(".extern .func <EXT>", text)
    return text


def categorize(line: str) -> str:
    for name, pat in CATEGORIES:
        if pat.search(line):
            return name
    return "MISC"


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: diff_ptx.py <d2e_ptx> <phaseA_ptx> <output_diff.txt>", file=sys.stderr)
        return 2
    d2e_path = Path(sys.argv[1])
    phaseA_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    d2e_text = normalize(d2e_path.read_text())
    phaseA_text = normalize(phaseA_path.read_text())

    d2e_lines = d2e_text.splitlines()
    phaseA_lines = phaseA_text.splitlines()

    import difflib
    diff_iter = difflib.unified_diff(
        d2e_lines, phaseA_lines,
        fromfile=str(d2e_path), tofile=str(phaseA_path),
        n=3, lineterm="",
    )

    category_counts: dict[str, int] = {}
    buffer: list[str] = []
    for raw in diff_iter:
        buffer.append(raw)
        if raw.startswith(("+", "-")) and not raw.startswith(("+++", "---")):
            cat = categorize(raw[1:])
            category_counts[cat] = category_counts.get(cat, 0) + 1

    header = [
        f"# PTX diff: {d2e_path.name} vs {phaseA_path.name}",
        f"# d2e lines:     {len(d2e_lines)}",
        f"# phaseA lines:  {len(phaseA_lines)}",
        "# category counts (changed lines, D2e->Phase A):",
    ]
    for cat, count in sorted(category_counts.items(), key=lambda kv: -kv[1]):
        header.append(f"#   {cat:<12} {count}")
    header.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header + buffer) + "\n")
    print(f"wrote {out_path} ({len(buffer)} diff lines)")
    print("category counts:")
    for cat, count in sorted(category_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<12} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
