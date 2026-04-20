# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""Phase D3a — tile-preset registry for `Phase_D_MLP_Kernel`.

This module exists as a sibling of `mlp_kernel.py`, NOT inside it, by
design. See the hard-won investigation below before collapsing it.

────────────────────────────────────────────────────────────────────────
Why this file is separate from `mlp_kernel.py`
────────────────────────────────────────────────────────────────────────

Empirical finding (Phase D3a investigation, 2026-04-19, branch
`feat/unreal-kernel-phase-d`): **any runtime-Python code addition to the
same module that hosts a `@cute.kernel`-decorated function can perturb
the compiled PTX enough to break FP4 decode numerics.**

The original D3a design put `_TILE_PRESETS`, `_DEFAULT_PRESET_NAME`, and
`_resolve_tile_preset` directly in `mlp_kernel.py` plus refactored
`Phase_D_MLP_Kernel.__init__` to call the resolver. A 4-preset sweep
under that design produced GSM8K 7/8 across all presets — including
`prefill-legacy`, which has bit-for-bit identical tile values (256, 640,
8) to the D2e shipped config. Every preset's Qwen3.5-27B Q2 decode
produced mathematically incoherent tokens (e.g. `'50/5. 5/12. 5'`) vs
the D2e image's correct `'10\\n...'`.

Bisect (evidence under `docs/research/phase_d3a_q2_bisect_*/`):

  - V1 (module-level registry only, D2e `__init__` preserved): +32
    line shift in file, Q2 raw `'50/5 = 10\\n...'` — trajectory drift
    but math correct. True 8/8.
  - V2 (V1 + `__init__` resolver refactor): +40 line shift, Q2 raw
    `'50/5 = 10\\nSo the answer is 1'` — still correct.
  - V2.5 (V2 + **10 pure `#` comment lines** at end of `__init__`):
    +50 line shift, Q2 raw `'10\\nTherefore...'` — direct-answer 8/8.
  - V2.6 (V2 + **one** extra `os.environ.get("CUTE_MLP_TILE")` assignment
    in `__init__`, unused beyond a local variable): +2 line shift, Q2
    raw `'50/5. 12/1. 12/'` — **math broken**, 7/8.
  - V3 (full D3a: V2.6 + extended assert f-strings): Q2 raw
    `'50/5. 5/12. 5'` — math broken.

Pure-comment additions (V1, V2, V2.5) preserved math correctness
regardless of line-shift magnitude. A single extra runtime statement
(V2.6) broke math. The `@cute.kernel` function body was byte-identical
in every variant. Likely cause: CuTe DSL JIT hashes or tokenizes the
whole module source as part of its compile-cache key, and a fresh
recompile produces PTX with ULP-level drift. At FP4 precision, that
drift is enough to flip Qwen3.5-27B's Q2 near-boundary token trajectory
from the correct `10` to incoherent output. Unverified upstream — this
module is the workaround, not the fix.

────────────────────────────────────────────────────────────────────────
Rule: keep runtime Python out of `mlp_kernel.py`
────────────────────────────────────────────────────────────────────────

Any new registry, resolver, config lookup, env read, cache structure,
or helper type that would naturally live at module scope in
`mlp_kernel.py` goes HERE (or another sibling module) instead. Any
change to `Phase_D_MLP_Kernel.__init__` or other non-kernel methods
on that class is also suspect — prefer adding kwargs that the caller
(`_backend.py::_resolve_mlp_weights`) passes explicitly.

Pure-comment edits to `mlp_kernel.py` are empirically safe per V2.5;
docstring and block-comment additions for documentation are fine. But
if you MUST make a runtime change, re-run GSM8K against Qwen3.5-27B
with a fresh image build before shipping.

See `memory:feedback_cute_source_sensitivity` for the condensed rule.
"""

from __future__ import annotations

import os
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Named-tile presets for the Phase D fused MLP kernel.
# ---------------------------------------------------------------------------
# `CUTE_MLP_TILE` env var picks one at kernel construct time;
# unset/empty → `_DEFAULT_PRESET_NAME`. Unknown name → ValueError at
# construct time (intentional: sweep runs must never silently use the
# wrong preset).
#
# Tuple order: (tile_s, tile_k, slice_ctas).
#
# Preset rationale: see
# `docs/superpowers/specs/2026-04-19-phase-d3a-mlp-decode-retune-design.md`.
# ---------------------------------------------------------------------------
_TILE_PRESETS: dict[str, tuple[int, int, int]] = {
    "prefill-legacy":     (256, 640, 8),     # baseline; matches D2e shipped config
    "decode-balanced":    (128, 640, 16),    # half tile_s, 2× CTAs
    "decode-small":       (64,  640, 32),    # quarter tile_s, 4× CTAs
    "decode-narrow-grid": (256, 1280, 8),    # same tile_s, 2× tile_k → halve num_k_tiles
}

_DEFAULT_PRESET_NAME: str = "prefill-legacy"


def _resolve_tile_preset(name: Optional[str]) -> tuple[int, int, int]:
    """Return (tile_s, tile_k, slice_ctas) for the given preset name.

    `None` or empty → the default preset. Unknown name → ValueError with
    the full list of valid preset names in the message.
    """
    key = name if name else _DEFAULT_PRESET_NAME
    if key not in _TILE_PRESETS:
        valid = sorted(_TILE_PRESETS)
        raise ValueError(
            f"Unknown CUTE_MLP_TILE={name!r}; valid: {valid}"
        )
    return _TILE_PRESETS[key]


def resolve_tile_preset_from_env() -> tuple[int, int, int]:
    """Convenience wrapper — reads `CUTE_MLP_TILE` env var once and
    returns the resolved tuple. Logs the chosen preset at INFO so the
    server startup log shows which preset was picked.
    """
    name = os.environ.get("CUTE_MLP_TILE")
    tile_s, tile_k, slice_ctas = _resolve_tile_preset(name)
    resolved_name = name if name else _DEFAULT_PRESET_NAME
    logger.info(
        "Phase D MLP tile preset: %s → tile_s=%d tile_k=%d slice_ctas=%d",
        resolved_name, tile_s, tile_k, slice_ctas,
    )
    return tile_s, tile_k, slice_ctas
