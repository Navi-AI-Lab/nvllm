"""Host-side reducer for the β-coop per-region timing buffer.

The kernel writes per-CTA u64 ticks (entry + exit) per region into a
(num_ctas, num_regions, 2) int64 tensor. This module:
  1. Computes per-CTA deltas (exit - entry).
  2. Derives the active-CTA mask for each region from
     (slice_ctas, num_k_tiles, num_seqs) — NEVER by slicing the first
     N rows. cta_id = bz*(slice_ctas*num_k_tiles) + by*slice_ctas + bx.
       - region 0 (Phase 0): bx==0 && by==0     → 1 CTA/seq
       - regions 1-3 (Phase 1): bx==0 && by<4   → 4 CTAs/seq
                                  cta_ids {0, slice_ctas, 2*slice_ctas,
                                            3*slice_ctas} per seq
       - regions 4-10: all CTAs in seq
  3. Computes percentiles (mean, median, p99) over only the active CTAs.
  4. Optionally calibrates against the nsys-reported total kernel
     duration to produce median_us + frac_of_kernel.

Tick units depend on the kernel-side timer (decided at runtime by the
Task 2 smoke test): %globaltimer is nanoseconds (cross-SM synced),
%clock64 is per-SM cycles (CTA-local diffs only). Reducer reports raw
ticks plus a tick_source label; converts to μs only when calibrated.

Region 4 (grid_barrier_wait) is reported separately and labelled
barrier_wait — it must not be summed into a "work fraction"
denominator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

REGION_NAMES = [
    "phase0_pre_attn",
    "phase1_attn_pre_wo",
    "phase1_wo_gemv",
    "phase1_wo_post",
    "grid_barrier_wait",
    "phase3_load_x",
    "phase3_partial_reset",
    "phase3_3a_fc1_silu",
    "phase3_3b_quant",
    "phase3_3c_fc2_atomic",
    "phase3_3d_arrival",
    "phase1_pre_wo_wait",      # NEW R11: bx>0 W_O CTAs wait for attn producers
    "phase1_gather_reduce",    # NEW R12: last-CTA gather of total_wo_slots partials
]
PHASE0_REGIONS = {0}                        # single CTA per seq
PHASE1_REGIONS = {1, 2, 3}                  # 4 CTAs per seq (bx==0, by<4)
WAIT_NOT_WORK_REGIONS = {4, 11}             # R4 grid barrier + R11 pre-W_O wait
DYNAMIC_SINGLE_CTA_REGIONS = {12}           # R12 elected gather/reduce
PHASE3_REGIONS = {5, 6, 7, 8, 9, 10}        # all CTAs


def _phase0_cta_ids(slice_ctas: int, num_k_tiles: int, num_seqs: int) -> np.ndarray:
    """Phase 0 fires on bx==0 && by==0 → cta_id_within_seq == 0."""
    return np.array(
        [s * slice_ctas * num_k_tiles for s in range(num_seqs)],
        dtype=np.int64,
    )


def _phase1_cta_ids(slice_ctas: int, num_k_tiles: int, num_seqs: int) -> np.ndarray:
    """Phase 1 fires on bx==0 && by<4 → cta_ids {0, s, 2s, 3s} per seq."""
    out: list[int] = []
    for s in range(num_seqs):
        base = s * slice_ctas * num_k_tiles
        for by in range(min(4, num_k_tiles)):
            out.append(base + by * slice_ctas)  # bx=0
    return np.array(out, dtype=np.int64)


def _phase1_wo_split_cta_ids(
    slice_ctas: int,
    num_k_tiles: int,
    num_seqs: int,
    wo_split: int,
    num_kv_heads: int,
) -> np.ndarray:
    """W_O active CTAs with K-parallel split: bx<wo_split && by<num_kv_heads.
    Each CTA's id = bz * (slice_ctas * num_k_tiles) + by * slice_ctas + bx.
    """
    out: list[int] = []
    for s in range(num_seqs):
        base = s * slice_ctas * num_k_tiles
        for by in range(min(num_kv_heads, num_k_tiles)):
            for bx in range(min(wo_split, slice_ctas)):
                out.append(base + by * slice_ctas + bx)
    return np.array(out, dtype=np.int64)


@dataclass
class RegionRow:
    region_id: int
    region: str
    n_active_ctas: int
    cta_class: str             # "phase0" | "phase1" | "phase3" | "barrier_wait" | "dynamic_single"
    tick_source: str           # "globaltimer" | "clock64"
    mean_ticks: float
    median_ticks: float
    p99_ticks: float
    raw_total_ticks: int
    median_us: float           # NaN unless nsys_total_us provided + globaltimer
    frac_of_kernel: float      # NaN unless nsys_total_us provided


def reduce_region_timings(
    buf,                                    # np.ndarray or torch tensor
    *,
    slice_ctas: int,
    num_k_tiles: int,
    num_seqs: int,
    tick_source: str,                       # "globaltimer" | "clock64"
    nsys_total_us: Optional[float] = None,
    wo_split: int = 1,
    num_kv_heads: int = 0,
) -> pd.DataFrame:
    """Reduce a (num_ctas, 13, 2) tick buffer to per-region rows.

    Active-CTA masks are derived from (slice_ctas, num_k_tiles, num_seqs)
    so callers do NOT pass a "num_attn_active_ctas" count — that count
    is wrong for Phase 0 (1 CTA/seq) vs Phase 1 (4 CTAs/seq) which the
    earlier draft conflated as "32".

    When wo_split > 1, regions {2, 3, 11, 12} are masked using the
    K-parallel W_O active-CTA layout (bx<wo_split && by<num_kv_heads)
    via _phase1_wo_split_cta_ids. wo_split=1 falls through to the
    legacy _phase1_cta_ids mask for full backward compatibility.

    median_us is reported ONLY when nsys_total_us is provided AND
    tick_source is globaltimer. With clock64, dynamic-clock effects make
    cycle→μs conversion unreliable, so the column stays NaN — caller
    interprets ticks-as-cycles diagnostically and uses frac_of_kernel
    (which is computed via tick-to-tick ratio against a synthetic total
    or, calibrated, against nsys ticks-equivalent).
    """
    assert tick_source in ("globaltimer", "clock64"), tick_source
    if hasattr(buf, "cpu"):
        arr = buf.cpu().numpy()
    else:
        arr = np.asarray(buf)
    assert arr.dtype == np.int64
    num_ctas, num_regions, two = arr.shape
    assert two == 2
    assert num_regions == len(REGION_NAMES)
    assert num_ctas == slice_ctas * num_k_tiles * num_seqs, (
        f"num_ctas {num_ctas} != slice_ctas*num_k_tiles*num_seqs "
        f"{slice_ctas*num_k_tiles*num_seqs}"
    )

    deltas = arr[:, :, 1] - arr[:, :, 0]   # (num_ctas, num_regions) int64

    p0_ids = _phase0_cta_ids(slice_ctas, num_k_tiles, num_seqs)
    p1_ids = _phase1_cta_ids(slice_ctas, num_k_tiles, num_seqs)
    all_ids = np.arange(num_ctas, dtype=np.int64)
    # K-parallel W_O mask: only valid when wo_split > 1 AND caller
    # supplied num_kv_heads. Used for R2/R3/R11/R12.
    if wo_split > 1:
        assert num_kv_heads > 0, (
            "wo_split>1 requires num_kv_heads>0 for the K-parallel mask"
        )
        wo_split_ids = _phase1_wo_split_cta_ids(
            slice_ctas, num_k_tiles, num_seqs, wo_split, num_kv_heads,
        )
    else:
        wo_split_ids = None

    rows: list[RegionRow] = []
    for r in range(num_regions):
        col = deltas[:, r]
        if r in PHASE0_REGIONS:
            active_ids = p0_ids
            cta_class = "phase0"
        elif r in PHASE1_REGIONS:
            # When wo_split>1, R2/R3 are the W_O GEMV/post regions and
            # use the K-parallel mask. R1 (phase1_attn_pre_wo) is still
            # the bx==0 && by<4 set so it stays on p1_ids.
            if wo_split_ids is not None and r in (2, 3):
                active_ids = wo_split_ids
            else:
                active_ids = p1_ids
            cta_class = "phase1"
        elif r in WAIT_NOT_WORK_REGIONS:
            # R11 (phase1_pre_wo_wait) uses the K-parallel mask when
            # wo_split>1 — it's the consumer wait for bx>0 W_O CTAs.
            # R4 (grid_barrier_wait) stays on all_ids.
            if wo_split_ids is not None and r == 11:
                active_ids = wo_split_ids
            else:
                active_ids = all_ids
            cta_class = "barrier_wait"
        elif r in DYNAMIC_SINGLE_CTA_REGIONS:
            # R12 (phase1_gather_reduce) is the elected single-CTA
            # gather. Even with wo_split>1 only one CTA writes a tick;
            # nonzero filter handles it. Mask is all_ids.
            active_ids = all_ids
            cta_class = "dynamic_single"
        else:
            active_ids = all_ids
            cta_class = "phase3"

        sliced = col[active_ids]
        # Defensive: drop zero-delta rows (CTA didn't reach the region —
        # shouldn't happen if the active-CTA mask is correct, but treat
        # as inactive rather than counting zero into the percentile).
        nonzero = sliced[sliced > 0]
        n = int(nonzero.size)
        if n == 0:
            mean = median = p99 = 0.0
            total = 0
        else:
            mean = float(np.mean(nonzero))
            median = float(np.median(nonzero))
            p99 = float(np.percentile(nonzero, 99))
            total = int(nonzero.sum())

        # μs conversion only when units are nanoseconds AND calibrated.
        if tick_source == "globaltimer" and nsys_total_us is not None:
            median_us = median / 1000.0
        else:
            median_us = float("nan")

        # frac_of_kernel: per-CTA median is the representative single-CTA
        # wall-time contribution (concurrent execution within a region).
        # Denominator: nsys_total_us converted to ticks of the same
        # source. For globaltimer that's *1000 (ns/μs); for clock64 we
        # cannot convert, so frac is reported as NaN unless caller
        # passes a clock64-calibrated total (not in the v1 reducer API).
        if r in WAIT_NOT_WORK_REGIONS or r in DYNAMIC_SINGLE_CTA_REGIONS:
            frac = float("nan")
        elif nsys_total_us is None:
            frac = float("nan")
        elif tick_source == "globaltimer":
            denom_ns = float(nsys_total_us) * 1000.0
            frac = median / denom_ns if denom_ns > 0 else float("nan")
        else:
            frac = float("nan")  # clock64: no calibrated total available

        rows.append(RegionRow(
            region_id=r,
            region=REGION_NAMES[r],
            n_active_ctas=n,
            cta_class=cta_class,
            tick_source=tick_source,
            mean_ticks=mean,
            median_ticks=median,
            p99_ticks=p99,
            raw_total_ticks=total,
            median_us=median_us,
            frac_of_kernel=frac,
        ))
    return pd.DataFrame([row.__dict__ for row in rows])
