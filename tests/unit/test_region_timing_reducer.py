"""Unit tests for the per-region timing reducer with synthetic buffers.

Per friend review: the reducer must derive active-CTA masks from
(slice_ctas, num_k_tiles, num_seqs) — NOT slice the first N rows. With
cta_id = by*slice_ctas + bx, Phase 0 active = {0}, Phase 1 active =
{0, slice_ctas, 2*slice_ctas, 3*slice_ctas} for slice_ctas=8 and
num_k_tiles>=4.
"""
from __future__ import annotations

import numpy as np
import pytest


def _phase0_active(slice_ctas, num_k_tiles, num_seqs):
    return [s * slice_ctas * num_k_tiles for s in range(num_seqs)]


def _phase1_active(slice_ctas, num_k_tiles, num_seqs):
    out = []
    for s in range(num_seqs):
        base = s * slice_ctas * num_k_tiles
        for by in range(min(4, num_k_tiles)):
            out.append(base + by * slice_ctas)  # bx=0
    return out


def _make_synthetic_buffer(slice_ctas=8, num_k_tiles=8, num_seqs=1):
    """Synthesize a (num_ctas, 11, 2) buffer with known deltas.

    Uses the CORRECT active masks (single CTA for Phase 0, four CTAs for
    Phase 1, all 64 for Phase 3). Inactive (cta, region) cells leave both
    ticks zero so the reducer's `delta>0` filter exercises the real
    inactive case.
    """
    num_ctas = slice_ctas * num_k_tiles * num_seqs
    base_delta = np.array(
        [100, 200, 5000, 50,    # P0, P1-pre-WO, P1-WO, P1-cleanup
         800,                    # barrier wait
         150, 100, 18000, 600, 12000, 200],  # P3 stages 5-10
        dtype=np.int64,
    )
    cta_jitter = np.linspace(0, 1000, num_ctas).astype(np.int64)
    buf = np.zeros((num_ctas, 11, 2), dtype=np.int64)
    base_t = 1_000_000_000  # arbitrary epoch
    p0 = set(_phase0_active(slice_ctas, num_k_tiles, num_seqs))
    p1 = set(_phase1_active(slice_ctas, num_k_tiles, num_seqs))
    for c in range(num_ctas):
        for r in range(11):
            if r == 0 and c not in p0:
                continue
            if r in (1, 2, 3) and c not in p1:
                continue
            # regions 4-10: all CTAs active
            buf[c, r, 0] = base_t + r * 100_000
            buf[c, r, 1] = buf[c, r, 0] + base_delta[r] + cta_jitter[c]
    return buf, base_delta, num_ctas


def test_reducer_extracts_correct_deltas():
    from vllm.v1.attention.backends.cute_paged.region_timing import (
        reduce_region_timings,
    )
    buf, base_delta, _ = _make_synthetic_buffer()
    df = reduce_region_timings(
        buf, slice_ctas=8, num_k_tiles=8, num_seqs=1,
        tick_source="globaltimer",
    )

    p3a = df[df.region == "phase3_3a_fc1_silu"]
    assert len(p3a) == 1
    # Region 7 fires on all 64 CTAs; median across cta_jitter ~500
    assert abs(p3a.iloc[0].median_ticks - (base_delta[7] + 500)) < 50

    # Region 0 (Phase 0) is single-CTA: n_active=1 (NOT 32)
    p0 = df[df.region == "phase0_pre_attn"].iloc[0]
    assert p0.n_active_ctas == 1, (
        f"Phase 0 fires on cta_id 0 only (bx==0 && by==0); "
        f"got n_active_ctas={p0.n_active_ctas}"
    )

    # Region 1/2/3 (Phase 1) are 4-CTA: n_active=4 (NOT 32)
    for name in ("phase1_attn_pre_wo", "phase1_wo_gemv", "phase1_wo_post"):
        row = df[df.region == name].iloc[0]
        assert row.n_active_ctas == 4, (
            f"{name} fires on cta_ids {{0, 8, 16, 24}}; "
            f"got n_active_ctas={row.n_active_ctas}"
        )

    # Region 7 (Phase 3 stage 3a) reports n_active=64
    assert p3a.iloc[0].n_active_ctas == 64


def test_reducer_filters_inactive_ctas_in_attn_regions():
    """Reducer must NOT include CTAs that didn't run a region."""
    from vllm.v1.attention.backends.cute_paged.region_timing import (
        reduce_region_timings,
    )
    buf, _, num_ctas = _make_synthetic_buffer()
    df = reduce_region_timings(
        buf, slice_ctas=8, num_k_tiles=8, num_seqs=1,
        tick_source="globaltimer",
    )
    # Phase 1 W_O: only 4 CTAs ran it, so percentiles are over those 4
    wo = df[df.region == "phase1_wo_gemv"].iloc[0]
    assert wo.n_active_ctas == 4
    # The other num_ctas-4 CTAs had delta=0 and must be excluded
    assert wo.median_ticks > 0


def test_reducer_emits_calibrated_us_only_after_calibration():
    from vllm.v1.attention.backends.cute_paged.region_timing import (
        reduce_region_timings,
    )
    buf, base_delta, _ = _make_synthetic_buffer()
    # Without calibration: median_us is NaN
    df_uncal = reduce_region_timings(
        buf, slice_ctas=8, num_k_tiles=8, num_seqs=1,
        tick_source="globaltimer",
    )
    assert df_uncal["median_us"].isna().all(), (
        "Without nsys_total_us, median_us must be NaN — units of ticks "
        "are unknown until calibrated."
    )
    # With calibration: median_us populated, frac_of_kernel populated
    nsys_total_us = float(base_delta.sum() / 1000.0)  # synthetic ns → μs
    df = reduce_region_timings(
        buf, slice_ctas=8, num_k_tiles=8, num_seqs=1,
        tick_source="globaltimer",
        nsys_total_us=nsys_total_us,
    )
    # Excluding wait region, fractions should sum to ~1.0
    work = df[df.cta_class != "barrier_wait"]
    assert abs(work.frac_of_kernel.sum() - 1.0) < 0.10


def test_reducer_records_tick_source():
    """tick_source must propagate so consumers can label units."""
    from vllm.v1.attention.backends.cute_paged.region_timing import (
        reduce_region_timings,
    )
    buf, _, _ = _make_synthetic_buffer()
    df = reduce_region_timings(
        buf, slice_ctas=8, num_k_tiles=8, num_seqs=1,
        tick_source="clock64",
    )
    assert (df.tick_source == "clock64").all()
