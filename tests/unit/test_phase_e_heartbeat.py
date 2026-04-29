"""Heartbeat daemon thread fires periodic log lines while cute.compile is hot."""

from __future__ import annotations

import logging
import time


def test_heartbeat_fires_periodic_log_with_increasing_counter(caplog):
    caplog.set_level(logging.INFO)
    from vllm.v1.attention.backends.cute_paged import phase_e_kernel

    # vllm propagates=False on its loggers; attach caplog directly so the
    # heartbeat's logger.info lines are visible.
    target = logging.getLogger(
        "vllm.v1.attention.backends.cute_paged.phase_e_kernel"
    )
    target.addHandler(caplog.handler)
    target.setLevel(logging.INFO)

    try:
        with phase_e_kernel._coop_full_compile_heartbeat(period_s=0.1):
            time.sleep(0.5)
    finally:
        target.removeHandler(caplog.handler)

    msgs = [r.message for r in caplog.records if "β-coop compile" in r.message]
    assert len(msgs) >= 2, f"expected >=2 heartbeat lines, got {len(msgs)}: {msgs}"
    counters = [int(m.split("#")[1].rstrip(")")) for m in msgs]
    assert counters == sorted(counters), f"counters must be monotonic: {counters}"
    assert counters[0] >= 1


def test_heartbeat_thread_dies_on_exit(caplog):
    caplog.set_level(logging.INFO)
    from vllm.v1.attention.backends.cute_paged import phase_e_kernel

    target = logging.getLogger(
        "vllm.v1.attention.backends.cute_paged.phase_e_kernel"
    )
    target.addHandler(caplog.handler)
    target.setLevel(logging.INFO)

    try:
        with phase_e_kernel._coop_full_compile_heartbeat(period_s=0.05):
            time.sleep(0.2)
        msgs_before = [r for r in caplog.records if "β-coop compile" in r.message]
        time.sleep(0.3)
        msgs_after = [r for r in caplog.records if "β-coop compile" in r.message]
    finally:
        target.removeHandler(caplog.handler)

    assert len(msgs_after) == len(msgs_before)
