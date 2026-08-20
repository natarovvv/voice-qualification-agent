"""Counters for the two questions an operator asks: is it up, and is it fast.

Prometheus text format, hand-rolled. prometheus_client would be a dependency
for four counters and one histogram, and emitting the format is a dozen lines
of printf - parsing it is the hard half, and that is the scraper's problem.

A histogram rather than a running p95, because percentiles do not average: an
in-process p95 stops meaning anything the moment there is a second worker.
Bucket counts add across workers and the scraper takes the quantile from them.

Everything here is per-process and dies with it. That is the normal shape for
a metrics endpoint - the scraper keeps the history, the process keeps a tally.
"""
from __future__ import annotations

import bisect

# Time to first audio, in seconds. The boundaries bracket the 1200 ms budget
# and crowd where the real numbers live (235-550 ms mid-call), so the share of
# turns inside budget reads straight off one bucket: le="1.2" over _count.
BUCKETS = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0)

LATENCY = "voice_first_audio_seconds"
HELP = {
    "voice_calls_total": "Calls accepted since this worker started.",
    "voice_calls_rejected_total": "Calls turned away before they began.",
    "voice_turns_total": "Turns the agent replied to.",
    "voice_turn_errors_total": "Turns that failed and got the apology line.",
    "voice_tool_calls_total": "Tool calls the agent made.",
    LATENCY: "Caller stopped talking -> first byte of audio out.",
}

# Seeded at zero so a quiet worker still exports the series. A rate() over a
# metric that only appears after its first event has nothing to rate at the
# moment you most want to look: right after a deploy, before any traffic.
_SEEDED = ("voice_calls_total", "voice_turns_total", "voice_turn_errors_total")

_counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
_hist: list[int] = []  # one slot per boundary, plus a last one for +Inf
_sum = 0.0


def reset() -> None:
    """Back to a clean slate. Runs at import; the tests call it between cases."""
    global _sum
    _counters.clear()
    _counters.update({(name, ()): 0 for name in _SEEDED})
    _hist[:] = [0] * (len(BUCKETS) + 1)
    _sum = 0.0


def count(name: str, **labels: str) -> None:
    """Add one to a counter.

    Labels take a small fixed set of values by design, and every value passed
    here is a literal from our own code. A label fed from the outside - a
    session id, a caller's email, a provider's error string - is how a metrics
    endpoint turns into a memory leak with a scrape that never finishes.
    """
    key = (name, tuple(sorted(labels.items())))
    _counters[key] = _counters.get(key, 0) + 1


def observe_first_audio(seconds: float) -> None:
    global _sum
    _sum += seconds
    # bisect_left, not right: a turn landing exactly on a boundary belongs in
    # that bucket, because the boundary is le - less than or *equal*.
    _hist[bisect.bisect_left(BUCKETS, seconds)] += 1


def _labels(pairs: tuple[tuple[str, str], ...]) -> str:
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in pairs) + "}"


def render(active_calls: int) -> str:
    """The whole exposition, as one scrape."""
    out = [
        "# HELP voice_calls_active Calls on this worker right now.",
        "# TYPE voice_calls_active gauge",
        f"voice_calls_active {active_calls}",
    ]
    for name in sorted({n for n, _ in _counters}):
        # .get, not [name]: a scrape is the thing you least want falling over
        # during an incident, and a missing HELP line is not worth a 500.
        out += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} counter"]
        for (found, labels), value in sorted(_counters.items()):
            if found == name:
                out.append(f"{name}{_labels(labels)} {value}")

    out += [f"# HELP {LATENCY} {HELP[LATENCY]}", f"# TYPE {LATENCY} histogram"]
    total = 0
    for edge, in_bucket in zip(BUCKETS, _hist):
        total += in_bucket  # buckets are cumulative on the wire, not here
        out.append(f'{LATENCY}_bucket{{le="{edge}"}} {total}')
    total += _hist[-1]
    out += [
        f'{LATENCY}_bucket{{le="+Inf"}} {total}',
        f"{LATENCY}_sum {_sum}",
        f"{LATENCY}_count {total}",
    ]
    return "\n".join(out) + "\n"


reset()
