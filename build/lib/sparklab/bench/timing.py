"""Timing helpers. Distributions, not means: a mean hides the slow tail that a
multi-turn agent is nearly guaranteed to hit (a 10-turn agent sees the p99 with
probability ~1 - 0.99^10 ~= 10%), so every summary here carries percentiles."""

import statistics
import time


def percentiles(xs, points=(50, 90, 99)):
    """Summarize a list of measurements (any unit) as percentiles plus mean/stdev."""
    if not xs:
        raise ValueError("percentiles() needs at least one measurement")
    s = sorted(xs)
    # value at position p% of the sorted list; index clamped so p100 stays in range
    q = lambda p: s[min(int(p / 100 * len(s)), len(s) - 1)]
    out = {f"p{p}": q(p) for p in points}
    out.update(
        mean=statistics.fmean(s),
        stdev=statistics.stdev(s) if len(s) > 1 else 0.0,
        min=s[0],
        max=s[-1],
        n=len(s),
    )
    return out


def bench(fn, warmup=5, iters=50):
    """Time fn() honestly: untimed warmup runs first so one-time startup costs
    don't pollute the numbers, then many timed repetitions, reported in ms."""
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    return percentiles(xs)
