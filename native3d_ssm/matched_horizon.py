"""Explicit raw, single-log and exact-flat operands for matched-horizon controls.

The generic feature extractor exports already logged ring amplitudes. These
kernels instead expose raw directional ring sums, preventing a second logarithm
from silently changing the tested formula. Finite-decay states retain the
float32 update and per-scale capped-age recurrence of ``core.NativeStateBank``.
Flat controls use integer event counts, not a large finite decay constant.
"""

from __future__ import annotations

import numpy as np

from .core import njit


def _events(events: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    events = np.ascontiguousarray(events, dtype=np.int64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape [N,4] ordered t,x,y,p")
    if width < 1 or height < 1 or radius < 1:
        raise ValueError("positive geometry and radius required")
    if events.size:
        if np.any(np.diff(events[:, 0]) < 0):
            raise ValueError("timestamps must be nondecreasing")
        if np.any((events[:, 1] < 0) | (events[:, 1] >= width)):
            raise ValueError("x outside sensor")
        if np.any((events[:, 2] < 0) | (events[:, 2] >= height)):
            raise ValueError("y outside sensor")
        if np.any((events[:, 3] != 0) & (events[:, 3] != 1)):
            raise ValueError("polarity must be 0 or 1")
    return events


@njit(cache=True)
def _exact_block_statistics(events, width, height, radius):
    counts = np.zeros((2, height, width), dtype=np.uint32)
    past_map = np.zeros((2, height, width), dtype=np.uint32)
    n = len(events)
    occupancy = np.zeros(n, dtype=np.uint8)
    multiplicity = np.zeros(n, dtype=np.uint32)
    past = np.zeros(n, dtype=np.uint32)
    future = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        x, y, p = events[i, 1], events[i, 2], events[i, 3]
        counts[p, y, x] += 1
    for i in range(n):
        x, y, p = events[i, 1], events[i, 2], events[i, 3]
        occupied, total, previous = 0, 0, 0
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 0 or yy >= height:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if (dx == 0 and dy == 0) or xx < 0 or xx >= width:
                    continue
                value = counts[p, yy, xx]
                occupied += value > 0
                total += value
                previous += past_map[p, yy, xx]
        occupancy[i] = occupied
        multiplicity[i] = total
        past[i] = previous
        future[i] = total - previous
        past_map[p, y, x] += 1
    return occupancy, multiplicity, past, future


def exact_block_statistics(events: np.ndarray, width: int, height: int,
                           radius: int = 2):
    """Return occupancy, multiplicity, past count, future count for each event.

    Counts include only same-polarity, in-sensor, centre-excluded ring pixels.
    Equal timestamps follow arrival order. The full-block outputs themselves
    are independent of that order. The uint32 counters require N < 2**32;
    occupancy is uint8 and requires a ring of at most 255 taps.
    """
    events = _events(events, width, height, radius)
    if len(events) >= 2**32 or (2 * radius + 1)**2 - 1 > 255:
        raise ValueError("counter representation exceeded")
    return _exact_block_statistics(events, width, height, radius)


@njit(cache=True)
def _raw_directional_amplitudes(events, width, height, radius, taus,
                                max_age_factor, reverse):
    states = np.zeros((2, height, width, len(taus)), dtype=np.float32)
    last_times = np.full((2, height, width), -1, dtype=np.int64)
    out = np.zeros((len(events), len(taus)), dtype=np.float64)
    end_time = events[-1, 0] if len(events) else 0
    for step in range(len(events)):
        i = len(events) - 1 - step if reverse else step
        timestamp = end_time - events[i, 0] if reverse else events[i, 0]
        x, y, p = events[i, 1], events[i, 2], events[i, 3]
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 0 or yy >= height:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if (dx == 0 and dy == 0) or xx < 0 or xx >= width:
                    continue
                last = last_times[p, yy, xx]
                if last < 0:
                    continue
                for k in range(len(taus)):
                    age = min(float(timestamp - last), max_age_factor * taus[k])
                    out[i, k] += states[p, yy, xx, k] * np.exp(-age / taus[k])
        previous = last_times[p, y, x]
        for k in range(len(taus)):
            value = 0.0
            if previous >= 0:
                age = min(float(timestamp - previous), max_age_factor * taus[k])
                value = states[p, y, x, k] * np.exp(-age / taus[k])
            states[p, y, x, k] = value + 1.0
        last_times[p, y, x] = timestamp
    return out


def raw_directional_amplitudes(events: np.ndarray, width: int, height: int,
                               taus_us=(250.0, 1000.0, 4000.0, 16000.0),
                               radius: int = 2, max_age_factor: float = 20.0):
    """Return explicit past/future raw ring amplitudes, shape [N,len(taus)].

    Each scale clamps at its own 20*tau, matching separate single-scale banks.
    Logarithms are intentionally absent; callers choose their reduction.
    """
    events = _events(events, width, height, radius)
    taus = np.asarray(taus_us, dtype=np.float64)
    if taus.ndim != 1 or not len(taus) or np.any(~np.isfinite(taus)) or np.any(taus <= 0):
        raise ValueError("finite positive taus required")
    if not np.isfinite(max_age_factor) or max_age_factor <= 0:
        raise ValueError("finite positive age cap required")
    forward = _raw_directional_amplitudes(events, width, height, radius, taus,
                                        max_age_factor, False)
    backward = _raw_directional_amplitudes(events, width, height, radius, taus,
                                         max_age_factor, True)
    return forward, backward


FINITE_TAUS_US = (250.0, 1000.0, 4000.0, 16000.0)
FINITE_NAMES = ("250us", "1ms", "4ms", "16ms")
SCORE_DEFINITIONS = {
    "count": "negative whole-block occupied same-polarity ring taps",
    "sum_flat": "negative exact whole-block integer same-polarity ring event count",
    "logdir_flat": "-log1p(exact past ring event count)-log1p(exact future ring event count)",
    "sum_finite": "-(raw past decayed ring amplitude+raw future decayed ring amplitude)",
    "logdir_finite": "-log1p(raw past decayed ring amplitude)-log1p(raw future decayed ring amplitude)",
    "nearflat": "finite tau=1e9 us diagnostic; not the exact matched-horizon control",
    "logproduct_flat": "negative exact uint64 (1+past_count)*(1+future_count), preserving ideal split-log ties; converted to float64 only after confirming product < 2**53",
    "numerics": "float32 finite-decay state updates; float64 raw ring sums; canonical logdir uses float32 log export and addition as phase8.operands; logdir64 isolates float64 reduction; integer flat counts",
}


def single_log_score(forward, backward):
    """One logarithm per direction, with the established feature precision."""
    return -(np.log1p(forward).astype(np.float32) +
             np.log1p(backward).astype(np.float32)).astype(np.float64)


def block_scores(events, width, height, radius=2):
    """All corrected controls plus near-flat diagnostics on one complete block."""
    occupancy, multiplicity, past, future = exact_block_statistics(events, width, height, radius)
    product = (1 + past.astype(np.uint64)) * (1 + future.astype(np.uint64))
    if len(product) and product.max() >= 2**53:
        raise ValueError("flat product diagnostic exceeds exact float64 integer range")
    out = {"count": -occupancy.astype(np.float64),
           "sum_flat": -multiplicity.astype(np.float64),
           "logproduct_flat": -product.astype(np.float64),
           "logdir_flat": single_log_score(past.astype(np.float64), future.astype(np.float64)),
           "logdir64_flat": -(np.log1p(past.astype(np.float64)) +
                              np.log1p(future.astype(np.float64)))}
    forward, backward = raw_directional_amplitudes(
        events, width, height, FINITE_TAUS_US + (1e9,), radius)
    for k, name in enumerate(FINITE_NAMES + ("nearflat",)):
        out["sum_" + name] = -(forward[:, k] + backward[:, k])
        out["logdir_" + name] = single_log_score(forward[:, k], backward[:, k])
        out["logdir64_" + name] = -(np.log1p(forward[:, k]) + np.log1p(backward[:, k]))
    return out, occupancy
