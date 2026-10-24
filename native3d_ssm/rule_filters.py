"""BAF and STCF, the two classical rule-based denoisers, as score functions.

WHY SCORES AND NOT DECISIONS. Both filters are defined as binary rules with a correlation
time `dt`: BAF passes an event if any of its eight neighbours fired within `dt`, and STCF
passes it if at least `k` of them did. An AUC therefore requires sweeping something, and
\\cite{guo2025ebf} sweeps `dt` from 1 to 50 ms to trace their ROC curves.

Sweeping a threshold over a scalar is the same operation as ranking by that scalar. For BAF
the scalar is the age of the most recent neighbour event: thresholding it at `dt` reproduces
BAF's decision exactly, for every `dt`, so the ROC of that score IS the `dt` sweep, at
infinite resolution instead of at whatever grid the sweep used. For STCF the same argument
applies per `k`: "at least `k` neighbours within `dt`" is exactly "the `k`-th smallest
neighbour age is below `dt`". So STCF is a family of eight scores, one per `k`, and sweeping
`k` as well as `dt` -- which is what a parameter sweep does -- means taking the best of them.

This construction can only help the baselines relative to a coarse grid, which is the point:
a re-implemented baseline that loses because it was run at the wrong constant is worthless as
evidence.

k = 1 IS BAF. The two kernels are written separately anyway, from the two source
descriptions, and `check_equivalence` requires them to agree; a shared misreading of the
neighbourhood or the update order would have to be identical in both to survive that.

STATE AND UPDATE ORDER. Both keep one last-timestamp map per pixel, polarity-agnostic
(STCF's polarity variant is PFD, which is a different row in the source and is not this).
The map is read before the current event is written into it, so both are strictly causal.
Pixels that have never fired carry no age and never count as correlated.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numba import njit

#: jaer's default search distance for both filters: a 3x3 window, 8 neighbours.
NEIGHBOURHOOD_RADIUS = 1
#: Age returned for a neighbour that has never fired. Larger than any real stream.
NEVER = np.int64(1) << np.int64(62)


@njit(cache=True)
def _baf_kernel(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                width: int, height: int, radius: int) -> np.ndarray:
    """Age of the most recent neighbour event, per event. Lower means more signal-like."""

    last = np.full(width * height, np.int64(-1), dtype=np.int64)
    out = np.full(t.shape[0], NEVER, dtype=np.int64)
    for n in range(t.shape[0]):
        xe, ye, te = x[n], y[n], t[n]
        x0 = max(0, xe - radius)
        x1 = min(width - 1, xe + radius)
        y0 = max(0, ye - radius)
        y1 = min(height - 1, ye + radius)
        newest = np.int64(-1)
        for j in range(y0, y1 + 1):
            base = j * width
            for i in range(x0, x1 + 1):
                if i == xe and j == ye:
                    continue
                v = last[base + i]
                if v > newest:
                    newest = v
        if newest >= 0:
            out[n] = te - newest
        last[ye * width + xe] = te
    return out


@njit(cache=True)
def _stcf_kernel(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                 width: int, height: int, radius: int, n_taps: int) -> np.ndarray:
    """Sorted neighbour ages, per event: column k-1 is the k-th smallest age."""

    last = np.full(width * height, np.int64(-1), dtype=np.int64)
    out = np.full((t.shape[0], n_taps), NEVER, dtype=np.int64)
    ages = np.empty(n_taps, dtype=np.int64)
    for n in range(t.shape[0]):
        xe, ye, te = x[n], y[n], t[n]
        x0 = max(0, xe - radius)
        x1 = min(width - 1, xe + radius)
        y0 = max(0, ye - radius)
        y1 = min(height - 1, ye + radius)
        count = 0
        for j in range(y0, y1 + 1):
            base = j * width
            for i in range(x0, x1 + 1):
                if i == xe and j == ye:
                    continue
                v = last[base + i]
                if v >= 0:
                    ages[count] = te - v
                    count += 1
        # Insertion sort over at most eight entries; the array is tiny and this keeps the
        # kernel free of any allocation inside the event loop.
        for a in range(1, count):
            key = ages[a]
            b = a - 1
            while b >= 0 and ages[b] > key:
                ages[b + 1] = ages[b]
                b -= 1
            ages[b + 1] = key
        for a in range(count):
            out[n, a] = ages[a]
        last[ye * width + xe] = te
    return out


def _columns(events: np.ndarray, width: int, height: int
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = np.ascontiguousarray(events, dtype=np.int64)
    if events.ndim != 2 or events.shape[1] < 3:
        raise ValueError(f"Expected [t,x,y,...] columns, got shape {events.shape}")
    t, x, y = events[:, 0], events[:, 1], events[:, 2]
    if np.any(x < 0) or np.any(x >= width) or np.any(y < 0) or np.any(y >= height):
        raise ValueError(f"Event coordinates fall outside {width}x{height}")
    if np.any(np.diff(t) < 0):
        raise ValueError("Timestamps must be nondecreasing; both filters are causal")
    return t, x, y


def baf_noise_score(events: np.ndarray, width: int, height: int,
                    radius: int = NEIGHBOURHOOD_RADIUS) -> np.ndarray:
    """BAF in this paper's orientation: larger means more noise-like.

    The raw quantity is already oriented that way -- a large age since the last neighbour
    event is what BAF filters out -- so no negation is needed, unlike EBF.
    """

    t, x, y = _columns(events, width, height)
    return _baf_kernel(t, x, y, int(width), int(height), radius).astype(np.float64)


def stcf_noise_scores(events: np.ndarray, width: int, height: int,
                      radius: int = NEIGHBOURHOOD_RADIUS) -> Dict[int, np.ndarray]:
    """STCF's score family, keyed by `k`. Larger means more noise-like, as for BAF.

    `scores[k]` is the age of the `k`-th most recent neighbour event; thresholding it at
    `dt` is exactly STCF's "at least `k` neighbours correlated within `dt`".
    """

    t, x, y = _columns(events, width, height)
    n_taps = (2 * radius + 1) ** 2 - 1
    sorted_ages = _stcf_kernel(t, x, y, int(width), int(height), radius, n_taps)
    return {k: sorted_ages[:, k - 1].astype(np.float64) for k in range(1, n_taps + 1)}


def check_equivalence(width: int = 48, height: int = 40, n_events: int = 5000,
                      seed: int = 0) -> float:
    """BAF must equal STCF at k = 1. Two kernels, two source descriptions, one answer."""

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 300_000, size=n_events)).astype(np.int64)
    x = rng.integers(0, width, size=n_events).astype(np.int64)
    y = rng.integers(0, height, size=n_events).astype(np.int64)
    events = np.column_stack((t, x, y, rng.integers(0, 2, size=n_events)))

    baf = baf_noise_score(events, width, height)
    stcf1 = stcf_noise_scores(events, width, height)[1]
    worst = float(np.max(np.abs(baf - stcf1)))
    if worst > 0.0:
        raise AssertionError(
            f"BAF disagrees with STCF at k=1 on "
            f"{int(np.count_nonzero(baf != stcf1))} of {n_events} events (worst {worst})")
    return worst


def check_properties(width: int = 48, height: int = 40, n_events: int = 5000,
                     seed: int = 1) -> None:
    """Properties taken from the filter definitions, not from either implementation."""

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 300_000, size=n_events)).astype(np.int64)
    x = rng.integers(0, width, size=n_events).astype(np.int64)
    y = rng.integers(0, height, size=n_events).astype(np.int64)
    events = np.column_stack((t, x, y, rng.integers(0, 2, size=n_events)))
    scores = stcf_noise_scores(events, width, height)

    # 1. Requiring more correlated neighbours can only make an event look worse, so the
    #    score family is nondecreasing in k. This is what makes each k a valid threshold.
    for k in range(2, 9):
        if np.any(scores[k] < scores[k - 1]):
            raise AssertionError(f"STCF score at k={k} is below k={k-1} on "
                                 f"{int(np.sum(scores[k] < scores[k-1]))} events")

    # 2. Ages are never negative: the map is read before it is written, so a neighbour can
    #    never carry a timestamp later than the event being scored.
    for k, v in scores.items():
        if np.any(v < 0):
            raise AssertionError(f"STCF k={k} produced a negative age")

    # 3. An event at a pixel whose neighbourhood has never fired must score NEVER at k=1.
    #    We construct one: a lone event in an otherwise empty corner.
    lone = np.array([[0, width - 1, height - 1, 1]], dtype=np.int64)
    if baf_noise_score(lone, width, height)[0] != float(NEVER):
        raise AssertionError("An event with no prior neighbour was assigned a finite age")

    # 4. Two events at the same pixel do not correlate with each other: the centre is
    #    excluded, so a pixel firing twice in isolation still scores NEVER.
    repeat = np.array([[0, 5, 5, 1], [10, 5, 5, 1]], dtype=np.int64)
    if baf_noise_score(repeat, width, height)[1] != float(NEVER):
        raise AssertionError("The centre pixel is correlating with itself")


if __name__ == "__main__":
    check_properties()
    worst = check_equivalence()
    print(f"BAF == STCF at k=1 exactly (worst |diff| = {worst}); "
          f"four definitional properties hold")
