"""The strictly causal, streaming form of the ring-occupancy operand.

WHY THIS FILE EXISTS. The paper's main comparison is confounded in two ways that no amount
of writing can remove. First, our score reads a whole 10,240-event block, forward and
backward, while EBF reads only the past; the asymmetry favours us. Second, our support is
an unweighted count over the block's whole extent while EBF's magnitude decays over a
32 ms window, so the two operands differ in amplitude representation *and* in temporal
horizon at the same time. This module removes both confounds by construction.

THE CONSTRUCTION. Take EBF's own data structure -- a Timestamp and Polarity Map holding the
last timestamp and last polarity at every pixel -- its own 5x5 neighbourhood, its own
polarity rule, and its own strictly causal update order. Change exactly one thing, the
accumulator:

    EBF          acc += 1 - dt/tau        for 0 <= dt < tau       (decayed magnitude)
    ours         acc += 1                 for 0 <= dt < W         (occupancy count)

Everything else is byte-for-byte the same traversal. With W = tau the two scores read the
same taps of the same map at the same instant and differ only in what they accumulate from
them, so a difference between them is the operand and nothing else. That is the matched
experiment the block-local comparison cannot give.

WHAT IT COSTS AND WHAT IT SAVES. The trailing-window form keeps EBF's map, so it does not
inherit the block form's two-bit-plane state; the memory is EBF's, exactly. What it removes
is the block buffer -- there is none, the filter is a single pass with O(1) work per event
-- and the arithmetic: one integer compare and an increment where EBF needs a subtraction,
a division and a floating-point add. The output is an integer in [0, 24], so the five-bit
result of Proposition 1 carries over unchanged and without a block reset to depend on.

VERIFICATION. `check_equivalence` requires the compiled kernel to agree with a literal
reference implementation, and `check_properties` requires four things the reference cannot
vouch for: the score is integer-valued, it lies in [0, 24], it counts exactly the taps EBF
gives nonzero weight when W = tau, and it never reads a future event.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit
except ImportError:                                              # pragma: no cover
    def njit(*args, **kwargs):
        def wrap(func):
            return func
        return wrap(args[0]) if args and callable(args[0]) else wrap

#: EBF's neighbourhood, borrowed so the traversal is matched. Sec. III of Guo et al.
OCCUPANCY_S = 5
#: The largest value the score can take: the 5x5 ring minus its centre.
OCCUPANCY_MAX = OCCUPANCY_S * OCCUPANCY_S - 1
#: Bits needed to hold it exactly.
OCCUPANCY_BITS = 5


@njit(cache=True, fastmath=False)
def _occupancy_kernel(t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray,
                      width: int, height: int, window_us: np.int64,
                      radius: int) -> np.ndarray:
    """Causal trailing-window ring occupancy per event. Higher is more signal-like."""

    n_events = t.shape[0]
    ts = np.full(width * height, np.int64(-1), dtype=np.int64)
    pol = np.zeros(width * height, dtype=np.int8)
    out = np.zeros(n_events, dtype=np.int64)

    for n in range(n_events):
        xe = x[n]
        ye = y[n]
        te = t[n]
        pe = np.int8(1) if p[n] > 0 else np.int8(-1)

        x0 = xe - radius
        if x0 < 0:
            x0 = 0
        x1 = xe + radius
        if x1 > width - 1:
            x1 = width - 1
        y0 = ye - radius
        if y0 < 0:
            y0 = 0
        y1 = ye + radius
        if y1 > height - 1:
            y1 = height - 1

        acc = 0
        for j in range(y0, y1 + 1):
            base = j * width
            for i in range(x0, x1 + 1):
                if i == xe and j == ye:
                    continue          # the centre pixel is excluded, as in EBF
                idx = base + i
                if pol[idx] != pe:
                    continue          # XNOR, and pol == 0 marks a tap that never fired
                dt = te - ts[idx]
                if dt < 0:
                    dt = -dt          # mirrors the released EBF code; causally dt >= 0
                if dt >= window_us:
                    continue          # outside the trailing window
                acc += 1
        out[n] = acc

        centre = ye * width + xe
        ts[centre] = te
        pol[centre] = pe
    return out


def occupancy_score(events: np.ndarray, width: int, height: int,
                    window_us: float, s: int = OCCUPANCY_S) -> np.ndarray:
    """Trailing-window ring occupancy for a `[t_us, x, y, p]` int64 stream.

    Higher means signal. `p` may be given as {0,1} or {-1,+1}. The stream must be in
    nondecreasing timestamp order: the filter is causal and a reordering would change it.
    """

    if s % 2 == 0:
        raise ValueError(f"neighbourhood size s must be odd, got {s}")
    events = np.ascontiguousarray(events, dtype=np.int64)
    if events.ndim != 2 or events.shape[1] < 4:
        raise ValueError(f"Expected [t,x,y,p] columns, got shape {events.shape}")
    t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    if np.any(x < 0) or np.any(x >= width) or np.any(y < 0) or np.any(y >= height):
        raise ValueError(f"Event coordinates fall outside {width}x{height}")
    return _occupancy_kernel(t, x, y, p, int(width), int(height),
                             np.int64(round(window_us)), (s - 1) // 2)


def occupancy_noise_score(events: np.ndarray, width: int, height: int,
                          window_us: float, s: int = OCCUPANCY_S) -> np.ndarray:
    """The same score negated into this paper's orientation: larger means noise-like."""

    return -occupancy_score(events, width, height, window_us, s=s).astype(np.float64)


def _occupancy_reference(events: np.ndarray, width: int, height: int,
                         window_us: float, s: int = OCCUPANCY_S) -> np.ndarray:
    """Literal, slow reference. Used only to check the compiled kernel."""

    radius = (s - 1) // 2
    ts = np.full((height, width), -1, dtype=np.int64)
    pol = np.zeros((height, width), dtype=np.int64)
    out = np.zeros(len(events), dtype=np.int64)
    for n, (te, xe, ye, pe_raw) in enumerate(np.asarray(events, dtype=np.int64)):
        pe = 1 if pe_raw > 0 else -1
        acc = 0
        for j in range(max(0, ye - radius), min(height, ye + radius + 1)):
            for i in range(max(0, xe - radius), min(width, xe + radius + 1)):
                if i == xe and j == ye:
                    continue
                if ts[j, i] < 0 or pol[j, i] != pe:
                    continue
                if abs(int(te) - int(ts[j, i])) < window_us:
                    acc += 1
        out[n] = acc
        ts[ye, xe] = te
        pol[ye, xe] = pe
    return out


def check_equivalence(width: int = 64, height: int = 48, n_events: int = 4000,
                      window_us: float = 32_000.0, seed: int = 0) -> Tuple[int, int]:
    """Compiled kernel against the literal reference. Returns (worst diff, events)."""

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 200_000, size=n_events)).astype(np.int64)
    events = np.column_stack((t,
                              rng.integers(0, width, n_events),
                              rng.integers(0, height, n_events),
                              rng.integers(0, 2, n_events))).astype(np.int64)
    fast = occupancy_score(events, width, height, window_us)
    slow = _occupancy_reference(events, width, height, window_us)
    worst = int(np.abs(fast - slow).max())
    if worst != 0:
        raise AssertionError(f"occupancy kernel disagrees with its reference by {worst}")
    return worst, n_events


def check_properties(width: int = 40, height: int = 32, n_events: int = 3000,
                     window_us: float = 32_000.0, seed: int = 1) -> None:
    """Four properties the reference implementation cannot vouch for.

    1. The score is integer-valued and fits in [0, 24], so five bits hold it exactly.
    2. With W = tau it counts exactly the taps EBF gives strictly positive weight, so the
       two scores are the count and the sum of the *same* term set -- the matched contrast.
    3. It is strictly causal: truncating the stream never changes an earlier event's score.
    4. Enlarging the window can only raise the count, never lower it.
    """

    from .ebf import _ebf_kernel

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 150_000, size=n_events)).astype(np.int64)
    events = np.column_stack((t,
                              rng.integers(0, width, n_events),
                              rng.integers(0, height, n_events),
                              rng.integers(0, 2, n_events))).astype(np.int64)

    score = occupancy_score(events, width, height, window_us)
    if score.dtype.kind != "i":
        raise AssertionError("occupancy score is not integer-valued")
    if score.min() < 0 or score.max() > OCCUPANCY_MAX:
        raise AssertionError(f"occupancy score left [0, {OCCUPANCY_MAX}]")
    if (1 << OCCUPANCY_BITS) <= OCCUPANCY_MAX:
        raise AssertionError("five bits do not hold the stated maximum")

    # Property 2: a tap contributes to EBF iff it contributes to the count, when W = tau.
    # EBF's weight 1 - dt/tau is strictly positive exactly on 0 <= dt < tau, which is the
    # count's admission rule. So the count is zero iff EBF's sum is zero, event by event.
    ebf = _ebf_kernel(events[:, 0], events[:, 1], events[:, 2], events[:, 3],
                      width, height, float(window_us), (OCCUPANCY_S - 1) // 2)
    if np.any((score == 0) != (ebf <= 0.0)):
        raise AssertionError("the count and EBF's sum do not share their term set")

    # Property 3: causality. Scores of the first half must not depend on the second.
    half = n_events // 2
    truncated = occupancy_score(events[:half], width, height, window_us)
    if not np.array_equal(truncated, score[:half]):
        raise AssertionError("the occupancy score reads future events")

    # Property 4: monotone in the window.
    wider = occupancy_score(events, width, height, window_us * 2)
    if np.any(wider < score):
        raise AssertionError("enlarging the window lowered the count")
