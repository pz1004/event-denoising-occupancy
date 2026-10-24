"""EBF: the Event-Based Bilateral Filter of Guo et al., IEEE TCSVT 2025.

WHY THIS FILE EXISTS. EBF is the strongest published rule-based denoiser we are aware
of and it appeared in the venue this paper targets, so it is the baseline a reader will
ask for first. The released repository (github.com/shicy17/EBF) cannot be run here: its
entry point imports `allrulefilters` and `custom_dataloader.dataloader`, neither of
which ships, and it hardcodes Windows paths and a 346x260 sensor. Re-implementation is
therefore forced, and re-implementation without verification is just an assertion. So
this module carries two implementations -- a fast causal kernel and a literal
transcription of the released array code -- and `check_equivalence` requires them to
agree to machine precision before either is used to produce a number for the paper.

THE ALGORITHM (paper Sec. III, Eqs. 2-4). Keep a Timestamp and Polarity Map (TPM)
holding the last timestamp and last polarity at every pixel. For an incoming event
e = (t_e, x_e, y_e, p_e) with p_e in {+1,-1}, over the s x s square centred on
(x_e, y_e):

    A(i,j) = max(0, 1 - (t_e - t_ij)/tau)      0 where the pixel has no past event
    P(i,j) = XNOR(p_e, p_ij) in {0,1}          P(x_e, y_e) = 0 always
    S      = sum_ij A(i,j) P(i,j)              higher means signal

then write (t_e, p_e) into the TPM at (x_e, y_e). The update follows the read, so EBF
is strictly causal -- it never looks at a future event. Ours is not: it takes a
block-local lookahead. That asymmetry favours us and has to be said wherever the rows
are compared.

Adopted constants are the paper's own: s = 5 and tau = 32 ms, selected in its Sec. V-A
on Driving and Campus, and matching `cmds.txt` in the release. T_EBF is swept, so it
never needs selecting for a ROC-AUC.

THREE DELIBERATE DEVIATIONS from the released script, each of which we consider a fix:

1.  `getageandpolstringTI25` computes the write offset of a clipped patch as
    `(yleftbound-(y-radius)) * radius + (xleftbound-(x-radius))` where the flattened
    patch stride is `patchsize`, not `radius`. For any event within `radius` of a
    border this scatters the gathered neighbourhood into the wrong cells. We gather
    the window correctly. The two agree exactly on interior events, which is what
    `check_equivalence` tests and what almost every event is.
2.  The released test path never calls `initializetsandpolmap`, so the TPM starts at
    zero. We match that: no random warm start. It makes no difference either way,
    because the zero-initialised polarity map already forces P = 0 at every pixel that
    has not yet fired.
3.  Sensor size is a parameter here rather than the hardcoded 346x260, since three of
    the four benchmarks are not that size.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit

#: The paper's adopted neighbourhood, s = 5, i.e. a radius-2 square. Sec. V-A.
EBF_S = 5
#: The paper's adopted time window in microseconds, tau = 32 ms. Sec. V-A.
EBF_TAU_US = 32_000


@njit(cache=True, fastmath=False)
def _ebf_kernel(t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray,
                width: int, height: int, tau_us: float, radius: int) -> np.ndarray:
    """Causal EBF correlation score per event. Higher is more signal-like."""

    n_events = t.shape[0]
    ts = np.full(width * height, np.int64(-1), dtype=np.int64)
    pol = np.zeros(width * height, dtype=np.int8)
    out = np.zeros(n_events, dtype=np.float64)

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

        acc = 0.0
        for j in range(y0, y1 + 1):
            base = j * width
            for i in range(x0, x1 + 1):
                if i == xe and j == ye:
                    continue          # P is zero at the centre pixel by definition
                idx = base + i
                if pol[idx] != pe:
                    continue          # XNOR: opposite polarity, or no past event
                dt = te - ts[idx]
                if dt >= tau_us:
                    continue          # A has already decayed to zero
                if dt < 0:
                    dt = -dt          # released code takes |dt|; causally dt >= 0
                acc += 1.0 - dt / tau_us
        out[n] = acc

        centre = ye * width + xe
        ts[centre] = te
        pol[centre] = pe
    return out


def ebf_score(events: np.ndarray, width: int, height: int,
              tau_us: float = EBF_TAU_US, s: int = EBF_S) -> np.ndarray:
    """EBF score for a `[t_us, x, y, p]` int64 stream. Higher means signal.

    `p` may be given as {0,1} or {-1,+1}; both are mapped to the paper's {-1,+1}.
    The stream must be in nondecreasing timestamp order, which every caller here
    guarantees, because EBF is causal and a reordering would silently change it.
    """

    if s % 2 == 0:
        raise ValueError(f"EBF neighbourhood size s must be odd, got {s}")
    events = np.ascontiguousarray(events, dtype=np.int64)
    if events.ndim != 2 or events.shape[1] < 4:
        raise ValueError(f"Expected [t,x,y,p] columns, got shape {events.shape}")
    t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    if np.any(x < 0) or np.any(x >= width) or np.any(y < 0) or np.any(y >= height):
        raise ValueError(f"Event coordinates fall outside {width}x{height}")
    return _ebf_kernel(t, x, y, p, int(width), int(height), float(tau_us), (s - 1) // 2)


def ebf_score_reference(events: np.ndarray, width: int, height: int,
                        tau_us: float = EBF_TAU_US, s: int = EBF_S,
                        patchsize: int = 7) -> np.ndarray:
    """Literal transcription of the released array code, used only to check the kernel.

    This follows `getageandpolstringTI25` (gather a `patchsize` window from the TPM),
    then `preprocessingresizefortest` (linear age ramp in milliseconds, polarity weight
    as a clipped product, centre zeroed), then `trainFunction` (crop to `s`, multiply,
    sum). It is deliberately slow and deliberately not refactored: its value is that it
    was written from the released source rather than from our understanding of it.
    """

    events = np.ascontiguousarray(events, dtype=np.int64)
    radius = (patchsize - 1) // 2
    timestamp_image = np.zeros((height, width), dtype=np.float64)
    last_pol = np.zeros((height, width), dtype=np.float64)

    age = np.zeros((len(events), patchsize, patchsize), dtype=np.float64)
    pol = np.zeros((len(events), patchsize, patchsize), dtype=np.float64)
    target_ts = np.zeros(len(events), dtype=np.float64)
    target_p = np.zeros(len(events), dtype=np.float64)

    for n, (te, xe, ye, pe) in enumerate(events[:, :4]):
        signed = 1.0 if pe > 0 else -1.0
        # Released code gathers a window clipped to the sensor; we place it at the
        # correct offset (deviation 1), which for interior events is offset zero.
        x0, x1 = max(0, xe - radius), min(width - 1, xe + radius)
        y0, y1 = max(0, ye - radius), min(height - 1, ye + radius)
        age[n, y0 - (ye - radius):y1 - (ye - radius) + 1,
            x0 - (xe - radius):x1 - (xe - radius) + 1] = timestamp_image[y0:y1 + 1, x0:x1 + 1]
        pol[n, y0 - (ye - radius):y1 - (ye - radius) + 1,
            x0 - (xe - radius):x1 - (xe - radius) + 1] = last_pol[y0:y1 + 1, x0:x1 + 1]
        target_ts[n] = te
        target_p[n] = signed
        timestamp_image[ye, xe] = te
        last_pol[ye, xe] = signed

    tau_ms = tau_us / 1000.0
    diff_ms = np.abs(age - target_ts[:, None, None]) / 1000.0
    age_w = np.clip((tau_ms - diff_ms) / tau_ms, 0.0, 1.0)
    pol_w = np.clip(pol * target_p[:, None, None], 0.0, 1.0)
    centre = patchsize // 2
    pol_w[:, centre, centre] = 0.0

    lo = (patchsize - s) // 2
    hi = patchsize - lo
    return (age_w[:, lo:hi, lo:hi] * pol_w[:, lo:hi, lo:hi]).sum(axis=(1, 2))


def check_equivalence(width: int = 64, height: int = 48, n_events: int = 4000,
                      seed: int = 0, margin: int = 4) -> Tuple[float, int]:
    """Kernel against literal transcription on a synthetic stream. Raises on mismatch.

    Events are drawn away from the border by `margin` so that the released code's
    patch-offset defect (deviation 1) cannot make the two disagree for a reason that
    is the released code's bug rather than ours.
    """

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 400_000, size=n_events)).astype(np.int64)
    x = rng.integers(margin, width - margin, size=n_events).astype(np.int64)
    y = rng.integers(margin, height - margin, size=n_events).astype(np.int64)
    p = rng.integers(0, 2, size=n_events).astype(np.int64)
    events = np.column_stack((t, x, y, p))

    fast = ebf_score(events, width, height)
    slow = ebf_score_reference(events, width, height)
    worst = float(np.max(np.abs(fast - slow)))
    if worst > 1e-9:
        raise AssertionError(
            f"EBF kernel disagrees with the literal transcription by {worst:.3e} "
            f"on {int(np.sum(np.abs(fast - slow) > 1e-9))} of {n_events} events")
    return worst, int(np.count_nonzero(fast))


def check_properties(width: int = 40, height: int = 32, n_events: int = 3000,
                     seed: int = 1) -> None:
    """Four properties EBF must satisfy by definition. Independent of the transcription.

    `check_equivalence` proves our kernel computes what the released code computes. It
    cannot prove either one computes what the paper describes, because both could share
    a misreading. These four do not depend on the released code at all.
    """

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 200_000, size=n_events)).astype(np.int64)
    x = rng.integers(4, width - 4, size=n_events).astype(np.int64)
    y = rng.integers(4, height - 4, size=n_events).astype(np.int64)
    events_on = np.column_stack((t, x, y, np.ones(n_events, dtype=np.int64)))
    mixed = np.column_stack((t, x, y, rng.integers(0, 2, size=n_events).astype(np.int64)))

    # 1. Bounded above by the neighbourhood size, since every weight is in [0,1] and the
    #    centre is excluded: s*s - 1 = 24 at s = 5.
    scores = ebf_score(mixed, width, height)
    if scores.max() > EBF_S * EBF_S - 1 + 1e-9:
        raise AssertionError(f"EBF exceeded its bound: {scores.max()} > {EBF_S**2 - 1}")
    if scores.min() < 0.0:
        raise AssertionError(f"EBF returned a negative score: {scores.min()}")

    # 2. With a window so long nothing decays, and one polarity only, the score is exactly
    #    the count of distinct occupied neighbouring pixels -- the same occupancy quantity
    #    Proposition 1 is about. This is the strongest check here: it pins the age weight,
    #    the centre exclusion and the neighbourhood shape simultaneously.
    huge = 10 ** 12
    counted = ebf_score(events_on, width, height, tau_us=huge)
    occupancy = np.zeros(n_events, dtype=np.float64)
    seen = np.zeros((height, width), dtype=bool)
    radius = (EBF_S - 1) // 2
    for n, (_te, xe, ye, _pe) in enumerate(events_on[:, :4]):
        window = seen[max(0, ye - radius):ye + radius + 1,
                      max(0, xe - radius):xe + radius + 1]
        occupancy[n] = window.sum() - (1.0 if seen[ye, xe] else 0.0)
        seen[ye, xe] = True
    # `huge` is finite, so each of at most 24 taps carries a weight 1 - dt/tau rather than
    # exactly 1. The residual is bounded by 24 * (span / tau) and the tolerance is that
    # bound, not a number chosen to make the test pass.
    tolerance = (EBF_S * EBF_S - 1) * float(t[-1] - t[0]) / huge * 1.001
    worst = float(np.max(np.abs(counted - occupancy)))
    if worst > tolerance:
        raise AssertionError(
            f"At tau -> infinity EBF should equal same-polarity ring occupancy; "
            f"worst disagreement {worst:.3e} exceeds the finite-tau residual bound "
            f"{tolerance:.3e}")

    # 3. The TPM holds ONE entry per pixel, not one per polarity, so an OFF event
    #    overwrites whatever ON history that pixel carried and the ON events that follow
    #    lose it. Interleaving opposite-polarity events must therefore weakly *lower* the
    #    ON scores and can never raise them. This is a real difference from our own
    #    extractor, which banks the two polarities separately, and it is the property that
    #    would break first if the polarity weight were read as a mask over per-polarity
    #    state rather than as an XNOR against a single shared map.
    alternating = np.column_stack((t, x, y, (np.arange(n_events) % 2).astype(np.int64)))
    on_rows = alternating[:, 3] == 1
    interleaved = ebf_score(alternating, width, height, tau_us=huge)[on_rows]
    alone = ebf_score(alternating[on_rows], width, height, tau_us=huge)
    if np.any(interleaved > alone + 1e-9):
        raise AssertionError(
            f"Interleaving OFF events raised the ON score on "
            f"{int(np.sum(interleaved > alone + 1e-9))} events; the timestamp-and-polarity "
            f"map is being treated as per-polarity, which EBF's is not")

    # 4. Monotone in the window: a longer tau can only keep more history, never less.
    short = ebf_score(mixed, width, height, tau_us=1_000.0)
    long_ = ebf_score(mixed, width, height, tau_us=100_000.0)
    if np.any(short > long_ + 1e-9):
        raise AssertionError("A shorter tau scored higher than a longer one on "
                             f"{int(np.sum(short > long_ + 1e-9))} events")


if __name__ == "__main__":
    check_properties()
    print("EBF properties hold: bounded by 24, equals same-polarity ring occupancy at "
          "tau -> infinity, one shared map per pixel, monotone in tau")
    worst, nonzero = check_equivalence()
    print(f"EBF kernel == literal transcription, max |diff| = {worst:.3e}, "
          f"{nonzero} events scored nonzero")
