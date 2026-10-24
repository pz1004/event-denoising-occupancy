"""Label-free per-block retention estimators (Phase 6, operating-point track).

Phases 0-4 evaluated ROC AUC, a *ranking* metric that is invariant to how many
events are finally kept. Phase 5 (E-MLB / MESR) forced an explicit retention
decision and used a fixed fraction `r`, which the sweep showed is clearly
suboptimal (MESR peaks at r ~ 0.5 on Daylight but r ~ 0.3 on Night).

This module supplies the *adaptive* replacement: given the frozen corr-gated
score `s = beta*G - R` of one block (noise scores high, signal low), estimate the
fraction of events to keep using only the score distribution -- no labels, no
per-dataset constant.

Four estimators are provided. The development calibration (Phase 6a) shows the
informative statistic is *regime-dependent*, exactly like the sign of the global
operand, so `gated_retention` is the deployable rule and the others are its
branches or references.

`gated_retention`  <- the frozen rule
    corr(G,R) < 0 -> `otsu_retention`; corr(G,R) >= 0 -> `below_top_retention`.
    Keyed by the same theory-fixed theta = 0 switch already frozen for beta, so
    it adds no parameter.

`below_top_retention`
    Fraction scoring strictly below the maximum. In the sparse (+corr) regime the
    score collapses onto an atom at 0 that holds the "no evidence" events.

`otsu_retention`
    Otsu's between-class-variance threshold on the score histogram. Zero tuned
    parameters (the bin count is a discretization, not a fitted value) and O(N)
    per block: a histogram plus two cumulative sums. Because it is computed
    *within* each block it cannot suffer the cross-dataset extrapolation failure
    that defeated the rho-keyed beta law in Phase 2'.

`gmm2_retention`
    Two-component 1-D Gaussian mixture; the fraction assigned to the low-mean
    (signal) component. More expressive but iterative, orders of magnitude
    slower, and therefore a development-time reference rather than a deployable
    rule.

Both fall back to `default` retention when the block carries no usable score
spread (constant score). The fallback is 1.0 = "keep everything", i.e. behave
like Raw, which is the conservative choice: Raw is the weakest row of E-MLB
Table 2, so the fallback cannot flatter the method.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

DEFAULT_BINS = 256
DEFAULT_RETENTION = 1.0


def otsu_threshold(values: np.ndarray, bins: int = DEFAULT_BINS) -> Optional[float]:
    """Otsu threshold maximizing between-class variance, or None if degenerate."""

    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    lo, hi = float(v.min()), float(v.max())
    if not (hi > lo):
        return None
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    weights = counts.astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_cum = np.cumsum(weights)[:-1]          # mass at or below each split
    moment_cum = np.cumsum(weights * centers)[:-1]
    moment_total = float((weights * centers).sum())
    weight_hi = total - weight_cum
    usable = (weight_cum > 0) & (weight_hi > 0)
    if not usable.any():
        return None
    mean_lo = np.divide(moment_cum, weight_cum, out=np.zeros_like(moment_cum), where=usable)
    mean_hi = np.divide(moment_total - moment_cum, weight_hi,
                        out=np.zeros_like(moment_cum), where=usable)
    between = weight_cum * weight_hi * (mean_lo - mean_hi) ** 2
    between[~usable] = -1.0
    return float(edges[int(np.argmax(between)) + 1])


def _clamp(retention: float, n_events: int) -> float:
    """Keep retention inside [1 event, all events]."""

    if n_events <= 0:
        return DEFAULT_RETENTION
    return float(min(1.0, max(1.0 / n_events, retention)))


def otsu_retention(scores: np.ndarray, bins: int = DEFAULT_BINS,
                   default: float = DEFAULT_RETENTION) -> float:
    """Fraction of events at or below the Otsu threshold (label-free)."""

    scores = np.asarray(scores)
    threshold = otsu_threshold(scores, bins)
    if threshold is None:
        return _clamp(default, scores.size)
    return _clamp(float(np.mean(scores <= threshold)), scores.size)


def gmm2_retention(scores: np.ndarray, seed: int = 0,
                   default: float = DEFAULT_RETENTION) -> float:
    """Fraction assigned to the low-mean component of a 2-component 1-D GMM."""

    from sklearn.mixture import GaussianMixture

    v = np.asarray(scores, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 4 or v.std() <= 0:
        return _clamp(default, v.size)
    model = GaussianMixture(n_components=2, random_state=seed, n_init=1).fit(v.reshape(-1, 1))
    low = int(np.argmin(model.means_.ravel()))
    return _clamp(float(np.mean(model.predict(v.reshape(-1, 1)) == low)), v.size)


def below_top_retention(scores: np.ndarray,
                        default: float = DEFAULT_RETENTION) -> float:
    """Fraction of events scoring strictly below the maximum score.

    In the sparse (+corr) regime the gate sets beta = 0, so `s = -R` and every
    event whose same-polarity ring is empty within 250 us lands on the single
    value 0 -- which is also the maximum. That atom is the "no evidence" class
    and holds 40-60% of a DVSCLEAN block. Rejecting exactly the atom therefore
    keeps the events carrying positive ring evidence, which tracks the true
    signal fraction closely. On a continuous score (the dense -corr regime) this
    degenerates to 1 - 1/N, i.e. "keep everything", so it must only be used
    where an atom is expected -- see `gated_retention`.
    """

    v = np.asarray(scores)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return _clamp(default, v.size)
    top = float(v.max())
    below = float(np.mean(v < top))
    if below <= 0.0:                     # every score identical: no information
        return _clamp(default, v.size)
    return _clamp(below, v.size)


def gated_retention(scores: np.ndarray, corr: float, bins: int = DEFAULT_BINS,
                    default: float = DEFAULT_RETENTION) -> float:
    """Regime-gated retention, keyed by the same theory-fixed corr(G,R) < 0 switch.

    corr < 0  (dense regime, beta = 1, continuous score)  -> Otsu.
    corr >= 0 (sparse regime, beta = 0, score has a top atom) -> `below_top`.

    No parameter is introduced beyond the threshold theta = 0 already frozen for
    beta in Phase 3+: the sign of corr(G,R) selects both the operand weighting
    and which retention statistic carries the signal fraction.
    """

    if corr < 0.0:
        return otsu_retention(scores, bins=bins, default=default)
    return below_top_retention(scores, default=default)


def balanced_accuracy_curve(scores: np.ndarray, labels: np.ndarray) -> Optional[np.ndarray]:
    """Balanced accuracy for every prefix size k = 1..N of the score ordering.

    Convention: label 1 = noise, and the k lowest-scoring events are kept
    (predicted signal). Returns None when the block is single-class.
    """

    labels = np.asarray(labels).astype(np.float64)
    n_events = labels.size
    noise_total = float(labels.sum())
    signal_total = n_events - noise_total
    if noise_total <= 0 or signal_total <= 0:
        return None
    order = np.argsort(np.asarray(scores), kind="stable")
    noise_kept = np.cumsum(labels[order])                 # k = 1..N
    k = np.arange(1, n_events + 1, dtype=np.float64)
    noise_rejected = noise_total - noise_kept
    signal_kept = k - noise_kept
    return 0.5 * (noise_rejected / noise_total + signal_kept / signal_total)


def oracle_retention(scores: np.ndarray, labels: np.ndarray) -> Optional[Tuple[float, float]]:
    """Label-using best retention and its balanced accuracy (a ceiling, not a rule)."""

    curve = balanced_accuracy_curve(scores, labels)
    if curve is None:
        return None
    best = int(np.argmax(curve))
    return (best + 1) / curve.size, float(curve[best])


def balanced_accuracy_at(scores: np.ndarray, labels: np.ndarray,
                         retention: float) -> Optional[float]:
    """Balanced accuracy when the lowest-scoring `retention` fraction is kept."""

    curve = balanced_accuracy_curve(scores, labels)
    if curve is None:
        return None
    n_events = curve.size
    k = int(min(n_events, max(1, round(retention * n_events))))
    return float(curve[k - 1])


__all__ = [
    "DEFAULT_BINS",
    "DEFAULT_RETENTION",
    "balanced_accuracy_at",
    "balanced_accuracy_curve",
    "below_top_retention",
    "gated_retention",
    "gmm2_retention",
    "oracle_retention",
    "otsu_retention",
    "otsu_threshold",
]
