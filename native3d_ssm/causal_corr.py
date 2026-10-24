"""Causal online corr(G,R) estimators (Phase 5b, the runtime-adaptation primitive).

Phases 2'-6 compute corr(G,R) once per 10,240-event block and then apply the gate
beta = 1 iff corr < 0 to that whole block. That is a *batch* statistic: event i is
gated using events after i. It is fine for offline benchmarking but it is not the
runtime adaptation the contribution claims, and a hardware implementation cannot
buffer a block before deciding.

This module provides the streamed replacement: running mean, variance and
covariance of the two operands, giving corr(G,R) at every event from events <= i
only. Two variants:

`cumulative_corr`
    Exact running Pearson correlation over the whole history. Zero parameters. In
    closed form from five cumulative sums, so it stays vectorized (a per-event
    Python loop over 38 M dev events would not be practical).

`ema_corr`
    Exponentially weighted with a forgetting factor `alpha`; effective window
    ~1/alpha events. Required for a non-stationary stream, where a cumulative
    estimator eventually stops responding. `alpha` is the "adaptation time
    constant" meta-parameter flagged in `improve_plan.md` §4 -- the one genuinely
    new tunable this phase introduces. Implemented as a first-order IIR filter
    (`scipy.signal.lfilter`), i.e. adds and multiplies only, which is the
    fixed-point-friendly form noted in `improve_plan.md` §6.

Cold start: correlation is undefined until both operands show spread, so the gate
falls back to `cold_start_beta`. The default is 0 = ring-only, because ring-only is
the branch validated on 3 of the 4 datasets (E-MLB, ED24, DVSCLEAN) while beta = 1
rests on DND21 alone -- so it is the conservative default.

Scope note: this makes *adaptation* causal. The operands themselves are still the
disclosed within-block bidirectional pass, so decay causality is unchanged. Pass
forward-only operands to close both axes at once; `phase5b_causal_corr.py` reports
that variant separately.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

EPS = 1e-12
DEFAULT_COLD_START_BETA = 0.0
DEFAULT_WARMUP = 1024


def cumulative_corr(g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Running Pearson corr(G,R) after each event, using events <= i only."""

    g = np.asarray(g, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = np.arange(1, g.size + 1, dtype=np.float64)
    mean_g = np.cumsum(g) / n
    mean_r = np.cumsum(r) / n
    var_g = np.cumsum(g * g) / n - mean_g * mean_g
    var_r = np.cumsum(r * r) / n - mean_r * mean_r
    cov = np.cumsum(g * r) / n - mean_g * mean_r
    denominator = np.sqrt(np.maximum(var_g, 0.0) * np.maximum(var_r, 0.0))
    corr = np.where(denominator > EPS, cov / np.maximum(denominator, EPS), 0.0)
    return np.clip(corr, -1.0, 1.0)


def _ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    """First-order IIR low-pass: y[i] = (1-alpha) y[i-1] + alpha x[i], y[-1] = x[0]."""

    from scipy.signal import lfilter

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    # Initialise the filter state at x[0] so the estimate starts unbiased instead
    # of decaying up from zero.
    zi = np.array([(1.0 - alpha) * values[0]])
    filtered, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], values, zi=zi)
    return filtered


def ema_corr(g: np.ndarray, r: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted running corr(G,R); effective window ~1/alpha events."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    g = np.asarray(g, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    mean_g = _ewma(g, alpha)
    mean_r = _ewma(r, alpha)
    var_g = _ewma(g * g, alpha) - mean_g * mean_g
    var_r = _ewma(r * r, alpha) - mean_r * mean_r
    cov = _ewma(g * r, alpha) - mean_g * mean_r
    denominator = np.sqrt(np.maximum(var_g, 0.0) * np.maximum(var_r, 0.0))
    corr = np.where(denominator > EPS, cov / np.maximum(denominator, EPS), 0.0)
    return np.clip(corr, -1.0, 1.0)


def causal_beta(corr: np.ndarray, warmup: int = DEFAULT_WARMUP,
                cold_start_beta: float = DEFAULT_COLD_START_BETA) -> np.ndarray:
    """Per-event gate from a running corr: beta = 1 iff corr < 0, cold start held."""

    corr = np.asarray(corr)
    beta = (corr < 0.0).astype(np.float64)
    hold = min(int(max(warmup, 0)), corr.size)
    if hold:
        beta[:hold] = cold_start_beta
    return beta


def causal_scores(g: np.ndarray, r: np.ndarray, corr: np.ndarray,
                  warmup: int = DEFAULT_WARMUP,
                  cold_start_beta: float = DEFAULT_COLD_START_BETA) -> Tuple[np.ndarray, np.ndarray]:
    """Score s_i = beta_i * G_i - R_i under a per-event causal gate."""

    beta = causal_beta(corr, warmup, cold_start_beta)
    return beta * np.asarray(g, dtype=np.float64) - np.asarray(r, dtype=np.float64), beta


def crossing_index(corr: np.ndarray, target_sign: int, offset: int = 0,
                   hold: int = 1) -> int:
    """First index >= offset where `corr` holds `target_sign` for `hold` events.

    Returns -1 if the sign is never held that long. Used to measure adaptation
    latency after a regime change: how many events the streamed estimator needs
    before the gate settles on the new regime.
    """

    corr = np.asarray(corr)
    if target_sign < 0:
        matches = corr < 0.0
    else:
        matches = corr >= 0.0
    matches = matches[offset:]
    if matches.size < hold:
        return -1
    if hold == 1:
        found = np.flatnonzero(matches)
        return int(offset + found[0]) if found.size else -1
    # Run-length: first window of `hold` consecutive matches.
    padded = np.concatenate(([0], np.cumsum(matches.astype(np.int64))))
    window = padded[hold:] - padded[:-hold]
    found = np.flatnonzero(window == hold)
    return int(offset + found[0]) if found.size else -1


__all__ = [
    "DEFAULT_COLD_START_BETA",
    "DEFAULT_WARMUP",
    "causal_beta",
    "causal_scores",
    "crossing_index",
    "cumulative_corr",
    "ema_corr",
]
