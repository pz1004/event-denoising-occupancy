"""Canonical zero-parameter Native3D local/global score definition."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .core import StateConfig


def analytic_contrast_score(
    raw_bidirectional_features: np.ndarray,
    config: StateConfig,
    quantization: Literal["float", "q12.8", "q8.4"] = "float",
) -> np.ndarray:
    """Return the bidirectional 250-us local/global noise score.

    Fixed-point modes quantize only the four nonnegative score operands. State
    storage, timestamp arithmetic, and exponential decay remain floating point,
    matching the bounded emulation reported in the manuscript.
    """

    raw = np.asarray(raw_bidirectional_features)
    if raw.shape != (len(raw), 2 * config.feature_dim):
        raise ValueError("Raw feature layout does not match state configuration")
    scales = len(config.taus_us)
    if config.taus_us[0] != 250.0:
        raise ValueError("The analytic score requires a leading 250-us state")
    future_offset = config.feature_dim
    ring_past = raw[:, 2 * scales]
    global_past = raw[:, 6 * scales]
    ring_future = raw[:, future_offset + 2 * scales]
    global_future = raw[:, future_offset + 6 * scales]
    if quantization != "float":
        formats = {"q12.8": (12, 8), "q8.4": (8, 4)}
        if quantization not in formats:
            raise ValueError(f"Unsupported score quantization: {quantization!r}")
        total_bits, fractional_bits = formats[quantization]
        scale = float(2**fractional_bits)
        maximum = float(2**total_bits - 1) / scale

        def quantize(values: np.ndarray) -> np.ndarray:
            return np.round(np.clip(values, 0.0, maximum) * scale) / scale

        ring_past = quantize(ring_past)
        global_past = quantize(global_past)
        ring_future = quantize(ring_future)
        global_future = quantize(global_future)
    return np.asarray(
        global_past + global_future - ring_past - ring_future,
        dtype=np.float32,
    )
