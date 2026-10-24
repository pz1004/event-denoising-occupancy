"""Apply the zero-parameter Native3D local/global contrast to raw AEDAT 3.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from .analytic import analytic_contrast_score
from .core import StateConfig, bidirectional_features
from .data import iter_aedat31


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Calibration-frozen score threshold; no universal threshold is implied.",
    )
    parser.add_argument(
        "--noise-when",
        choices=("higher", "lower"),
        default="higher",
        help="Reject scores at/above the threshold, or at/below it.",
    )
    parser.add_argument(
        "--quantization", choices=("float", "q12.8", "q8.4"), default="q8.4"
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=10_240)
    return parser.parse_args()


def score_analytic_events(
    events: np.ndarray,
    config: StateConfig,
    block_size: int = 10_240,
    quantization: Literal["float", "q12.8", "q8.4"] = "q8.4",
) -> np.ndarray:
    """Score ordered events in independent bounded-bidirectional blocks."""

    events = np.ascontiguousarray(events, dtype=np.int64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape [N, 4]")
    if len(events) and np.any(np.diff(events[:, 0]) < 0):
        raise ValueError("events must be ordered by nondecreasing timestamp")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    scores = np.empty(len(events), dtype=np.float32)
    for start in range(0, len(events), block_size):
        end = min(start + block_size, len(events))
        features = bidirectional_features(events[start:end], config)
        scores[start:end] = analytic_contrast_score(
            features, config, quantization
        )
    return scores


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if not np.isfinite(args.threshold):
        raise ValueError("threshold must be finite")
    # The final score consumes only the 250-us scale. Keeping the generic
    # four-scale state in evaluators reproduces historical artifacts, while the
    # deployable raw path prunes the three unused independent state planes.
    config = StateConfig(width=args.width, height=args.height, taus_us=(250.0,))
    packets = list(iter_aedat31(args.input))
    events = (
        np.ascontiguousarray(np.concatenate(packets), dtype=np.int64)
        if packets
        else np.empty((0, 4), dtype=np.int64)
    )
    differences = np.diff(events[:, 0])
    backward = differences[differences < 0]
    if len(backward):
        # The released user08_led recording contains one overlapping packet.
        # A stable sort preserves AEDAT order for timestamp ties and changes no
        # event value, matching the downstream raw-data decoder.
        events = np.ascontiguousarray(
            events[np.argsort(events[:, 0], kind="mergesort")]
        )
    scores = score_analytic_events(
        events,
        config,
        block_size=args.block_size,
        quantization=args.quantization,
    )
    noise = (
        scores >= args.threshold
        if args.noise_when == "higher"
        else scores <= args.threshold
    )
    retained_events = events[~noise].copy()
    retained_scores = scores[~noise].copy()
    input_events = len(events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        events=retained_events,
        noise_score=retained_scores,
        event_columns=np.asarray(("timestamp", "x", "y", "polarity")),
        method=np.asarray("bidirectional_250us_local_global_contrast"),
        quantization=np.asarray(args.quantization),
        threshold=np.asarray(args.threshold, dtype=np.float32),
        noise_when=np.asarray(args.noise_when),
        block_size=np.asarray(args.block_size, dtype=np.int64),
        timestamp_stable_sort_applied=np.asarray(bool(len(backward))),
        timestamp_backward_jumps=np.asarray(len(backward), dtype=np.int64),
    )
    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "input_events": input_events,
                "retained_events": int(len(retained_events)),
                "retention": len(retained_events) / input_events if input_events else 0.0,
                "threshold": args.threshold,
                "noise_when": args.noise_when,
                "quantization": args.quantization,
                "block_size": args.block_size,
                "timestamp_repair": {
                    "stable_sort_applied": bool(len(backward)),
                    "backward_jumps": int(len(backward)),
                    "maximum_backward_us": (
                        int(-backward.min()) if len(backward) else 0
                    ),
                },
                "state_config": config.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
