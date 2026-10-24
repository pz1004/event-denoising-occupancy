"""Apply a trained Native3D-SSM checkpoint directly to raw AEDAT 3.1 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core import NativeStateBank, StateConfig, bidirectional_features
from .data import iter_aedat31
from .model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    model, training_config, metadata = load_checkpoint(args.checkpoint)
    config = StateConfig(
        width=args.width,
        height=args.height,
        taus_us=training_config.taus_us,
        radius=training_config.radius,
        max_age_factor=training_config.max_age_factor,
        global_reference_pixels=training_config.global_reference_pixels,
    )
    state = NativeStateBank(config)
    device = torch.device(args.device)
    model.to(device).eval()
    retained_event_chunks = []
    retained_probability_chunks = []
    input_events = 0
    pending = np.empty((0, 4), dtype=np.int64)
    for packet in iter_aedat31(args.input):
        pending = np.concatenate((pending, packet))
        while len(pending) >= args.block_size:
            events = pending[: args.block_size]
            pending = pending[args.block_size :]
            features = (
                state.extract(events)
                if model.context_mode == "causal"
                else bidirectional_features(events, config)
            )
            noise_probability = model.noise_probability(features, device)
            retained = noise_probability < args.threshold
            retained_event_chunks.append(events[retained].copy())
            retained_probability_chunks.append(noise_probability[retained].copy())
            input_events += len(events)
    if len(pending):
        events = pending
        features = (
            state.extract(events)
            if model.context_mode == "causal"
            else bidirectional_features(events, config)
        )
        noise_probability = model.noise_probability(features, device)
        retained = noise_probability < args.threshold
        retained_event_chunks.append(events[retained].copy())
        retained_probability_chunks.append(noise_probability[retained].copy())
        input_events += len(events)
    retained_events = (
        np.concatenate(retained_event_chunks)
        if retained_event_chunks
        else np.empty((0, 4), dtype=np.int64)
    )
    retained_probabilities = (
        np.concatenate(retained_probability_chunks)
        if retained_probability_chunks
        else np.empty(0, dtype=np.float32)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        events=retained_events,
        noise_probability=retained_probabilities,
        event_columns=np.asarray(("timestamp", "x", "y", "polarity")),
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
                "training_metadata": metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
