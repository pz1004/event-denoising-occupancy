"""Evaluate a Native3D-SSM checkpoint on released DND21 labeled streams."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .core import NativeStateBank, bidirectional_features
from .data import iter_labeled_text
from .model import load_checkpoint


PAPER_AUC = {
    "1hz": {"hotel-bar": 0.9963, "driving": 0.9694},
    "3hz": {"hotel-bar": 0.9949, "driving": 0.9710},
    "5hz": {"hotel-bar": 0.9955, "driving": 0.9734},
    "7hz": {"hotel-bar": 0.9939, "driving": 0.9689},
    "10hz": {"hotel-bar": 0.9916, "driving": 0.9636},
}
SCENE_FILES = {"hotel-bar": "mix_result.txt", "driving": "driving_mix_result.txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", nargs="+", default=list(PAPER_AUC))
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_FILES))
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate_stream(path: Path, model, config, device, block_size: int, max_blocks):
    state = NativeStateBank(config)
    block_aucs = []
    scores_all = []
    labels_all = []
    feature_seconds = 0.0
    classifier_seconds = 0.0
    wall_start = time.perf_counter()
    for block_index, values in enumerate(iter_labeled_text(path, block_size), start=1):
        if max_blocks is not None and len(block_aucs) >= max_blocks:
            break
        if len(values) != block_size:
            break
        started = time.perf_counter()
        features = (
            state.extract(values[:, :4])
            if model.context_mode == "causal"
            else bidirectional_features(values[:, :4], config)
        )
        feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        scores = model.noise_probability(features, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        classifier_seconds += time.perf_counter() - started
        labels = values[:, 4]
        block_aucs.append(float(roc_auc_score(labels, scores)))
        scores_all.append(scores)
        labels_all.append(labels.copy())
        if block_index == 1 or block_index % 100 == 0:
            print(f"{path.name}: blocks={block_index} auc={np.mean(block_aucs):.6f}", flush=True)
    if not block_aucs:
        raise RuntimeError(f"No complete blocks found in {path}")
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    return {
        "source": str(path.resolve()),
        "blocks": len(block_aucs),
        "events": int(len(labels)),
        "mean_block_auc": float(np.mean(block_aucs)),
        "std_block_auc": float(np.std(block_aucs, ddof=1)) if len(block_aucs) > 1 else 0.0,
        "global_event_auc": float(roc_auc_score(labels, scores)),
        "feature_seconds": feature_seconds,
        "classifier_seconds": classifier_seconds,
        "wall_seconds": time.perf_counter() - wall_start,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, metadata = load_checkpoint(args.checkpoint)
    model.to(device).eval()
    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata": metadata,
        "protocol": {
            "causal_continuous_state": True,
            "block_size": args.block_size,
            "tail_policy": "drop incomplete final block",
        },
        "model": {
            "parameters": model.trainable_parameters,
            "context_mode": model.context_mode,
            "state_config": config.to_dict(),
            "state_bytes_float32": NativeStateBank(config).state_bytes,
        },
        "results": {},
    }
    for rate in args.rates:
        if rate not in PAPER_AUC:
            raise ValueError(f"Unsupported rate: {rate}")
        results["results"][rate] = {}
        for scene in args.scenes:
            path = args.dataset_root / rate / SCENE_FILES[scene]
            stream = evaluate_stream(
                path, model, config, device, args.block_size, args.max_blocks
            )
            stream["edmamba_paper_auc"] = PAPER_AUC[rate][scene]
            stream["delta_vs_edmamba"] = (
                stream["mean_block_auc"] - PAPER_AUC[rate][scene]
            )
            results["results"][rate][scene] = stream
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
            temporary.replace(args.output)
            print(
                f"RESULT {rate} {scene}: {stream['mean_block_auc']:.6f} "
                f"delta={stream['delta_vs_edmamba']:+.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
