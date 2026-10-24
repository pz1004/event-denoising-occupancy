"""Evaluate a split-state/motion Native3D-SSM checkpoint on DND21."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .advanced import advanced_block_features
from .advanced_model import load_advanced_checkpoint
from .core import bidirectional_features
from .data import iter_labeled_text
from .evaluate_dnd21 import PAPER_AUC, SCENE_FILES
from .model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", nargs="+", default=list(PAPER_AUC))
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_FILES))
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate_stream(path, model, bootstrap, config, device, block_size, max_blocks):
    block_aucs = []
    all_scores = []
    all_labels = []
    started = time.perf_counter()
    for values in iter_labeled_text(path, block_size):
        if max_blocks is not None and len(block_aucs) >= max_blocks:
            break
        if len(values) != block_size:
            break
        events = values[:, :4]
        raw = bidirectional_features(events, config.state)
        bootstrap_probability = bootstrap.noise_probability(raw, device)
        features = advanced_block_features(
            events, bootstrap_probability, config, raw_features=raw
        )
        scores = model.noise_probability(features, device)
        labels = values[:, 4]
        block_aucs.append(float(roc_auc_score(labels, scores)))
        all_scores.append(scores)
        all_labels.append(labels.copy())
        if len(block_aucs) == 1 or len(block_aucs) % 100 == 0:
            print(
                f"{path.name}: blocks={len(block_aucs)} auc={np.mean(block_aucs):.6f}",
                flush=True,
            )
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    return {
        "blocks": len(block_aucs),
        "events": int(len(labels)),
        "mean_block_auc": float(np.mean(block_aucs)),
        "std_block_auc": float(np.std(block_aucs, ddof=1)),
        "global_event_auc": float(roc_auc_score(labels, scores)),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, metadata = load_advanced_checkpoint(args.checkpoint)
    bootstrap, bootstrap_config, bootstrap_metadata = load_checkpoint(
        args.bootstrap_checkpoint
    )
    if bootstrap_config != config.state:
        raise ValueError("Bootstrap and advanced state configurations differ")
    expected_hash = metadata.get("bootstrap_sha256")
    if expected_hash:
        from .train_advanced import sha256

        if sha256(args.bootstrap_checkpoint) != expected_hash:
            raise ValueError("Bootstrap checkpoint hash does not match training metadata")
    model.to(device).eval()
    bootstrap.to(device).eval()
    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "bootstrap_checkpoint": str(args.bootstrap_checkpoint.resolve()),
        "metadata": metadata,
        "parameters": model.trainable_parameters,
        "qat_enabled": model.is_qat_enabled,
        "results": {},
    }
    for rate in args.rates:
        results["results"][rate] = {}
        for scene in args.scenes:
            path = args.dataset_root / rate / SCENE_FILES[scene]
            stream = evaluate_stream(
                path,
                model,
                bootstrap,
                config,
                device,
                args.block_size,
                args.max_blocks,
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
