"""Evaluate robust float/QAT checkpoints and the analytic contrast on DND21."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .core import bidirectional_features
from .data import iter_labeled_text
from .evaluate_advanced_suite import (
    _atomic_write,
    _candidate_groups,
    _score_models,
    summarize,
)
from .evaluate_dnd21 import PAPER_AUC, SCENE_FILES
from .robust import (
    analytic_contrast_score,
    load_robust_checkpoint,
    rate_invariant_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", nargs="+", default=list(PAPER_AUC))
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_FILES))
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_suite(paths, device):
    models = {}
    descriptions = {}
    common_config = None
    for path in paths:
        model, config, metadata = load_robust_checkpoint(path)
        identifier = f"seed{int(metadata['seed'])}_{metadata['stage']}"
        if identifier in models:
            raise ValueError(f"Duplicate checkpoint identity: {identifier}")
        if common_config is None:
            common_config = config
        elif config != common_config:
            raise ValueError("Robust checkpoint state configurations differ")
        model.to(device).eval()
        models[identifier] = model
        descriptions[identifier] = {
            "checkpoint": str(path.resolve()),
            "seed": int(metadata["seed"]),
            "stage": str(metadata["stage"]),
            "validation_auc": float(metadata["best_validation_auc"]),
            "parameters": model.trainable_parameters,
            "qat_enabled": model.is_qat_enabled,
        }
    if not models:
        raise ValueError("No robust checkpoints supplied")
    return models, descriptions, common_config


def evaluate_stream(
    path,
    models,
    descriptions,
    config,
    device,
    block_size,
    max_blocks,
):
    groups = _candidate_groups(descriptions)
    candidate_ids = list(models) + list(groups) + ["analytic_contrast"]
    block_aucs = {identifier: [] for identifier in candidate_ids}
    score_batches = {identifier: [] for identifier in candidate_ids}
    label_batches = []
    feature_seconds = 0.0
    classifier_seconds = 0.0
    wall_started = time.perf_counter()
    for values in iter_labeled_text(path, block_size):
        if max_blocks is not None and len(label_batches) >= max_blocks:
            break
        if len(values) != block_size:
            break
        started = time.perf_counter()
        raw = bidirectional_features(values[:, :4], config)
        features = rate_invariant_features(raw, config)
        analytic = analytic_contrast_score(raw, config)
        feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        scores = _score_models(models, features, device)
        for group_id, members in groups.items():
            scores[group_id] = np.mean([scores[member] for member in members], axis=0)
        scores["analytic_contrast"] = analytic
        if device.type == "cuda":
            torch.cuda.synchronize()
        classifier_seconds += time.perf_counter() - started
        labels = values[:, 4].astype(np.int64, copy=True)
        label_batches.append(labels)
        for identifier in candidate_ids:
            block_aucs[identifier].append(
                float(roc_auc_score(labels, scores[identifier]))
            )
            score_batches[identifier].append(scores[identifier])
        blocks = len(label_batches)
        if blocks == 1 or blocks % 100 == 0:
            print(
                f"{path.name}: blocks={blocks} "
                f"robust={np.mean(block_aucs['ensemble_float']):.6f} "
                f"analytic={np.mean(block_aucs['analytic_contrast']):.6f}",
                flush=True,
            )
    if not label_batches:
        raise RuntimeError(f"No complete blocks found in {path}")
    labels = np.concatenate(label_batches)
    candidates = {}
    for identifier in candidate_ids:
        scores = np.concatenate(score_batches[identifier])
        candidates[identifier] = {
            "mean_block_auc": float(np.mean(block_aucs[identifier])),
            "std_block_auc": (
                float(np.std(block_aucs[identifier], ddof=1))
                if len(block_aucs[identifier]) > 1
                else 0.0
            ),
            "global_event_auc": float(roc_auc_score(labels, scores)),
        }
    return {
        "source": str(path.resolve()),
        "blocks": len(label_batches),
        "events": int(len(labels)),
        "feature_seconds": feature_seconds,
        "classifier_seconds_all_models": classifier_seconds,
        "wall_seconds": time.perf_counter() - wall_started,
        "groups": groups,
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    models, descriptions, config = load_suite(args.checkpoints, device)
    payload = {
        "device": str(device),
        "protocol": {
            "block_size": args.block_size,
            "tail_policy": "drop incomplete final block",
            "shared_feature_extraction": True,
            "analytic_contrast_selected_on": "ED24/synthetic-development; DND diagnostic disclosed",
        },
        "state_config": config.to_dict(),
        "models": descriptions,
        "results": {},
    }
    for rate in args.rates:
        if rate not in PAPER_AUC:
            raise ValueError(f"Unsupported rate: {rate}")
        payload["results"][rate] = {}
        for scene in args.scenes:
            if scene not in SCENE_FILES:
                raise ValueError(f"Unsupported scene: {scene}")
            stream = evaluate_stream(
                args.dataset_root / rate / SCENE_FILES[scene],
                models,
                descriptions,
                config,
                device,
                args.block_size,
                args.max_blocks,
            )
            target = PAPER_AUC[rate][scene]
            stream["edmamba_paper_auc"] = target
            for metrics in stream["candidates"].values():
                metrics["delta_vs_edmamba"] = metrics["mean_block_auc"] - target
            payload["results"][rate][scene] = stream
            payload["summary"] = summarize(payload["results"], descriptions)
            _atomic_write(args.output, payload)
            best_id, best = max(
                stream["candidates"].items(), key=lambda item: item[1]["mean_block_auc"]
            )
            print(
                f"RESULT {rate} {scene}: {best_id}={best['mean_block_auc']:.6f} "
                f"delta={best['delta_vs_edmamba']:+.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
