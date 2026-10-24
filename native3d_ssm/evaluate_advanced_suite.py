"""Evaluate multiple advanced checkpoints with one shared DND21 feature pass."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .advanced import advanced_block_features
from .advanced_model import load_advanced_checkpoint
from .core import bidirectional_features
from .data import iter_labeled_text
from .evaluate_dnd21 import PAPER_AUC, SCENE_FILES
from .model import load_checkpoint
from .train_advanced import sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", nargs="+", default=list(PAPER_AUC))
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_FILES))
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _checkpoint_id(metadata: Dict) -> str:
    if "seed" not in metadata or "stage" not in metadata:
        raise ValueError("Every checkpoint needs seed and stage metadata")
    stage = str(metadata["stage"])
    if stage not in ("float", "qat"):
        raise ValueError(f"Unsupported checkpoint stage: {stage}")
    return f"seed{int(metadata['seed'])}_{stage}"


def load_suite(paths: Sequence[Path], bootstrap_path: Path, device: torch.device):
    bootstrap, bootstrap_config, _ = load_checkpoint(bootstrap_path)
    bootstrap.to(device).eval()
    digest = sha256(bootstrap_path)
    models = {}
    descriptions = {}
    common_config = None
    for path in paths:
        model, config, metadata = load_advanced_checkpoint(path)
        identifier = _checkpoint_id(metadata)
        if identifier in models:
            raise ValueError(f"Duplicate checkpoint identity: {identifier}")
        if common_config is None:
            common_config = config
        elif config != common_config:
            raise ValueError("Advanced checkpoint configurations differ")
        if config.state != bootstrap_config:
            raise ValueError("Bootstrap and advanced state configurations differ")
        expected_hash = metadata.get("bootstrap_sha256")
        if expected_hash and expected_hash != digest:
            raise ValueError(f"Bootstrap hash mismatch for {identifier}")
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
        raise ValueError("No checkpoints supplied")
    return models, descriptions, bootstrap, common_config


@torch.inference_mode()
def _score_models(models: Dict, features: np.ndarray, device: torch.device) -> Dict:
    # Upload a feature block only once; all classifiers consume the same tensor.
    batch = torch.from_numpy(
        np.array(features, dtype=np.float32, order="C", copy=True)
    ).to(device)
    scores = {}
    for identifier, model in models.items():
        scores[identifier] = (
            torch.softmax(model(batch), dim=-1)[:, 1].detach().cpu().numpy()
        )
    return scores


def _candidate_groups(descriptions: Dict) -> Dict[str, List[str]]:
    groups = {}
    for stage in ("float", "qat"):
        members = [
            identifier
            for identifier, description in descriptions.items()
            if description["stage"] == stage
        ]
        if len(members) > 1:
            groups[f"ensemble_{stage}"] = members
    return groups


def evaluate_stream(
    path: Path,
    models: Dict,
    descriptions: Dict,
    bootstrap,
    config,
    device: torch.device,
    block_size: int,
    max_blocks,
) -> Dict:
    groups = _candidate_groups(descriptions)
    candidate_ids = list(models) + list(groups)
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
        events = values[:, :4]
        started = time.perf_counter()
        raw = bidirectional_features(events, config.state)
        bootstrap_probability = bootstrap.noise_probability(raw, device)
        features = advanced_block_features(
            events, bootstrap_probability, config, raw_features=raw
        )
        feature_seconds += time.perf_counter() - started

        started = time.perf_counter()
        scores = _score_models(models, features, device)
        for group_id, members in groups.items():
            scores[group_id] = np.mean([scores[member] for member in members], axis=0)
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
            ensemble = "ensemble_float" if "ensemble_float" in block_aucs else candidate_ids[0]
            print(
                f"{path.name}: blocks={blocks} "
                f"{ensemble}_auc={np.mean(block_aucs[ensemble]):.6f}",
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


def summarize(results: Dict, descriptions: Dict) -> Dict:
    rows = []
    candidate_values = {}
    for rate, scenes in results.items():
        for scene, stream in scenes.items():
            target = PAPER_AUC[rate][scene]
            for identifier, metrics in stream["candidates"].items():
                value = metrics["mean_block_auc"]
                candidate_values.setdefault(identifier, []).append(value)
                rows.append((identifier, rate, scene, value, target))

    candidate_summary = {}
    for identifier, values in candidate_values.items():
        matching = [row for row in rows if row[0] == identifier]
        deltas = [row[3] - row[4] for row in matching]
        candidate_summary[identifier] = {
            "mean_auc": float(np.mean(values)),
            "mean_delta_vs_edmamba": float(np.mean(deltas)),
            "minimum_delta_vs_edmamba": float(np.min(deltas)),
            "conditions_beating_edmamba": int(np.sum(np.asarray(deltas) > 0.0)),
            "conditions": len(deltas),
        }

    seed_statistics = {}
    for stage in ("float", "qat"):
        members = [
            identifier
            for identifier, description in descriptions.items()
            if description["stage"] == stage
        ]
        if len(members) < 2:
            continue
        condition_stats = {}
        for rate in results:
            for scene in results[rate]:
                values = [
                    results[rate][scene]["candidates"][member]["mean_block_auc"]
                    for member in members
                ]
                condition_stats[f"{rate}/{scene}"] = {
                    "mean_auc": float(np.mean(values)),
                    "std_auc": float(np.std(values, ddof=1)),
                    "edmamba_paper_auc": PAPER_AUC[rate][scene],
                    "mean_delta_vs_edmamba": float(
                        np.mean(values) - PAPER_AUC[rate][scene]
                    ),
                }
        seed_statistics[stage] = {
            "members": members,
            "conditions": condition_stats,
            "mean_across_conditions": float(
                np.mean([value["mean_auc"] for value in condition_stats.values()])
            ),
        }
    return {
        "candidates": candidate_summary,
        "across_seed_statistics": seed_statistics,
    }


def _atomic_write(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    models, descriptions, bootstrap, config = load_suite(
        args.checkpoints, args.bootstrap_checkpoint, device
    )
    payload = {
        "bootstrap_checkpoint": str(args.bootstrap_checkpoint.resolve()),
        "device": str(device),
        "protocol": {
            "block_size": args.block_size,
            "tail_policy": "drop incomplete final block",
            "shared_feature_extraction": True,
            "ensemble": "unweighted arithmetic mean of event noise probabilities",
        },
        "advanced_config": config.to_dict(),
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
                bootstrap,
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
