"""Train the rate-invariant Native3D scorer with ED24 Poisson stress data."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .advanced_model import AdvancedEventClassifier
from .core import StateConfig, bidirectional_features
from .data import iter_labeled_text
from .model import load_checkpoint
from .robust import (
    TRANSFORM_VERSION,
    rate_invariant_features,
    robust_feature_dim,
    robust_feature_names,
    save_robust_checkpoint,
)
from .train import split_scene_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path, required=True)
    parser.add_argument("--real-feature-cache", type=Path, required=True)
    parser.add_argument("--synthetic-feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--split-seed", type=int, default=1024)
    parser.add_argument(
        "--synthetic-rates-hz",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument("--synthetic-duration-us", type=int, default=500_000)
    parser.add_argument(
        "--max-synthetic-scenes",
        type=int,
        default=0,
        help="Bound synthetic scenes per split for smoke tests; zero uses all scenes.",
    )
    parser.add_argument("--samples-per-block", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=10_240)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--float-epochs", type=int, default=20)
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


_WORKER_CONFIG = None


def _initialize_worker(config_values: Dict) -> None:
    global _WORKER_CONFIG
    torch.set_num_threads(1)
    _WORKER_CONFIG = StateConfig.from_dict(config_values)


def _read_clean_prefix(path: Path, maximum_duration_us: int) -> np.ndarray:
    batches = []
    first_timestamp = None
    for values in iter_labeled_text(path, 65_536):
        if not len(values):
            continue
        if first_timestamp is None:
            first_timestamp = int(values[0, 0])
        keep = values[:, 0] <= first_timestamp + maximum_duration_us
        if np.any(keep):
            batches.append(values[keep])
        if not np.all(keep):
            break
    if not batches:
        raise RuntimeError(f"No events found in {path}")
    return np.concatenate(batches)


def _synthetic_file_task(task):
    (
        task_index,
        path_text,
        rate_hz,
        group_id,
        seed,
        maximum_duration_us,
        chunk_size,
        samples_per_block,
        balance,
    ) = task
    generator = np.random.default_rng(seed + task_index * 1_000_003)
    clean = _read_clean_prefix(Path(path_text), maximum_duration_us)
    clean_events = np.ascontiguousarray(clean[:, :4], dtype=np.int64)
    clean_labels = clean[:, 4].astype(np.int64, copy=False)
    start = int(clean_events[0, 0])
    stop = int(clean_events[-1, 0])
    duration_us = max(1, stop - start)
    expected_noise = (
        rate_hz
        * _WORKER_CONFIG.width
        * _WORKER_CONFIG.height
        * duration_us
        / 1_000_000.0
    )
    noise_count = int(generator.poisson(expected_noise))
    noise_events = np.empty((noise_count, 4), dtype=np.int64)
    noise_events[:, 0] = generator.integers(start, stop + 1, size=noise_count)
    noise_events[:, 1] = generator.integers(
        0, _WORKER_CONFIG.width, size=noise_count
    )
    noise_events[:, 2] = generator.integers(
        0, _WORKER_CONFIG.height, size=noise_count
    )
    noise_events[:, 3] = generator.integers(0, 2, size=noise_count)
    events = np.concatenate((clean_events, noise_events))
    labels = np.concatenate((clean_labels, np.ones(noise_count, dtype=np.int64)))
    tie_breaker = generator.random(len(events))
    order = np.lexsort((tie_breaker, events[:, 0]))
    events = events[order]
    labels = labels[order]

    feature_batches = []
    label_batches = []
    group_batches = []
    complete_events = 0
    for start_index in range(0, len(events), chunk_size):
        stop_index = start_index + chunk_size
        if stop_index > len(events):
            break
        block_events = events[start_index:stop_index]
        block_labels = labels[start_index:stop_index]
        raw = bidirectional_features(block_events, _WORKER_CONFIG)
        features = rate_invariant_features(raw, _WORKER_CONFIG)
        if balance:
            selected_parts = []
            target = max(1, samples_per_block // 2)
            for label in (0, 1):
                candidates = np.flatnonzero(block_labels == label)
                count = min(target, len(candidates))
                if count:
                    selected_parts.append(
                        generator.choice(candidates, size=count, replace=False)
                    )
            selected = np.concatenate(selected_parts)
            generator.shuffle(selected)
        else:
            count = min(samples_per_block, len(block_labels))
            selected = generator.choice(len(block_labels), size=count, replace=False)
        feature_batches.append(features[selected])
        label_batches.append(block_labels[selected])
        group_batches.append(np.full(len(selected), group_id, dtype=np.int16))
        complete_events += chunk_size
    if not feature_batches:
        raise RuntimeError(f"Synthetic stream has no complete block: {path_text}")
    return (
        task_index,
        np.concatenate(feature_batches),
        np.concatenate(label_batches),
        np.concatenate(group_batches),
        len(events),
        complete_events,
        noise_count,
    )


def _cache_specification(
    config: StateConfig,
    train_files: Sequence[Path],
    validation_files: Sequence[Path],
    dataset_root: Path,
    args: argparse.Namespace,
) -> Dict:
    return {
        "schema_version": 1,
        "transform_version": TRANSFORM_VERSION,
        "state_config": config.to_dict(),
        "split_seed": args.split_seed,
        "synthetic_rates_hz": list(args.synthetic_rates_hz),
        "synthetic_duration_us": args.synthetic_duration_us,
        "max_synthetic_scenes": args.max_synthetic_scenes,
        "chunk_size": args.chunk_size,
        "samples_per_block": args.samples_per_block,
        "train_files": [str(path.relative_to(dataset_root)) for path in train_files],
        "validation_files": [
            str(path.relative_to(dataset_root)) for path in validation_files
        ],
    }


def _write_cache(path: Path, specification: Dict, arrays: Dict, metadata: Dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    descriptions = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        temporary = path / f"{filename}.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path / filename)
        descriptions[filename] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    manifest = {**specification, "arrays": descriptions, "metadata": metadata}
    temporary = path / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path / "manifest.json")


def _load_cache(path: Path, specification: Dict):
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    for key, expected in specification.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Synthetic feature cache mismatch for {key}")
    arrays = {}
    for filename, description in manifest["arrays"].items():
        array = np.load(path / filename, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != description["shape"] or str(array.dtype) != description["dtype"]:
            raise ValueError(f"Synthetic cached array mismatch: {filename}")
        arrays[filename[:-4]] = array
    return arrays, manifest["metadata"]


def extract_synthetic(
    files: Sequence[Path],
    rates: Sequence[float],
    config: StateConfig,
    args: argparse.Namespace,
    balance: bool,
    seed_offset: int,
):
    tasks = []
    clean_paths = sorted(
        {
            path.parent / f"{path.parent.name}_0.0.csv"
            for path in files
        }
    )
    if args.max_synthetic_scenes > 0:
        clean_paths = clean_paths[: args.max_synthetic_scenes]
    for file_index, clean_path in enumerate(clean_paths):
        for rate_index, rate in enumerate(rates):
            task_index = file_index * len(rates) + rate_index
            tasks.append(
                (
                    task_index,
                    str(clean_path),
                    float(rate),
                    rate_index + 1,
                    args.split_seed + seed_offset,
                    args.synthetic_duration_us,
                    args.chunk_size,
                    args.samples_per_block,
                    balance,
                )
            )
    feature_batches = []
    label_batches = []
    group_batches = []
    total_events = 0
    complete_events = 0
    injected_noise = 0
    started = time.perf_counter()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(config.to_dict(),),
    ) as executor:
        for completed, result in enumerate(
            executor.map(_synthetic_file_task, tasks, chunksize=1), start=1
        ):
            _, features, labels, groups, events, usable, noise = result
            feature_batches.append(features)
            label_batches.append(labels)
            group_batches.append(groups)
            total_events += events
            complete_events += usable
            injected_noise += noise
            if completed == 1 or completed % 25 == 0:
                print(
                    f"synthetic-features tasks={completed}/{len(tasks)} "
                    f"events={total_events:,}",
                    flush=True,
                )
    return (
        np.concatenate(feature_batches),
        np.concatenate(label_batches),
        np.concatenate(group_batches),
        {
            "tasks": len(tasks),
            "total_events": total_events,
            "complete_block_events": complete_events,
            "injected_noise_events": injected_noise,
            "sampled_events": int(sum(len(batch) for batch in label_batches)),
            "feature_seconds": time.perf_counter() - started,
            "workers": args.workers,
        },
    )


@torch.inference_mode()
def validation_metrics(model, features, labels, groups, device) -> Dict:
    probabilities = model.noise_probability(features, device)
    metrics = {}
    for group in np.unique(groups):
        selected = groups == group
        if np.unique(labels[selected]).size == 2:
            metrics[str(int(group))] = float(
                roc_auc_score(labels[selected], probabilities[selected])
            )
    if not metrics:
        raise RuntimeError("No validation group contains both classes")
    return {
        "selection_auc": float(np.mean(list(metrics.values()))),
        "minimum_group_auc": float(np.min(list(metrics.values()))),
        "group_auc": metrics,
    }


def train_epochs(
    model,
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    validation_groups,
    device,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    seed,
):
    counts = np.bincount(train_labels, minlength=2).astype(np.float64)
    weights = counts.sum() / np.maximum(2.0 * counts, 1.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = np.random.default_rng(seed)
    best = {"selection_auc": float("-inf"), "state_dict": None, "epoch": 0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = generator.permutation(len(train_labels))
        loss_sum = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_features = torch.from_numpy(train_features[indices]).to(device)
            batch_labels = torch.from_numpy(train_labels[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(indices)
        metrics = validation_metrics(
            model,
            validation_features,
            validation_labels,
            validation_groups,
            device,
        )
        record = {"epoch": epoch, "loss": loss_sum / len(train_labels), **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["selection_auc"] > best["selection_auc"]:
            best = {
                **metrics,
                "epoch": epoch,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
    model.load_state_dict(best["state_dict"], strict=True)
    return best, history


def main() -> None:
    args = parse_args()
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("workers must be positive and seeds must be unique")
    if any(rate <= 0 for rate in args.synthetic_rates_hz):
        raise ValueError("Synthetic rates must be positive")
    device = torch.device(args.device)
    _, config, _ = load_checkpoint(args.bootstrap_checkpoint)
    train_files, validation_files, train_scenes, validation_scenes = split_scene_files(
        args.dataset_root, 0.2, args.split_seed, 0
    )
    specification = _cache_specification(
        config, train_files, validation_files, args.dataset_root, args
    )
    cached = _load_cache(args.synthetic_feature_cache, specification)
    if cached is None:
        train_synthetic = extract_synthetic(
            train_files,
            args.synthetic_rates_hz,
            config,
            args,
            balance=True,
            seed_offset=100_000,
        )
        validation_synthetic = extract_synthetic(
            validation_files,
            args.synthetic_rates_hz,
            config,
            args,
            balance=False,
            seed_offset=200_000,
        )
        arrays = {
            "train_features": train_synthetic[0],
            "train_labels": train_synthetic[1],
            "train_groups": train_synthetic[2],
            "validation_features": validation_synthetic[0],
            "validation_labels": validation_synthetic[1],
            "validation_groups": validation_synthetic[2],
        }
        synthetic_metadata = {
            "train": train_synthetic[3],
            "validation": validation_synthetic[3],
        }
        _write_cache(
            args.synthetic_feature_cache,
            specification,
            arrays,
            synthetic_metadata,
        )
        print(f"WROTE SYNTHETIC CACHE {args.synthetic_feature_cache.resolve()}")
    else:
        arrays, synthetic_metadata = cached
        print(f"LOADED SYNTHETIC CACHE {args.synthetic_feature_cache.resolve()}")

    real_manifest = json.loads(
        (args.real_feature_cache / "manifest.json").read_text()
    )
    if real_manifest["advanced_config"]["state"] != config.to_dict():
        raise ValueError("Real feature cache state configuration differs")
    if real_manifest["train_files"] != specification["train_files"]:
        raise ValueError("Real feature cache training manifest differs")
    if real_manifest["validation_files"] != specification["validation_files"]:
        raise ValueError("Real feature cache validation manifest differs")
    real_train_raw = np.load(
        args.real_feature_cache / "train_features.npy", mmap_mode="r"
    )[:, : 2 * config.feature_dim]
    real_validation_raw = np.load(
        args.real_feature_cache / "validation_features.npy", mmap_mode="r"
    )[:, : 2 * config.feature_dim]
    real_train_features = rate_invariant_features(real_train_raw, config)
    real_validation_features = rate_invariant_features(real_validation_raw, config)
    real_train_labels = np.load(
        args.real_feature_cache / "train_labels.npy", mmap_mode="r"
    )
    real_validation_labels = np.load(
        args.real_feature_cache / "validation_labels.npy", mmap_mode="r"
    )

    train_features = np.concatenate((real_train_features, arrays["train_features"]))
    train_labels = np.concatenate((real_train_labels, arrays["train_labels"]))
    validation_features = np.concatenate(
        (real_validation_features, arrays["validation_features"])
    )
    validation_labels = np.concatenate(
        (real_validation_labels, arrays["validation_labels"])
    )
    validation_groups = np.concatenate(
        (
            np.zeros(len(real_validation_labels), dtype=np.int16),
            arrays["validation_groups"],
        )
    )
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_metadata = {
        "split_seed": args.split_seed,
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "real_feature_cache": str(args.real_feature_cache.resolve()),
        "synthetic_feature_cache": str(args.synthetic_feature_cache.resolve()),
        "synthetic_rates_hz": list(args.synthetic_rates_hz),
        "synthetic_metadata": synthetic_metadata,
        "train_samples": int(len(train_labels)),
        "validation_samples": int(len(validation_labels)),
        "validation_group_map": {
            "0": "real_ED24",
            **{
                str(index + 1): f"synthetic_{rate:g}Hz_per_pixel"
                for index, rate in enumerate(args.synthetic_rates_hz)
            },
        },
        "feature_names": list(robust_feature_names(config)),
    }
    summary = {"transform_version": TRANSFORM_VERSION, "runs": []}
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = AdvancedEventClassifier(robust_feature_dim(config), args.hidden_dims)
        model.set_normalizer(mean, scale)
        model.to(device)
        print(f"ROBUST FLOAT seed={seed}", flush=True)
        float_best, float_history = train_epochs(
            model,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            validation_groups,
            device,
            args.float_epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            seed,
        )
        float_path = args.output_dir / f"robust_seed{seed}_float.pth"
        save_robust_checkpoint(
            float_path,
            model,
            config,
            {
                **common_metadata,
                "seed": seed,
                "stage": "float",
                "best_epoch": float_best["epoch"],
                "best_validation_auc": float_best["selection_auc"],
                "best_validation_metrics": {
                    key: value for key, value in float_best.items() if key != "state_dict"
                },
                "history": float_history,
            },
        )
        model.calibrate_qat(train_features, device)
        model.enable_qat(12, 8, 8)
        print(f"ROBUST QAT seed={seed}", flush=True)
        qat_best, qat_history = train_epochs(
            model,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            validation_groups,
            device,
            args.qat_epochs,
            args.batch_size,
            args.learning_rate * 0.1,
            args.weight_decay,
            seed + 10_000,
        )
        qat_path = args.output_dir / f"robust_seed{seed}_qat.pth"
        save_robust_checkpoint(
            qat_path,
            model,
            config,
            {
                **common_metadata,
                "seed": seed,
                "stage": "qat",
                "float_validation_auc": float_best["selection_auc"],
                "best_epoch": qat_best["epoch"],
                "best_validation_auc": qat_best["selection_auc"],
                "best_validation_metrics": {
                    key: value for key, value in qat_best.items() if key != "state_dict"
                },
                "history": qat_history,
                "quantization": {
                    "input_bits": 12,
                    "weight_bits": 8,
                    "activation_bits": 8,
                    "scheme": "symmetric per-tensor STE fake quantization",
                },
            },
        )
        summary["runs"].append(
            {
                "seed": seed,
                "float_checkpoint": str(float_path.resolve()),
                "float_validation_auc": float_best["selection_auc"],
                "qat_checkpoint": str(qat_path.resolve()),
                "qat_validation_auc": qat_best["selection_auc"],
            }
        )
        temporary = args.output_dir / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
