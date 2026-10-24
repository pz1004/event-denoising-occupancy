"""Train motion/split-state Native3D-SSM models and QAT variants on ED24."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .advanced import (
    AdvancedConfig,
    MotionConfig,
    advanced_block_features,
    advanced_feature_names,
)
from .advanced_model import (
    AdvancedEventClassifier,
    load_advanced_checkpoint,
    save_advanced_checkpoint,
)
from .core import bidirectional_features
from .data import iter_labeled_text
from .model import load_checkpoint
from .train import split_scene_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Persist or reuse the exact sampled advanced feature matrices.",
    )
    parser.add_argument(
        "--resume-float-checkpoint",
        type=Path,
        default=None,
        help="Reuse a matching float checkpoint and run only QAT after feature extraction.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--split-seed", type=int, default=1024)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-events-per-file", type=int, default=250_000)
    parser.add_argument("--chunk-size", type=int, default=10_240)
    parser.add_argument("--samples-per-chunk", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--float-epochs", type=int, default=15)
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--motion-delays-us",
        type=float,
        nargs="+",
        default=[250, 500, 1000, 2000, 4000, 8000],
    )
    parser.add_argument("--motion-strides-px", type=int, nargs="+", default=[1])
    parser.add_argument("--motion-steps", type=int, default=3)
    parser.add_argument("--qat-input-bits", type=int, default=12)
    parser.add_argument("--qat-weight-bits", type=int, default=8)
    parser.add_argument("--qat-activation-bits", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent CPU workers for per-file feature extraction.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_feature_cache(
    cache_dir: Path,
    specification: Dict,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    train_data: Dict,
    validation_data: Dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "train_features.npy": train_features,
        "train_labels.npy": train_labels,
        "validation_features.npy": validation_features,
        "validation_labels.npy": validation_labels,
    }
    for name, array in arrays.items():
        temporary = cache_dir / f"{name}.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(cache_dir / name)
    manifest = {
        **specification,
        "train_data": train_data,
        "validation_data": validation_data,
        "arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in arrays.items()
        },
    }
    temporary = cache_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(cache_dir / "manifest.json")


def load_feature_cache(cache_dir: Path, specification: Dict):
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    for key, expected in specification.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Feature cache mismatch for {key}")
    array_names = (
        "train_features.npy",
        "train_labels.npy",
        "validation_features.npy",
        "validation_labels.npy",
    )
    arrays = [
        np.load(cache_dir / name, mmap_mode="r", allow_pickle=False)
        for name in array_names
    ]
    for name, array in zip(array_names, arrays):
        if name not in manifest["arrays"]:
            raise ValueError(f"Cached array is absent from manifest: {name}")
        expected = manifest["arrays"][name]
        if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
            raise ValueError(f"Cached array metadata mismatch: {name}")
    return (*arrays, manifest["train_data"], manifest["validation_data"])


_WORKER_BOOTSTRAP = None
_WORKER_CONFIG = None


def _initialize_feature_worker(bootstrap_path: str, config_values: Dict) -> None:
    global _WORKER_BOOTSTRAP, _WORKER_CONFIG
    torch.set_num_threads(1)
    _WORKER_BOOTSTRAP, _, _ = load_checkpoint(Path(bootstrap_path))
    _WORKER_BOOTSTRAP.eval()
    _WORKER_CONFIG = AdvancedConfig.from_dict(config_values)


def _sample_one_file(task):
    (
        file_index,
        path_text,
        chunk_size,
        samples_per_chunk,
        max_events_per_file,
        seed,
        balance,
    ) = task
    generator = np.random.default_rng(seed + file_index * 1_000_003)
    feature_batches = []
    label_batches = []
    processed_events = 0
    for values in iter_labeled_text(Path(path_text), chunk_size):
        if max_events_per_file > 0:
            remaining = max_events_per_file - processed_events
            if remaining <= 0:
                break
            values = values[:remaining]
        if not len(values):
            break
        events = values[:, :4]
        raw = bidirectional_features(events, _WORKER_CONFIG.state)
        bootstrap_probability = _WORKER_BOOTSTRAP.noise_probability(
            raw, torch.device("cpu")
        )
        features = advanced_block_features(
            events,
            bootstrap_probability,
            _WORKER_CONFIG,
            raw_features=raw,
        )
        labels = values[:, 4].astype(np.int64, copy=False)
        if balance:
            selected_parts = []
            target = max(1, samples_per_chunk // 2)
            for label in (0, 1):
                candidates = np.flatnonzero(labels == label)
                count = min(target, len(candidates))
                if count:
                    selected_parts.append(
                        generator.choice(candidates, size=count, replace=False)
                    )
            selected = np.concatenate(selected_parts)
            generator.shuffle(selected)
        else:
            count = min(samples_per_chunk, len(labels))
            selected = generator.choice(len(labels), size=count, replace=False)
        feature_batches.append(features[selected])
        label_batches.append(labels[selected])
        processed_events += len(values)
        if max_events_per_file > 0 and processed_events >= max_events_per_file:
            break
    return (
        file_index,
        np.concatenate(feature_batches),
        np.concatenate(label_batches),
        processed_events,
    )


def sample_advanced_streams(
    files: Sequence[Path],
    bootstrap,
    config: AdvancedConfig,
    device: torch.device,
    chunk_size: int,
    samples_per_chunk: int,
    max_events_per_file: int,
    seed: int,
    balance: bool,
    workers: int = 1,
    bootstrap_checkpoint: Path = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > 1:
        if device.type != "cpu":
            raise ValueError("Parallel feature extraction currently requires CPU")
        if bootstrap_checkpoint is None:
            raise ValueError("Parallel extraction requires the bootstrap checkpoint path")
        tasks = [
            (
                file_index,
                str(path),
                chunk_size,
                samples_per_chunk,
                max_events_per_file,
                seed,
                balance,
            )
            for file_index, path in enumerate(files)
        ]
        feature_batches = []
        label_batches = []
        processed_events = 0
        started = time.perf_counter()
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_feature_worker,
            initargs=(str(bootstrap_checkpoint), config.to_dict()),
        ) as executor:
            for completed, result in enumerate(
                executor.map(_sample_one_file, tasks, chunksize=1), start=1
            ):
                _, features, labels, file_events = result
                feature_batches.append(features)
                label_batches.append(labels)
                processed_events += file_events
                if completed == 1 or completed % 25 == 0:
                    print(
                        f"advanced-features files={completed}/{len(files)} "
                        f"events={processed_events:,}",
                        flush=True,
                    )
        features = np.concatenate(feature_batches)
        labels = np.concatenate(label_batches)
        return features, labels, {
            "files": len(files),
            "processed_events": processed_events,
            "sampled_events": int(len(labels)),
            "class_counts": np.bincount(labels, minlength=2).tolist(),
            "feature_seconds": time.perf_counter() - started,
            "workers": workers,
            "sampling": "deterministic per-file RNG",
        }

    generator = np.random.default_rng(seed)
    feature_batches = []
    label_batches = []
    processed_events = 0
    started = time.perf_counter()
    for file_index, path in enumerate(files, start=1):
        file_events = 0
        for values in iter_labeled_text(path, chunk_size):
            if max_events_per_file > 0:
                remaining = max_events_per_file - file_events
                if remaining <= 0:
                    break
                values = values[:remaining]
            if not len(values):
                break
            events = values[:, :4]
            raw = bidirectional_features(events, config.state)
            bootstrap_probability = bootstrap.noise_probability(raw, device)
            features = advanced_block_features(
                events, bootstrap_probability, config, raw_features=raw
            )
            labels = values[:, 4].astype(np.int64, copy=False)
            if balance:
                selected_parts = []
                target = max(1, samples_per_chunk // 2)
                for label in (0, 1):
                    candidates = np.flatnonzero(labels == label)
                    count = min(target, len(candidates))
                    if count:
                        selected_parts.append(
                            generator.choice(candidates, size=count, replace=False)
                        )
                selected = np.concatenate(selected_parts)
                generator.shuffle(selected)
            else:
                count = min(samples_per_chunk, len(labels))
                selected = generator.choice(len(labels), size=count, replace=False)
            feature_batches.append(features[selected])
            label_batches.append(labels[selected])
            file_events += len(values)
            processed_events += len(values)
            if max_events_per_file > 0 and file_events >= max_events_per_file:
                break
        if file_index == 1 or file_index % 25 == 0:
            print(
                f"advanced-features files={file_index}/{len(files)} "
                f"events={processed_events:,}",
                flush=True,
            )
    if not feature_batches:
        raise RuntimeError("No advanced features were sampled")
    features = np.concatenate(feature_batches)
    labels = np.concatenate(label_batches)
    return features, labels, {
        "files": len(files),
        "processed_events": processed_events,
        "sampled_events": int(len(labels)),
        "class_counts": np.bincount(labels, minlength=2).tolist(),
        "feature_seconds": time.perf_counter() - started,
        "workers": 1,
        "sampling": "single-stream RNG",
    }


def auc(model, features, labels, device) -> float:
    if np.unique(labels).size != 2:
        raise ValueError("ROC AUC validation requires both signal and noise labels")
    return float(roc_auc_score(labels, model.noise_probability(features, device)))


def train_epochs(
    model: AdvancedEventClassifier,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Tuple[Dict, List[Dict]]:
    counts = np.bincount(train_labels, minlength=2).astype(np.float64)
    weights = counts.sum() / np.maximum(2.0 * counts, 1.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = np.random.default_rng(seed)
    best = {"auc": float("-inf"), "epoch": 0, "state_dict": None}
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
        validation_auc = auc(
            model, validation_features, validation_labels, device
        )
        record = {
            "epoch": epoch,
            "loss": loss_sum / len(train_labels),
            "validation_auc": validation_auc,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_auc > best["auc"]:
            best = {
                "auc": validation_auc,
                "epoch": epoch,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
    model.load_state_dict(best["state_dict"], strict=True)
    return best, history


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be unique")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.resume_float_checkpoint is not None and len(args.seeds) != 1:
        raise ValueError("QAT-only resume requires exactly one seed")
    device = torch.device(args.device)
    bootstrap, state_config, bootstrap_metadata = load_checkpoint(
        args.bootstrap_checkpoint
    )
    if bootstrap.context_mode != "bidirectional":
        raise ValueError("The advanced extractor requires a bidirectional bootstrap")
    bootstrap.to(device).eval()
    config = AdvancedConfig(
        state=state_config,
        motion=MotionConfig(
            delays_us=tuple(args.motion_delays_us),
            spatial_strides=tuple(args.motion_strides_px),
            spatial_steps=args.motion_steps,
        ),
        gate_threshold=args.gate_threshold,
    )
    train_files, validation_files, train_scenes, validation_scenes = split_scene_files(
        args.dataset_root,
        args.validation_fraction,
        args.split_seed,
        args.max_files,
    )
    bootstrap_digest = sha256(args.bootstrap_checkpoint)
    cache_specification = {
        "schema_version": 1,
        "advanced_config": config.to_dict(),
        "bootstrap_sha256": bootstrap_digest,
        "split_seed": args.split_seed,
        "chunk_size": args.chunk_size,
        "samples_per_chunk": args.samples_per_chunk,
        "max_events_per_file": args.max_events_per_file,
        "sampling_schema": (
            "per_file_rng_v2" if args.workers > 1 else "single_stream_rng_v1"
        ),
        "train_files": [str(path.relative_to(args.dataset_root)) for path in train_files],
        "validation_files": [
            str(path.relative_to(args.dataset_root)) for path in validation_files
        ],
    }
    cached = (
        load_feature_cache(args.feature_cache, cache_specification)
        if args.feature_cache is not None
        else None
    )
    if cached is None:
        train_features, train_labels, train_data = sample_advanced_streams(
            train_files,
            bootstrap,
            config,
            device,
            args.chunk_size,
            args.samples_per_chunk,
            args.max_events_per_file,
            args.split_seed,
            balance=True,
            workers=args.workers,
            bootstrap_checkpoint=args.bootstrap_checkpoint,
        )
        validation_features, validation_labels, validation_data = sample_advanced_streams(
            validation_files,
            bootstrap,
            config,
            device,
            args.chunk_size,
            args.samples_per_chunk,
            args.max_events_per_file,
            args.split_seed + 1,
            balance=False,
            workers=args.workers,
            bootstrap_checkpoint=args.bootstrap_checkpoint,
        )
        if args.feature_cache is not None:
            write_feature_cache(
                args.feature_cache,
                cache_specification,
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                train_data,
                validation_data,
            )
            print(f"WROTE FEATURE CACHE {args.feature_cache.resolve()}", flush=True)
    else:
        (
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            train_data,
            validation_data,
        ) = cached
        print(f"LOADED FEATURE CACHE {args.feature_cache.resolve()}", flush=True)
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_metadata = {
        "split_seed": args.split_seed,
        "bootstrap_checkpoint": str(args.bootstrap_checkpoint.resolve()),
        "bootstrap_sha256": bootstrap_digest,
        "bootstrap_validation_auc": bootstrap_metadata.get("best_validation_auc"),
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "train_files": [str(path.relative_to(args.dataset_root)) for path in train_files],
        "validation_files": [
            str(path.relative_to(args.dataset_root)) for path in validation_files
        ],
        "train_data": train_data,
        "validation_data": validation_data,
        "feature_names": list(advanced_feature_names(config)),
    }
    summary = {"config": config.to_dict(), "runs": []}

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if args.resume_float_checkpoint is not None:
            model, resumed_config, resumed_metadata = load_advanced_checkpoint(
                args.resume_float_checkpoint
            )
            if resumed_config != config:
                raise ValueError("Resumed float checkpoint configuration differs")
            if resumed_metadata.get("seed") != seed:
                raise ValueError("Resumed float checkpoint seed differs")
            if model.is_qat_enabled:
                raise ValueError("Resumed checkpoint is already QAT-enabled")
            model.to(device)
            float_best = {
                "auc": float(resumed_metadata["best_validation_auc"]),
                "epoch": int(resumed_metadata["best_epoch"]),
            }
            float_history = list(resumed_metadata.get("history", []))
            float_path = args.resume_float_checkpoint
            print(f"RESUME FLOAT seed={seed} auc={float_best['auc']:.6f}", flush=True)
        else:
            model = AdvancedEventClassifier(config.feature_dim, args.hidden_dims)
            model.set_normalizer(mean, scale)
            model.to(device)
            print(f"FLOAT seed={seed}", flush=True)
            float_best, float_history = train_epochs(
                model,
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                device,
                args.float_epochs,
                args.batch_size,
                args.learning_rate,
                args.weight_decay,
                seed,
            )
            float_path = args.output_dir / f"advanced_seed{seed}_float.pth"
            float_metadata = {
                **common_metadata,
                "seed": seed,
                "stage": "float",
                "best_epoch": float_best["epoch"],
                "best_validation_auc": float_best["auc"],
                "history": float_history,
            }
            save_advanced_checkpoint(float_path, model, config, float_metadata)

        model.calibrate_qat(train_features, device)
        model.enable_qat(
            args.qat_input_bits,
            args.qat_weight_bits,
            args.qat_activation_bits,
        )
        print(f"QAT seed={seed}", flush=True)
        qat_best, qat_history = train_epochs(
            model,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            device,
            args.qat_epochs,
            args.batch_size,
            args.learning_rate * 0.1,
            args.weight_decay,
            seed + 10_000,
        )
        qat_path = args.output_dir / f"advanced_seed{seed}_qat.pth"
        qat_metadata = {
            **common_metadata,
            "seed": seed,
            "stage": "qat",
            "float_validation_auc": float_best["auc"],
            "best_epoch": qat_best["epoch"],
            "best_validation_auc": qat_best["auc"],
            "history": qat_history,
            "quantization": {
                "input_bits": args.qat_input_bits,
                "weight_bits": args.qat_weight_bits,
                "activation_bits": args.qat_activation_bits,
                "scheme": "symmetric per-tensor STE fake quantization",
            },
        }
        save_advanced_checkpoint(qat_path, model, config, qat_metadata)
        summary["runs"].append(
            {
                "seed": seed,
                "float_checkpoint": str(float_path.resolve()),
                "float_validation_auc": float_best["auc"],
                "qat_checkpoint": str(qat_path.resolve()),
                "qat_validation_auc": qat_best["auc"],
            }
        )
        temporary = args.output_dir / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
