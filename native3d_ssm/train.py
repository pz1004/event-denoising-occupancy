"""Train the tiny Native3D-SSM classifier on raw ED24 text streams."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .core import NativeStateBank, StateConfig, bidirectional_features, feature_names
from .data import iter_labeled_text
from .model import TinyEventClassifier, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--context-mode",
        choices=("causal", "bidirectional"),
        default="bidirectional",
        help="Bidirectional mode uses two native passes within each bounded chunk.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=10_240)
    parser.add_argument("--samples-per-chunk", type=int, default=512)
    parser.add_argument("--max-events-per-file", type=int, default=500_000)
    parser.add_argument(
        "--max-files",
        type=int,
        default=200,
        help="Maximum total noise-level CSV files; zero means all files.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument(
        "--taus-us", type=float, nargs="+", default=[250, 1000, 4000, 16000]
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def split_scene_files(
    root: Path, validation_fraction: float, seed: int, max_files: int
) -> Tuple[List[Path], List[Path], List[str], List[str]]:
    scene_directories = sorted(path for path in root.iterdir() if path.is_dir())
    if len(scene_directories) < 2:
        raise ValueError("ED24 root must contain at least two scene directories")
    generator = random.Random(seed)
    generator.shuffle(scene_directories)
    validation_count = max(1, round(len(scene_directories) * validation_fraction))
    validation_scenes = scene_directories[:validation_count]
    training_scenes = scene_directories[validation_count:]
    training_files = [path for scene in training_scenes for path in sorted(scene.glob("*.csv"))]
    validation_files = [
        path for scene in validation_scenes for path in sorted(scene.glob("*.csv"))
    ]
    generator.shuffle(training_files)
    generator.shuffle(validation_files)
    if max_files > 0:
        validation_limit = max(1, round(max_files * validation_fraction))
        training_limit = max(1, max_files - validation_limit)
        training_files = training_files[:training_limit]
        validation_files = validation_files[:validation_limit]
    return (
        training_files,
        validation_files,
        sorted(path.name for path in training_scenes),
        sorted(path.name for path in validation_scenes),
    )


def sample_streams(
    files: Sequence[Path],
    config: StateConfig,
    chunk_size: int,
    samples_per_chunk: int,
    max_events_per_file: int,
    seed: int,
    balance: bool,
    context_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    generator = np.random.default_rng(seed)
    feature_batches = []
    label_batches = []
    processed_events = 0
    started = time.perf_counter()

    for file_index, path in enumerate(files, start=1):
        state = NativeStateBank(config) if context_mode == "causal" else None
        file_events = 0
        for values in iter_labeled_text(path, chunk_size):
            if max_events_per_file > 0:
                remaining = max_events_per_file - file_events
                if remaining <= 0:
                    break
                values = values[:remaining]
            if not len(values):
                break
            features = (
                state.extract(values[:, :4])
                if context_mode == "causal"
                else bidirectional_features(values[:, :4], config)
            )
            labels = values[:, 4].astype(np.int64, copy=False)
            if balance:
                target_per_class = max(1, samples_per_chunk // 2)
                selected_parts = []
                for label in (0, 1):
                    candidates = np.flatnonzero(labels == label)
                    count = min(target_per_class, len(candidates))
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
                f"features: files={file_index}/{len(files)} events={processed_events:,}",
                flush=True,
            )

    if not feature_batches:
        raise RuntimeError("No events were sampled")
    features = np.concatenate(feature_batches)
    labels = np.concatenate(label_batches)
    metadata = {
        "files": len(files),
        "processed_events": processed_events,
        "sampled_events": int(len(labels)),
        "class_counts": np.bincount(labels, minlength=2).tolist(),
        "feature_seconds": time.perf_counter() - started,
    }
    return features, labels, metadata


def score(model: TinyEventClassifier, features: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    probabilities = model.noise_probability(features, device)
    return float(roc_auc_score(labels, probabilities))


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    state_config = StateConfig(
        width=args.width,
        height=args.height,
        taus_us=tuple(args.taus_us),
        radius=args.radius,
    )
    train_files, validation_files, train_scenes, validation_scenes = split_scene_files(
        args.dataset_root, args.validation_fraction, args.seed, args.max_files
    )
    train_features, train_labels, train_data = sample_streams(
        train_files,
        state_config,
        args.chunk_size,
        args.samples_per_chunk,
        args.max_events_per_file,
        args.seed,
        balance=True,
        context_mode=args.context_mode,
    )
    validation_features, validation_labels, validation_data = sample_streams(
        validation_files,
        state_config,
        args.chunk_size,
        args.samples_per_chunk,
        args.max_events_per_file,
        args.seed + 1,
        balance=False,
        context_mode=args.context_mode,
    )

    feature_dim = state_config.feature_dim * (2 if args.context_mode == "bidirectional" else 1)
    model = TinyEventClassifier(
        feature_dim, args.hidden_dim, context_mode=args.context_mode
    )
    model.set_normalizer(train_features.mean(axis=0), train_features.std(axis=0))
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    counts = np.bincount(train_labels, minlength=2).astype(np.float64)
    class_weights = counts.sum() / np.maximum(2.0 * counts, 1.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    generator = np.random.default_rng(args.seed)
    best_auc = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = generator.permutation(len(train_labels))
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            features = torch.from_numpy(train_features[indices]).to(device)
            labels = torch.from_numpy(train_labels[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(indices)
            seen += len(indices)
        validation_auc = score(model, validation_features, validation_labels, device)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "validation_auc": validation_auc,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_auc > best_auc:
            best_auc = validation_auc
            save_checkpoint(
                args.output,
                model,
                state_config,
                {
                    "seed": args.seed,
                    "best_epoch": epoch,
                    "best_validation_auc": best_auc,
                    "train_scenes": train_scenes,
                    "validation_scenes": validation_scenes,
                    "train_files": [
                        str(path.relative_to(args.dataset_root)) for path in train_files
                    ],
                    "validation_files": [
                        str(path.relative_to(args.dataset_root))
                        for path in validation_files
                    ],
                    "train_data": train_data,
                    "validation_data": validation_data,
                    "history": history,
                    "context_mode": args.context_mode,
                    "feature_names": [
                        f"{direction}_{name}"
                        for direction in (
                            ("past", "future")
                            if args.context_mode == "bidirectional"
                            else ("past",)
                        )
                        for name in feature_names(
                            state_config.taus_us, state_config.radius
                        )
                    ],
                },
            )

    print(
        json.dumps(
            {
                "checkpoint": str(args.output.resolve()),
                "best_validation_auc": best_auc,
                "parameters": model.trainable_parameters,
                "state_bytes": NativeStateBank(state_config).state_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
