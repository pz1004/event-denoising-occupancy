#!/usr/bin/env python3
"""Matched DVSCLEAN evaluation for EDmamba and the Native3D contrast.

The released EDmamba DVSCLEAN loader names the ten paper test streams but does
not load the distributed HDF5 layout.  This runner streams those HDF5 files
directly, splits each stream into consecutive complete 10,240-event blocks, and
applies the same EDmamba transforms used by the locally reproduced DND21 path.

DVSCLEAN stores a continuous pre-conversion ``polarity`` field, whereas the
released EDmamba NPZ loader expects binary polarity before applying ``2*p-1``.
Both methods therefore use the same declared, label-blind adapter ``p >= 0.5``.
Labels are used only for evaluation, with one denoting injected noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDMAMBA_ROOT = PROJECT_ROOT / "EDmamba"
POINTOPS_BUILD = EDMAMBA_ROOT / "libs/pointops/build/lib.linux-x86_64-cpython-38"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EDMAMBA_ROOT))
sys.path.insert(0, str(POINTOPS_BUILD))

from native3d_ssm.analytic import analytic_contrast_score
from native3d_ssm.core import StateConfig, bidirectional_features
from reproduction.reproduce_edmamba_dnd21 import (
    build_checkpoint_model,
    seed_everything,
    write_results,
)


# The aliases are verified from label-0 signal-event accumulation images and
# ordered to match the paper table's five column groups. Stems remain canonical.
SEQUENCES = {
    "MAH00446": "double-bracket",
    "MAH00456": "double-ship",
    "MAH00454": "double-airplane",
    "MAH00455": "multi-helicopter",
    "MAH00465": "multi-car",
}

PAPER_AUC = {
    "MAH00446": {50: 0.9457, 100: 0.9248},
    "MAH00456": {50: 0.9684, 100: 0.9572},
    "MAH00454": {50: 0.9684, 100: 0.8704},
    "MAH00455": {50: 0.9247, 100: 0.9122},
    "MAH00465": {50: 0.9218, 100: 0.9036},
}

CANDIDATES = ("native_float", "native_q12_8", "native_q8_4")
TAU_DIAGNOSTICS = ("native_tau1000us", "native_tau4000us", "native_tau16000us")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "DVSCLEAN/simulated_data",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=EDMAMBA_ROOT / "exp/ED24/EDmamba/model/model_best.pth",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EDMAMBA_ROOT / "configs/ED24/EDmamba.py",
    )
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--noise-levels", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument(
        "--edmamba-coordinate-mode",
        choices=("raw-1280x720", "scale-to-ed24"),
        default="raw-1280x720",
        help="Protocol sensitivity for the missing HDF5-to-NPZ converter.",
    )
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--cluster-blocks", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reproduction/results/dvsclean_paired.json",
    )
    parser.add_argument(
        "--hard-exit",
        action="store_true",
        help="Bypass CUDA destructor cleanup for the legacy EDmamba environment.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_hdf5_full_blocks(
    path: Path, block_size: int, max_blocks: Optional[int] = None
) -> Iterable[Dict[str, np.ndarray]]:
    """Yield consecutive complete blocks without materializing a full stream."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    with h5py.File(path, "r") as handle:
        if "events" not in handle:
            raise ValueError(f"Missing events group in {path}")
        group = handle["events"]
        required = ("timestamp", "x", "y", "polarity", "label")
        missing = [key for key in required if key not in group]
        if missing:
            raise ValueError(f"Missing DVSCLEAN arrays {missing} in {path}")
        lengths = {key: len(group[key]) for key in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Mismatched DVSCLEAN array lengths in {path}: {lengths}")
        block_count = lengths["label"] // block_size
        if max_blocks is not None:
            block_count = min(block_count, max_blocks)
        for block_index in range(block_count):
            start = block_index * block_size
            stop = start + block_size
            block = {key: np.asarray(group[key][start:stop]) for key in required}
            timestamps = block["timestamp"].astype(np.float64, copy=False)
            if np.any(np.diff(timestamps) < 0):
                raise ValueError(f"Timestamp regression in {path} block {block_index}")
            labels = block["label"]
            if np.any((labels != 0) & (labels != 1)):
                raise ValueError(f"Nonbinary label in {path} block {block_index}")
            yield block


def binary_polarity(values: np.ndarray) -> np.ndarray:
    """Apply the shared label-blind raw-HDF5 polarity conversion."""

    return (np.asarray(values, dtype=np.float32) >= 0.5).astype(np.float32)


def edmamba_coordinates(
    x: np.ndarray, y: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one explicit spatial adapter for the missing EDmamba converter."""

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if mode == "raw-1280x720":
        return x, y
    if mode == "scale-to-ed24":
        return x * (346.0 / 1280.0), y * (260.0 / 720.0)
    raise ValueError(f"Unknown EDmamba coordinate mode: {mode}")


def prepare_edmamba_block(
    block: Dict[str, np.ndarray], coordinate_mode: str = "raw-1280x720"
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Apply EDmamba's first released test transform to one raw HDF5 block."""

    from pointcept.datasets.transform import GridSample_V3

    timestamps = block["timestamp"].astype(np.float64, copy=False)
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0:
        raise ValueError("A block must span a positive timestamp interval")
    normalized_time = ((timestamps - timestamps[0]) / duration).astype(np.float32)
    x, y = edmamba_coordinates(block["x"], block["y"], coordinate_mode)
    coord = np.column_stack((x, y, normalized_time)).astype(np.float32)
    strength = (binary_polarity(block["polarity"]) * 2.0 - 1.0).reshape(-1, 1)
    labels = block["label"].astype(np.int64, copy=False)

    data = {"coord": coord, "strength": strength, "segment": labels.copy()}
    sampler = GridSample_V3(
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "strength", "segment"),
        return_inverse=True,
    )
    data = sampler(data)
    return data, data["inverse"].copy(), data["segment"].copy()


def score_edmamba_block(
    model: torch.nn.Module,
    block: Dict[str, np.ndarray],
    coordinate_mode: str = "raw-1280x720",
) -> Tuple[np.ndarray, int, int, float]:
    from pointcept.datasets.transform import Collect, GridSample_V3, ToTensor
    from pointcept.datasets.utils import collate_fn

    data, inverse, sampled_labels = prepare_edmamba_block(block, coordinate_mode)
    voxelizer = GridSample_V3(
        grid_size=0.1,
        hash_type="fnv",
        mode="test",
        keys=("coord", "strength"),
        return_grid_coord=True,
    )
    fragments = voxelizer(data)
    to_tensor = ToTensor()
    collect = Collect(
        keys=("coord", "grid_coord", "index"),
        feat_keys=("coord", "strength"),
    )
    probabilities = torch.zeros((sampled_labels.size, 2), device="cuda")
    counts = torch.zeros((sampled_labels.size, 1), device="cuda")
    inference_seconds = 0.0
    for fragment in fragments:
        input_dict = collate_fn([collect(to_tensor(fragment))])
        input_dict = {
            key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in input_dict.items()
        }
        index = input_dict["index"]
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            probability = F.softmax(model(input_dict)["seg_logits"], dim=-1)
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        probabilities[index] += probability
        counts[index] += 1
    if torch.any(counts == 0):
        raise RuntimeError("EDmamba test fragments did not cover every sampled event")
    sampled_scores = (probabilities / counts)[:, 1].cpu().numpy()
    return sampled_scores[inverse], int(sampled_labels.size), len(fragments), inference_seconds


def prepare_native_events(block: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert one block to Native3D's [microseconds, x, y, binary p] contract."""

    timestamp_seconds = block["timestamp"].astype(np.float64, copy=False)
    timestamp_us = np.rint((timestamp_seconds - timestamp_seconds[0]) * 1_000_000.0)
    timestamp_us = timestamp_us.astype(np.int64)
    polarity = binary_polarity(block["polarity"]).astype(np.int64)
    events = np.column_stack((timestamp_us, block["x"], block["y"], polarity))
    return np.ascontiguousarray(events, dtype=np.int64)


def score_native_block(
    block: Dict[str, np.ndarray], config: StateConfig
) -> Dict[str, np.ndarray]:
    raw = bidirectional_features(prepare_native_events(block), config)
    scores = {
        "native_float": analytic_contrast_score(raw, config, "float"),
        "native_q12_8": analytic_contrast_score(raw, config, "q12.8"),
        "native_q8_4": analytic_contrast_score(raw, config, "q8.4"),
    }
    scale_count = len(config.taus_us)
    future_offset = config.feature_dim
    for tau, name in zip(config.taus_us[1:], TAU_DIAGNOSTICS):
        scale_index = config.taus_us.index(tau)
        ring_past = raw[:, 2 * scale_count + scale_index]
        global_past = raw[:, 6 * scale_count + scale_index]
        ring_future = raw[:, future_offset + 2 * scale_count + scale_index]
        global_future = raw[:, future_offset + 6 * scale_count + scale_index]
        scores[name] = np.asarray(
            global_past + global_future - ring_past - ring_future,
            dtype=np.float32,
        )
    return scores


def evaluate_stream(
    model: torch.nn.Module,
    path: Path,
    config: StateConfig,
    block_size: int,
    max_blocks: Optional[int],
    edmamba_coordinate_mode: str,
) -> Dict:
    method_names = CANDIDATES + TAU_DIAGNOSTICS
    block_aucs = {"edmamba": [], **{name: [] for name in method_names}}
    all_labels = []
    all_scores = {name: [] for name in block_aucs}
    sampled_events = 0
    fragments = 0
    edmamba_inference_seconds = 0.0
    edmamba_wall_seconds = 0.0
    native_wall_seconds = 0.0
    source_events = 0
    block_durations_seconds = []

    with h5py.File(path, "r") as handle:
        total_events = int(len(handle["events/label"]))
        source_labels = handle["events/label"][:]
        source_polarity = handle["events/polarity"][:]
        signal_events = int(np.count_nonzero(source_labels == 0))
        noise_events = total_events - signal_events
        timestamp_start = float(handle["events/timestamp"][0])
        timestamp_end = float(handle["events/timestamp"][-1])
        duration_seconds = timestamp_end - timestamp_start
        binary_source_polarity = binary_polarity(source_polarity)
        positive_polarity_fraction = float(np.mean(binary_source_polarity))
        positive_polarity_fraction_signal = float(
            np.mean(binary_source_polarity[source_labels == 0])
        )
        positive_polarity_fraction_noise = float(
            np.mean(binary_source_polarity[source_labels == 1])
        )

    for block_index, block in enumerate(
        iter_hdf5_full_blocks(path, block_size, max_blocks), start=1
    ):
        labels = block["label"].astype(np.int64, copy=False)
        block_durations_seconds.append(
            float(block["timestamp"][-1] - block["timestamp"][0])
        )
        edmamba_started = time.perf_counter()
        edmamba_scores, sampled, part_count, inference = score_edmamba_block(
            model, block, edmamba_coordinate_mode
        )
        edmamba_wall_seconds += time.perf_counter() - edmamba_started
        native_started = time.perf_counter()
        native_scores = score_native_block(block, config)
        native_wall_seconds += time.perf_counter() - native_started

        scores = {"edmamba": edmamba_scores, **native_scores}
        for name, values in scores.items():
            block_aucs[name].append(float(roc_auc_score(labels, values)))
            all_scores[name].append(np.asarray(values, dtype=np.float32))
        all_labels.append(labels.copy())
        source_events += len(labels)
        sampled_events += sampled
        fragments += part_count
        edmamba_inference_seconds += inference
        if block_index == 1 or block_index % 10 == 0:
            print(
                f"{path.name}: blocks={block_index} "
                f"edmamba={np.mean(block_aucs['edmamba']):.6f} "
                f"native={np.mean(block_aucs['native_float']):.6f}",
                flush=True,
            )

    if not block_aucs["edmamba"]:
        raise RuntimeError(f"No complete {block_size}-event blocks found in {path}")
    labels = np.concatenate(all_labels)
    return {
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "total_source_events": total_events,
        "signal_events": signal_events,
        "noise_events": noise_events,
        "duration_seconds": duration_seconds,
        "event_rate_hz": total_events / duration_seconds,
        "event_rate_per_pixel_hz": total_events / duration_seconds / (1280 * 720),
        "binary_positive_polarity_fraction": positive_polarity_fraction,
        "binary_positive_polarity_fraction_signal": positive_polarity_fraction_signal,
        "binary_positive_polarity_fraction_noise": positive_polarity_fraction_noise,
        "evaluated_blocks": len(block_aucs["edmamba"]),
        "evaluated_events": source_events,
        "dropped_tail_events": total_events - source_events,
        "block_duration_seconds": {
            "mean": float(np.mean(block_durations_seconds)),
            "median": float(np.median(block_durations_seconds)),
            "minimum": float(np.min(block_durations_seconds)),
            "maximum": float(np.max(block_durations_seconds)),
        },
        "edmamba_sampled_events": sampled_events,
        "edmamba_fragments": fragments,
        "timing": {
            "edmamba_gpu_inference_seconds": edmamba_inference_seconds,
            "edmamba_block_wall_seconds": edmamba_wall_seconds,
            "native_block_wall_seconds": native_wall_seconds,
        },
        "methods": {
            name: {
                "mean_block_auc": float(np.mean(values)),
                "std_block_auc": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                "event_global_auc": float(
                    roc_auc_score(labels, np.concatenate(all_scores[name]))
                ),
                "block_aucs": values,
            }
            for name, values in block_aucs.items()
        },
    }


def paired_cluster_bootstrap(
    condition_differences,
    seed: int,
    replicates: int,
    cluster_blocks: int,
) -> Dict:
    """Condition-stratified paired bootstrap over contiguous block clusters."""

    if replicates <= 0 or cluster_blocks <= 0:
        raise ValueError("Bootstrap replicates and cluster size must be positive")
    observed = float(np.mean([np.mean(values) for values in condition_differences]))
    clusters_by_condition = []
    for values in condition_differences:
        values = np.asarray(values, dtype=np.float64)
        clusters = []
        for start in range(0, len(values), cluster_blocks):
            chunk = values[start : start + cluster_blocks]
            clusters.append((float(chunk.sum()), len(chunk)))
        clusters_by_condition.append(clusters)

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        condition_means = []
        for clusters in clusters_by_condition:
            indices = rng.integers(0, len(clusters), size=len(clusters))
            total = sum(clusters[index][0] for index in indices)
            count = sum(clusters[index][1] for index in indices)
            condition_means.append(total / count)
        bootstrap[replicate] = np.mean(condition_means)
    return {
        "observed_paired_mean_delta": observed,
        "95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "replicates": replicates,
        "cluster_blocks": cluster_blocks,
        "cluster_events": cluster_blocks * 10_240,
        "stratification": "unweighted mean of condition-level paired deltas",
    }


def summarize(
    results: Dict,
    seed: int,
    bootstrap_replicates: int,
    cluster_blocks: int,
) -> Dict:
    conditions = [
        stream
        for sequence_results in results.values()
        for stream in sequence_results.values()
    ]
    paper_values = [stream["paper_edmamba_auc"] for stream in conditions]
    edmamba_values = [stream["methods"]["edmamba"]["mean_block_auc"] for stream in conditions]
    summary = {
        "conditions": len(conditions),
        "paper_edmamba_mean_auc": float(np.mean(paper_values)),
        "reproduced_edmamba_mean_auc": float(np.mean(edmamba_values)),
        "reproduced_edmamba_delta_vs_paper": float(
            np.mean(edmamba_values) - np.mean(paper_values)
        ),
        "candidates": {},
        "posthoc_tau_sweep_diagnostic": {},
    }
    for name in CANDIDATES:
        values = [stream["methods"][name]["mean_block_auc"] for stream in conditions]
        paired = []
        for stream in conditions:
            candidate_blocks = np.asarray(stream["methods"][name]["block_aucs"])
            edmamba_blocks = np.asarray(stream["methods"]["edmamba"]["block_aucs"])
            if candidate_blocks.shape != edmamba_blocks.shape:
                raise ValueError("Paired DVSCLEAN block arrays have different shapes")
            paired.append(candidate_blocks - edmamba_blocks)
        mean_auc = float(np.mean(values))
        summary["candidates"][name] = {
            "mean_auc": mean_auc,
            "mean_oriented_auc_diagnostic": float(
                np.mean([max(value, 1.0 - value) for value in values])
            ),
            "delta_vs_reproduced_edmamba": mean_auc - float(np.mean(edmamba_values)),
            "conditions_beating_reproduced_edmamba": int(
                sum(value > baseline for value, baseline in zip(values, edmamba_values))
            ),
            "paired_cluster_bootstrap": paired_cluster_bootstrap(
                paired,
                seed=seed,
                replicates=bootstrap_replicates,
                cluster_blocks=cluster_blocks,
            ),
        }
    for name in TAU_DIAGNOSTICS:
        values = [stream["methods"][name]["mean_block_auc"] for stream in conditions]
        paired = []
        for stream in conditions:
            candidate_blocks = np.asarray(stream["methods"][name]["block_aucs"])
            edmamba_blocks = np.asarray(stream["methods"]["edmamba"]["block_aucs"])
            paired.append(candidate_blocks - edmamba_blocks)
        mean_auc = float(np.mean(values))
        summary["posthoc_tau_sweep_diagnostic"][name] = {
            "mean_auc_frozen_direction": mean_auc,
            "mean_oriented_auc": float(
                np.mean([max(value, 1.0 - value) for value in values])
            ),
            "delta_vs_reproduced_edmamba": mean_auc - float(np.mean(edmamba_values)),
            "conditions_beating_reproduced_edmamba": int(
                sum(value > baseline for value, baseline in zip(values, edmamba_values))
            ),
            "paired_cluster_bootstrap": paired_cluster_bootstrap(
                paired,
                seed=seed,
                replicates=bootstrap_replicates,
                cluster_blocks=cluster_blocks,
            ),
            "status": "post-hoc diagnostic; not a test-set-selected primary result",
        }
    return summary


def validate_args(args: argparse.Namespace) -> None:
    unknown = sorted(set(args.sequences) - set(SEQUENCES))
    if unknown:
        raise ValueError(f"Unknown test sequence(s): {unknown}")
    if any(level not in (50, 100) for level in args.noise_levels):
        raise ValueError("DVSCLEAN paper noise levels are 50 and 100")
    if args.max_blocks is not None and args.max_blocks <= 0:
        raise ValueError("max_blocks must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, checkpoint_metadata = build_checkpoint_model(args.config, args.checkpoint)
    native_config = StateConfig(width=1280, height=720)
    payload = {
        "benchmark": "DVSCLEAN labeled simulated paper subset",
        "protocol": {
            "sequence_mapping_provenance": (
                "label-0 signal-event accumulation images matched to paper object-composition names"
            ),
            "block_size": args.block_size,
            "tail_policy": "drop incomplete final block",
            "aggregation": "unweighted mean of per-block ROC AUC within each condition; unweighted mean of ten conditions",
            "label_convention": "0=simulated signal, 1=injected noise",
            "edmamba_input": "HDF5 polarity threshold p>=0.5, then transformed to 2*p-1",
            "edmamba_coordinate_mode": args.edmamba_coordinate_mode,
            "native_input": "timestamp seconds rounded to relative microseconds; polarity threshold p>=0.5",
            "native_score_direction": "DND21-frozen: higher score means noise",
            "oriented_auc": "diagnostic only; not used for the primary comparison",
            "tau_sweep": "post-hoc float diagnostic at 1, 4, and 16 ms; primary remains frozen 250 us",
            "first_grid_size": 0.05,
            "test_grid_size": 0.1,
            "seed": args.seed,
            "max_blocks_per_stream": args.max_blocks,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "edmamba": checkpoint_metadata,
        "native": {
            "method": "bidirectional 250-us same-polarity local/global log contrast",
            "trained_parameters": 0,
            "state_config": native_config.to_dict(),
            "fixed_point": {
                "native_q12_8": "four unsigned score operands, 12 total / 8 fractional bits",
                "native_q8_4": "four unsigned score operands, 8 total / 4 fractional bits",
            },
        },
        "results": {},
    }
    print(json.dumps({"environment": payload["environment"], "edmamba": payload["edmamba"]}, indent=2))

    for sequence in args.sequences:
        payload["results"].setdefault(sequence, {})
        for level in args.noise_levels:
            path = args.dataset_root / f"{sequence}_{level}.hdf5"
            if not path.is_file():
                raise FileNotFoundError(path)
            torch.cuda.reset_peak_memory_stats()
            stream = evaluate_stream(
                model,
                path,
                native_config,
                args.block_size,
                args.max_blocks,
                args.edmamba_coordinate_mode,
            )
            stream["scene_alias"] = SEQUENCES[sequence]
            stream["noise_level_percent_of_signal_count"] = level
            stream["paper_edmamba_auc"] = PAPER_AUC[sequence][level]
            stream["edmamba_delta_vs_paper"] = (
                stream["methods"]["edmamba"]["mean_block_auc"]
                - stream["paper_edmamba_auc"]
            )
            stream["peak_cuda_memory_mib"] = float(
                torch.cuda.max_memory_allocated() / 2**20
            )
            payload["results"][sequence][str(level)] = stream
            payload["summary"] = summarize(
                payload["results"],
                seed=args.seed,
                bootstrap_replicates=args.bootstrap_replicates,
                cluster_blocks=args.cluster_blocks,
            )
            write_results(args.output, payload)
            print(
                f"RESULT {sequence} {level}: "
                f"edmamba={stream['methods']['edmamba']['mean_block_auc']:.6f} "
                f"native={stream['methods']['native_float']['mean_block_auc']:.6f} "
                f"q8.4={stream['methods']['native_q8_4']['mean_block_auc']:.6f} "
                f"paper={stream['paper_edmamba_auc']:.4f}",
                flush=True,
            )

    write_results(args.output, payload)
    print(f"Wrote {args.output.resolve()}", flush=True)
    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
