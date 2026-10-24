#!/usr/bin/env python3
"""Reproduce EDmamba's DND21 AUC evaluation from the raw ECCV2024 files.

This follows the released dataset/test implementation without materializing the
intermediate NPZ files:

1. split each labeled stream into consecutive 10,240-event blocks;
2. normalize timestamps independently inside every block;
3. apply the config's 0.05 temporal-grid random sampling;
4. apply its exhaustive 0.1 temporal-grid test partitioning;
5. average per-block AUC, as pointcept/engines/test.py does.

The script additionally reports an event-global AUC, which is not the metric
aggregation used by the released tester but is useful as a cross-check.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDMAMBA_ROOT = PROJECT_ROOT / "EDmamba"
POINTOPS_BUILD = (
    EDMAMBA_ROOT / "libs/pointops/build/lib.linux-x86_64-cpython-38"
)
sys.path.insert(0, str(EDMAMBA_ROOT))
sys.path.insert(0, str(POINTOPS_BUILD))

PAPER_AUC = {
    "1hz": {"hotel-bar": 0.9963, "driving": 0.9694},
    "3hz": {"hotel-bar": 0.9949, "driving": 0.9710},
    "5hz": {"hotel-bar": 0.9955, "driving": 0.9734},
    "7hz": {"hotel-bar": 0.9939, "driving": 0.9689},
    "10hz": {"hotel-bar": 0.9916, "driving": 0.9636},
}

SCENE_FILES = {
    "hotel-bar": "mix_result.txt",
    "driving": "driving_mix_result.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test",
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
    parser.add_argument(
        "--rates",
        nargs="+",
        default=list(PAPER_AUC),
        choices=[f"{rate}hz" for rate in range(1, 11)],
    )
    parser.add_argument(
        "--scenes", nargs="+", default=list(SCENE_FILES), choices=list(SCENE_FILES)
    )
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Limit blocks per stream for a smoke test; omit for full evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reproduction/results/dnd21_auc.json",
    )
    parser.add_argument(
        "--hard-exit",
        action="store_true",
        help="Bypass CUDA destructor cleanup (workaround for old CUDA 11.6 on WSL).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_checkpoint_model(config_path: Path, checkpoint_path: Path) -> Tuple[torch.nn.Module, Dict]:
    from pointcept.models import build_model
    from pointcept.utils.config import Config

    cfg = Config.fromfile(str(config_path))
    model = build_model(cfg.model)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in checkpoint["state_dict"].items()
    )
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_best_metric": float(checkpoint["best_metric_value"]),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    return model, metadata


def full_blocks(
    path: Path, block_size: int, max_blocks: Optional[int]
) -> Iterable[np.ndarray]:
    reader = pd.read_csv(
        path,
        sep=r"\s+",
        dtype=np.int64,
        chunksize=block_size,
        engine="c",
    )
    for index, frame in enumerate(reader):
        if max_blocks is not None and index >= max_blocks:
            break
        values = frame.to_numpy(copy=False)
        if values.shape != (block_size, 5):
            break
        yield values


def prepare_block(values: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
    from pointcept.datasets.transform import GridSample_V3

    timestamps = values[:, 0]
    denominator = timestamps[-1] - timestamps[0]
    if denominator <= 0:
        raise ValueError("A block must span a positive timestamp interval")
    normalized_time = (timestamps - timestamps.min()) / denominator
    coord = np.column_stack(
        (values[:, 1], values[:, 2], normalized_time)
    ).astype(np.float32)
    strength = (values[:, 3] * 2 - 1).reshape(-1, 1).astype(np.float32)
    labels = values[:, 4].astype(np.int64, copy=False)

    # This is the config's first test transform. It deliberately uses mode="train"
    # and therefore randomly chooses one event from each 0.05-bin/polarity voxel.
    data = {"coord": coord, "strength": strength, "segment": labels.copy()}
    sampler = GridSample_V3(
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "strength", "segment"),
        return_inverse=True,
    )
    data = sampler(data)
    inverse = data["inverse"].copy()
    sampled_labels = data["segment"].copy()
    return data, inverse, sampled_labels


def score_block(model: torch.nn.Module, values: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    from pointcept.datasets.transform import Collect, GridSample_V3, ToTensor
    from pointcept.datasets.utils import collate_fn

    data, inverse, sampled_labels = prepare_block(values)
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
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_dict)["seg_logits"]
            part_probability = F.softmax(logits, dim=-1)
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - start
        probabilities[index] += part_probability
        counts[index] += 1

    sampled_scores = (probabilities / counts)[:, 1].cpu().numpy()
    original_scores = sampled_scores[inverse]
    auc = float(roc_auc_score(values[:, 4], original_scores))
    return original_scores, auc, int(sampled_labels.size), len(fragments), inference_seconds


def evaluate_stream(
    model: torch.nn.Module,
    path: Path,
    block_size: int,
    max_blocks: Optional[int],
) -> Dict:
    block_aucs = []
    all_scores = []
    all_labels = []
    sampled_events = 0
    fragment_count = 0
    inference_seconds = 0.0
    wall_start = time.perf_counter()

    for block_index, values in enumerate(full_blocks(path, block_size, max_blocks), start=1):
        scores, block_auc, sampled, fragments, block_inference = score_block(model, values)
        block_aucs.append(block_auc)
        all_scores.append(scores)
        all_labels.append(values[:, 4].copy())
        sampled_events += sampled
        fragment_count += fragments
        inference_seconds += block_inference
        if block_index == 1 or block_index % 50 == 0:
            print(
                f"{path.parent.name}/{path.name}: blocks={block_index} "
                f"mean_auc={np.mean(block_aucs):.6f}",
                flush=True,
            )

    if not block_aucs:
        raise RuntimeError(f"No complete {block_size}-event blocks found in {path}")
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    return {
        "source": str(path.resolve()),
        "blocks": len(block_aucs),
        "original_events": int(labels.size),
        "sampled_events": sampled_events,
        "fragments": fragment_count,
        "mean_block_auc": float(np.mean(block_aucs)),
        "std_block_auc": float(np.std(block_aucs, ddof=1)) if len(block_aucs) > 1 else 0.0,
        "block_aucs": [float(value) for value in block_aucs],
        "event_global_auc": float(roc_auc_score(labels, scores)),
        "inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - wall_start,
        "peak_cuda_memory_mib": float(torch.cuda.max_memory_allocated() / 2**20),
    }


def write_results(path: Path, results: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, checkpoint_metadata = build_checkpoint_model(args.config, args.checkpoint)
    results = {
        "protocol": {
            "block_size": args.block_size,
            "seed": args.seed,
            "first_grid_size": 0.05,
            "test_grid_size": 0.1,
            "aggregation": "unweighted mean of per-block ROC AUC (released tester)",
            "max_blocks_per_stream": args.max_blocks,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "model": checkpoint_metadata,
        "results": {},
    }
    print(json.dumps({"environment": results["environment"], "model": results["model"]}, indent=2))

    for rate in args.rates:
        results["results"].setdefault(rate, {})
        for scene in args.scenes:
            source = args.dataset_root / rate / SCENE_FILES[scene]
            if not source.is_file():
                raise FileNotFoundError(source)
            torch.cuda.reset_peak_memory_stats()
            stream_result = evaluate_stream(model, source, args.block_size, args.max_blocks)
            target = PAPER_AUC.get(rate, {}).get(scene)
            stream_result["paper_auc"] = target
            stream_result["delta_from_paper"] = (
                stream_result["mean_block_auc"] - target if target is not None else None
            )
            results["results"][rate][scene] = stream_result
            write_results(args.output, results)
            print(
                f"RESULT {rate} {scene}: mean_block_auc={stream_result['mean_block_auc']:.6f} "
                f"global_auc={stream_result['event_global_auc']:.6f} paper={target}",
                flush=True,
            )

    write_results(args.output, results)
    print(f"Wrote {args.output.resolve()}", flush=True)
    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
