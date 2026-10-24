"""Evaluate the zero-parameter Native3D local/global contrast on DVSCLEAN.

This path depends only on ``native3d_ssm`` and ``h5py``: no EDmamba, torch, or
pointcept import is required, unlike ``reproduction/evaluate_dvsclean.py``.
Native3D block AUCs are recomputed fresh from the released HDF5 files; the
comparison EDmamba block AUCs are read from the already-computed, frozen
``reproduction/results/dvsclean_paired.json`` (produced by the released
EDmamba checkpoint under CUDA). Before pairing, this script checks that the
source HDF5 file hashes and per-condition block counts match that frozen
record exactly, so a stale or substituted baseline cannot be silently paired
against fresh Native3D scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

from .analytic import analytic_contrast_score
from .core import StateConfig, bidirectional_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Verified from label-0 signal-event accumulation images; see
# DVSCLEAN_REPRODUCIBILITY_REPORT.md for the montage that fixes this mapping.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "DVSCLEAN/simulated_data"
    )
    parser.add_argument(
        "--edmamba-results",
        type=Path,
        default=PROJECT_ROOT / "reproduction/results/dvsclean_paired.json",
        help="Frozen paired EDmamba/Native3D JSON supplying the EDmamba baseline.",
    )
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--noise-levels", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--cluster-blocks", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/dvsclean_native_only.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_polarity(values: np.ndarray) -> np.ndarray:
    """Shared label-blind adapter: DVSCLEAN stores a continuous polarity field."""

    return (np.asarray(values, dtype=np.float32) >= 0.5).astype(np.float32)


def iter_hdf5_full_blocks(
    path: Path, block_size: int, max_blocks: Optional[int] = None
) -> Iterable[Dict[str, np.ndarray]]:
    """Yield consecutive complete blocks without materializing the full stream."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    with h5py.File(path, "r") as handle:
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


def prepare_native_events(block: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert one HDF5 block to Native3D's [microseconds, x, y, binary p] contract."""

    timestamp_seconds = block["timestamp"].astype(np.float64, copy=False)
    timestamp_us = np.rint((timestamp_seconds - timestamp_seconds[0]) * 1_000_000.0)
    polarity = binary_polarity(block["polarity"]).astype(np.int64)
    events = np.column_stack(
        (timestamp_us.astype(np.int64), block["x"], block["y"], polarity)
    )
    return np.ascontiguousarray(events, dtype=np.int64)


def score_native_block(block: Dict[str, np.ndarray], config: StateConfig) -> Dict[str, np.ndarray]:
    raw = bidirectional_features(prepare_native_events(block), config)
    return {
        "native_float": analytic_contrast_score(raw, config, "float"),
        "native_q12_8": analytic_contrast_score(raw, config, "q12.8"),
        "native_q8_4": analytic_contrast_score(raw, config, "q8.4"),
    }


def evaluate_stream(
    path: Path, config: StateConfig, block_size: int, max_blocks: Optional[int]
) -> Dict:
    method_names = ("native_float", "native_q12_8", "native_q8_4")
    block_aucs = {name: [] for name in method_names}
    labels_by_block = []
    started = time.perf_counter()
    for block in iter_hdf5_full_blocks(path, block_size, max_blocks):
        labels = block["label"].astype(np.int64, copy=False)
        scores = score_native_block(block, config)
        for name, values in scores.items():
            block_aucs[name].append(float(roc_auc_score(labels, values)))
        labels_by_block.append(len(labels))
    if not block_aucs["native_float"]:
        raise RuntimeError(f"No complete {block_size}-event blocks found in {path}")
    return {
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "evaluated_blocks": len(block_aucs["native_float"]),
        "evaluated_events": sum(labels_by_block),
        "wall_seconds": time.perf_counter() - started,
        "methods": {
            name: {
                "mean_block_auc": float(np.mean(values)),
                "std_block_auc": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "block_aucs": values,
            }
            for name, values in block_aucs.items()
        },
    }


def paired_cluster_bootstrap(
    condition_differences, seed: int, replicates: int, cluster_blocks: int
) -> Dict:
    """Condition-stratified paired bootstrap over contiguous block clusters."""

    observed = float(np.mean([np.mean(values) for values in condition_differences]))
    clusters_by_condition = []
    for values in condition_differences:
        values = np.asarray(values, dtype=np.float64)
        clusters = [
            (float(values[start : start + cluster_blocks].sum()), len(values[start : start + cluster_blocks]))
            for start in range(0, len(values), cluster_blocks)
        ]
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
        "95ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "replicates": replicates,
        "cluster_blocks": cluster_blocks,
    }


def load_frozen_edmamba(path: Path) -> Dict:
    payload = json.loads(path.read_text())
    return payload["results"]


def summarize(results: Dict, frozen_edmamba: Dict, seed: int, replicates: int, cluster_blocks: int) -> Dict:
    conditions = [
        (sequence, level, stream)
        for sequence, levels in results.items()
        for level, stream in levels.items()
    ]
    edmamba_values = [
        stream["frozen_edmamba_mean_block_auc"] for _, _, stream in conditions
    ]
    summary = {
        "conditions": len(conditions),
        "paper_edmamba_mean_auc": float(
            np.mean([PAPER_AUC[sequence][int(level)] for sequence, level, _ in conditions])
        ),
        "frozen_local_edmamba_mean_auc": float(np.mean(edmamba_values)),
        "candidates": {},
    }
    for name in ("native_float", "native_q12_8", "native_q8_4"):
        values = [stream["methods"][name]["mean_block_auc"] for _, _, stream in conditions]
        paired = []
        for _, _, stream in conditions:
            candidate_blocks = np.asarray(stream["methods"][name]["block_aucs"])
            edmamba_blocks = np.asarray(stream["frozen_edmamba_block_aucs"])
            if candidate_blocks.shape != edmamba_blocks.shape:
                raise ValueError("Paired DVSCLEAN block arrays have different shapes")
            paired.append(candidate_blocks - edmamba_blocks)
        mean_auc = float(np.mean(values))
        summary["candidates"][name] = {
            "mean_auc": mean_auc,
            "delta_vs_frozen_local_edmamba": mean_auc - float(np.mean(edmamba_values)),
            "conditions_beating_frozen_local_edmamba": int(
                sum(value > baseline for value, baseline in zip(values, edmamba_values))
            ),
            "paired_cluster_bootstrap": paired_cluster_bootstrap(
                paired, seed=seed, replicates=replicates, cluster_blocks=cluster_blocks
            ),
        }
    return summary


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = StateConfig(width=1280, height=720)
    frozen = load_frozen_edmamba(args.edmamba_results)
    payload = {
        "benchmark": "DVSCLEAN labeled simulated paper subset",
        "note": (
            "Native3D scores are freshly computed here; the EDmamba baseline is "
            "read from the frozen reproduction/results/dvsclean_paired.json "
            "(released checkpoint, CUDA inference) rather than re-run, so this "
            "evaluator has no EDmamba/torch/pointcept dependency. Table 1 of "
            "main_cas-sc.tex does not include DVSCLEAN; the comparison baseline "
            "here is the EDmamba paper value and the frozen local reproduction."
        ),
        "method": "bidirectional 250-us same-polarity local/global log contrast",
        "trained_parameters": 0,
        "state_config": config.to_dict(),
        "protocol": {
            "block_size": args.block_size,
            "tail_policy": "drop incomplete final block",
            "label_convention": "0=simulated signal, 1=injected noise",
            "native_score_direction": "DND21-frozen: higher score means noise",
        },
        "results": {},
    }
    for sequence in args.sequences:
        payload["results"].setdefault(sequence, {})
        for level in args.noise_levels:
            path = args.dataset_root / f"{sequence}_{level}.hdf5"
            if not path.is_file():
                raise FileNotFoundError(path)
            frozen_stream = frozen[sequence][str(level)]
            frozen_sha256 = frozen_stream["source_sha256"]
            actual_sha256 = sha256_file(path)
            if actual_sha256 != frozen_sha256:
                raise ValueError(
                    f"{path} sha256 changed since the frozen EDmamba baseline was recorded"
                )
            stream = evaluate_stream(path, config, args.block_size, args.max_blocks)
            frozen_edmamba_blocks = frozen_stream["methods"]["edmamba"]["block_aucs"]
            if args.max_blocks is None and stream["evaluated_blocks"] != len(frozen_edmamba_blocks):
                raise ValueError(
                    f"{path}: fresh block count {stream['evaluated_blocks']} does not match "
                    f"frozen baseline block count {len(frozen_edmamba_blocks)}"
                )
            stream["frozen_edmamba_block_aucs"] = frozen_edmamba_blocks[: stream["evaluated_blocks"]]
            stream["frozen_edmamba_mean_block_auc"] = float(
                np.mean(stream["frozen_edmamba_block_aucs"])
            )
            stream["scene_alias"] = SEQUENCES[sequence]
            stream["noise_level_percent_of_signal_count"] = level
            stream["paper_edmamba_auc"] = PAPER_AUC[sequence][level]
            payload["results"][sequence][str(level)] = stream
            payload["summary"] = summarize(
                payload["results"],
                frozen,
                seed=args.seed,
                replicates=args.bootstrap_replicates,
                cluster_blocks=args.cluster_blocks,
            )
            _write(args.output, payload)
            print(
                f"RESULT {sequence} {level}: "
                f"native_float={stream['methods']['native_float']['mean_block_auc']:.6f} "
                f"q8.4={stream['methods']['native_q8_4']['mean_block_auc']:.6f} "
                f"frozen_edmamba={stream['frozen_edmamba_mean_block_auc']:.6f} "
                f"paper={stream['paper_edmamba_auc']:.4f}",
                flush=True,
            )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
