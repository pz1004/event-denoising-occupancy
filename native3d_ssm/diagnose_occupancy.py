"""Phase 0 diagnostic: same-polarity ring occupancy vs. operand informativeness.

Read-only. For DVSCLEAN and DND21 it recovers, per event and per tau plane, the
unlogged same-polarity ring operand ``R`` (occupancy proxy) and the log operands
that enter the score, then splits them by signal/noise label to test the
sparsity-floor hypothesis in improve_plan.md Section 3.3:

- Is the ring operand structurally zero on DVSCLEAN (R==0 fraction)?
- Does the signal-minus-noise occupancy gap grow with tau?
- Per tau plane, does the *global* operand carry the separation while the *ring*
  operand is near chance (ring dead weight), and does the combined contrast beat
  either operand alone?

Feature layout (four scales, radius 1, per pass of width 95; see core.py):
  ring_sum_same[scale] = index 2*4 + scale = 8 + scale   (log1p compressed)
  global_density[scale] = index 6*4 + scale = 24 + scale  (log1p compressed)
Bidirectional features concatenate [forward(95) | reverse(95)].
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .data import iter_labeled_text
from .evaluate_dvsclean_analytic import (
    SEQUENCES,
    iter_hdf5_full_blocks,
    prepare_native_events,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALE_LABELS = ("250us", "1ms", "4ms", "16ms")

# DND21 conditions: the known weak cell, a strong dense cell, and the densest.
DND21_CONDITIONS = {
    "1hz_driving": "ECCV2024_datasets/AUC_test/1hz/driving_mix_result.txt",
    "5hz_driving": "ECCV2024_datasets/AUC_test/5hz/driving_mix_result.txt",
    "10hz_driving": "ECCV2024_datasets/AUC_test/10hz/driving_mix_result.txt",
}


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """ROC AUC oriented so label 1 (noise) should score high; None if 1-class."""

    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def analyze_condition(
    block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
    config: StateConfig,
    max_blocks: Optional[int],
) -> Dict:
    """Accumulate per-scale operands over a condition and summarize by label."""

    scale_count = len(config.taus_us)
    ring_forward: List[np.ndarray] = [[] for _ in range(scale_count)]
    contrast: List[np.ndarray] = [[] for _ in range(scale_count)]
    ring_only: List[np.ndarray] = [[] for _ in range(scale_count)]
    global_only: List[np.ndarray] = [[] for _ in range(scale_count)]
    all_labels: List[np.ndarray] = []
    blocks = 0

    for events, labels in block_iter:
        if max_blocks is not None and blocks >= max_blocks:
            break
        raw = bidirectional_features(events, config)
        width = config.feature_dim
        for scale in range(scale_count):
            ring_log_f = raw[:, 2 * scale_count + scale]
            ring_log_r = raw[:, width + 2 * scale_count + scale]
            global_log_f = raw[:, 6 * scale_count + scale]
            global_log_r = raw[:, width + 6 * scale_count + scale]
            # Forward-pass occupancy proxy (unlogged sum of decayed states).
            ring_forward[scale].append(np.expm1(ring_log_f))
            # Bidirectional operands as actually consumed by the score.
            ring_sum = ring_log_f + ring_log_r
            global_sum = global_log_f + global_log_r
            contrast[scale].append(global_sum - ring_sum)      # higher => noise
            ring_only[scale].append(-ring_sum)                 # less ring => noise
            global_only[scale].append(global_sum)              # more global => noise
        all_labels.append(labels.astype(np.int64, copy=False))
        blocks += 1

    if not all_labels:
        raise RuntimeError("No complete blocks in this condition")
    labels = np.concatenate(all_labels)
    signal = labels == 0
    noise = labels == 1
    per_scale = {}
    for scale in range(scale_count):
        ring_fwd = np.concatenate(ring_forward[scale])
        con = np.concatenate(contrast[scale])
        ring_sc = np.concatenate(ring_only[scale])
        glob_sc = np.concatenate(global_only[scale])
        per_scale[SCALE_LABELS[scale]] = {
            "frac_ring_zero_all": float(np.mean(ring_fwd == 0.0)),
            "frac_ring_zero_signal": float(np.mean(ring_fwd[signal] == 0.0)),
            "frac_ring_zero_noise": float(np.mean(ring_fwd[noise] == 0.0)),
            "mean_ring_signal": float(np.mean(ring_fwd[signal])),
            "mean_ring_noise": float(np.mean(ring_fwd[noise])),
            "occupancy_gap_signal_minus_noise": float(
                np.mean(ring_fwd[signal]) - np.mean(ring_fwd[noise])
            ),
            "contrast_auc": _safe_auc(labels, con),
            "ring_only_auc": _safe_auc(labels, ring_sc),
            "global_only_auc": _safe_auc(labels, glob_sc),
        }
    return {
        "blocks": blocks,
        "events": int(labels.size),
        "signal_events": int(signal.sum()),
        "noise_events": int(noise.sum()),
        "per_scale": per_scale,
    }


def dvsclean_blocks(path: Path, block_size: int, max_blocks: Optional[int]):
    for block in iter_hdf5_full_blocks(path, block_size, max_blocks):
        yield prepare_native_events(block), block["label"].astype(np.int64, copy=False)


def dnd21_blocks(path: Path, block_size: int, max_blocks: Optional[int]):
    produced = 0
    for values in iter_labeled_text(path, block_size):
        if len(values) != block_size:
            break
        if max_blocks is not None and produced >= max_blocks:
            break
        yield values[:, :4], values[:, 4]
        produced += 1


def summarize_dataset(conditions: Dict) -> Dict:
    """Unweighted mean of each per-scale metric across a dataset's conditions."""

    summary = {}
    for scale in SCALE_LABELS:
        keys = conditions[next(iter(conditions))]["per_scale"][scale].keys()
        summary[scale] = {}
        for key in keys:
            values = [
                cond["per_scale"][scale][key]
                for cond in conditions.values()
                if cond["per_scale"][scale][key] is not None
            ]
            summary[scale][key] = float(np.mean(values)) if values else None
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dvsclean-root", type=Path, default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--dvsclean-max-blocks", type=int, default=None)
    parser.add_argument("--dnd21-max-blocks", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase0_occupancy.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    payload = {
        "phase": "0 occupancy diagnostic",
        "hypothesis": "improve_plan.md Section 3.3 sparsity floor",
        "scales_us": [250, 1000, 4000, 16000],
        "note": "ring occupancy is the forward-pass unlogged same-polarity ring; AUCs are bidirectional and oriented so label 1 (noise) scores high",
        "datasets": {},
    }

    # DVSCLEAN (target dataset): all five sequences at both noise levels.
    dvsclean_config = StateConfig(width=1280, height=720)
    dvsclean = {}
    for sequence in SEQUENCES:
        for level in (50, 100):
            path = args.dvsclean_root / f"{sequence}_{level}.hdf5"
            name = f"{sequence}_{level}"
            dvsclean[name] = analyze_condition(
                dvsclean_blocks(path, args.block_size, args.dvsclean_max_blocks),
                dvsclean_config,
                args.dvsclean_max_blocks,
            )
            top = dvsclean[name]["per_scale"]["250us"]
            print(
                f"DVSCLEAN {name}: 250us ring_zero_signal={top['frac_ring_zero_signal']:.4f} "
                f"contrast_auc={top['contrast_auc']:.4f} global_only_auc={top['global_only_auc']:.4f}",
                flush=True,
            )
    payload["datasets"]["DVSCLEAN"] = {
        "sensor": [1280, 720],
        "alpha_100k": 100000.0 / (1280 * 720),
        "conditions": dvsclean,
        "summary_per_scale": summarize_dataset(dvsclean),
    }

    # DND21 (reference regime): representative dense conditions.
    dnd21_config = StateConfig()
    dnd21 = {}
    for name, rel in DND21_CONDITIONS.items():
        path = PROJECT_ROOT / rel
        dnd21[name] = analyze_condition(
            dnd21_blocks(path, args.block_size, args.dnd21_max_blocks),
            dnd21_config,
            args.dnd21_max_blocks,
        )
        top = dnd21[name]["per_scale"]["250us"]
        print(
            f"DND21 {name}: 250us ring_zero_signal={top['frac_ring_zero_signal']:.4f} "
            f"contrast_auc={top['contrast_auc']:.4f} global_only_auc={top['global_only_auc']:.4f}",
            flush=True,
        )
    payload["datasets"]["DND21"] = {
        "sensor": [346, 260],
        "alpha_100k": 100000.0 / (346 * 260),
        "conditions": dnd21,
        "summary_per_scale": summarize_dataset(dnd21),
    }

    payload["wall_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output.resolve()} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
