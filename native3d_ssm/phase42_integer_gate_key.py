"""Phase 42: integer and covariance alternatives for the gated-control key.

The gated control uses ``corr(G, R_same)``, where ``R_same`` is the radius-1 decayed
magnitude, even when its score uses bounded support. These measurements separate
score-path arithmetic from the additional operations required by the gate.

Compare four keys with the score, radius-2 support operand, block size and zero
threshold held fixed:

``magnitude``    ``beta = 1 iff corr(G, R_same) < 0``
``log_support``  ``beta = 1 iff corr(G, log1p(S1)) < 0``
``support``      ``beta = 1 iff corr(G, S1) < 0``
``covariance``   ``beta = 1 iff Cov(G, R_same) < 0``

When both variances are positive, ``sign(corr) = sign(Cov)``. The covariance key
therefore removes normalisation, division and square root; disagreements with
the correlation key identify degenerate-variance blocks.

This is a gated-control comparison. Nothing is trained or fitted, and no selected
configuration is changed.
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
from .evaluate_adaptive_beta import (
    dnd21_blocks,
    dvsclean_blocks,
    per_block_operands_both,
)
from .evaluate_dnd21 import PAPER_AUC as DND21_RATES
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase14_quantizable_operand import BLOCK_SIZE, ring_support, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYS = ("magnitude", "log_support", "support", "covariance")

#: Anchors this run must reproduce from the adopted configuration. If the ``magnitude``
#: row drifts from these, the pipeline moved underneath and the comparison is void.
ANCHORS = {"DND21": 0.9917, "DVSCLEAN": 0.9706, "overall_20": 0.9811}


def safe_cov(a: np.ndarray, b: np.ndarray) -> float:
    """Population covariance, the quantity whose sign the gate actually needs."""

    if len(a) < 2:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())))


def gate_keys(g: np.ndarray, r_same: np.ndarray,
              support_r1: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Per-key (statistic, beta) for one block, all read off the same operands."""

    statistics = {
        "magnitude": safe_corr(g, r_same),
        "log_support": safe_corr(g, np.log1p(support_r1)),
        "support": safe_corr(g, support_r1),
        "covariance": safe_cov(g, r_same),
    }
    return {name: (value, 1.0 if value < 0.0 else 0.0) for name, value in statistics.items()}


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, score_config: Optional[StateConfig],
                   max_blocks: Optional[int] = None) -> Dict:
    cfg_s = score_config if score_config is not None else config
    scale_count = len(config.taus_us)
    auc: Dict[str, List[float]] = {}
    open_fraction: Dict[str, List[float]] = {name: [] for name in KEYS}
    agreement: Dict[str, List[float]] = {name: [] for name in KEYS}
    produced = 0

    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue

        raw = bidirectional_features(events, config)
        g, r_same, _ = per_block_operands_both(raw, scale_count, config.feature_dim)
        support_r1 = ring_support(raw, config)

        raw_s = bidirectional_features(events, cfg_s) if score_config is not None else raw
        subtrahend = np.log1p(ring_support(raw_s, cfg_s))

        decided = gate_keys(g, r_same, support_r1)
        reference = decided["magnitude"][1]
        for name, (_, beta) in decided.items():
            open_fraction[name].append(beta)
            agreement[name].append(float(beta == reference))
            score = beta * g - subtrahend
            if score.std() <= 0:
                continue
            auc.setdefault(name, []).append(roc_auc_score(labels, score))

    if not auc:
        return {"blocks": 0}
    return {
        "blocks": produced,
        "auc": {k: float(np.mean(v)) for k, v in auc.items()},
        "gate_open_fraction": {k: float(np.mean(v)) for k, v in open_fraction.items() if v},
        "agreement_with_adopted": {k: float(np.mean(v)) for k, v in agreement.items() if v},
    }


def aggregate(conditions: Dict[str, Dict], field: str) -> Dict[str, float]:
    usable = [c for c in conditions.values() if c.get("blocks") and c.get(field)]
    if not usable:
        return {}
    names = set(usable[0][field])
    for c in usable[1:]:
        names &= set(c[field])
    return {n: float(np.mean([c[field][n] for c in usable])) for n in sorted(names)}


def paired_interval(per_condition: List[float], resamples: int = 10_000,
                    seed: int = 20260819) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap interval over conditions, as elsewhere in this work."""

    values = np.asarray(per_condition, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return (float(values.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT
                        / "native3d_ssm/experiments/analytic_contrast/phase42_integer_gate_key.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    def configs(width: int, height: int):
        gate = StateConfig(width=width, height=height, taus_us=(250.0,), radius=1)
        score = (StateConfig(width=width, height=height, taus_us=(250.0,),
                             radius=args.score_radius)
                 if args.score_radius != 1 else None)
        return gate, score

    dvs_gate, dvs_score = configs(1280, 720)
    dnd_gate, dnd_score = configs(346, 260)

    dvsclean: Dict[str, Dict] = {}
    for sequence in SEQUENCES:
        for level in (50, 100):
            name = f"{sequence}_{level}"
            dvsclean[name] = scan_condition(
                dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE,
                                args.max_blocks),
                dvs_gate, dvs_score, args.max_blocks)
            print(f"DVSCLEAN {name}: " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(dvsclean[name].get("auc", {}).items())),
                flush=True)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_RATES:
        for scene, filename in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / filename, BLOCK_SIZE, args.max_blocks),
                dnd_gate, dnd_score, args.max_blocks)
            print(f"DND21 {name}: " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(dnd21[name].get("auc", {}).items())),
                flush=True)

    everything = {**dvsclean, **dnd21}
    dv, dn = aggregate(dvsclean, "auc"), aggregate(dnd21, "auc")
    overall = {k: 0.5 * (dv[k] + dn[k]) for k in dv if k in dn}

    per_condition = {
        key: [c["auc"][key] for c in everything.values() if c.get("blocks") and key in c["auc"]]
        for key in KEYS
    }
    reference = np.asarray(per_condition["magnitude"], dtype=np.float64)
    deltas = {}
    for key in KEYS:
        if key == "magnitude":
            continue
        values = np.asarray(per_condition[key], dtype=np.float64)
        if len(values) != len(reference):
            continue
        mean, low, high = paired_interval(list(values - reference))
        deltas[key] = {
            "paired_mean_vs_magnitude": mean,
            "ci95": [low, high],
            "conditions_not_worse": int(np.sum(values >= reference - 1e-12)),
            "conditions": int(len(values)),
        }

    payload = {
        "phase": "42 integer gate key",
        "question": ("The score is an integer count; the gate still keys on the radius-1 "
                     "decayed ring magnitude. Can an integer count key it instead, and is "
                     "the covariance sign enough to avoid the division and square root?"),
        "frozen": ("score s = beta*G - log1p(support@r2), threshold theta = 0, block 10,240. "
                   "Only the gate key changes between rows."),
        "anchors": ANCHORS,
        "config": {"gate_radius": 1, "score_radius": args.score_radius,
                   "block_size": BLOCK_SIZE},
        "summary": {"DVSCLEAN": dv, "DND21": dn, "overall_20": overall},
        "gate_open_fraction": {
            "DVSCLEAN": aggregate(dvsclean, "gate_open_fraction"),
            "DND21": aggregate(dnd21, "gate_open_fraction"),
        },
        "agreement_with_adopted": {
            "DVSCLEAN": aggregate(dvsclean, "agreement_with_adopted"),
            "DND21": aggregate(dnd21, "agreement_with_adopted"),
        },
        "paired_vs_magnitude": deltas,
        "per_condition": {"DVSCLEAN": dvsclean, "DND21": dnd21},
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall_20": overall, "paired_vs_magnitude": deltas}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
