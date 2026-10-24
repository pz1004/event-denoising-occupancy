"""Phase 48: is the support advantage occupancy, or is it the fusion convention?

THE CONFOUND. The manuscript substitutes one local operand for another and attributes
the gain to counting occupancy instead of accumulating amplitude. But the two operands
of Section 3 differ in more than that. Written out:

    support    S = sum_ring 1[ a_fwd + a_rev > 0 ]          fuse taps, THEN count
    magnitude  R = sum_d log(1 + sum_ring a^d)              transform, THEN sum passes

Three differences ride together: (a) count versus summed amplitude, (b) the passes are
fused before the nonlinearity for support and after it for magnitude, and (c) support
carries no temporal decay while magnitude does. A reviewer is entitled to ask which one
the 0.95-point pooled gain belongs to.

(c) IS ALREADY SETTLED and needs no arm here. Phase 47 showed S is *exactly* whole-block
occupancy: the extractor clamps age at 20*tau, flooring the decay at exp(-20) > 0, so no
tap ever decays away and S carries no age information at all. There is no temporal-window
difference left to ablate -- the difference is total, and it is part of (a).

That leaves a clean 2x2 on the fusion point and the operand type:

                     fuse-then-transform          transform-then-sum
    count            support        (adopted)     support_split
    amplitude        magnitude_fused              magnitude      (adopted)

    support         S       = sum_ring 1[a_fwd + a_rev > 0]           in [0, 24]
    support_split   S_split = sum_ring 1[a_fwd>0] + sum_ring 1[a_rev>0]  in [0, 48]
    magnitude_fused Rf      = log(1 + sum_ring(a_fwd + a_rev))
    magnitude       R       = log(1 + sum_ring a_fwd) + log(1 + sum_ring a_rev)

The two contrasts that decompose the headline:

    support - magnitude_fused   count vs amplitude AT MATCHED FUSION   <- the real claim
    magnitude_fused - magnitude the fusion convention ALONE
    support_split - support     the fusion convention on the count side

If `support - magnitude_fused` keeps the sign and most of the size of
`support - magnitude`, the operand claim survives. If instead the gain migrates into
`magnitude_fused - magnitude`, the paper has been crediting occupancy for a fusion
convention and the central claim has to be rewritten.

Every arm uses the same gate (radius-1 magnitude key, beta = 1 iff corr(G,K) < 0), the
same blocks, and the same score shape s = beta*G - operand, so the arms are exactly
paired and differ in the operand alone.

The `support` arm must reproduce Phase 14's `log_support|float`, which certifies the
harness measures the paper's headline. Read-only; nothing is trained.

Unit of analysis: 20 labelled development conditions, paired within condition. Those 20
nest in 2 datasets, so a 20-condition interval understates between-condition correlation;
the two dataset means are reported alongside it, never in place of it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import (
    dnd21_blocks,
    dvsclean_blocks,
    per_block_operands_both,
)
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase14_quantizable_operand import ring_support, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK_SIZE = 10_240
SCORE_RADIUS = 2
GATE_RADIUS = 1

ARMS = ("support", "support_split", "magnitude_fused", "magnitude")

#: (arm, reference). Each contrast isolates exactly one cell move in the 2x2.
CONTRASTS = (
    ("support", "magnitude"),          # the headline substitution, both differences at once
    ("support", "magnitude_fused"),    # count vs amplitude, under support's fusion rule
    ("support_split", "magnitude"),    # count vs amplitude, under magnitude's fusion rule
    ("magnitude_fused", "magnitude"),  # fusion alone, operand held fixed
    ("support_split", "support"),      # fusion alone, on the count side
)

#: Phase 14 `log_support|float`; the `support` arm must reproduce these exactly.
ANCHORS = {"DND21": 0.9916848287458562, "DVSCLEAN": 0.9705824757446344}


def operands(raw_s: np.ndarray, score_config: StateConfig) -> Dict[str, np.ndarray]:
    """The four local terms, all from one radius-2 extraction so the arms are paired."""

    sc = len(score_config.taus_us)
    width = score_config.feature_dim
    nc = (2 * score_config.radius + 1) ** 2 - 1
    tap_cols = [7 * sc + k * sc for k in range(nc)]

    # Per-tap features are stored as log1p(value); log1p is monotone and zero only at
    # zero, so testing the stored value against zero tests the raw value against zero.
    fwd_taps = np.stack([raw_s[:, c] for c in tap_cols], axis=1)
    rev_taps = np.stack([raw_s[:, width + c] for c in tap_cols], axis=1)

    support = ring_support(raw_s, score_config)
    support_split = ((fwd_taps > 0).sum(axis=1) + (rev_taps > 0).sum(axis=1)).astype(np.float64)

    # Column 2*sc holds log1p(ring_sum) per pass, so expm1 recovers the raw ring sum.
    log_rs_fwd = raw_s[:, 2 * sc].astype(np.float64)
    log_rs_rev = raw_s[:, width + 2 * sc].astype(np.float64)
    magnitude = log_rs_fwd + log_rs_rev
    magnitude_fused = np.log1p(np.expm1(log_rs_fwd) + np.expm1(log_rs_rev))

    return {
        "support": np.log1p(support),
        "support_split": np.log1p(support_split),
        "magnitude_fused": magnitude_fused,
        "magnitude": magnitude,
        "_support_raw": support,
        "_support_split_raw": support_split,
    }


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   gate_config: StateConfig, score_config: StateConfig,
                   max_blocks: Optional[int]) -> Optional[Dict]:
    per_block: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    bounds = {"support_max": 0.0, "support_split_max": 0.0}
    produced = 0

    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue
        events = np.asarray(events)

        sc = len(gate_config.taus_us)
        raw_g = bidirectional_features(events, gate_config)
        g, k, _ = per_block_operands_both(raw_g, sc, gate_config.feature_dim)
        beta = 1.0 if safe_corr(g, k) < 0.0 else 0.0

        raw_s = bidirectional_features(events, score_config)
        ops = operands(raw_s, score_config)
        bounds["support_max"] = max(bounds["support_max"], float(ops["_support_raw"].max()))
        bounds["support_split_max"] = max(bounds["support_split_max"],
                                          float(ops["_support_split_raw"].max()))

        for arm in ARMS:
            per_block[arm].append(float(roc_auc_score(labels, beta * g - ops[arm])))

    if not per_block["support"]:
        return None
    out = {arm: float(np.mean(per_block[arm])) for arm in ARMS}
    out["blocks"] = len(per_block["support"])
    out.update(bounds)
    return out


def paired_stats(deltas: List[float]) -> Dict:
    a = np.asarray(deltas, dtype=np.float64)
    n = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    # Wilcoxon is the paper's convention for these 20 paired conditions; it needs no
    # normality assumption, which a 20-point paired sample cannot establish.
    try:
        stat, p = wilcoxon(a)
        p = float(p)
    except ValueError:          # every difference identical, including all-zero
        stat, p = float("nan"), float("nan")
    return {
        "n": n,
        "mean_delta": mean,
        "sd": sd,
        "ci95_mean_delta": [mean - 1.96 * se, mean + 1.96 * se],
        "wilcoxon_p": p,
        "wins": int((a > 0).sum()),
        "losses": int((a < 0).sum()),
        "ties": int((a == 0).sum()),
        # Paired Cohen's d. Reported because a sign count alone hides effect size.
        "cohen_d_paired": float(mean / sd) if sd > 0 else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase48_operand_decomposition.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    conditions: Dict[str, Dict] = {}

    def run(name: str, block_iter, w: int, h: int) -> None:
        gate = StateConfig(width=w, height=h, taus_us=(250.0,), radius=GATE_RADIUS)
        score = StateConfig(width=w, height=h, taus_us=(250.0,), radius=SCORE_RADIUS)
        row = scan_condition(block_iter, gate, score, args.max_blocks)
        if row is None:
            return
        conditions[name] = row
        print(f"{name}: sup={row['support']:.4f} sup_split={row['support_split']:.4f} "
              f"mag_fused={row['magnitude_fused']:.4f} mag={row['magnitude']:.4f}",
              flush=True)

    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            run(f"DND21 {rate} {scene}",
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                346, 260)
    for seq in SEQUENCES:
        for level in (50, 100):
            run(f"DVSCLEAN {seq}_{level}",
                dvsclean_blocks(args.dvsclean_root / f"{seq}_{level}.hdf5",
                                BLOCK_SIZE, args.max_blocks),
                1280, 720)

    names = list(conditions)
    dataset_means = {
        ds: {arm: float(np.mean([conditions[n][arm] for n in names if n.startswith(ds)]))
             for arm in ARMS}
        for ds in ("DND21", "DVSCLEAN")
    }
    pooled = {arm: 0.5 * (dataset_means["DND21"][arm] + dataset_means["DVSCLEAN"][arm])
              for arm in ARMS}

    comparisons = {}
    for arm, ref in CONTRASTS:
        deltas = [conditions[n][arm] - conditions[n][ref] for n in names]
        entry = paired_stats(deltas)
        entry["pooled_delta"] = pooled[arm] - pooled[ref]
        entry["per_dataset_delta"] = {
            ds: dataset_means[ds][arm] - dataset_means[ds][ref] for ds in dataset_means
        }
        comparisons[f"{arm}_minus_{ref}"] = entry

    # The 2x2 is only a 2x2 if the fourth cell is used. Measuring the operand effect
    # under BOTH fusion rules gives an interaction term, and an interaction indistinguishable
    # from zero is what licenses talking about "the operand effect" as one number rather
    # than as two that happen to be close.
    inter = [(conditions[n]["support_split"] - conditions[n]["magnitude"])
             - (conditions[n]["support"] - conditions[n]["magnitude_fused"]) for n in names]
    interaction = paired_stats(inter)
    interaction["reading"] = ("operand effect under magnitude's fusion rule minus the same "
                              "under support's; zero means the operand effect does not "
                              "depend on where the two passes are combined")

    head = comparisons["support_minus_magnitude"]["pooled_delta"]
    matched = comparisons["support_minus_magnitude_fused"]["pooled_delta"]
    fusion = comparisons["magnitude_fused_minus_magnitude"]["pooled_delta"]
    attribution = {
        "headline_pooled_delta": head,
        "operand_at_matched_fusion": matched,
        "fusion_convention_alone": fusion,
        # How much of the headline survives when the fusion convention is held fixed.
        "operand_share_of_headline": float(matched / head) if head != 0 else float("nan"),
        "fusion_share_of_headline": float(fusion / head) if head != 0 else float("nan"),
        "additivity_residual": float(head - matched - fusion),
    }

    anchor_drift = {ds: dataset_means[ds]["support"] - ANCHORS[ds] for ds in ANCHORS}

    payload = {
        "phase": "48 matched operand decomposition: occupancy versus fusion convention",
        "purpose": ("the headline support-minus-magnitude gain moves both the operand type "
                    "and the pass-fusion point at once; this separates them. The temporal "
                    "window arm is not run because phase 47 proved S carries no age "
                    "information, so that difference is total and belongs to the operand"),
        "arms": {
            "support": "sum_ring 1[a_fwd + a_rev > 0], then log1p; in [0, 24]",
            "support_split": "sum_ring 1[a_fwd>0] + sum_ring 1[a_rev>0], then log1p; in [0, 48]",
            "magnitude_fused": "log(1 + sum_ring(a_fwd + a_rev))",
            "magnitude": "log(1 + sum_ring a_fwd) + log(1 + sum_ring a_rev)",
        },
        "metric": "block-mean AUC over 10,240-event blocks",
        "pooling": "0.5 * (DND21 mean + DVSCLEAN mean), each a mean over 10 conditions",
        "unit_of_analysis": ("20 development conditions paired within condition; they nest "
                             "in 2 datasets, so the 20-condition interval understates "
                             "between-condition correlation and the dataset means are "
                             "reported alongside it"),
        "score_radius": SCORE_RADIUS,
        "gate_radius": GATE_RADIUS,
        "conditions": conditions,
        "dataset_means": dataset_means,
        "pooled": pooled,
        "comparisons": comparisons,
        "attribution": attribution,
        "fusion_operand_interaction": interaction,
        "anchor_drift_vs_phase14": anchor_drift,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print("\npooled:", json.dumps(pooled, indent=2))
    print("attribution:", json.dumps(attribution, indent=2))
    print("anchor drift vs Phase 14:", anchor_drift)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
