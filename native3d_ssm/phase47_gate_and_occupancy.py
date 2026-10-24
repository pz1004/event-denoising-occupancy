"""Phase 47: gate-key geometry, occupancy identity and retention ties.

The gate key K is a radius-1 magnitude, whereas the support score uses
L=log(1+S) at radius 2. Evaluate beta=0 versus beta=1 for both the key score
``beta*G-K`` and the support score ``beta*G-L``. This distinguishes a key-score
selection rate of 94.2% from performance on the different support operand.

The analysis separates two sign conditions:

    (A1a) Delta K < 0 -- gate implication: rho < 0 => Delta G > 0
    (A1b) Delta L < 0 -- score implication: Delta G > 0 widens the mean gap

The occupancy arm compares ``sum_ring 1[a_fwd+a_rev>0]`` with a reference count
built from (x,y,p) alone. The extractor clamps age at ``20*taus_us[-1]``; with
one tau the maximum age is 5 ms and exp(-20)=2.06e-9 remains positive in float32.
Thus combined forward and reverse positivity represents same-polarity occupancy
anywhere in the block, independent of event age and tau.

The retention arm measures tie mass at the cutoff and compares stable arrival-order
ties with random tie-breaking. With S in [0,24] and beta=0, the score takes at most
25 distinct values. No parameter is trained or fitted.
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
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase14_quantizable_operand import ring_support, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK_SIZE = 10_240
FROZEN_RETENTION = 0.28
MIN_PER_CLASS = 50
SCORE_RADIUS = 2
GATE_RADIUS = 1


def occupancy_support(events: np.ndarray, radius: int, width: int,
                      height: int) -> np.ndarray:
    """Reference `S` from whole-block same-polarity occupancy alone.

    Uses only (x, y, p). No decayed state, no tau, no forward/reverse pass. A tap
    counts if ANY event in the block fired at it with the same polarity. The centre
    is excluded, and the event's own contribution cannot enter because the centre is
    the only tap it writes.
    """

    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    p = events[:, 3].astype(np.int64)
    occupied = np.zeros((2, height, width), dtype=bool)
    occupied[p, y, x] = True

    total = np.zeros(len(events), dtype=np.int64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            hit = np.zeros(len(events), dtype=bool)
            hit[inside] = occupied[p[inside], ny[inside], nx[inside]]
            total += hit
    return total.astype(np.float64)


def tie_statistics(scores: np.ndarray, labels: np.ndarray,
                   rng: np.random.Generator) -> Dict[str, float]:
    """Tie mass at the r = 0.28 cut, and what random tie-breaking costs."""

    n = len(scores)
    n_keep = max(1, int(round(n * FROZEN_RETENTION)))
    order = np.argsort(scores, kind="stable")
    cut_value = scores[order[n_keep - 1]]
    tied = scores == cut_value
    # How many of the tie group are kept versus how many exist: the cut is arbitrary
    # inside this group, and this is the number of events the convention decides.
    kept_from_tie = int(tied[order[:n_keep]].sum())

    stable_keep = np.zeros(n, dtype=bool)
    stable_keep[order[:n_keep]] = True
    jitter_order = np.lexsort((rng.random(n), scores))
    random_keep = np.zeros(n, dtype=bool)
    random_keep[jitter_order[:n_keep]] = True

    return {
        "tie_group_size": int(tied.sum()),
        "tie_group_fraction": float(tied.sum()) / n,
        "kept_from_tie_group": kept_from_tie,
        "undetermined_fraction": float(min(kept_from_tie, int(tied.sum()) - kept_from_tie)) / n,
        "keep_set_disagreement": float((stable_keep != random_keep).sum()) / n,
        "signal_recall_stable": float(stable_keep[labels == 0].mean()),
        "signal_recall_random": float(random_keep[labels == 0].mean()),
    }


def block_terms(events: np.ndarray, labels: np.ndarray,
                gate_config: StateConfig, score_config: StateConfig,
                rng: np.random.Generator) -> Optional[Dict]:
    noise, signal = labels == 1, labels == 0
    if noise.sum() < MIN_PER_CLASS or signal.sum() < MIN_PER_CLASS:
        return None

    scale_count = len(gate_config.taus_us)
    raw_g = bidirectional_features(events, gate_config)
    g, k, _ = per_block_operands_both(raw_g, scale_count, gate_config.feature_dim)
    raw_s = bidirectional_features(events, score_config)
    support = ring_support(raw_s, score_config)
    ell = np.log1p(support)

    if g.std() <= 0 or k.std() <= 0 or ell.std() <= 0:
        return None

    reference = occupancy_support(events, SCORE_RADIUS,
                                  score_config.width, score_config.height)
    mismatches = int((support != reference).sum())

    pi = float(noise.mean())
    delta_g = float(g[noise].mean() - g[signal].mean())
    delta_k = float(k[noise].mean() - k[signal].mean())
    delta_l = float(ell[noise].mean() - ell[signal].mean())
    c_within = float(pi * np.cov(g[noise], k[noise])[0, 1]
                     + (1.0 - pi) * np.cov(g[signal], k[signal])[0, 1])
    rho = float(np.corrcoef(g, k)[0, 1])

    # Does admitting G help? Measured separately on the score the paper adopts and on
    # the score phase 26 actually measured.
    auc_adopted_gain = float(roc_auc_score(labels, g - ell) - roc_auc_score(labels, -ell))
    auc_key_gain = float(roc_auc_score(labels, g - k) - roc_auc_score(labels, -k))

    opened = rho < 0.0
    scores = (g - ell) if opened else (-ell)
    return {
        "rho": rho, "delta_G": delta_g, "delta_K": delta_k, "delta_L": delta_l,
        "pi_noise": pi, "C_within": c_within,
        "C_between_key": float(pi * (1.0 - pi) * delta_g * delta_k),
        "auc_adopted_gain": auc_adopted_gain,
        "auc_key_gain": auc_key_gain,
        "gate_correct_adopted": bool(opened == (auc_adopted_gain > 0)),
        "gate_correct_key": bool(opened == (auc_key_gain > 0)),
        "a1a_delta_K_negative": bool(delta_k < 0),
        "a1b_delta_L_negative": bool(delta_l < 0),
        "a2_C_within_positive": bool(c_within > 0),
        "theory_matches_implemented": bool((delta_g > 0) == opened),
        "support_mismatches": mismatches,
        "support_max": float(support.max()),
        "block_duration_us": float(events[-1, 0] - events[0, 0]),
        "distinct_scores": int(len(np.unique(scores))),
        "gate_open": bool(opened),
        **tie_statistics(scores, labels, rng),
    }


def scan(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
         gate_config: StateConfig, score_config: StateConfig,
         max_blocks: Optional[int], seed: int) -> Optional[Dict]:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for produced, (events, labels) in enumerate(block_iter):
        if max_blocks is not None and produced >= max_blocks:
            break
        row = block_terms(np.asarray(events), np.asarray(labels),
                          gate_config, score_config, rng)
        if row is not None:
            rows.append(row)
    if not rows:
        return None

    out: Dict[str, float] = {"blocks": len(rows)}
    for key in ("rho", "delta_G", "delta_K", "delta_L", "pi_noise",
                "C_within", "C_between_key", "auc_adopted_gain", "auc_key_gain",
                "tie_group_fraction", "undetermined_fraction",
                "keep_set_disagreement", "signal_recall_stable",
                "signal_recall_random", "block_duration_us", "distinct_scores"):
        out[key] = float(np.mean([r[key] for r in rows]))
    for key in ("gate_correct_adopted", "gate_correct_key", "a1a_delta_K_negative",
                "a1b_delta_L_negative", "a2_C_within_positive",
                "theory_matches_implemented", "gate_open"):
        out[f"frac_{key}"] = float(np.mean([r[key] for r in rows]))
    out["support_mismatch_total"] = int(sum(r["support_mismatches"] for r in rows))
    out["support_max"] = float(max(r["support_max"] for r in rows))
    out["block_duration_us_max"] = float(max(r["block_duration_us"] for r in rows))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase47_gate_and_occupancy.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    conditions: Dict[str, Dict] = {}

    def run(name: str, block_iter, w: int, h: int) -> None:
        gate = StateConfig(width=w, height=h, taus_us=(250.0,), radius=GATE_RADIUS)
        score = StateConfig(width=w, height=h, taus_us=(250.0,), radius=SCORE_RADIUS)
        row = scan(block_iter, gate, score, args.max_blocks, args.seed)
        if row is None:
            return
        conditions[name] = row
        print(f"{name}: mismatch={row['support_mismatch_total']} "
              f"dL={row['delta_L']:+.4f} "
              f"gate_ok_adopted={row['frac_gate_correct_adopted']:.4f} "
              f"gate_ok_key={row['frac_gate_correct_key']:.4f} "
              f"tie={row['tie_group_fraction']:.4f}", flush=True)

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

    rows = list(conditions.values())
    blocks = sum(r["blocks"] for r in rows)

    def wmean(key: str) -> float:
        return float(sum(r[key] * r["blocks"] for r in rows) / blocks)

    summary = {
        "conditions": len(rows),
        "blocks": blocks,
        "support_equals_occupancy": all(r["support_mismatch_total"] == 0 for r in rows),
        "support_mismatch_total": int(sum(r["support_mismatch_total"] for r in rows)),
        "support_max": float(max(r["support_max"] for r in rows)),
        "block_duration_us_max": float(max(r["block_duration_us_max"] for r in rows)),
        "max_age_us": 20.0 * 250.0,
        "decay_floor": float(np.exp(-20.0)),
        "frac_a1a_delta_K_negative": wmean("frac_a1a_delta_K_negative"),
        "frac_a1b_delta_L_negative": wmean("frac_a1b_delta_L_negative"),
        "frac_a2_C_within_positive": wmean("frac_a2_C_within_positive"),
        "frac_theory_matches_implemented": wmean("frac_theory_matches_implemented"),
        "frac_gate_correct_adopted": wmean("frac_gate_correct_adopted"),
        "frac_gate_correct_key": wmean("frac_gate_correct_key"),
        "frac_gate_open": wmean("frac_gate_open"),
        "mean_delta_L": wmean("delta_L"),
        "mean_delta_K": wmean("delta_K"),
        "mean_tie_group_fraction": wmean("tie_group_fraction"),
        "mean_undetermined_fraction": wmean("undetermined_fraction"),
        "mean_keep_set_disagreement": wmean("keep_set_disagreement"),
        "mean_signal_recall_stable": wmean("signal_recall_stable"),
        "mean_signal_recall_random": wmean("signal_recall_random"),
        "mean_distinct_scores": wmean("distinct_scores"),
    }

    payload = {
        "phase": "47 three-variable gate check and support-versus-occupancy identity",
        "purpose": (
            "phase 26 measured the gate on beta*G - K, not on the adopted beta*G - "
            "log(1+S); and the age clamp max_age = 20*tau floors the decay at exp(-20) "
            "> 0, which predicts that S is exactly whole-block occupancy"
        ),
        "score_radius": SCORE_RADIUS,
        "gate_radius": GATE_RADIUS,
        "retention": FROZEN_RETENTION,
        "seed": args.seed,
        "summary": summary,
        "conditions": conditions,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print("\nsummary:", json.dumps(summary, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
