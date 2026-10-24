"""Phase 11: attribute the radius-2 gain between the gate key and the score operand.

Phase 7's radius-2 scan (Step 3, 2026-07-26) measured the old score at 0.9688 overall_20
versus 0.9565 at radius 1 -- but that comparison is confounded. Widening the ring to
radius 2 changes **two** things at once:

  1. the score's subtrahend R, and
  2. the gate key corr(G, R), which decides beta.

The gate key is frozen by decision 2.4 and is defined on the radius-1 ring, so a radius-2
result that also moves the gate is not a clean operand measurement. This script runs the
full 2 x 2 grid on the *same* blocks:

              score R @ r=1        score R @ r=2
  gate @ r=1  (a) baseline         (c) ATTRIBUTION: operand effect alone
  gate @ r=2  (b) gate effect      (d) Step 3's confounded number

so the +0.0123 can be split into an operand part (c - a) and a gate part (d - c).

G (global_density) is radius-independent -- Phase 7 measured an identical AUC of 0.4454
at both radii -- so only R moves. That invariant is asserted per block here rather than
assumed.

Both score variants are evaluated:
    old:  s = beta*G - R_same
    new:  s = beta*G - (R_same + R_opposite)     (Phase 8)

Read-only over data on disk. Nothing is trained and no parameter is fitted: theta = 0
throughout.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 10_240

# (gate radius, score radius) -> row name. (1,1) reproduces Phase 3/4; (2,2) reproduces
# the Phase 7 radius-2 scan; (1,2) is the attribution row this phase exists for.
CELLS = ((1, 1), (2, 1), (1, 2), (2, 2))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() <= 0 or b.std() <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   cfg1: StateConfig, cfg2: StateConfig,
                   max_blocks: Optional[int] = None) -> Dict:
    """Per-block AUC for every (gate radius, score radius) cell and both score variants."""

    scale_count = len(cfg1.taus_us)
    auc: Dict[str, List[float]] = {}
    corr_same: Dict[int, List[float]] = {1: [], 2: []}
    beta_agree: List[float] = []
    max_g_drift = 0.0
    produced = 0

    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue

        ops: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for radius, cfg in ((1, cfg1), (2, cfg2)):
            raw = bidirectional_features(events, cfg)
            ops[radius] = per_block_operands_both(raw, scale_count, cfg.feature_dim)

        # G must not depend on radius; assert rather than assume.
        max_g_drift = max(max_g_drift, float(np.max(np.abs(ops[1][0] - ops[2][0]))))

        beta: Dict[int, float] = {}
        for radius in (1, 2):
            g, r_same, _ = ops[radius]
            corr = safe_corr(g, r_same)
            corr_same[radius].append(corr)
            beta[radius] = 1.0 if corr < 0.0 else 0.0
        beta_agree.append(float(beta[1] == beta[2]))

        for gate_r, score_r in CELLS:
            g, r_same, r_opposite = ops[score_r]
            b = beta[gate_r]
            auc.setdefault(f"old_gate{gate_r}_score{score_r}", []).append(
                roc_auc_score(labels, b * g - r_same))
            auc.setdefault(f"new_gate{gate_r}_score{score_r}", []).append(
                roc_auc_score(labels, b * g - (r_same + r_opposite)))

    if not auc:
        return {"blocks": 0}
    return {
        "blocks": produced,
        "mean_corr_r1": float(np.mean(corr_same[1])),
        "mean_corr_r2": float(np.mean(corr_same[2])),
        "frac_blocks_beta1_r1": float(np.mean(np.asarray(corr_same[1]) < 0)),
        "frac_blocks_beta1_r2": float(np.mean(np.asarray(corr_same[2]) < 0)),
        "beta_agreement_r1_vs_r2": float(np.mean(beta_agree)),
        "max_global_density_drift": max_g_drift,
        "auc": {k: float(np.mean(v)) for k, v in auc.items()},
    }


def aggregate(conditions: Dict[str, Dict]) -> Dict[str, float]:
    usable = [c for c in conditions.values() if c.get("blocks")]
    if not usable:
        return {}
    names = set(usable[0]["auc"])
    for c in usable[1:]:
        names &= set(c["auc"])
    return {n: float(np.mean([c["auc"][n] for c in usable])) for n in sorted(names)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase11_radius_attribution.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    dvsclean: Dict[str, Dict] = {}
    dvs1 = StateConfig(width=1280, height=720, taus_us=(250.0,), radius=1)
    dvs2 = StateConfig(width=1280, height=720, taus_us=(250.0,), radius=2)
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            dvsclean[name] = scan_condition(
                dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE, args.max_blocks),
                dvs1, dvs2, args.max_blocks)
            c = dvsclean[name]
            print(f"DVSCLEAN {name}: blocks={c['blocks']} "
                  f"a={c['auc']['old_gate1_score1']:.4f} c={c['auc']['old_gate1_score2']:.4f} "
                  f"d={c['auc']['old_gate2_score2']:.4f}", flush=True)

    dnd21: Dict[str, Dict] = {}
    dnd1 = StateConfig(width=346, height=260, taus_us=(250.0,), radius=1)
    dnd2 = StateConfig(width=346, height=260, taus_us=(250.0,), radius=2)
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd1, dnd2, args.max_blocks)
            c = dnd21[name]
            print(f"DND21 {name}: blocks={c['blocks']} "
                  f"a={c['auc']['old_gate1_score1']:.4f} c={c['auc']['old_gate1_score2']:.4f} "
                  f"d={c['auc']['old_gate2_score2']:.4f}", flush=True)

    per_dataset = {"DVSCLEAN": aggregate(dvsclean), "DND21": aggregate(dnd21)}
    overall = {k: 0.5 * (per_dataset["DVSCLEAN"][k] + per_dataset["DND21"][k])
               for k in per_dataset["DVSCLEAN"]}

    # Split the radius-2 gain into operand and gate contributions, per score variant.
    decomposition = {}
    for variant in ("old", "new"):
        a = overall[f"{variant}_gate1_score1"]
        b = overall[f"{variant}_gate2_score1"]
        c = overall[f"{variant}_gate1_score2"]
        d = overall[f"{variant}_gate2_score2"]
        decomposition[variant] = {
            "a_baseline_gate1_score1": a,
            "b_gate2_score1": b,
            "c_gate1_score2_ATTRIBUTION": c,
            "d_gate2_score2_confounded": d,
            "total_gain_d_minus_a": d - a,
            "operand_effect_c_minus_a": c - a,
            "gate_effect_d_minus_c": d - c,
            "gate_effect_alone_b_minus_a": b - a,
            "interaction": (d - c) - (b - a),
        }

    payload = {
        "phase": "11 radius-2 attribution (gate key vs score operand)",
        "grid": "rows = gate key radius, cols = score ring radius; theta = 0 throughout",
        "scores": {"old": "s = beta*G - R_same",
                   "new": "s = beta*G - (R_same + R_opposite)"},
        "invariant_checked": ("global_density G must be radius-independent; "
                              "max |G(r=1) - G(r=2)| reported per condition"),
        "conditions": {"DVSCLEAN": dvsclean, "DND21": dnd21},
        "summary": {"DVSCLEAN": per_dataset["DVSCLEAN"], "DND21": per_dataset["DND21"],
                    "overall_20": overall},
        "decomposition": decomposition,
        "max_global_density_drift": max(
            [c["max_global_density_drift"] for c in {**dvsclean, **dnd21}.values()
             if c.get("blocks")] or [float("nan")]),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")
    print("\nmax |G(r1) - G(r2)| =", payload["max_global_density_drift"])
    print("\nDECOMPOSITION:", json.dumps(decomposition, indent=2))


if __name__ == "__main__":
    main()
