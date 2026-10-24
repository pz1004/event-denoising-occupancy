"""Phase 26: why rho = 0 is the right gate threshold, derived rather than asserted.

The manuscript has always justified the gate structurally: theta = 0 is the one value fixed by
the hypothesis rather than by data. That is honest but it is not a derivation, and Section 3.4
says so. This phase supplies the missing step and then checks it against the data.

**The claim.** Write pi for the noise fraction in a block and, for any operand X,
    Delta X = E[X | noise] - E[X | signal].
The score is s = beta*G - R, and it ranks noise above signal by the between-class separation
    Delta s = beta * Delta G - Delta R.
Since Delta R < 0 uniformly (the ring term is sign-invariant -- Section 3.3), -Delta R > 0 always
contributes, and the only question is the sign of Delta G: adding G helps iff Delta G > 0.

By the law of total covariance, over the two-class mixture,
    Cov(G, R) = E[Cov(G, R | class)]  +  pi (1 - pi) * Delta G * Delta R
              =        C_within       +           C_between.
Empirically C_within > 0 always (both operands rise together with instantaneous activity inside
either class), and Delta R < 0 always. Therefore

    Delta G > 0   =>  C_between < 0, and rho < 0 once |C_between| > C_within;
    Delta G < 0   =>  C_between > 0, so rho > 0;
    Delta G -> 0  =>  C_between -> 0, so rho -> C_within > 0.

So `beta = 1 iff rho < 0` is exactly `beta = 1 iff G separates noise upward by enough to
overcome the within-class covariance`. The threshold sits at zero because zero is where the
between-class term changes sign, i.e. where G stops carrying between-class information at all.
The degenerate case is the useful one: as Delta G -> 0, G becomes uninformative and rho falls
back on C_within > 0, which switches the gate OFF -- the safe default, reached automatically.

This also explains why rho is a better gate statistic than Delta G, even though Delta G is what
the decision actually depends on: Delta G needs labels, rho does not, and rho degrades to the
safe branch precisely where Delta G is too small to be worth acting on.

Read-only; nothing is trained or fitted. Labels are used ONLY to compute the diagnostic
quantities Delta G, Delta R and pi that the derivation refers to -- the gate itself never sees
them.
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
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK_SIZE = 10_240
MIN_PER_CLASS = 50


def block_terms(events: np.ndarray, labels: np.ndarray, config: StateConfig) -> Optional[Dict]:
    """The decomposition for one block. Label 1 is noise (verified by AUC sign below)."""

    raw = bidirectional_features(events, config)
    g, r, _ = per_block_operands_both(raw, len(config.taus_us), config.feature_dim)
    noise, signal = labels == 1, labels == 0
    if noise.sum() < MIN_PER_CLASS or signal.sum() < MIN_PER_CLASS:
        return None
    pi = float(noise.mean())
    delta_g = float(g[noise].mean() - g[signal].mean())
    delta_r = float(r[noise].mean() - r[signal].mean())
    c_within = float(pi * np.cov(g[noise], r[noise])[0, 1]
                     + (1.0 - pi) * np.cov(g[signal], r[signal])[0, 1])
    c_between = float(pi * (1.0 - pi) * delta_g * delta_r)
    if g.std() <= 0 or r.std() <= 0:
        return None
    return {
        "rho": float(np.corrcoef(g, r)[0, 1]),
        "delta_G": delta_g, "delta_R": delta_r, "pi_noise": pi,
        "C_within": c_within, "C_between": c_between,
        # which beta is actually better here, by AUC on this block
        "auc_beta1_minus_beta0": float(roc_auc_score(labels, g - r)
                                       - roc_auc_score(labels, -r)),
    }


def scan(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]], config: StateConfig,
         max_blocks: Optional[int]) -> Dict:
    rows: List[Dict] = []
    for produced, (events, labels) in enumerate(block_iter):
        if max_blocks is not None and produced >= max_blocks:
            break
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue
        row = block_terms(events, labels, config)
        if row is not None:
            rows.append(row)
    if not rows:
        return {}
    keys = rows[0].keys()
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    mean["blocks"] = len(rows)
    # Agreement rates, per block rather than per condition: the gate is a per-block decision.
    mean["blocks_theory_matches_implemented"] = int(sum(
        (r["delta_G"] > 0) == (r["rho"] < 0) for r in rows))
    mean["blocks_implemented_picks_better_beta"] = int(sum(
        (r["rho"] < 0) == (r["auc_beta1_minus_beta0"] > 0) for r in rows))
    mean["blocks_delta_R_negative"] = int(sum(r["delta_R"] < 0 for r in rows))
    mean["blocks_C_within_positive"] = int(sum(r["C_within"] > 0 for r in rows))
    # Where does the rule disagree with the oracle? The derivation predicts: exactly where the
    # between-class term fails to dominate the within-class one, since that is the only way
    # sign(rho) can come apart from -sign(Delta G).
    agree = [r for r in rows if (r["delta_G"] > 0) == (r["rho"] < 0)]
    differ = [r for r in rows if (r["delta_G"] > 0) != (r["rho"] < 0)]
    def ratio(rs: List[Dict]) -> Optional[float]:
        v = [abs(r["C_between"]) / max(r["C_within"], 1e-12) for r in rs if r["C_within"] > 0]
        return float(np.median(v)) if v else None
    mean["dominance_ratio_median_when_agree"] = ratio(agree)
    mean["dominance_ratio_median_when_differ"] = ratio(differ)
    mean["abs_delta_G_median_when_agree"] = (
        float(np.median([abs(r["delta_G"]) for r in agree])) if agree else None)
    mean["abs_delta_G_median_when_differ"] = (
        float(np.median([abs(r["delta_G"]) for r in differ])) if differ else None)
    return mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase26_gate_derivation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dnd = StateConfig(width=346, height=260, taus_us=(250.0,), radius=1)
    dvs = StateConfig(width=1280, height=720, taus_us=(250.0,), radius=1)

    conditions: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"DND21 {rate} {scene}"
            conditions[name] = scan(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd, args.max_blocks)
            print(f"{name}: rho={conditions[name]['rho']:+.3f} "
                  f"dG={conditions[name]['delta_G']:+.3f} "
                  f"dR={conditions[name]['delta_R']:+.4f}", flush=True)
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"DVSCLEAN {seq}_{level}"
            conditions[name] = scan(
                dvsclean_blocks(args.dvsclean_root / f"{seq}_{level}.hdf5",
                                BLOCK_SIZE, args.max_blocks),
                dvs, args.max_blocks)
            print(f"{name}: rho={conditions[name]['rho']:+.3f} "
                  f"dG={conditions[name]['delta_G']:+.3f} "
                  f"dR={conditions[name]['delta_R']:+.4f}", flush=True)

    usable = [c for c in conditions.values() if c]
    total_blocks = sum(c["blocks"] for c in usable)
    summary = {
        "conditions": len(usable),
        "blocks": total_blocks,
        "assumption_delta_R_negative": {
            "blocks": sum(c["blocks_delta_R_negative"] for c in usable), "of": total_blocks},
        "assumption_C_within_positive": {
            "blocks": sum(c["blocks_C_within_positive"] for c in usable), "of": total_blocks},
        "theory_matches_implemented": {
            "blocks": sum(c["blocks_theory_matches_implemented"] for c in usable),
            "of": total_blocks},
        "implemented_picks_better_beta": {
            "blocks": sum(c["blocks_implemented_picks_better_beta"] for c in usable),
            "of": total_blocks},
        "conditions_delta_G_positive": sum(1 for c in usable if c["delta_G"] > 0),
        "conditions_rho_negative": sum(1 for c in usable if c["rho"] < 0),
    }
    for key in ("assumption_delta_R_negative", "assumption_C_within_positive",
                "theory_matches_implemented", "implemented_picks_better_beta"):
        d = summary[key]
        d["fraction"] = d["blocks"] / d["of"] if d["of"] else None

    payload = {
        "phase": "26 derivation of the rho = 0 gate threshold",
        "claim": ("beta = 1 iff Delta G > 0; and because Delta R < 0 uniformly and C_within > 0, "
                  "sign(Cov(G,R)) tracks -sign(Delta G), so 'beta = 1 iff rho < 0' implements "
                  "'beta = 1 iff G separates noise upward' without ever seeing a label"),
        "decomposition": "Cov(G,R) = C_within + pi(1-pi) * Delta_G * Delta_R",
        "label_convention": ("label 1 = noise: the adopted score is high for label 1 and "
                             "retention keeps the lowest-scoring fraction"),
        "note": ("labels are used here only to expose Delta G, Delta R and pi, which the "
                 "derivation refers to. The gate itself is computed from rho alone."),
        "summary": summary,
        "conditions": conditions,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
