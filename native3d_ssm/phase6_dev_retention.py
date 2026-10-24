"""Phase 6a: calibrate a label-free retention rule on the labeled development sets.

Phase 5 exposed retention `r` as a first-class, un-adapted operating point: MESR on
E-MLB peaks near r = 0.5 on Daylight but near r = 0.3 on Night, so one constant is
wrong somewhere. E-MLB itself cannot calibrate `r` honestly -- MESR is a
no-reference metric, so choosing `r` to maximize MESR is label-free yet is direct
optimization of the very number being reported.

So the rule is fitted here instead, on the two *labeled* development sets
(DND21 x 10 conditions, DVSCLEAN x 10 conditions), and frozen before it ever sees
E-MLB. Score is unchanged from Phase 3/4: beta = 1 iff corr(G,R) < 0, s = beta*G - R,
250 us bidirectional, 10,240-event blocks.

Rows compared, per block, on balanced accuracy (label 1 = noise; the lowest-scoring
`r` fraction is kept):

  oracle_per_block      label-using argmax over all prefix sizes -- a ceiling
  true_signal_fraction  r = 1 - noise fraction; the theory-preferred target
  gated                 corr<0 -> otsu, corr>=0 -> below_top   <- candidate rule
  otsu                  label-free, zero-parameter, per block
  below_top             label-free: reject the "no evidence" atom at the top score
  gmm2                  label-free, 2-component 1-D mixture (slow; reference only)
  fixed_best_dev        the single best constant r fitted on dev *with* labels --
                        the honest non-adaptive competitor the rule must beat
  fixed_0.3 ... 0.9     the Phase 5 Table 2 rows

The gate is needed because the informative statistic is regime-dependent, just
like the sign of the global operand: on DND21 (corr < 0, beta = 1) the score is
continuous and Otsu works while `below_top` degenerates to "keep everything"; on
DVSCLEAN (corr > 0, beta = 0) the score collapses onto an atom at 0 holding
40-60% of the block, where `below_top` recovers the signal fraction and Otsu cuts
far too deep into the negative tail.

Diagnostics: mean r per row, MAE against the true signal fraction, MAE against the
oracle retention. Read-only over data on disk; no model is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import (
    dnd21_blocks,
    dvsclean_blocks,
    per_block_operands_both,
    ring_support,
)
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .retention import (
    balanced_accuracy_curve,
    below_top_retention,
    gated_retention,
    gmm2_retention,
    otsu_retention,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 10_240
FIXED_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)   # candidates for fixed_best_dev
PHASE5_FIXED = (0.3, 0.5, 0.7, 0.9)                      # the Table 2 rows
LABEL_FREE_ROWS = ("gated", "otsu", "below_top")          # gmm2 added only when enabled


def _accuracy_at(curve: np.ndarray, retention: float) -> float:
    n_events = curve.size
    k = int(min(n_events, max(1, round(retention * n_events))))
    return float(curve[k - 1])


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, with_gmm: bool,
                   max_blocks: Optional[int] = None, score: str = "new",
                   score_config: Optional[StateConfig] = None) -> Dict:
    """Per-block retention estimates and the balanced accuracy each one realizes.

    `score` selects the subtrahend: "old" = R_same (Phase 3/4), "new" = R_same + R_opposite
    (Phase 8). The gate statistic is corr(G, R_same) either way -- frozen by decision 2.4.

    `score_config`, when given, supplies the ring used by the *score* while `config` still
    supplies the ring used by the *gate*. That is the Phase 11 result: the radius-2 gain is
    entirely the operand, and widening the gate key costs 0.0028, so the score may use
    radius 2 while the gate key stays at radius 1.
    """

    scale_count = len(config.taus_us)
    feature_width = config.feature_dim
    records: List[Dict[str, float]] = []
    grid_accuracy: List[np.ndarray] = []
    produced = 0
    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        raw = bidirectional_features(events, config)
        g, r, r_opposite = per_block_operands_both(raw, scale_count, feature_width)
        corr = float(np.corrcoef(g, r)[0, 1]) if g.std() > 0 and r.std() > 0 else 0.0
        beta = 1.0 if corr < 0.0 else 0.0
        raw_s = None
        if score_config is not None:
            raw_s = bidirectional_features(events, score_config)
            g, r, r_opposite = per_block_operands_both(raw_s, scale_count,
                                                       score_config.feature_dim)
        if score == "support":
            cfg_s = score_config if score_config is not None else config
            support = ring_support(raw_s if score_config is not None else raw,
                                   scale_count, cfg_s.feature_dim, cfg_s.radius)
            scores = beta * g - np.log1p(support)
        else:
            scores = beta * g - (r if score == "old" else r + r_opposite)
        curve = balanced_accuracy_curve(scores, labels)
        if curve is None:                      # single-class block: no operating point
            continue
        oracle_index = int(np.argmax(curve))
        record = {
            "corr": corr,
            "beta": beta,
            "signal_fraction": float(1.0 - np.mean(labels)),
            "r_oracle": (oracle_index + 1) / curve.size,
            "acc_oracle": float(curve[oracle_index]),
            "r_gated": gated_retention(scores, corr),
            "r_otsu": otsu_retention(scores),
            "r_below_top": below_top_retention(scores),
        }
        if with_gmm:
            record["r_gmm2"] = gmm2_retention(scores)
        for row in (LABEL_FREE_ROWS + (("gmm2",) if with_gmm else ())):
            record[f"acc_{row}"] = _accuracy_at(curve, record[f"r_{row}"])
        record["acc_signal_fraction"] = _accuracy_at(curve, record["signal_fraction"])
        records.append(record)
        grid_accuracy.append(np.array([_accuracy_at(curve, float(r_fixed))
                                       for r_fixed in FIXED_GRID]))
    if not records:
        return {"blocks": 0}

    def mean(key: str) -> float:
        return float(np.mean([rec[key] for rec in records]))

    def mae(key: str, reference: str) -> float:
        return float(np.mean([abs(rec[key] - rec[reference]) for rec in records]))

    summary = {
        "blocks": len(records),
        "mean_corr": mean("corr"),
        "frac_blocks_beta1": mean("beta"),
        "mean_signal_fraction": mean("signal_fraction"),
        "mean_r_oracle": mean("r_oracle"),
        "acc_oracle": mean("acc_oracle"),
        "acc_signal_fraction": mean("acc_signal_fraction"),
        "acc_by_fixed_r": np.mean(np.vstack(grid_accuracy), axis=0),   # kept as array
    }
    for row in (LABEL_FREE_ROWS + (("gmm2",) if with_gmm else ())):
        summary[f"mean_r_{row}"] = mean(f"r_{row}")
        summary[f"acc_{row}"] = mean(f"acc_{row}")
        summary[f"mae_{row}_vs_signal_fraction"] = mae(f"r_{row}", "signal_fraction")
        summary[f"mae_{row}_vs_oracle"] = mae(f"r_{row}", "r_oracle")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=None,
                        help="Per-condition block cap (debug only; default = all).")
    parser.add_argument("--no-gmm", action="store_true",
                        help="Skip the slow GMM reference row.")
    parser.add_argument("--score-radius", type=int, default=1,
                        help="Ring radius used by the SCORE. The gate key always stays at "
                             "radius 1 (Phase 11: widening the gate costs 0.0028).")
    parser.add_argument("--score", choices=("old", "new", "support"), default="new",
                        help="old = beta*G - R_same (Phase 3/4); new = beta*G - (R_same + R_opposite) "
                             "(Phase 8). Gate key is corr(G,R_same) either way.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase6_dev_retention_phase8score.json")
    return parser.parse_args()


def aggregate(conditions: Dict[str, Dict], keys: List[str]) -> Dict[str, float]:
    """Unweighted mean over conditions (the convention used in Phases 2'-5)."""

    usable = [c for c in conditions.values() if c.get("blocks")]
    return {k: float(np.mean([c[k] for c in usable])) for k in keys if all(k in c for c in usable)}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    with_gmm = not args.no_gmm

    dvs_config = StateConfig(width=1280, height=720, taus_us=(250.0,))
    dnd_config = StateConfig(width=346, height=260, taus_us=(250.0,))
    # Score ring may be wider than the gate ring (Phase 11). None = same as the gate ring.
    dvs_score_config = dnd_score_config = None
    if args.score_radius != 1:
        dvs_score_config = StateConfig(width=1280, height=720, taus_us=(250.0,),
                                       radius=args.score_radius)
        dnd_score_config = StateConfig(width=346, height=260, taus_us=(250.0,),
                                       radius=args.score_radius)

    dvsclean: Dict[str, Dict] = {}
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            path = args.dvsclean_root / f"{name}.hdf5"
            dvsclean[name] = scan_condition(
                dvsclean_blocks(path, BLOCK_SIZE, args.max_blocks),
                dvs_config, with_gmm, args.max_blocks, args.score, dvs_score_config)
            c = dvsclean[name]
            print(f"DVSCLEAN {name}: blocks={c['blocks']} f_sig={c['mean_signal_fraction']:.3f} "
                  f"r: gated={c['mean_r_gated']:.3f} otsu={c['mean_r_otsu']:.3f} "
                  f"oracle={c['mean_r_oracle']:.3f} | acc: gated={c['acc_gated']:.4f} "
                  f"otsu={c['acc_otsu']:.4f} oracle={c['acc_oracle']:.4f}", flush=True)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            path = args.dnd21_root / rate / fname
            dnd21[name] = scan_condition(
                dnd21_blocks(path, BLOCK_SIZE, args.max_blocks),
                dnd_config, with_gmm, args.max_blocks, args.score, dnd_score_config)
            c = dnd21[name]
            print(f"DND21 {name}: blocks={c['blocks']} f_sig={c['mean_signal_fraction']:.3f} "
                  f"r: gated={c['mean_r_gated']:.3f} otsu={c['mean_r_otsu']:.3f} "
                  f"oracle={c['mean_r_oracle']:.3f} | acc: gated={c['acc_gated']:.4f} "
                  f"otsu={c['acc_otsu']:.4f} oracle={c['acc_oracle']:.4f}", flush=True)

    rows = LABEL_FREE_ROWS + (("gmm2",) if with_gmm else ())
    scalar_keys = ["mean_corr", "frac_blocks_beta1", "mean_signal_fraction",
                   "mean_r_oracle", "acc_oracle", "acc_signal_fraction"]
    for row in rows:
        scalar_keys += [f"mean_r_{row}", f"acc_{row}",
                        f"mae_{row}_vs_signal_fraction", f"mae_{row}_vs_oracle"]

    all_conditions = {**{f"DVSCLEAN/{k}": v for k, v in dvsclean.items()},
                      **{f"DND21/{k}": v for k, v in dnd21.items()}}

    def fixed_curve(conditions: Dict[str, Dict]) -> np.ndarray:
        usable = [c["acc_by_fixed_r"] for c in conditions.values() if c.get("blocks")]
        return np.mean(np.vstack(usable), axis=0)

    overall_fixed = fixed_curve(all_conditions)
    best_index = int(np.argmax(overall_fixed))
    fixed_best_dev = float(FIXED_GRID[best_index])

    summary = {
        "DVSCLEAN": aggregate(dvsclean, scalar_keys),
        "DND21": aggregate(dnd21, scalar_keys),
        "overall_20": aggregate(all_conditions, scalar_keys),
        "fixed_r_curve_overall": {str(r): float(a) for r, a in zip(FIXED_GRID, overall_fixed)},
        "fixed_r_curve_DVSCLEAN": {str(r): float(a) for r, a in zip(FIXED_GRID, fixed_curve(dvsclean))},
        "fixed_r_curve_DND21": {str(r): float(a) for r, a in zip(FIXED_GRID, fixed_curve(dnd21))},
        "fixed_best_dev": {"r": fixed_best_dev, "acc_overall_20": float(overall_fixed[best_index])},
        "phase5_fixed_rows": {
            str(r): {"overall_20": float(overall_fixed[int(np.argmin(np.abs(FIXED_GRID - r)))]),
                     "DVSCLEAN": float(fixed_curve(dvsclean)[int(np.argmin(np.abs(FIXED_GRID - r)))]),
                     "DND21": float(fixed_curve(dnd21)[int(np.argmin(np.abs(FIXED_GRID - r)))])}
            for r in PHASE5_FIXED},
    }

    # Which label-free row wins on dev, and does it beat the best fitted constant?
    overall = summary["overall_20"]
    ranked = sorted(((row, overall[f"acc_{row}"]) for row in rows),
                    key=lambda pair: -pair[1])
    summary["verdict"] = {
        "ranked_label_free_rows": [{"row": row, "acc_overall_20": acc} for row, acc in ranked],
        "winner": ranked[0][0],
        "winner_acc_overall_20": ranked[0][1],
        "beats_fixed_best_dev": bool(ranked[0][1] > summary["fixed_best_dev"]["acc_overall_20"]),
        "margin_over_fixed_best_dev": ranked[0][1] - summary["fixed_best_dev"]["acc_overall_20"],
        "gap_to_oracle": overall["acc_oracle"] - ranked[0][1],
    }

    def serialize(conditions: Dict[str, Dict]) -> Dict[str, Dict]:
        return {name: {k: v for k, v in c.items() if not isinstance(v, np.ndarray)}
                for name, c in conditions.items()}

    payload = {
        "phase": "6a dev calibration of label-free retention",
        "score_variant": args.score,
        "score_radius": args.score_radius,
        "gate_radius": 1,
        "score": ("frozen corr gate: beta=1 iff corr(G,R_same)<0, 250us bidirectional, "
                  "10,240-event blocks; s = beta*G - "
                  + {"old": "R_same", "new": "(R_same + R_opposite)",
                     "support": "log1p(ring_support)"}[args.score]),
        "metric": "balanced accuracy at the chosen retention (label 1 = noise; lowest-scoring r fraction kept)",
        "note": ("ROC AUC is invariant to retention, so this is purely the Track-B "
                 "operating-point calibration. Rule is frozen here and applied blind to E-MLB in Phase 6b."),
        "fixed_grid": FIXED_GRID.tolist(),
        "conditions": {"DVSCLEAN": serialize(dvsclean), "DND21": serialize(dnd21)},
        "summary": summary,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
