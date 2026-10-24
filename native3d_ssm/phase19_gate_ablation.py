"""Phase 19: what the gate actually buys once the support operand is adopted.

Motivation (REVIEW_theory_empirics.md, P1-3). The draft motivates the beta gate with the
*magnitude* operand, where ungating collapses DVSCLEAN from 0.9286 to 0.5198. But that is no
longer the adopted score. Phase 5b already showed `sup_beta0_ring_only` = 0.9627 overall against
`sup_batch_gate` = 0.9811 -- and on DVSCLEAN the two agree to four decimals, because the gate
always selects beta = 0 there. The ungated-always-on variant of the adopted score
(`beta == 1` with the support operand) was never measured at all.

This phase fills the 2x3 cell grid on the same 20 development conditions and the same blocks:

    operand in {R_same (magnitude), log1p(support)}  x  beta in {gated, 0, 1}

so Table 3 can state the gate's contribution for the score the paper actually adopts, rather than
for the score it used to adopt. Float precision only -- Phase 13/14 already cover fixed point.

Read-only. Nothing is trained, fitted or selected here. Gate key stays radius 1, score radius 2,
block size 10,240, all frozen from Phases 8/11/14.

Audit: the `log_support|gated` column must reproduce phase14_quantizable_operand.json's
`log_support|float`, and `current|gated` must reproduce its `current|float`, condition by
condition. Both equalities are asserted in the payload.
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
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks, per_block_operands_both
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase14_quantizable_operand import BLOCK_SIZE, ring_support, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BETA_MODES = ("gated", "0", "1")
OPERANDS = ("current", "log_support")


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, score_config: Optional[StateConfig],
                   max_blocks: Optional[int] = None) -> Dict:
    cfg_s = score_config if score_config is not None else config
    scale_count = len(config.taus_us)
    auc: Dict[str, List[float]] = {}
    beta_hist: List[float] = []
    produced = 0

    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue

        raw = bidirectional_features(events, config)
        g, r_gate, _ = per_block_operands_both(raw, scale_count, config.feature_dim)
        corr = safe_corr(g, r_gate)
        beta_gated = 1.0 if corr < 0.0 else 0.0
        beta_hist.append(beta_gated)

        raw_s = bidirectional_features(events, cfg_s) if score_config is not None else raw
        _, r_same, _ = per_block_operands_both(raw_s, scale_count, cfg_s.feature_dim)
        support = ring_support(raw_s, cfg_s)

        subtrahends = {"current": r_same, "log_support": np.log1p(support)}
        betas = {"gated": beta_gated, "0": 0.0, "1": 1.0}
        for op_name, sub in subtrahends.items():
            for b_name, b in betas.items():
                score = b * g - sub
                if score.std() <= 0:
                    continue
                auc.setdefault(f"{op_name}|{b_name}", []).append(roc_auc_score(labels, score))

    if not auc:
        return {"blocks": 0}
    return {"blocks": produced,
            "frac_blocks_beta1": float(np.mean(beta_hist)) if beta_hist else None,
            "auc": {k: float(np.mean(v)) for k, v in auc.items()}}


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
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT
                        / "native3d_ssm/experiments/analytic_contrast/phase19_gate_ablation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    def configs(w: int, h: int):
        gate = StateConfig(width=w, height=h, taus_us=(250.0,), radius=1)
        score = (StateConfig(width=w, height=h, taus_us=(250.0,), radius=args.score_radius)
                 if args.score_radius != 1 else None)
        return gate, score

    dvs_gate, dvs_score = configs(1280, 720)
    dnd_gate, dnd_score = configs(346, 260)

    dvsclean: Dict[str, Dict] = {}
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            dvsclean[name] = scan_condition(
                dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE, args.max_blocks),
                dvs_gate, dvs_score, args.max_blocks)
            print(f"DVSCLEAN {name}: "
                  + " ".join(f"{k}={v:.4f}" for k, v in
                             sorted(dvsclean[name].get("auc", {}).items())), flush=True)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd_gate, dnd_score, args.max_blocks)
            print(f"DND21 {name}: "
                  + " ".join(f"{k}={v:.4f}" for k, v in
                             sorted(dnd21[name].get("auc", {}).items())), flush=True)

    conditions = {"DVSCLEAN": dvsclean, "DND21": dnd21}
    summary = {ds: aggregate(cs) for ds, cs in conditions.items()}
    overall = {}
    names = set(summary["DVSCLEAN"]) & set(summary["DND21"])
    for n in sorted(names):
        overall[n] = float(np.mean([summary["DVSCLEAN"][n], summary["DND21"][n]]))
    summary["overall_20"] = overall

    payload = {
        "phase": "19 beta-gate ablation on the adopted operand (REVIEW P1-3)",
        "question": ("How much does the beta gate contribute once the score subtrahend is "
                     "log1p(ring support) rather than the ring magnitude?"),
        "grid": "operand {current = R_same, log_support = log1p(support)} x beta {gated, 0, 1}",
        "gate": "gated = 1 iff corr(G, R_same) < 0, theta = 0; '0' and '1' force beta",
        "config": {"score_radius": args.score_radius, "gate_radius": 1,
                   "block_size": BLOCK_SIZE, "taus_us": [250.0], "precision": "float64"},
        "audit": ("current|gated must equal phase14 current|float and log_support|gated must "
                  "equal phase14 log_support|float, condition by condition."),
        "conditions": conditions,
        "summary": summary,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {time.perf_counter() - started:.1f}s")
    for n in sorted(overall):
        print(f"  overall_20 {n:24s} {overall[n]:.4f}")


if __name__ == "__main__":
    main()
