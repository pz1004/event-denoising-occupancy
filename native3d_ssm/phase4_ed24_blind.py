"""Phase 4: blind validation of the corr(G,R) crossover on ED24, then test.

The corr<0 gate (beta = 1 if corr(G,R) < 0 else 0) is parameter-free in theory.
Phase 4 checks that this is not a DND21/DVSCLEAN coincidence:

  1. ED24 (dev, fully disjoint from both test sets) spans sparse->dense noise
     levels, so it should span both corr(G,R) signs within one dataset.
  2. On ED24, does sign(corr) predict which beta wins per block/condition?
  3. Fit the gate threshold on ED24 ONLY (maximize ED24 gated AUC) -> theta*.
     If theta* ~ 0, the derived corr=0 crossover is validated on independent data.
  4. Apply the ED24-frozen theta* (and the theory value 0) BLIND to DND21 and
     DVSCLEAN -- labels of the test sets never touch the threshold.

Read-only; beta is a fixed function. Only AUC at beta=0 and beta=1 is needed
because the gate selects between them per block.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .data import iter_labeled_text
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks, per_block_operands_both
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ED24_ROOT = PROJECT_ROOT / "ECCV2024_datasets/ED24"
ED24_LEVELS = ["0.1", "0.3", "0.5", "1.0", "2.0", "4.0"]


def scan_min(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]], config: StateConfig,
             max_blocks: Optional[int]) -> Optional[Dict]:
    """Per block: corr(G,R_same), plus AUC at beta=0 and beta=1 for both scores.

    Two score variants are evaluated on the *same* blocks, so the comparison is exactly
    paired and the old rows double as a regression check on this plumbing:

        old (Phase 3/4):  s = beta*G - R_same
        new (Phase 8):    s = beta*G - (R_same + R_opposite)

    The gate statistic stays corr(G, R_same) for both -- frozen by decision 2.4, and
    deliberately NOT updated to the new ring (keying on corr(G,R_both) costs 0.0058 AUC).
    """

    sc = len(config.taus_us)
    width = config.feature_dim
    corr, auc0, auc1, auc0_new, auc1_new = [], [], [], [], []
    produced = 0
    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        if labels.min() == labels.max():
            produced += 1
            continue
        raw = bidirectional_features(events, config)
        g, r, r_opposite = per_block_operands_both(raw, sc, width)
        r_both = r + r_opposite
        corr.append(float(np.corrcoef(g, r)[0, 1]) if g.std() > 0 and r.std() > 0 else 0.0)
        auc0.append(roc_auc_score(labels, -r))          # beta = 0
        auc1.append(roc_auc_score(labels, g - r))        # beta = 1
        auc0_new.append(roc_auc_score(labels, -r_both))
        auc1_new.append(roc_auc_score(labels, g - r_both))
        produced += 1
    if not auc0:
        return None
    return {"corr": np.asarray(corr),
            "auc0": np.asarray(auc0), "auc1": np.asarray(auc1),
            "auc0_new": np.asarray(auc0_new), "auc1_new": np.asarray(auc1_new)}


def ed24_blocks(scene: str, level: str, block_size: int, max_blocks: Optional[int]):
    path = ED24_ROOT / scene / f"{scene}_{level}.csv"
    produced = 0
    for values in iter_labeled_text(path, block_size):
        if len(values) != block_size:
            break
        if max_blocks is not None and produced >= max_blocks:
            break
        yield values[:, :4], values[:, 4]
        produced += 1


def gated_auc(conditions: Dict[str, Dict], threshold: float, suffix: str = "") -> float:
    """Mean over conditions of mean-over-blocks AUC under beta=1 iff corr<threshold.

    `suffix` selects the score variant: "" = old (beta*G - R_same),
    "_new" = Phase 8 (beta*G - (R_same + R_opposite)).
    """

    means = []
    for c in conditions.values():
        chosen = np.where(c["corr"] < threshold, c[f"auc1{suffix}"], c[f"auc0{suffix}"])
        means.append(float(np.mean(chosen)))
    return float(np.mean(means))


def fit_threshold(conditions: Dict[str, Dict], suffix: str = "") -> Dict[str, float]:
    """Fit theta on these conditions.

    Returns the max gated AUC, the LARGEST theta achieving it (the identifiable
    upper edge of the optimal plateau -- meaningful when the set is single-regime),
    and the plateau width. If the dev set spans only one regime the plateau is
    wide and theta is only bounded from one side.
    """

    grid = np.round(np.arange(-0.40, 0.201, 0.01), 3)
    scores = np.array([gated_auc(conditions, float(t), suffix) for t in grid])
    best = float(scores.max())
    at_max = grid[scores >= best - 1e-9]
    return {
        "best_gated_auc": best,
        "theta_upper_edge": float(at_max.max()),
        "theta_lower_edge": float(at_max.min()),
        "plateau_width": float(at_max.max() - at_max.min()),
    }


def dataset_endpoints(conditions: Dict[str, Dict], suffix: str = "") -> Dict[str, float]:
    a0 = np.mean([np.mean(c[f"auc0{suffix}"]) for c in conditions.values()])
    a1 = np.mean([np.mean(c[f"auc1{suffix}"]) for c in conditions.values()])
    return {"beta0_ring_only": float(a0), "beta1_contrast": float(a1)}


def corr_predicts_winner(conditions: Dict[str, Dict], suffix: str = "") -> Dict[str, float]:
    """Per-block: does sign(corr) agree with which beta actually wins?"""

    agree = total = 0
    neg_corr_beta1_better = neg = 0
    for c in conditions.values():
        beta1_better = c[f"auc1{suffix}"] > c[f"auc0{suffix}"]
        pred_beta1 = c["corr"] < 0.0
        agree += int(np.sum(pred_beta1 == beta1_better))
        total += len(c["corr"])
        neg += int(np.sum(c["corr"] < 0))
        neg_corr_beta1_better += int(np.sum((c["corr"] < 0) & beta1_better))
    return {
        "block_sign_agreement": agree / total,
        "frac_blocks_corr_negative": neg / total,
        "P(beta1_better | corr<0)": (neg_corr_beta1_better / neg) if neg else float("nan"),
    }


def main() -> None:
    started = time.perf_counter()
    dnd_cfg = StateConfig(width=346, height=260, taus_us=(250.0,))
    dvs_cfg = StateConfig(width=1280, height=720, taus_us=(250.0,))

    # ED24 dev set: a spread of scenes x noise levels.
    scenes = sorted(p.name for p in ED24_ROOT.iterdir() if p.is_dir())[::6][:16]
    ed24 = {}
    corr_by_level = {lvl: [] for lvl in ED24_LEVELS}
    for scene in scenes:
        for level in ED24_LEVELS:
            res = scan_min(ed24_blocks(scene, level, 10240, 8), dnd_cfg, 8)
            if res is not None:
                ed24[f"{scene}_{level}"] = res
                corr_by_level[level].append(float(np.mean(res["corr"])))
        print(f"ED24 {scene}: done", flush=True)

    theta = 0.0  # theory crossover; ED24 is single-regime so it only bounds theta from above
    ed24_fit = fit_threshold(ed24)
    ed24_fit_new = fit_threshold(ed24, "_new")
    ed24_end = dataset_endpoints(ed24)
    ed24_end_new = dataset_endpoints(ed24, "_new")
    ed24_pred = corr_predicts_winner(ed24)
    ed24_pred_new = corr_predicts_winner(ed24, "_new")
    print(f"\nED24 fit (old score): {ed24_fit}", flush=True)
    print(f"ED24 fit (NEW score): {ed24_fit_new}", flush=True)
    print(f"Using theta=0 (theory; nothing is adopted from the ED24 fit). "
          f"ED24 old beta0={ed24_end['beta0_ring_only']:.4f} beta1={ed24_end['beta1_contrast']:.4f} "
          f"| NEW beta0={ed24_end_new['beta0_ring_only']:.4f} "
          f"beta1={ed24_end_new['beta1_contrast']:.4f}", flush=True)
    print("corr(G,R) mean by ED24 noise level:",
          {lvl: round(float(np.mean(v)), 3) for lvl, v in corr_by_level.items() if v}, flush=True)
    print("ED24 corr predicts winner (old):", {k: round(v, 3) for k, v in ed24_pred.items()}, flush=True)
    print("ED24 corr predicts winner (NEW):", {k: round(v, 3) for k, v in ed24_pred_new.items()}, flush=True)

    # Test sets (labels never used to set theta).
    dnd = {}
    for rate in ("1hz", "3hz", "5hz", "7hz", "10hz"):
        for scene, fname in SCENE_FILES.items():
            dnd[f"{rate}_{scene}"] = scan_min(
                dnd21_blocks(PROJECT_ROOT / f"ECCV2024_datasets/AUC_test/{rate}/{fname}", 10240, None),
                dnd_cfg, None)
        print(f"DND21 {rate}: done", flush=True)
    dvs = {}
    for seq in SEQUENCES:
        for lvl in (50, 100):
            dvs[f"{seq}_{lvl}"] = scan_min(
                dvsclean_blocks(PROJECT_ROOT / f"DVSCLEAN/simulated_data/{seq}_{lvl}.hdf5", 10240, None),
                dvs_cfg, None)
        print(f"DVSCLEAN {seq}: done", flush=True)

    def report(conds, suffix: str, fit: Dict[str, float]):
        end = dataset_endpoints(conds, suffix)
        return {
            "beta0_ring_only": end["beta0_ring_only"],
            "beta1_contrast": end["beta1_contrast"],
            "gate_theta0_theory": gated_auc(conds, 0.0, suffix),
            "gate_theta_ED24_upper_edge": gated_auc(conds, fit["theta_upper_edge"], suffix),
        }

    def blind_block(suffix: str, fit: Dict[str, float]):
        block = {"DND21": report(dnd, suffix, fit), "DVSCLEAN": report(dvs, suffix, fit)}
        block["overall_theta0"] = 0.5 * (block["DND21"]["gate_theta0_theory"]
                                        + block["DVSCLEAN"]["gate_theta0_theory"])
        return block

    result = {
        "phase": "4 (re-validated on the Phase 8 score)",
        "old_score": "s = beta*G - R_same",
        "new_score": "s = beta*G - (R_same + R_opposite)",
        "gate_key": "corr(G, R_same) for BOTH scores -- frozen by decision 2.4",
        "theta_note": ("theta = 0 is the theory value and is what is applied blind. The ED24 fit "
                       "is reported as a bound only; nothing is adopted from it, so ED24 remains "
                       "unburned as a source of fitted parameters."),
        "ed24": {
            "scenes": scenes, "levels": ED24_LEVELS,
            "theta_fit": ed24_fit, "theta_fit_new": ed24_fit_new, "theta_used": theta,
            "endpoints": ed24_end, "endpoints_new": ed24_end_new,
            "corr_predicts_winner": ed24_pred, "corr_predicts_winner_new": ed24_pred_new,
            "corr_mean_by_level": {lvl: float(np.mean(v)) for lvl, v in corr_by_level.items() if v},
        },
        "blind_test": blind_block("", ed24_fit),
        "blind_test_new_score": blind_block("_new", ed24_fit_new),
    }
    result["verdict"] = {
        "old_overall_theta0": result["blind_test"]["overall_theta0"],
        "new_overall_theta0": result["blind_test_new_score"]["overall_theta0"],
        "gain_new_minus_old": (result["blind_test_new_score"]["overall_theta0"]
                               - result["blind_test"]["overall_theta0"]),
    }
    result["wall_seconds"] = time.perf_counter() - started

    out = (PROJECT_ROOT
           / "native3d_ssm/experiments/analytic_contrast/phase4_ed24_blind_phase8score.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("\nOLD score:\n" + json.dumps(result["blind_test"], indent=2))
    print("\nNEW score:\n" + json.dumps(result["blind_test_new_score"], indent=2))
    print("\nVERDICT:", json.dumps(result["verdict"], indent=2))
    print(f"\nWrote {out} in {result['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
