"""Phase 22b: theta = 0 on ED24, under the score this paper actually adopts.

Phase 22 answered "what does theta = 0 score on ED24?" for two score variants:

    old (Phase 3/4):  s = beta*G - R_same                (radius-1 ring magnitude)
    new (Phase 8):    s = beta*G - (R_same + R_opposite) (radius-1, both polarities)

Neither is the adopted score, which Phase 14 settled as

    adopted:          s = beta*G - log(1 + S_r2)         (radius-2 integer ring support)

so Section 3.4 of the manuscript was pricing theta = 0 on a configuration the method no
longer uses. Nothing else in the paper cites ED24, and no ED24 artifact carries a
support-operand run, so the number simply did not exist. This phase measures it.

Everything except the score operand is reused verbatim from Phase 4 -- the scene subset,
the block size, the theta grid, `fit_threshold` and `gated_auc` -- so the rows below are
exactly comparable to Phase 22's and the difference is attributable to the operand alone.
The gate key stays corr(G, R_same) at radius 1, frozen by decision 2.4 and not re-tuned
here.

Two pairs are produced, matching the two sentences in Section 3.4:

    ED24:               theta = 0 against theta* fitted on ED24 itself
    DND21 + DVSCLEAN:   theta = 0 against that ED24-fitted theta*, carried unchanged

The transfer rows double as a regression check: at theta = 0 they must reproduce Phase
19's `log_support|gated` condition means (DND21 0.9917, DVSCLEAN 0.9706), because that is
the same score under the same gate. `audit` reports the deviation.

Read-only; nothing is trained. Writes a NEW artifact and does not touch
`phase22_ed24_theta0.json` or `phase4_ed24_blind*.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks, per_block_operands_both
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase4_ed24_blind import ED24_LEVELS, ED24_ROOT, ed24_blocks, fit_threshold, gated_auc
from .phase14_quantizable_operand import ring_support, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK_SIZE = 10_240

#: Phase 19's `log_support|gated` condition means, i.e. the adopted score at theta = 0.
#: The transfer rows must land here; see `audit`.
PHASE19_GATED = {"DND21": 0.9917, "DVSCLEAN": 0.9706}


def scan_support(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                 gate_config: StateConfig, score_config: StateConfig,
                 max_blocks: Optional[int]) -> Optional[Dict]:
    """Per block: corr(G, R_same) at radius 1, plus AUC at beta = 0 and beta = 1.

    Column conventions and the operand itself follow `phase14_quantizable_operand`, so the
    score evaluated here is bit-for-bit the one behind Table 1: G is read from the
    radius-1 extraction (it takes no spatial argument) and the subtrahend is log1p of the
    radius-2 support count.
    """

    scale_count = len(gate_config.taus_us)
    corr, auc0, auc1 = [], [], []
    produced = 0
    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue

        raw = bidirectional_features(events, gate_config)
        g, r_same, _ = per_block_operands_both(raw, scale_count, gate_config.feature_dim)
        raw_score = bidirectional_features(events, score_config)
        subtrahend = np.log1p(ring_support(raw_score, score_config))

        corr.append(safe_corr(g, r_same))
        auc0.append(roc_auc_score(labels, -subtrahend))
        auc1.append(roc_auc_score(labels, g - subtrahend))

    if not auc0:
        return None
    return {"corr": np.asarray(corr),
            "auc0": np.asarray(auc0), "auc1": np.asarray(auc1)}


def endpoints(conditions: Dict[str, Dict]) -> Dict[str, float]:
    return {
        "beta0_ring_only": float(np.mean([np.mean(c["auc0"]) for c in conditions.values()])),
        "beta1_contrast": float(np.mean([np.mean(c["auc1"]) for c in conditions.values()])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-blocks", type=int, default=8,
                        help="per ED24 scene x level; 8 matches Phase 4 and Phase 22")
    parser.add_argument("--skip-transfer", action="store_true",
                        help="ED24 only; skips the DND21/DVSCLEAN carry-over rows")
    parser.add_argument("--levels", nargs="+", default=None,
                        help="ED24 voltages. ED24 defines 21 noise levels, 1.5-3.5V; the paper "
                             "excludes below 1.5V (BA noise too sparse) and above 3.5V (nearly "
                             "uniform). Phase 4's original six put five outside that range.")
    parser.add_argument("--all-scenes", action="store_true",
                        help="score all 100 ED24 scenes instead of Phase 4's [::6][:16] subsample")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase22b_ed24_theta0_support.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    def configs(width: int, height: int) -> Tuple[StateConfig, StateConfig]:
        gate = StateConfig(width=width, height=height, taus_us=(250.0,), radius=1)
        score = StateConfig(width=width, height=height, taus_us=(250.0,), radius=2)
        return gate, score

    # --max-blocks 0 reads naturally as "no cap" but `scan_support` treats any non-None value as
    # a hard limit, so 0 scored nothing and fit_threshold got an empty grid. Non-positive now
    # means uncapped, which is what full-coverage runs want.
    max_blocks = args.max_blocks if (args.max_blocks or 0) > 0 else None

    dnd_gate, dnd_score = configs(346, 260)

    # Same scene selection as Phase 4 and Phase 22, so all three describe one ED24 subset.
    scenes = sorted(p.name for p in ED24_ROOT.iterdir() if p.is_dir())
    if not args.all_scenes:
        scenes = scenes[::6][:16]
    levels = args.levels or ED24_LEVELS
    ed24: Dict[str, Dict] = {}
    for scene in scenes:
        for level in levels:
            res = scan_support(ed24_blocks(scene, level, BLOCK_SIZE, max_blocks),
                               dnd_gate, dnd_score, max_blocks)
            if res is not None:
                ed24[f"{scene}_{level}"] = res
        print(f"ED24 {scene}: done", flush=True)

    if not ed24:
        raise ValueError(
            "No ED24 condition produced a scored block. With --max-blocks 0 this used to mean "
            "'score nothing' and fit_threshold then failed on an empty grid; check the level "
            "names exist on disk and that the cap is not zero")
    fit = fit_threshold(ed24)
    at_zero = gated_auc(ed24, 0.0)
    payload: Dict = {
        "phase": "22b theta = 0 on ED24 under the adopted radius-2 support score",
        "score": "s = beta*G - log(1 + S_r2)",
        "gate_key": "corr(G, R_same) at radius 1 -- unchanged from Phase 4",
        "scenes": scenes,
        "levels": levels,
        "max_blocks_per_condition": max_blocks,
        "ed24": {
            "gate_theta0_on_ed24": at_zero,
            "best_fitted_gated_auc": fit["best_gated_auc"],
            "theta_plateau": [fit["theta_lower_edge"], fit["theta_upper_edge"]],
            "plateau_width": fit["plateau_width"],
            "cost_of_theta0_vs_fitted": at_zero - fit["best_gated_auc"],
            "theta0_inside_plateau": bool(
                fit["theta_lower_edge"] <= 0.0 <= fit["theta_upper_edge"]),
            "endpoints": endpoints(ed24),
        },
        "frac_blocks_corr_negative": float(
            np.mean(np.concatenate([c["corr"] for c in ed24.values()]) < 0.0)),
    }
    print("\nED24:", json.dumps(payload["ed24"], indent=2), flush=True)

    if not args.skip_transfer:
        dnd: Dict[str, Dict] = {}
        for rate in ("1hz", "3hz", "5hz", "7hz", "10hz"):
            for scene, fname in SCENE_FILES.items():
                path = PROJECT_ROOT / f"ECCV2024_datasets/AUC_test/{rate}/{fname}"
                dnd[f"{rate}_{scene}"] = scan_support(
                    dnd21_blocks(path, BLOCK_SIZE, None), dnd_gate, dnd_score, None)
            print(f"DND21 {rate}: done", flush=True)

        dvs_gate, dvs_score = configs(1280, 720)
        dvs: Dict[str, Dict] = {}
        for seq in SEQUENCES:
            for level in (50, 100):
                path = PROJECT_ROOT / f"DVSCLEAN/simulated_data/{seq}_{level}.hdf5"
                dvs[f"{seq}_{level}"] = scan_support(
                    dvsclean_blocks(path, BLOCK_SIZE, None), dvs_gate, dvs_score, None)
            print(f"DVSCLEAN {seq}: done", flush=True)

        transfer = {}
        for name, conds in (("DND21", dnd), ("DVSCLEAN", dvs)):
            transfer[name] = {
                **endpoints(conds),
                "gate_theta0_theory": gated_auc(conds, 0.0),
                "gate_theta_ED24_upper_edge": gated_auc(conds, fit["theta_upper_edge"]),
            }
        transfer["overall_theta0"] = 0.5 * (transfer["DND21"]["gate_theta0_theory"]
                                            + transfer["DVSCLEAN"]["gate_theta0_theory"])
        transfer["overall_theta_ED24_upper_edge"] = 0.5 * (
            transfer["DND21"]["gate_theta_ED24_upper_edge"]
            + transfer["DVSCLEAN"]["gate_theta_ED24_upper_edge"])
        transfer["cost_of_theta0_vs_fitted"] = (transfer["overall_theta0"]
                                                - transfer["overall_theta_ED24_upper_edge"])
        payload["blind_transfer"] = transfer
        payload["audit"] = {
            "note": "theta = 0 here is the adopted filter, so these must match Phase 19's "
                    "log_support|gated condition means.",
            **{f"{k}_deviation_vs_phase19": transfer[k]["gate_theta0_theory"] - v
               for k, v in PHASE19_GATED.items()},
        }
        print("\nTransfer:", json.dumps(transfer, indent=2), flush=True)
        print("Audit:", json.dumps(payload["audit"], indent=2), flush=True)

    payload["wall_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
