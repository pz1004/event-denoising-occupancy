"""Phase 22: what does theta = 0 actually score on ED24 itself?

Phase 4 fitted theta on ED24 and applied theta = 0 blind to DND21 and DVSCLEAN, but it
never evaluated theta = 0 *on ED24*. The draft consequently claimed that theta = 0
"reproduces the best fitted rule exactly (both 0.9565)", which conflated two different
quantities: 0.9565 is the DND21+DVSCLEAN transfer score, not an ED24 score, and the
ED24 fit peaks on a plateau of negative theta that excludes zero.

This phase measures the missing number directly, reusing Phase 4's scan and fit verbatim
so the comparison is exactly paired:

    gated AUC on ED24 at theta = 0          <- the theory value
    gated AUC on ED24 at the fitted theta*  <- the plateau maximum
    the plateau [theta_lower, theta_upper]  <- where the fit actually peaks

Read-only; nothing is trained. Deliberately writes a NEW artifact and does not touch
`phase4_ed24_blind*.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import StateConfig
from .phase4_ed24_blind import (
    ED24_LEVELS,
    ED24_ROOT,
    ed24_blocks,
    fit_threshold,
    gated_auc,
    scan_min,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-blocks", type=int, default=8,
                        help="per scene x level; 8 matches Phase 4")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase22_ed24_theta0.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cfg = StateConfig(width=346, height=260, taus_us=(250.0,))

    # Same scene selection as Phase 4, so the two artifacts describe the same ED24 subset.
    scenes = sorted(p.name for p in ED24_ROOT.iterdir() if p.is_dir())[::6][:16]
    ed24 = {}
    for scene in scenes:
        for level in ED24_LEVELS:
            res = scan_min(ed24_blocks(scene, level, 10240, args.max_blocks), cfg, args.max_blocks)
            if res is not None:
                ed24[f"{scene}_{level}"] = res
        print(f"ED24 {scene}: done", flush=True)

    payload = {"phase": "22 theta = 0 evaluated on ED24 itself",
               "scenes": scenes, "levels": ED24_LEVELS,
               "max_blocks_per_condition": args.max_blocks}
    for suffix, label in (("", "old_score"), ("_new", "new_score")):
        fit = fit_threshold(ed24, suffix)
        at_zero = gated_auc(ed24, 0.0, suffix)
        payload[label] = {
            "gate_theta0_on_ed24": at_zero,
            "best_fitted_gated_auc": fit["best_gated_auc"],
            "theta_plateau": [fit["theta_lower_edge"], fit["theta_upper_edge"]],
            "plateau_width": fit["plateau_width"],
            "cost_of_theta0_vs_fitted": at_zero - fit["best_gated_auc"],
            "theta0_inside_plateau": bool(
                fit["theta_lower_edge"] <= 0.0 <= fit["theta_upper_edge"]),
        }
    payload["frac_blocks_corr_negative"] = float(
        np.mean(np.concatenate([c["corr"] for c in ed24.values()]) < 0.0))
    payload["reading"] = (
        "ED24 is effectively single-regime (corr > 0 almost everywhere), so the fit cannot "
        "locate theta from below and its plateau runs to the grid edge. theta = 0 sits just "
        "outside that plateau: the cost is what the gate pays for occasionally activating G "
        "on a dataset where it never helps.")
    payload["wall_seconds"] = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")
    print(json.dumps({k: payload[k] for k in ("old_score", "new_score",
                                              "frac_blocks_corr_negative")}, indent=2))


if __name__ == "__main__":
    main()
