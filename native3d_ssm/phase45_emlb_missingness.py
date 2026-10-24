"""Phase 45: bounding the Night ND16/ND64 deficit under MESR missingness.

Reviewer question this answers
------------------------------
MESR is undefined once a filtered stream holds fewer than one 30,000-event slice, and the
reproduced transformer filters hardest exactly where the manuscript reports an unresolved
deficit. A reviewer asked what that deficit is once the missingness is accounted for,
rather than left as a caveat.

The manuscript currently compares the two methods on the sequences where both are defined.
That is a *complete-case* analysis, and complete-case analysis is unbiased only if the
missingness is unrelated to the value --- which here it plainly is not, since a sequence
goes missing precisely because the baseline removed most of it.

This phase replaces the caveat with an interval. For each cell it computes:

  * the complete-case paired margin, which is what the manuscript reports;
  * a **worst case** for us, imputing every missing baseline sequence at the largest value
    that baseline achieved anywhere in the cell;
  * a **best case** for us, imputing at the smallest;
  * a mean-imputed margin, which is the assumption complete-case analysis makes implicitly.

If the sign of the margin is the same at both extremes, the missingness cannot explain it.
If it is not, the honest statement is that the deficit is unresolved --- and this phase
says by how much, which the manuscript could not before.

Read-only over two stored artefacts. Nothing is re-run and no constant moves.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PART_NAMES = {"D-END": "Daylight", "N-END": "Night"}
ND_DISPLAY = {"ND00": "ND1", "ND04": "ND4", "ND16": "ND16", "ND64": "ND64"}
#: Our per-sequence field: the adopted support score at the frozen r = 0.28.
OURS_FIELD = "support_score_fixed_mesr"


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def paired_bounds(ours: Dict[str, float], theirs: Dict[str, Optional[float]]) -> Dict:
    """Complete-case margin and the interval the missingness leaves open.

    Both rows are improvements over the same Raw sequence, so the Raw term cancels and a
    sequence contributes ``ours - theirs`` exactly.
    """

    shared = [k for k in ours if k in theirs and not is_missing(ours[k])]
    observed = [k for k in shared if not is_missing(theirs[k])]
    missing = [k for k in shared if is_missing(theirs[k])]
    if not observed:
        return {"sequences": 0}

    their_values = np.asarray([theirs[k] for k in observed], dtype=np.float64)
    deltas_observed = np.asarray([ours[k] - theirs[k] for k in observed], dtype=np.float64)

    result = {
        "sequences": len(shared),
        "observed": len(observed),
        "missing": len(missing),
        "complete_case_margin": float(deltas_observed.mean()),
        "baseline_observed_min": float(their_values.min()),
        "baseline_observed_max": float(their_values.max()),
        "baseline_observed_mean": float(their_values.mean()),
    }
    if not missing:
        result.update({
            "worst_case_margin": result["complete_case_margin"],
            "best_case_margin": result["complete_case_margin"],
            "mean_imputed_margin": result["complete_case_margin"],
            "sign_is_robust_to_missingness": True,
        })
        return result

    our_missing = np.asarray([ours[k] for k in missing], dtype=np.float64)
    # Worst case for us: the baseline would have scored its cell maximum on every
    # sequence it could not be scored on. Best case: its cell minimum.
    worst = np.concatenate([deltas_observed, our_missing - their_values.max()])
    best = np.concatenate([deltas_observed, our_missing - their_values.min()])
    mean_imputed = np.concatenate([deltas_observed, our_missing - their_values.mean()])

    result.update({
        "worst_case_margin": float(worst.mean()),
        "best_case_margin": float(best.mean()),
        "mean_imputed_margin": float(mean_imputed.mean()),
        "sign_is_robust_to_missingness": bool(
            np.sign(worst.mean()) == np.sign(best.mean())
        ),
        "our_mean_on_missing_sequences": float(our_missing.mean()),
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path,
                        default=PROJECT_ROOT
                        / "native3d_ssm/experiments/analytic_contrast/phase24_emlb_per_sequence.json")
    parser.add_argument("--baseline", type=Path,
                        default=PROJECT_ROOT
                        / "reproduction/results/emlb_edformer_mesr_per_sequence.json")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT
                        / "native3d_ssm/experiments/analytic_contrast/phase45_emlb_missingness.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    ours_all = json.loads(args.ours.read_text())["results"]
    theirs_all = json.loads(args.baseline.read_text())["results"]

    cells: Dict[str, Dict] = {}
    for part, part_name in PART_NAMES.items():
        for nd, nd_name in ND_DISPLAY.items():
            if part not in ours_all or nd not in ours_all[part]:
                continue
            if part not in theirs_all or nd not in theirs_all[part]:
                continue
            ours = {k: v[OURS_FIELD]
                    for k, v in ours_all[part][nd]["per_sequence"].items()}
            theirs = {k: v["edformer_mesr"]
                      for k, v in theirs_all[part][nd]["per_sequence"].items()}
            report = paired_bounds(ours, theirs)
            if report.get("sequences"):
                cells[f"{part_name} {nd_name}"] = report
                print(json.dumps({"cell": f"{part_name} {nd_name}", **report}), flush=True)

    resolved = [k for k, v in cells.items() if v.get("sign_is_robust_to_missingness")]
    unresolved = [k for k, v in cells.items() if not v.get("sign_is_robust_to_missingness")]

    payload = {
        "phase": "45 bounding the E-MLB comparison under MESR missingness",
        "question": ("The baseline's MESR is undefined on the sequences it filters hardest. "
                     "What does that leave open about the per-cell margin?"),
        "method": ("Complete-case margin, plus the interval spanned by imputing every "
                   "missing baseline sequence at the cell's observed baseline maximum "
                   "(worst case for us) and minimum (best case for us), on the rep-1 "
                   "sequences both artefacts share."),
        "ours_field": OURS_FIELD,
        "cells": cells,
        "cells_with_sign_robust_to_missingness": sorted(resolved),
        "cells_left_open_by_missingness": sorted(unresolved),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"robust": sorted(resolved), "open": sorted(unresolved)}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
