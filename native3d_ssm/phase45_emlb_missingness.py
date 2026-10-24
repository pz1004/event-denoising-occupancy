"""Phase 45: E-MLB MESR sensitivity to undefined baseline outputs.

MESR is undefined when filtering leaves fewer than one 30,000-event slice. The
Night ND16/ND64 comparison therefore uses paired sequences on which both methods
have finite values. This complete-case estimate need not represent the full cell:
missingness is related to how many events the baseline removes.

Using two stored artifacts, compute for each cell:

  * the complete-case paired margin;
  * a lower margin, imputing missing baseline values at that baseline's observed
    maximum within the cell;
  * an upper margin, imputing them at the observed minimum;
  * a mean-imputed margin.

These are sensitivity bounds within the observed baseline range, not assumptions
that missing values must lie inside that range. If both bounds have the same sign,
that sign is unchanged by these imputations; otherwise the margin remains
unresolved under this sensitivity analysis. No experiment is rerun or parameter
changed.
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
