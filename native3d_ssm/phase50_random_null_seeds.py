"""Phase 50: the E-MLB random-deletion null over multiple seeds.

Section 3.4 leans on one number: randomly discarding the same 72% of events at r = 0.28
lowers mean E-MLB MESR by 0.053, which is what separates "the score discriminates" from
"MESR rewards discarding". Until now that came from a single seed, and a control reported
without its own uncertainty is a weak control -- a reviewer cannot tell whether 0.053 is a
stable property of undirected deletion or one draw that happened to land there.

This aggregates independent replications of `phase20_emlb_random_null`, each a full pass
over all eight cells with a different seed for the uniform per-event ranking. Seed
20260726 is the published one and must reproduce the stored artefact cell for cell; the
rest are new draws.

The expectation is a tight spread, and the reason is structural rather than lucky: each
cell averages 144 sequences, and each sequence's MESR averages ESR over consecutive
30,000-event slices, so a per-event uniform ranking has a great deal to average over
before it reaches the reported figure. Reporting the spread is still the point --
"stable for a reason we can name" is a stronger claim than "stable", and only the
measurement distinguishes them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
PUBLISHED_SEED = 20260726


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=ARTIFACTS / "phase20_emlb_random_null.json")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase50_random_null_seeds.json")
    args = parser.parse_args()

    per_seed: Dict[str, Dict] = {}
    cell_deltas: Dict[str, List[float]] = {}
    for path in args.inputs:
        payload = json.loads(path.read_text())
        seed = int(payload["settings"]["seed"])
        results = payload["results"]
        deltas = {f"{p}/{n}": results[p][n]["delta_over_raw"]
                  for p in results for n in results[p]}
        for cell, value in deltas.items():
            cell_deltas.setdefault(cell, []).append(value)
        per_seed[str(seed)] = {
            "mean_delta_over_raw": float(np.mean(list(deltas.values()))),
            "per_cell_delta": deltas,
        }

    means = np.array([v["mean_delta_over_raw"] for v in per_seed.values()])
    n = len(means)

    # The published seed must still reproduce the stored artefact exactly, otherwise this
    # is a different experiment wearing the same name.
    reference = json.loads(args.reference.read_text())["results"]
    ref_worst = max(
        abs(reference[p][nd]["random_null_fixed_mesr"]
            - json.loads(path.read_text())["results"][p][nd]["random_null_fixed_mesr"])
        for path in args.inputs
        if int(json.loads(path.read_text())["settings"]["seed"]) == PUBLISHED_SEED
        for p in reference for nd in reference[p]
    )

    payload = {
        "phase": "50 random-deletion null over independent seeds",
        "purpose": ("the manuscript's deletion control was reported from one seed; this "
                    "gives it an uncertainty so the control can be read as a property of "
                    "undirected deletion rather than of one draw"),
        "retention": 0.28,
        "metric": "mean over 8 E-MLB cells of (null MESR - raw MESR)",
        "seeds": sorted(int(s) for s in per_seed),
        "n_seeds": n,
        "published_seed": PUBLISHED_SEED,
        "published_seed_reproduces_artefact": ref_worst == 0.0,
        "published_seed_max_cell_difference": ref_worst,
        "mean_delta_over_raw": float(means.mean()),
        "sd_across_seeds": float(means.std(ddof=1)),
        "min_delta": float(means.min()),
        "max_delta": float(means.max()),
        "range_across_seeds": float(means.max() - means.min()),
        # A t interval on 5 draws is the honest summary; the range matters more here and
        # is reported beside it rather than hidden behind the interval.
        "ci95_mean": [float(means.mean() - 2.776 * means.std(ddof=1) / np.sqrt(n)),
                      float(means.mean() + 2.776 * means.std(ddof=1) / np.sqrt(n))],
        "per_cell": {cell: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                            "range": float(max(v) - min(v))}
                     for cell, v in cell_deltas.items()},
        "per_seed": per_seed,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("per_seed", "per_cell")}, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
