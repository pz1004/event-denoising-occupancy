"""Phase 29: does the effective-sample-size-2 ceiling actually bind the operand result?

Section 6.1 of the manuscript concedes that the twenty labelled conditions are not twenty draws:
they are ten conditions from each of two datasets that differ in noise model, sensor geometry,
density regime and preprocessing provenance, so the effective sample size is 2. Re-clustering the
paired bootstrap by dataset instead of by condition was measured once before, on the EDZero versus
reproduced-EDmamba margin, and it widened that interval substantially.

The manuscript then generalised: *every* labelled interval inherits the ceiling, the operand's
included. That generalisation is wrong, and this phase is the measurement that shows why.

Clustering can only widen an interval through variation BETWEEN clusters. The EDmamba margin has
plenty: it is positive on DND21 and negative on DVSCLEAN, so resampling two datasets moves the
mean a long way. The operand gain has almost none -- it is the same size on both datasets -- so a
dataset-clustered resample of it collapses onto its own point estimate instead of widening.

Two intervals are therefore computed for each paired quantity available in phase 23:

    condition-level   ordinary bootstrap over the 20 conditions (what the paper reports)
    dataset-clustered resample the 2 datasets with replacement, take every condition in each

The second is reported with an explicit warning attached, because with two clusters its support
has three atoms -- both draws DND21, one of each, both draws DVSCLEAN -- so the percentile interval
spans the two dataset means and carries no coverage guarantee. It is a diagnostic for how much of
the variance sits between datasets, not a confidence interval, and the manuscript quotes it as the
former.

Pure post-processing: reads phase23_ablation_statistic.json and writes a JSON. Nothing is scored,
trained or fitted, so this runs anywhere in a second without the benchmarks present.

Run:  python -m native3d_ssm.phase29_operand_cluster
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
SOURCE = ARTIFACTS / "phase23_ablation_statistic.json"
BOOTSTRAP_REPLICATES = 10_000
SEED = 20260730

#: label -> (variant scored, variant subtracted). Both are keys of phase 23's per-condition
#: block-mean AUC, so every delta below is paired condition by condition on identical blocks.
QUANTITIES = {
    "operand_vs_magnitude": ("gated", "magnitude_gated"),
    "gate_vs_beta0": ("gated", "beta0"),
    "gate_vs_beta1": ("gated", "beta1"),
}
STATS = ("mean_block_auc", "global_event_auc", "rank_normalised_event_auc")


def bootstrap(values: Sequence[float], rng: np.random.Generator) -> Dict:
    """Percentile interval for the mean, resampling the values as independent units."""
    arr = np.asarray(values, dtype=np.float64)
    means = np.array([arr[rng.integers(0, arr.size, arr.size)].mean()
                      for _ in range(BOOTSTRAP_REPLICATES)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(arr.mean()), "n": int(arr.size),
            "ci95": [float(lo), float(hi)], "width": float(hi - lo),
            "bootstrap_se": float(means.std(ddof=1))}


def cluster_bootstrap(groups: Sequence[Sequence[float]], rng: np.random.Generator) -> Dict:
    """Percentile interval resampling whole clusters, taking every unit in a drawn cluster."""
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    n = len(arrays)
    means = np.array([np.concatenate([arrays[i] for i in rng.integers(0, n, n)]).mean()
                      for _ in range(BOOTSTRAP_REPLICATES)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    pooled = np.concatenate(arrays)
    return {"mean": float(pooled.mean()), "clusters": n,
            "cluster_means": [float(a.mean()) for a in arrays],
            "ci95": [float(lo), float(hi)], "width": float(hi - lo),
            "bootstrap_se": float(means.std(ddof=1)),
            "warning": ("with 2 clusters the bootstrap support has 3 atoms and the interval spans "
                        "the two cluster means; read it as a between-dataset variance diagnostic, "
                        "not as an interval with 95% coverage")}


def deltas(conditions: Dict, dataset: str, stat: str, scored: str, subtracted: str) -> List[float]:
    return [v[stat][scored] - v[stat][subtracted] for v in conditions[dataset].values()]


def summarise(conditions: Dict, stat: str, scored: str, subtracted: str,
              rng: np.random.Generator) -> Dict:
    per_dataset = {ds: deltas(conditions, ds, stat, scored, subtracted) for ds in conditions}
    pooled = [d for values in per_dataset.values() for d in values]
    condition_level = bootstrap(pooled, rng)
    clustered = cluster_bootstrap(list(per_dataset.values()), rng)
    return {
        "per_dataset_mean": {ds: float(np.mean(v)) for ds, v in per_dataset.items()},
        "per_dataset_positive": {ds: int(np.sum(np.asarray(v) > 0)) for ds, v in per_dataset.items()},
        "conditions_positive": int(np.sum(np.asarray(pooled) > 0)),
        "conditions_total": len(pooled),
        "smallest_gain": float(np.min(pooled)),
        "condition_level": condition_level,
        "dataset_clustered": clustered,
        "width_ratio_clustered_over_condition": (
            float(clustered["width"] / condition_level["width"])
            if condition_level["width"] > 0 else None),
        "se_ratio_clustered_over_condition": (
            float(clustered["bootstrap_se"] / condition_level["bootstrap_se"])
            if condition_level["bootstrap_se"] > 0 else None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase29_operand_cluster.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conditions = json.loads(args.source.read_text())["conditions"]
    rng = np.random.default_rng(SEED)

    payload = {
        "phase": "29 condition-level versus dataset-clustered intervals",
        "question": ("Section 6.1 claims every labelled interval inherits the effective-sample-"
                     "size-2 ceiling. Clustering widens an interval only through between-cluster "
                     "variation. Does the operand gain have any?"),
        "source": str(args.source.relative_to(PROJECT_ROOT)),
        "config": {"replicates": BOOTSTRAP_REPLICATES, "seed": SEED,
                   "clusters": "the 2 labelled datasets", "units": "the 20 labelled conditions"},
        "audit": ("mean_block_auc/operand_vs_magnitude must reproduce the manuscript's +0.0095 "
                  "and its condition-level interval [+0.0071, +0.0121]."),
        "quantities": {
            label: {stat: summarise(conditions, stat, scored, subtracted, rng) for stat in STATS}
            for label, (scored, subtracted) in QUANTITIES.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}\n")

    for label in QUANTITIES:
        block = payload["quantities"][label]["mean_block_auc"]
        cond, clus = block["condition_level"], block["dataset_clustered"]
        print(f"{label} (block-mean AUC)")
        print(f"  per dataset      : " + ", ".join(
            f"{ds} {m:+.4f}" for ds, m in block["per_dataset_mean"].items()))
        print(f"  condition-level  : {cond['mean']:+.4f} "
              f"[{cond['ci95'][0]:+.4f}, {cond['ci95'][1]:+.4f}]  width {cond['width']:.4f}")
        print(f"  dataset-clustered: {clus['mean']:+.4f} "
              f"[{clus['ci95'][0]:+.4f}, {clus['ci95'][1]:+.4f}]  width {clus['width']:.4f}"
              f"  ({block['width_ratio_clustered_over_condition']:.2f}x)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
