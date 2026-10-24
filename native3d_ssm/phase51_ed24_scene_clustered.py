"""Phase 51: ED24 inference at the right level of clustering.

THE DEFECT. Section 4.4 reports the ED24 operand effect as a mean over the 2,100
scene-by-level conditions with a bootstrap interval obtained by resampling those 2,100
conditions independently. They are not independent. ED24 anneals 21 attenuator-voltage
levels over a *fixed* set of 100 scenes, so each scene contributes 21 correlated
observations and voltage is crossed with scene rather than nested in it. Resampling
conditions independently treats one scene's 21 levels as 21 draws and understates the
uncertainty.

This is the same error the manuscript already refuses to make twice: Section 4.1 declines
to treat E-MLB's eight crossed acquisition cells as replicates, and Phase 49 declines to
treat its 384 scene-by-ND units as 384 clusters when they hold only 96 distinct scenes.
ED24 was left at the condition level. This phase corrects it.

THE CORRECTION. Resample the 100 scenes with replacement; a drawn scene carries all 21 of
its voltage levels together, or none of them. The pooled statistic is recomputed inside
each replicate, so a scene that appears twice weighs twice at every level. The sign-flip
test permutes at the same level: a whole scene's 21 deltas flip sign together.

Reported alongside:
  * the condition-level interval, unchanged, so the narrowing factor is visible;
  * a scene-clustered interval at every one of the 21 levels, which is what decides
    whether the negative value at the sparsest level is separated from zero or is noise;
  * the level at which the scene-clustered interval first excludes zero on the positive
    side, which is the honest way to state "the gain rises with density".

INPUT. Read-only, from stored artefacts. `phase36_ed24_inrange_ours.json` holds, for each
of the 2,100 conditions, the per-block AUC series for every arm, so nothing is rescored
and the point estimates must reproduce the published ones exactly.

ARMS. `adopted` is beta*G - log(1 + S_r2); `magnitude` is beta*G - R_same_r2 on the same
radius-2 ring with the same gate. The contrast is the operand and nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
SOURCE = ARTIFACTS / "phase36_ed24_inrange_ours.json"
OUT = ARTIFACTS / "phase51_ed24_scene_clustered.json"

#: Published condition-level figures this phase must reproduce before it may correct them.
ANCHORS = {
    "operand_gain_block_mean_mean": 0.01732762320124535,
    "operand_gain_block_mean_ci_lo": 0.016397195136327872,
    "operand_gain_block_mean_ci_hi": 0.01826799648851823,
    "conditions": 2100,
    "scenes": 100,
    "levels": 21,
}
TOLERANCE = 5e-7


def split_condition(key: str, scenes: List[str]) -> Tuple[str, str]:
    """`Bicycle_01_1.5` -> (`Bicycle_01`, `1.5`). Scene names contain underscores."""

    for scene in scenes:
        if key.startswith(scene + "_"):
            return scene, key[len(scene) + 1:]
    raise KeyError(f"condition {key!r} matches no scene")


def cluster_bootstrap(clusters: List[Tuple[str, List[float]]], replicates: int,
                      rng: np.random.Generator) -> Dict:
    """Resample whole scenes; a scene's voltage levels are drawn together or not at all.

    The pooled mean is recomputed inside the replicate as sum/count over the drawn
    scenes, so scenes contributing unequal numbers of conditions weigh correctly.
    """

    blocks = [np.asarray(v, dtype=np.float64) for _, v in clusters]
    n = len(blocks)
    flat = np.concatenate(blocks) if blocks else np.array([])
    if n < 2:
        return {"n_clusters": n, "mean": float(flat.mean()) if flat.size else float("nan"),
                "ci95": [float("nan"), float("nan")], "replicates": 0}
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([b.size for b in blocks])
    draws = rng.integers(0, n, size=(replicates, n))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    cluster_means = np.array([b.mean() for b in blocks])
    return {
        "n_clusters": n,
        "mean": float(flat.mean()),
        "sd_across_clusters": float(cluster_means.std(ddof=1)),
        "bootstrap_se": float(means.std(ddof=1)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "replicates": replicates,
        "clusters_ahead": int((cluster_means > 0).sum()),
        "clusters_behind": int((cluster_means < 0).sum()),
    }


def naive_bootstrap(values: np.ndarray, replicates: int, rng: np.random.Generator) -> Dict:
    """The condition-level interval the manuscript currently prints. Kept for contrast."""

    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "bootstrap_se": float(means.std(ddof=1)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "replicates": replicates,
    }


def sign_flip_test(clusters: List[Tuple[str, List[float]]], draws: int,
                   rng: np.random.Generator) -> Dict:
    """Cluster-level permutation: a whole scene's 21 deltas flip sign together."""

    cluster_means = np.array([np.mean(v) for _, v in clusters], dtype=np.float64)
    observed = float(cluster_means.mean())
    signs = rng.choice((-1.0, 1.0), size=(draws, len(cluster_means)))
    null_means = (signs * cluster_means).mean(axis=1)
    p = float((np.abs(null_means) >= abs(observed) - 1e-15).mean())
    return {"n_clusters": int(len(cluster_means)), "observed_mean": observed,
            "draws": draws, "p_two_sided": max(p, 1.0 / draws)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200_000)
    parser.add_argument("--flip-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--left", default="adopted",
                        help="arm on the plus side of the contrast")
    parser.add_argument("--right", default="magnitude",
                        help="arm on the minus side of the contrast")
    parser.add_argument("--output", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    source = json.loads(SOURCE.read_text())
    scenes: List[str] = source["scenes"]
    levels: List[str] = source["levels"]
    conditions: Dict[str, Dict] = source["conditions"]

    # delta[scene][level] = block-mean AUC(support) - block-mean AUC(magnitude)
    delta: Dict[str, Dict[str, float]] = {s: {} for s in scenes}
    for key, record in conditions.items():
        scene, level = split_condition(key, scenes)
        bm = record["block_mean"]
        delta[scene][level] = float(bm[args.left]) - float(bm[args.right])

    flat = np.array([delta[s][l] for s in scenes for l in levels], dtype=np.float64)
    if flat.size != ANCHORS["conditions"]:
        raise SystemExit(f"expected {ANCHORS['conditions']} conditions, got {flat.size}")

    default_pair = (args.left, args.right) == ("adopted", "magnitude")
    anchor_drift = (abs(float(flat.mean()) - ANCHORS["operand_gain_block_mean_mean"])
                    if default_pair else float("nan"))
    if default_pair and anchor_drift > TOLERANCE:
        raise SystemExit(
            f"pooled operand gain drifted by {anchor_drift:.3e} from the published "
            f"{ANCHORS['operand_gain_block_mean_mean']:.6f}; refusing to publish a "
            "corrected interval around a point estimate that no longer reproduces")

    clusters = [(s, [delta[s][l] for l in levels]) for s in scenes]
    pooled_clustered = cluster_bootstrap(clusters, args.replicates, rng)
    pooled_naive = naive_bootstrap(flat, args.replicates, rng)
    flip = sign_flip_test(clusters, args.flip_draws, rng)

    naive_width = pooled_naive["ci95"][1] - pooled_naive["ci95"][0]
    clustered_width = pooled_clustered["ci95"][1] - pooled_clustered["ci95"][0]

    per_level: Dict[str, Dict] = {}
    for level in levels:
        per_scene = [(s, [delta[s][level]]) for s in scenes]
        per_level[level] = cluster_bootstrap(per_scene, args.replicates, rng)

    # The first level, scanning from the sparse end, whose scene-clustered interval lies
    # entirely above zero -- and, separately, whether any level is separated below zero.
    first_positive = next((l for l in levels if per_level[l]["ci95"][0] > 0.0), None)
    separated_negative = [l for l in levels if per_level[l]["ci95"][1] < 0.0]
    unresolved = [l for l in levels
                  if per_level[l]["ci95"][0] <= 0.0 <= per_level[l]["ci95"][1]]

    result = {
        "phase": "51 ED24 operand effect, resampled over scenes rather than conditions",
        "source": str(SOURCE.relative_to(PROJECT_ROOT)),
        "rescored": False,
        "arms": {"left": args.left, "right": args.right},
        "seed": args.seed,
        "unit": {
            "scenes": len(scenes),
            "levels": len(levels),
            "conditions": int(flat.size),
            "why": "voltage is crossed with scene, not nested in it, so the scene is the "
                   "cluster and its 21 levels are carried together",
        },
        "anchor_drift_vs_published_point_estimate": anchor_drift,
        "pooled_scene_clustered": pooled_clustered,
        "pooled_condition_level_for_contrast": pooled_naive,
        "interval_width_ratio_naive_over_clustered": naive_width / clustered_width,
        "sign_flip_scene_clustered": flip,
        "per_level_scene_clustered": per_level,
        "first_level_separated_above_zero": first_positive,
        "levels_separated_below_zero": separated_negative,
        "levels_not_separated_from_zero": unresolved,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"pooled operand gain      {pooled_clustered['mean']:+.6f}")
    print(f"  condition-level CI     [{pooled_naive['ci95'][0]:+.6f}, "
          f"{pooled_naive['ci95'][1]:+.6f}]  (width {naive_width:.6f})")
    print(f"  scene-clustered CI     [{pooled_clustered['ci95'][0]:+.6f}, "
          f"{pooled_clustered['ci95'][1]:+.6f}]  (width {clustered_width:.6f})")
    print(f"  narrowing factor       {naive_width / clustered_width:.2f}x")
    print(f"  scenes ahead           {pooled_clustered['clusters_ahead']}/{len(scenes)}")
    print(f"  sign-flip p            {flip['p_two_sided']:.2e}")
    print(f"  first level separated above zero: {first_positive}")
    print(f"  levels separated below zero:      {separated_negative}")
    print(f"  levels not separated from zero:   {unresolved}")
    for level in levels:
        r = per_level[level]
        print(f"    {level:>4} V  {r['mean']:+.6f}  "
              f"[{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]  "
              f"{r['clusters_ahead']:3d}/{len(scenes)}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
