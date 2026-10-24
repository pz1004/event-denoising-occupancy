"""Phase 55: E-MLB with the gate closed, because LED said to close it.

WHAT FORCED THIS. Phase 54 is the first labelled benchmark that actually exercises the
correlation-sign gate's open branch: it opens on 23.0% of LED blocks, against 1.9% on
E-MLB and 2.5% on ED24. Exercised, it loses. Against the ungated score the gate is worth
-0.0122 block-mean AUC, scene-clustered interval [-0.0156, -0.0090], and it is behind on
**49 of the 66 LED scenes and ahead on none**. (An earlier version of this note said "all
66". That count belongs to a different contrast -- the gate forced *open*, which is worth
-0.0550 and is behind on all 66. The adopted gated score is behind on 49 and tied on the
rest, because on those scenes the gate never opens.) The only benchmark on which the gate
has ever helped is the pair of development sets that selected it.

That leaves the held-out E-MLB result attached to a component the evidence rejects. The
gate is nearly inert there -- 1.9% of blocks -- so the ungated numbers are expected to sit
very close to the gated ones, but "expected to" is not a measurement, and a headline row
should not rest on one. This phase re-runs E-MLB with the gate closed so the adopted
configuration can be reported from a measured row rather than an argued one.

ARMS, all at the same frozen retention r = 0.28 on the same event stream:

    support_gated     beta*G - log(1 + S_r2)   the previously adopted score  (anchor)
    magnitude_gated   beta*G - R_same_r2       its radius-matched control    (anchor)
    support_beta0     -log(1 + S_r2)           the ungated score  <- the new adopted arm
    magnitude_beta0   -R_same_r2               its radius-matched control

The two gated arms exist to anchor: they must reproduce Phase 49's cell means, or the
ungated rows are not comparable to anything already published. Ranking by `support_beta0`
is ranking by the 5-bit integer S itself, since -log(1 + S) is a strictly decreasing
function of it, so this arm is the one the occupancy-bit-plane implementation computes
exactly and with no floating point anywhere in the score.

Unit of analysis, dropout accounting and scene clustering are Phase 49's, unchanged: the
cluster is the scene within illumination part, 96 of them, and a scene's four ND levels
move together. Read-only; nothing is trained, and no constant is refitted.
"""


from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import ring_support
from .phase5_emlb_mesr import (
    ND_LEVELS,
    PARTS,
    SLICE,
    mesr,
    read_aedat4,
    retain_mask,
)
from .phase8_score_upgrade import operands, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK = 10_240
RETENTION = 0.28
ARMS = ("support_gated", "magnitude_gated", "support_beta0", "magnitude_beta0")
CONTRASTS = (
    ("support_beta0", "magnitude_beta0"),  # the operand, ungated -- the new headline
    ("support_gated", "magnitude_gated"),  # the operand, gated -- Phase 49's claim
    ("support_beta0", "support_gated"),    # what closing the gate costs or buys here
)
#: Phase 49 cell means the gated arms must reproduce, or nothing here is comparable.
ANCHORS = {"raw": 0.8147, "support_gated": 1.0035, "magnitude_gated": 0.9426}


def score_streams(events: np.ndarray, gate_config: StateConfig,
                  score_config: StateConfig) -> Dict[str, np.ndarray]:
    """Per-event scores for the three arms, blockwise, sharing one gate decision."""

    ordered = np.ascontiguousarray(
        np.column_stack((events["timestamp"].astype(np.int64),
                         events["x"].astype(np.int64),
                         events["y"].astype(np.int64),
                         (events["polarity"].astype(np.int64) > 0).astype(np.int64))),
        dtype=np.int64)
    out = {arm: np.zeros(len(ordered), dtype=np.float32) for arm in ARMS}

    for start in range(0, len(ordered), BLOCK):
        chunk = ordered[start:start + BLOCK]
        if len(chunk) < 2:
            continue
        stop = start + len(chunk)

        raw_g = bidirectional_features(chunk, gate_config)
        g, r_same_r1, _ = operands(raw_g, gate_config)
        # Gate key frozen at radius 1 for every arm, so no arm gets a gate tuned to it.
        beta = 1.0 if safe_corr(g, r_same_r1) < 0.0 else 0.0

        raw_s = bidirectional_features(chunk, score_config)
        _, r_same_r2, _ = operands(raw_s, score_config)
        support = ring_support(raw_s, len(score_config.taus_us),
                               score_config.feature_dim, score_config.radius)

        local = np.log1p(support)
        out["support_gated"][start:stop] = beta * g - local
        out["magnitude_gated"][start:stop] = beta * g - r_same_r2
        out["support_beta0"][start:stop] = -local
        out["magnitude_beta0"][start:stop] = -r_same_r2
    return out


def cluster_bootstrap_blocks(clusters: List[Tuple[str, List[float]]], replicates: int,
                             rng: np.random.Generator) -> Dict:
    """Resample whole scenes. A scene's ND levels are drawn together or not at all."""

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


def sign_flip_test_blocks(blocks: List[np.ndarray], draws: int,
                          rng: np.random.Generator) -> Dict:
    """Cluster-level permutation: a whole scene's sign flips together."""

    n = len(blocks)
    cluster_means = np.array([b.mean() for b in blocks])
    observed = float(cluster_means.mean())
    if n == 0:
        return {"n_clusters": 0, "p_two_sided": float("nan")}
    signs = rng.choice((-1.0, 1.0), size=(draws, n))
    null_means = (signs * cluster_means).mean(axis=1)
    p = float((np.abs(null_means) >= abs(observed) - 1e-15).mean())
    return {"n_clusters": n, "observed_mean": observed, "draws": draws,
            "p_two_sided": max(p, 1.0 / draws)}


def cluster_bootstrap(by_scene: List[Tuple[str, float]], replicates: int,
                      rng: np.random.Generator) -> Dict:
    """Resample SCENES with replacement. Reps within a scene move together."""

    values = np.asarray([v for _, v in by_scene], dtype=np.float64)
    n = len(values)
    if n < 2:
        return {"n": n, "mean": float(values.mean()) if n else float("nan"),
                "ci95": [float("nan"), float("nan")], "replicates": 0}
    draws = rng.integers(0, n, size=(replicates, n))
    means = values[draws].mean(axis=1)
    return {
        "n": n,
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "bootstrap_se": float(means.std(ddof=1)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "replicates": replicates,
        "scenes_ahead": int((values > 0).sum()),
        "scenes_behind": int((values < 0).sum()),
    }


def sign_flip_test(values: np.ndarray, draws: int, rng: np.random.Generator) -> Dict:
    """Paired permutation test: under the null, each scene's delta sign is exchangeable."""

    n = len(values)
    observed = float(values.mean())
    if n == 0:
        return {"n": 0, "p_two_sided": float("nan")}
    signs = rng.choice((-1.0, 1.0), size=(draws, n))
    null_means = (signs * values).mean(axis=1)
    p = float((np.abs(null_means) >= abs(observed) - 1e-15).mean())
    return {"n": n, "observed_mean": observed, "draws": draws,
            "p_two_sided": max(p, 1.0 / draws)}


def recompute_dropouts(previous: Dict) -> Dict:
    """Re-derive the dropout classification from the stored per-sequence evidence."""

    old = previous["dropouts"]
    entries = []
    common, differential = 0, 0
    for entry in old["arm_specific_keys"]:
        failed = entry["nonfinite_arms"]
        is_common = len(failed) == len(ARMS)
        common += int(is_common)
        differential += int(not is_common)
        entries.append({**entry, "common_mode": is_common})
    return {
        "attempted": old["attempted"],
        "complete": old["complete"],
        "dropped_raw_nonfinite": old["dropped_raw_nonfinite"],
        "dropped_arm_nonfinite": old["dropped_arm_nonfinite"],
        "dropped_all_arms_common_mode": common,
        "dropped_differential": differential,
        "per_arm_nonfinite": old["per_arm_nonfinite"],
        "arm_specific_keys": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--nd-levels", nargs="+", default=list(ND_LEVELS))
    parser.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--recompute-from", type=Path, default=None,
                        help=("rebuild the summary statistics from a previous artefact's "
                              "stored per-sequence values instead of re-scoring E-MLB"))
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase55_emlb_ungated.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    gate_config = StateConfig(width=346, height=260, taus_us=(250.0,), radius=1)
    score_config = StateConfig(width=346, height=260, taus_us=(250.0,), radius=2)

    cap: Optional[int] = args.max_events if args.max_events > 0 else None
    results: Dict[str, Dict[str, Dict]] = {}
    reused: Optional[Dict] = None
    if args.recompute_from is not None:
        reused = json.loads(args.recompute_from.read_text())
        results = reused["results"]
    #: Dropout accounting. The claim that this pairing is complete -- and therefore free
    #: of the complete-case bias that forced the n = 8 headline elsewhere -- is only
    #: worth making if the dropouts are recorded and shown not to be arm-specific.
    #: A sequence whose RAW stream has no finite MESR is unusable for every arm and
    #: cannot bias a paired contrast. A sequence lost because ONE arm went non-finite
    #: can, so the two cases are counted apart rather than together.
    dropouts: Dict[str, object] = {
        "attempted": 0,
        "complete": 0,
        "dropped_raw_nonfinite": 0,
        "dropped_arm_nonfinite": 0,
        "dropped_all_arms_common_mode": 0,
        "dropped_differential": 0,
        "per_arm_nonfinite": {arm: 0 for arm in ARMS},
        "arm_specific_keys": [],
    }

    for part in (() if reused else args.parts):
        root = args.emlb_root / part
        scenes = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
        results[part] = {}
        for nd in args.nd_levels:
            per_sequence: Dict[str, Dict[str, float]] = {}
            raw_list: List[float] = []
            arm_lists: Dict[str, List[float]] = {arm: [] for arm in ARMS}

            for scene in scenes:
                for rep in args.reps:
                    path = root / scene / f"{scene}-{nd}-{rep}.aedat4"
                    if not path.exists():
                        continue
                    events = read_aedat4(path, cap)
                    if events is None or len(events) < SLICE:
                        continue
                    x, y = events["x"], events["y"]
                    raw_value = mesr(x, y, gate_config.width, gate_config.height)
                    streams = score_streams(events, gate_config, score_config)

                    row = {"raw_mesr": float(raw_value), "scene": scene, "rep": rep}
                    dropouts["attempted"] += 1
                    raw_ok = bool(np.isfinite(raw_value))
                    arms_ok = True
                    for arm in ARMS:
                        keep = retain_mask(streams[arm], RETENTION, BLOCK)
                        value = mesr(x[keep], y[keep], gate_config.width, gate_config.height)
                        row[arm] = float(value)
                        if not np.isfinite(value):
                            dropouts["per_arm_nonfinite"][arm] += 1
                            arms_ok = False
                    if not raw_ok:
                        dropouts["dropped_raw_nonfinite"] += 1
                        continue
                    if not arms_ok:
                        # Raw was fine but at least one arm was not. Two very different
                        # cases hide here and only one of them can bias a paired
                        # contrast. If EVERY arm went non-finite the sequence is simply
                        # unusable -- retention left too few events for a 30,000-event
                        # slice -- and dropping it removes the same sequence from both
                        # sides, which a paired estimator is indifferent to. If only
                        # SOME arms failed, one side is missing where the other survives
                        # and the complete-case estimate is biased, exactly the defect
                        # phase25's BIAS_WARNING records. Counted apart, not together.
                        failed = [a for a in ARMS if not np.isfinite(row[a])]
                        dropouts["dropped_arm_nonfinite"] += 1
                        if len(failed) == len(ARMS):
                            dropouts["dropped_all_arms_common_mode"] += 1
                        else:
                            dropouts["dropped_differential"] += 1
                        dropouts["arm_specific_keys"].append(
                            {"key": f"{scene}-{nd}-{rep}", "cell": f"{part}/{nd}",
                             "nonfinite_arms": failed,
                             "common_mode": len(failed) == len(ARMS)})
                        continue
                    dropouts["complete"] += 1
                    raw_list.append(raw_value)
                    for arm in ARMS:
                        arm_lists[arm].append(row[arm])
                    per_sequence[f"{scene}-{nd}-{rep}"] = row

            cell = {"sequences": len(raw_list), "per_sequence": per_sequence,
                    "raw_mesr": float(np.mean(raw_list)) if raw_list else None}
            for arm in ARMS:
                cell[arm] = float(np.mean(arm_lists[arm])) if arm_lists[arm] else None
            results[part][nd] = cell
            args.output.parent.mkdir(parents=True, exist_ok=True)
            print(f"{part} {nd}: raw={cell['raw_mesr']:.4f} "
                  f"sup0={cell['support_beta0']:.4f} mag0={cell['magnitude_beta0']:.4f} "
                  f"supG={cell['support_gated']:.4f} magG={cell['magnitude_gated']:.4f} "
                  f"n={cell['sequences']}", flush=True)

    if reused:
        dropouts = recompute_dropouts(reused)

    # ---- cell means and the published anchors -------------------------------------
    cells = [results[p][nd] for p in results for nd in results[p]]
    cell_means = {arm: float(np.mean([c[arm] for c in cells])) for arm in ARMS}
    cell_means["raw"] = float(np.mean([c["raw_mesr"] for c in cells]))
    anchor_drift = {k: cell_means[k] - v for k, v in ANCHORS.items()}

    # ---- scene-clustered paired analysis -------------------------------------------
    comparisons: Dict[str, Dict] = {}
    for arm, ref in CONTRASTS:
        # Reps of one scene in one cell are repeated measures, so they average first.
        by_scene_cell: Dict[Tuple[str, str, str], List[float]] = {}
        for part in results:
            for nd in results[part]:
                for key, row in results[part][nd]["per_sequence"].items():
                    by_scene_cell.setdefault((part, nd, row["scene"]), []).append(
                        row[arm] - row[ref])
        collapsed = [(f"{p}/{nd}/{s}", float(np.mean(v)))
                     for (p, nd, s), v in by_scene_cell.items()]

        # THE CLUSTER. Scene is crossed with ND, not nested in it, so a scene's four ND
        # observations move together or not at all. 96 clusters, not 384 observations.
        clusters: Dict[Tuple[str, str], List[float]] = {}
        for (pt, nd, sc), v in by_scene_cell.items():
            clusters.setdefault((pt, sc), []).append(float(np.mean(v)))
        cluster_list = [(f"{pt}/{sc}", vals) for (pt, sc), vals in clusters.items()]

        per_cell = {}
        for part in results:
            for nd in results[part]:
                vals = [v for k, v in collapsed if k.startswith(f"{part}/{nd}/")]
                per_cell[f"{part}/{nd}"] = float(np.mean(vals)) if vals else None

        entry = cluster_bootstrap_blocks(cluster_list, args.replicates, rng)
        entry["observations"] = len(collapsed)
        entry["sign_flip_test"] = sign_flip_test_blocks(
            [np.asarray(v, dtype=np.float64) for _, v in cluster_list],
            args.permutations, rng)
        entry["per_cell_mean_delta"] = per_cell
        entry["cells_positive"] = int(sum(1 for v in per_cell.values()
                                          if v is not None and v > 0))
        entry["cell_mean_delta"] = cell_means[arm] - cell_means[ref]
        comparisons[f"{arm}_minus_{ref}"] = entry

    operand_ungated = comparisons["support_beta0_minus_magnitude_beta0"]["cell_mean_delta"]
    operand_gated = comparisons["support_gated_minus_magnitude_gated"]["cell_mean_delta"]
    gate_cost = comparisons["support_beta0_minus_support_gated"]["cell_mean_delta"]
    attribution = {
        "operand_ungated": operand_ungated,
        "operand_gated": operand_gated,
        "gate_effect_on_headline": -gate_cost,
        "operand_change_from_closing_the_gate": operand_ungated - operand_gated,
    }

    payload = {
        "phase": "55 E-MLB with the gate closed, the configuration LED selects",
        "purpose": ("Phase 54 rejects the correlation-sign gate on labelled recorded "
                    "noise. This re-measures the held-out E-MLB rows with the gate "
                    "closed, so the adopted configuration rests on a measurement rather "
                    "than on the gate being nearly inert here"),
        "why_pairing_should_be_complete": (
            "both arms are our own implementation at one frozen retention, so each should "
            "be defined wherever the raw stream is, and the complete-case bias that forced "
            "the n = 8 headline for the EDformer contrast (phase25 BIAS_WARNING) should not "
            "arise. `dropouts` tests that rather than assuming it: sequences whose RAW MESR "
            "is non-finite are unusable for every arm and cannot bias a paired contrast, "
            "while any sequence lost to a SINGLE arm is counted and named separately. Read "
            "`pairing_is_complete` before relying on the clustered intervals"),
        "arms": {
            "support_gated": "beta*G - log(1 + S_r2); the adopted score (anchor)",
            "magnitude_gated": "beta*G - R_same_r2; its radius-matched control (anchor)",
            "support_beta0": "-log(1 + S_r2); ranking by the 5-bit integer S alone",
            "magnitude_beta0": "-R_same_r2; the radius-matched control, ungated",
        },
        "gate": "beta = 1 iff corr(G, R_same_r1) < 0 in the two gated arms; the two beta0 "
                "arms ignore the decision entirely",
        "retention": RETENTION,
        "metric": "MESR = mean ESR over 30,000-event slices",
        "unit_of_analysis": ("scene within cell; the three reps of a scene are repeated "
                             "measures of one recording and are averaged before clustering"),
        "seed": args.seed,
        "dropouts": dropouts,
        "pairing_is_complete": dropouts["dropped_differential"] == 0,
        "cell_means": cell_means,
        "anchor_drift_vs_published": anchor_drift,
        "comparisons": comparisons,
        "attribution": attribution,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print("\ndropouts:", json.dumps({k: v for k, v in dropouts.items()
                                       if k != "arm_specific_keys"}, indent=2))
    print("pairing_is_complete:", dropouts["dropped_differential"] == 0)
    print("cell means:", json.dumps(cell_means, indent=2))
    print("anchor drift:", json.dumps(anchor_drift, indent=2))
    print("attribution:", json.dumps(attribution, indent=2))
    for name, entry in comparisons.items():
        print(f"{name}: mean={entry['mean']:+.4f} "
              f"ci=[{entry['ci95'][0]:+.5f}, {entry['ci95'][1]:+.5f}] "
              f"p={entry['sign_flip_test']['p_two_sided']:.2e} "
              f"cells+={entry['cells_positive']}/8 "
              f"scenes={entry['clusters_ahead']}/{entry['n_clusters']}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
