"""Phase 58: EBF on held-out E-MLB, at retention matched to ours.

WHY MATCHED RETENTION IS THE ONLY HONEST COMPARISON HERE. MESR is not a labelled accuracy
metric; it rewards spatial concentration, and concentration can be bought by deleting more
events. Table 4 already carries a random-deletion null for exactly that reason. A filter
compared at a different retention is therefore not compared at all, so EBF is run at the
same frozen r = 0.28 our own row uses: keep the 28% of each block its score ranks most
signal-like. EBF's own threshold T_EBF is the one constant its authors call
application-dependent and never fix, so there is no published operating point to honour
instead, and matching ours is both the fair choice and the only defined one.

WHAT IS RECOMPUTED AND WHAT IS JOINED. This phase reads each sequence once, computes the
Raw MESR and EBF's MESR, and then *joins* our own arms by sequence key from Phase 55's
artefact rather than re-scoring them. That makes the pairing exact -- same sequence, same
event cap, same slicing -- at a quarter of the cost, and it makes the comparison impossible
to fudge, because our side of every pair is a number that was already published. The join is
checked: the Raw MESR recomputed here must equal the Raw MESR stored there, to six decimals,
for every sequence. Raw depends on no filter, so any disagreement means the two runs did not
read the same events, and the run refuses to continue.

Unit of analysis is Phase 49's and Phase 55's, unchanged: the cluster is the scene within
illumination part, 96 of them, and a scene's four ND levels move together. Resampling the
384 scene-by-cell units instead would be wrong for the same reason it is wrong there.

EBF runs at its published s = 5, tau = 32 ms. Nothing is refitted on E-MLB for either side.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ebf import EBF_S, EBF_TAU_US, check_equivalence, check_properties
from .phase5_emlb_mesr import ND_LEVELS, PARTS, SLICE, mesr, read_aedat4, retain_mask
from .phase55_emlb_ungated import ARTIFACTS, BLOCK, PROJECT_ROOT, RETENTION
from .phase57_ebf_labelled import ebf_noise_score
from .retention import otsu_retention

WIDTH, HEIGHT = 346, 260
#: Our arms, joined by sequence key from the artefact that published them.
OURS_ARTEFACT = "phase55_emlb_ungated.json"
JOIN_ARMS = ("support_beta0", "support_gated", "magnitude_beta0", "magnitude_gated")
ANCHOR_TOLERANCE = 1e-6


def ebf_streams(events: np.ndarray) -> Dict[str, np.ndarray]:
    """EBF's score in this paper's orientation, blockwise as ours is scored."""

    ordered = np.ascontiguousarray(
        np.column_stack((events["timestamp"].astype(np.int64),
                         events["x"].astype(np.int64),
                         events["y"].astype(np.int64),
                         (events["polarity"].astype(np.int64) > 0).astype(np.int64))),
        dtype=np.int64)
    # EBF is causal and keeps one map for the whole stream, so it is scored once end to
    # end rather than per block. Only the retention rule is blockwise, as ours is.
    return {"ebf": ebf_noise_score(ordered, WIDTH, HEIGHT, float(EBF_TAU_US))}


def otsu_mask(scores: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Keep the lowest-scoring share of each block, with the share set by Otsu."""

    keep = np.zeros(len(scores), dtype=bool)
    for start in range(0, len(scores), block):
        chunk = scores[start:start + block]
        if len(chunk) == 0:
            continue
        n_keep = max(1, int(round(len(chunk) * otsu_retention(chunk))))
        keep[start + np.argsort(chunk, kind="stable")[:n_keep]] = True
    return keep


def cluster_bootstrap(clusters: List[Tuple[str, List[float]]], replicates: int,
                      rng: np.random.Generator) -> Dict:
    """Resample whole scenes. A scene's ND levels are drawn together or not at all."""

    blocks = [np.asarray(v, dtype=np.float64) for _, v in clusters]
    n = len(blocks)
    if n < 2:
        return {"n_clusters": n, "mean": float("nan"), "ci95": [float("nan")] * 2}
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([b.size for b in blocks])
    draws = rng.integers(0, n, size=(replicates, n))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    flat = np.concatenate(blocks)
    per_cluster = np.array([b.mean() for b in blocks])
    return {
        "n_clusters": n,
        "mean": float(flat.sum() / counts.sum()),
        "bootstrap_se": float(means.std(ddof=1)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "clusters_ahead": int((per_cluster > 0).sum()),
        "clusters_behind": int((per_cluster < 0).sum()),
        "clusters_tied": int((per_cluster == 0).sum()),
        "replicates": replicates,
    }


def sign_flip(clusters: List[Tuple[str, List[float]]], draws: int,
              rng: np.random.Generator) -> Dict:
    values = np.array([np.mean(v) for _, v in clusters], dtype=np.float64)
    if values.size == 0:
        return {"n_clusters": 0, "p_two_sided": float("nan")}
    observed = float(values.mean())
    signs = rng.choice((-1.0, 1.0), size=(draws, values.size))
    null_means = (signs * values).mean(axis=1)
    p = float((np.abs(null_means) >= abs(observed) - 1e-15).mean())
    return {"n_clusters": int(values.size), "observed_mean": observed, "draws": draws,
            "p_two_sided": max(p, 1.0 / draws)}


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
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--ours", type=Path, default=ARTIFACTS / OURS_ARTEFACT)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase58_ebf_emlb.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    check_properties()
    worst, _ = check_equivalence()
    print(f"EBF kernel verified against the released code, max |diff| = {worst:.2e}, "
          f"and against four properties the released code cannot vouch for")

    ours = json.loads(args.ours.read_text())["results"]
    joined: Dict[str, Dict[str, float]] = {}
    for part, levels in ours.items():
        for nd, cell in levels.items():
            for key, row in cell["per_sequence"].items():
                joined[key] = row
    if not joined:
        raise SystemExit(f"{args.ours} carries no per-sequence rows to join against.")
    # A stale artefact -- one written by an earlier phase under different arm names --
    # would otherwise join cleanly and produce empty contrasts without ever erroring.
    sample = next(iter(joined.values()))
    missing = [arm for arm in JOIN_ARMS if arm not in sample]
    if missing:
        raise SystemExit(
            f"{args.ours} does not carry the arms {missing}; it holds "
            f"{sorted(k for k in sample if k not in ('scene', 'rep'))}. This is the "
            f"stale artefact, not the re-run. Refusing to produce empty contrasts.")

    cap: Optional[int] = args.max_events if args.max_events > 0 else None
    results: Dict[str, Dict[str, Dict]] = {}
    anchor_worst = 0.0
    unjoined = 0

    for part in args.parts:
        root = args.emlb_root / part
        scenes = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
        results[part] = {}
        for nd in args.nd_levels:
            per_sequence: Dict[str, Dict[str, float]] = {}
            step = time.perf_counter()
            for scene in scenes:
                for rep in args.reps:
                    key = f"{scene}-{nd}-{rep}"
                    path = root / scene / f"{key}.aedat4"
                    if not path.exists():
                        continue
                    if key not in joined:
                        # Phase 55 dropped this sequence. Keeping it here would make the
                        # contrast a complete-case estimate on one side only.
                        unjoined += 1
                        continue
                    events = read_aedat4(path, cap)
                    if events is None or len(events) < SLICE:
                        continue
                    x, y = events["x"], events["y"]
                    raw_value = float(mesr(x, y, WIDTH, HEIGHT))
                    drift = abs(raw_value - float(joined[key]["raw_mesr"]))
                    anchor_worst = max(anchor_worst, drift)
                    if drift > ANCHOR_TOLERANCE:
                        raise SystemExit(
                            f"Raw MESR anchor failed on {key}: {raw_value:.8f} here "
                            f"against {joined[key]['raw_mesr']:.8f} in {OURS_ARTEFACT} "
                            f"(drift {drift:.2e}). The two runs did not read the same "
                            f"events, so the join is not a pairing. Refusing to continue.")

                    scores = ebf_streams(events)["ebf"]
                    keep = retain_mask(scores, RETENTION, BLOCK)
                    ebf_value = float(mesr(x[keep], y[keep], WIDTH, HEIGHT))
                    otsu_keep = otsu_mask(scores)
                    otsu_value = float(mesr(x[otsu_keep], y[otsu_keep], WIDTH, HEIGHT))
                    if not (np.isfinite(raw_value) and np.isfinite(ebf_value)):
                        continue
                    per_sequence[key] = {
                        "scene": scene, "rep": rep, "raw_mesr": raw_value,
                        "ebf": ebf_value,
                        "ebf_otsu": otsu_value,
                        "ebf_otsu_retention": float(otsu_keep.mean()),
                        **{arm: float(joined[key][arm]) for arm in JOIN_ARMS
                           if arm in joined[key]},
                    }
            cell = {"sequences": len(per_sequence), "per_sequence": per_sequence}
            for field in ("raw_mesr", "ebf", "ebf_otsu") + JOIN_ARMS:
                vals = [r[field] for r in per_sequence.values() if field in r]
                cell[field] = float(np.mean(vals)) if vals else None
            results[part][nd] = cell
            print(f"{part}/{nd}: n={cell['sequences']:4d} raw={cell['raw_mesr']:.4f} "
                  f"EBF={cell['ebf']:.4f} ours={cell.get('support_beta0')} "
                  f"[{time.perf_counter() - step:.0f}s]", flush=True)

    cell_means = {f"{p}/{nd}": results[p][nd] for p in results for nd in results[p]}
    summary = {field: float(np.mean([c[field] for c in cell_means.values()
                                     if c.get(field) is not None]))
               for field in ("raw_mesr", "ebf", "ebf_otsu") + JOIN_ARMS}

    contrasts: Dict[str, Dict] = {}
    for arm in JOIN_ARMS:
        by_scene: Dict[str, List[float]] = {}
        for part in results:
            for nd in results[part]:
                for row in results[part][nd]["per_sequence"].values():
                    if arm not in row:
                        continue
                    by_scene.setdefault(f"{part}/{row['scene']}", []).append(row["ebf"] - row[arm])
        clusters = list(by_scene.items())
        if not clusters:
            continue
        contrasts[f"ebf_minus_{arm}"] = {
            "cluster_bootstrap": cluster_bootstrap(clusters, args.replicates, rng),
            "sign_flip": sign_flip(clusters, args.permutations, rng),
            "cell_mean_delta": summary["ebf"] - summary[arm],
        }

    payload = {
        "phase": "58 EBF on held-out E-MLB at matched retention",
        "baseline": {
            "method": "EBF, Guo et al., IEEE TCSVT 35(10):9889-9894, 2025",
            "implementation": "ours, verified against a literal transcription of the "
                              "released repository (github.com/shicy17/EBF)",
            "kernel_vs_released_max_abs_diff": worst,
            "s": EBF_S, "tau_us": EBF_TAU_US,
            "retention": RETENTION,
            "retention_note": "matched to ours; T_EBF is application-dependent and "
                              "never fixed by its authors, so no published point exists",
            "causality": "strictly causal; ours takes a block-local bidirectional pass",
        },
        "join": {"source": OURS_ARTEFACT, "arms": list(JOIN_ARMS),
                 "raw_mesr_anchor_worst_drift": anchor_worst,
                 "sequences_absent_from_source": unjoined},
        "unit_of_analysis": "scene within illumination part; a scene's ND levels move together",
        "metric": "MESR, mean over 30,000-event slices",
        "cell_means": {k: {f: v[f] for f in ("sequences", "raw_mesr", "ebf", "ebf_otsu")
                           + JOIN_ARMS if f in v} for k, v in cell_means.items()},
        "summary": summary,
        "contrasts": contrasts,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\nraw MESR anchor: worst drift {anchor_worst:.2e} "
          f"({unjoined} sequences absent from {OURS_ARTEFACT})")
    print("cell-mean MESR: " + "  ".join(f"{f}={summary[f]:.4f}" for f in
          ("raw_mesr", "ebf", "ebf_otsu") + JOIN_ARMS if summary.get(f) is not None))
    for name, c in contrasts.items():
        b = c["cluster_bootstrap"]
        print(f"{name}: {b['mean']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] "
              f"ahead {b['clusters_ahead']} behind {b['clusters_behind']} "
              f"tied {b['clusters_tied']} of {b['n_clusters']}, "
              f"p={c['sign_flip']['p_two_sided']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
