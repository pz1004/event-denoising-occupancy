"""Phase 20: the random-subsampling null on E-MLB, in Phase 10's exact units.

Why this exists
---------------
Table 2 reports our MESR at the frozen ``r = 0.28``, i.e. after discarding 72 % of every
block. MESR rises when a stream is concentrated, and deleting events concentrates it, so
*some* of that gain is available to any filter that merely throws events away. Without a
null the table cannot separate "our score discriminates" from "MESR rewards discarding".

The null is a **seeded uniform ranking**: it removes signal and noise in exactly the
proportion they occur, so any MESR it gains at ``r < 1`` is a gain from discarding alone.

This script deliberately mirrors ``phase10_emlb_blind.main`` cell for cell -- same reader,
same ``SLICE``, same ``BLOCK``, same ``retain_mask``, same parts/ND levels/reps, same
uncapped default -- so its rows are directly comparable with Table 2's. It is a separate
script rather than a flag on Phase 10 because Phase 10's artifact is already published and
regenerating it must stay byte-reproducible.

It skips ``score_stream_both`` entirely, which is where Phase 10 spent nearly all of its
5,847 s, so this pass is dominated by file reads and the MESR computation.

The seed matches the sibling critique project's ``RANDOM_NULL_SEED`` so the two lines of
work describe the same null.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .core import StateConfig
from .phase5_emlb_mesr import (
    ND_LEVELS,
    PAPER_RAW,
    PARTS,
    SLICE,
    mesr,
    read_aedat4,
    retain_mask,
)
from .phase10_emlb_blind import BLOCK, DEV_CALIBRATED_RETENTION, PROJECT_ROOT

#: Same seed as the sibling project's `dataset_assessment.denoisors.RANDOM_NULL_SEED`, so
#: "the random null" means one thing across both papers.
RANDOM_NULL_SEED = 20260726


def random_null_scores(n_events: int, seed: int = RANDOM_NULL_SEED) -> np.ndarray:
    """A uniform ranking independent of the events. Reseeded per recording.

    Reseeding per recording (rather than once per run) keeps each cell's null independent
    of how many recordings happened to precede it, so a partial run and a full run agree
    on every cell they share.
    """

    return np.random.default_rng(seed).random(n_events, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--retention", type=float, default=DEV_CALIBRATED_RETENTION,
                        help="Must match the retention Table 2's row uses.")
    parser.add_argument("--max-events", type=int, default=0, help="0 = uncapped, as Phase 10.")
    parser.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--nd-levels", nargs="+", default=list(ND_LEVELS))
    parser.add_argument("--seed", type=int, default=RANDOM_NULL_SEED)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
                             / "phase20_emlb_random_null.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = StateConfig(width=346, height=260, taus_us=(250.0,))
    cap: Optional[int] = args.max_events if args.max_events > 0 else None
    results: Dict[str, Dict[str, Dict]] = {}

    for part in args.parts:
        root = args.emlb_root / part
        scenes = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
        results[part] = {}
        for nd in args.nd_levels:
            raw_list: List[float] = []
            null_fixed: List[float] = []
            for scene in scenes:
                for rep in args.reps:
                    path = root / scene / f"{scene}-{nd}-{rep}.aedat4"
                    if not path.exists():
                        continue
                    events = read_aedat4(path, cap)
                    if events is None or len(events) < SLICE:
                        continue
                    x, y = events["x"], events["y"]
                    raw_list.append(mesr(x, y, config.width, config.height))
                    scores = random_null_scores(len(events), args.seed)
                    keep = retain_mask(scores, args.retention, BLOCK)
                    null_fixed.append(mesr(x[keep], y[keep], config.width, config.height))
            raw_mean = float(np.nanmean(raw_list)) if raw_list else None
            null_mean = float(np.nanmean(null_fixed)) if null_fixed else None
            results[part][nd] = {
                "sequences": len(raw_list),
                "raw_mesr": raw_mean,
                "random_null_fixed_mesr": null_mean,
                "delta_over_raw": (None if raw_mean is None or null_mean is None
                                   else null_mean - raw_mean),
                "paper_raw": PAPER_RAW[part][nd],
            }
            print(f"{part}/{nd}: n={len(raw_list)} raw={raw_mean} null={null_mean}",
                  flush=True)

    payload = {
        "phase": "20 random-subsampling null on E-MLB, Phase 10 units",
        "purpose": ("separate 'the score discriminates' from 'MESR rewards discarding' for "
                    "the Table 2 row, which is measured at the same fixed retention"),
        "null": ("seeded uniform per-event ranking, reseeded per recording; removes signal "
                 "and noise in the proportion they occur"),
        "retention": f"fixed r = {args.retention}, matching the Table 2 row",
        "metric": "MESR = mean ESR over 30,000-event slices (same mesr() as Phase 5/10)",
        "settings": {"max_events_per_sequence": args.max_events, "reps": args.reps,
                     "slice": SLICE, "block": BLOCK, "seed": args.seed},
        "results": results,
        "timestamp_repairs": int(read_aedat4.repairs),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
