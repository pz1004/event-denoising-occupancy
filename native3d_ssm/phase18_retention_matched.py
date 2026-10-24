"""Phase 18: retention-matched E-MLB comparison against the reproduced transformer.

Motivation (REVIEW_theory_empirics.md, P0-2). Phase 10 scored our filter at a single frozen
retention r = 0.28 on every cell, while the reproduced EDformer's released rule
(`sigmoid(f) >= 0.005`) is adaptive and spans 0.483 -> 0.081 across the eight cells. Regressing
the Phase 10 per-cell margin on the retention differential gives r = 0.83, R^2 = 0.69, so most of
the reported win/loss pattern tracks how many events each method discarded rather than how well it
discarded them. The structural ratio rewards removal -- the draft says so itself in Sec. 6.2 -- so
an unmatched comparison is not interpretable.

This phase re-scores the *same* adopted support score at each cell's EDformer retention, plus a
sweep, so the margin can be read at matched removal and as a function of retention.

Nothing here is selected on E-MLB. The retentions are read from the EDformer reproduction JSON;
the score, gate, radius and block size are all frozen from Phases 8/11/14.

Scores come from `phase10_emlb_blind.score_stream_both` unchanged, so the r = 0.28 column of this
run must reproduce `phase10_emlb_blind_support.json` exactly. That equality is the correctness
check for this script and is asserted in the payload.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .core import StateConfig
from .phase5_emlb_mesr import ND_LEVELS, PARTS, SLICE, mesr, read_aedat4, retain_mask
from .phase10_emlb_blind import BLOCK, score_stream_both

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Frozen from Phase 9 dev MESR. Reproduced here only as the audit column.
DEV_CALIBRATED_RETENTION = 0.28

# Retention sweep, chosen to bracket the EDformer range [0.081, 0.483] without being
# selected on E-MLB outcomes.
SWEEP = (0.08, 0.10, 0.15, 0.20, 0.25, 0.28, 0.35, 0.45, 0.50)

EDFORMER_JSON = PROJECT_ROOT / "reproduction/results/emlb_edformer_mesr.json"


def edformer_reference() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per-cell EDformer MESR and mean retention from the reproduction run."""

    payload = json.loads(EDFORMER_JSON.read_text())
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for part, cells in payload["results"].items():
        out[part] = {}
        for nd, cell in cells.items():
            out[part][nd] = {
                "edformer_mesr": float(cell["edformer_mesr"]),
                "edformer_retention": float(cell["mean_retention"]),
                "edformer_raw_mesr": float(cell["raw_mesr"]),
            }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--score-radius", type=int, default=2,
                        help="Ring radius for the support score; gate key stays radius 1.")
    parser.add_argument("--max-events", type=int, default=0, help="0 = uncapped, as Phase 10.")
    parser.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--nd-levels", nargs="+", default=list(ND_LEVELS))
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT
                        / "native3d_ssm/experiments/analytic_contrast/phase18_retention_matched.json")
    return parser.parse_args()


def build_payload(args, results, reference, started) -> Dict:
    return {
        "phase": "18 retention-matched E-MLB comparison (REVIEW P0-2)",
        "question": ("Does our Phase 10 E-MLB margin survive when our retention is matched to the "
                     "reproduced EDformer's per-cell retention?"),
        "score": "s = beta*G - log(1 + ring_support), score radius 2, gate key radius 1",
        "gate": "beta = 1 iff corr(G, R_same) < 0, theta = 0 theory-fixed",
        "retention_source": ("per-cell mean_retention of the reproduced EDformer, read from "
                             "reproduction/results/emlb_edformer_mesr.json. Not selected on E-MLB "
                             "outcomes -- it is the baseline's own operating point."),
        "audit": ("the r = 0.28 column must equal phase10_emlb_blind_support.json's "
                  "support_score_fixed_mesr for every cell; scores come from the same "
                  "score_stream_both call."),
        "metric": "MESR = mean ESR over 30,000-event slices (numpy ESR, bit-exact vs cuke-emlb)",
        "settings": {"max_events_per_sequence": args.max_events, "reps": args.reps,
                     "slice": SLICE, "block": BLOCK, "sweep": list(SWEEP),
                     "score_radius": args.score_radius},
        "edformer_reference": reference,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    reference = edformer_reference()
    config = StateConfig(width=346, height=260, taus_us=(250.0,))
    score_config = (StateConfig(width=346, height=260, taus_us=(250.0,),
                                radius=args.score_radius)
                    if args.score_radius != 1 else None)
    results: Dict[str, Dict[str, Dict]] = {}
    cap: Optional[int] = args.max_events if args.max_events > 0 else None

    for part in args.parts:
        root = args.emlb_root / part
        scenes = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
        results[part] = {}
        for nd in args.nd_levels:
            matched_r = reference[part][nd]["edformer_retention"]
            grid = sorted(set(SWEEP) | {round(matched_r, 4)})
            raw_list: List[float] = []
            per_r: Dict[float, List[float]] = {r: [] for r in grid}
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
                    _, _, sup, _, _ = score_stream_both(events, config, BLOCK, score_config)
                    for r in grid:
                        keep = retain_mask(sup, r, BLOCK)
                        per_r[r].append(mesr(x[keep], y[keep], config.width, config.height))

            raw_mean = float(np.nanmean(raw_list)) if raw_list else None
            by_retention = {f"{r:.4f}": (float(np.nanmean(v)) if v else None)
                            for r, v in per_r.items()}
            ours_matched = by_retention[f"{round(matched_r, 4):.4f}"]
            ours_frozen = by_retention[f"{DEV_CALIBRATED_RETENTION:.4f}"]
            ed = reference[part][nd]
            cell = {
                "sequences": len(raw_list),
                "raw_mesr": raw_mean,
                "mesr_by_retention": by_retention,
                "matched_retention": matched_r,
                "ours_at_matched_retention": ours_matched,
                "ours_at_frozen_0.28": ours_frozen,
                "edformer_mesr": ed["edformer_mesr"],
                "margin_matched": (None if ours_matched is None
                                   else ours_matched - ed["edformer_mesr"]),
                "margin_frozen_0.28": (None if ours_frozen is None
                                       else ours_frozen - ed["edformer_mesr"]),
            }
            results[part][nd] = cell
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(build_payload(args, results, reference, started),
                                              indent=2, sort_keys=True) + "\n")
            print(f"{part} {nd}: raw={raw_mean:.4f} | ours@{matched_r:.3f}(matched)="
                  f"{ours_matched:.4f} | ours@0.28={ours_frozen:.4f} | "
                  f"EDformer={ed['edformer_mesr']:.4f} | "
                  f"margin_matched={cell['margin_matched']:+.4f} "
                  f"(frozen {cell['margin_frozen_0.28']:+.4f}) | n={len(raw_list)}", flush=True)

    args.output.write_text(json.dumps(build_payload(args, results, reference, started),
                                      indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
