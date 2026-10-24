"""Phase 6b: blind E-MLB MESR with the frozen adaptive-retention rule.

Phase 5 reported E-MLB MESR at four *fixed* retentions and noted the optimum moves
(r ~ 0.5 Daylight, r ~ 0.3 Night). Picking r from that sweep is metric selection --
MESR is a no-reference metric, so maximizing it on E-MLB is label-free yet is direct
optimization of the reported number. The r = 0.3 headline of Phase 5 is therefore an
*oracle* on the test metric, not a blind operating point.

This script instead applies `retention.gated_retention`, frozen on the labeled
development sets in Phase 6a (`phase6_dev_retention.json`), and never refitted here:

    corr(G,R) <  0  ->  Otsu threshold on the score histogram
    corr(G,R) >= 0  ->  reject the "no evidence" atom at the top score

Everything else is byte-identical to Phase 5: the same frozen gate beta = 1 iff
corr(G,R) < 0, s = beta*G - R, 250 us bidirectional, 10,240-event blocks, the same
numpy ESR (verified bit-exact against `cuke-emlb`), the same MESR over consecutive
30,000-event slices, the same 144 sequences per cell, uncapped.

Raw MESR is recomputed as a regression check: it must reproduce
`phase5_emlb_mesr.json` exactly, which proves the two runs are comparable.

E-MLB is +corr at every ND level, so this run exercises the corr >= 0 branch almost
exclusively; the corr < 0 branch remains validated on DND21 only. Reported honestly
rather than hidden: `frac_blocks_beta1` per cell records how often the other branch
fired.
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
    PAPER_EDMAMBA,
    PAPER_RAW,
    PARTS,
    SLICE,
    mesr,
    read_aedat4,
    score_stream,
)
from .retention import gated_retention

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK = 10_240

# Phase 5 fixed-retention MESR, for side-by-side reporting only (not recomputed).
PHASE5_FIXED_MESR = {
    "D-END": {"ND00": {"0.3": 0.8863, "0.5": 0.8953, "0.7": 0.8805, "0.9": 0.8443},
              "ND04": {"0.3": 0.8950, "0.5": 0.9009, "0.7": 0.8839, "0.9": 0.8459},
              "ND16": {"0.3": 0.8919, "0.5": 0.8971, "0.7": 0.8749, "0.9": 0.8365},
              "ND64": {"0.3": 0.8846, "0.5": 0.8815, "0.7": 0.8514, "0.9": 0.8084}},
    "N-END": {"ND00": {"0.3": 1.0360, "0.5": 1.0105, "0.7": 0.9712, "0.9": 0.9199},
              "ND04": {"0.3": 1.0346, "0.5": 0.9864, "0.7": 0.9205, "0.9": 0.8542},
              "ND16": {"0.3": 0.9754, "0.5": 0.9297, "0.7": 0.8691, "0.9": 0.8115},
              "ND64": {"0.3": 0.9610, "0.5": 0.9082, "0.7": 0.8437, "0.9": 0.7884}},
}


def adaptive_retain_mask(scores: np.ndarray, corrs: List[float],
                         block: int = BLOCK) -> tuple[np.ndarray, List[float]]:
    """Per-block keep-mask under the frozen gated retention rule.

    `corrs` comes from `score_stream` and holds one corr(G,R) per scored block, in
    order. Blocks too short to score (< 2 events) produce no corr, so they are kept
    whole -- the same conservative fallback the estimators use.
    """

    keep = np.zeros(len(scores), dtype=bool)
    retentions: List[float] = []
    index = 0
    for start in range(0, len(scores), block):
        chunk = scores[start:start + block]
        if len(chunk) == 0:
            continue
        if len(chunk) < 2 or index >= len(corrs):
            keep[start:start + len(chunk)] = True
            continue
        retention = gated_retention(chunk, corrs[index])
        index += 1
        n_keep = max(1, int(round(len(chunk) * retention)))
        order = np.argsort(chunk, kind="stable")[:n_keep]
        keep[start + order] = True
        retentions.append(retention)
    return keep, retentions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--max-events", type=int, default=0,
                        help="Per-sequence event cap; 0 = uncapped, matching Phase 5.")
    parser.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--nd-levels", nargs="+", default=list(ND_LEVELS))
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase6_emlb_adaptive.json")
    return parser.parse_args()


def build_payload(args, results, started) -> Dict:
    return {
        "phase": "6b E-MLB MESR with frozen adaptive retention",
        "metric": "MESR = mean ESR over 30,000-event slices (official cuke-emlb definition, numpy verified bit-exact)",
        "denoiser": ("frozen corr-gated Native3D: beta=1 iff corr(G,R)<0, s=beta*G-R, "
                     "250us bidirectional, 10,240-event blocks"),
        "retention_rule": ("frozen gated_retention from Phase 6a: corr<0 -> Otsu on the score "
                           "histogram; corr>=0 -> reject the top-score atom. Calibrated on "
                           "labeled DND21+DVSCLEAN only; not refitted on E-MLB."),
        "integrity_note": ("MESR is no-reference, so Phase 5's r=0.3 headline is an oracle on the "
                           "test metric. This run chooses retention blind, per block."),
        "settings": {"max_events_per_sequence": args.max_events, "reps": args.reps,
                     "slice": SLICE},
        "results": results,
        "timestamp_repairs": int(read_aedat4.repairs),
        "wall_seconds": time.perf_counter() - started,
    }


def _write(args, results, started) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_payload(args, results, started),
                                      indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = StateConfig(width=346, height=260, taus_us=(250.0,))
    results: Dict[str, Dict[str, Dict]] = {}

    for part in args.parts:
        root = args.emlb_root / part
        scenes = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
        results[part] = {}
        for nd in args.nd_levels:
            raw_mesr, adaptive_mesr = [], []
            corr_all, beta_all, retention_all, actual_all = [], [], [], []
            for scene in scenes:
                for rep in args.reps:
                    path = root / scene / f"{scene}-{nd}-{rep}.aedat4"
                    if not path.exists():
                        continue
                    cap: Optional[int] = args.max_events if args.max_events > 0 else None
                    events = read_aedat4(path, cap)
                    if events is None or len(events) < SLICE:
                        continue
                    x, y = events["x"], events["y"]
                    raw_mesr.append(mesr(x, y, config.width, config.height))
                    scores, corrs, betas = score_stream(events, config, BLOCK)
                    corr_all.extend(corrs)
                    beta_all.extend(betas)
                    keep, retentions = adaptive_retain_mask(scores, corrs, BLOCK)
                    retention_all.extend(retentions)
                    actual_all.append(float(keep.mean()))
                    adaptive_mesr.append(mesr(x[keep], y[keep], config.width, config.height))
            cell = {
                "sequences": len(raw_mesr),
                "raw_mesr": float(np.nanmean(raw_mesr)) if raw_mesr else None,
                "adaptive_mesr": float(np.nanmean(adaptive_mesr)) if adaptive_mesr else None,
                "mean_corr_GR": float(np.mean(corr_all)) if corr_all else None,
                "frac_blocks_beta1": float(np.mean(beta_all)) if beta_all else None,
                "mean_block_retention": float(np.mean(retention_all)) if retention_all else None,
                "std_block_retention": float(np.std(retention_all)) if retention_all else None,
                "mean_actual_retention": float(np.mean(actual_all)) if actual_all else None,
                "paper_raw": PAPER_RAW[part][nd],
                "paper_edmamba": PAPER_EDMAMBA[part][nd],
                "phase5_fixed_mesr": PHASE5_FIXED_MESR[part][nd],
            }
            results[part][nd] = cell
            _write(args, results, started)
            print(f"{part} {nd}: raw={cell['raw_mesr']:.4f} (paper {cell['paper_raw']:.3f}) "
                  f"adaptive={cell['adaptive_mesr']:.4f} "
                  f"r={cell['mean_block_retention']:.3f}+-{cell['std_block_retention']:.3f} "
                  f"corr={cell['mean_corr_GR']:+.3f} beta1={cell['frac_blocks_beta1']:.3f} "
                  f"n={cell['sequences']}", flush=True)

    _write(args, results, started)
    elapsed = time.perf_counter() - started
    print(f"\nWrote {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
