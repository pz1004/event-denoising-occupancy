"""Phase 10: blind E-MLB MESR with the Phase 8 score and the dev-calibrated retention.

Pre-registered in status.md **before this ran**: retention **r = 0.28** comes from Phase 9's
sweep of MESR over the two *labeled dev* sets and has never been fitted, tuned, or inspected
on E-MLB. Score is the Phase 8 winner, unchanged:

    s = beta * G - (R_same + R_opposite),   beta = 1 iff corr(G, R_same) < 0,  theta = 0

Note the gate key stays `corr(G, R_same)` -- Phase 8 measured that keying on the new ring
costs 0.0058 AUC because it pushes DND21 toward theta = 0.

Why a new script rather than `phase5_emlb_mesr.py --retentions 0.28`: that script hardcodes
the **old** score `s = beta*G - R_same`. Applying a retention calibrated on the new score to
the old score would be a mismatched test, and its existing rows are cited in status.md, so it
is left untouched.

Four rows, all on the identical event stream so the comparison is clean:

  raw                  regression anchor -- must reproduce 0.8147 (uncapped, reps 1-3)
  new score @ r=0.28   **the blind result** Phase 9 predicts
  new score @ adaptive `gated_retention` -- does the Phase 9 "retention drift" persist
                       once the score changes the atom structure?
  old score @ r=0.28   ablation isolating the score change from the retention change

Protocol identical to Phase 5/6b: 250 us bidirectional, 10,240-event blocks, MESR = mean ESR
over consecutive 30,000-event slices (numpy ESR verified bit-exact against official
cuke-emlb), 144 sequences per cell (reps 1-3), uncapped.
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
    PAPER_EDMAMBA,
    PAPER_RAW,
    PARTS,
    SLICE,
    mesr,
    read_aedat4,
    retain_mask,
)
from .phase8_score_upgrade import operands, safe_corr
from .retention import gated_retention

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK = 10_240
DEV_CALIBRATED_RETENTION = 0.28      # Phase 9, dev MESR only. Never fitted on E-MLB.

# Uncapped reference rows already on disk, for side-by-side reporting only.
REFERENCE_UNCAPPED = {
    "raw": 0.8147,
    "old_score_fixed_0.3": 0.9456,
    "old_score_adaptive": 0.8773,
    "edformer_reproduced": 1.0069,
}


def score_stream_both(events: np.ndarray, config: StateConfig,
                      block: int = BLOCK, score_config: Optional[StateConfig] = None):
    """Per-event old / new / support scores, plus per-block corr and beta.

    `score_config` (radius 2) supplies the ring used by the Phase 14 support operand; the
    gate key stays on `config` (radius 1), per Phase 11.
    """

    ordered = np.ascontiguousarray(
        np.column_stack((events["timestamp"].astype(np.int64),
                         events["x"].astype(np.int64),
                         events["y"].astype(np.int64),
                         (events["polarity"].astype(np.int64) > 0).astype(np.int64))),
        dtype=np.int64)
    old = np.zeros(len(ordered), dtype=np.float32)
    new = np.zeros(len(ordered), dtype=np.float32)
    sup = np.zeros(len(ordered), dtype=np.float32)
    corrs: List[float] = []
    betas: List[float] = []
    for start in range(0, len(ordered), block):
        chunk = ordered[start:start + block]
        if len(chunk) < 2:
            continue
        raw = bidirectional_features(chunk, config)
        g, r_same, r_opposite = operands(raw, config)
        corr = safe_corr(g, r_same)              # frozen Phase-3 key, deliberately unchanged
        beta = 1.0 if corr < 0.0 else 0.0
        old[start:start + len(chunk)] = beta * g - r_same
        new[start:start + len(chunk)] = beta * g - (r_same + r_opposite)
        cfg_s = score_config if score_config is not None else config
        raw_s = bidirectional_features(chunk, cfg_s) if score_config is not None else raw
        support = ring_support(raw_s, len(cfg_s.taus_us), cfg_s.feature_dim, cfg_s.radius)
        sup[start:start + len(chunk)] = beta * g - np.log1p(support)
        corrs.append(corr)
        betas.append(beta)
    return old, new, sup, corrs, betas


def adaptive_mask(scores: np.ndarray, corrs: List[float],
                  block: int = BLOCK) -> Tuple[np.ndarray, List[float]]:
    """Per-block keep-mask under the frozen `gated_retention` rule."""

    keep = np.zeros(len(scores), dtype=bool)
    chosen: List[float] = []
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
        chosen.append(retention)
    return keep, chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--retention", type=float, default=DEV_CALIBRATED_RETENTION,
                        help="Frozen retention from Phase 9 dev MESR. Do not tune on E-MLB.")
    parser.add_argument("--score-radius", type=int, default=2,
                        help="Ring radius for the support score; gate key stays radius 1.")
    parser.add_argument("--max-events", type=int, default=0, help="0 = uncapped (as Phase 5/6b).")
    parser.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--nd-levels", nargs="+", default=list(ND_LEVELS))
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase10_emlb_blind.json")
    return parser.parse_args()


def build_payload(args, results, started) -> Dict:
    return {
        "phase": "10 blind E-MLB MESR, Phase 8 score + dev-calibrated retention",
        "score": "s = beta*G - (R_same + R_opposite)",
        "gate": "beta = 1 iff corr(G, R_same) < 0, theta = 0 theory-fixed",
        "retention": (f"frozen r = {args.retention}, calibrated on dev MESR in Phase 9; "
                      "never fitted or inspected on E-MLB"),
        "preregistration": ("r = 0.28 and this protocol were written into status.md before the "
                            "run. Phase 9 selected it from DND21+DVSCLEAN MESR alone."),
        "metric": "MESR = mean ESR over 30,000-event slices (numpy ESR, bit-exact vs cuke-emlb)",
        "settings": {"max_events_per_sequence": args.max_events, "reps": args.reps,
                     "slice": SLICE, "block": BLOCK,
                     # Recorded explicitly: the adopted score uses radius 2 and the gate radius 1.
                     # Earlier artifacts omitted this, so the configuration behind the headline MESR
                     # had to be established by matching values against phase 18 rather than read.
                     "gate_radius": 1, "score_radius": args.score_radius},
        "reference_uncapped_rows": REFERENCE_UNCAPPED,
        "results": results,
        "timestamp_repairs": int(read_aedat4.repairs),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
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
            raw_list, new_fixed, new_adaptive, old_fixed = [], [], [], []
            sup_fixed, sup_adaptive = [], []
            corr_all, beta_all, adaptive_r = [], [], []
            #: Per-sequence values keyed by "{scene}-{nd}-{rep}", matching the key the
            #: EDformer reproduction emits, so the two sides can be paired sequence by
            #: sequence instead of only at the cell means.
            per_sequence: Dict[str, Dict[str, float]] = {}
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
                    old, new, sup, corrs, betas = score_stream_both(
                        events, config, BLOCK, score_config)
                    corr_all.extend(corrs)
                    beta_all.extend(betas)

                    keep = retain_mask(new, args.retention, BLOCK)
                    new_fixed.append(mesr(x[keep], y[keep], config.width, config.height))

                    keep_old = retain_mask(old, args.retention, BLOCK)
                    old_fixed.append(mesr(x[keep_old], y[keep_old], config.width, config.height))

                    keep_sup = retain_mask(sup, args.retention, BLOCK)
                    sup_fixed.append(mesr(x[keep_sup], y[keep_sup], config.width, config.height))
                    keep_sa, _ = adaptive_mask(sup, corrs, BLOCK)
                    sup_adaptive.append(mesr(x[keep_sa], y[keep_sa], config.width, config.height))

                    keep_ad, chosen = adaptive_mask(new, corrs, BLOCK)
                    adaptive_r.extend(chosen)
                    new_adaptive.append(mesr(x[keep_ad], y[keep_ad], config.width, config.height))
                    per_sequence[f"{scene}-{nd}-{rep}"] = {
                        "raw_mesr": float(raw_list[-1]),
                        "support_score_fixed_mesr": float(sup_fixed[-1]),
                        "support_score_adaptive_mesr": float(sup_adaptive[-1]),
                    }
            cell = {
                "sequences": len(raw_list),
                "per_sequence": per_sequence,
                "raw_mesr": float(np.nanmean(raw_list)) if raw_list else None,
                "new_score_fixed_mesr": float(np.nanmean(new_fixed)) if new_fixed else None,
                "new_score_adaptive_mesr": float(np.nanmean(new_adaptive)) if new_adaptive else None,
                "old_score_fixed_mesr": float(np.nanmean(old_fixed)) if old_fixed else None,
                "support_score_fixed_mesr": float(np.nanmean(sup_fixed)) if sup_fixed else None,
                "support_score_adaptive_mesr": (float(np.nanmean(sup_adaptive))
                                                if sup_adaptive else None),
                "mean_adaptive_retention": float(np.mean(adaptive_r)) if adaptive_r else None,
                "mean_corr_GR": float(np.mean(corr_all)) if corr_all else None,
                "frac_blocks_beta1": float(np.mean(beta_all)) if beta_all else None,
                "paper_raw": PAPER_RAW[part][nd],
                "paper_edmamba": PAPER_EDMAMBA[part][nd],
            }
            results[part][nd] = cell
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(build_payload(args, results, started),
                                              indent=2, sort_keys=True) + "\n")
            print(f"{part} {nd}: raw={cell['raw_mesr']:.4f} | NEW@r={args.retention}="
                  f"{cell['new_score_fixed_mesr']:.4f} | NEW@adaptive="
                  f"{cell['new_score_adaptive_mesr']:.4f} (r={cell['mean_adaptive_retention']:.3f}) "
                  f"| SUP@r={args.retention}={cell['support_score_fixed_mesr']:.4f} "
                  f"| OLD@r={args.retention}={cell['old_score_fixed_mesr']:.4f} "
                  f"| n={cell['sequences']}", flush=True)

    args.output.write_text(json.dumps(build_payload(args, results, started),
                                      indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
