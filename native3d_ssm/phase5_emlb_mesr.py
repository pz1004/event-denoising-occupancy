"""Phase 5: E-MLB MESR evaluation of the corr-gated Native3D denoiser.

E-MLB is UNLABELED, so its metric is MESR (Mean Event Structural Ratio), not ROC
AUC. MESR scores the *denoised output*, so unlike Phases 0-4 this requires an
explicit retention decision (the operating-point track).

ESR is reimplemented in numpy and verified bit-exact against the official
`cuke-emlb` implementation (`EventStructuralRatio._calc_esr`, metric.py L178+):

    n    = per-pixel event count (ignore polarity, no decay, contribution 1)
    N    = events in slice, M = int(2N/3), K = W*H
    ntss = sum(n*(n-1)) / (N*(N-1))
    ln   = K - sum((1 - M/N)**n)
    ESR  = sqrt(ntss * ln)

MESR = mean ESR over consecutive 30,000-event slices (the official
`evalEventStorePerNumber` default), then averaged over sequences in a cell.

Denoiser: the frozen Phase-3/4 rule. Per 10,240-event block, compute the
bidirectional 250-us global (G) and same-polarity ring (R) log operands, gate
beta = 1 if corr(G,R) < 0 else 0 (theta = 0, theory-fixed), score
s = beta*G - R, and retain the lowest-scoring fraction `r` of each block.
Because MESR has no labels, `r` cannot be calibrated on E-MLB; a sweep over r is
reported so the retention/structure trade-off is explicit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import StateConfig, bidirectional_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ND_LEVELS = ("ND00", "ND04", "ND16", "ND64")
ND_DISPLAY = {"ND00": "ND1", "ND04": "ND4", "ND16": "ND16", "ND64": "ND64"}
PARTS = {"D-END": "Daylight", "N-END": "Night"}
SLICE = 30_000

# EDmamba paper Table 2 reference rows (for context only; not recomputed here).
PAPER_RAW = {"D-END": {"ND00": 0.821, "ND04": 0.824, "ND16": 0.815, "ND64": 0.786},
             "N-END": {"ND00": 0.890, "ND04": 0.824, "ND16": 0.786, "ND64": 0.768}}
PAPER_EDMAMBA = {"D-END": {"ND00": 0.976, "ND04": 0.990, "ND16": 0.985, "ND64": 0.972},
                 "N-END": {"ND00": 1.002, "ND04": 1.025, "ND16": 1.082, "ND64": 1.089}}


def esr(x: np.ndarray, y: np.ndarray, width: int, height: int) -> float:
    """Official E-MLB Event Structural Ratio for one slice (numpy, bit-exact)."""

    n_events = len(x)
    if n_events < 2:
        return float("nan")
    pixels = width * height
    counts = np.bincount(y.astype(np.int64) * width + x.astype(np.int64),
                         minlength=pixels).astype(np.float64)
    m = int(n_events * 2 / 3)
    eps = np.spacing(1)
    ntss = (counts * (counts - 1)).sum() / (n_events + eps) / (n_events - 1 + eps)
    ln = pixels - ((1 - m / n_events) ** counts).sum()
    return float(np.sqrt(ntss * ln))


def mesr(x: np.ndarray, y: np.ndarray, width: int, height: int,
         slice_size: int = SLICE) -> float:
    """Mean ESR over consecutive complete slices; incomplete tail dropped."""

    scores = [esr(x[i:i + slice_size], y[i:i + slice_size], width, height)
              for i in range(0, len(x) - slice_size + 1, slice_size)]
    return float(np.mean(scores)) if scores else float("nan")


def read_aedat4(path: Path, max_events: Optional[int]) -> Optional[np.ndarray]:
    import dv_processing as dv

    recording = dv.io.MonoCameraRecording(str(path))
    chunks: List[np.ndarray] = []
    total = 0
    while recording.isRunning():
        batch = recording.getNextEventBatch()
        if batch is None:
            continue
        array = batch.numpy()
        chunks.append(array)
        total += len(array)
        if max_events is not None and total >= max_events:
            break
    if not chunks:
        return None
    events = np.concatenate(chunks)
    if max_events is not None:
        events = events[:max_events]
    # Some released recordings contain overlapping packets. A stable sort keeps
    # source order for timestamp ties and changes no event value, matching the
    # repair already used by filter_analytic_aedat.py.
    if len(events) > 1 and np.any(np.diff(events["timestamp"].astype(np.int64)) < 0):
        events = events[np.argsort(events["timestamp"], kind="mergesort")]
        read_aedat4.repairs += 1
    return events


read_aedat4.repairs = 0


def score_stream(events: np.ndarray, config: StateConfig,
                 block: int = 10_240) -> Tuple[np.ndarray, List[float], List[float]]:
    """Frozen corr-gated score per event, plus per-block corr(G,R) and beta."""

    scale_count = len(config.taus_us)
    width = config.feature_dim
    ordered = np.ascontiguousarray(
        np.column_stack((events["timestamp"].astype(np.int64),
                         events["x"].astype(np.int64),
                         events["y"].astype(np.int64),
                         (events["polarity"].astype(np.int64) > 0).astype(np.int64))),
        dtype=np.int64)
    scores = np.zeros(len(ordered), dtype=np.float32)
    corrs: List[float] = []
    betas: List[float] = []
    for start in range(0, len(ordered), block):
        chunk = ordered[start:start + block]
        if len(chunk) < 2:
            continue
        raw = bidirectional_features(chunk, config)
        g = raw[:, 6 * scale_count] + raw[:, width + 6 * scale_count]
        r = raw[:, 2 * scale_count] + raw[:, width + 2 * scale_count]
        corr = float(np.corrcoef(g, r)[0, 1]) if g.std() > 0 and r.std() > 0 else 0.0
        beta = 1.0 if corr < 0.0 else 0.0
        scores[start:start + len(chunk)] = beta * g - r
        corrs.append(corr)
        betas.append(beta)
    return scores, corrs, betas


def retain_mask(scores: np.ndarray, retention: float, block: int = 10_240) -> np.ndarray:
    """Keep the lowest-scoring `retention` fraction within each block."""

    keep = np.zeros(len(scores), dtype=bool)
    for start in range(0, len(scores), block):
        chunk = scores[start:start + block]
        if len(chunk) == 0:
            continue
        n_keep = max(1, int(round(len(chunk) * retention)))
        order = np.argsort(chunk, kind="stable")[:n_keep]
        keep[start + order] = True
    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--max-events", type=int, default=300_000,
                        help="Per-sequence event cap (applied identically to Raw and filtered).")
    parser.add_argument("--reps", nargs="+", type=int, default=[1])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--nd-levels", nargs="+", default=list(ND_LEVELS))
    parser.add_argument("--retentions", nargs="+", type=float,
                        default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--headline-retention", type=float, default=0.5)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase5_emlb_mesr.json")
    return parser.parse_args()


def _dump(args, results, started) -> None:
    payload = {
        "phase": "5 E-MLB MESR",
        "metric": "MESR = mean ESR over 30,000-event slices (official cuke-emlb definition, numpy verified bit-exact)",
        "denoiser": "frozen corr-gated Native3D: beta=1 iff corr(G,R)<0, s=beta*G-R, 250us bidirectional, 10,240-event blocks",
        "caveat": "retention is a free operating point; E-MLB is unlabeled so it cannot be calibrated there. Sweep reported.",
        "settings": {"max_events_per_sequence": args.max_events, "reps": args.reps,
                     "retentions": args.retentions, "headline_retention": args.headline_retention,
                     "slice": SLICE},
        "results": results,
        "timestamp_repairs": int(read_aedat4.repairs),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
            raw_scores, filt_scores = [], {r: [] for r in args.retentions}
            corr_all, beta_all, retain_frac = [], [], []
            for scene in scenes:
                for rep in args.reps:
                    path = root / scene / f"{scene}-{nd}-{rep}.aedat4"
                    if not path.exists():
                        continue
                    cap = args.max_events if args.max_events and args.max_events > 0 else None
                    events = read_aedat4(path, cap)
                    if events is None or len(events) < SLICE:
                        continue
                    x, y = events["x"], events["y"]
                    raw_scores.append(mesr(x, y, config.width, config.height))
                    scores, corrs, betas = score_stream(events, config)
                    corr_all.extend(corrs)
                    beta_all.extend(betas)
                    for r in args.retentions:
                        keep = retain_mask(scores, r)
                        filt_scores[r].append(mesr(x[keep], y[keep], config.width, config.height))
                        if r == args.headline_retention:
                            retain_frac.append(float(keep.mean()))
            results[part][nd] = {
                "sequences": len(raw_scores),
                "raw_mesr": float(np.nanmean(raw_scores)) if raw_scores else None,
                "ours_mesr_by_retention": {str(r): (float(np.nanmean(v)) if v else None)
                                           for r, v in filt_scores.items()},
                "mean_corr_GR": float(np.mean(corr_all)) if corr_all else None,
                "frac_blocks_beta1": float(np.mean(beta_all)) if beta_all else None,
                "mean_actual_retention": float(np.mean(retain_frac)) if retain_frac else None,
                "paper_raw": PAPER_RAW[part][nd],
                "paper_edmamba": PAPER_EDMAMBA[part][nd],
            }
            cell = results[part][nd]
            _dump(args, results, started)
            print(f"{part} {nd}: raw={cell['raw_mesr']:.4f} (paper {cell['paper_raw']:.3f}) "
                  f"ours@{args.headline_retention}={cell['ours_mesr_by_retention'][str(args.headline_retention)]:.4f} "
                  f"corr={cell['mean_corr_GR']:+.3f} beta1_frac={cell['frac_blocks_beta1']:.2f} "
                  f"n={cell['sequences']}", flush=True)

    payload = {
        "phase": "5 E-MLB MESR",
        "metric": "MESR = mean ESR over 30,000-event slices (official cuke-emlb definition, numpy verified bit-exact)",
        "denoiser": "frozen corr-gated Native3D: beta=1 iff corr(G,R)<0, s=beta*G-R, 250us bidirectional, 10,240-event blocks",
        "caveat": "retention is a free operating point; E-MLB is unlabeled so it cannot be calibrated there. Sweep reported.",
        "settings": {"max_events_per_sequence": args.max_events, "reps": args.reps,
                     "retentions": args.retentions, "headline_retention": args.headline_retention,
                     "slice": SLICE},
        "results": results,
        "timestamp_repairs": int(read_aedat4.repairs),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
