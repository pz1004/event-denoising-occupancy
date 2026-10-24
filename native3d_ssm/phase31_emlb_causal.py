"""Phase 31: E-MLB MESR under strict per-event causality.

The manuscript prices the block-local bidirectional pass on the *labelled* track only
(Section 4.6: pooled AUC 0.9811 -> 0.9097). E-MLB, the one held-out benchmark and the
one carrying the parity claim against EDformer, was never re-scored without it. So the
headline "parity at >=102x less arithmetic" was stated for a configuration that takes a
lookahead, and nothing on disk said whether parity survives removing it.

This phase runs exactly that measurement, in one pass so the two rows are exactly paired:

  raw                 unfiltered anchor -- must reproduce phase10's 0.8147
  bidirectional       the adopted headline score at frozen r = 0.28 -- must reproduce
                      phase10_emlb_blind_support.json's 1.0035 (regression anchor)
  fully causal        forward-only operands + cumulative streaming gate, same r = 0.28

"Fully causal" is the same construction `phase5b_causal_corr.py` reports as
`sup_fully_causal_cumulative_forward`, the row behind the manuscript's 0.9097:

    forward_corr = cumulative_corr(G_fwd, R_same_fwd)      # radius-1 gate key
    beta_i       = 1 iff forward_corr_i < 0, cold start 0 for the first 1024 events
    s_i          = beta_i * G_fwd_i - log1p(ring_support_fwd_i)   # radius-2 score ring

Two differences from the labelled track, both forced by the benchmark rather than chosen:
the correlation state persists across blocks *within a recording* and resets at each
recording (E-MLB sequences are independent recordings, not one stream), and there are no
labels, so the metric is MESR rather than AUC.

Nothing here is tuned on E-MLB. r = 0.28 is the frozen dev-calibrated retention of
Phase 9; the warmup and cold-start branch are Phase 5b's defaults, unchanged.

Read-only over data on disk. No model is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .causal_corr import DEFAULT_WARMUP, causal_beta, cumulative_corr
from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import ring_support
from .phase5_emlb_mesr import (
    ND_LEVELS,
    PAPER_RAW,
    PARTS,
    SLICE,
    mesr,
    read_aedat4,
    retain_mask,
)
from .phase8_score_upgrade import operands, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK = 10_240
DEV_CALIBRATED_RETENTION = 0.28      # Phase 9, dev MESR only. Never fitted on E-MLB.

#: Anchors this run must reproduce, from phase10_emlb_blind_support.json. If either
#: drifts, the pipeline changed underneath and the causal row is not comparable to the
#: manuscript's numbers.
ANCHORS = {"raw_mean": 0.8147, "bidirectional_mean": 1.0035}


def score_stream_causal(events: np.ndarray, config: StateConfig,
                        block: int = BLOCK,
                        score_config: Optional[StateConfig] = None,
                        warmup: int = DEFAULT_WARMUP):
    """Bidirectional (adopted) and fully causal support scores on one event stream.

    Both rows come out of the same feature extraction so the pair is exact: the
    bidirectional operands are the forward and reverse halves summed, the causal ones
    are the forward half alone. `score_config` (radius 2) supplies the score ring; the
    gate key stays on `config` (radius 1), per Phase 11.
    """

    ordered = np.ascontiguousarray(
        np.column_stack((events["timestamp"].astype(np.int64),
                         events["x"].astype(np.int64),
                         events["y"].astype(np.int64),
                         (events["polarity"].astype(np.int64) > 0).astype(np.int64))),
        dtype=np.int64)
    n = len(ordered)
    scale_count = len(config.taus_us)

    sup_bidir = np.zeros(n, dtype=np.float64)
    g_fwd = np.zeros(n, dtype=np.float64)
    r_fwd = np.zeros(n, dtype=np.float64)
    log_sup_fwd = np.zeros(n, dtype=np.float64)
    corrs: List[float] = []
    betas: List[float] = []

    for start in range(0, n, block):
        chunk = ordered[start:start + block]
        if len(chunk) < 2:
            continue
        stop = start + len(chunk)
        raw = bidirectional_features(chunk, config)
        g, r_same, _r_opposite = operands(raw, config)

        # --- adopted row: per-block batch gate on bidirectional operands ---------
        corr = safe_corr(g, r_same)
        beta = 1.0 if corr < 0.0 else 0.0
        cfg_s = score_config if score_config is not None else config
        raw_s = bidirectional_features(chunk, cfg_s) if score_config is not None else raw
        support = ring_support(raw_s, scale_count, cfg_s.feature_dim, cfg_s.radius)
        sup_bidir[start:stop] = beta * g - np.log1p(support)
        corrs.append(float(corr))
        betas.append(beta)

        # --- causal row: forward halves only, no reverse pass consulted ----------
        g_fwd[start:stop] = raw[:, 6 * scale_count]
        r_fwd[start:stop] = raw[:, 2 * scale_count]
        log_sup_fwd[start:stop] = np.log1p(
            ring_support(raw_s, scale_count, cfg_s.feature_dim, cfg_s.radius,
                         forward_only=True))

    # Streaming gate over the whole recording: beta_i depends on events <= i only.
    forward_corr = cumulative_corr(g_fwd, r_fwd)
    beta_causal = causal_beta(forward_corr, warmup)
    sup_causal = beta_causal * g_fwd - log_sup_fwd
    return sup_bidir, sup_causal, corrs, betas, beta_causal


def exact_randomisation_p(deltas: np.ndarray) -> float:
    """Two-sided exact paired sign-flip test, enumerating all 2^n assignments."""

    deltas = np.asarray(deltas, dtype=np.float64)
    n = deltas.size
    observed = abs(float(np.mean(deltas)))
    signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    means = np.abs((signs * deltas).mean(axis=1))
    return float((means >= observed - 1e-15).mean())


def bootstrap_ci(deltas: np.ndarray, replicates: int = 10_000,
                 seed: int = 20260813) -> List[float]:
    """Percentile bootstrap over the paired per-cell differences."""

    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas, dtype=np.float64)
    draws = rng.choice(deltas, size=(replicates, deltas.size), replace=True)
    return [float(np.percentile(draws.mean(axis=1), 2.5)),
            float(np.percentile(draws.mean(axis=1), 97.5))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--retention", type=float, default=DEV_CALIBRATED_RETENTION,
                        help="Frozen retention from Phase 9. Do not tune on E-MLB.")
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--max-events", type=int, default=0, help="0 = uncapped.")
    parser.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--nd-levels", nargs="+", default=list(ND_LEVELS))
    parser.add_argument("--resummarise", action="store_true",
                        help="Rebuild the summary from an existing artifact; no rescoring.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase31_emlb_causal.json")
    return parser.parse_args()


def edformer_margins(results: Dict) -> Dict:
    """Paired per-cell margins against the local EDformer reproduction.

    Pairing convention is phase21's: a per-cell difference of raw MESR values, not of
    improvements over Raw. The bidirectional row is recomputed here as a control -- it
    must reproduce phase21's `emlb/frozen_r0.28/observed_mean_margin` of -0.003386, and
    if it does not, the causal margin beside it is not comparable to the manuscript.
    """

    reference = json.loads(
        (PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
         / "phase18_retention_matched.json").read_text())["edformer_reference"]
    cells, bidir, causal = [], [], []
    for part in sorted(results):
        for nd in sorted(results[part]):
            cell = results[part][nd]
            if cell.get("raw_mesr") is None or part not in reference:
                continue
            cells.append(f"{part}/{nd}")
            bidir.append(cell["bidirectional_mesr"] - reference[part][nd]["edformer_mesr"])
            causal.append(cell["causal_mesr"] - reference[part][nd]["edformer_mesr"])
    if not cells:
        return {"cells": 0}
    out = {}
    for name, margins in (("bidirectional", np.asarray(bidir)),
                          ("causal", np.asarray(causal))):
        out[name] = {
            "mean": float(margins.mean()),
            "per_cell": {c: float(m) for c, m in zip(cells, margins)},
            "cells_ahead": int((margins > 0).sum()),
            "ci95": bootstrap_ci(margins),
            "exact_randomisation_p": exact_randomisation_p(margins),
        }
    out["control"] = {
        "phase21_frozen_margin_expected": -0.003386298726542794,
        "bidirectional_abs_error": abs(out["bidirectional"]["mean"] + 0.003386298726542794),
    }
    return out


def summarise(results: Dict) -> Dict:
    """Cell-level aggregation and the paired causal-vs-bidirectional test."""

    cells, raw, bidir, causal = [], [], [], []
    for part in sorted(results):
        for nd in sorted(results[part]):
            cell = results[part][nd]
            if cell.get("raw_mesr") is None:
                continue
            cells.append(f"{part}/{nd}")
            raw.append(cell["raw_mesr"])
            bidir.append(cell["bidirectional_mesr"])
            causal.append(cell["causal_mesr"])
    if not cells:
        return {"cells": 0}
    raw = np.asarray(raw); bidir = np.asarray(bidir); causal = np.asarray(causal)
    delta = causal - bidir
    summary = {
        "cells": len(cells),
        "cell_order": cells,
        "raw_mean": float(raw.mean()),
        "bidirectional_mean": float(bidir.mean()),
        "causal_mean": float(causal.mean()),
        "bidirectional_delta_raw": float((bidir - raw).mean()),
        "causal_delta_raw": float((causal - raw).mean()),
        "causal_minus_bidirectional": {
            "mean": float(delta.mean()),
            "per_cell": {c: float(d) for c, d in zip(cells, delta)},
            "cells_causal_ahead": int((delta > 0).sum()),
            "ci95": bootstrap_ci(delta),
            "exact_randomisation_p": exact_randomisation_p(delta),
        },
        "vs_edformer_reproduction": edformer_margins(results),
        "anchor_check": {
            "raw_mean_expected": ANCHORS["raw_mean"],
            "raw_mean_abs_error": abs(float(raw.mean()) - ANCHORS["raw_mean"]),
            "bidirectional_mean_expected": ANCHORS["bidirectional_mean"],
            "bidirectional_mean_abs_error": abs(float(bidir.mean())
                                                - ANCHORS["bidirectional_mean"]),
        },
    }
    return summary


def build_payload(args, results, started) -> Dict:
    return {
        "phase": "31 E-MLB MESR under strict per-event causality",
        "question": ("does the E-MLB parity claim survive removing the block-local "
                     "bidirectional pass, which Section 4.6 prices at 7.15 AUC points "
                     "on the labelled track but had never been priced here"),
        "rows": {
            "raw": "unfiltered anchor",
            "bidirectional": ("adopted headline: per-block batch gate on bidirectional "
                              "operands, s = beta*G - log1p(ring_support)"),
            "causal": ("forward-only operands + cumulative streaming gate, the E-MLB "
                       "analogue of phase5b's sup_fully_causal_cumulative_forward"),
        },
        "gate_causal": (f"beta_i = 1 iff cumulative_corr(G_fwd, R_same_fwd)_i < 0, "
                        f"warmup {args.warmup} events held at beta = 0; state persists "
                        "across blocks within a recording and resets per recording"),
        "retention": (f"frozen r = {args.retention}, Phase 9 dev calibration; "
                      "not fitted or inspected on E-MLB, and unchanged for the causal row"),
        "metric": "MESR = mean ESR over 30,000-event slices (numpy ESR, bit-exact vs cuke-emlb)",
        "settings": {"max_events_per_sequence": args.max_events, "reps": args.reps,
                     "slice": SLICE, "block": BLOCK, "warmup": args.warmup,
                     "gate_radius": 1, "score_radius": args.score_radius},
        "anchors": ANCHORS,
        "summary": summarise(results),
        "results": results,
        "timestamp_repairs": int(read_aedat4.repairs),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.resummarise:
        existing = json.loads(args.output.read_text())
        payload = build_payload(args, existing["results"], started)
        payload["wall_seconds"] = existing["wall_seconds"]
        payload["resummarised"] = True
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        s = payload["summary"]
        m = s["vs_edformer_reproduction"]
        print(f"raw {s['raw_mean']:.4f} | bidirectional {s['bidirectional_mean']:.4f} "
              f"| causal {s['causal_mean']:.4f}")
        print(f"dRaw: bidirectional {s['bidirectional_delta_raw']:+.4f} | "
              f"causal {s['causal_delta_raw']:+.4f}")
        for row in ("bidirectional", "causal"):
            r = m[row]
            print(f"vs EDformer ({row:>13}): {r['mean']:+.4f} {[round(v,4) for v in r['ci95']]} "
                  f"p={r['exact_randomisation_p']:.3f} ahead {r['cells_ahead']}/8")
        print(f"control |bidir - phase21| = {m['control']['bidirectional_abs_error']:.2e}")
        return
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
            raw_list, bidir_list, causal_list = [], [], []
            corr_all, beta_all, beta_causal_all = [], [], []
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

                    sup_bidir, sup_causal, corrs, betas, beta_causal = score_stream_causal(
                        events, config, BLOCK, score_config, args.warmup)
                    corr_all.extend(corrs)
                    beta_all.extend(betas)
                    beta_causal_all.append(float(np.mean(beta_causal)))

                    keep_b = retain_mask(sup_bidir, args.retention, BLOCK)
                    bidir_list.append(mesr(x[keep_b], y[keep_b], config.width, config.height))
                    keep_c = retain_mask(sup_causal, args.retention, BLOCK)
                    causal_list.append(mesr(x[keep_c], y[keep_c], config.width, config.height))

                    per_sequence[f"{scene}-{nd}-{rep}"] = {
                        "raw_mesr": float(raw_list[-1]),
                        "bidirectional_mesr": float(bidir_list[-1]),
                        "causal_mesr": float(causal_list[-1]),
                    }
            cell = {
                "sequences": len(raw_list),
                "per_sequence": per_sequence,
                "raw_mesr": float(np.nanmean(raw_list)) if raw_list else None,
                "bidirectional_mesr": float(np.nanmean(bidir_list)) if bidir_list else None,
                "causal_mesr": float(np.nanmean(causal_list)) if causal_list else None,
                "frac_blocks_beta1_batch": float(np.mean(beta_all)) if beta_all else None,
                "frac_events_beta1_causal": (float(np.mean(beta_causal_all))
                                             if beta_causal_all else None),
                "mean_corr_GR": float(np.mean(corr_all)) if corr_all else None,
                "paper_raw": PAPER_RAW[part][nd],
            }
            results[part][nd] = cell
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(build_payload(args, results, started),
                                              indent=2, sort_keys=True) + "\n")
            print(f"{part} {nd}: raw={cell['raw_mesr']:.4f} | "
                  f"bidir={cell['bidirectional_mesr']:.4f} | "
                  f"causal={cell['causal_mesr']:.4f} | "
                  f"n={cell['sequences']} | "
                  f"beta1_causal={cell['frac_events_beta1_causal']:.4f}",
                  flush=True)

    payload = build_payload(args, results, started)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    s = payload["summary"]
    print(f"\nraw {s['raw_mean']:.4f} | bidirectional {s['bidirectional_mean']:.4f} "
          f"| causal {s['causal_mean']:.4f}")
    print(f"delta Raw: bidirectional {s['bidirectional_delta_raw']:+.4f} | "
          f"causal {s['causal_delta_raw']:+.4f}")
    cb = s["causal_minus_bidirectional"]
    print(f"causal - bidirectional: {cb['mean']:+.4f} {cb['ci95']} "
          f"p={cb['exact_randomisation_p']:.3f} ahead on {cb['cells_causal_ahead']}/8")
    print(f"anchors: raw err {s['anchor_check']['raw_mean_abs_error']:.4f} | "
          f"bidir err {s['anchor_check']['bidirectional_mean_abs_error']:.4f}")
    print(f"wall {payload['wall_seconds']/60:.1f} min")


if __name__ == "__main__":
    main()
