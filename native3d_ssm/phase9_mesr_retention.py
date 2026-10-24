"""Phase 9: is dev MESR a legitimate calibration target for retention at all?

Phase 6a calibrated retention against **balanced accuracy** and landed on mean r ~0.47
on dev / ~0.785 on E-MLB. Phase 6b then showed MESR wants far more aggressive filtering,
and the reproduced EDformer picks r = 0.081-0.483 (mean 0.307) while scoring
MESR 1.0069 against our 0.8773. That looks like a straightforward objective mismatch, and
the obvious fix is to re-calibrate retention against MESR measured on *dev* data (legitimate,
because MESR is no-reference and dev != test).

**But that fix is only sound if dev MESR has an interior optimum.** Phase 5 swept
r in {0.3, 0.5, 0.7, 0.9} and reported a peak "at r ~0.3-0.5" -- yet 0.3 was the *lowest*
value tested, so the sweep cannot distinguish "interior peak near 0.3" from "rises without
bound as r -> 0". If it is the latter, MESR is trivially gamed by over-filtering, chasing it
is unscientific, and Phase 6a's balanced-accuracy calibration is the defensible choice after
all. EDformer's own paper flags exactly this: "we identified a certain limitation of MESR in
evaluating event over-denoising."

So this phase answers, on both labeled dev sets, in one pass:

1. **Shape of MESR vs retention** on a fine grid down to r = 0.02, per condition.
   Is the argmax interior, or pinned at the smallest r?
2. **What retention would each objective ask for** -- r*(MESR) vs r*(balanced accuracy)
   vs what the frozen `gated_retention` actually picks.
3. **What it costs in detection terms** to move to the MESR-preferred operating point:
   balanced accuracy, signal recall and noise rejection evaluated *at* r*(MESR).

That third item is the honesty check. If reaching MESR parity means discarding most true
signal, then MESR parity is not a goal worth having, and the right output of this phase is
that statement plus the numbers behind it -- not a re-tuned rule.

DVSCLEAN (1280x720) and DND21 (346x260) have very different pixel counts K, and MESR scales
with K, so absolute MESR is NOT comparable across the two. Only the *shape* in r and the
argmax location are compared. Read-only; nothing is fitted here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import ring_support, dnd21_blocks, dvsclean_blocks
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase8_score_upgrade import operands, safe_corr
from .retention import gated_retention

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 10_240
SLICE = 30_000
# Fine grid, deliberately reaching far below Phase 5's minimum of 0.3.
RETENTIONS = np.round(np.concatenate([np.arange(0.02, 0.30, 0.02),
                                      np.arange(0.30, 1.001, 0.05)]), 3)


def esr(x: np.ndarray, y: np.ndarray, width: int, height: int) -> float:
    """The Phase-5 numpy ESR, verified bit-exact against official cuke-emlb."""

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


def mesr(x: np.ndarray, y: np.ndarray, width: int, height: int) -> float:
    scores = [esr(x[i:i + SLICE], y[i:i + SLICE], width, height)
              for i in range(0, len(x) - SLICE + 1, SLICE)]
    return float(np.mean(scores)) if scores else float("nan")


def retain_mask(scores: np.ndarray, retention: float, block: int = BLOCK_SIZE) -> np.ndarray:
    keep = np.zeros(len(scores), dtype=bool)
    for start in range(0, len(scores), block):
        chunk = scores[start:start + block]
        if len(chunk) == 0:
            continue
        n_keep = max(1, int(round(len(chunk) * retention)))
        order = np.argsort(chunk, kind="stable")[:n_keep]
        keep[start + order] = True
    return keep


def detection_at(labels: np.ndarray, keep: np.ndarray) -> Tuple[float, float, float]:
    """Balanced accuracy, signal recall, noise rejection for a keep-mask (1 = noise)."""

    noise = labels == 1
    signal = ~noise
    if noise.sum() == 0 or signal.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    signal_recall = float(keep[signal].mean())
    noise_rejection = float((~keep[noise]).mean())
    return 0.5 * (signal_recall + noise_rejection), signal_recall, noise_rejection


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]], config: StateConfig,
                   use_both_ring: bool, max_blocks: Optional[int] = None,
                   score_config: Optional[StateConfig] = None,
                   use_support: bool = False) -> Dict:
    """Collect the whole condition's stream, then sweep retention over it.

    `score_config` supplies a wider ring for the SCORE only; the gate key keeps using
    `config` (radius 1), per Phase 11.
    """

    xs, ys, scores, labels, gated = [], [], [], [], []
    produced = 0
    for events, block_labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        raw = bidirectional_features(events, config)
        g, r_same, r_opposite = operands(raw, config)
        corr = safe_corr(g, r_same)          # Phase 8: the original key is the better one
        beta = 1.0 if corr < 0.0 else 0.0
        cfg_s = score_config if score_config is not None else config
        raw_s = raw
        if score_config is not None:         # Phase 11: wider score ring, radius-1 gate
            raw_s = bidirectional_features(events, score_config)
            _, r_same, r_opposite = operands(raw_s, score_config)
        if use_support:                      # Phase 14 operand
            ring = np.log1p(ring_support(raw_s, len(cfg_s.taus_us),
                                         cfg_s.feature_dim, cfg_s.radius))
        else:
            ring = r_same + r_opposite if use_both_ring else r_same
        block_scores = beta * g - ring
        xs.append(events[:, 1])
        ys.append(events[:, 2])
        scores.append(block_scores)
        labels.append(np.asarray(block_labels))
        gated.append(gated_retention(block_scores, corr))
    if not produced:
        return {"blocks": 0}

    x = np.concatenate(xs).astype(np.int64)
    y = np.concatenate(ys).astype(np.int64)
    score = np.concatenate(scores)
    label = np.concatenate(labels)

    raw_mesr = mesr(x, y, config.width, config.height)
    curve = []
    for retention in RETENTIONS:
        keep = retain_mask(score, float(retention))
        value = mesr(x[keep], y[keep], config.width, config.height)
        accuracy, recall, rejection = detection_at(label, keep)
        curve.append({"r": float(retention), "mesr": value, "balanced_accuracy": accuracy,
                      "signal_recall": recall, "noise_rejection": rejection,
                      "kept_events": int(keep.sum())})

    usable = [c for c in curve if np.isfinite(c["mesr"])]
    best_mesr = max(usable, key=lambda c: c["mesr"]) if usable else None
    best_accuracy = max((c for c in curve if np.isfinite(c["balanced_accuracy"])),
                        key=lambda c: c["balanced_accuracy"], default=None)
    smallest = min(usable, key=lambda c: c["r"]) if usable else None

    return {
        "blocks": produced,
        "events": int(len(score)),
        "signal_fraction": float(1.0 - label.mean()),
        "raw_mesr": raw_mesr,
        "gated_retention_mean": float(np.mean(gated)),
        "curve": curve,
        "r_star_mesr": best_mesr["r"] if best_mesr else None,
        "mesr_at_star": best_mesr["mesr"] if best_mesr else None,
        "accuracy_at_mesr_star": best_mesr["balanced_accuracy"] if best_mesr else None,
        "signal_recall_at_mesr_star": best_mesr["signal_recall"] if best_mesr else None,
        "r_star_accuracy": best_accuracy["r"] if best_accuracy else None,
        "accuracy_at_star": best_accuracy["balanced_accuracy"] if best_accuracy else None,
        # The decisive diagnostic. A short condition cannot be measured at aggressive
        # retention: below ~30,000 surviving events there is no complete MESR slice and
        # the value is NaN. So record the smallest *evaluable* r explicitly -- "pinned at
        # the floor" is only meaningful when that floor is genuinely small.
        "smallest_evaluable_r": smallest["r"] if smallest else None,
        "valid_grid_points": len(usable),
        "grid_points": len(curve),
        "mesr_argmax_at_evaluable_floor": bool(best_mesr and smallest
                                               and best_mesr["r"] == smallest["r"]),
        "mesr_at_evaluable_floor": smallest["mesr"] if smallest else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--both-ring", action="store_true",
                        help="Use the Phase 8 score s = beta*G - (R_same+R_opp).")
    parser.add_argument("--support", action="store_true",
                        help="Phase 14 operand: s = beta*G - log1p(ring_support).")
    parser.add_argument("--score-radius", type=int, default=1,
                        help="Ring radius for the SCORE; gate key stays at radius 1 (Phase 11).")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=["DVSCLEAN", "DND21"])
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase9_mesr_retention.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    results: Dict[str, Dict[str, Dict]] = {"DVSCLEAN": {}, "DND21": {}}

    if "DVSCLEAN" in args.datasets:
        config = StateConfig(width=1280, height=720, taus_us=(250.0,))
        score_config = (StateConfig(width=1280, height=720, taus_us=(250.0,),
                                    radius=args.score_radius)
                        if args.score_radius != 1 else None)
        for seq in SEQUENCES:
            for level in (50, 100):
                name = f"{seq}_{level}"
                results["DVSCLEAN"][name] = scan_condition(
                    dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE, args.max_blocks),
                    config, args.both_ring, args.max_blocks, score_config, args.support)
                c = results["DVSCLEAN"][name]
                print(f"DVSCLEAN {name}: r*(MESR)={c['r_star_mesr']} MESR={c['mesr_at_star']:.4f} "
                      f"(raw {c['raw_mesr']:.4f}) | r*(acc)={c['r_star_accuracy']} "
                      f"| acc@r*MESR={c['accuracy_at_mesr_star']:.4f} "
                      f"recall={c['signal_recall_at_mesr_star']:.4f} "
                      f"| eval_floor={c['smallest_evaluable_r']} pinned={c['mesr_argmax_at_evaluable_floor']}", flush=True)

    if "DND21" in args.datasets:
        config = StateConfig(width=346, height=260, taus_us=(250.0,))
        score_config = (StateConfig(width=346, height=260, taus_us=(250.0,),
                                    radius=args.score_radius)
                        if args.score_radius != 1 else None)
        for rate in DND21_PAPER:
            for scene, fname in SCENE_FILES.items():
                name = f"{rate}_{scene}"
                results["DND21"][name] = scan_condition(
                    dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                    config, args.both_ring, args.max_blocks, score_config, args.support)
                c = results["DND21"][name]
                print(f"DND21 {name}: r*(MESR)={c['r_star_mesr']} MESR={c['mesr_at_star']:.4f} "
                      f"(raw {c['raw_mesr']:.4f}) | r*(acc)={c['r_star_accuracy']} "
                      f"| acc@r*MESR={c['accuracy_at_mesr_star']:.4f} "
                      f"recall={c['signal_recall_at_mesr_star']:.4f} "
                      f"| eval_floor={c['smallest_evaluable_r']} pinned={c['mesr_argmax_at_evaluable_floor']}", flush=True)

    everything = {f"{ds}/{k}": v for ds, conditions in results.items()
                  for k, v in conditions.items() if v.get("blocks")}
    pinned = [n for n, c in everything.items() if c["mesr_argmax_at_evaluable_floor"]]
    # Only conditions long enough to be measured at genuinely aggressive retention can
    # answer the monotonicity question at all.
    decisive = {n: c for n, c in everything.items()
                if c["smallest_evaluable_r"] is not None and c["smallest_evaluable_r"] <= 0.10}
    verdict = {
        "conditions": len(everything),
        "conditions_measurable_below_r0.10": len(decisive),
        "mesr_argmax_pinned_at_evaluable_floor": len(pinned),
        "pinned_conditions": pinned,
        "pinned_among_decisive": sorted(n for n in pinned if n in decisive),
        "grid_floor": float(RETENTIONS.min()),
        "mean_smallest_evaluable_r": float(np.mean(
            [c["smallest_evaluable_r"] for c in everything.values()
             if c["smallest_evaluable_r"] is not None])),
        "mean_r_star_mesr": float(np.mean([c["r_star_mesr"] for c in everything.values()])),
        "mean_r_star_accuracy": float(np.mean([c["r_star_accuracy"] for c in everything.values()])),
        "mean_gated_retention": float(np.mean([c["gated_retention_mean"] for c in everything.values()])),
        "mean_accuracy_at_mesr_star": float(np.mean([c["accuracy_at_mesr_star"] for c in everything.values()])),
        "mean_accuracy_at_accuracy_star": float(np.mean([c["accuracy_at_star"] for c in everything.values()])),
        "mean_signal_recall_at_mesr_star": float(np.mean([c["signal_recall_at_mesr_star"] for c in everything.values()])),
    }
    verdict["accuracy_cost_of_chasing_mesr"] = (verdict["mean_accuracy_at_accuracy_star"]
                                                - verdict["mean_accuracy_at_mesr_star"])
    # Judge gameability only on conditions that could actually be measured low.
    verdict["mesr_is_gameable_by_overfiltering"] = bool(
        decisive and len(verdict["pinned_among_decisive"]) > 0.5 * len(decisive))
    verdict["per_dataset_mean_r_star_mesr"] = {
        ds: float(np.mean([c["r_star_mesr"] for c in conditions.values() if c.get("blocks")]))
        for ds, conditions in results.items() if any(c.get("blocks") for c in conditions.values())}

    payload = {
        "phase": "9 is dev MESR a legitimate retention target",
        # The local operand is whichever of the three --support / --both-ring select.
        # This string reported only the --both-ring branch until 2026-08-13, so the
        # archived phase9_mesr_retention_support.json carries a stale label: it was run
        # with --support --score-radius 2 and holds the adopted score, not R_same.
        "score": ("s = beta*G - log1p(ring_support)" if args.support
                  else "s = beta*G - (R_same + R_opposite)" if args.both_ring
                  else "s = beta*G - R_same"),
        "score_radius": args.score_radius,
        "gate_key": "corr(G, R_same), theta = 0 (Phase 8 showed the original key is better)",
        "retention_grid": RETENTIONS.tolist(),
        "caveat": ("MESR scales with pixel count K, so DVSCLEAN (1280x720) and DND21 (346x260) "
                   "absolute values are not comparable; only the shape in r and the argmax are."),
        "results": results,
        "verdict": verdict,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("\nVERDICT:", json.dumps(verdict, indent=2))
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
