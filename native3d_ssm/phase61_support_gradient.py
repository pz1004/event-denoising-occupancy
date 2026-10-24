"""Phase 61: the occupancy account of the LED gap, tested at the scene level.

THE HONEST ORDER OF EVENTS, BECAUSE IT MATTERS FOR HOW THIS IS READ.

Phase 59 found EBF ahead of us on all 66 LED scenes by +0.1498 block-mean AUC. Phase 60 was
written to test one explanation that had been committed to print before LED was ever scored:
Supplementary S5 derives the operand from a Binomial occupancy model and names a failure
regime at a nearly empty ring. Phase 60 fixed its decision rule *before* running -- LED counts
as being in the regime if its mean support is below half of development's and more than 30% of
its events have zero support -- and LED **failed** that rule. Mean support is 2.54 of 24
against development's 4.53, a factor of 1.8, not the factor of 2 the rule demanded. At the
benchmark level, the occupancy model does not explain the gap, and Phase 60 says so.

That rule was the wrong test, and this phase says why rather than quietly replacing it. The
model does not predict that one benchmark sits in a bad regime and another does not. It
predicts a *gradient*: separation should fall as the ring empties, wherever that happens.
Comparing two pooled means cannot see a gradient. The unit that can is the scene, and LED
supplies 66 of them spanning a wide range of occupancy.

So this phase is post-hoc with respect to Phase 60's rule and is reported as such. What is
*not* post-hoc is the quantity, the direction, or the mechanism: all three were fixed in S5
before any LED number existed. Only the unit of analysis changed, and it changed because the
pooled comparison was the wrong instrument, not because it gave the wrong answer.

WHAT IS COMPUTED. Per-scene mean ring support against per-scene block-mean AUC, for our arms
and for EBF, over LED's 66 scenes and the 20 development conditions. Then the one comparison
that carries the claim without needing a correlation at all: the LED scenes whose occupancy
reaches development levels, and what our filter scores on them.

Read-only, and recomputed entirely from two stored artefacts. Nothing is re-scanned, so this
phase cannot disagree with the numbers it draws on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_corr(a: np.ndarray, b: np.ndarray, replicates: int,
                   rng: np.random.Generator) -> Dict[str, float]:
    """Percentile interval on a correlation by resampling the units themselves."""

    n = a.size
    draws = rng.integers(0, n, size=(replicates, n))
    values = np.array([corr(a[d], b[d]) for d in draws])
    values = values[np.isfinite(values)]
    return {"point": corr(a, b),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "replicates": int(values.size)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase61_support_gradient.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    occ = json.loads((ARTIFACTS / "phase60_occupancy_regime.json").read_text())
    led = json.loads((ARTIFACTS / "phase59_ebf_led.json").read_text())["per_scene"]
    dev = json.loads((ARTIFACTS / "phase57_ebf_labelled.json").read_text())["per_condition"]

    scenes = sorted(s for s in led if s in occ["led"])
    support = np.array([occ["led"][s]["mean_support"] for s in scenes])
    zero = np.array([occ["led"][s]["fraction_support_zero"] for s in scenes])
    ours = np.array([led[s]["support"]["block_mean_auc"] for s in scenes])
    beta0 = np.array([led[s]["beta0_support"]["block_mean_auc"] for s in scenes])
    ebf = np.array([led[s]["ebf"]["block_mean_auc"] for s in scenes])

    conditions = sorted(k for k in dev if k in occ["development"])
    d_support = np.array([occ["development"][k]["mean_support"] for k in conditions])
    d_ours = np.array([dev[k]["ours_block_mean_auc"] for k in conditions])
    d_ebf = np.array([dev[k]["block_mean_auc"] for k in conditions])

    #: The threshold is development's own mean occupancy, not a value chosen here.
    threshold = float(occ["summary"]["development"]["mean_support"])
    at_dev_level = support >= threshold

    payload = {
        "phase": "61 the support gradient behind the LED gap",
        "provenance": {
            "inputs": ["phase60_occupancy_regime.json", "phase59_ebf_led.json",
                       "phase57_ebf_labelled.json"],
            "recomputed": "nothing; every value is read from a stored artefact",
            "status": ("POST-HOC with respect to Phase 60's pre-registered pooled rule, "
                       "which LED failed. The quantity, direction and mechanism were "
                       "fixed in Supplementary S5 before LED was scored; only the unit "
                       "of analysis changed, from the benchmark to the scene."),
            "phase60_verdict": occ["verdict"],
        },
        "led": {
            "scenes": len(scenes),
            "mean_support_range": [float(support.min()), float(support.max())],
            "corr_support_vs_ours_gated": bootstrap_corr(support, ours, args.replicates, rng),
            "corr_support_vs_ours_beta0": bootstrap_corr(support, beta0, args.replicates, rng),
            "corr_support_vs_ebf": bootstrap_corr(support, ebf, args.replicates, rng),
            "corr_support_vs_gap": bootstrap_corr(support, ebf - ours, args.replicates, rng),
            "corr_zero_fraction_vs_ours_gated": corr(zero, ours),
            "corr_zero_fraction_vs_gap": corr(zero, ebf - ours),
        },
        "development": {
            "conditions": len(conditions),
            "mean_support_range": [float(d_support.min()), float(d_support.max())],
            "corr_support_vs_ours": corr(d_support, d_ours),
            "corr_support_vs_ebf": corr(d_support, d_ebf),
            # The account is only worth much if it also predicts where EBF wins on the
            # development sets, which were scored long before any of this was written.
            "corr_support_vs_gap": corr(d_support, d_ebf - d_ours),
            "ebf_wins": int((d_ebf > d_ours).sum()),
            "ebf_losses": int((d_ebf < d_ours).sum()),
            "mean_support_where_ebf_wins": float(d_support[d_ebf > d_ours].mean()),
            "mean_support_where_ebf_loses": float(d_support[d_ebf < d_ours].mean()),
            "max_support_where_ebf_wins": float(d_support[d_ebf > d_ours].max()),
            "conditions_ebf_wins": sorted(
                c for c, w in zip(conditions, d_ebf > d_ours) if w),
            "note": ("On development the two methods move opposite ways in occupancy: "
                     "ours improves as the ring fills, EBF degrades, which is what a "
                     "window fixed in time does when the event rate rises."),
        },
        "pooled_86_units": {
            "corr_support_vs_ours": corr(np.r_[support, d_support], np.r_[ours, d_ours]),
            "corr_support_vs_ebf": corr(np.r_[support, d_support], np.r_[ebf, d_ebf]),
        },
        "led_scenes_at_development_occupancy": {
            "threshold_mean_support": threshold,
            "threshold_source": "development's own mean occupancy, not chosen here",
            "n_at_or_above": int(at_dev_level.sum()),
            "n_below": int((~at_dev_level).sum()),
            "ours_at_or_above": float(ours[at_dev_level].mean()),
            "ours_below": float(ours[~at_dev_level].mean()),
            "ebf_at_or_above": float(ebf[at_dev_level].mean()),
            "ebf_below": float(ebf[~at_dev_level].mean()),
            "gap_at_or_above": float((ebf - ours)[at_dev_level].mean()),
            "gap_below": float((ebf - ours)[~at_dev_level].mean()),
        },
        "block_span": {
            "led_median_ms": occ["summary"]["led"]["block_span_ms_median"],
            "development_median_ms": occ["summary"]["development"]["block_span_ms_median"],
            "ebf_window_ms": occ["ebf_comparison"]["ebf_window_ms"],
            "note": ("Our block is fixed in events, so its temporal extent shrinks as the "
                     "event rate rises; EBF's window is fixed in time and does not."),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    l = payload["led"]
    print(f"LED, {len(scenes)} scenes, mean support {support.min():.2f}-{support.max():.2f} of 24")
    for label, key in (("ours (gated)", "corr_support_vs_ours_gated"),
                       ("ours (beta=0)", "corr_support_vs_ours_beta0"),
                       ("EBF", "corr_support_vs_ebf"),
                       ("EBF - ours gap", "corr_support_vs_gap")):
        c = l[key]
        print(f"  corr(mean support, {label:15s}) = {c['point']:+.3f} "
              f"[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]")
    a = payload["led_scenes_at_development_occupancy"]
    print(f"\nLED scenes at or above development's mean occupancy ({threshold:.2f} of 24):")
    print(f"  n={a['n_at_or_above']:2d}  ours {a['ours_at_or_above']:.4f}  "
          f"EBF {a['ebf_at_or_above']:.4f}  gap {a['gap_at_or_above']:+.4f}")
    print(f"  n={a['n_below']:2d}  ours {a['ours_below']:.4f}  "
          f"EBF {a['ebf_below']:.4f}  gap {a['gap_below']:+.4f}   (below)")
    d = payload["development"]
    print(f"\ndevelopment: corr(support, ours) = {d['corr_support_vs_ours']:+.3f}, "
          f"corr(support, EBF) = {d['corr_support_vs_ebf']:+.3f}, "
          f"corr(support, gap) = {d['corr_support_vs_gap']:+.3f}")
    print(f"  EBF wins {d['ebf_wins']} of {d['conditions']} conditions, mean support "
          f"{d['mean_support_where_ebf_wins']:.2f} against "
          f"{d['mean_support_where_ebf_loses']:.2f} where it loses; every win is at "
          f"support <= {d['max_support_where_ebf_wins']:.2f}")
    b = payload["block_span"]
    print(f"block span: LED {b['led_median_ms']:.2f} ms, development "
          f"{b['development_median_ms']:.2f} ms, EBF's window {b['ebf_window_ms']:.0f} ms")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
