"""Phase 15: choosing the Track-B operating point without letting MESR pick it.

**Why the previous rule broke.** Phase 9 selected `r` as the argmax of mean dev-MESR gain,
and validated that choice by showing 0/20 dev conditions peaked at the evaluable retention
floor -- i.e. MESR was not "gameable by over-filtering". With the Phase 14 support operand
that check degrades: **5/20 conditions become monotone down to the floor** (all DVSCLEAN),
and every one of them is *decreasing in r from its first evaluable point* (slopes -0.09 to
-0.38 MESR per unit r), so the optimum is at or below the floor rather than being a shallow
interior peak we merely failed to resolve.

The mechanism is structural, not a bug. `ring_support` counts spatially clustered
neighbours; MESR's `ntss` term is a second moment of per-pixel counts, so it rewards spatial
concentration. Support and MESR therefore optimise nearly the same thing, and maximising dev
MESR drives `r` toward zero. The better the operand gets at finding clustered events, the
more MESR rewards discarding everything else.

Note the strict flag in Phase 9 still reads False -- it fires only when pinned conditions are
a *majority* of the decisive ones (here 3 of 15). The criterion is not formally overturned;
what collapses is the margin, and with it the safety of using the argmax.

**The fix: stop using the MESR argmax.** Select `r` by jointly maximising

    normalised dev-MESR gain   (gain / per-condition-oracle gain, in [0,1])
    dev balanced accuracy      (label-based, in [0,1])

Both are measured on **dev only**, so E-MLB is never touched and the choice stays
non-circular. Balanced accuracy is *not* the reported Track-B metric, which is precisely why
it is a safe second objective: it cannot be gamed in MESR's direction.

Two scalarisations are reported -- product and harmonic mean. They are a design choice, not
forced; they are reported together because when they agree the choice of scalarisation is
not doing the work. On the support score they both select the same `r`.

Read-only over the Phase 9 artifacts. Nothing is trained; no threshold is tuned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AC = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"

SOURCES = {
    "old_score_r1": AC / "phase9_mesr_retention.json",
    "old_score_r2": AC / "phase9_mesr_retention_radius2.json",
    "support_r2": AC / "phase9_mesr_retention_support.json",
}


def load_curves(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict]:
    payload = json.load(open(path))
    grid = np.asarray(payload["retention_grid"], dtype=float)
    gains, accuracies, recalls = [], [], []
    for dataset in ("DVSCLEAN", "DND21"):
        for condition in payload["results"].get(dataset, {}).values():
            if not condition.get("blocks"):
                continue
            raw = condition["raw_mesr"]
            points = {p["r"]: p for p in condition["curve"]}

            def at(field: str, default=np.nan):
                return [points[r][field] if r in points and points[r][field] is not None
                        else default for r in grid]

            mesr = np.asarray(at("mesr"), dtype=float)
            gains.append(np.where(np.isfinite(mesr), mesr - raw, np.nan))
            accuracies.append(np.asarray(at("balanced_accuracy"), dtype=float))
            recalls.append(np.asarray(at("signal_recall"), dtype=float))
    gain_matrix = np.asarray(gains, dtype=float)
    oracle = float(np.nanmean(np.nanmax(gain_matrix, axis=1)))
    return (grid, np.nanmean(gain_matrix, axis=0),
            np.nanmean(np.asarray(accuracies, dtype=float), axis=0),
            np.nanmean(np.asarray(recalls, dtype=float), axis=0),
            oracle, payload.get("verdict", {}))


def select(path: Path) -> Dict:
    grid, gain, accuracy, recall, oracle, verdict = load_curves(path)
    normalised = gain / oracle
    product = normalised * accuracy
    harmonic = 2 * normalised * accuracy / (normalised + accuracy)

    def row(index: int) -> Dict:
        return {"r": float(grid[index]),
                "dev_mesr_gain": float(gain[index]),
                "pct_of_oracle": float(100 * gain[index] / oracle),
                "balanced_accuracy": float(accuracy[index]),
                "signal_recall": float(recall[index])}

    chosen_product = int(np.nanargmax(product))
    chosen_harmonic = int(np.nanargmax(harmonic))
    return {
        "oracle_gain": oracle,
        "rules": {
            "argmax_dev_mesr_DEGENERATE": row(int(np.nanargmax(gain))),
            "argmax_normalised_mesr_x_balanced_accuracy": row(chosen_product),
            "argmax_harmonic_mean": row(chosen_harmonic),
            "argmax_balanced_accuracy": row(int(np.nanargmax(accuracy))),
        },
        "scalarisations_agree": bool(grid[chosen_product] == grid[chosen_harmonic]),
        "selected_r": float(grid[chosen_product]),
        "pinned_at_evaluable_floor": verdict.get("mesr_argmax_pinned_at_evaluable_floor"),
        "pinned_conditions": verdict.get("pinned_conditions"),
        "strict_gameability_flag": verdict.get("mesr_is_gameable_by_overfiltering"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=AC / "phase15_trackb_operating_point.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {name: select(path) for name, path in SOURCES.items() if path.exists()}

    payload = {
        "phase": "15 Track-B operating point without a MESR argmax",
        "why": ("With the Phase 14 support operand the dev-MESR argmax is degenerate: 5/20 "
                "conditions are monotone to the evaluable floor because support and MESR both "
                "reward spatial concentration. Selecting on MESR alone drives r toward zero."),
        "rule": ("maximise (dev-MESR gain / oracle) x (dev balanced accuracy); both measured "
                 "on dev only, so E-MLB is untouched. Balanced accuracy is not the reported "
                 "Track-B metric, so it cannot be gamed in MESR's direction."),
        "caveat": ("The scalarisation is a design choice. Product and harmonic mean are both "
                   "reported; they agree on the support score, so the choice is not load-bearing "
                   "there."),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for name, res in results.items():
        print(f"\n=== {name}   (oracle {res['oracle_gain']:+.4f}, "
              f"pinned {res['pinned_at_evaluable_floor']}, "
              f"strict-gameable {res['strict_gameability_flag']})")
        for rule, row in res["rules"].items():
            print(f"  {rule:<44} r={row['r']:<5} MESR {row['dev_mesr_gain']:+.4f} "
                  f"({row['pct_of_oracle']:3.0f}%)  acc {row['balanced_accuracy']:.4f}  "
                  f"recall {row['signal_recall']:.4f}")
        print(f"  -> SELECTED r = {res['selected_r']}   "
              f"(scalarisations agree: {res['scalarisations_agree']})")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
