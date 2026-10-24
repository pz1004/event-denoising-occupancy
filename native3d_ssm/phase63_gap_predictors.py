"""Phase 63: what predicts the gap to EBF, once a fifth benchmark is in evidence.

WHY THIS EXISTS. Section IV-E attributes our deficit to EBF to ring occupancy, which our
fixed-event-count block sets, and an earlier draft of that section ended with the sentence
"one quantity orders every comparison in this paper between the two filters, and it is not
the noise process". BUAA_Campus was run after that sentence was written and it is a
counterexample: its mean occupancy is 5.62 of 24, the highest of the three labelled
benchmarks and above development's own 4.53, and we still lose there by +0.018 block-mean
AUC. A claim that survived four benchmarks and fails on the fifth has to be weakened, and
this phase measures by how much rather than guessing.

WHAT IS COMPUTED. Every labelled unit for which both filters were scored on identical blocks
-- 20 development conditions, 66 LED scenes, 6 BUAA_Campus sequences, 92 in all -- with its
mean ring occupancy, its noise fraction, and the paired block-mean AUC gap. Then the two
candidate explanations are compared, alone and together, both pooled and within each
benchmark.

WHY THE WITHIN-BENCHMARK NUMBERS ARE THE ONES TO READ. Pooled across benchmarks, occupancy
and noise fraction are both confounded with benchmark identity, and there are three
benchmarks, so the between-benchmark part of any pooled fit rests on an effective sample of
three. The within-benchmark correlations do not have that problem, and they are reported
separately for exactly that reason.

Read-only, and recomputed entirely from stored artefacts. Nothing is re-scanned.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
#: Mean ring occupancy above which the benchmark is not in the near-empty regime. This is
#: development's own mean, taken from Phase 60, not a value chosen here.
SUPPORT_MAX = 24


def collect(arm: str = "gated") -> List[Tuple[str, str, float, float, float, float]]:
    """(unit, benchmark, occupancy, noise fraction, our block AUC, EBF block AUC).

    `arm` selects which of our two configurations supplies the "ours" column. The three
    source artefacts name the same two arms differently, which is exactly how the gated
    values reached a paragraph about the adopted score, so the mapping is written out:

        arm      development (phase57)   LED (phase59)    BUAA (phase62)
        gated    ours_block_mean_auc     support          support
        beta0    --- (phase69 `count`)   beta0_support    beta0_support

    phase57 carries no beta == 0 arm, so the development rows come from phase69, which
    scores the same 20 conditions on the same 10,240-event blocks.
    """

    if arm not in ("gated", "beta0"):
        raise SystemExit(f"unknown arm {arm!r}; expected 'gated' or 'beta0'")

    occ = json.loads((ARTIFACTS / "phase60_occupancy_regime.json").read_text())
    dev = json.loads((ARTIFACTS / "phase57_ebf_labelled.json").read_text())["per_condition"]
    led = json.loads((ARTIFACTS / "phase59_ebf_led.json").read_text())["per_scene"]
    campus = json.loads((ARTIFACTS / "phase62_buaa_campus.json").read_text())["per_sequence"]
    iso = json.loads((ARTIFACTS / "phase69_operand_isolation.json").read_text())["per_unit"]
    ours_key = "support" if arm == "gated" else "beta0_support"

    rows: List[Tuple[str, str, float, float, float, float]] = []
    for name, row in dev.items():
        if name not in occ["development"]:
            continue
        if arm == "gated":
            ours = row["ours_block_mean_auc"]
        else:
            ours = iso[name]["count"]["block_mean_auc"]
        rows.append((name, "development", occ["development"][name]["mean_support"],
                     row["noise_fraction"], ours, row["block_mean_auc"]))
    for name, row in led.items():
        if name in occ["led"]:
            rows.append((name, "LED", occ["led"][name]["mean_support"],
                         row["noise_fraction"], row[ours_key]["block_mean_auc"],
                         row["ebf"]["block_mean_auc"]))
    for name, row in campus.items():
        # The repaired stream, so the release's label leak cannot enter either arm.
        rep = row["repaired"]
        rows.append((name, "BUAA_Campus", rep["_occupancy"]["mean_support"],
                     row["released"]["diagnostics"]["noise_fraction"],
                     rep[ours_key]["block_mean_auc"], rep["ebf"]["block_mean_auc"]))
    return rows


def r_squared(design: np.ndarray, target: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    return float(1.0 - (residual ** 2).sum() / ((target - target.mean()) ** 2).sum())


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase63_gap_predictors.json")
    parser.add_argument("--arm", choices=("gated", "beta0"), default="gated",
                        help="which of our configurations supplies the 'ours' column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    rows = collect(args.arm)
    names = [r[0] for r in rows]
    bench = np.array([r[1] for r in rows])
    support = np.array([r[2] for r in rows])
    noise = np.array([r[3] for r in rows])
    ours = np.array([r[4] for r in rows])
    ebf = np.array([r[5] for r in rows])
    gap = ebf - ours

    ones = np.ones(len(rows))
    pooled = {
        "units": len(rows),
        "corr_support_vs_gap": corr(support, gap),
        "corr_noise_vs_gap": corr(noise, gap),
        "r2_support_only": r_squared(np.column_stack([ones, support]), gap),
        "r2_noise_only": r_squared(np.column_stack([ones, noise]), gap),
        "r2_both": r_squared(np.column_stack([ones, support, noise]), gap),
        "caveat": ("pooled across three benchmarks, both predictors are confounded with "
                   "benchmark identity, so the between-benchmark part of this fit rests "
                   "on an effective sample of three"),
    }

    per_benchmark: Dict[str, Dict] = {}
    for b in ("development", "LED", "BUAA_Campus"):
        m = bench == b
        per_benchmark[b] = {
            "units": int(m.sum()),
            "mean_support": float(support[m].mean()),
            "support_range": [float(support[m].min()), float(support[m].max())],
            "mean_noise_fraction": float(noise[m].mean()),
            "ours_block_mean_auc": float(ours[m].mean()),
            "ebf_block_mean_auc": float(ebf[m].mean()),
            "mean_gap": float(gap[m].mean()),
            "corr_support_vs_gap": corr(support[m], gap[m]),
            "corr_noise_vs_gap": corr(noise[m], gap[m]),
        }

    dev_support = per_benchmark["development"]["mean_support"]
    campus = per_benchmark["BUAA_Campus"]
    payload = {
        "phase": "63 predictors of the gap to EBF across 92 labelled units",
        "unit": "one condition, scene or sequence on which both filters scored the same blocks",
        "arm": args.arm,
        "metric": "paired block-mean AUC, EBF minus ours",
        "support_max": SUPPORT_MAX,
        "pooled": pooled,
        "per_benchmark": per_benchmark,
        "counterexample": {
            "benchmark": "BUAA_Campus",
            "mean_support": campus["mean_support"],
            "development_mean_support": dev_support,
            "support_is_above_development": bool(campus["mean_support"] > dev_support),
            "mean_gap": campus["mean_gap"],
            "verdict": ("occupancy is the strongest single predictor of the gap but is not "
                        "sufficient: BUAA_Campus sits above development's own mean occupancy "
                        "and we still lose there"
                        if campus["mean_support"] > dev_support and campus["mean_gap"] > 0
                        else "BUAA_Campus is consistent with the occupancy account"),
        },
        "units": [{"name": n, "benchmark": b, "mean_support": float(s),
                   "noise_fraction": float(f), "ours": float(o), "ebf": float(e),
                   "gap": float(e - o)}
                  for (n, b, s, f, o, e) in rows],
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"{len(rows)} units across three labelled benchmarks\n")
    print(f"{'benchmark':13s} {'n':>3s} {'S':>6s} {'noise':>7s} {'ours':>7s} {'EBF':>7s} "
          f"{'gap':>8s} {'r(S,gap)':>9s} {'r(nf,gap)':>10s}")
    for b, v in per_benchmark.items():
        print(f"{b:13s} {v['units']:3d} {v['mean_support']:6.2f} {v['mean_noise_fraction']:7.3f} "
              f"{v['ours_block_mean_auc']:7.4f} {v['ebf_block_mean_auc']:7.4f} "
              f"{v['mean_gap']:+8.4f} {v['corr_support_vs_gap']:+9.3f} "
              f"{v['corr_noise_vs_gap']:+10.3f}")
    print(f"\npooled ({pooled['units']} units): r(S,gap) = {pooled['corr_support_vs_gap']:+.3f}, "
          f"r(noise,gap) = {pooled['corr_noise_vs_gap']:+.3f}")
    print(f"  R^2: support only {pooled['r2_support_only']:.3f}, "
          f"noise only {pooled['r2_noise_only']:.3f}, both {pooled['r2_both']:.3f}")
    print(f"\nverdict: {payload['counterexample']['verdict']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
