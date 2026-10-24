"""Phase 64: the operand claim reverses on BUAA_Campus, and Supplementary S5 says why.

WHAT HAPPENED. On BUAA_Campus, ring *magnitude* beats ring *support* by about $+0.042$
event-level AUC on all six released sequences. That is the first reversal of this paper's
central substitution on any benchmark: support led on all 20 development conditions, all
eight held-out E-MLB cells, ED24 pooled, and 63 of 66 LED scenes.

WHY IT IS NOT A REFUTATION OF THE MODEL. Supplementary S5 derives the substitution from a
Binomial occupancy model whose criterion is the standardised mean deflection, and it states
plainly that the clean case in which amplitude carries no class information is falsified by
our data. The prediction that follows is conditional, not absolute: counting wins where the
amplitude accumulated at an occupied tap is similar for the two classes, and loses where it
is not. No benchmark used before this one sampled the second regime.

THE TEST. For every labelled unit, compute the class-conditional means and the standardised
mean deflection of both operands, and the mean amplitude per *occupied* tap by class, which
is the quantity the model says decides it. Then ask a yes/no question unit by unit: does
d(R) > d(S) predict that magnitude outranks support? A criterion that only agreed where we
already win would be worthless.

Blocks are subsampled by `--stride` because the deflection is a distributional quantity and
does not need every block; the AUC values it is checked against are the full-stream ones
already published by Phases 57 and 62.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import ring_support
from .phase8_score_upgrade import operands
from .phase57_ebf_labelled import GEOMETRY, conditions, load_condition
from .phase62_buaa_campus import BUAA_ROOT, HEIGHT, SEQUENCES, WIDTH, load_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK = 10_240
TAU_US = 250.0
SCORE_RADIUS = 2


def deflection(events: np.ndarray, labels: np.ndarray, width: int, height: int,
               stride: int) -> Dict[str, float]:
    """Class-conditional moments of both operands on one stream. Label 1 is noise."""

    config = StateConfig(width=width, height=height, taus_us=(TAU_US,), radius=SCORE_RADIUS)
    support_all: List[np.ndarray] = []
    magnitude_all: List[np.ndarray] = []
    label_all: List[np.ndarray] = []
    for start in range(0, len(events) - BLOCK + 1, BLOCK * stride):
        chunk = events[start:start + BLOCK]
        raw = bidirectional_features(chunk, config)
        _, r_same, _ = operands(raw, config)
        support_all.append(ring_support(raw, len(config.taus_us),
                                        config.feature_dim, config.radius))
        magnitude_all.append(r_same)
        label_all.append(labels[start:start + BLOCK])
    support = np.concatenate(support_all).astype(np.float64)
    magnitude = np.concatenate(magnitude_all).astype(np.float64)
    y = np.concatenate(label_all)
    signal, noise = y == 0, y == 1
    if not signal.any() or not noise.any():
        return {}

    def d(v: np.ndarray) -> float:
        pooled = np.sqrt(0.5 * (v[signal].var() + v[noise].var()))
        return float((v[signal].mean() - v[noise].mean()) / pooled) if pooled > 0 else float("nan")

    def per_tap(mask: np.ndarray) -> float:
        occupied = support[mask].mean()
        return float(magnitude[mask].mean() / occupied) if occupied > 0 else float("nan")

    return {
        "blocks_sampled": len(support_all),
        "support_signal_mean": float(support[signal].mean()),
        "support_noise_mean": float(support[noise].mean()),
        "magnitude_signal_mean": float(magnitude[signal].mean()),
        "magnitude_noise_mean": float(magnitude[noise].mean()),
        "d_support": d(support),
        "d_magnitude": d(magnitude),
        "amplitude_per_occupied_tap_signal": per_tap(signal),
        "amplitude_per_occupied_tap_noise": per_tap(noise),
        "predicts_magnitude_wins": bool(d(magnitude) > d(support)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--campus-root", type=Path, default=BUAA_ROOT)
    parser.add_argument("--stride", type=int, default=8, help="sample every Nth block")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase64_deflection_reversal.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    p57 = json.loads((ARTIFACTS / "phase57_ebf_labelled.json").read_text())["per_condition"]
    p62 = json.loads((ARTIFACTS / "phase62_buaa_campus.json").read_text())["per_sequence"]
    p23 = json.loads((ARTIFACTS / "phase23_ablation_statistic.json").read_text())["conditions"]

    units: Dict[str, Dict] = {}

    for dataset, name in conditions():
        width, height = GEOMETRY[dataset]
        events, labels = load_condition(dataset, name, args.dnd21_root, args.dvsclean_root)
        record = deflection(events, labels, width, height, args.stride)
        mine = p23[dataset][name]["global_event_auc"]
        record["benchmark"] = dataset
        record["support_event_auc"] = float(mine["gated"])
        record["magnitude_event_auc"] = float(mine["magnitude_gated"])
        record["magnitude_wins"] = record["magnitude_event_auc"] > record["support_event_auc"]
        units[f"{dataset}/{name}"] = record
        print(f"{dataset}/{name:24s} d_S={record['d_support']:+.3f} d_R={record['d_magnitude']:+.3f} "
              f"predict_mag={record['predicts_magnitude_wins']!s:5s} "
              f"actual_mag={record['magnitude_wins']!s:5s}", flush=True)

    for name in SEQUENCES:
        events, labels, _diag = load_sequence(args.campus_root / f"{name}.npy", True, 0)
        record = deflection(events, labels, WIDTH, HEIGHT, args.stride)
        arm = p62[name]["repaired"]
        record["benchmark"] = "BUAA_Campus"
        record["support_event_auc"] = float(arm["support"]["event_auc"])
        record["magnitude_event_auc"] = float(arm["magnitude"]["event_auc"])
        record["magnitude_wins"] = record["magnitude_event_auc"] > record["support_event_auc"]
        units[f"BUAA_Campus/{name}"] = record
        print(f"BUAA_Campus/{name:22s} d_S={record['d_support']:+.3f} d_R={record['d_magnitude']:+.3f} "
              f"predict_mag={record['predicts_magnitude_wins']!s:5s} "
              f"actual_mag={record['magnitude_wins']!s:5s}", flush=True)

    agree = sum(1 for u in units.values()
                if u["predicts_magnitude_wins"] == u["magnitude_wins"])
    by_bench: Dict[str, Dict] = {}
    for b in ("DND21", "DVSCLEAN", "BUAA_Campus"):
        sel = [u for u in units.values() if u["benchmark"] == b]
        if not sel:
            continue
        by_bench[b] = {
            "units": len(sel),
            "d_support": float(np.mean([u["d_support"] for u in sel])),
            "d_magnitude": float(np.mean([u["d_magnitude"] for u in sel])),
            "amplitude_per_tap_signal":
                float(np.mean([u["amplitude_per_occupied_tap_signal"] for u in sel])),
            "amplitude_per_tap_noise":
                float(np.mean([u["amplitude_per_occupied_tap_noise"] for u in sel])),
            "magnitude_wins": int(sum(u["magnitude_wins"] for u in sel)),
        }

    payload = {
        "phase": "64 the deflection criterion and the BUAA_Campus reversal",
        "criterion": ("Supplementary S5's standardised mean deflection; it predicts that "
                      "magnitude outranks support wherever d(R) > d(S)"),
        "unit": "one labelled condition or sequence",
        "stride": args.stride,
        "units_total": len(units),
        "criterion_agrees": agree,
        "criterion_disagrees": len(units) - agree,
        "per_benchmark": by_bench,
        "units": units,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\ndeflection criterion agrees with the measured ordering on "
          f"{agree} of {len(units)} units")
    for b, v in by_bench.items():
        print(f"  {b:12s} n={v['units']:2d}  d_S={v['d_support']:+.3f}  d_R={v['d_magnitude']:+.3f}  "
              f"amp/tap sig={v['amplitude_per_tap_signal']:.3f} noi={v['amplitude_per_tap_noise']:.3f}  "
              f"magnitude wins {v['magnitude_wins']}/{v['units']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
