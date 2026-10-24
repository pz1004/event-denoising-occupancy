"""Phase 68: the same matched causal contrast, on the development conditions and BUAA.

Phase 67 runs the matched contrast on LED. This one runs it on the other two labelled
benchmarks so the operand question is answered on every labelled unit this paper has,
under one protocol: EBF's timestamp and polarity map, EBF's 5x5 neighbourhood, EBF's
polarity rule, EBF's strictly causal update order, and one line changed -- count the
admitted taps instead of summing their decayed weights.

WHY BOTH BENCHMARKS MATTER HERE. The 20 development conditions are where the operand was
selected, so they are the weakest evidence in the paper and are reported as such; what they
add is that the *matched* comparison can be checked against the confounded one on the data
where the confounded one was strongest. BUAA_Campus is where the block-local comparison
reversed, and the matched contrast says whether the reversal survives once the horizon,
the state and the causality stop differing.

Aggregation is the 10,240-event partition every other table in this paper uses, so the rows
land beside Tables III and VIII without translation.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .causal_occupancy import check_equivalence, check_properties, occupancy_noise_score
from .ebf import EBF_TAU_US
from .phase57_ebf_labelled import GEOMETRY, conditions, ebf_noise_score, load_condition
from .phase62_buaa_campus import BUAA_ROOT, MAX_EVENTS, SEQUENCES, load_sequence
from .phase62_buaa_campus import HEIGHT as BUAA_HEIGHT, WIDTH as BUAA_WIDTH
from .phase66_time_budget import T_STAR_US

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK = 10_240
WINDOWS = (("occupancy_w32ms", float(EBF_TAU_US)),
           ("occupancy_wTstar", float(T_STAR_US)))
ARMS = tuple(name for name, _ in WINDOWS) + ("ebf",)


def block_aucs(scores: np.ndarray, labels: np.ndarray) -> List[float]:
    out: List[float] = []
    for start in range(0, len(scores) - BLOCK + 1, BLOCK):
        y = labels[start:start + BLOCK]
        if y.min() == y.max():
            continue
        out.append(float(roc_auc_score(y, scores[start:start + BLOCK])))
    return out


def measure(events: np.ndarray, labels: np.ndarray, width: int, height: int) -> Dict:
    record: Dict = {"events": int(len(events)), "noise_fraction": float(labels.mean())}
    scores = {"ebf": ebf_noise_score(events, width, height, tau_us=float(EBF_TAU_US))}
    for name, window in WINDOWS:
        scores[name] = occupancy_noise_score(events, width, height, window)
    record["mean_occupancy_w32ms"] = float(np.mean(-scores["occupancy_w32ms"]))
    for arm in ARMS:
        aucs = block_aucs(scores[arm], labels)
        record[arm] = {
            "block_mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "event_auc": float(roc_auc_score(labels, scores[arm])),
            "blocks": len(aucs),
        }
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--buaa-root", type=Path, default=BUAA_ROOT)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-conditions", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase68_matched_causal_labelled.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    worst, checked = check_equivalence()
    check_properties()
    print(f"occupancy kernel verified on {checked} events, worst diff {worst}")

    per_unit: Dict[str, Dict] = {}

    todo: List[Tuple[str, str]] = conditions()
    if args.max_conditions:
        todo = todo[:args.max_conditions]
    for dataset, name in todo:
        step = time.perf_counter()
        width, height = GEOMETRY[dataset]
        events, labels = load_condition(dataset, name, args.dnd21_root, args.dvsclean_root)
        record = measure(events, labels, width, height)
        record["benchmark"] = dataset
        per_unit[f"{dataset}/{name}"] = record
        print(f"{dataset}/{name:22s} occ32={record['occupancy_w32ms']['block_mean_auc']:.4f} "
              f"occT*={record['occupancy_wTstar']['block_mean_auc']:.4f} "
              f"ebf={record['ebf']['block_mean_auc']:.4f} "
              f"occ={record['mean_occupancy_w32ms']:5.2f} "
              f"[{time.perf_counter() - step:.0f}s]", flush=True)

    for sequence in SEQUENCES:
        step = time.perf_counter()
        events, labels, _ = load_sequence(args.buaa_root / f"{sequence}.npy", True, args.seed)
        record = measure(events, labels, BUAA_WIDTH, BUAA_HEIGHT)
        record["benchmark"] = "BUAA_Campus"
        per_unit[f"BUAA_Campus/{sequence}"] = record
        print(f"BUAA_Campus/{sequence:16s} occ32={record['occupancy_w32ms']['block_mean_auc']:.4f} "
              f"occT*={record['occupancy_wTstar']['block_mean_auc']:.4f} "
              f"ebf={record['ebf']['block_mean_auc']:.4f} "
              f"occ={record['mean_occupancy_w32ms']:5.2f} "
              f"[{time.perf_counter() - step:.0f}s]", flush=True)

    summary: Dict[str, Dict] = {}
    for bench in sorted({r["benchmark"] for r in per_unit.values()}) + ["development"]:
        if bench == "development":
            units = [r for r in per_unit.values() if r["benchmark"] in GEOMETRY]
        else:
            units = [r for r in per_unit.values() if r["benchmark"] == bench]
        if not units:
            continue
        summary[bench] = {
            "units": len(units),
            "mean_occupancy_w32ms": float(np.mean([u["mean_occupancy_w32ms"] for u in units])),
            **{arm: {f: float(np.mean([u[arm][f] for u in units]))
                     for f in ("block_mean_auc", "event_auc")} for arm in ARMS},
        }

    payload = {
        "phase": "68 matched causal operand contrast on development and BUAA_Campus",
        "windows_us": {name: window for name, window in WINDOWS},
        "primary": "occupancy_w32ms",
        "block": BLOCK,
        "buaa_max_events": MAX_EVENTS,
        "kernel_check": {"worst_diff_vs_reference": worst, "events": checked},
        "summary": summary,
        "per_unit": per_unit,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.output}")
    for bench, s in summary.items():
        print(f"  {bench:14s} n={s['units']:3d} occ={s['mean_occupancy_w32ms']:5.2f} "
              f"count={s['occupancy_w32ms']['block_mean_auc']:.4f} "
              f"countT*={s['occupancy_wTstar']['block_mean_auc']:.4f} "
              f"ebf={s['ebf']['block_mean_auc']:.4f}")


if __name__ == "__main__":
    main()
