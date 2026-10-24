"""Phase 57: EBF on the two labelled development benchmarks, DND21 and DVSCLEAN.

WHY. Every cross-method row in this paper so far is transcribed from a source paper, and
the caption has to say so, because the two available sources disagree by up to 0.34 AUC on
the same classical filters. That is a weak position. EBF (Guo et al., IEEE TCSVT 2025) is
the strongest published rule-based denoiser we know of, its code is released, and it is
close enough to us in kind -- hand-designed, no trained weights -- that running it
ourselves on our own streams gives a row that is matched on data, on aggregation, and on
preprocessing all at once. That is a better comparison than any transcription, and it is
the one a reader of a training-free denoising paper will want.

WHAT IS MATCHED AND WHAT IS NOT. Both filters see the same events, the same labels, the
same block partition and the same two aggregations. They differ in causality, and the
difference favours us: EBF is strictly causal, updating its map only after scoring, while
our score takes a block-local bidirectional pass over 10,240 events. Any row that puts the
two side by side has to say this, and Table 3's caption already does for the learned rows.

EBF runs at its own published operating point -- s = 5, tau = 32 ms, its Sec. V-A choice --
across every condition. It is not refitted per dataset, which is the same discipline we
hold ourselves to. Because ROC-AUC is threshold-free, T_EBF never has to be chosen, so the
one parameter EBF says is application-dependent cannot advantage or disadvantage it here.

The tau sweep is run on the development sets only and is reported as a property of EBF, not
used to select anything: the adopted row stays at the published 32 ms whatever the sweep
says. If the sweep shows 32 ms is not EBF's best on our streams, that is worth printing --
it bounds how much of any gap is EBF's method and how much is its constant.

Read-only. Our own numbers are read from the published Phase 23 artefact rather than
recomputed, so this phase cannot silently disagree with the tables it will sit beside.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .ebf import EBF_S, EBF_TAU_US, check_equivalence, check_properties, ebf_score
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks
from .core import StateConfig
from .phase23_ablation_statistic import scan_condition
from .evaluate_dnd21 import PAPER_AUC as DND21_RATES
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK = 10_240
GEOMETRY = {"DND21": (346, 260), "DVSCLEAN": (1280, 720)}
#: Reported as a property of EBF. Nothing is selected on it. 32 ms is EBF's own choice.
TAU_SWEEP_MS = (1, 2, 4, 8, 16, 32, 64, 128)
#: Our published values, so the EBF rows land beside numbers this phase did not compute.
OURS_ARTEFACT = "phase23_ablation_statistic.json"
#: Conditions on which our own path is re-run and required to reproduce the artefact.
ANCHOR_CONDITIONS = (("DND21", "1hz_driving"), ("DVSCLEAN", "MAH00446_100"))
ANCHOR_TOLERANCE = 1e-6


def ebf_noise_score(events: np.ndarray, width: int, height: int,
                    tau_us: float) -> np.ndarray:
    """EBF's score, negated into this paper's orientation.

    Every AUC in this paper is computed as `roc_auc_score(label, score)` where the label
    is 1 for *noise* and the score is oriented so that larger means more noise-like -- our
    own score is, since the retention rule keeps the lowest-scoring events. EBF is defined
    the other way: "If S(x,y) is larger than T_EBF, it is classified as a signal event"
    (Guo et al., Sec. III). Negating is therefore a change of convention and not a change
    of method; it leaves EBF's ranking, and so its ROC curve, untouched.
    """

    return -ebf_score(events, width, height, tau_us=tau_us, s=EBF_S)


def load_condition(dataset: str, name: str, dnd21_root: Path,
                   dvsclean_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Every complete 10,240-event block of one condition, concatenated in stream order.

    Concatenating the blocks reproduces the stream prefix exactly, so EBF's map carries
    across block boundaries as a causal filter's must. Only the incomplete tail is
    dropped, and it is dropped for both filters.
    """

    if dataset == "DND21":
        rate, scene = name.split("_", 1)
        blocks = list(dnd21_blocks(dnd21_root / rate / SCENE_FILES[scene], BLOCK, None))
    else:
        blocks = list(dvsclean_blocks(dvsclean_root / f"{name}.hdf5", BLOCK, None))
    events = np.ascontiguousarray(np.concatenate([b for b, _ in blocks]), dtype=np.int64)
    labels = np.concatenate([l for _, l in blocks]).astype(np.int64)
    return events, labels


def block_mean_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mean of per-block AUCs, skipping blocks with only one class present."""

    per_block = []
    for start in range(0, len(scores) - BLOCK + 1, BLOCK):
        y = labels[start:start + BLOCK]
        if y.min() == y.max():
            continue
        per_block.append(roc_auc_score(y, scores[start:start + BLOCK]))
    return float(np.mean(per_block)) if per_block else float("nan")


def score_condition(events: np.ndarray, labels: np.ndarray, width: int, height: int,
                    tau_us: float) -> Dict[str, float]:
    started = time.perf_counter()
    scores = ebf_noise_score(events, width, height, tau_us)
    elapsed = time.perf_counter() - started
    return {
        "block_mean_auc": block_mean_auc(scores, labels),
        "event_auc": float(roc_auc_score(labels, scores)),
        "events": int(len(events)),
        "blocks": int(len(events) // BLOCK),
        "noise_fraction": float(labels.mean()),  # label 1 is noise
        "score_seconds": elapsed,
        "throughput_mev_s": float(len(events) / elapsed / 1e6),
    }


def anchor_our_own_path(dnd21_root: Path, dvsclean_root: Path,
                        published: Dict) -> Dict[str, float]:
    """Re-run our own score through Phase 23's helper and require the artefact back.

    The EBF rows are only worth printing if the AUC path that produced them -- the label
    convention, the score orientation, the block partition, the single-class skip -- is
    the same path that produced our published rows. Asserting that in prose is cheap.
    Re-running one condition per dataset and demanding the published number to six
    decimals is not, and it is what fails loudly if any of those conventions drifts.
    """

    drifts: Dict[str, float] = {}
    for dataset, name in ANCHOR_CONDITIONS:
        width, height = GEOMETRY[dataset]
        gate = StateConfig(width=width, height=height, taus_us=(250.0,), radius=1)
        score = StateConfig(width=width, height=height, taus_us=(250.0,), radius=2)
        if dataset == "DND21":
            rate, scene = name.split("_", 1)
            blocks = dnd21_blocks(dnd21_root / rate / SCENE_FILES[scene], BLOCK, None)
        else:
            blocks = dvsclean_blocks(dvsclean_root / f"{name}.hdf5", BLOCK, None)
        got = float(scan_condition(blocks, gate, score)["mean_block_auc"]["gated"])
        want = float(published[dataset][name]["mean_block_auc"]["gated"])
        drift = abs(got - want)
        drifts[f"{dataset}/{name}"] = drift
        if drift > ANCHOR_TOLERANCE:
            raise SystemExit(
                f"Anchor failed on {dataset}/{name}: re-running our own score through "
                f"this phase's AUC path gives {got:.8f} where {OURS_ARTEFACT} published "
                f"{want:.8f} (drift {drift:.2e} > {ANCHOR_TOLERANCE:.0e}). The EBF rows "
                f"would not be comparable to the published rows; refusing to write them.")
        print(f"anchor {dataset}/{name}: reproduced {got:.6f}, drift {drift:.2e}")
    return drifts


def conditions() -> List[Tuple[str, str]]:
    out = [("DND21", f"{rate}_{scene}") for rate in DND21_RATES for scene in SCENE_FILES]
    out += [("DVSCLEAN", f"{seq}_{level}") for seq in SEQUENCES for level in (50, 100)]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase57_ebf_labelled.json")
    parser.add_argument("--tau-sweep", action="store_true",
                        help="Also sweep tau. Reported, never used to select.")
    parser.add_argument("--max-conditions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    check_properties()
    worst, _ = check_equivalence()
    print(f"EBF kernel verified against the released code, max |diff| = {worst:.2e}, "
          f"and against four properties the released code cannot vouch for")

    ours = json.loads((ARTIFACTS / OURS_ARTEFACT).read_text())["conditions"]
    anchor_drifts = anchor_our_own_path(args.dnd21_root, args.dvsclean_root, ours)

    todo = conditions()[: args.max_conditions]
    per_condition: Dict[str, Dict] = {}
    sweep: Dict[str, Dict[str, float]] = {}

    for index, (dataset, name) in enumerate(todo, start=1):
        width, height = GEOMETRY[dataset]
        events, labels = load_condition(dataset, name, args.dnd21_root, args.dvsclean_root)
        record = score_condition(events, labels, width, height, EBF_TAU_US)
        record["dataset"] = dataset
        mine = ours[dataset][name]
        record["ours_block_mean_auc"] = float(mine["mean_block_auc"]["gated"])
        record["ours_event_auc"] = float(mine["global_event_auc"]["gated"])
        record["delta_block_mean_vs_ours"] = record["block_mean_auc"] - record["ours_block_mean_auc"]
        per_condition[f"{dataset}/{name}"] = record
        print(f"[{index}/{len(todo)}] {dataset}/{name}: EBF block {record['block_mean_auc']:.4f} "
              f"event {record['event_auc']:.4f} | ours block {record['ours_block_mean_auc']:.4f} "
              f"({record['throughput_mev_s']:.1f} Mev/s)")

        if args.tau_sweep:
            sweep[f"{dataset}/{name}"] = {
                f"{ms}ms": block_mean_auc(
                    ebf_noise_score(events, width, height, ms * 1000.0), labels)
                for ms in TAU_SWEEP_MS}
            best = max(sweep[f"{dataset}/{name}"].items(), key=lambda kv: kv[1])
            print(f"        tau sweep best {best[0]} at {best[1]:.4f}")

    def dataset_mean(dataset: str, key: str) -> float:
        vals = [r[key] for r in per_condition.values() if r["dataset"] == dataset]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {}
    for key in ("block_mean_auc", "event_auc", "ours_block_mean_auc", "ours_event_auc"):
        dnd, dvs = dataset_mean("DND21", key), dataset_mean("DVSCLEAN", key)
        summary[key] = {"DND21": dnd, "DVSCLEAN": dvs, "pooled": float(np.mean([dnd, dvs]))}

    deltas = np.array([r["delta_block_mean_vs_ours"] for r in per_condition.values()])
    payload = {
        "phase": "57 EBF on the labelled development benchmarks",
        "baseline": {
            "method": "EBF, Guo et al., IEEE TCSVT 35(10):9889-9894, 2025",
            "implementation": "ours, verified against a literal transcription of the "
                              "released repository (github.com/shicy17/EBF)",
            "kernel_vs_released_max_abs_diff": worst,
            "s": EBF_S, "tau_us": EBF_TAU_US,
            "operating_point": "the paper's own Sec. V-A choice, not refitted per dataset",
            "threshold": "not required: ROC-AUC is threshold-free",
            "causality": "strictly causal; ours takes a block-local bidirectional pass",
        },
        "ours_source": f"{OURS_ARTEFACT} (published values, not recomputed)",
        "anchor_drift_vs_published": anchor_drifts,
        "block_size": BLOCK,
        "conditions": len(per_condition),
        "summary": summary,
        "paired_block_mean": {
            "mean_delta_ebf_minus_ours": float(deltas.mean()),
            "conditions_ebf_ahead": int((deltas > 0).sum()),
            "conditions_ours_ahead": int((deltas < 0).sum()),
            "conditions_tied": int((deltas == 0).sum()),
        },
        "tau_sweep": {
            "note": "reported as a property of EBF; the adopted row stays at 32 ms",
            "grid_ms": list(TAU_SWEEP_MS),
            "per_condition": sweep,
        } if sweep else None,
        "per_condition": per_condition,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\nEBF   block-mean: DND21 {summary['block_mean_auc']['DND21']:.4f} "
          f"DVSCLEAN {summary['block_mean_auc']['DVSCLEAN']:.4f} "
          f"pooled {summary['block_mean_auc']['pooled']:.4f}")
    print(f"ours  block-mean: DND21 {summary['ours_block_mean_auc']['DND21']:.4f} "
          f"DVSCLEAN {summary['ours_block_mean_auc']['DVSCLEAN']:.4f} "
          f"pooled {summary['ours_block_mean_auc']['pooled']:.4f}")
    p = payload["paired_block_mean"]
    print(f"paired: EBF ahead on {p['conditions_ebf_ahead']}, behind on "
          f"{p['conditions_ours_ahead']}, tied on {p['conditions_tied']} "
          f"of {len(per_condition)}; mean delta {p['mean_delta_ebf_minus_ours']:+.4f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
