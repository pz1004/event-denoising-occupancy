"""Phase 67: the operand, isolated -- same map, same window, same causality, on LED.

THE CONFOUND THIS REMOVES. Table VII compares a block-local bidirectional occupancy count
over a 10,240-event block against a strictly causal decayed magnitude over a 32 ms window.
Three things differ at once: what is accumulated (a count against a decayed sum), over how
long (the block's extent against 32 ms), and with what access to the future (both
directions against the past only). A difference between those two columns cannot be
attributed to the operand, and this paper's central claim is about the operand.

`causal_occupancy` removes all three. It is EBF's own traversal -- EBF's timestamp and
polarity map, EBF's 5x5 neighbourhood, EBF's polarity rule, EBF's causal update order --
with one line changed, the accumulator. At W = tau = 32 ms the two scores read the same
taps of the same map at the same instant and differ only in whether they count them or sum
their decayed weights. Whatever separates the two rows below is the operand, and nothing
else is left to carry it.

TWO WINDOWS ARE REPORTED, both fixed before the run:

    W = 32 ms      EBF's own constant, borrowed so the horizon is matched exactly. This is
                   the isolating contrast and the primary row.
    W = 24.302 ms  the mean median span of our frozen 10,240-event block over the 20
                   development conditions (Phase 60). This is what our own block length
                   already spends, restated in time, and it reads no LED.

AGGREGATION. The scores are streaming, so the partition affects only how block-mean AUC is
formed. We use the 10,240-event windows of Phase 54 so that every number here is directly
comparable with Table VII, and report the pooled event-level AUC beside it.

WHAT IS NOT CLAIMED. This phase does not show the block-local filter is causal; it is not.
It shows what the occupancy operand is worth once causality, horizon, neighbourhood and
state are held fixed, which is a different and narrower question, and the one the operand
claim actually needs.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .causal_occupancy import (
    OCCUPANCY_MAX,
    check_equivalence,
    check_properties,
    occupancy_noise_score,
)
from .ebf import EBF_TAU_US
from .phase54_led import (
    ARTIFACTS,
    BLOCK,
    HEIGHT,
    LED_ROOT,
    RETENTION,
    WIDTH,
    _rates,
    cluster_bootstrap,
    iter_scene,
    sign_flip,
)
from .phase57_ebf_labelled import ebf_noise_score
from .phase66_time_budget import T_STAR_US
from .retention import otsu_retention

#: (name, window in us). The first is the matched-horizon contrast and the primary row.
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


def operating_point(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """The frozen r, Otsu, and the count-matched reference -- the rows Table VII carries."""

    n = len(scores) - (len(scores) % BLOCK)
    keep = np.zeros(len(scores), dtype=bool)
    keep_adaptive = np.zeros(len(scores), dtype=bool)
    keep_oracle = np.zeros(len(scores), dtype=bool)
    for start in range(0, n - BLOCK + 1, BLOCK):
        block = scores[start:start + BLOCK]
        order = np.argsort(block, kind="stable")
        keep[start + order[:int(np.floor(RETENTION * BLOCK))]] = True
        share = otsu_retention(block)
        keep_adaptive[start + order[:int(np.floor(share * BLOCK))]] = True
        n_signal = int((labels[start:start + BLOCK] == 0).sum())
        keep_oracle[start + order[:n_signal]] = True
    sl = slice(0, n)
    return {**_rates(keep[sl], labels[sl], ""),
            **_rates(keep_adaptive[sl], labels[sl], "_adaptive"),
            **_rates(keep_oracle[sl], labels[sl], "_oracle_r")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LED_ROOT)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=200_000)
    parser.add_argument("--flip-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase67_matched_causal_led.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    worst, checked = check_equivalence()
    check_properties()
    print(f"occupancy kernel verified against its reference on {checked} events "
          f"(worst diff {worst}), and against four properties including the shared "
          f"term set with EBF at W = tau")

    scenes = sorted(p.name for p in (args.root / "raw_event").iterdir() if p.is_dir())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    per_scene: Dict[str, Dict] = {}
    per_block: Dict[str, Dict[str, List[float]]] = {arm: {} for arm in ARMS}
    saturation: List[float] = []

    for index, scene in enumerate(scenes, start=1):
        step = time.perf_counter()
        events, labels, _, _, _ = iter_scene(args.root, scene)
        if len(events) < BLOCK:
            continue
        n = len(events) - (len(events) % BLOCK)
        y = labels[:n]
        if y.min() == y.max():
            continue

        scores: Dict[str, np.ndarray] = {
            "ebf": ebf_noise_score(events, WIDTH, HEIGHT, tau_us=float(EBF_TAU_US))}
        for name, window in WINDOWS:
            scores[name] = occupancy_noise_score(events, WIDTH, HEIGHT, window)
        saturation.append(float(np.mean(-scores["occupancy_w32ms"][:n] == OCCUPANCY_MAX)))

        record: Dict = {"blocks": n // BLOCK, "events": int(n),
                        "noise_fraction": float(y.mean()),
                        "mean_occupancy_w32ms": float(np.mean(-scores["occupancy_w32ms"][:n]))}
        for arm in ARMS:
            s = scores[arm][:n]
            aucs = block_aucs(s, y)
            per_block[arm][scene] = aucs
            record[arm] = {
                "block_mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
                "event_auc": float(roc_auc_score(y, s)),
                "pr_auc_noise": float(average_precision_score(y, s)),
                **operating_point(s, labels),
            }
        per_scene[scene] = record
        print(f"[{index:3d}/{len(scenes)}] {scene:14s} "
              f"occ32={record['occupancy_w32ms']['block_mean_auc']:.4f} "
              f"occT*={record['occupancy_wTstar']['block_mean_auc']:.4f} "
              f"ebf={record['ebf']['block_mean_auc']:.4f} "
              f"mean_occ={record['mean_occupancy_w32ms']:5.2f} "
              f"[{time.perf_counter() - step:.0f}s]", flush=True)

    def pooled(arm: str, field: str) -> float:
        vals = [r[arm][field] for r in per_scene.values() if not np.isnan(r[arm][field])]
        return float(np.mean(vals)) if vals else float("nan")

    fields = ("block_mean_auc", "event_auc", "pr_auc_noise", "retained_fraction",
              "signal_recall", "noise_rejection", "balanced_accuracy",
              "balanced_accuracy_adaptive", "retained_fraction_adaptive",
              "balanced_accuracy_oracle_r")
    summary = {arm: {f: pooled(arm, f) for f in fields} for arm in ARMS}

    contrasts: Dict[str, Dict] = {}
    for left, right in (("occupancy_w32ms", "ebf"),
                        ("occupancy_wTstar", "ebf"),
                        ("occupancy_w32ms", "occupancy_wTstar")):
        clusters = [(s, [a - b for a, b in zip(per_block[left][s], per_block[right][s])])
                    for s in per_scene]
        contrasts[f"{left}_minus_{right}"] = {
            **cluster_bootstrap(clusters, args.replicates, rng),
            **sign_flip(clusters, args.flip_draws, rng),
        }

    payload = {
        "phase": "67 matched causal operand contrast on LED",
        "question": ("With EBF's map, window, neighbourhood, polarity rule and causal "
                     "order held fixed, what is the occupancy operand worth?"),
        "windows_us": {name: window for name, window in WINDOWS},
        "primary": "occupancy_w32ms",
        "aggregation": {"block": BLOCK, "note": "Phase 54's partition, so the rows are "
                                                "directly comparable with Table VII"},
        "kernel_check": {"worst_diff_vs_reference": worst, "events": checked},
        "occupancy_max": OCCUPANCY_MAX,
        "fraction_saturated_w32ms": float(np.mean(saturation)) if saturation else float("nan"),
        "scenes": len(per_scene),
        "summary": summary,
        "contrasts": contrasts,
        "per_scene": per_scene,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.output}")
    for arm in ARMS:
        print(f"  {arm:18s} block={summary[arm]['block_mean_auc']:.4f} "
              f"event={summary[arm]['event_auc']:.4f} "
              f"bal={summary[arm]['balanced_accuracy']:.4f} "
              f"bal_adapt={summary[arm]['balanced_accuracy_adaptive']:.4f}")
    for key, value in contrasts.items():
        print(f"  {key}: mean {value['mean']:+.4f} ci {value['ci95']} "
              f"ahead {value.get('clusters_ahead')} / {value['n_clusters']} "
              f"p {value['p_two_sided']:.2e}")


if __name__ == "__main__":
    main()
