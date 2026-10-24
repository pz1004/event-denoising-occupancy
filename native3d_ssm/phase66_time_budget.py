"""Phase 66: is the LED deficit caused by the block-length constant? A controlled test.

WHAT SECTION IV-E CURRENTLY CLAIMS, AND WHY IT IS NOT ENOUGH. Our block is fixed in
*events*, so its temporal extent shrinks as the event rate rises: 24.3 ms on the
development conditions, 0.77 ms on LED. Mean ring support falls with it, and across LED's
66 scenes our block-mean AUC tracks occupancy at r = +0.91. That is a correlation. It is
consistent with "the block length delivers an empty ring, and the empty ring costs the
accuracy", but it is equally consistent with occupancy being a marker for something else
that varies across scenes. Nothing in the paper so far *intervenes* on block length.

THIS PHASE INTERVENES. It repartitions the same events, the same labels and the same
scenes by a fixed *time* budget T instead of a fixed event count, and re-measures. If the
occupancy account is right, occupancy must rise with T and accuracy must rise with it; if
accuracy does not move, the account is wrong and the paper has to say so.

THE PRE-DECLARED T. T* = 24,302 us, the mean over the 20 development conditions of the
median span of a 10,240-event block (Phase 60, `ebf_comparison.development_block_span_ms
_median`). It is a development-only quantity: it is what our own frozen block length
*already spends* on the data the paper selected on, restated in time. Nothing about it
reads LED. It is fixed before this script runs and is the primary row.

The other T values are a sweep and are reported as a sweep. They are exploratory, they are
not a selection, and no number anywhere else in the paper is chosen from them. T = 775 us
is included because it is LED's own median block span at N = 10,240, so it is the status
quo restated in time and shows what the repartition alone costs or buys.

CAUSALITY, WHICH THREE READERS ASKED ABOUT TOGETHER. The same feature pass yields the
forward-only support at no extra cost, because `bidirectional_features` concatenates the
two passes and `ring_support(forward_only=True)` reads only the first half. Every T is
therefore reported twice, block-local bidirectional and strictly causal, against an EBF
that is strictly causal at every T. This removes the lookahead asymmetry from the
comparison rather than describing it.

BETA = 0 THROUGHOUT. The gate is not exercised here. Section V adopts beta = 0 and this
phase scores the adopted filter, not the gated one.

EBF IS PARTITION-INDEPENDENT and is scored once per scene over the whole stream, because
it is causal and keeps no block state. Only the aggregation into blocks changes with T,
which is what makes every row here exactly paired with ours: same events, same labels,
same windows, same single-class skip.

THE ANCHOR. Before any new number is written, the fixed-N = 10,240 partition is run through
this file's own code path and required to reproduce Phase 54's published per-scene
`beta0_support` block-mean AUC to 1e-6. If it does not, the repartition is being compared
against something other than the published table and the run refuses to produce a number.

WHAT IT COSTS. A time budget is not free: the block buffer grows with the event rate,
where a fixed event count bounds it by construction. Mean and maximum events per window
are recorded at every T, and the paper reports the accuracy against the buffer, not the
accuracy alone.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import ring_support
from .phase8_score_upgrade import operands
from .phase54_led import (
    ARTIFACTS,
    BLOCK,
    HEIGHT,
    LED_ROOT,
    SCORE_RADIUS,
    TAU_US,
    WIDTH,
    cluster_bootstrap,
    iter_scene,
    sign_flip,
)
from .phase57_ebf_labelled import ebf_noise_score
from .ebf import EBF_TAU_US, check_equivalence, check_properties

#: Mean over the 20 development conditions of the median 10,240-event block span, in us.
#: Phase 60, `ebf_comparison.development_block_span_ms_median` = 24.3024 ms. Frozen here.
T_STAR_US = 24_302

#: The sweep. Exploratory; T_STAR_US is the pre-declared point and is a member of it.
#: 775 us is LED's own median block span at N = 10,240 -- the status quo restated in time.
T_GRID_US = (775, 3_000, 10_000, T_STAR_US, 50_000)

#: Windows below this are dropped rather than scored; the count is reported.
MIN_EVENTS = 1_024

#: Our arms. Every one is beta = 0; the gate is not exercised in this phase.
ARMS = ("support", "support_causal", "magnitude", "magnitude_causal")

#: Phase 54's published per-scene values the fixed-N anchor must reproduce.
ANCHOR_ARTEFACT = "phase54_led.json"
ANCHOR_ARM = "beta0_support"
ANCHOR_TOLERANCE = 1e-6

#: Packed event width used by the cost accounting of Section IV-H, in bytes.
PACKED_EVENT_BYTES = 8


def time_windows(t: np.ndarray, budget_us: int) -> List[Tuple[int, int]]:
    """Half-open [start, stop) index ranges of consecutive fixed-duration windows."""

    origin = int(t[0])
    edges = np.arange(origin, int(t[-1]) + budget_us + 1, budget_us, dtype=np.int64)
    cuts = np.searchsorted(t, edges, side="left")
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= MIN_EVENTS]


def count_windows(n: int, block: int) -> List[Tuple[int, int]]:
    """The fixed-event-count partition, exactly as Phase 54 forms it."""

    return [(s, s + block) for s in range(0, n - block + 1, block)]


def score_windows(events: np.ndarray, windows: List[Tuple[int, int]],
                  config: StateConfig) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Per-event scores for every arm over the given windows, and per-event support.

    Events outside every window keep NaN and are excluded from every metric.
    """

    out = {arm: np.full(len(events), np.nan, dtype=np.float64) for arm in ARMS}
    support_all = np.full(len(events), np.nan, dtype=np.float64)
    scale_count = len(config.taus_us)
    for start, stop in windows:
        raw = bidirectional_features(events[start:stop], config)
        _, r_same, _ = operands(raw, config)
        r_same_causal = raw[:, 2 * scale_count]
        s_bidir = ring_support(raw, scale_count, config.feature_dim, config.radius)
        s_causal = ring_support(raw, scale_count, config.feature_dim, config.radius,
                                forward_only=True)
        out["support"][start:stop] = -np.log1p(s_bidir)
        out["support_causal"][start:stop] = -np.log1p(s_causal)
        out["magnitude"][start:stop] = -r_same
        out["magnitude_causal"][start:stop] = -r_same_causal
        support_all[start:stop] = s_bidir
    return out, support_all


def window_aucs(scores: np.ndarray, labels: np.ndarray,
                windows: List[Tuple[int, int]]) -> Tuple[List[float], List[int]]:
    """Per-window ROC-AUC, and the indices of the windows that produced one."""

    values: List[float] = []
    kept: List[int] = []
    for index, (start, stop) in enumerate(windows):
        y = labels[start:stop]
        if y.min() == y.max():
            continue
        values.append(float(roc_auc_score(y, scores[start:stop])))
        kept.append(index)
    return values, kept


def paired(ours: List[float], theirs: List[float]) -> List[float]:
    return [a - b for a, b in zip(ours, theirs)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LED_ROOT)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--budgets", type=int, nargs="*", default=list(T_GRID_US))
    parser.add_argument("--replicates", type=int, default=200_000)
    parser.add_argument("--flip-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--skip-anchor", action="store_true",
                        help="debug only; the anchor is what ties this run to Table VII")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase66_time_budget.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    check_properties()
    worst, _ = check_equivalence()
    print(f"EBF kernel verified against the released code, max |diff| = {worst:.2e}")

    published = json.loads((ARTIFACTS / ANCHOR_ARTEFACT).read_text())["per_scene"]
    config = StateConfig(width=WIDTH, height=HEIGHT, taus_us=(TAU_US,), radius=SCORE_RADIUS)

    scenes = sorted(p.name for p in (args.root / "raw_event").iterdir() if p.is_dir())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    partitions: List[Tuple[str, Optional[int]]] = [("fixed_n_10240", None)]
    partitions += [(f"t_{b}us", b) for b in args.budgets]

    # partition -> arm -> scene -> per-window AUC
    per_window: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        name: {arm: {} for arm in (*ARMS, "ebf")} for name, _ in partitions}
    # partition -> scene -> record
    per_scene: Dict[str, Dict[str, Dict]] = {name: {} for name, _ in partitions}
    anchor_worst = 0.0
    anchor_checked = 0

    for index, scene in enumerate(scenes, start=1):
        step = time.perf_counter()
        events, labels, _, _, _ = iter_scene(args.root, scene)
        if len(events) < BLOCK:
            continue
        # EBF is causal and keeps no block state, so one pass over the scene serves every
        # partition. Only the aggregation into windows changes.
        ebf = ebf_noise_score(events, WIDTH, HEIGHT, tau_us=float(EBF_TAU_US))

        for name, budget in partitions:
            windows = (count_windows(len(events), BLOCK) if budget is None
                       else time_windows(events[:, 0], budget))
            if not windows:
                continue
            scores, support = score_windows(events, windows, config)
            scored = ~np.isnan(scores["support"])
            y = labels[scored]
            if y.min() == y.max():
                continue

            sizes = np.array([stop - start for start, stop in windows], dtype=np.int64)
            spans = np.array([int(events[stop - 1, 0] - events[start, 0])
                              for start, stop in windows], dtype=np.int64)
            record: Dict = {
                "windows": len(windows),
                "events": int(scored.sum()),
                "noise_fraction": float(y.mean()),
                "mean_support": float(np.nanmean(support)),
                "fraction_support_zero": float(np.nanmean(support[scored] == 0)),
                "fraction_support_saturated": float(np.nanmean(support[scored] == 24)),
                "events_per_window_mean": float(sizes.mean()),
                "events_per_window_max": int(sizes.max()),
                "span_us_median": float(np.median(spans)),
            }
            ebf_aucs, ebf_kept = window_aucs(ebf, labels, windows)
            per_window[name]["ebf"][scene] = ebf_aucs
            record["ebf"] = {"block_mean_auc": float(np.mean(ebf_aucs)) if ebf_aucs
                             else float("nan"),
                             "event_auc": float(roc_auc_score(y, ebf[scored]))}
            for arm in ARMS:
                aucs, kept = window_aucs(scores[arm], labels, windows)
                if kept != ebf_kept:
                    raise SystemExit(f"{scene}/{name}/{arm}: window sets diverged; the "
                                     f"paired difference would not be paired")
                per_window[name][arm][scene] = aucs
                record[arm] = {
                    "block_mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
                    "event_auc": float(roc_auc_score(y, scores[arm][scored])),
                }
            per_scene[name][scene] = record

            if budget is None and not args.skip_anchor and scene in published:
                was = published[scene][ANCHOR_ARM]["block_mean_auc"]
                drift = abs(was - record["support"]["block_mean_auc"])
                anchor_worst = max(anchor_worst, drift)
                anchor_checked += 1

        primary = per_scene[f"t_{T_STAR_US}us"].get(scene, {})
        base = per_scene["fixed_n_10240"].get(scene, {})
        print(f"[{index:3d}/{len(scenes)}] {scene:14s} "
              f"N=10240: auc={base.get('support', {}).get('block_mean_auc', float('nan')):.4f} "
              f"occ={base.get('mean_support', float('nan')):.2f} | "
              f"T*={T_STAR_US/1000:.1f}ms: "
              f"auc={primary.get('support', {}).get('block_mean_auc', float('nan')):.4f} "
              f"occ={primary.get('mean_support', float('nan')):.2f} "
              f"n={primary.get('events_per_window_mean', float('nan')):.0f} | "
              f"ebf={primary.get('ebf', {}).get('block_mean_auc', float('nan')):.4f} "
              f"[{time.perf_counter() - step:.0f}s]", flush=True)

    if not args.skip_anchor:
        if anchor_checked < 1:
            raise SystemExit("the fixed-N anchor checked no scene; refusing to write")
        if anchor_worst > ANCHOR_TOLERANCE:
            raise SystemExit(f"the fixed-N partition does not reproduce {ANCHOR_ARTEFACT}: "
                             f"worst per-scene block-mean AUC drift {anchor_worst:.2e} "
                             f"exceeds {ANCHOR_TOLERANCE:.0e}")
        print(f"anchor: fixed-N reproduces {ANCHOR_ARTEFACT} on {anchor_checked} scenes, "
              f"worst drift {anchor_worst:.2e}")

    summary: Dict[str, Dict] = {}
    contrasts: Dict[str, Dict] = {}
    for name, budget in partitions:
        scenes_here = per_scene[name]
        if not scenes_here:
            continue

        def pooled(arm: str, field: str) -> float:
            vals = [r[arm][field] for r in scenes_here.values()
                    if not np.isnan(r[arm][field])]
            return float(np.mean(vals)) if vals else float("nan")

        block = {arm: {f: pooled(arm, f) for f in ("block_mean_auc", "event_auc")}
                 for arm in (*ARMS, "ebf")}
        sizes = np.array([r["events_per_window_mean"] for r in scenes_here.values()])
        maxima = np.array([r["events_per_window_max"] for r in scenes_here.values()])
        summary[name] = {
            "budget_us": budget,
            "scenes": len(scenes_here),
            "windows": int(sum(r["windows"] for r in scenes_here.values())),
            "events": int(sum(r["events"] for r in scenes_here.values())),
            "mean_support": float(np.mean([r["mean_support"] for r in scenes_here.values()])),
            "fraction_support_zero": float(np.mean(
                [r["fraction_support_zero"] for r in scenes_here.values()])),
            "fraction_support_saturated": float(np.mean(
                [r["fraction_support_saturated"] for r in scenes_here.values()])),
            "span_us_median": float(np.median(
                [r["span_us_median"] for r in scenes_here.values()])),
            "events_per_window_mean": float(sizes.mean()),
            "events_per_window_max": int(maxima.max()),
            "buffer_bytes_mean": float(sizes.mean() * PACKED_EVENT_BYTES),
            "buffer_bytes_max": int(maxima.max() * PACKED_EVENT_BYTES),
            **block,
        }

        pairs = {}
        for left, right in (("support", "ebf"), ("support_causal", "ebf"),
                            ("support", "magnitude"), ("support", "support_causal")):
            clusters = [(s, paired(per_window[name][left][s], per_window[name][right][s]))
                        for s in scenes_here]
            pairs[f"{left}_minus_{right}"] = {
                **cluster_bootstrap(clusters, args.replicates, rng),
                **sign_flip(clusters, args.flip_draws, rng),
            }
        contrasts[name] = pairs

    payload = {
        "phase": "66 time-budget block partition on LED",
        "question": ("Section IV-E attributes the LED deficit to a block fixed in events. "
                     "Repartitioning by time is the intervention that claim implies."),
        "pre_declared_budget_us": T_STAR_US,
        "pre_declared_source": ("phase60_occupancy_regime.json -> "
                                "ebf_comparison.development_block_span_ms_median"),
        "grid_us": list(args.budgets),
        "min_events_per_window": MIN_EVENTS,
        "beta": 0,
        "frozen": {"tau_us": TAU_US, "score_radius": SCORE_RADIUS,
                   "sensor": [WIDTH, HEIGHT], "ebf_tau_us": EBF_TAU_US},
        "anchor": {"artefact": ANCHOR_ARTEFACT, "arm": ANCHOR_ARM,
                   "scenes_checked": anchor_checked, "worst_drift": anchor_worst,
                   "tolerance": ANCHOR_TOLERANCE},
        "packed_event_bytes": PACKED_EVENT_BYTES,
        "summary": summary,
        "contrasts": contrasts,
        "per_scene": per_scene,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.output}")
    for name in summary:
        s = summary[name]
        print(f"  {name:16s} occ={s['mean_support']:5.2f} "
              f"ours={s['support']['block_mean_auc']:.4f} "
              f"causal={s['support_causal']['block_mean_auc']:.4f} "
              f"ebf={s['ebf']['block_mean_auc']:.4f} "
              f"buf={s['buffer_bytes_mean']/1024:8.1f}kB "
              f"windows={s['windows']}")


if __name__ == "__main__":
    main()
