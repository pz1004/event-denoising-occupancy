"""Phase 70: the amplitude isolation of Phase 69, on LED, under both partitions.

Phase 69 asks what discarding amplitude is worth once the magnitude control is given the
count's own horizon, and answers it on the development conditions and BUAA_Campus. LED is
the benchmark that decides whether that answer is a property of injected noise, and it is
also the one where the partition itself is in question, so it gets both partitions:

    fixed_n_10240   the frozen block, where Section IV-E measures a nearly empty ring
    t_24302us       the pre-declared time budget of Phase 66

Same corrected raw/single-log arms and exact integer flat controls as Phase 69.
Finite-decay states use independent per-tau age caps; flat does not use a decay
approximation. The only change between partitions is how the stream is cut.
The final artifact checks occupancy AUC against every legacy scene/partition.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .matched_horizon import block_scores, SCORE_DEFINITIONS
from .phase54_led import ARTIFACTS, BLOCK, HEIGHT, LED_ROOT, WIDTH, iter_scene
from .phase66_time_budget import T_STAR_US, count_windows, time_windows
from .phase69_operand_isolation import (ARMS, SCORE_RADIUS,
                                        TAU_NAMES, TAUS_US, archive_legacy, write_json,
                                        SCORE_SIGNATURE, runtime_metadata)


def score_windows(events: np.ndarray, windows: List[Tuple[int, int]], width: int,
                  height: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Every arm over the given windows. Events outside every window stay NaN."""

    out = {arm: np.full(len(events), np.nan, dtype=np.float64) for arm in ARMS}
    support_all = np.full(len(events), np.nan, dtype=np.float64)
    for start, stop in windows:
        values, support = block_scores(events[start:stop], width, height, SCORE_RADIUS)
        for arm in ARMS:
            out[arm][start:stop] = values[arm]
        support_all[start:stop] = support
    return out, support_all


def window_aucs(scores: np.ndarray, labels: np.ndarray,
                windows: List[Tuple[int, int]]) -> List[float]:
    out: List[float] = []
    for start, stop in windows:
        y = labels[start:stop]
        if y.min() == y.max():
            continue
        out.append(float(roc_auc_score(y, scores[start:stop])))
    return out


def measure_scene(root: Path, scene: str, partitions) -> Tuple[str, Dict, float]:
    started = time.perf_counter()
    events, labels, _, _, _ = iter_scene(root, scene)
    records = {}
    if len(events) < BLOCK:
        return scene, records, time.perf_counter() - started
    for name, budget in partitions:
        windows = (count_windows(len(events), BLOCK) if budget is None
                   else time_windows(events[:, 0], budget))
        if not windows:
            continue
        scores, support = score_windows(events, windows, WIDTH, HEIGHT)
        scored = ~np.isnan(scores["count"])
        y = labels[scored]
        if y.min() == y.max():
            continue
        record = {"windows": len(windows), "events": int(scored.sum()),
                  "noise_fraction": float(y.mean()),
                  "mean_support": float(np.nanmean(support))}
        for arm in ARMS:
            aucs = window_aucs(scores[arm], labels, windows)
            record[arm] = {
                "block_mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
                "event_auc": float(roc_auc_score(y, scores[arm][scored])),
                "blocks": len(aucs)}
        records[name] = record
    return scene, records, time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LED_ROOT)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume schema-2 scene checkpoints")
    parser.add_argument("--workers", type=int, default=1, help="Independent scene worker processes")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase70_operand_isolation_led.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    scenes = sorted(p.name for p in (args.root / "raw_event").iterdir() if p.is_dir())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    partitions: List[Tuple[str, Optional[int]]] = [("fixed_n_10240", None),
                                                   (f"t_{T_STAR_US}us", T_STAR_US)]
    archive_legacy(args.output)
    per_scene: Dict[str, Dict[str, Dict]] = {name: {} for name, _ in partitions}
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("schema_version") != 2 or previous.get("score_signature") != SCORE_SIGNATURE:
            raise ValueError("cannot resume legacy score definitions")
        per_scene = previous.get("per_scene", per_scene)

    def checkpoint():
        write_json(args.output, {"schema_version": 2, "complete": False,
                                "score_signature": SCORE_SIGNATURE,
                                "score_definitions": SCORE_DEFINITIONS,
                                "per_scene": per_scene})

    pending = [scene for scene in scenes
               if not all(scene in per_scene[name] for name, _ in partitions)]
    resumed_scenes = len(scenes) - len(pending)
    if args.workers < 1:
        raise ValueError("workers must be positive")

    def accept(result):
        scene, records, elapsed = result
        for name, record in records.items():
            per_scene[name][scene] = record
        checkpoint()
        base = per_scene["fixed_n_10240"].get(scene, {})
        prim = per_scene[f"t_{T_STAR_US}us"].get(scene, {})
        completed = sum(all(s in per_scene[name] for name, _ in partitions) for s in scenes)
        print(f"[{completed:3d}/{len(scenes)}] {scene:14s} "
              f"N: count={base.get('count', {}).get('block_mean_auc', float('nan')):.4f} "
              f"flat={base.get('sum_flat', {}).get('block_mean_auc', float('nan')):.4f} | "
              f"T*: count={prim.get('count', {}).get('block_mean_auc', float('nan')):.4f} "
              f"flat={prim.get('sum_flat', {}).get('block_mean_auc', float('nan')):.4f} "
              f"250us={prim.get('logdir_250us', {}).get('block_mean_auc', float('nan')):.4f} "
              f"[{elapsed:.0f}s]", flush=True)

    if args.workers == 1:
        for scene in pending:
            accept(measure_scene(args.root, scene, partitions))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(measure_scene, args.root, scene, partitions)
                       for scene in pending]
            for future in as_completed(futures):
                accept(future.result())

    summary: Dict[str, Dict] = {}
    for name, budget in partitions:
        units = list(per_scene[name].values())
        if not units:
            continue
        entry = {"budget_us": budget, "scenes": len(units),
                 "mean_support": float(np.mean([u["mean_support"] for u in units]))}
        for arm in ARMS:
            entry[arm] = {f: float(np.mean([u[arm][f] for u in units]))
                          for f in ("block_mean_auc", "event_auc")}
        entry["count_beats_flat_scenes"] = int(sum(
            u["count"]["block_mean_auc"] > u["sum_flat"]["block_mean_auc"] for u in units))
        entry["count_beats_250us_scenes"] = int(sum(
            u["count"]["block_mean_auc"] > u["sum_250us"]["block_mean_auc"] for u in units))
        summary[name] = entry

    payload = {
        "schema_version": 2, "complete": True,
        "worker_processes": args.workers, "resumed_complete_scenes": resumed_scenes,
        "wall_seconds_scope": "current invocation, excluding earlier checkpointed work",
        **runtime_metadata(),
        "full_benchmark_run": args.max_scenes is None and all(len(records) == 66 for records in per_scene.values()),
        "score_definitions": SCORE_DEFINITIONS,
        "flat_control": "exact integer counts; no decay approximation",
        "nearflat_tau_us": 1e9,
        "phase": "70 operand isolation on LED, under both partitions",
        "taus_us": list(TAUS_US),
        "tau_names": list(TAU_NAMES),
        "matched_control": "sum_flat",
        "pre_declared_budget_us": T_STAR_US,
        "summary": summary,
        "per_scene": per_scene,
        "wall_seconds": time.perf_counter() - started,
    }
    legacy_path = args.output.with_name(args.output.stem + ".legacy_mislabelled.json")
    if legacy_path.exists():
        old = json.loads(legacy_path.read_text())["per_scene"]
        drift = [abs(record["count"]["block_mean_auc"] - old[name][scene]["count"]["block_mean_auc"])
                 for name, records in per_scene.items() for scene, record in records.items()
                 if scene in old.get(name, {})]
        if drift and max(drift) > 1e-12:
            raise AssertionError("occupancy differs from independent legacy extractor")
        payload["legacy_anchor_comparison"] = {"partition_scene_pairs_checked": len(drift),
            "max_occupancy_auc_drift": max(drift, default=0)}
    write_json(args.output, payload)
    print(f"\nwrote {args.output}")
    for name, s in summary.items():
        row = " ".join(f"{a}={s[a]['block_mean_auc']:.4f}" for a in ARMS)
        print(f"  {name:16s} occ={s['mean_support']:5.2f} {row} "
              f"| count>flat on {s['count_beats_flat_scenes']}/{s['scenes']}")


if __name__ == "__main__":
    main()
