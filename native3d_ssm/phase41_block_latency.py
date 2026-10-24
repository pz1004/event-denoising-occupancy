"""Phase 41: how long a 10,240-event block actually lasts, in wall-clock time.

Reviewer question this answers
------------------------------
The block-local bidirectional pass is disclosed and priced in AUC and MESR, but the
manuscript never says what the lookahead *costs in time*. Because a block holds a fixed
number of *events* rather than a fixed *duration*, its temporal span is a function of
scene activity and illumination: the same 10,240 events span milliseconds in a bright
busy scene and a substantial fraction of a second in a dark quiet one. A reader deciding
whether this filter can sit immediately after a sensor needs that distribution, not the
event count.

What this phase measures
------------------------
For every dataset and every condition, the span ``t[last] - t[first]`` of each complete
``N``-event block, reported as median / p90 / p95 / max in milliseconds, together with the
implied event rate. Blocks are cut exactly as the filter cuts them, so these are the
spans the adopted configuration actually waits for.

This is a pure measurement over the event timestamps. No score is computed, nothing is
fitted, and the numbers do not depend on any constant of the method other than ``N``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

# Imports are deliberately deferred into the measure functions. The labelled loaders pull
# in scikit-learn and the E-MLB reader pulls in dv-processing, and no single environment
# on this machine holds both: the EDformer environment carries the authors' dv-processing
# pin and nothing else may be installed into it. Deferring lets one script serve both.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK = 10_240
PERCENTILES = (50, 90, 95, 99)


def span_statistics(spans_us: List[float], block: int) -> Dict[str, float]:
    """Summarise one condition's block spans in milliseconds."""

    values = np.asarray([s for s in spans_us if np.isfinite(s)], dtype=np.float64)
    if values.size == 0:
        return {"blocks": 0}
    report: Dict[str, float] = {"blocks": int(values.size)}
    for percentile in PERCENTILES:
        report[f"p{percentile}_ms"] = float(np.percentile(values, percentile) / 1000.0)
    report["max_ms"] = float(values.max() / 1000.0)
    report["mean_ms"] = float(values.mean() / 1000.0)
    # Median event rate implied by the median span, in events per second.
    median_us = float(np.median(values))
    report["median_event_rate_hz"] = float(block / median_us * 1e6) if median_us > 0 else float("inf")
    return report


def spans_from_blocks(block_iter: Iterator, block: int,
                      max_blocks: Optional[int]) -> List[float]:
    spans: List[float] = []
    for index, item in enumerate(block_iter):
        if max_blocks is not None and index >= max_blocks:
            break
        events = item[0] if isinstance(item, tuple) else item
        timestamps = np.asarray(events)[:, 0].astype(np.float64)
        if len(timestamps) < block:
            continue
        spans.append(float(timestamps[-1] - timestamps[0]))
    return spans


def spans_from_stream(timestamps: np.ndarray, block: int) -> List[float]:
    """Cut a whole recording into consecutive complete blocks, as the filter does."""

    t = np.asarray(timestamps, dtype=np.float64)
    usable = (len(t) // block) * block
    if usable < block:
        return []
    cut = t[:usable].reshape(-1, block)
    return list(cut[:, -1] - cut[:, 0])


def measure_dnd21(root: Path, block: int, max_blocks: Optional[int]) -> Dict[str, Dict]:
    from .evaluate_adaptive_beta import dnd21_blocks
    from .evaluate_dnd21 import PAPER_AUC as DND21_RATES
    from .evaluate_dnd21 import SCENE_FILES

    out: Dict[str, Dict] = {}
    for rate in DND21_RATES:
        for scene, filename in SCENE_FILES.items():
            path = root / rate / filename
            if not path.exists():
                continue
            condition = f"{rate}_{scene}"
            spans = spans_from_blocks(dnd21_blocks(path, block, max_blocks), block, max_blocks)
            out[condition] = span_statistics(spans, block)
            print(json.dumps({"dataset": "DND21", "condition": condition, **out[condition]}),
                  flush=True)
    return out


def measure_dvsclean(root: Path, block: int, max_blocks: Optional[int]) -> Dict[str, Dict]:
    from .evaluate_adaptive_beta import dvsclean_blocks
    from .evaluate_dvsclean_analytic import SEQUENCES

    out: Dict[str, Dict] = {}
    for sequence in SEQUENCES:
        for level in (50, 100):
            condition = f"{sequence}_{level}"
            path = root / f"{condition}.hdf5"
            if not path.exists():
                continue
            spans = spans_from_blocks(dvsclean_blocks(path, block, max_blocks), block, max_blocks)
            out[condition] = span_statistics(spans, block)
            print(json.dumps({"dataset": "DVSCLEAN", "condition": condition, **out[condition]}),
                  flush=True)
    return out


def measure_ed24(block: int, scenes: int, levels: List[str],
                 max_blocks: Optional[int]) -> Dict[str, Dict]:
    from .phase4_ed24_blind import ED24_ROOT, ed24_blocks

    out: Dict[str, Dict] = {}
    if not ED24_ROOT.exists():
        return out
    names = sorted(p.name for p in ED24_ROOT.iterdir() if p.is_dir())[:scenes]
    for level in levels:
        spans: List[float] = []
        for scene in names:
            spans.extend(
                spans_from_blocks(ed24_blocks(scene, level, block, max_blocks),
                                  block, max_blocks)
            )
        if spans:
            out[level] = span_statistics(spans, block)
            print(json.dumps({"dataset": "ED24", "condition": level, **out[level]}),
                  flush=True)
    return out


def measure_emlb(root: Path, block: int, reps: List[int],
                 max_scenes: Optional[int]) -> Dict[str, Dict]:
    from .phase5_emlb_mesr import ND_DISPLAY, ND_LEVELS, PARTS, read_aedat4

    out: Dict[str, Dict] = {}
    for part, part_name in PARTS.items():
        part_root = root / part
        if not part_root.exists():
            continue
        scenes = sorted(d.name for d in part_root.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
        if max_scenes:
            scenes = scenes[:max_scenes]
        for nd in ND_LEVELS:
            spans: List[float] = []
            for scene in scenes:
                for rep in reps:
                    path = part_root / scene / f"{scene}-{nd}-{rep}.aedat4"
                    if not path.exists():
                        continue
                    events = read_aedat4(path, None)
                    if events is None or len(events) < block:
                        continue
                    spans.extend(
                        spans_from_stream(events["timestamp"].astype(np.float64), block)
                    )
            if spans:
                cell = f"{part_name} {ND_DISPLAY[nd]}"
                out[cell] = span_statistics(spans, block)
                print(json.dumps({"dataset": "E-MLB", "condition": cell, **out[cell]}),
                      flush=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, default=BLOCK)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--ed24-scenes", type=int, default=20)
    parser.add_argument("--ed24-levels", nargs="*", default=None,
                        help="default is phase4's ED24_LEVELS, resolved lazily")
    parser.add_argument("--emlb-reps", nargs="*", type=int, default=[1])
    parser.add_argument("--emlb-max-scenes", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--merge", action="store_true",
                        help="merge into an existing output file instead of replacing it")
    parser.add_argument("--datasets", nargs="*",
                        default=["DND21", "DVSCLEAN", "ED24", "E-MLB"])
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT
                        / "native3d_ssm/experiments/analytic_contrast/phase41_block_latency.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    results: Dict[str, Dict] = {}
    if "DND21" in args.datasets:
        results["DND21"] = measure_dnd21(args.dnd21_root, args.block, args.max_blocks)
    if "DVSCLEAN" in args.datasets:
        results["DVSCLEAN"] = measure_dvsclean(args.dvsclean_root, args.block, args.max_blocks)
    if "ED24" in args.datasets:
        levels = args.ed24_levels
        if levels is None:
            from .phase4_ed24_blind import ED24_LEVELS

            levels = list(ED24_LEVELS)
        results["ED24"] = measure_ed24(args.block, args.ed24_scenes, levels, args.max_blocks)
    if "E-MLB" in args.datasets:
        results["E-MLB"] = measure_emlb(args.emlb_root, args.block, args.emlb_reps,
                                        args.emlb_max_scenes)

    pooled: Dict[str, Dict] = {}
    for dataset, conditions in results.items():
        usable = [c for c in conditions.values() if c.get("blocks")]
        if not usable:
            continue
        weights = np.asarray([c["blocks"] for c in usable], dtype=np.float64)
        pooled[dataset] = {
            "conditions": len(usable),
            "blocks": int(weights.sum()),
            "median_of_condition_p50_ms": float(np.median([c["p50_ms"] for c in usable])),
            "worst_condition_p95_ms": float(max(c["p95_ms"] for c in usable)),
            "worst_condition_max_ms": float(max(c["max_ms"] for c in usable)),
            "best_condition_p50_ms": float(min(c["p50_ms"] for c in usable)),
            "worst_condition_p50_ms": float(max(c["p50_ms"] for c in usable)),
        }

    payload = {
        "phase": "41 wall-clock temporal span of one event block",
        "question": ("A block holds a fixed number of events, so the lookahead latency it "
                     "implies varies with scene activity. What is that distribution?"),
        "block_size": args.block,
        "note": ("Spans are measured over complete blocks only. The adopted configuration "
                 "waits one block before emitting, so p95 is the latency figure a "
                 "deployment budget should use, not the median."),
        "per_condition": results,
        "pooled": pooled,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.merge and args.output.exists():
        previous = json.loads(args.output.read_text())
        merged_conditions = previous.get("per_condition", {})
        merged_conditions.update(payload["per_condition"])
        merged_pooled = previous.get("pooled", {})
        merged_pooled.update(payload["pooled"])
        payload["per_condition"] = merged_conditions
        payload["pooled"] = merged_pooled
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "complete", "pooled": pooled}, indent=2), flush=True)


if __name__ == "__main__":
    main()
