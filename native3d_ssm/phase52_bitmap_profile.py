"""Phase 52: implement the primitive the paper claims, and price it on this host.

THE GAP. Section 3.1 proves that the support operand is exactly whole-block same-polarity
occupancy, then adds that "the operand needs neither the decayed state nor the reverse pass
that produce it here, since two polarity occupancy bitmaps of the block suffice", and
flags that route as "an available reduction we did not implement". Every cost the paper
reports -- 177 operation-equivalents per event, 4.32 MB of persistent state at 346x260,
44.24 MB at 1280x720 -- belongs to the float extractor, not to the primitive being
claimed. A reviewer is right to say the claimed primitive and the priced implementation
are different objects. This phase closes that gap by building the bitmap path, proving it
exact, and measuring both on one machine.

TWO IMPLEMENTATIONS, both exact by construction and both checked event by event:

  bitmap_gather   one vectorised bit test per ring offset: 24 gathers at radius 2.
                  Obviously correct; the reference for the second.
  bitmap_rowmask  the hardware form. The block is held as two padded bit planes, one per
                  polarity. Each event reads a 5-bit window out of each of the five rows
                  its ring spans, population-counts the five windows, and subtracts the
                  centre bit. No decay, no timestamps, no floating point, no reverse pass.

Padding by two pixels on every side makes out-of-sensor taps read as zero without a
boundary test, which is what the float extractor does implicitly and what Phase 47's
reference count does explicitly.

WHAT IS AND IS NOT COMPARED. The two implementations are timed on the same blocks, in the
same process, on this host, through the same NumPy. That comparison is controlled. The
float path additionally produces G and the gate key K, which the bitmap path does not and
cannot: the bitmap replaces the *support-only* score s = -log(1 + S), which is the closed-
gate branch, and ranking by it is ranking by the integer S itself. The gated score still
needs the decayed state. The timing table therefore prices two ways of computing one
operand, plus the extra work the gate costs, and it is labelled that way rather than as an
end-to-end system comparison against a learned baseline.

STATE. Reported as bytes actually allocated, at both sensor sizes, for the float bank and
for the bit planes. This is the number that answers "the bidirectional persistent state
reaches 44.24 MB at 1280x720, exceeding the audited float32 weight storage of EDformer".

Read-only with respect to every existing artefact. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase14_quantizable_operand import ring_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK = 10_240
RADIUS = 2
PAD = RADIUS
WINDOW_BITS = 2 * RADIUS + 1

#: popcount of every value a 5-bit window can take.
POPCOUNT5 = np.array([bin(v).count("1") for v in range(1 << WINDOW_BITS)], dtype=np.uint8)

#: Sensor sizes the manuscript prices state at.
SENSORS = {"DND21 346x260": (346, 260), "1280x720": (1280, 720)}


def build_planes(events: np.ndarray, width: int, height: int
                 ) -> Tuple[np.ndarray, int]:
    """Two padded bit planes, one per polarity. Bit (x + PAD) of row (y + PAD)."""

    padded_w = width + 2 * PAD
    # +1 spare word so the two-word read at the right edge is always in range
    words = (padded_w + 63) // 64 + 1
    planes = np.zeros((2, height + 2 * PAD, words), dtype=np.uint64)
    x = events[:, 1].astype(np.int64) + PAD
    y = events[:, 2].astype(np.int64) + PAD
    p = events[:, 3].astype(np.int64)
    np.bitwise_or.at(planes, (p, y, x >> 6), np.uint64(1) << (x & 63).astype(np.uint64))
    return planes, words


def support_rowmask(events: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """The hardware form: five 5-bit window reads, five popcounts, minus the centre bit."""

    x = events[:, 1].astype(np.int64)          # padded start of the window is exactly x
    y = events[:, 2].astype(np.int64)
    p = events[:, 3].astype(np.int64)
    word = x >> 6
    off = (x & 63).astype(np.uint64)
    total = np.zeros(len(events), dtype=np.int64)
    for dy in range(-RADIUS, RADIUS + 1):
        row = y + PAD + dy
        w0 = planes[p, row, word]
        w1 = planes[p, row, word + 1]
        hi = np.where(off == 0, np.uint64(0), w1 << ((np.uint64(64) - off) & np.uint64(63)))
        window = ((w0 >> off) | hi) & np.uint64((1 << WINDOW_BITS) - 1)
        total += POPCOUNT5[window].astype(np.int64)
        if dy == 0:
            total -= ((window >> np.uint64(RADIUS)) & np.uint64(1)).astype(np.int64)
    return total


def support_gather(events: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """One vectorised bit test per ring offset. Independent route to the same count."""

    x = events[:, 1].astype(np.int64) + PAD
    y = events[:, 2].astype(np.int64) + PAD
    p = events[:, 3].astype(np.int64)
    total = np.zeros(len(events), dtype=np.int64)
    for dy in range(-RADIUS, RADIUS + 1):
        for dx in range(-RADIUS, RADIUS + 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            bit = (planes[p, ny, nx >> 6] >> (nx & 63).astype(np.uint64)) & np.uint64(1)
            total += bit.astype(np.int64)
    return total


def state_bytes(width: int, height: int, scales: int, directions: int) -> Dict[str, int]:
    """What each route actually allocates, at one sensor size."""

    decayed = width * height * 2 * scales * 4 * directions      # float32
    timestamps = width * height * 2 * 8 * directions            # int64
    planes_w = (width + 2 * PAD + 63) // 64 + 1
    planes = 2 * (height + 2 * PAD) * planes_w * 8              # uint64 words
    return {
        "float_decayed_bytes": decayed,
        "float_timestamp_bytes": timestamps,
        "float_total_bytes": decayed + timestamps,
        "float_total_mb": (decayed + timestamps) / 1e6,
        "bitplane_bytes": planes,
        "bitplane_kb": planes / 1e3,
        "reduction_factor": (decayed + timestamps) / planes,
    }


def iter_blocks(dnd21_root: Path, dvsclean_root: Path, limit_per_condition: int | None
                ) -> Iterator[Tuple[str, int, int, np.ndarray]]:
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            for events, _ in dnd21_blocks(dnd21_root / rate / fname, BLOCK,
                                          limit_per_condition):
                yield f"DND21/{rate}_{scene}", 346, 260, events
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            for events, _ in dvsclean_blocks(dvsclean_root / f"{name}.hdf5", BLOCK,
                                             limit_per_condition):
                yield f"DVSCLEAN/{name}", 1280, 720, events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=None,
                        help="per condition; None walks all 3,708 development blocks")
    parser.add_argument("--timing-blocks", type=int, default=200,
                        help="blocks per sensor size used for the timing table")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase52_bitmap_profile.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    score_config = {size: StateConfig(width=w, height=h, taus_us=(250.0,), radius=RADIUS)
                    for size, (w, h) in {"346x260": (346, 260),
                                         "1280x720": (1280, 720)}.items()}

    mismatch_rowmask = 0
    mismatch_gather = 0
    events_checked = 0
    blocks_checked = 0
    max_support = 0
    timing_pool: Dict[str, List[np.ndarray]] = {"346x260": [], "1280x720": []}

    for name, width, height, events in iter_blocks(args.dnd21_root, args.dvsclean_root,
                                                   args.max_blocks):
        key = f"{width}x{height}"
        config = score_config[key]
        raw = bidirectional_features(events, config)
        reference = ring_support(raw, config).astype(np.int64)

        planes, _ = build_planes(events, width, height)
        rowmask = support_rowmask(events, planes)
        gather = support_gather(events, planes)

        mismatch_rowmask += int((rowmask != reference).sum())
        mismatch_gather += int((gather != reference).sum())
        events_checked += len(events)
        blocks_checked += 1
        max_support = max(max_support, int(reference.max(initial=0)))
        if len(timing_pool[key]) < args.timing_blocks:
            timing_pool[key].append(events)

    if mismatch_rowmask or mismatch_gather:
        raise SystemExit(
            f"bitmap path is not exact: {mismatch_rowmask} rowmask and {mismatch_gather} "
            f"gather mismatches over {events_checked} events; refusing to publish a "
            "timing comparison between operands that are not the same operand")
    if max_support > (2 * RADIUS + 1) ** 2 - 1:
        raise SystemExit(f"support exceeded its 5-bit bound: {max_support}")

    timing: Dict[str, Dict] = {}
    for key, blocks in timing_pool.items():
        if not blocks:
            continue
        config = score_config[key]
        n_events = sum(len(b) for b in blocks)

        def time_it(fn) -> float:
            best = float("inf")
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - t0)
            return best

        def float_path() -> None:
            for b in blocks:
                ring_support(bidirectional_features(b, config), config)

        def bitmap_path() -> None:
            for b in blocks:
                w, h = config.width, config.height
                support_rowmask(b, build_planes(b, w, h)[0])

        def extractor_only() -> None:
            for b in blocks:
                bidirectional_features(b, config)

        t_float = time_it(float_path)
        t_bitmap = time_it(bitmap_path)
        t_extract = time_it(extractor_only)
        timing[key] = {
            "blocks": len(blocks),
            "events": n_events,
            "float_path_seconds": t_float,
            "float_path_events_per_second": n_events / t_float,
            "extractor_only_seconds": t_extract,
            "bitmap_path_seconds": t_bitmap,
            "bitmap_path_events_per_second": n_events / t_bitmap,
            "speedup_bitmap_over_float": t_float / t_bitmap,
        }

    memory = {label: state_bytes(w, h, scales=1, directions=2)
              for label, (w, h) in SENSORS.items()}

    payload = {
        "phase": "52 the occupancy-bitmap primitive, implemented, proved exact, and timed",
        "question": ("Section 3.1 claims two bitmaps suffice for the support operand but "
                     "prices the float extractor instead. Build the bitmap path, check it "
                     "event by event, and measure both on one host."),
        "scope": ("The bitmap replaces the closed-gate score s = -log(1 + S), whose "
                  "ranking is the ranking of the integer S. The gated score still needs "
                  "the decayed state for G and K; the gate cost is reported separately as "
                  "the extractor-only row."),
        "exactness": {
            "blocks": blocks_checked,
            "events": events_checked,
            "mismatches_rowmask_vs_extractor": mismatch_rowmask,
            "mismatches_gather_vs_extractor": mismatch_gather,
            "max_support_observed": max_support,
            "support_upper_bound": (2 * RADIUS + 1) ** 2 - 1,
            "bits_required": int(np.ceil(np.log2((2 * RADIUS + 1) ** 2))),
        },
        "timing": {
            "host_note": "same process, same NumPy, same input blocks; best of "
                         f"{args.repeats} repeats",
            "numpy_version": np.__version__,
            "timing_blocks_per_sensor": args.timing_blocks,
            "by_sensor": timing,
        },
        "persistent_state": memory,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"exact on {events_checked:,} events in {blocks_checked:,} blocks; "
          f"max support {max_support} (bound {(2 * RADIUS + 1) ** 2 - 1})")
    for key, t in timing.items():
        print(f"{key:>9}  float {t['float_path_events_per_second']:>12,.0f} ev/s   "
              f"bitmap {t['bitmap_path_events_per_second']:>12,.0f} ev/s   "
              f"{t['speedup_bitmap_over_float']:.1f}x")
    for label, m in memory.items():
        print(f"{label:>14}  float {m['float_total_mb']:8.2f} MB   "
              f"bitplanes {m['bitplane_kb']:8.1f} kB   "
              f"{m['reduction_factor']:.0f}x")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
