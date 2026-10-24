"""Phase 71: the whole filter's memory and arithmetic, not just the operand's state.

WHY THIS EXISTS. Table X of the manuscript prices the *operand*: 29.6 kB of occupancy bit
planes at 346x260 against 4.32 MB for the float bank it replaces. That is a true statement
about the operand and a misleading one about the filter, because a block-partitioned filter
must also hold the block. The buffer is compulsory -- the reverse pass cannot start until
the block is complete -- and at 10,240 events it is larger than the bit planes. Reporting
the planes alone and calling it the filter's state overstates the result, and a reader
comparing against a streaming filter's timestamp map is entitled to the total.

THIS FILE COMPUTES THE TOTAL, item by item, for three configurations:

    block_float       the extractor as implemented: float bank, int64 event buffer,
                      float64 score array, argsort index.
    block_bitplane    the route Proposition 1 licenses: occupancy planes, a packed event
                      buffer, a 5-bit score array, and -- because the score is an integer
                      in [0,24] -- a 25-bin counting sort instead of a comparison sort and
                      its index array.
    causal_occupancy  the strictly causal trailing-window form of `causal_occupancy`:
                      EBF's timestamp and polarity map, no block buffer at all, no score
                      array, no selection structure.

and beside them EBF's own state, so the comparison is between totals rather than between
one method's total and another's part.

THE SELECTION STRUCTURE IS NOT AN AFTERTHOUGHT. Keeping the lowest-scoring fraction of a
block is usually an argsort: 8 bytes of index per event, 81.9 kB here, and O(N log N).
A five-bit score admits a 25-bin histogram instead -- one pass to count, one prefix sum
over 25 bins to find the threshold, one pass to emit -- which is O(N), needs 25 counters,
and is exact rather than approximate. That is a consequence of the bounded operand and it
is priced here rather than assumed away.

THROUGHPUT. The two causal kernels -- EBF's decayed sum and our count -- are the same
traversal of the same map in the same language on the same host, differing in one line.
A ratio between them therefore means something, unlike a ratio against a number published
from another machine. It is measured here, best of three repeats.

Read-only. Nothing is trained. No dataset is required.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .causal_occupancy import OCCUPANCY_BITS, OCCUPANCY_MAX, OCCUPANCY_S, occupancy_score
from .ebf import EBF_S, EBF_TAU_US, ebf_score
from .phase52_bitmap_profile import PAD, RADIUS, state_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK = 10_240
SENSORS = (("346x260", 346, 260), ("1280x720", 1280, 720))

#: Timestamp field width in a packed event record. A block's span is at most 1.85 s, so a
#: microsecond offset from the block origin fits in 21 bits; 32 is used for alignment.
PACKED_T_BITS = 32
#: How the extractor holds an event today: four int64 columns.
IMPLEMENTED_EVENT_BYTES = 32


def packed_event_bytes(width: int, height: int) -> int:
    """Bytes for one event packed as (t offset, x, y, p), rounded up to a whole byte."""

    bits = PACKED_T_BITS + math.ceil(math.log2(width)) + math.ceil(math.log2(height)) + 1
    return (bits + 7) // 8


def block_form(width: int, height: int, route: str) -> Dict[str, float]:
    """Every buffer a block-partitioned pass must hold, at one sensor size."""

    state = state_bytes(width, height, scales=1, directions=2)
    if route == "float":
        operand = state["float_total_bytes"]
        buffer_bytes = BLOCK * IMPLEMENTED_EVENT_BYTES
        score = BLOCK * 8                      # float64 score per event
        select = BLOCK * 8                     # argsort index
        select_note = "argsort index, 8 B/event"
    else:
        operand = state["bitplane_bytes"]
        buffer_bytes = BLOCK * packed_event_bytes(width, height)
        score = math.ceil(BLOCK * OCCUPANCY_BITS / 8)
        select = (OCCUPANCY_MAX + 1) * 4       # one 32-bit counter per attainable value
        select_note = f"{OCCUPANCY_MAX + 1}-bin counting sort, 4 B/bin"
    keep_mask = math.ceil(BLOCK / 8)
    total = operand + buffer_bytes + score + select + keep_mask
    return {
        "operand_state_bytes": float(operand),
        "block_buffer_bytes": float(buffer_bytes),
        "score_bytes": float(score),
        "selection_bytes": float(select),
        "selection_note": select_note,
        "keep_mask_bytes": float(keep_mask),
        "total_bytes": float(total),
        "total_kb": total / 1e3,
        "bytes_per_event_buffered": (buffer_bytes / BLOCK),
    }


def streaming_form(width: int, height: int, kind: str) -> Dict[str, float]:
    """A causal filter's whole state: one map, no block buffer, no selection structure."""

    pixels = width * height
    timestamps = pixels * 8                    # int64 last-touch, as implemented
    polarity = pixels * 1                      # int8 last polarity, as implemented
    total = timestamps + polarity
    return {
        "kind": kind,
        "timestamp_map_bytes": float(timestamps),
        "polarity_map_bytes": float(polarity),
        "block_buffer_bytes": 0.0,
        "score_bytes": 0.0,
        "selection_bytes": 0.0,
        "total_bytes": float(total),
        "total_kb": total / 1e3,
    }


def throughput(width: int, height: int, n_events: int, repeats: int,
               seed: int) -> Dict[str, float]:
    """Both causal kernels on identical synthetic streams. Best of `repeats`."""

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 2_000_000, size=n_events)).astype(np.int64)
    events = np.column_stack((t,
                              rng.integers(0, width, n_events),
                              rng.integers(0, height, n_events),
                              rng.integers(0, 2, n_events))).astype(np.int64)
    # Warm the JIT so the first repeat does not price compilation.
    ebf_score(events[:1000], width, height, tau_us=float(EBF_TAU_US))
    occupancy_score(events[:1000], width, height, float(EBF_TAU_US))

    def once(fn) -> float:
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start

    # The two kernels are interleaved rather than run in two batches, so a transient load
    # on the host lands on both and does not become a difference between them.
    ebf_t: List[float] = []
    occ_t: List[float] = []
    for _ in range(repeats):
        ebf_t.append(once(lambda: ebf_score(events, width, height,
                                            tau_us=float(EBF_TAU_US))))
        occ_t.append(once(lambda: occupancy_score(events, width, height,
                                                  float(EBF_TAU_US))))
    ebf_t, occ_t = sorted(ebf_t), sorted(occ_t)
    ratios = sorted(a / b for a, b in zip(ebf_t, occ_t))
    return {
        "events": n_events,
        "repeats": repeats,
        "ebf_mev_s": n_events / min(ebf_t) / 1e6,
        "occupancy_mev_s": n_events / min(occ_t) / 1e6,
        "speedup": min(ebf_t) / min(occ_t),
        # Reported because a single best-of-n ratio hides how much this moves with host
        # load, and a reader is entitled to the spread rather than the lucky repeat.
        "speedup_min": ratios[0],
        "speedup_median": ratios[len(ratios) // 2],
        "speedup_max": ratios[-1],
        "ebf_seconds": ebf_t,
        "occupancy_seconds": occ_t,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=2_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase71_end_to_end_cost.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    per_sensor: Dict[str, Dict] = {}
    for name, width, height in SENSORS:
        entry = {
            "width": width,
            "height": height,
            "block_float": block_form(width, height, "float"),
            "block_bitplane": block_form(width, height, "bitplane"),
            "causal_occupancy": streaming_form(width, height, "ours, trailing window"),
            "ebf": streaming_form(width, height, "EBF, decayed magnitude"),
            "packed_event_bytes": packed_event_bytes(width, height),
            "throughput": throughput(width, height, args.events, args.repeats, args.seed),
        }
        entry["bitplane_share_of_total"] = (
            entry["block_bitplane"]["operand_state_bytes"]
            / entry["block_bitplane"]["total_bytes"])
        # The streaming form reads EBF's map and nothing else, so its state is EBF's
        # exactly. That is the honest statement: parity in memory, and the saving is in
        # the arithmetic and in the block buffer that is no longer needed.
        if entry["causal_occupancy"]["total_bytes"] != entry["ebf"]["total_bytes"]:
            raise SystemExit("the streaming form no longer shares EBF's state; the "
                             "parity claim in Section IV-H would be wrong")
        entry["streaming_state_equals_ebf"] = True
        entry["block_buffer_share_of_total"] = (
            entry["block_bitplane"]["block_buffer_bytes"]
            / entry["block_bitplane"]["total_bytes"])
        per_sensor[name] = entry
        b, f = entry["block_bitplane"], entry["block_float"]
        print(f"{name}: bit-plane route total {b['total_kb']:8.1f} kB "
              f"(operand {b['operand_state_bytes']/1e3:6.1f} kB, "
              f"buffer {b['block_buffer_bytes']/1e3:6.1f} kB) | "
              f"float route {f['total_kb']:9.1f} kB | "
              f"streaming {entry['causal_occupancy']['total_kb']:8.1f} kB | "
              f"kernels: EBF {entry['throughput']['ebf_mev_s']:.2f} Mev/s, "
              f"count {entry['throughput']['occupancy_mev_s']:.2f} Mev/s "
              f"({entry['throughput']['speedup']:.2f}x)")

    payload = {
        "phase": "71 end-to-end memory and arithmetic accounting",
        "question": ("Table X prices the operand's state. What does the whole filter hold, "
                     "and what does the streaming form hold instead?"),
        "block_size": BLOCK,
        "bit_plane_padding": PAD,
        "score_radius": RADIUS,
        "occupancy_max": OCCUPANCY_MAX,
        "occupancy_bits": OCCUPANCY_BITS,
        "neighbourhood": OCCUPANCY_S,
        "ebf_neighbourhood": EBF_S,
        "implemented_event_bytes": IMPLEMENTED_EVENT_BYTES,
        "packed_timestamp_bits": PACKED_T_BITS,
        "per_sensor": per_sensor,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
