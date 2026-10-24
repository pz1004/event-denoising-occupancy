"""Fair fixed-block occupancy versus exact event-multiplicity filtering.

Both Numba implementations consume the same int64 [t,x,y,p] events, use the
same fixed retention, break score ties by arrival index, and emit events in
arrival order. The complete path includes a block copy, map construction,
scoring, histogram selection and output copy. File I/O and compilation are
excluded. No adaptive retention, decay, logarithms or timestamps in the maps.

Quality is replayed on every LED scene. Timing uses a predetermined prefix of
one DND21 condition and the lexicographically first LED scene, with warmups and
alternating paired repeats. Run --stage quality while other experiments run;
run --stage timing only after they finish to avoid CPU contention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import llvmlite
import numba
import numpy as np
from numba import njit
from sklearn.metrics import roc_auc_score

from .phase54_led import ARTIFACTS, LED_ROOT, iter_scene

BLOCK = 10_240
RETENTION = 0.28
ARMS = ("occupancy", "multiplicity")
POPCOUNT = np.array([int(i).bit_count() for i in range(32)], dtype=np.uint8)


@njit(cache=True)
def occupancy_scores(events, width, height, radius=2):
    """Packed two-plane occupancy, five/three row reads per radius-2/1 event."""
    words = (width + 2 * radius + 63) // 64 + 1
    planes = np.zeros((2, height + 2 * radius, words), np.uint64)
    for i in range(len(events)):
        x, y, p = events[i, 1] + radius, events[i, 2] + radius, events[i, 3]
        planes[p, y, x >> 6] |= np.uint64(1) << np.uint64(x & 63)
    scores = np.empty(len(events), np.uint8)
    mask = np.uint64((1 << (2 * radius + 1)) - 1)
    for i in range(len(events)):
        x, y, p = events[i, 1], events[i, 2], events[i, 3]
        word, off = x >> 6, x & 63
        total = 0
        for dy in range(2 * radius + 1):
            window = planes[p, y + dy, word] >> np.uint64(off)
            if off != 0:
                window |= planes[p, y + dy, word + 1] << np.uint64(64 - off)
            window &= mask
            total += POPCOUNT[window]
            if dy == radius:
                total -= int((window >> np.uint64(radius)) & np.uint64(1))
        scores[i] = total
    return scores


@njit(cache=True)
def multiplicity_scores(events, width, height, radius=2):
    """Exact flat ring event counts; uint16 is sufficient for N <= 65535."""
    if len(events) > 65535:
        raise ValueError("uint16 implementation requires N <= 65535")
    counts = np.zeros((2, height + 2 * radius, width + 2 * radius), np.uint16)
    for i in range(len(events)):
        counts[events[i, 3], events[i, 2] + radius, events[i, 1] + radius] += 1
    scores = np.empty(len(events), np.uint16)
    for i in range(len(events)):
        x, y, p = events[i, 1], events[i, 2], events[i, 3]
        total = 0
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                if dx != radius or dy != radius:
                    total += int(counts[p, y + dy, x + dx])
        scores[i] = total
    return scores


@njit(cache=True)
def select_emit(events, scores, maximum, retention):
    """Stable highest-score retention with arrival-order output, without sorting."""
    keep_n = int(math.floor(retention * len(scores)))
    histogram = np.zeros(maximum + 1, np.int32)
    observed_max = 0
    for score in scores:
        histogram[score] += 1
        observed_max = max(observed_max, int(score))
    threshold = observed_max
    above = 0
    while threshold > 0 and above + histogram[threshold] < keep_n:
        above += histogram[threshold]
        threshold -= 1
    ties_left = keep_n - above
    emitted = np.empty((keep_n, 4), np.int64)
    indices = np.empty(keep_n, np.int32)
    count = 0
    for i in range(len(scores)):
        take = scores[i] > threshold
        if scores[i] == threshold and ties_left > 0:
            take = True
            ties_left -= 1
        if take:
            emitted[count] = events[i]
            indices[count] = i
            count += 1
    return emitted, indices


@njit(cache=True)
def filter_occupancy(events, width, height, radius=2, retention=RETENTION):
    buffered = events.copy()
    scores = occupancy_scores(buffered, width, height, radius)
    emitted, indices = select_emit(buffered, scores, (2 * radius + 1) ** 2 - 1, retention)
    return scores, emitted, indices


@njit(cache=True)
def filter_multiplicity(events, width, height, radius=2, retention=RETENTION):
    buffered = events.copy()
    scores = multiplicity_scores(buffered, width, height, radius)
    emitted, indices = select_emit(buffered, scores, max(0, len(events) - 1), retention)
    return scores, emitted, indices


def working_storage(width, height, radius=2, block=BLOCK, retention=RETENTION):
    """Conservative sum of array capacities, not process RSS or allocator peak.

    Packed design uses the same byte-rounded (32-bit time offset, x,y,p)
    buffer and output for both arms. It omits diagnostic retained indices,
    whereas implemented-array accounting includes them. State padding is held
    fixed. Real code has non-overlapping map/histogram lifetimes, so their sum
    is a conservative upper bound. Full source streams are evaluation data,
    not algorithm working state. External acquisition/output queues excluded.
    """
    n_keep = math.floor(retention * block)
    event_bits = 32 + math.ceil(math.log2(width)) + math.ceil(math.log2(height)) + 1
    event_bytes = math.ceil(event_bits / 8)
    occupancy_max = (2 * radius + 1) ** 2 - 1
    occupancy_bits = occupancy_max.bit_length()
    multiplicity_bits = (block - 1).bit_length()
    count_map_bits = block.bit_length()  # all N events may fire at one pixel
    planes = 2 * (height + 2 * radius) * ((width + 2 * radius + 63) // 64 + 1) * 8
    count_cells = 2 * (height + 2 * radius) * (width + 2 * radius)
    result = {"width": width, "height": height, "block": block, "radius": radius,
              "retention": retention, "kept_events": n_keep,
              "common_packed_event_bytes": event_bytes,
              "scope": "Conservative sum of working-array capacities; not measured RSS; input/output queues excluded."}
    for arm in ARMS:
        occupancy = arm == "occupancy"
        maximum = occupancy_max if occupancy else block - 1
        bits = occupancy_bits if occupancy else multiplicity_bits
        actual = {"operand_bytes": planes if occupancy else count_cells * 2,
                  "readonly_popcount_table_bytes": 32 if occupancy else 0,
                  "input_buffer_bytes": block * 32,
                  "score_bytes": block * (1 if occupancy else 2),
                  "selection_histogram_bytes": (maximum + 1) * 4,
                  "output_events_bytes": n_keep * 32,
                  "diagnostic_output_indices_bytes": n_keep * 4}
        packed = {"operand_bytes": planes if occupancy else math.ceil(count_cells * count_map_bits / 8),
                  "readonly_popcount_table_bytes": 32 if occupancy else 0,
                  "input_buffer_bytes": block * event_bytes,
                  "score_bytes": math.ceil(block * bits / 8),
                  "selection_histogram_bytes": (maximum + 1) * 4,
                  "output_events_bytes": n_keep * event_bytes}
        actual["total_bytes"] = sum(actual.values())
        packed["total_bytes"] = sum(packed.values())
        result[arm] = {"score_bits": bits, "map_counter_bits": 1 if occupancy else count_map_bits,
                       "implemented_arrays": actual, "packed_design": packed}
    return result


def environment():
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    return {"cpu": cpu, "platform": platform.platform(), "python": sys.version,
            "numpy": np.__version__, "numba": numba.__version__, "llvmlite": llvmlite.__version__,
            "logical_cpus": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "kernel_threads": 1, "parallel_numba": False,
            "fastmath": False, "compiler": "Numba nopython via LLVM", "timing_clock": "time.perf_counter"}


def write_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(a)), "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "min": float(np.min(a)), "max": float(np.max(a))}


def quality_scene(events, labels, width, height):
    n = len(events) // BLOCK * BLOCK
    y = labels[:n]
    scores = {a: np.empty(n, np.uint16) for a in ARMS}
    kept = {a: np.zeros(n, bool) for a in ARMS}
    waits = np.empty(n, np.int64)
    spans = []
    for start in range(0, n, BLOCK):
        b = events[start:start + BLOCK]
        waits[start:start + BLOCK] = b[-1, 0] - b[:, 0]
        spans.append(int(b[-1, 0] - b[0, 0]))
        for arm, fn in (("occupancy", filter_occupancy), ("multiplicity", filter_multiplicity)):
            s, emitted, indices = fn(b, width, height)
            if not np.array_equal(emitted, b[indices]):
                raise AssertionError("emission changed event identity/order")
            scores[arm][start:start + BLOCK] = s
            kept[arm][start + indices] = True
    result = {"events": n, "blocks": n // BLOCK, "discarded_tail_events": len(events) - n,
              "noise_fraction": float(y.mean()), "acquisition_to_block_close_us": distribution(waits),
              "block_span_us": distribution(spans)}
    for arm in ARMS:
        k = kept[arm]
        signal_recall = float(k[y == 0].mean())
        noise_rejection = float((~k[y == 1]).mean())
        aucs = [roc_auc_score(y[s:s + BLOCK], -scores[arm][s:s + BLOCK].astype(np.float64))
                for s in range(0, n, BLOCK) if y[s:s + BLOCK].min() != y[s:s + BLOCK].max()]
        result[arm] = {"signal_recall": signal_recall, "noise_rejection": noise_rejection,
                       "balanced_accuracy": .5 * (signal_recall + noise_rejection),
                       "retained_fraction": float(k.mean()),
                       "event_auc": float(roc_auc_score(y, -scores[arm].astype(np.float64))),
                       "block_mean_auc": float(np.mean(aucs)),
                       "maximum_score": int(scores[arm].max())}
    return result


def quality(args, payload):
    scenes = sorted(p.name for p in (args.led_root / "raw_event").iterdir() if p.is_dir())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    per_scene = payload.setdefault("quality", {}).setdefault("per_scene", {})
    for scene in scenes:
        if scene in per_scene:
            continue
        events, labels, ambiguous, skipped, reordered = iter_scene(args.led_root, scene)
        record = quality_scene(events, labels, 1280, 720)
        record.update(ambiguous_labels=ambiguous, skipped_files=skipped, reordered=reordered)
        per_scene[scene] = record
        write_payload(args.output, payload)
        print(f"quality {scene}: events={record['events']} BA=" +
              "/".join(f"{record[a]['balanced_accuracy']:.5f}" for a in ARMS), flush=True)
    records = list(per_scene.values())
    anchor_path = ARTIFACTS / "phase54_led.json"
    anchors = json.loads(anchor_path.read_text())["per_scene"]
    anchor_fields = ("signal_recall", "noise_rejection", "balanced_accuracy",
                     "retained_fraction", "event_auc", "block_mean_auc")
    max_error = max(abs(record["occupancy"][field] - anchors[scene]["beta0_support"][field])
                    for scene, record in per_scene.items() for field in anchor_fields)
    if max_error > 1e-12:
        raise AssertionError(f"fixed-retention occupancy differs from published path: {max_error}")
    payload["quality"]["anchor"] = {"artifact": anchor_path.name, "arm": "beta0_support",
        "fields": list(anchor_fields), "checks": len(per_scene) * len(anchor_fields),
        "max_absolute_error": max_error}
    summary = {"scenes": len(records), "events": sum(r["events"] for r in records),
               "aggregation": "Equal scene means; no independent new benchmark claim."}
    for arm in ARMS:
        summary[arm] = {key: float(np.mean([r[arm][key] for r in records]))
                        for key in ("signal_recall", "noise_rejection", "balanced_accuracy",
                                    "retained_fraction", "event_auc", "block_mean_auc")}
    summary["block_wait_scene_mean_us"] = float(np.mean([r["acquisition_to_block_close_us"]["mean"] for r in records]))
    payload["quality"]["summary"] = summary
    payload["quality"]["latency_scope"] = "Acquisition to last event of fixed-count block only; processing/emission time reported separately by timing replay. Offline input file loading excluded."
    write_payload(args.output, payload)


def timing_pool(args):
    from .evaluate_adaptive_beta import dnd21_blocks
    first_scene = sorted(p.name for p in (args.led_root / "raw_event").iterdir() if p.is_dir())[0]
    events, _, _, _, _ = iter_scene(args.led_root, first_scene)
    led = [np.ascontiguousarray(events[s:s + BLOCK])
           for s in range(0, min(len(events) // BLOCK, args.timing_blocks) * BLOCK, BLOCK)]
    dnd_path = args.dnd21_root / "1hz/mix_result.txt"
    dnd = [np.ascontiguousarray(e) for e, _ in dnd21_blocks(dnd_path, BLOCK, args.timing_blocks)]
    return {"346x260": (346, 260, dnd, "DND21/1hz_hotel-bar"),
            "1280x720": (1280, 720, led, "LED/" + first_scene)}


def timed_pass(fn, blocks, width, height, radius):
    began = time.perf_counter()
    checksum = 0
    for b in blocks:
        out = fn(b, width, height, radius)
        if isinstance(out, tuple):
            checksum += int(out[0][0]) + len(out[1])
        else:
            checksum += int(out[0])
    return time.perf_counter() - began, checksum


def emission_replay(fn, blocks, width, height):
    """Recorded arrival times plus measured block service, single-server FIFO.

    This is a replay model, not a live-camera deadline measurement. A block is
    available at its final recorded arrival; complete-filter wall time is then
    added after any previous service completes. Output transmission is excluded.
    """
    ready = 0.0
    delays, service, backlog = [], [], []
    close_times = np.array([b[-1, 0] for b in blocks], dtype=np.int64)
    arrival_times = np.concatenate([b[:, 0] for b in blocks])
    queued_blocks, queued_events = [], []
    for index, b in enumerate(blocks):
        began = time.perf_counter()
        _, _, indices = fn(b, width, height, 2)
        us = (time.perf_counter() - began) * 1e6
        close = float(b[-1, 0])
        start = max(close, ready)
        ready = start + us
        delays.append(ready - b[indices, 0].astype(np.float64))
        service.append(us)
        backlog.append(start - close)
        queued_blocks.append(max(0, int(np.searchsorted(close_times, ready, side="left")) - index - 1))
        queued_events.append(max(0, int(np.searchsorted(arrival_times, ready, side="left")) - (index + 1) * BLOCK))
    packed_event_bytes = math.ceil((32 + math.ceil(math.log2(width)) + math.ceil(math.log2(height)) + 1) / 8)
    return {"scope": "Single-server FIFO replay using recorded block-close timestamps and measured complete-filter service; no file-I/O or output-transmission time; not a live deadline guarantee.",
            "acquisition_to_emission_us": distribution(np.concatenate(delays)),
            "processing_per_block_us": distribution(service),
            "queued_before_processing_us": distribution(backlog),
            "peak_additional_queued_complete_blocks": max(queued_blocks),
            "peak_additional_queued_events_including_partial_block": max(queued_events),
            "peak_additional_queue_bytes_int64_events": max(queued_events) * 32,
            "peak_additional_queue_bytes_packed_events": max(queued_events) * packed_event_bytes,
            "queue_scope": "Input events arriving during service, excluding the block currently processed; observed prefix only, no general stream bound or output queue model."}


def timing(args, payload, pools=None):
    payload["timing_environment"] = environment()
    payload["timing_protocol"] = {"warmup_passes": 2, "repeats": args.repeats,
        "order": "Alternating occupancy then multiplicity / multiplicity then occupancy each repeat",
        "scope": "In-memory fixed-block replay; complete includes block copy, map and score, histogram retention, original-order event copy and diagnostic retained indices. File I/O, JIT compilation and external queues excluded.",
        "contention": args.contention_note, "selection": "First lexicographic LED scene and DND21 1hz_hotel-bar; no performance-based selection."}
    timing_result = {}
    for key, (width, height, blocks, source) in (timing_pool(args) if pools is None else pools).items():
        event_count = len(blocks) * BLOCK
        record = {"source": source, "blocks": len(blocks), "events": event_count,
                  "input_sha256": hashlib.sha256(b"".join(b.tobytes() for b in blocks)).hexdigest()}
        for radius in (1, 2):
            rad = {}
            for scope, functions in (("score", (occupancy_scores, multiplicity_scores)),
                                     ("complete", (filter_occupancy, filter_multiplicity))):
                for fn in functions:
                    for _ in range(2):
                        timed_pass(fn, blocks, width, height, radius)
                samples = {a: [] for a in ARMS}
                for repeat in range(args.repeats):
                    indices = (0, 1) if repeat % 2 == 0 else (1, 0)
                    for idx in indices:
                        elapsed, checksum = timed_pass(functions[idx], blocks, width, height, radius)
                        if checksum <= 0:
                            raise AssertionError("unexpected empty computation")
                        samples[ARMS[idx]].append(elapsed)
                stats = {}
                for arm in ARMS:
                    seconds = np.array(samples[arm])
                    stats[arm] = {"seconds": seconds.tolist(),
                                  "throughput_mev_s": distribution(event_count / seconds / 1e6),
                                  "mean_block_processing_us": distribution(seconds / len(blocks) * 1e6)}
                stats["occupancy_speedup_median"] = float(np.median(samples["multiplicity"]) / np.median(samples["occupancy"]))
                rad[scope] = stats
            rad["storage"] = working_storage(width, height, radius)
            record[f"radius_{radius}"] = rad
            print(f"timing {key} r={radius}: " + json.dumps({scope: rad[scope]["occupancy_speedup_median"] for scope in ("score", "complete")}), flush=True)
        timing_result[key] = record
        record["emission_replay_radius_2"] = {
            arm: emission_replay(fn, blocks, width, height)
            for arm, fn in (("occupancy", filter_occupancy), ("multiplicity", filter_multiplicity))}
    payload["timing"] = timing_result
    write_payload(args.output, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("quality", "timing", "all"), default="all")
    parser.add_argument("--led-root", type=Path, default=LED_ROOT)
    parser.add_argument("--dnd21-root", type=Path, default=Path(__file__).resolve().parents[1] / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase75_integer_count_cost.json")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--timing-blocks", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--contention-note", default="Not independently audited; record concurrent workloads before using timing results.")
    args = parser.parse_args()
    payload = json.loads(args.output.read_text()) if args.output.exists() else {
        "phase": "75 exact integer count versus occupancy: matched full filter",
        "schema_version": 1, "block": BLOCK, "retention": RETENTION,
        "score_definitions": {"occupancy": "Number of occupied same-polarity radius-2 ring taps in the whole block",
                              "multiplicity": "Number of same-polarity radius-2 ring events in the whole block, including repeated firings; exact zero decay"},
        "selection": "Keep floor(r*N) largest integer scores, arrival-index ties, emit in arrival order",
        "quality_radius": 2, "working_storage": {f"{w}x{h}": working_storage(w, h) for w, h in ((346, 260), (1280, 720))}}
    payload["working_storage"] = {f"{w}x{h}": working_storage(w, h) for w, h in ((346, 260), (1280, 720))}
    if args.stage in ("quality", "all"):
        quality(args, payload)
    if args.stage in ("timing", "all"):
        timing(args, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
