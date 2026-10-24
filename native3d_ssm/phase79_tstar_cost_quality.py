"""Complete full-stream T* filtering under the frozen phase79 protocol.

The reusable filter accepts only integer [t_us,x,y,p] events and never labels.
Every nonempty first-event-anchored window is scored, including its final tail.
EBF state carries across windows; all methods use the same exact rational quota
and stable arrival-order ties. The window iterator emits copied payloads without
retaining an entire output stream. ``filter_stream`` is a convenience collector,
whose concatenated outputs and optional diagnostic scores are harness storage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from numba import njit
from sklearn.metrics import roc_auc_score

from .phase75_integer_count_cost import occupancy_scores, distribution, environment, write_payload

WINDOW_US = 24_302
RETAIN_NUM, RETAIN_DEN = 4, 5
RADIUS = 2
EBF_TAU_US = 32_000
METHODS = ("occupancy", "multiplicity", "ebf")
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "plan/protocol_tstar_fullstream_20261022.md"
OUTPUT_PATH = ROOT / "native3d_ssm/experiments/analytic_contrast/phase79_tstar_cost_quality.json"


@dataclass
class WindowOutput:
    start: int
    stop: int
    close_us: int
    emitted: np.ndarray
    indices: np.ndarray
    scores: np.ndarray
    service_us: float | None = None
    startup_us: float = 0.0


@dataclass
class FilterOutput:
    emitted: np.ndarray
    indices: np.ndarray
    scores: np.ndarray | None
    windows: list[dict]


def counter_dtype(population: int):
    """Smallest supported native width that can hold every event at one pixel."""
    if population < 0 or population > np.iinfo(np.intp).max:
        raise ValueError("population must fit the native array index range")
    if population <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if population <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


@njit(cache=True)
def _validate_values(events, width, height):
    for i in range(len(events)):
        if events[i, 0] < 0 or (i and events[i, 0] < events[i - 1, 0]):
            return 1
        if not (0 <= events[i, 1] < width and 0 <= events[i, 2] < height):
            return 2
        if events[i, 3] != 0 and events[i, 3] != 1:
            return 3
    return 0


def validate_events(events, width, height):
    if not isinstance(events, np.ndarray) or events.dtype != np.dtype(np.int64):
        raise TypeError("events must be an int64 array; floating timestamps are not converted")
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have exactly [t_us,x,y,p] columns")
    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)) or width <= 0 or height <= 0:
        raise ValueError("positive integer sensor dimensions are required")
    error = _validate_values(events, width, height)
    if error:
        raise ValueError({1: "timestamps must be nonnegative and nondecreasing",
                          2: "coordinates are outside the sensor",
                          3: "polarity must be 0 or 1"}[error])


def window_bounds(events, window_us=WINDOW_US):
    """Yield (start, stop, scheduled right edge), skipping only empty windows.

    Timestamp arithmetic uses Python integers, so a right edge past int64 max
    does not wrap. A timestamp exactly on an edge belongs to the next window.
    The caller validates event order once before iteration.
    """
    if not isinstance(window_us, (int, np.integer)) or window_us <= 0:
        raise ValueError("window_us must be a positive integer")
    if not len(events):
        return
    anchor = int(events[0, 0])
    start = 0
    while start < len(events):
        slot = (int(events[start, 0]) - anchor) // int(window_us)
        close = anchor + (slot + 1) * int(window_us)
        stop = len(events) if close > np.iinfo(np.int64).max else int(np.searchsorted(events[:, 0], close, side="left"))
        yield start, stop, close
        start = stop


@njit(cache=True)
def _multiplicity_scores(events, counts, radius):
    for i in range(len(events)):
        counts[events[i, 3], events[i, 2] + radius, events[i, 1] + radius] += 1
    scores = np.empty(len(events), dtype=counts.dtype)
    for i in range(len(events)):
        x, y, p = events[i, 1], events[i, 2], events[i, 3]
        total = np.uint64(0)
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                if dx != radius or dy != radius:
                    total += np.uint64(counts[p, y + dy, x + dx])
        scores[i] = total
    return scores


def multiplicity_scores(events, width, height, radius=RADIUS):
    counts = np.zeros((2, height + 2 * radius, width + 2 * radius), dtype=counter_dtype(len(events)))
    return _multiplicity_scores(events, counts, radius)


@njit(cache=True, fastmath=False)
def _ebf_stateful_scores(events, width, height, tau_us, radius, timestamps, polarities):
    """Same arithmetic/order as ebf._ebf_kernel, with caller-owned state."""
    scores = np.zeros(len(events), np.float64)
    for n in range(len(events)):
        te, xe, ye = events[n, 0], events[n, 1], events[n, 2]
        pe = np.int8(1) if events[n, 3] > 0 else np.int8(-1)
        acc = 0.0
        for y in range(max(0, ye - radius), min(height - 1, ye + radius) + 1):
            base = y * width
            for x in range(max(0, xe - radius), min(width - 1, xe + radius) + 1):
                if x == xe and y == ye:
                    continue
                index = base + x
                if polarities[index] != pe:
                    continue
                dt = te - timestamps[index]
                if dt >= tau_us:
                    continue
                if dt < 0:
                    dt = -dt
                acc += 1.0 - dt / tau_us
        scores[n] = acc
        centre = ye * width + xe
        timestamps[centre] = te
        polarities[centre] = pe
    return scores


@njit(cache=True)
def _emit_from_histogram(events, scores, histogram, keep_n):
    observed_max = 0
    for score in scores:
        index = int(score)
        histogram[index] += 1
        if index > observed_max:
            observed_max = index
    threshold, above = observed_max, 0
    while threshold > 0 and above + int(histogram[threshold]) < keep_n:
        above += int(histogram[threshold])
        threshold -= 1
    ties_left = keep_n - above
    emitted = np.empty((keep_n, 4), np.int64)
    indices = np.empty(keep_n, np.int64)
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
def _emit_from_threshold(events, scores, keep_n, threshold):
    above = 0
    for score in scores:
        if score > threshold:
            above += 1
    ties_left = keep_n - above
    emitted = np.empty((keep_n, 4), np.int64)
    indices = np.empty(keep_n, np.int64)
    target = 0
    for i in range(len(scores)):
        take = scores[i] > threshold
        if scores[i] == threshold and ties_left > 0:
            take = True
            ties_left -= 1
        if take:
            emitted[target] = events[i]
            indices[target] = i
            target += 1
    return emitted, indices


def _filter_window(events, width, height, method, radius, keep_n, ebf_tau_us, ebf_state):
    buffered = events.copy()
    if method == "occupancy":
        scores = occupancy_scores(buffered, width, height, radius)
        histogram = np.zeros((2 * radius + 1) ** 2, dtype=counter_dtype(len(events)))
        emitted, indices = _emit_from_histogram(buffered, scores, histogram, keep_n)
    elif method == "multiplicity":
        scores = multiplicity_scores(buffered, width, height, radius)
        histogram = np.zeros(len(events), dtype=counter_dtype(len(events)))
        emitted, indices = _emit_from_histogram(buffered, scores, histogram, keep_n)
    else:
        scores = _ebf_stateful_scores(buffered, width, height, float(ebf_tau_us), radius, *ebf_state)
        # NumPy introselect needs one score-sized copy, without a full sort.
        threshold = float(np.partition(scores, len(scores) - keep_n)[len(scores) - keep_n]) if keep_n else float("inf")
        emitted, indices = _emit_from_threshold(buffered, scores, keep_n, threshold)
    return scores, emitted, indices


def iter_filtered_windows(events, width, height, method="occupancy", *, window_us=WINDOW_US,
                          retain_num=RETAIN_NUM, retain_den=RETAIN_DEN, radius=RADIUS,
                          ebf_tau_us=EBF_TAU_US, measure_service=False):
    """Yield independent copied output payloads and source indices per window.

    The input is a complete, ordered int64 source stream. Kernel state is owned
    by this iterator; EBF processes rejected events too. ``measure_service`` adds
    per-window clock reads and is used only by the separate FIFO replay. The
    complete timing loop measures this whole iterator, including validation,
    boundary discovery and state initialization, without those extra clocks.
    """
    setup_began = time.perf_counter() if measure_service else None
    validate_events(events, width, height)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if radius not in (1, 2):
        raise ValueError("the packed occupancy implementation supports radius 1 or 2")
    if not all(isinstance(v, (int, np.integer)) for v in (retain_num, retain_den)) or not 0 <= retain_num <= retain_den or retain_den <= 0:
        raise ValueError("retention must be an exact integer fraction in [0,1]")
    if not np.isfinite(ebf_tau_us) or ebf_tau_us <= 0:
        raise ValueError("ebf_tau_us must be positive and finite")
    state = (np.full(width * height, -1, np.int64), np.zeros(width * height, np.int8)) if method == "ebf" else None
    bounds = iter(window_bounds(events, window_us))
    setup_us = (time.perf_counter() - setup_began) * 1e6 if measure_service else 0.0
    while True:
        began = time.perf_counter() if measure_service else None
        try:
            start, stop, close = next(bounds)
        except StopIteration:
            break
        keep_n = (stop - start) * int(retain_num) // int(retain_den)
        scores, emitted, indices = _filter_window(events[start:stop], width, height, method, radius, keep_n, ebf_tau_us, state)
        indices += start
        startup_us = setup_us
        service_us = (time.perf_counter() - began) * 1e6 + setup_us if measure_service else None
        setup_us = 0.0
        yield WindowOutput(start, stop, close, emitted, indices, scores, service_us, startup_us)
        # Consumers used for timing must also release their item before next().
        # Whole-stream collectors intentionally retain output as harness memory.
        del scores, emitted, indices


def filter_stream(events, width, height, method="occupancy", *, diagnostics=False, **kwargs):
    """Collect emitted events; optional full-source scores are evaluation memory."""
    outputs, indices, windows = [], [], []
    scores = np.empty(len(events), np.float64) if diagnostics else None
    for item in iter_filtered_windows(events, width, height, method, **kwargs):
        outputs.append(item.emitted)
        indices.append(item.indices)
        if diagnostics:
            scores[item.start:item.stop] = item.scores
        windows.append({"start": item.start, "stop": item.stop, "scheduled_close_us": item.close_us,
                        "events": item.stop - item.start, "retained_events": len(item.indices)})
    return FilterOutput(np.concatenate(outputs) if outputs else np.empty((0, 4), np.int64),
                        np.concatenate(indices) if indices else np.empty(0, np.int64), scores, windows)


def working_storage(width, height, population, method, *, radius=RADIUS,
                    retain_num=RETAIN_NUM, retain_den=RETAIN_DEN):
    """Conservative sum of explicit arrays, not RSS or maximum live allocation.

    Full source arrays, collected output and diagnostics belong to the harness.
    External acquisition/output queues are excluded and separately replayed.
    Input/output events use the same 4xint64 layout for every method. The
    NumPy partition score copy is counted for EBF; its auxiliary workspace is O(1).
    """
    if method not in METHODS or population < 1:
        raise ValueError("a supported method and nonempty population are required")
    n, keep = int(population), int(population) * retain_num // retain_den
    count_bytes = counter_dtype(n).itemsize
    planes = 2 * (height + 2 * radius) * ((width + 2 * radius + 63) // 64 + 1) * 8
    cells = 2 * (height + 2 * radius) * (width + 2 * radius)
    parts = {"input_buffer_bytes": n * 32, "output_events_bytes": keep * 32,
             "output_indices_bytes": keep * 8, "operand_bytes": 0,
             "score_bytes": 0, "selection_histogram_bytes": 0,
             "selection_partition_copy_bytes": 0, "readonly_popcount_table_bytes": 0}
    if method == "occupancy":
        parts.update(operand_bytes=planes, score_bytes=n,
                     selection_histogram_bytes=(2 * radius + 1) ** 2 * count_bytes,
                     readonly_popcount_table_bytes=32)
    elif method == "multiplicity":
        parts.update(operand_bytes=cells * count_bytes, score_bytes=n * count_bytes,
                     selection_histogram_bytes=n * count_bytes)
    else:
        parts.update(operand_bytes=width * height * 9, score_bytes=n * 8,
                     selection_partition_copy_bytes=n * 8)
    return {"method": method, "population": n, "retained_events": keep,
            "count_dtype": counter_dtype(n).name if method != "ebf" else None,
            "components": parts, "total_bytes": sum(parts.values()),
            "scope": "Conservative sum of explicit working arrays, not measured RSS; complete source/output harness arrays and external queues excluded."}


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def confusion_metrics(labels, kept):
    # Evaluation only; the filtering API never receives labels.
    from .phase77_led_operating_points import count_metrics
    return count_metrics({"kept_signal": int(np.count_nonzero(kept & (labels == 0))),
                          "kept_noise": int(np.count_nonzero(kept & (labels == 1))),
                          "rejected_signal": int(np.count_nonzero(~kept & (labels == 0))),
                          "rejected_noise": int(np.count_nonzero(~kept & (labels == 1)))})


def paired_bootstrap(values, names, replicates=200_000, seed=20261022):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("paired inference needs finite, nonempty sequence contrasts")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for start in range(0, replicates, 4096):
        count = min(4096, replicates - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples[start:start + count] = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
            "per_scene_difference": dict(zip(names, values.tolist())),
            "scenes": len(values), "scenes_ahead": int((values > 0).sum()),
            "scenes_behind": int((values < 0).sum()), "scenes_tied": int((values == 0).sum()),
            "replicates": replicates, "seed": seed,
            "scope": "Equal-sequence paired percentile bootstrap; assumes independent sequences; no multiplicity correction or independent-session guarantee."}


def _reference_samples(events, scores, method, start, stop):
    """Literal event-difference oracle, on deterministic targets in a full window."""
    block = events[start:stop]
    targets = sorted(set((0, len(block) // 2, len(block) - 1)))
    expected = []
    for i in targets:
        dx, dy = np.abs(block[:, 1] - block[i, 1]), np.abs(block[:, 2] - block[i, 2])
        mask = (np.maximum(dx, dy) <= RADIUS) & ((dx + dy) > 0) & (block[:, 3] == block[i, 3])
        expected.append(int(len(np.unique(block[mask, 1:3], axis=0))) if method == "occupancy" else int(mask.sum()))
    actual = scores[targets].astype(np.uint64).tolist()
    if actual != expected:
        raise AssertionError(f"{method} differs from independent ring oracle")
    return {"start": start, "stop": stop, "local_indices": targets,
            "expected_scores": expected, "actual_scores": actual}


def quality_scene(events, labels, width, height, anchor=None):
    validate_events(events, width, height)
    if labels.shape != (len(events),) or np.any((labels != 0) & (labels != 1)):
        raise ValueError("aligned binary released labels are required")
    bounds = list(window_bounds(events))
    windows = [{"start": a, "stop": b, "scheduled_close_us": c,
                "events": b - a, "retained_events": (b - a) * RETAIN_NUM // RETAIN_DEN,
                "signal_events": int(np.count_nonzero(labels[a:b] == 0)),
                "noise_events": int(np.count_nonzero(labels[a:b] == 1))}
               for a, b, c in bounds]
    source_hash, label_hash = digest_array(events.astype('<i8', copy=False)), digest_array(labels.astype(np.uint8))
    result = {"source_events": len(events), "source_sha256": source_hash,
              "labels_sha256": label_hash, "windows": windows,
              "window_event_count": distribution([b-a for a,b,_ in bounds]),
              "single_class_windows": sum(w["signal_events"] == 0 or w["noise_events"] == 0 for w in windows),
              "windows_below_1024_events": sum(w["events"] < 1024 for w in windows),
              "methods": {}, "reference_checks": {}, "common_target_bridge": {}}
    common = None
    if anchor is not None:
        if source_hash != anchor["source_stream_sha256"] or label_hash != anchor["labels_sha256"]:
            raise AssertionError("source or labels differ from the prior common-target study")
        masks = []
        for partition in ("fixed_n_10240", "t_24302us"):
            mask = np.zeros(len(events), bool)
            for a, b in anchor["native"][partition]["eligible_intervals"]:
                mask[a:b] = True
            masks.append(mask)
        common = masks[0] & masks[1]
        from .phase74_common_event_led import mask_digest
        if mask_digest(common) != anchor["common"]["primary_pair"]["mask_sha256"]:
            raise AssertionError("common-mask bridge failed")
        result["common_target_bridge"].update(events=int(common.sum()), mask_sha256=mask_digest(common))
    reference_windows = sorted(set((0, len(bounds) // 2, len(bounds) - 1)))
    for method in METHODS:
        all_scores = np.empty(len(events), np.float64)
        kept = np.zeros(len(events), bool)
        output_digest, index_digest = hashlib.sha256(), hashlib.sha256()
        reference = []
        for window_index, item in enumerate(iter_filtered_windows(events, width, height, method)):
            a, b, c = bounds[window_index]
            if (item.start, item.stop, item.close_us) != (a, b, c):
                raise AssertionError("methods used different boundaries")
            if len(item.indices) != (b-a) * 4 // 5 or not np.array_equal(item.emitted, events[item.indices]):
                raise AssertionError("quota, payload or output order mismatch")
            if len(item.indices) > 1 and np.any(np.diff(item.indices) <= 0):
                raise AssertionError("output indices are not strictly increasing")
            all_scores[a:b] = item.scores
            kept[item.indices] = True
            output_digest.update(item.emitted.tobytes())
            index_digest.update(item.indices.tobytes())
            if method != "ebf" and window_index in reference_windows:
                reference.append(_reference_samples(events, item.scores, method, a, b))
            del item
        metrics = confusion_metrics(labels, kept)
        metrics.update(event_auc=float(roc_auc_score(labels, -all_scores)) if len(np.unique(labels)) == 2 else None,
                       event_auc_defined=bool(len(np.unique(labels)) == 2),
                       maximum_score=float(all_scores.max()), output_sha256=output_digest.hexdigest(),
                       indices_sha256=index_digest.hexdigest())
        result["methods"][method] = metrics
        if method == "ebf":
            from .ebf import ebf_score
            expected = ebf_score(events, width, height)
            error = float(np.max(np.abs(all_scores - expected)))
            if error != 0:
                raise AssertionError("window-carried EBF scores differ from original continuous reference")
            result["reference_checks"][method] = {"events": len(events), "max_absolute_error": error}
            del expected
        else:
            result["reference_checks"][method] = reference
        if common is not None and method != "multiplicity":
            auc = float(roc_auc_score(labels[common], -all_scores[common]))
            expected = anchor["common"]["primary_pair"]["ebf_event_auc"] if method == "ebf" else anchor["common"]["primary_pair"]["rsf_event_auc"]["t_24302us"]
            if abs(auc - expected) > 1e-12:
                raise AssertionError("common-target score AUC differs from phase74")
            result["common_target_bridge"][method] = {"event_auc": auc, "reference_event_auc": expected,
                                                        "absolute_error": abs(auc - expected)}
        del all_scores, kept
    maximum = max(w["events"] for w in windows)
    result["storage_max_window"] = {m: working_storage(width, height, maximum, m) for m in METHODS}
    return result


METRICS = ("balanced_accuracy", "event_auc", "signal_recall", "noise_rejection",
           "signal_precision", "signal_f1", "retained_fraction")


def summarize_quality(records):
    names = sorted(records)
    result = {"scenes": len(names), "source_events": sum(r["source_events"] for r in records.values()),
              "windows": sum(len(r["windows"]) for r in records.values()),
              "single_class_windows": sum(r["single_class_windows"] for r in records.values()),
              "windows_below_1024_events": sum(r["windows_below_1024_events"] for r in records.values()),
              "aggregation": "Equal-sequence means of full-source event metrics; no exclusions based on quality or class composition.",
              "methods": {}, "paired_occupancy_minus": {}}
    for method in METHODS:
        values = [records[name]["methods"][method] for name in names]
        defined_ba = [name for name in names if records[name]["methods"][method]["balanced_accuracy"] is not None]
        result["methods"][method] = {
            "equal_scene_mean": {metric: float(np.mean([v[metric] for v in values if v[metric] is not None])) if any(v[metric] is not None for v in values) else None for metric in METRICS},
            "undefined_scene_metrics": {metric: sum(v[metric] is None for v in values) for metric in METRICS},
            "pooled_counts": {key: sum(v[key] for v in values) for key in ("kept_signal", "kept_noise", "rejected_signal", "rejected_noise", "retained_events")},
            "worst_balanced_accuracy": {"scene": min(defined_ba, key=lambda n: records[n]["methods"][method]["balanced_accuracy"]),
                                       "value": min(records[n]["methods"][method]["balanced_accuracy"] for n in defined_ba)} if defined_ba else None}
    for comparator in ("multiplicity", "ebf"):
        result["paired_occupancy_minus"][comparator] = {}
        for metric in ("balanced_accuracy", "event_auc"):
            eligible = [name for name in names if all(records[name]["methods"][m][metric] is not None for m in ("occupancy", comparator))]
            contrast = [records[name]["methods"]["occupancy"][metric] - records[name]["methods"][comparator][metric] for name in eligible]
            result["paired_occupancy_minus"][comparator][metric] = paired_bootstrap(contrast, eligible) if eligible else None
    return result


def quality(args, payload):
    from .phase54_led import iter_scene
    anchor_path = ROOT / "native3d_ssm/experiments/analytic_contrast/phase74_common_event_led.json"
    anchor = json.loads(anchor_path.read_text())["per_scene"]
    scenes = sorted(p.name for p in (args.led_root / "raw_event").iterdir() if p.is_dir())
    if scenes != sorted(anchor):
        raise AssertionError("LED source cohort differs from the existing 66-sequence cohort")
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    records = payload.setdefault("quality", {}).setdefault("per_scene", {})
    for scene in scenes:
        if scene in records:
            continue
        events, labels, ambiguous, skipped, reordered = iter_scene(args.led_root, scene)
        if ambiguous or skipped:
            raise AssertionError("unresolved label assignment or skipped source chunk")
        record = quality_scene(events, labels, 1280, 720, anchor[scene])
        record.update(ambiguous_events=ambiguous, skipped_chunks=skipped, reordered=bool(reordered))
        records[scene] = record
        write_payload(args.output, payload)
        print(f"quality {len(records)}/{len(scenes)} {scene}: n={len(events)} BA=" + "/".join(f"{record['methods'][m]['balanced_accuracy']:.6f}" for m in METHODS), flush=True)
    payload["quality"]["summary"] = summarize_quality(records)
    payload["quality_complete"] = len(records) == 66 and not args.max_scenes
    write_payload(args.output, payload)


def timed_pass(events, width, height, method):
    started = time.perf_counter()
    checksum, emitted = 0, 0
    for item in iter_filtered_windows(events, width, height, method):
        emitted += len(item.indices)
        if len(item.indices):
            checksum += int(item.emitted[0, 0]) + int(item.indices[-1])
        del item  # release the previous window before allocating its successor
    return time.perf_counter() - started, emitted, checksum


def emission_replay(events, width, height, method):
    bounds = list(window_bounds(events))
    closes = np.array([c for _, _, c in bounds], np.int64)
    ready = float(events[0, 0])
    delays, records = [], []
    for window_index, item in enumerate(iter_filtered_windows(events, width, height, method, measure_service=True)):
        service = item.service_us
        begin = max(float(item.close_us), ready)
        ready = begin + service
        delay = ready - events[item.indices, 0].astype(np.float64)
        delays.append(delay)
        queued_events = max(0, int(np.searchsorted(events[:, 0], ready, side="left")) - item.stop)
        queued_windows = max(0, int(np.searchsorted(closes, ready, side="left")) - window_index - 1)
        records.append({"start": item.start, "stop": item.stop, "scheduled_close_us": item.close_us,
                        "retained_events": len(item.indices), "service_us": service,
                        "startup_validation_and_state_us": item.startup_us,
                        "service_start_us": begin, "emission_time_us": ready,
                        "queue_wait_us": begin - item.close_us,
                        "queued_source_events_after_service": queued_events,
                        "queued_complete_windows_after_service": queued_windows,
                        "retained_arrival_us_sum": len(item.indices) * int(events[item.start, 0]) + int((events[item.indices, 0] - events[item.start, 0]).sum()),
                        "retained_arrival_us_min": int(events[item.indices[0], 0]) if len(item.indices) else None,
                        "retained_arrival_us_max": int(events[item.indices[-1], 0]) if len(item.indices) else None,
                        "emitted_delay_us_sum": float(delay.sum()),
                        "emitted_delay_us_min": float(delay.min()) if len(delay) else None,
                        "emitted_delay_us_max": float(delay.max()) if len(delay) else None})
        del item
    all_delays = np.concatenate(delays)
    peak_events = max(r["queued_source_events_after_service"] for r in records)
    return {"windows": records, "acquisition_to_emission_us": distribution(all_delays),
            "processing_per_window_us": distribution([r["service_us"] for r in records]),
            "queue_wait_us": distribution([r["queue_wait_us"] for r in records]),
            "peak_additional_queued_events": peak_events,
            "peak_additional_queue_bytes": peak_events * 32,
            "peak_additional_queued_complete_windows": max(r["queued_complete_windows_after_service"] for r in records),
            "scope": "Single-server FIFO replay, scheduled right-edge closure including final partial window; first service includes whole-source validation and EBF state initialization. Per-window clocks include boundary discovery/input copy/scoring/selection/output. Harness bookkeeping, I/O and output transmission excluded; no live timer or deadline guarantee."}


def timing_pool(args):
    from .phase54_led import iter_scene
    from .data import iter_labeled_text
    first_scene = sorted(p.name for p in (args.led_root / "raw_event").iterdir() if p.is_dir())[0]
    led, _, ambiguous, skipped, _ = iter_scene(args.led_root, first_scene)
    if ambiguous or skipped:
        raise AssertionError("timing source failed label/source alignment")
    dnd_path = args.dnd21_root / "1hz/mix_result.txt"
    dnd_chunks, n = [], 0
    for chunk in iter_labeled_text(dnd_path, 100_000):
        take = min(len(chunk), args.timing_events - n)
        dnd_chunks.append(chunk[:take, :4])
        n += take
        if n == args.timing_events:
            break
    dnd = np.ascontiguousarray(np.concatenate(dnd_chunks), dtype=np.int64)
    return {"346x260": (346, 260, dnd, "DND21/1hz_hotel-bar"),
            "1280x720": (1280, 720, np.ascontiguousarray(led[:args.timing_events]), "LED/" + first_scene)}


def timing(args, payload):
    payload["timing_environment"] = environment()
    payload["timing_protocol"] = {"repeats": args.repeats, "warmup_passes": 2,
        "prefix_max_events": args.timing_events,
        "order": "Cyclic occupancy/multiplicity/EBF order rotated by repeat; nine repeats give every arm each position three times.",
        "scope": "Complete iterator: validation, state initialization, boundary discovery, input copy, scoring, selection, copied ordered event payloads and indices. Excludes I/O, JIT compilation, evaluation metrics and collected full-output concatenation. Includes Python iterator dispatch and a scalar checksum per window.",
        "contention_note": args.contention_note}
    records = payload.setdefault("timing", {})
    for geometry, (width, height, events, source) in timing_pool(args).items():
        for method in METHODS:
            for _ in range(2):
                timed_pass(events, width, height, method)
        samples = {m: [] for m in METHODS}
        outputs = {}
        orders = []
        for repeat in range(args.repeats):
            order = METHODS[repeat % 3:] + METHODS[:repeat % 3]
            orders.append(list(order))
            for method in order:
                elapsed, retained, checksum = timed_pass(events, width, height, method)
                samples[method].append(elapsed)
                outputs[method] = {"retained_events": retained, "checksum": checksum}
        bounds = list(window_bounds(events))
        maximum = max(b-a for a,b,_ in bounds)
        records[geometry] = {"source": source, "source_events": len(events), "source_sha256": digest_array(events),
            "windows": len(bounds), "window_event_count": distribution([b-a for a,b,_ in bounds]),
            "repeat_orders": orders,
            "methods": {m: {"seconds": samples[m], "throughput_mev_s": distribution(len(events)/np.asarray(samples[m])/1e6),
                             **outputs[m], "working_storage_at_max_window": working_storage(width,height,maximum,m)} for m in METHODS}}
        records[geometry]["emission_replay"] = {m: emission_replay(events,width,height,m) for m in METHODS}
        write_payload(args.output, payload)
        print("timing " + geometry + " " + "/".join(f"{m}={records[geometry]['methods'][m]['throughput_mev_s']['median']:.3f}Mev/s" for m in METHODS), flush=True)
    payload["timing_complete"] = args.repeats == 9 and args.timing_events == 4_000_000
    write_payload(args.output, payload)


def provenance():
    files = ("native3d_ssm/phase79_tstar_cost_quality.py", "native3d_ssm/phase75_integer_count_cost.py",
             "native3d_ssm/phase77_led_operating_points.py", "native3d_ssm/phase74_common_event_led.py",
             "native3d_ssm/phase54_led.py", "native3d_ssm/ebf.py", "native3d_ssm/data.py")
    code = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    protocol = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps({"code":code,"protocol":protocol},sort_keys=True).encode()).hexdigest()
    return {"code_sha256":code,"protocol_path":str(PROTOCOL_PATH.relative_to(ROOT)),
            "protocol_sha256":protocol,"run_signature":signature}


def main():
    from .phase54_led import LED_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("quality","timing","all"), default="quality")
    parser.add_argument("--led-root", type=Path, default=LED_ROOT)
    parser.add_argument("--dnd21-root",type=Path,default=ROOT/"ECCV2024_datasets/AUC_test")
    parser.add_argument("--output",type=Path,default=OUTPUT_PATH)
    parser.add_argument("--max-scenes",type=int)
    parser.add_argument("--timing-events",type=int,default=4_000_000)
    parser.add_argument("--repeats",type=int,default=9)
    parser.add_argument("--contention-note",default="Not independently audited; formal timing requires serialized experiments.")
    args = parser.parse_args()
    meta = provenance()
    payload = json.loads(args.output.read_text()) if args.output.exists() else {
        "schema_version":1,"phase":79,"window_us":WINDOW_US,"retain_num":RETAIN_NUM,"retain_den":RETAIN_DEN,
        "radius":RADIUS,"ebf_tau_us":EBF_TAU_US,"methods":list(METHODS),
        "quality_complete":False,"timing_complete":False,
        "runtime_environment":environment(),
        "configuration_scope":"Full original source stream, all nonempty anchored windows including tail; same exact quota per window; no label-dependent selection or common-mask restriction.",
        **meta}
    if payload["run_signature"] != meta["run_signature"]:
        raise RuntimeError("Code/protocol changed since this artifact began; choose a fresh output path")
    if args.stage in ("quality","all"):
        quality(args,payload)
    if args.stage in ("timing","all"):
        timing(args,payload)
    payload["complete"] = bool(payload["quality_complete"] and payload["timing_complete"])
    write_payload(args.output,payload)
    print(f"wrote {args.output}",flush=True)


if __name__ == "__main__":
    main()
