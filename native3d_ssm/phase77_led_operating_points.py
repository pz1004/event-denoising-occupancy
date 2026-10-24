"""Matched offline output budgets on the Phase 74 primary common LED events.

The two frozen partitions retain their full scoring contexts. EBF is scored once
on the continuous original scene. Within each partition window, both methods
rank ONLY common target events and retain floor(r*n_common), breaking score ties
by original arrival index. Labels are used only after selection. This is an
offline output-budget experiment, not a new causal emission policy or cost run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from .ebf import EBF_TAU_US
from .phase54_led import ARTIFACTS, BLOCK, HEIGHT, LED_ROOT, WIDTH, iter_scene
from .phase57_ebf_labelled import ebf_noise_score
from .phase66_time_budget import T_STAR_US, count_windows, time_windows
from .phase74_common_event_led import (
    PRIMARY, count_auc, digest_bytes, eligibility, mask_digest,
    paired_inference, support_scores,
)

RETENTIONS = (.10, .20, .28, .40, .50, .60, .70, .80, .90, .95, 1.0)
PRIMARY_RETENTION = .80
METHODS = ("rsf", "ebf")
METRICS = ("signal_recall", "noise_rejection", "signal_precision",
           "balanced_accuracy", "signal_f1", "retained_fraction")
COUNT_NAMES = ("kept_signal", "kept_noise", "rejected_signal", "rejected_noise")
SOURCE_FILES = ("phase74_common_event_led.json", "phase75_integer_count_cost.json")
CODE_FILES = ("phase77_led_operating_points.py", "phase74_common_event_led.py",
              "phase66_time_budget.py", "phase54_led.py", "phase52_bitmap_profile.py",
              "phase57_ebf_labelled.py", "ebf.py")
TOLERANCE = 1e-12


def retention_key(r: float) -> str:
    return f"{r:.2f}"


def exact_budget(n: int, retention: float) -> int:
    """Floor the specified decimal retention exactly, without binary-float drift."""
    if n < 0 or not 0 <= retention <= 1:
        raise ValueError("nonnegative count and retention in [0,1] required")
    fraction = Fraction(str(retention))
    return n * fraction.numerator // fraction.denominator


def rank_common_targets(noise_scores: np.ndarray, common_mask: np.ndarray) -> np.ndarray:
    """Label-free stable ordering: lower noise scores are retained first."""
    scores = np.asarray(noise_scores)
    mask = np.asarray(common_mask)
    if scores.ndim != 1 or mask.shape != scores.shape or mask.dtype != np.bool_:
        raise ValueError("aligned one-dimensional scores and boolean mask required")
    targets = np.flatnonzero(mask)
    if not np.isfinite(scores[targets]).all():
        raise ValueError("common target scores must be finite")
    return targets[np.argsort(scores[targets], kind="stable")]


def stable_keep_indices(noise_scores: np.ndarray, common_mask: np.ndarray,
                        retention: float) -> np.ndarray:
    """Original-order selected indices; intentionally accepts no labels."""
    if not 0 <= retention <= 1:
        raise ValueError("retention must be in [0,1]")
    ranked = rank_common_targets(noise_scores, common_mask)
    return np.sort(ranked[:exact_budget(len(ranked), retention)])


def count_metrics(counts: dict) -> dict:
    """Signal-positive operating metrics; AUC elsewhere remains noise-positive."""
    ks, kn, rs, rn = (int(counts[name]) for name in COUNT_NAMES)
    if min(ks, kn, rs, rn) < 0:
        raise ValueError("negative confusion count")
    signal, noise, retained = ks + rs, kn + rn, ks + kn
    total = signal + noise
    recall = ks / signal if signal else None
    rejection = rn / noise if noise else None
    return {**dict(zip(COUNT_NAMES, (ks, kn, rs, rn))),
            "signal_events": signal, "noise_events": noise,
            "events": total, "retained_events": retained,
            "signal_recall": recall, "noise_rejection": rejection,
            "signal_precision": ks / retained if retained else None,
            "balanced_accuracy": .5 * (recall + rejection)
            if recall is not None and rejection is not None else None,
            "signal_f1": 2 * ks / (2 * ks + kn + rs) if 2 * ks + kn + rs else None,
            "retained_fraction": retained / total if total else None}


def measure_selection(noise_scores: np.ndarray, labels: np.ndarray,
                      common_mask: np.ndarray, windows: list[tuple[int, int]],
                      retentions=RETENTIONS) -> tuple[dict, list[list[int]]]:
    """Pool exact counts across ALL target windows, including single-class ones.

    Sorting is independent of labels. Labels are indexed only afterwards to
    measure the class counts in each predetermined prefix of that ordering.
    """
    if len(noise_scores) != len(labels) or len(labels) != len(common_mask):
        raise ValueError("scores, labels and target mask must align")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("binary released labels required")
    eligible = eligibility(len(labels), windows)
    if np.any(common_mask & ~eligible):
        raise ValueError("common targets outside scoring windows")
    if any(not 0 <= r <= 1 for r in retentions):
        raise ValueError("retention must be in [0,1]")
    totals = {retention_key(r): dict.fromkeys(COUNT_NAMES, 0) for r in retentions}
    ties = {retention_key(r): {"windows_with_nonempty_selection": 0,
                             "partial_cut_tie_windows": 0, "cut_tie_events": 0,
                             "kept_from_cut_ties": 0, "ambiguous_kept_events": 0,
                             "maximum_cut_tie_fraction": 0.0} for r in retentions}
    retained_by_window = []
    for start, stop in windows:
        ranked = rank_common_targets(noise_scores[start:stop], common_mask[start:stop])
        ranked_y = labels[start:stop][ranked]
        n = len(ranked)
        noise = int(ranked_y.sum())
        signal = n - noise
        cumulative_noise = np.r_[0, np.cumsum(ranked_y, dtype=np.int64)]
        sorted_scores = noise_scores[start:stop][ranked]
        kept_row = []
        for r in retentions:
            key = retention_key(r)
            k = exact_budget(n, r)
            kn = int(cumulative_noise[k])
            ks = k - kn
            for name, value in zip(COUNT_NAMES, (ks, kn, signal - ks, noise - kn)):
                totals[key][name] += value
            kept_row.append(k)
            if k:
                cutoff = sorted_scores[k - 1]
                left = int(np.searchsorted(sorted_scores, cutoff, side="left"))
                right = int(np.searchsorted(sorted_scores, cutoff, side="right"))
                t = ties[key]
                t["windows_with_nonempty_selection"] += 1
                t["partial_cut_tie_windows"] += int(left < k < right)
                t["cut_tie_events"] += right - left
                t["kept_from_cut_ties"] += k - left
                t["ambiguous_kept_events"] += min(k - left, right - k)
                t["maximum_cut_tie_fraction"] = max(t["maximum_cut_tie_fraction"],
                                                      (right - left) / n)
        retained_by_window.append(kept_row)
    return {key: {**count_metrics(value), "tie_cut": ties[key]}
            for key, value in totals.items()}, retained_by_window


def measure_scene(root: Path, scene: str, anchor74: dict, anchor75: dict) -> tuple[str, dict]:
    started = time.perf_counter()
    events, labels, ambiguous, skipped, reordered = iter_scene(root, scene)
    if not len(events) or ambiguous or skipped:
        raise ValueError(f"{scene}: incomplete or ambiguous input")
    source_hash = digest_bytes(events.astype('<i8', copy=False).tobytes())
    label_hash = digest_bytes(labels.astype(np.uint8).tobytes())
    if source_hash != anchor74["source_stream_sha256"] or label_hash != anchor74["labels_sha256"]:
        raise AssertionError(f"{scene}: source/label hash drift")
    windows = {PRIMARY[0]: count_windows(len(events), BLOCK),
               PRIMARY[1]: time_windows(events[:, 0], T_STAR_US)}
    masks = {name: eligibility(len(events), w) for name, w in windows.items()}
    common = masks[PRIMARY[0]] & masks[PRIMARY[1]]
    primary = anchor74["common"]["primary_pair"]
    if mask_digest(common) != primary["mask_sha256"] or int(common.sum()) != primary["events"]:
        raise AssertionError(f"{scene}: common-mask drift")
    # Establish, rather than assume, equivalence to the native fixed-count selection.
    if not np.array_equal(common, masks[PRIMARY[0]]):
        raise AssertionError(f"{scene}: fixed-N common mask is not its native mask")
    ebf = ebf_noise_score(events, WIDTH, HEIGHT, tau_us=float(EBF_TAU_US))
    ebf_auc = float(roc_auc_score(labels[common], ebf[common]))
    differences = [abs(ebf_auc - primary["ebf_event_auc"])]
    result = {"source_events": len(events), "source_stream_sha256": source_hash,
              "labels_sha256": label_hash, "common_mask_sha256": mask_digest(common),
              "common_events": int(common.sum()), "common_signal_events": int((labels[common] == 0).sum()),
              "common_noise_events": int(labels[common].sum()),
              "ambiguous_labels": ambiguous, "skipped_chunks": skipped,
              "reordered": bool(reordered), "fixed_n_common_equals_native": True,
              "common_event_auc": {"ebf": ebf_auc, "rsf": {}}, "partitions": {}}
    for name in PRIMARY:
        counts = support_scores(events, windows[name])
        rsf_auc = count_auc(labels[common], counts[common])
        differences.append(abs(rsf_auc - primary["rsf_event_auc"][name]))
        result["common_event_auc"]["rsf"][name] = rsf_auc
        record = {"full_context_events": int(masks[name].sum()),
                  "full_context_mask_sha256": mask_digest(masks[name]),
                  "windows": [], "methods": {}}
        for start, stop in windows[name]:
            y = labels[start:stop][common[start:stop]]
            record["windows"].append({"start": start, "stop": stop,
                "full_context_events": stop - start, "common_events": len(y),
                "common_signal_events": int((y == 0).sum()), "common_noise_events": int(y.sum()),
                "retained_by_method": {}})
        record["window_count"] = len(windows[name])
        record["target_window_count"] = sum(w["common_events"] > 0 for w in record["windows"])
        record["single_class_target_windows"] = sum(
            w["common_events"] > 0 and min(w["common_signal_events"], w["common_noise_events"]) == 0
            for w in record["windows"])
        for method, scores in (("rsf", -counts.astype(np.int16)), ("ebf", ebf)):
            measurements, retained = measure_selection(scores, labels, common, windows[name])
            record["methods"][method] = {"by_retention": measurements}
            for row, actual in zip(record["windows"], retained):
                expected = [exact_budget(row["common_events"], r) for r in RETENTIONS]
                if actual != expected:
                    raise AssertionError("window budget mismatch")
                row["retained_by_method"][method] = actual
        for row in record["windows"]:
            if row["retained_by_method"]["rsf"] != row["retained_by_method"]["ebf"]:
                raise AssertionError("unequal method output budgets")
        record["exact_window_retention_equality"] = True
        for method in METHODS:
            for measurement in record["methods"][method]["by_retention"].values():
                if measurement["events"] != int(common.sum()):
                    raise AssertionError("selection skipped target events")
        if name == PRIMARY[0]:
            block_aucs = [count_auc(labels[a:b], counts[a:b]) for a, b in windows[name]
                          if np.unique(labels[a:b]).size == 2]
            measured = {**record["methods"]["rsf"]["by_retention"]["0.28"],
                        "event_auc": rsf_auc, "block_mean_auc": float(np.mean(block_aucs))}
            fields = ("signal_recall", "noise_rejection", "balanced_accuracy", "retained_fraction",
                      "event_auc", "block_mean_auc")
            quality_differences = {field: abs(measured[field] - anchor75["occupancy"][field])
                                   for field in fields}
            if max(quality_differences.values()) > TOLERANCE:
                raise AssertionError(f"{scene}: phase75 quality anchor drift {quality_differences}")
            result["phase75_reference"] = {field: measured[field] for field in fields}
            result["phase75_max_absolute_error"] = max(quality_differences.values())
        result["partitions"][name] = record
    if max(differences) > TOLERANCE:
        raise AssertionError(f"{scene}: phase74 common AUC drift {differences}")
    result["phase74_auc_max_absolute_error"] = max(differences)
    result["runtime_seconds"] = time.perf_counter() - started
    return scene, result


def summarize(records: dict, replicates: int, draws: int, seed: int) -> dict:
    summary = {}
    scenes = sorted(records)
    for partition_index, partition in enumerate(PRIMARY):
        rows = {s: records[s]["partitions"][partition] for s in scenes}
        output = {"scenes": len(scenes), "common_events": sum(records[s]["common_events"] for s in scenes),
                  "full_context_events": sum(row["full_context_events"] for row in rows.values()),
                  "windows": sum(row["window_count"] for row in rows.values()),
                  "single_class_target_windows": sum(row["single_class_target_windows"] for row in rows.values()),
                  "by_retention": {}}
        for r in RETENTIONS:
            key = retention_key(r)
            row = {"methods": {}}
            for method in METHODS:
                values = {s: rows[s]["methods"][method]["by_retention"][key] for s in scenes}
                pooled_counts = {name: sum(v[name] for v in values.values()) for name in COUNT_NAMES}
                means = {metric: float(np.mean([v[metric] for v in values.values()])) for metric in METRICS}
                worst = {metric: {"scene": min(scenes, key=lambda s: values[s][metric]),
                                  "value": min(v[metric] for v in values.values())} for metric in METRICS}
                row["methods"][method] = {"equal_scene_mean": means, "pooled_counts": pooled_counts,
                    "pooled_event_metrics_descriptive": count_metrics(pooled_counts), "worst_scene": worst,
                    "minimum_scene_counts": {name: min(v[name] for v in values.values())
                        for name in (*COUNT_NAMES, "events", "signal_events", "noise_events", "retained_events")},
                    "tie_cut_totals": {field: sum(v["tie_cut"][field] for v in values.values())
                        for field in ("windows_with_nonempty_selection", "partial_cut_tie_windows", "cut_tie_events",
                                      "kept_from_cut_ties", "ambiguous_kept_events")}}
            output["by_retention"][key] = row
        paired = {}
        for metric_index, metric in enumerate(("balanced_accuracy", "signal_recall", "noise_rejection")):
            delta = {s: rows[s]["methods"]["rsf"]["by_retention"]["0.80"][metric]
                     - rows[s]["methods"]["ebf"]["by_retention"]["0.80"][metric] for s in scenes}
            paired[metric] = {**paired_inference(np.asarray(list(delta.values())), replicates, draws,
                                  seed + 10 * partition_index + metric_index),
                              "per_scene_difference": delta}
        output["primary_r080_rsf_minus_ebf"] = paired
        summary[partition] = output
    return summary


def hashes(paths: list[Path]) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LED_ROOT)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase77_led_operating_points.json")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replicates", type=int, default=200_000)
    parser.add_argument("--flip-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    source_hashes = hashes([ARTIFACTS / name for name in SOURCE_FILES])
    code_hashes = hashes([Path(__file__).parent / name for name in CODE_FILES])
    source74 = json.loads((ARTIFACTS / SOURCE_FILES[0]).read_text())
    source75 = json.loads((ARTIFACTS / SOURCE_FILES[1]).read_text())
    all_scenes = sorted(source74["per_scene"])
    scenes = all_scenes[:args.max_scenes] if args.max_scenes else all_scenes
    records = {}
    prior_runtime = 0.0
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if (previous.get("schema_version") != 1 or previous.get("code_sha256") != code_hashes
                or previous.get("source_sha256") != source_hashes
                or previous.get("requested_scenes") != scenes):
            raise ValueError("resume source/code/protocol mismatch")
        records = previous["per_scene"]
        prior_runtime = previous["runtime"]["cumulative_wall_seconds"]
    payload = {"phase": "77 matched offline LED output-budget operating points", "schema_version": 1,
        "complete": False, "requested_scenes": scenes, "expected_full_scenes": 66,
        "source_sha256": source_hashes, "code_sha256": code_hashes,
        "protocol": {"retention_grid": list(RETENTIONS), "primary_retention": PRIMARY_RETENTION,
            "reference_retention": .28, "primary_metric": "balanced_accuracy",
            "complementary_dependent_metrics": ["signal_recall", "noise_rejection"],
            "partitions": list(PRIMARY), "block_events": BLOCK, "time_budget_us": T_STAR_US,
            "score_radius": 2, "sensor": [WIDTH, HEIGHT], "ebf_tau_us": EBF_TAU_US,
            "target_set": "phase74 primary intersection; scoring contexts unchanged",
            "selection": "floor(r*n_common) within each original window; lowest noise score; stable arrival ties",
            "budget_arithmetic": "exact decimal-rational integer division, not binary-float multiplication",
            "score_context": "RSF whole original partition window; EBF continuous original scene, no resets",
            "label_use": "metrics only, never ranking, budget, windowing or tie breaking",
            "window_policy": "all common target windows, including single-class windows",
            "metric_orientation": "signal-positive precision/F1/counts; noise-positive AUC anchors",
            "aggregation": "equal scene mean of scene event-pooled metrics; raw pooled counts also provided",
            "inference": "paired scene percentile bootstrap and two-sided sign flips; BA primary at r=.80",
            "multiplicity": "unadjusted; recall/rejection dependent complements; other r values descriptive",
            "native_fixed_n_equivalence": "asserted common mask equals every native full fixed-N block",
            "time_window_scope": "common target subset selected using full-window occupancy, not a native full-window deployment",
            "resource_scope": "no timing, resource or live-emission claims for r=.80 or the sweep",
            "independent_validation": False},
        "statistics": {"replicates": args.replicates, "sign_flip_draws": args.flip_draws, "seed": args.seed},
        "per_scene": records}

    def checkpoint():
        payload["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "workers": args.workers,
            "cumulative_wall_seconds": prior_runtime + time.perf_counter() - started}
        write_json(args.output, payload)

    def accept(result):
        scene, record = result
        records[scene] = record
        checkpoint()
        primary = record["partitions"][PRIMARY[1]]["methods"]
        print(f"[{len(records):02d}/{len(scenes)}] {scene}: T* BA(.80) "
              + "/".join(f"{primary[m]['by_retention']['0.80']['balanced_accuracy']:.6f}" for m in METHODS)
              + f" [{record['runtime_seconds']:.1f}s]", flush=True)

    checkpoint()
    pending = [s for s in scenes if s not in records]
    if args.workers == 1:
        for scene in pending:
            accept(measure_scene(args.root, scene, source74["per_scene"][scene],
                                 source75["quality"]["per_scene"][scene]))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(measure_scene, args.root, scene, source74["per_scene"][scene],
                        source75["quality"]["per_scene"][scene]) for scene in pending]
            for future in as_completed(futures):
                accept(future.result())
    payload["summary"] = summarize(records, args.replicates, args.flip_draws, args.seed)
    payload["anchors"] = {"phase74_auc_checks": 3 * len(records), "phase75_quality_checks": 6 * len(records),
        "phase74_max_absolute_error": max(r["phase74_auc_max_absolute_error"] for r in records.values()),
        "phase75_max_absolute_error": max(r["phase75_max_absolute_error"] for r in records.values()),
        "source_label_common_mask_hashes_match": True, "tolerance": TOLERANCE,
        "exact_window_retention_equality": True, "single_class_windows_included": True}
    full = scenes == all_scenes and len(records) == 66
    if full and sum(r["common_events"] for r in records.values()) != 111_134_720:
        raise AssertionError("primary common event total changed")
    if hashes([Path(__file__).parent / name for name in CODE_FILES]) != code_hashes:
        raise AssertionError("source code changed during run")
    payload["run_complete"] = True
    payload["complete"] = full
    checkpoint()
    print(f"wrote {args.output}; full={full}", flush=True)


if __name__ == "__main__":
    main()
