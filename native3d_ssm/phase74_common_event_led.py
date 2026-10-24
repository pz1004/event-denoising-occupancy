"""Common-evaluation-event LED partition sensitivity (phase 74).

Scoring contexts and phase66 windows are unchanged. Masks restrict evaluation only.
Primary: fixed-N intersect T*. Secondary: all six partitions' intersection.
The primary statistic is an equally weighted mean of per-scene event-level ROC-AUC;
paired bootstrap and sign-flip inference use exactly the same scene-level estimand.
This retrospective sensitivity analysis does not create an untouched validation set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import ring_support
from .phase52_bitmap_profile import build_planes, support_rowmask
from .phase54_led import ARTIFACTS, BLOCK, HEIGHT, LED_ROOT, WIDTH, iter_scene
from .phase57_ebf_labelled import ebf_noise_score
from .phase66_time_budget import MIN_EVENTS, T_GRID_US, T_STAR_US, count_windows, time_windows
from .ebf import EBF_TAU_US, check_equivalence, check_properties

PARTITIONS = ("fixed_n_10240", *(f"t_{t}us" for t in T_GRID_US))
PRIMARY = ("fixed_n_10240", f"t_{T_STAR_US}us")
ANCHOR = ARTIFACTS / "phase66_time_budget.json"
TOLERANCE = 1e-12


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mask_digest(mask: np.ndarray) -> str:
    """Hash length and packed membership, including distinct duplicate-event indices."""
    return digest_bytes(np.asarray([len(mask)], dtype='<u8').tobytes()
                        + np.packbits(mask, bitorder='little').tobytes())


def eligibility(n: int, windows: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    last_stop = 0
    for start, stop in windows:
        if not (last_stop <= start < stop <= n):
            raise ValueError("windows must be ordered, disjoint, nonempty and in bounds")
        mask[start:stop] = True
        last_stop = stop
    return mask


def common_masks(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"primary_pair": masks[PRIMARY[0]] & masks[PRIMARY[1]],
            "all_partitions": np.logical_and.reduce([masks[name] for name in PARTITIONS])}


def support_scores(events: np.ndarray, windows: list[tuple[int, int]]) -> np.ndarray:
    """Exact occupancy, with -1 for events not scored by this partition."""
    counts = np.full(len(events), -1, dtype=np.int8)
    for start, stop in windows:
        chunk = events[start:stop]
        planes, _ = build_planes(chunk, WIDTH, HEIGHT)
        counts[start:stop] = support_rowmask(chunk, planes)
    return counts


def count_auc(labels: np.ndarray, counts: np.ndarray) -> float:
    """Noise-positive AUC of -count, including half-credit for ties."""
    if len(labels) != len(counts) or not len(labels):
        raise ValueError("labels and counts must be nonempty and aligned")
    if np.any((counts < 0) | (counts > 24)) or np.any((labels != 0) & (labels != 1)):
        raise ValueError("expected scores in 0..24 and binary labels")
    noise = np.bincount(counts[labels == 1], minlength=25).astype(np.float64)
    signal = np.bincount(counts[labels == 0], minlength=25).astype(np.float64)
    if not noise.sum() or not signal.sum():
        raise ValueError("ROC-AUC requires both classes")
    # Noise has the higher score if it has the smaller support count.
    signal_larger = signal.sum() - np.cumsum(signal)
    return float(np.dot(noise, signal_larger + 0.5 * signal) / (noise.sum() * signal.sum()))


def native_metrics(labels: np.ndarray, counts: np.ndarray, ebf: np.ndarray,
                   windows: list[tuple[int, int]], mask: np.ndarray) -> dict:
    block_rsf, block_ebf = [], []
    for start, stop in windows:
        y = labels[start:stop]
        if np.all(y == y[0]):
            continue
        block_rsf.append(count_auc(y, counts[start:stop]))
        block_ebf.append(float(roc_auc_score(y, ebf[start:stop])))
    return {"events": int(mask.sum()), "windows": len(windows),
            "valid_auc_windows": len(block_rsf), "mask_sha256": mask_digest(mask),
            "eligible_intervals": windows,
            "rsf": {"event_auc": count_auc(labels[mask], counts[mask]),
                    "block_mean_auc": float(np.mean(block_rsf))},
            "ebf": {"event_auc": float(roc_auc_score(labels[mask], ebf[mask])),
                    "block_mean_auc": float(np.mean(block_ebf))}}


def paired_inference(differences: np.ndarray, replicates: int, draws: int,
                     seed: int) -> dict:
    """Mean, interval and null statistic all give each scene equal weight."""
    values = np.asarray(differences, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("all scene differences must be finite")
    if replicates < 2 or draws < 1:
        raise ValueError("at least two bootstrap replicates and one sign-flip draw required")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    batch = 4096
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[index].mean(axis=1)
    observed = float(values.mean())
    exceedances = 0
    for start in range(0, draws, batch):
        size = min(batch, draws - start)
        signs = 2 * rng.integers(0, 2, size=(size, len(values)), dtype=np.int8) - 1
        null = (signs * values).mean(axis=1)
        exceedances += int(np.sum(np.abs(null) >= abs(observed) - 1e-15))
    return {"n_scenes": len(values), "mean": observed,
            "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
            "bootstrap_se": float(samples.std(ddof=1)),
            "scenes_ahead": int((values > 0).sum()),
            "scenes_behind": int((values < 0).sum()),
            "scenes_tied": int((values == 0).sum()),
            "replicates": replicates, "sign_flip_draws": draws, "seed": seed,
            "sign_flip_exceedances": exceedances,
            "p_two_sided": (exceedances + 1) / (draws + 1)}


def summarize(records: dict, replicates: int, draws: int, seed: int) -> dict:
    summary = {}
    for domain in ("primary_pair", "all_partitions"):
        rows = [r["common"][domain] for r in records.values()]
        ebf = np.array([r["ebf_event_auc"] for r in rows])
        names = PRIMARY if domain == "primary_pair" else PARTITIONS
        by_partition = {name: np.array([r["rsf_event_auc"][name] for r in rows])
                        for name in names}
        contrasts = {}
        for name, values in by_partition.items():
            contrasts[f"{name}_minus_ebf"] = paired_inference(values - ebf, replicates, draws, seed)
        fixed = by_partition[PRIMARY[0]]
        for name in names[1:]:
            contrasts[f"{name}_minus_fixed_n"] = paired_inference(
                by_partition[name] - fixed, replicates, draws, seed)
        summary[domain] = {
            "scenes": len(rows), "events": sum(r["events"] for r in rows),
            "noise_events": sum(r["noise_events"] for r in rows),
            "signal_events": sum(r["signal_events"] for r in rows),
            "ebf_event_auc": float(ebf.mean()),
            "rsf_event_auc": {name: float(values.mean()) for name, values in by_partition.items()},
            "contrasts": contrasts,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LED_ROOT)
    parser.add_argument("--max-scenes", type=int, default=None, help="debug only")
    parser.add_argument("--replicates", type=int, default=200000)
    parser.add_argument("--flip-draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase74_common_event_led.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    anchor = json.loads(ANCHOR.read_text())
    scenes = sorted(anchor["per_scene"][PRIMARY[0]])
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    if args.output.resolve() == ANCHOR.resolve():
        raise SystemExit("refusing to overwrite phase66")
    check_properties()
    ebf_worst, _ = check_equivalence()
    records, anchor_worst = {}, 0.0
    exact_events, exact_mismatches = 0, 0
    config = StateConfig(width=WIDTH, height=HEIGHT, taus_us=(250.0,), radius=2)
    for index, scene in enumerate(scenes, 1):
        step = time.perf_counter()
        events, labels, ambiguous, skipped, reordered = iter_scene(args.root, scene)
        if not len(events) or ambiguous or skipped:
            raise RuntimeError(f"{scene}: missing data or label ambiguity; refusing partial evaluation")
        windows = {"fixed_n_10240": count_windows(len(events), BLOCK)}
        windows.update({f"t_{budget}us": time_windows(events[:, 0], budget) for budget in T_GRID_US})
        masks = {name: eligibility(len(events), w) for name, w in windows.items()}
        shared = common_masks(masks)
        ebf = ebf_noise_score(events, WIDTH, HEIGHT, tau_us=float(EBF_TAU_US))
        common = {}
        for domain, mask in shared.items():
            y = labels[mask]
            if len(np.unique(y)) != 2:
                raise RuntimeError(f"{scene}/{domain}: common set lacks a class")
            common[domain] = {
                "events": int(mask.sum()), "noise_events": int(y.sum()),
                "signal_events": int((y == 0).sum()), "mask_sha256": mask_digest(mask),
                "ebf_event_auc": float(roc_auc_score(y, ebf[mask])), "rsf_event_auc": {},
            }
        native = {}
        for name in PARTITIONS:
            counts = support_scores(events, windows[name])
            if not np.array_equal(counts >= 0, masks[name]):
                raise RuntimeError("score availability and eligibility disagree")
            # Independent float-extractor identity on actual source windows, one of each
            # partition type on the first scene; all per-scene metrics also anchor below.
            if index == 1:
                start, stop = windows[name][0]
                raw = bidirectional_features(events[start:stop], config)
                reference = ring_support(raw, 1, config.feature_dim, config.radius).astype(np.int8)
                exact_events += len(reference)
                exact_mismatches += int(np.sum(reference != counts[start:stop]))
                if exact_mismatches:
                    raise RuntimeError("bit-plane and float-extractor counts disagree")
            native[name] = native_metrics(labels, counts, ebf, windows[name], masks[name])
            was = anchor["per_scene"][name][scene]
            if native[name]["events"] != was["events"] or native[name]["windows"] != was["windows"]:
                raise RuntimeError(f"{scene}/{name}: native event/window counts changed")
            for method, source_arm in (("rsf", "support"), ("ebf", "ebf")):
                for metric in ("event_auc", "block_mean_auc"):
                    if not np.isfinite([native[name][method][metric], was[source_arm][metric]]).all():
                        raise RuntimeError(f"{scene}/{name}/{method}/{metric}: nonfinite anchor value")
                    drift = abs(native[name][method][metric] - was[source_arm][metric])
                    anchor_worst = max(anchor_worst, drift)
                    if drift > TOLERANCE:
                        raise RuntimeError(f"{scene}/{name}/{method}/{metric}: anchor drift {drift}")
            for domain, mask in shared.items():
                if np.any(mask & ~masks[name]):
                    # The pairwise primary set need not be eligible under exploratory
                    # partitions. They are omitted in that domain rather than silently
                    # intersected again with a different mask.
                    if domain == "primary_pair" and name not in PRIMARY:
                        continue
                    raise RuntimeError(f"{scene}/{name}: common set includes unscored events")
                common[domain]["rsf_event_auc"][name] = count_auc(labels[mask], counts[mask])
            del counts
        # Primary domain contains only the two planned partition arms, irrespective of
        # accidental eligibility of any exploratory partition in a particular scene.
        common["primary_pair"]["rsf_event_auc"] = {
            name: common["primary_pair"]["rsf_event_auc"][name] for name in PRIMARY}
        records[scene] = {"source_events": len(events),
                          "source_stream_sha256": digest_bytes(events.astype('<i8', copy=False).tobytes()),
                          "labels_sha256": digest_bytes(labels.astype(np.uint8).tobytes()),
                          "ambiguous_events": ambiguous, "skipped_chunks": skipped,
                          "reordered": bool(reordered), "native": native, "common": common}
        p = common["primary_pair"]
        print(f"[{index:02d}/{len(scenes)}] {scene}: common={p['events']:,}, "
              f"N={p['rsf_event_auc'][PRIMARY[0]]:.6f}, T*={p['rsf_event_auc'][PRIMARY[1]]:.6f}, "
              f"EBF={p['ebf_event_auc']:.6f} [{time.perf_counter()-step:.1f}s]", flush=True)
    payload = {"phase": "74 common-evaluation-event LED partition sensitivity", "schema_version": 1,
               "source_artifact": ANCHOR.name, "source_artifact_sha256": digest_bytes(ANCHOR.read_bytes()),
               "protocol": {"primary_partitions": list(PRIMARY), "all_partitions": list(PARTITIONS),
                            "mask_application": "evaluation only; original streams, windows and scoring context unchanged",
                            "primary_metric": "equal-scene mean event-level noise-positive ROC-AUC",
                            "primary_contrast": "t_24302us_minus_fixed_n",
                            "bootstrap": "paired scene percentile, equal weights",
                            "sign_flip": "scene-level paired differences; (exceedances+1)/(draws+1)",
                            "multiplicity": "unadjusted; secondary sweep exploratory",
                            "independent_validation": False},
               "frozen": {"block": BLOCK, "budgets_us": list(T_GRID_US), "beta": 0,
                          "score_radius": 2, "min_events_per_window": MIN_EVENTS,
                          "sensor": [WIDTH, HEIGHT], "ebf_tau_us": EBF_TAU_US},
               "statistics": {"seed": args.seed, "replicates": args.replicates, "flip_draws": args.flip_draws},
               "anchor": {"scenes": len(records), "partitions": len(PARTITIONS),
                          "metrics_compared": len(records)*len(PARTITIONS)*4,
                          "worst_auc_drift": anchor_worst, "tolerance": TOLERANCE,
                          "float_identity_events": exact_events, "float_identity_mismatches": exact_mismatches,
                          "ebf_reference_max_error": ebf_worst},
               "per_scene": records, "summary": summarize(records, args.replicates, args.flip_draws, args.seed),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                           "sklearn": sklearn.__version__,
                           "wall_seconds": time.perf_counter()-started},
               "code_sha256": {name: digest_bytes((Path(__file__).parent/name).read_bytes())
                               for name in ('phase74_common_event_led.py','phase66_time_budget.py',
                                            'phase54_led.py','phase52_bitmap_profile.py','ebf.py',
                                            'core.py','evaluate_adaptive_beta.py','phase57_ebf_labelled.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(f"wrote {args.output}; max native AUC drift {anchor_worst:.3g}", flush=True)


if __name__ == '__main__':
    main()
