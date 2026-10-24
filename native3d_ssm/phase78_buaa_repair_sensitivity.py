"""Frozen ten-seed BUAA polarity-imputation sensitivity, without seed selection.

Each seed is an imputation of the same released stream, not a new independent
dataset. All methods receive one identical repaired first-3M-event array. The
four block methods reset on complete 10,240-event blocks; EBF remains causal
and continuous. Both aggregations use the same complete-block event prefix.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numba
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from .ebf import EBF_S, EBF_TAU_US
from .matched_horizon import (exact_block_statistics, raw_directional_amplitudes,
                              single_log_score)
from .phase57_ebf_labelled import ebf_noise_score
from .phase62_buaa_campus import (ARTIFACTS, BLOCK, BUAA_ROOT, HEIGHT, MAX_EVENTS,
                                  SEQUENCES, VALID_POLARITY, WIDTH, load_sequence)

SEEDS = tuple(range(20260825, 20260835))
ARMS = ("count", "sum_flat", "logdir_flat", "logdir_250us", "ebf")
METRICS = ("block_mean_auc", "event_auc")
CLEAN = ("campus1_1Hz", "campus1_3Hz")
CORRUPT = tuple(name for name in SEQUENCES if name not in CLEAN)
GROUPS = {"all_six": SEQUENCES, "corrupt_four": CORRUPT, "clean_two": CLEAN}
CONTRASTS = tuple((a, b) for a, b in itertools.combinations(ARMS, 2))
SCHEMA_VERSION = 1
PRECISION = "integer_flat_counts_float32_single_log_export_and_addition_v1"
SOURCE_FILES = ("phase78_buaa_repair_sensitivity.py", "matched_horizon.py",
                "core.py", "ebf.py", "phase57_ebf_labelled.py", "phase62_buaa_campus.py")
ANCHOR_FILES = ("phase69_operand_isolation.json", "phase72_buaa_beta0_magnitude.json",
                "phase62_buaa_campus.json")


def digest_array(array: np.ndarray) -> str:
    """SHA256 includes dtype and shape, followed by C-order array bytes."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(json.dumps({"dtype": array.dtype.str,
                                  "shape": list(array.shape)}, sort_keys=True).encode())
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "seeds": list(SEEDS),
        "sequences": list(SEQUENCES), "arms": list(ARMS), "metrics": list(METRICS),
        "first_events": MAX_EVENTS, "block_events": BLOCK, "geometry": [WIDTH, HEIGHT],
        "score_radius": 2, "finite_tau_us": 250.0, "maximum_age_factor": 20.0,
        "ebf_s": EBF_S, "ebf_tau_us": EBF_TAU_US, "precision": PRECISION,
        "seed_scope": "independent default_rng(seed) reinitialization per sequence",
        "repair": "replace only released polarities outside {-1,+1} uniformly in {-1,+1}",
        "target_mask": "complete fixed-N10240 prefix of each first-3M stream",
        "causality": "four block-local bidirectional arms; continuous causal EBF",
        "aggregation": "equal valid-block AUC and pooled-event AUC per sequence; equal sequence weights per seed",
        "uncertainty": "descriptive imputation sensitivity; seeds and sequences are not flattened or bootstrapped",
        "plan": "plan/task_plan_operating_points_and_repair_seeds.md",
    }


def run_identity() -> dict:
    directory = Path(__file__).resolve().parent
    identity = {"protocol": protocol(),
                "source_sha256": {name: file_digest(directory / name) for name in SOURCE_FILES},
                "anchor_sha256": {name: file_digest(ARTIFACTS / name) for name in ANCHOR_FILES},
                "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                     "numba": numba.__version__, "sklearn": sklearn.__version__}}
    identity["protocol_signature"] = hashlib.sha256(
        json.dumps(identity["protocol"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity["run_signature"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity


def audit_repair(raw: np.ndarray, events: np.ndarray, labels: np.ndarray) -> dict:
    """Independently audit the legacy loader against the actual released array."""
    original = raw[:, 4].astype(np.int64)
    corrupt = ~np.isin(original, VALID_POLARITY)
    np.testing.assert_array_equal(labels, (raw[:, 0] == 0).astype(np.int64))
    np.testing.assert_array_equal(events[:, :3], raw[:, 1:4].astype(np.int64))
    np.testing.assert_array_equal(events[~corrupt, 3], (original[~corrupt] > 0).astype(np.int64))
    if np.any(corrupt & (labels == 0)):
        raise ValueError("released corruption now occurs on signal events; protocol assumptions changed")
    if np.any((events[:, 3] != 0) & (events[:, 3] != 1)):
        raise ValueError("repair did not produce binary polarities")
    return {"corrupt_entries": int(corrupt.sum()), "corrupt_on_signal": 0,
            "corrupt_on_count_after_repair": int(events[corrupt, 3].sum()),
            "corrupt_off_count_after_repair": int(corrupt.sum() - events[corrupt, 3].sum()),
            "changed_sign_entries": int(np.count_nonzero(events[:, 3] != (original > 0))),
            "valid_polarities_unchanged": True, "timestamps_coordinates_labels_unchanged": True,
            "corrupt_mask_sha256": digest_array(corrupt)}


def subset_scores(events: np.ndarray, width: int, height: int, block: int = BLOCK) -> dict:
    """Only the five frozen study arms; no extra scales, gates or diagnostics."""
    n = len(events) // block * block
    if n == 0:
        raise ValueError("no complete evaluation block")
    scores = {arm: np.empty(n, dtype=np.float64) for arm in ARMS if arm != "ebf"}
    for start in range(0, n, block):
        stop = start + block
        chunk = events[start:stop]
        occupancy, total, past, future = exact_block_statistics(chunk, width, height, radius=2)
        scores["count"][start:stop] = -occupancy.astype(np.float64)
        scores["sum_flat"][start:stop] = -total.astype(np.float64)
        scores["logdir_flat"][start:stop] = single_log_score(
            past.astype(np.float64), future.astype(np.float64))
        f, b = raw_directional_amplitudes(chunk, width, height, taus_us=(250.0,), radius=2)
        scores["logdir_250us"][start:stop] = single_log_score(f[:, 0], b[:, 0])
    # Keep EBF's map across block boundaries, exactly as phases 62 and 72.
    scores["ebf"] = ebf_noise_score(events, width, height, float(EBF_TAU_US))[:n]
    return scores


def score_metrics(scores: dict, labels: np.ndarray, block: int = BLOCK) -> dict:
    n = len(next(iter(scores.values())))
    y = labels[:n]
    if n % block or any(len(s) != n for s in scores.values()):
        raise ValueError("methods must share the complete-block target mask")
    if y.min() == y.max():
        raise ValueError("AUC requires both classes in the event pool")
    out = {}
    for arm, values in scores.items():
        aucs = []
        for start in range(0, n, block):
            yy = y[start:start + block]
            aucs.append(None if yy.min() == yy.max() else
                        float(roc_auc_score(yy, values[start:start + block])))
        valid = [v for v in aucs if v is not None]
        if not valid:
            raise ValueError("no two-class blocks")
        out[arm] = {"event_auc": float(roc_auc_score(y, values)),
                    "block_mean_auc": float(np.mean(valid)),
                    "valid_auc_blocks": len(valid), "block_aucs": aucs}
    return out


def evaluate_record(root: str, name: str, seed: int, clean_base: dict | None = None) -> dict:
    path = Path(root) / f"{name}.npy"
    raw = np.asarray(np.load(path, mmap_mode="r")[:MAX_EVENTS])
    events, labels, diagnostics = load_sequence(path, repair=True, seed=seed)
    if len(events) != MAX_EVENTS:
        raise ValueError(f"{name}: expected the full first {MAX_EVENTS} events")
    audit = audit_repair(raw, events, labels)
    if bool(audit["corrupt_entries"]) != (name in CORRUPT):
        raise ValueError(f"{name}: corrupt/clean subgroup differs from the frozen protocol")
    n = len(events) // BLOCK * BLOCK
    hashes = {"raw_first3m_sha256": digest_array(raw),
              "repaired_first3m_sha256": digest_array(events),
              "labels_first3m_sha256": digest_array(labels),
              "scored_events_sha256": digest_array(events[:n]),
              "scored_labels_sha256": digest_array(labels[:n]),
              "target_mask_sha256": digest_array(np.arange(len(events)) < n)}
    if clean_base is not None:
        if audit["corrupt_entries"] or hashes != clean_base["hashes"]:
            raise ValueError("clean-score reuse requires independently identical loaded input and mask")
        metrics = copy.deepcopy(clean_base["metrics"])
        mode = "clean_input_invariant_copy"
    else:
        metrics = score_metrics(subset_scores(events, WIDTH, HEIGHT), labels)
        mode = "recomputed"
    return {"sequence": name, "seed": seed, "metrics": metrics,
            "hashes": hashes, "repair_audit": audit, "diagnostics": diagnostics,
            "scored_events": n, "blocks": n // BLOCK,
            "discarded_tail_events": len(events) - n,
            "methods_share_one_repaired_array": True,
            "evaluation_mode": mode, "copied_from_seed": SEEDS[0] if clean_base else None}


def stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 and np.ptp(values) else 0.0,
            "min": float(values.min()), "max": float(values.max()),
            "range": float(np.ptp(values)), "n": len(values)}


def sign_counts(values) -> dict:
    values = np.asarray(values)
    return {"positive": int(np.count_nonzero(values > 0)),
            "zero": int(np.count_nonzero(values == 0)),
            "negative": int(np.count_nonzero(values < 0))}


def contrast_key(a, b):
    return f"{a}_minus_{b}"


def summarize(records: dict) -> dict:
    """Preserve the sequence and imputation axes; never treat 60 rows as cases."""
    summary = {"per_seed": {}, "per_sequence_seed_sensitivity": {},
               "group_seed_sensitivity": {},
               "interpretation": "descriptive imputation sensitivity, not sampling confidence intervals",
               "standard_deviation": "sample SD over the ten fixed imputation seeds (ddof=1)",
               "contrast_orientation": "first arm AUC minus second arm AUC; positive favors first arm"}
    for seed in SEEDS:
        groups = {}
        for group, names in GROUPS.items():
            arms = {arm: {metric: float(np.mean([records[name][str(seed)]["metrics"][arm][metric]
                                                for name in names])) for metric in METRICS} for arm in ARMS}
            contrasts = {contrast_key(a, b): {
                metric: {"mean": arms[a][metric] - arms[b][metric],
                         "sequence_sign_counts": sign_counts([
                             records[name][str(seed)]["metrics"][a][metric] -
                             records[name][str(seed)]["metrics"][b][metric] for name in names])}
                for metric in METRICS} for a, b in CONTRASTS}
            groups[group] = {"sequences": list(names), "arms": arms, "contrasts": contrasts}
        summary["per_seed"][str(seed)] = groups
    for name in SEQUENCES:
        arms = {arm: {metric: stats([records[name][str(seed)]["metrics"][arm][metric]
                                    for seed in SEEDS]) for metric in METRICS} for arm in ARMS}
        contrasts = {}
        for a, b in CONTRASTS:
            contrasts[contrast_key(a, b)] = {}
            for metric in METRICS:
                values = [records[name][str(seed)]["metrics"][a][metric] -
                          records[name][str(seed)]["metrics"][b][metric] for seed in SEEDS]
                contrasts[contrast_key(a, b)][metric] = {**stats(values), "seed_sign_counts": sign_counts(values)}
        summary["per_sequence_seed_sensitivity"][name] = {"arms": arms, "contrasts": contrasts}
    for group in GROUPS:
        arms = {arm: {metric: stats([summary["per_seed"][str(seed)][group]["arms"][arm][metric]
                                    for seed in SEEDS]) for metric in METRICS} for arm in ARMS}
        contrasts = {}
        for a, b in CONTRASTS:
            key = contrast_key(a, b)
            contrasts[key] = {}
            for metric in METRICS:
                values = [summary["per_seed"][str(seed)][group]["contrasts"][key][metric]["mean"] for seed in SEEDS]
                contrasts[key][metric] = {**stats(values), "seed_sign_counts": sign_counts(values)}
        summary["group_seed_sensitivity"][group] = {"arms": arms, "contrasts": contrasts}
    return summary


def verify_anchors(records: dict) -> dict:
    sources = {name: json.loads((ARTIFACTS / name).read_text()) for name in ANCHOR_FILES}
    results = {}
    for name in SEQUENCES:
        actual = records[name][str(SEEDS[0])]["metrics"]
        isolation = sources[ANCHOR_FILES[0]]["per_unit"]["BUAA_Campus/" + name]
        diffs = {}
        for arm in ARMS[:-1]:
            for metric in METRICS:
                diffs[f"phase69/{arm}/{metric}"] = actual[arm][metric] - isolation[arm][metric]
        for filename in ANCHOR_FILES[1:]:
            reference = sources[filename]["per_sequence"][name]["repaired"]
            for metric in METRICS:
                diffs[f"{filename}/ebf/{metric}"] = actual["ebf"][metric] - reference["ebf"][metric]
            if "beta0_magnitude" in reference:
                for metric in METRICS:
                    diffs[f"{filename}/beta0_magnitude/{metric}"] = (
                        actual["logdir_250us"][metric] - reference["beta0_magnitude"][metric])
        if max(abs(v) for v in diffs.values()) > 1e-12:
            raise ValueError(f"canonical-seed anchor failed for {name}: {diffs}")
        results[name] = diffs
    return {"canonical_seed": SEEDS[0], "absolute_tolerance": 1e-12,
            "all_passed": True, "differences": results}


def finish(payload: dict) -> None:
    records = payload["records"]
    complete = (set(records) == set(SEQUENCES) and
                all(set(records[name]) == set(map(str, SEEDS)) for name in SEQUENCES))
    payload["completed_records"] = sum(len(v) for v in records.values())
    payload["complete"] = complete
    if complete:
        consistency = {}
        for name in SEQUENCES:
            rows = [records[name][str(seed)] for seed in SEEDS]
            invariant_fields = ("raw_first3m_sha256", "labels_first3m_sha256",
                                "scored_labels_sha256", "target_mask_sha256")
            stable = all(len({row["hashes"][key] for row in rows}) == 1 for key in invariant_fields)
            stable = stable and len({row["repair_audit"]["corrupt_mask_sha256"] for row in rows}) == 1
            for seed, row in zip(SEEDS, rows):
                stable = stable and row["sequence"] == name and row["seed"] == seed
                stable = stable and row["scored_events"] == MAX_EVENTS // BLOCK * BLOCK
                stable = stable and row["blocks"] == MAX_EVENTS // BLOCK
                stable = stable and row["discarded_tail_events"] == MAX_EVENTS % BLOCK
            if not stable:
                raise ValueError(f"{name}: stream, mask or record identity changed across seeds")
            consistency[name] = {"raw_labels_mask_invariant": True,
                                 "distinct_repaired_input_hashes": len({row["hashes"]["repaired_first3m_sha256"] for row in rows})}
        payload["cross_seed_input_audit"] = consistency
        payload["anchor_validation"] = verify_anchors(records)
        invariance = {}
        for name in CLEAN:
            hashes = [records[name][str(seed)]["hashes"] for seed in SEEDS]
            metrics = [records[name][str(seed)]["metrics"] for seed in SEEDS]
            invariance[name] = {"independent_loads": len(hashes),
                                "hashes_identical": all(h == hashes[0] for h in hashes),
                                "metrics_identical": all(m == metrics[0] for m in metrics),
                                "all_corrupt_masks_empty": all(records[name][str(seed)]["repair_audit"]["corrupt_entries"] == 0 for seed in SEEDS)}
        if not all(all(v[k] for k in ("hashes_identical", "metrics_identical", "all_corrupt_masks_empty")) for v in invariance.values()):
            raise ValueError("clean-pair seed invariance failed")
        payload["clean_seed_invariance"] = invariance
        payload["summary"] = summarize(records)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BUAA_ROOT)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase78_buaa_repair_sensitivity.json")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    identity = run_identity()
    if args.output.exists():
        payload = json.loads(args.output.read_text())
        if payload.get("run_signature") != identity["run_signature"]:
            raise SystemExit("Resume refused: code, protocol, anchor or runtime signature changed")
    else:
        payload = {"phase": "78 BUAA polarity repair imputation sensitivity", **identity,
                   "records": {name: {} for name in SEQUENCES}, "complete": False,
                   "execution": {"workers": args.workers, "root": str(args.root.resolve())}}
    records = payload["records"]
    # Before resuming, reject replaced data even if all code and parameters match.
    for name in SEQUENCES:
        if records[name]:
            raw = np.load(args.root / f"{name}.npy", mmap_mode="r")[:MAX_EVENTS]
            known = next(iter(records[name].values()))["hashes"]["raw_first3m_sha256"]
            if digest_array(raw) != known:
                raise SystemExit(f"Resume refused: released input changed for {name}")
    start = time.monotonic()
    # Establish the canonical seed first, so clean reuse always has a scored anchor.
    for canonical_only in (True, False):
        jobs = []
        for name in SEQUENCES:
            for seed in ((SEEDS[0],) if canonical_only else SEEDS[1:]):
                if str(seed) not in records[name]:
                    base = records[name].get(str(SEEDS[0])) if name in CLEAN and not canonical_only else None
                    jobs.append((name, seed, base))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(evaluate_record, str(args.root), name, seed, base): (name, seed)
                       for name, seed, base in jobs}
            for future in as_completed(pending):
                name, seed = pending[future]
                records[name][str(seed)] = future.result()
                finish(payload)
                write_json(args.output, payload)
                print(f"completed {payload['completed_records']}/60: {name} seed {seed}", flush=True)
    finish(payload)
    payload["execution"]["last_invocation_wall_seconds"] = time.monotonic() - start
    write_json(args.output, payload)
    print(f"complete={payload['complete']}; anchors passed; saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
