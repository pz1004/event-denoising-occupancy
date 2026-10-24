"""Corrected matched-horizon operand controls on development and BUAA.

Schema 2 exposes raw directional amplitudes and applies each requested logarithm
exactly once. sum_flat uses exact integer whole-block event counts; logdir_flat
uses exact integer past/future counts. tau=1e9 is retained only as a named
near-flat diagnostic. Legacy schema-1 artifacts are preserved as mislabelled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .matched_horizon import block_scores, SCORE_DEFINITIONS
from .phase57_ebf_labelled import GEOMETRY, conditions, load_condition
from .phase62_buaa_campus import BUAA_ROOT, SEQUENCES, load_sequence
from .phase62_buaa_campus import HEIGHT as BUAA_HEIGHT, WIDTH as BUAA_WIDTH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK = 10_240
SCORE_RADIUS = 2

# None denotes exact zero decay, not a finite approximation.
TAUS_US = (250.0, 1_000.0, 4_000.0, 16_000.0, None)
TAU_NAMES = ("sum_250us", "sum_1ms", "sum_4ms", "sum_16ms", "sum_flat")
LOGDIR_NAMES = tuple(name.replace("sum_", "logdir_") for name in TAU_NAMES)
ARMS = (("count",) + TAU_NAMES + LOGDIR_NAMES + ("sum_nearflat", "logdir_nearflat") +
        ("logdir64_250us", "logdir64_flat", "logdir64_nearflat", "logproduct_flat"))
SCORE_SIGNATURE = "explicit_raw_canonical_float32_logs_exact_integer_flat_v1"


def score_condition(events: np.ndarray, width: int, height: int
                    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Corrected controls on complete fixed-count blocks."""
    n = len(events) - (len(events) % BLOCK)
    out = {arm: np.empty(n, dtype=np.float64) for arm in ARMS}
    support_mean = []
    for start in range(0, n, BLOCK):
        stop = start + BLOCK
        values, support = block_scores(events[start:stop], width, height, SCORE_RADIUS)
        for arm in ARMS:
            out[arm][start:stop] = values[arm]
        support_mean.append(float(support.mean()))
    return out, np.asarray(support_mean)


def _support_at(raw: np.ndarray, config: StateConfig, scale_count: int,
                index: int) -> np.ndarray:
    """Occupancy count read from one decay plane of the bank."""

    neighbor_count = (2 * config.radius + 1) ** 2 - 1
    base = 7 * scale_count
    columns = [base + k * scale_count + index for k in range(neighbor_count)]
    stack = np.stack([raw[:, c] + raw[:, config.feature_dim + c] for c in columns], axis=1)
    return (stack > 0).sum(axis=1).astype(np.float64)


def block_aucs(scores: np.ndarray, labels: np.ndarray) -> List[float]:
    out: List[float] = []
    for start in range(0, len(scores) - BLOCK + 1, BLOCK):
        y = labels[start:start + BLOCK]
        if y.min() == y.max():
            continue
        out.append(float(roc_auc_score(y, scores[start:start + BLOCK])))
    return out


def measure(events: np.ndarray, labels: np.ndarray, width: int, height: int) -> Dict:
    scores, support_mean = score_condition(events, width, height)
    n = len(scores["count"])
    y = labels[:n]
    record: Dict = {"events": int(n), "noise_fraction": float(y.mean()),
                    "mean_support": float(support_mean.mean())}
    # The log compression cannot change an AUC. Asserted, not assumed.
    plain = float(roc_auc_score(y, scores["count"]))
    logged = float(roc_auc_score(y, -np.log1p(-scores["count"])))
    if abs(plain - logged) > 1e-12:
        raise SystemExit("log1p changed an AUC; the ranking argument is wrong")
    record["log1p_invariance_checked"] = True
    for arm in ARMS:
        aucs = block_aucs(scores[arm], y)
        record[arm] = {"block_mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
                       "event_auc": float(roc_auc_score(y, scores[arm])),
                       "blocks": len(aucs)}
    return record


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def runtime_metadata() -> Dict:
    import numba
    import sklearn
    files = ("core.py", "matched_horizon.py", "phase69_operand_isolation.py",
             "phase70_operand_isolation_led.py")
    directory = Path(__file__).resolve().parent
    return {"score_signature": SCORE_SIGNATURE,
            "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "numba": numba.__version__, "sklearn": sklearn.__version__},
            "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                              for name in files}}


def archive_legacy(path: Path) -> None:
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("schema_version") != 2:
            legacy = path.with_name(path.stem + ".legacy_mislabelled.json")
            if not legacy.exists():
                legacy.write_text(json.dumps({**old,
                    "obsolete_reason": "sum_* contained one log and logdir_* two logs; use corrected schema 2",
                    "superseded_by": path.name}, indent=2, sort_keys=True))


def compare_legacy(path: Path, records: Dict, key: str) -> Dict:
    legacy_path = path.with_name(path.stem + ".legacy_mislabelled.json")
    if not legacy_path.exists():
        return {}
    old = json.loads(legacy_path.read_text())[key]
    count_drifts, single_log_drifts = [], []
    for name, record in records.items():
        if name not in old:
            continue
        if record["benchmark"] == "BUAA_Campus":
            # Legacy isolation used 20260826, whereas the main BUAA/phase72
            # scorer used 20260825 for repairing leaked noise polarities.
            # Current runs align to the main scorer; compare those inputs in
            # audit_matched_horizon_results rather than unlike legacy streams.
            continue
        count_drifts.append(abs(record["count"]["block_mean_auc"] - old[name]["count"]["block_mean_auc"]))
        single_log_drifts.append(abs(record["logdir_nearflat"]["block_mean_auc"] - old[name]["sum_flat"]["block_mean_auc"]))
    if count_drifts and max(count_drifts) > 1e-12:
        raise AssertionError("occupancy AUC changed from the independent legacy extractor")
    return {"units_checked": len(count_drifts),
            "excluded_benchmark": "BUAA_Campus: aligned repair seed to established main scorer (20260825 vs legacy 20260826)",
            "max_occupancy_auc_drift": max(count_drifts, default=0),
            "max_nearflat_single_log_auc_drift": max(single_log_drifts, default=0),
            "single_log_drift_note": "legacy isolation added float32 logged features in float64; corrected canonical arm uses phase8 float32 addition"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--buaa-root", type=Path, default=BUAA_ROOT)
    parser.add_argument("--seed", type=int, default=20260825,
                        help="BUAA polarity-repair seed, matching phase62/phase72")
    parser.add_argument("--resume", action="store_true", help="Resume schema-2 per-unit checkpoints")
    parser.add_argument("--max-conditions", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase69_operand_isolation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    archive_legacy(args.output)
    per_unit: Dict[str, Dict] = {}
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("schema_version") != 2 or previous.get("score_signature") != SCORE_SIGNATURE:
            raise ValueError("cannot resume legacy score definitions")
        per_unit = previous.get("per_unit", {})
        if previous.get("seed") != args.seed and any(
                row["benchmark"] == "BUAA_Campus" for row in per_unit.values()):
            raise ValueError("cannot resume BUAA units with a different repair seed")

    def checkpoint():
        write_json(args.output, {"schema_version": 2, "complete": False,
                                "score_signature": SCORE_SIGNATURE,
                                "seed": args.seed,
                                "score_definitions": SCORE_DEFINITIONS,
                                "per_unit": per_unit})

    todo = conditions()
    if args.max_conditions:
        todo = todo[:args.max_conditions]
    for dataset, name in todo:
        if f"{dataset}/{name}" in per_unit:
            continue
        step = time.perf_counter()
        width, height = GEOMETRY[dataset]
        events, labels = load_condition(dataset, name, args.dnd21_root, args.dvsclean_root)
        record = measure(events, labels, width, height)
        record["benchmark"] = dataset
        per_unit[f"{dataset}/{name}"] = record
        checkpoint()
        print(f"{dataset}/{name:22s} count={record['count']['block_mean_auc']:.4f} "
              f"flat={record['sum_flat']['block_mean_auc']:.4f} "
              f"250us={record['sum_250us']['block_mean_auc']:.4f} "
              f"occ={record['mean_support']:5.2f} [{time.perf_counter() - step:.0f}s]",
              flush=True)

    for sequence in SEQUENCES:
        if f"BUAA_Campus/{sequence}" in per_unit:
            continue
        step = time.perf_counter()
        events, labels, _ = load_sequence(args.buaa_root / f"{sequence}.npy", True, args.seed)
        record = measure(events, labels, BUAA_WIDTH, BUAA_HEIGHT)
        record["benchmark"] = "BUAA_Campus"
        per_unit[f"BUAA_Campus/{sequence}"] = record
        checkpoint()
        print(f"BUAA_Campus/{sequence:16s} count={record['count']['block_mean_auc']:.4f} "
              f"flat={record['sum_flat']['block_mean_auc']:.4f} "
              f"250us={record['sum_250us']['block_mean_auc']:.4f} "
              f"occ={record['mean_support']:5.2f} [{time.perf_counter() - step:.0f}s]",
              flush=True)

    summary: Dict[str, Dict] = {}
    groups = {"development": lambda r: r["benchmark"] in GEOMETRY,
              "DND21": lambda r: r["benchmark"] == "DND21",
              "DVSCLEAN": lambda r: r["benchmark"] == "DVSCLEAN",
              "BUAA_Campus": lambda r: r["benchmark"] == "BUAA_Campus"}
    for label, keep in groups.items():
        units = [r for r in per_unit.values() if keep(r)]
        if not units:
            continue
        entry = {"units": len(units),
                 "mean_support": float(np.mean([u["mean_support"] for u in units]))}
        for arm in ARMS:
            entry[arm] = {f: float(np.mean([u[arm][f] for u in units]))
                          for f in ("block_mean_auc", "event_auc")}
        entry["count_beats_flat_units"] = int(sum(
            u["count"]["block_mean_auc"] > u["sum_flat"]["block_mean_auc"] for u in units))
        entry["count_beats_250us_units"] = int(sum(
            u["count"]["block_mean_auc"] > u["sum_250us"]["block_mean_auc"] for u in units))
        summary[label] = entry

    payload = {
        "schema_version": 2, "complete": True,
        **runtime_metadata(),
        "seed": args.seed,
        "seed_scope": "BUAA repaired polarities only; canonical default matches phase62/phase72",
        "full_benchmark_run": args.max_conditions is None and len(per_unit) == 26,
        "score_definitions": SCORE_DEFINITIONS,
        "flat_control": "exact integer counts; no decay approximation",
        "nearflat_tau_us": 1e9,
        "phase": "69 operand isolation: amplitude at a matched horizon",
        "question": ("Does discarding amplitude help once the magnitude control is given "
                     "the same temporal horizon as the count?"),
        "taus_us": list(TAUS_US),
        "tau_names": list(TAU_NAMES),
        "matched_control": "sum_flat",
        "block": BLOCK,
        "score_radius": SCORE_RADIUS,
        "summary": summary,
        "per_unit": per_unit,
        "wall_seconds": time.perf_counter() - started,
    }
    payload["legacy_anchor_comparison"] = compare_legacy(args.output, per_unit, "per_unit")
    write_json(args.output, payload)
    print(f"\nwrote {args.output}")
    for label, s in summary.items():
        row = " ".join(f"{a}={s[a]['block_mean_auc']:.4f}" for a in ARMS)
        print(f"  {label:14s} n={s['units']:3d} {row} "
              f"| count>flat on {s['count_beats_flat_units']}/{s['units']}")


if __name__ == "__main__":
    main()
