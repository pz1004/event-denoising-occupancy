"""Phase 53: event-level and rank-normalised block-size sensitivity.

Block-mean AUC changes its aggregation when block size N changes. Larger blocks
also grant more lookahead. The block-mean value 0.9937 at four times N therefore
requires comparison using a common event population.

Report two additional aggregations:

    event_level     one ROC-AUC per condition over all scored events, with the
                    same event set and labels at every N;
    rank_normalised replace each score by its within-block rank divided by block
                    size before pooling, removing between-block score offsets.

Compare these with block-mean AUC to distinguish aggregation effects from changes
in event ranking. Everything except block size follows Phase 28 and Phase 19:
20 labelled conditions, the radius-1 key corr(G,R_same), the gated score
``s=beta*G-log(1+S_r2)``, and float64 arithmetic. The block-mean column must reproduce
Phase 28 at every N; the audit reports deviations and refuses large mismatches.
Nothing is trained, fitted or selected.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks, per_block_operands_both
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase14_quantizable_operand import ring_support, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
PHASE28 = ARTIFACTS / "phase28_block_size_sweep.json"

#: N/8 .. 8N around the fixed 10,240-event configuration.
BLOCK_SIZES: Tuple[int, ...] = (1280, 2560, 5120, 10240, 20480, 40960, 81920)
ADOPTED_BLOCK_SIZE = 10240
AUDIT_TOLERANCE = 1e-4
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260825

MODES = ("gated", "beta0", "beta1")
AGGREGATIONS = ("block_mean", "event_level", "rank_normalised")


def _rank_normalise(values: np.ndarray) -> np.ndarray:
    """Within-block rank in [0, 1); ties take their average rank, as `rankdata` does."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # average ties so the transform is a function of the score, not of arrival order
    uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(uniq) != len(values):
        sums = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks / max(len(values), 1)


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   gate_config: StateConfig, score_config: StateConfig) -> Dict:
    """Score one condition at one block size under all three aggregations."""

    scale_count = len(gate_config.taus_us)
    per_block: Dict[str, List[float]] = {m: [] for m in MODES}
    pooled_scores: Dict[str, List[np.ndarray]] = {m: [] for m in MODES}
    pooled_ranks: Dict[str, List[np.ndarray]] = {m: [] for m in MODES}
    pooled_labels: List[np.ndarray] = []
    corr: List[float] = []
    produced = 0
    scored = 0
    events_seen = 0

    for events, labels in block_iter:
        produced += 1
        events_seen += len(events)
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue

        raw = bidirectional_features(events, gate_config)
        g, r_same, _ = per_block_operands_both(raw, scale_count, gate_config.feature_dim)
        raw_score = bidirectional_features(events, score_config)
        subtrahend = np.log1p(ring_support(raw_score, score_config))

        rho = safe_corr(g, r_same)
        score = {"beta0": -subtrahend, "beta1": g - subtrahend}
        score["gated"] = score["beta1"] if rho < 0.0 else score["beta0"]
        if score["beta0"].std() <= 0 or score["beta1"].std() <= 0:
            continue

        scored += 1
        corr.append(rho)
        pooled_labels.append(labels)
        for mode in MODES:
            per_block[mode].append(roc_auc_score(labels, score[mode]))
            pooled_scores[mode].append(score[mode])
            pooled_ranks[mode].append(_rank_normalise(score[mode]))

    if not scored:
        return {"blocks": produced, "scored_blocks": 0}

    labels_all = np.concatenate(pooled_labels)
    out: Dict = {
        "blocks": produced,
        "scored_blocks": scored,
        "events": events_seen,
        "events_scored": int(labels_all.size),
        "frac_blocks_beta1": float(np.mean(np.asarray(corr) < 0.0)),
        "mean_corr": float(np.mean(corr)),
    }
    for mode in MODES:
        out[f"block_mean|{mode}"] = float(np.mean(per_block[mode]))
        out[f"event_level|{mode}"] = float(
            roc_auc_score(labels_all, np.concatenate(pooled_scores[mode])))
        out[f"rank_normalised|{mode}"] = float(
            roc_auc_score(labels_all, np.concatenate(pooled_ranks[mode])))
    return out


COLUMNS = tuple(f"{a}|{m}" for a in AGGREGATIONS for m in MODES) + (
    "frac_blocks_beta1", "mean_corr")


def aggregate(conditions: Dict[str, Dict]) -> Dict[str, float]:
    usable = [c for c in conditions.values() if c.get("scored_blocks")]
    if not usable:
        return {}
    out = {name: float(np.mean([c[name] for c in usable])) for name in COLUMNS}
    out["conditions"] = len(usable)
    out["scored_blocks"] = int(sum(c["scored_blocks"] for c in usable))
    return out


def paired_interval(conditions: Dict[str, Dict], left: str, right: str,
                    rng: np.random.Generator) -> Dict[str, float]:
    """Percentile bootstrap over conditions. n = 20, clustering into 2 datasets."""

    diffs = np.asarray([c[left] - c[right]
                        for c in conditions.values() if c.get("scored_blocks")])
    if diffs.size == 0:
        return {}
    draws = rng.choice(diffs, size=(BOOTSTRAP_REPLICATES, diffs.size), replace=True)
    means = draws.mean(axis=1)
    return {"point": float(diffs.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "n_conditions": int(diffs.size)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=list(BLOCK_SIZES))
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase53_block_size_eventlevel.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reference = json.loads(PHASE28.read_text())["sweep"]

    def configs(width: int, height: int) -> Tuple[StateConfig, StateConfig]:
        gate = StateConfig(width=width, height=height, taus_us=(250.0,), radius=1)
        score = StateConfig(width=width, height=height, taus_us=(250.0,),
                            radius=args.score_radius)
        return gate, score

    dvs_gate, dvs_score = configs(1280, 720)
    dnd_gate, dnd_score = configs(346, 260)

    sweep: Dict[str, Dict] = {}
    audit: Dict[str, float] = {}
    for block_size in args.block_sizes:
        step = time.perf_counter()

        dvsclean: Dict[str, Dict] = {}
        for seq in SEQUENCES:
            for level in (50, 100):
                name = f"{seq}_{level}"
                dvsclean[name] = scan_condition(
                    dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5",
                                    block_size, args.max_blocks),
                    dvs_gate, dvs_score)

        dnd21: Dict[str, Dict] = {}
        for rate in DND21_PAPER:
            for scene, fname in SCENE_FILES.items():
                name = f"{rate}_{scene}"
                dnd21[name] = scan_condition(
                    dnd21_blocks(args.dnd21_root / rate / fname,
                                 block_size, args.max_blocks),
                    dnd_gate, dnd_score)

        per_dataset = {"DND21": aggregate(dnd21), "DVSCLEAN": aggregate(dvsclean)}
        merged = {**dnd21, **dvsclean}
        overall = {name: float(np.mean([per_dataset["DND21"][name],
                                        per_dataset["DVSCLEAN"][name]]))
                   for name in COLUMNS}

        # audit against Phase 28's block-mean series, dataset by dataset
        ref = reference.get(str(block_size))
        if ref is not None:
            for dataset in ("DND21", "DVSCLEAN"):
                for mode in MODES:
                    key = {"gated": "gated", "beta0": "beta0", "beta1": "beta1"}[mode]
                    delta = per_dataset[dataset][f"block_mean|{mode}"] - \
                        ref["per_dataset"][dataset][key]
                    audit[f"{dataset}_{mode}_N{block_size}"] = float(delta)

        sweep[str(block_size)] = {
            "per_dataset": per_dataset,
            "overall_20": overall,
            "conditions": merged,
            "wall_seconds": time.perf_counter() - step,
        }
        print(f"N={block_size:6d}  "
              f"block_mean={overall['block_mean|gated']:.4f}  "
              f"event_level={overall['event_level|gated']:.4f}  "
              f"rank_norm={overall['rank_normalised|gated']:.4f}  "
              f"[{time.perf_counter() - step:.0f}s]", flush=True)

    worst = max((abs(v) for v in audit.values()), default=0.0)
    if worst > AUDIT_TOLERANCE:
        raise SystemExit(
            f"block-mean column deviates from Phase 28 by {worst:.3e} > {AUDIT_TOLERANCE}; "
            "the frozen configuration is not being reproduced, so the new columns are "
            "not comparable to the published series")

    # Differences against the frozen block size, under each aggregation.
    adopted = str(ADOPTED_BLOCK_SIZE)
    versus_adopted: Dict[str, Dict] = {}
    for block_size in args.block_sizes:
        if str(block_size) == adopted:
            continue
        merged = sweep[str(block_size)]["conditions"]
        base = sweep[adopted]["conditions"]
        entry: Dict[str, Dict] = {}
        for aggregation in AGGREGATIONS:
            key = f"{aggregation}|gated"
            diffs = np.asarray([merged[c][key] - base[c][key]
                                for c in base if merged.get(c, {}).get("scored_blocks")
                                and base[c].get("scored_blocks")])
            draws = rng.choice(diffs, size=(BOOTSTRAP_REPLICATES, diffs.size), replace=True)
            means = draws.mean(axis=1)
            entry[aggregation] = {"point": float(diffs.mean()),
                                  "lo": float(np.percentile(means, 2.5)),
                                  "hi": float(np.percentile(means, 97.5)),
                                  "conditions_improved": int((diffs > 0).sum()),
                                  "n_conditions": int(diffs.size)}
        versus_adopted[str(block_size)] = entry

    payload = {
        "phase": "53 block-size sweep under aggregations block size cannot inflate",
        "question": ("Section 5 asserts the block-mean gain from larger N is an "
                     "aggregation artefact. Event-level and within-block rank-normalised "
                     "AUC test that assertion directly."),
        "frozen": "everything except block_size; 20 labelled conditions, gate radius 1, "
                  "score radius 2, taus_us=[250.0], float64",
        "score": "s = beta*G - log(1 + S_r2)",
        "aggregations": {
            "block_mean": "mean over blocks of the per-block ROC-AUC; a different "
                          "statistic at every N",
            "event_level": "one ROC-AUC per condition over all scored events; the event "
                           "set and labels are identical at every N",
            "rank_normalised": "scores replaced by within-block average rank / block size "
                               "before pooling; removes every between-block offset",
        },
        "block_sizes": list(args.block_sizes),
        "adopted_block_size": ADOPTED_BLOCK_SIZE,
        "audit_vs_phase28_block_mean": audit,
        "audit_worst_abs_deviation": worst,
        "bootstrap": {"method": "percentile, resampling conditions",
                      "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "unit_caveat": "n = 20 conditions clustering into 2 datasets"},
        "sweep": sweep,
        "versus_adopted_block_size": versus_adopted,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\naudit worst |deviation| vs Phase 28 block-mean: {worst:.3e}")
    print(f"{'N':>7}  {'block_mean':>11}  {'event_level':>11}  {'rank_norm':>11}")
    for block_size in args.block_sizes:
        o = sweep[str(block_size)]["overall_20"]
        print(f"{block_size:7d}  {o['block_mean|gated']:11.4f}  "
              f"{o['event_level|gated']:11.4f}  {o['rank_normalised|gated']:11.4f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
