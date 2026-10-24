"""Phase 21: uncertainty quantification and the block-mean/global AUC protocol gap.

Two review defects are answered here, both from the same pass over the labelled data.

**E1 -- no uncertainty anywhere.** Every AUC in the paper was a point estimate. Phase 14
kept condition-level means only, so no interval could be recovered after the fact. This
phase keeps the *per-block* AUC vector for every condition and reports:

  * per-condition mean with a contiguous-10-block cluster bootstrap CI,
  * the pooled 20-condition mean with a cluster bootstrap over conditions,
  * the paired `log_support - current` delta with a paired bootstrap CI and a win count,
    which is what the "improves 20 of 20 conditions" claim actually rests on.

Conventions (10,000 replicates, 10-block clusters) follow the existing
`dnd21_fixed_point_paired.json` so the new intervals match the ones already in the repo.

**E2 -- block-mean AUC is not the baselines' AUC.** The paper reports a mean over
per-block ROC AUCs; the transcribed baseline rows report AUC over events. These are
different statistics and averaging per-block AUCs can differ systematically, because a
block mean discards any inconsistency in score scale *between* blocks. `evaluate_dnd21.py`
has always computed `global_event_auc` but no artifact stored it, so the size of the gap
was unknown. Here both are computed on identical scores and blocks, so the difference is
attributable to the statistic alone.

Only the adopted configuration is measured: gate radius 1, score radius 2, float,
`s = beta*G - log1p(support)`, with the ring magnitude `current` carried alongside as the
paired comparator. Read-only; nothing is trained or fitted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import (
    dnd21_blocks,
    dvsclean_blocks,
    per_block_operands_both,
)
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase14_quantizable_operand import ring_support, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK_SIZE = 10_240
VARIANTS = ("log_support", "current")
BOOTSTRAP_REPLICATES = 10_000
CLUSTER_BLOCKS = 10
SEED = 20260730


def cluster_bootstrap(values: Sequence[float], rng: np.random.Generator,
                      cluster: int = 1, replicates: int = BOOTSTRAP_REPLICATES) -> Dict:
    """Percentile CI for the mean of `values`, resampling contiguous clusters.

    `cluster = 1` is an ordinary bootstrap over independent units (used when the unit is a
    whole condition). Larger clusters preserve short-range correlation between adjacent
    blocks of one recording.
    """

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    groups = [arr[i:i + cluster] for i in range(0, arr.size, cluster)]
    n = len(groups)
    if n < 2:
        return {"mean": float(arr.mean()), "n": int(arr.size), "ci95": None,
                "note": "too few clusters to bootstrap"}
    means = np.empty(replicates, dtype=np.float64)
    for rep in range(replicates):
        pick = rng.integers(0, n, size=n)
        means[rep] = np.concatenate([groups[i] for i in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(arr.mean()),
            "n": int(arr.size),
            "clusters": n,
            "cluster_size": cluster,
            "ci95": [float(lo), float(hi)],
            "bootstrap_se": float(means.std(ddof=1)),
            "replicates": replicates}


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, score_config: Optional[StateConfig],
                   max_blocks: Optional[int] = None) -> Dict:
    """Per-block AUCs plus the pooled-event AUC, for each variant, on identical blocks."""

    cfg_s = score_config if score_config is not None else config
    scale_count = len(config.taus_us)
    block_auc: Dict[str, List[float]] = {v: [] for v in VARIANTS}
    pooled_scores: Dict[str, List[np.ndarray]] = {v: [] for v in VARIANTS}
    pooled_labels: List[np.ndarray] = []
    produced = 0

    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue

        raw = bidirectional_features(events, config)
        g, r_gate, _ = per_block_operands_both(raw, scale_count, config.feature_dim)
        beta = 1.0 if safe_corr(g, r_gate) < 0.0 else 0.0

        raw_s = bidirectional_features(events, cfg_s) if score_config is not None else raw
        _, r_same, _ = per_block_operands_both(raw_s, scale_count, cfg_s.feature_dim)
        support = ring_support(raw_s, cfg_s)

        scores = {"log_support": beta * g - np.log1p(support),
                  "current": beta * g - r_same}
        # A block enters both statistics or neither, so the two are never computed on
        # different block sets -- that is the whole point of the comparison.
        if any(s.std() <= 0 for s in scores.values()):
            continue
        pooled_labels.append(labels)
        for name, score in scores.items():
            block_auc[name].append(float(roc_auc_score(labels, score)))
            pooled_scores[name].append(score)

    if not pooled_labels:
        return {"blocks": 0}

    labels_all = np.concatenate(pooled_labels)
    result: Dict = {"blocks": produced,
                    "blocks_scored": len(block_auc[VARIANTS[0]]),
                    "events_scored": int(labels_all.size),
                    "mean_block_auc": {}, "global_event_auc": {},
                    "block_auc_ci": {}, "block_aucs": {}}
    for name in VARIANTS:
        aucs = block_auc[name]
        result["mean_block_auc"][name] = float(np.mean(aucs))
        result["global_event_auc"][name] = float(
            roc_auc_score(labels_all, np.concatenate(pooled_scores[name])))
        result["block_auc_ci"][name] = cluster_bootstrap(
            aucs, np.random.default_rng(SEED), cluster=CLUSTER_BLOCKS)
        result["block_aucs"][name] = [float(a) for a in aucs]
    result["protocol_gap"] = {
        name: result["mean_block_auc"][name] - result["global_event_auc"][name]
        for name in VARIANTS}
    return result


def summarise(conditions: Dict[str, Dict], rng: np.random.Generator) -> Dict:
    """Pooled statistics over conditions, bootstrapping conditions as the unit."""

    usable = [c for c in conditions.values() if c.get("blocks_scored")]
    if not usable:
        return {}
    out: Dict = {"conditions": len(usable)}
    for stat in ("mean_block_auc", "global_event_auc"):
        out[stat] = {}
        for name in VARIANTS:
            per_condition = [c[stat][name] for c in usable]
            out[stat][name] = cluster_bootstrap(per_condition, rng, cluster=1)
    # The "20 of 20" claim, as a paired quantity.
    for stat in ("mean_block_auc", "global_event_auc"):
        deltas = [c[stat]["log_support"] - c[stat]["current"] for c in usable]
        out.setdefault("paired_support_minus_magnitude", {})[stat] = {
            **cluster_bootstrap(deltas, rng, cluster=1),
            "conditions_improved": int(sum(1 for d in deltas if d > 0)),
            "conditions_total": len(deltas),
            "min_delta": float(min(deltas)),
            "max_delta": float(max(deltas)),
        }
    return out


def emlb_cell_level_uncertainty(rng: np.random.Generator) -> Dict:
    """E-MLB gap CI at the only granularity the stored artifacts support: the cell.

    `phase18_retention_matched.json` holds, per cell, our MESR at the frozen r = 0.28 and
    the locally reproduced EDformer's MESR. Neither that file nor
    `phase10_emlb_blind_support.json` keeps per-sequence values, so the finest available
    unit is the cell and n = 8. This is a weak interval and is labelled as one; a
    per-sequence paired bootstrap would require re-running EDformer inference over all
    1152 sequences.
    """

    p18 = json.loads((ARTIFACTS / "phase18_retention_matched.json").read_text())
    cells, frozen, matched = [], [], []
    for group in ("D-END", "N-END"):
        for nd in ("ND00", "ND04", "ND16", "ND64"):
            cell = p18["results"][group][nd]
            cells.append(f"{group}/{nd}")
            frozen.append(float(cell["margin_frozen_0.28"]))
            matched.append(float(cell["margin_matched"]))

    def wins_binomial(margins: Sequence[float]) -> Dict:
        wins = sum(1 for m in margins if m > 0)
        n = len(margins)
        # Exact two-sided binomial p against p = 0.5, computed directly: no scipy needed.
        from math import comb
        probs = [comb(n, k) * 0.5 ** n for k in range(n + 1)]
        observed = probs[wins]
        p_two_sided = float(sum(p for p in probs if p <= observed + 1e-15))
        return {"cells_ahead": wins, "cells_total": n,
                "exact_binomial_p_two_sided": p_two_sided}

    def exact_permutation(margins: Sequence[float]) -> Dict:
        """Exact paired randomisation test: enumerate all 2^n sign assignments.

        A percentile bootstrap over eight points is not trustworthy -- with n = 8 the
        resampling distribution is coarse and the tails are estimated from a handful of
        distinct values. The paired permutation test has no such problem: under the null
        that the two methods are equivalent, the sign of each cell's margin is exchangeable,
        and with n = 8 all 256 assignments can simply be enumerated, so the p-value is
        exact rather than approximate. This does not enlarge n; it makes the inference valid
        at the n we have.
        """

        arr = np.asarray(margins, dtype=np.float64)
        n = arr.size
        observed = float(arr.mean())
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        means = (signs * arr).mean(axis=1)
        p = float(np.mean(np.abs(means) >= abs(observed) - 1e-15))
        return {"observed_mean_margin": observed,
                "exact_permutation_p_two_sided": p,
                "enumerations": int(2 ** n)}

    #: The operand ablation on the TEST set. Both rows are ours, scored on the same sequences at
    #: the same fixed r = 0.28, so retention is matched and the comparison is exactly paired --
    #: none of the missingness that makes the cross-method comparison unresolvable applies here.
    p10 = json.loads((ARTIFACTS / "phase10_emlb_blind_support.json").read_text())
    support, magnitude = [], []
    for part, nd in [tuple(c.split("/")) for c in cells]:
        cell = p10["results"][part][nd]
        support.append(float(cell["support_score_fixed_mesr"]))
        magnitude.append(float(cell["old_score_fixed_mesr"]))
    operand = [s - m for s, m in zip(support, magnitude)]

    return {
        "unit": "E-MLB cell (8 cells: {Daylight,Night} x ND1/4/16/64)",
        "operand_support_minus_magnitude": {
            "note": ("the paper's lead contribution, tested on the held-out real-sensor set. "
                     "Both operands are ours at the same frozen r = 0.28, so this is retention-"
                     "matched and exactly paired."),
            "mean_support": float(np.mean(support)),
            "mean_magnitude": float(np.mean(magnitude)),
            **cluster_bootstrap(operand, rng, cluster=1),
            **wins_binomial(operand), **exact_permutation(operand),
            "per_cell": dict(zip(cells, operand)),
        },
        "caveat": ("n = 8. Per-sequence values are not stored for either method, so a "
                   "per-sequence paired bootstrap is not recoverable from the artifacts; "
                   "regenerating them costs ~1.6 h for ours plus ~6.3 h of GPU re-inference "
                   "for EDformer (emlb_edformer_mesr.json wall_seconds = 22546). The exact "
                   "permutation p-value below is valid at n = 8 and is the primary test; the "
                   "bootstrap interval is reported for magnitude only."),
        "structure": ("the 8 cells are 2 lighting conditions x 4 ND levels and are not "
                      "exchangeable draws from one population; the per-cell margins are "
                      "listed so the structure is visible rather than averaged away"),
        "cells": cells,
        "per_cell_margin_frozen": dict(zip(cells, frozen)),
        "per_cell_margin_matched": dict(zip(cells, matched)),
        "frozen_r0.28": {**cluster_bootstrap(frozen, rng, cluster=1),
                         **wins_binomial(frozen), **exact_permutation(frozen)},
        "matched_retention": {**cluster_bootstrap(matched, rng, cluster=1),
                              **wins_binomial(matched), **exact_permutation(matched)},
        "note": ("margin = ours - reproduced EDformer, per cell. 'frozen' is the paper's "
                 "r = 0.28 operating point; 'matched' uses the baseline's own per-cell "
                 "retention, read from phase18."),
    }


def attribute_gap_to_gate(conditions: Dict[str, Dict[str, Dict]]) -> Dict:
    """Is the block-mean/event-level gap explained by beta varying within a condition?

    beta is decided per block, so a beta = 1 block and a beta = 0 block place their scores on
    different offsets. Ranking inside a block is unaffected; ranking across blocks is. If
    that is the mechanism, the gap should track how *mixed* a condition's beta is, where
    mixedness = min(frac_beta1, 1 - frac_beta1) and 0 means the whole condition agrees.
    `frac_blocks_beta1` is read from the Phase 19 artifact, which scored the same blocks.
    """

    p19 = json.loads((ARTIFACTS / "phase19_gate_ablation.json").read_text())
    mixedness, gaps, rows = [], [], []
    for dataset, per_condition in conditions.items():
        for name, cell in per_condition.items():
            if not cell.get("blocks_scored"):
                continue
            frac = float(p19["conditions"][dataset][name]["frac_blocks_beta1"])
            mix = min(frac, 1.0 - frac)
            gap = float(cell["protocol_gap"]["log_support"])
            mixedness.append(mix)
            gaps.append(gap)
            rows.append({"dataset": dataset, "condition": name,
                         "frac_blocks_beta1": frac, "mixedness": mix, "gap": gap})
    uniform = [r["gap"] for r in rows if r["mixedness"] < 0.01]
    return {
        "hypothesis": ("the gap is created by beta differing between blocks of one condition, "
                       "which shifts the score offset and degrades only cross-block ranking"),
        "pearson_r_mixedness_vs_gap": float(np.corrcoef(mixedness, gaps)[0, 1]),
        "uniform_beta_conditions": len(uniform),
        "uniform_beta_max_gap": max(uniform) if uniform else None,
        "uniform_beta_under_0.001": int(sum(1 for g in uniform if abs(g) < 0.001)),
        "max_gap_overall": max(gaps),
        "max_gap_condition": max(rows, key=lambda r: r["gap"])["condition"],
        "per_condition": sorted(rows, key=lambda r: -r["gap"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase21_uncertainty.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    def configs(w: int, h: int) -> Tuple[StateConfig, Optional[StateConfig]]:
        gate = StateConfig(width=w, height=h, taus_us=(250.0,), radius=1)
        score = (StateConfig(width=w, height=h, taus_us=(250.0,), radius=args.score_radius)
                 if args.score_radius != 1 else None)
        return gate, score

    dvs_gate, dvs_score = configs(1280, 720)
    dnd_gate, dnd_score = configs(346, 260)

    dvsclean: Dict[str, Dict] = {}
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            dvsclean[name] = scan_condition(
                dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE, args.max_blocks),
                dvs_gate, dvs_score, args.max_blocks)
            c = dvsclean[name]
            print(f"DVSCLEAN {name}: block-mean={c['mean_block_auc']['log_support']:.4f} "
                  f"global={c['global_event_auc']['log_support']:.4f} "
                  f"gap={c['protocol_gap']['log_support']:+.4f}", flush=True)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd_gate, dnd_score, args.max_blocks)
            c = dnd21[name]
            print(f"DND21 {name}: block-mean={c['mean_block_auc']['log_support']:.4f} "
                  f"global={c['global_event_auc']['log_support']:.4f} "
                  f"gap={c['protocol_gap']['log_support']:+.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    per_dataset = {"DND21": summarise(dnd21, rng), "DVSCLEAN": summarise(dvsclean, rng)}
    pooled = summarise({**dnd21, **dvsclean}, rng)

    payload = {
        "phase": "21 uncertainty quantification and the block-mean/global AUC protocol gap",
        "answers": {
            "E1": ("per-block AUC vectors retained, so per-condition and pooled CIs are "
                   "now derivable; E-MLB is covered only at cell granularity (n = 8)."),
            "E2": ("mean_block_auc and global_event_auc computed on identical scores and "
                   "blocks, isolating the statistic as the only difference."),
        },
        "config": {"gate_radius": 1, "score_radius": args.score_radius,
                   "block_size": BLOCK_SIZE, "arithmetic": "float",
                   "score": "s = beta*G - log1p(support); comparator s = beta*G - R_same",
                   "bootstrap": {"replicates": BOOTSTRAP_REPLICATES,
                                 "cluster_blocks_within_condition": CLUSTER_BLOCKS,
                                 "cluster_unit_across_conditions": "condition",
                                 "seed": SEED}},
        "audit": ("mean_block_auc/log_support must equal phase14_quantizable_operand.json's "
                  "log_support|float condition by condition; any drift means the two runs "
                  "did not see the same blocks."),
        "pooled_20": pooled,
        "per_dataset": per_dataset,
        "gap_attribution": attribute_gap_to_gate({"DND21": dnd21, "DVSCLEAN": dvsclean}),
        "emlb": emlb_cell_level_uncertainty(rng),
        "conditions": {"DND21": dnd21, "DVSCLEAN": dvsclean},
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")
    for stat in ("mean_block_auc", "global_event_auc"):
        s = pooled[stat]["log_support"]
        print(f"pooled {stat:17s} = {s['mean']:.4f}  95% CI [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
