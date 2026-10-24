"""Phase 23: does each ablated component still earn its place under the event-level AUC?

Phase 21 found that the block-mean and event-level AUC differ by 0.0225 pooled, and that the
difference is *caused by the gate*: beta is decided per block, so beta = 0 and beta = 1 blocks
put their scores on different offsets and only cross-block ranking suffers.

That raises an obvious and uncomfortable question. Table 4 of the manuscript ablates the gate,
the operand, the radius and the lookahead, and every number in it is a block mean. The gate is
precisely the component whose measured value the block-mean statistic could be flattering,
because the gate is what breaks cross-block comparability in the first place. If the gate's
contribution shrinks or reverses under the event-level statistic, the manuscript's central
architectural claim needs rewording.

This phase re-runs the two statistic-sensitive ablations under BOTH statistics on identical
blocks:

    gated       s = beta*G - log1p(S)      the adopted score
    beta = 0    s =      - log1p(S)        ring only; no G, so no beta-induced offset
    beta = 1    s =        G - log1p(S)     always global; also a single offset
    magnitude   s = beta*G - R_same        the operand ablation, gate still on

beta = 0 and beta = 1 are single-branch scores and should show little gap; the gated score is
the mixed one. Whether the gate still helps once that advantage is removed is the question.

**And whether the damage is repairable.** If the gap is only a per-branch *offset* -- beta = 0
scores live in [-log 25, 0] while beta = 1 scores are displaced by G -- then it is not a ranking
failure and a within-block monotone rescaling should undo it. So each score is also pooled after
being replaced by its within-block average rank. That transform costs nothing and is already
implicit in the pipeline: retention keeps the lowest-scoring fraction *of each block*, so the
deployed filter only ever uses within-block order anyway.

Read-only; nothing is trained or fitted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata
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
from .phase21_uncertainty import cluster_bootstrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK_SIZE = 10_240
VARIANTS = ("gated", "beta0", "beta1", "magnitude_gated")
STATS = ("mean_block_auc", "global_event_auc", "rank_normalised_event_auc")
SEED = 20260730


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, score_config: Optional[StateConfig],
                   max_blocks: Optional[int] = None) -> Dict:
    cfg_s = score_config if score_config is not None else config
    scale_count = len(config.taus_us)
    block_auc: Dict[str, List[float]] = {v: [] for v in VARIANTS}
    pooled: Dict[str, List[np.ndarray]] = {v: [] for v in VARIANTS}
    pooled_rank: Dict[str, List[np.ndarray]] = {v: [] for v in VARIANTS}
    labels_all: List[np.ndarray] = []
    beta_sum = 0
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
        support = np.log1p(ring_support(raw_s, cfg_s))

        scores = {"gated": beta * g - support,
                  "beta0": -support,
                  "beta1": g - support,
                  "magnitude_gated": beta * g - r_same}
        if any(s.std() <= 0 for s in scores.values()):
            continue
        beta_sum += int(beta)
        labels_all.append(labels)
        for name, score in scores.items():
            block_auc[name].append(float(roc_auc_score(labels, score)))
            pooled[name].append(score)
            # Average ranks, not ordinal: log1p(support) takes at most 25 distinct values, so
            # ties are common and breaking them arbitrarily would corrupt the pooled AUC.
            pooled_rank[name].append(rankdata(score, method="average") / score.size)

    if not labels_all:
        return {"blocks": 0}

    flat_labels = np.concatenate(labels_all)
    scored = len(block_auc["gated"])
    out: Dict = {"blocks": produced, "blocks_scored": scored,
                 "frac_blocks_beta1": beta_sum / scored if scored else None,
                 "mean_block_auc": {}, "global_event_auc": {},
                 "rank_normalised_event_auc": {}, "protocol_gap": {}, "gap_after_rank": {}}
    for name in VARIANTS:
        out["mean_block_auc"][name] = float(np.mean(block_auc[name]))
        out["global_event_auc"][name] = float(
            roc_auc_score(flat_labels, np.concatenate(pooled[name])))
        out["rank_normalised_event_auc"][name] = float(
            roc_auc_score(flat_labels, np.concatenate(pooled_rank[name])))
        out["protocol_gap"][name] = out["mean_block_auc"][name] - out["global_event_auc"][name]
        out["gap_after_rank"][name] = (out["mean_block_auc"][name]
                                       - out["rank_normalised_event_auc"][name])
    return out


def summarise(conditions: Dict[str, Dict], rng: np.random.Generator) -> Dict:
    usable = [c for c in conditions.values() if c.get("blocks_scored")]
    if not usable:
        return {}
    out: Dict = {"conditions": len(usable)}
    for stat in STATS:
        out[stat] = {n: cluster_bootstrap([c[stat][n] for c in usable], rng, cluster=1)
                     for n in VARIANTS}
    # What the gate is worth: gated minus the better of the two fixed branches, paired.
    out["gate_value_vs_best_fixed"] = {}
    for stat in STATS:
        deltas, losers = [], 0
        for c in usable:
            best_fixed = max(c[stat]["beta0"], c[stat]["beta1"])
            deltas.append(c[stat]["gated"] - best_fixed)
            losers += int(c[stat]["gated"] < best_fixed)
        out["gate_value_vs_best_fixed"][stat] = {
            **cluster_bootstrap(deltas, rng, cluster=1),
            "conditions_where_gate_loses": losers,
            "conditions_total": len(deltas),
        }
    # And versus each fixed branch separately, which is what Table 4 reports.
    out["gate_value_vs_each_branch"] = {}
    for stat in STATS:
        out["gate_value_vs_each_branch"][stat] = {
            branch: cluster_bootstrap([c[stat]["gated"] - c[stat][branch] for c in usable],
                                      rng, cluster=1)
            for branch in ("beta0", "beta1")}
    out["operand_value"] = {
        stat: cluster_bootstrap([c[stat]["gated"] - c[stat]["magnitude_gated"] for c in usable],
                                rng, cluster=1)
        for stat in STATS}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase23_ablation_statistic.json")
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
            print(f"DVSCLEAN {name}: gated block={c['mean_block_auc']['gated']:.4f} "
                  f"event={c['global_event_auc']['gated']:.4f} | "
                  f"beta0 event={c['global_event_auc']['beta0']:.4f}", flush=True)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd_gate, dnd_score, args.max_blocks)
            c = dnd21[name]
            print(f"DND21 {name}: gated block={c['mean_block_auc']['gated']:.4f} "
                  f"event={c['global_event_auc']['gated']:.4f} | "
                  f"beta0 event={c['global_event_auc']['beta0']:.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    payload = {
        "phase": "23 ablation under both AUC statistics",
        "question": ("the block-mean/event-level gap is caused by the gate, and Table 4 measures "
                     "the gate with the block-mean statistic. Does the gate still pay under the "
                     "event-level statistic?"),
        "config": {"gate_radius": 1, "score_radius": args.score_radius,
                   "block_size": BLOCK_SIZE, "arithmetic": "float", "seed": SEED},
        "audit": ("mean_block_auc/gated must match phase21's mean_block_auc/log_support and "
                  "mean_block_auc/magnitude_gated must match phase21's current, condition by "
                  "condition."),
        "pooled_20": summarise({**dnd21, **dvsclean}, rng),
        "per_dataset": {"DND21": summarise(dnd21, rng),
                        "DVSCLEAN": summarise(dvsclean, rng)},
        "conditions": {"DND21": dnd21, "DVSCLEAN": dvsclean},
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s\n")

    p = payload["pooled_20"]
    print(f"{'variant':18s} {'block-mean':>10s} {'event-level':>12s} {'event+rank':>12s}")
    for name in VARIANTS:
        print(f"{name:18s} {p['mean_block_auc'][name]['mean']:10.4f} "
              f"{p['global_event_auc'][name]['mean']:12.4f} "
              f"{p['rank_normalised_event_auc'][name]['mean']:12.4f}")
    print()
    for stat in STATS:
        v = p["gate_value_vs_best_fixed"][stat]
        print(f"gate vs BEST fixed branch, {stat:17s}: {v['mean']:+.4f} "
              f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  "
              f"gate loses in {v['conditions_where_gate_loses']}/{v['conditions_total']}")


if __name__ == "__main__":
    main()
