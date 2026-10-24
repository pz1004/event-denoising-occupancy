"""Phase 32: ED24 as a third labelled benchmark, at full scene coverage.

Section 4.4 of the manuscript says a third labelled dataset with a different noise model
"is the single measurement that would most improve Table 1, and we do not have one". That
sentence is wrong: ED24 is on disk, it has a loader that runs, and two earlier phases
already score it. What they do not do is score it as a *benchmark*. Phase 4 and Phase 22b
read 16 of the 100 scenes, capped at 8 blocks per condition, for one purpose only -- to
check that theta = 0 is not a DND21/DVSCLEAN coincidence. This phase reads all 100 scenes
with no block cap and reports the statistics Table 1 and Table 6 report, so that ED24 can
enter the paper as a labelled benchmark rather than a threshold sanity check.

What that buys, and it is the paper's largest open weakness:

  * The effective sample size behind every labelled interval is currently 2 -- two datasets
    differing in noise model, sensor geometry, density regime and provenance, with the 20
    conditions nested inside them (Section 4.4). ED24 is a third cluster, and unlike
    DVSCLEAN it needs no reconstruction, so its rows are protocol-matched by construction.
  * The lead contribution is "the count improves 20 of 20 labelled conditions". ED24 adds
    600 more scene x level conditions under a third noise model. Either the claim survives
    that, in which case it is a much stronger claim, or it does not, in which case the
    paper needs to know before a reviewer does.

WHAT THIS SET IS NOT. ED24 is not held out the way E-MLB is. Phase 4 and Phase 22b already
looked at it: theta = 0 was checked against an ED24-fitted threshold, on one side only, and
no constant was changed as a result (the fitted plateau [-0.40, -0.03] excludes zero, and
carrying the fitted value to the dev sets *costs* 0.0005). So ED24 is prior-contact data
for exactly one constant. Any text this artifact feeds must say so. It is a third noise
model and a third cluster; it is not a second held-out set, and E-MLB remains the only
benchmark nothing in the paper has ever been selected or checked on.

Arms, all on identical blocks so every comparison is exactly paired:

    adopted    s = beta*G - log(1 + S_r2)     the filter the paper proposes
    magnitude  s = beta*G - R_r2              the conventional operand it replaces
    beta0/1    the two gate branches, for both operands

Operand construction, the gate key and beta are taken verbatim from
`phase14_quantizable_operand.scan_condition`, so an ED24 row is the same measurement as a
DND21 row and the two are directly comparable. Read-only; nothing is trained.
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
from .evaluate_adaptive_beta import per_block_operands_both
from .phase4_ed24_blind import ED24_LEVELS, ED24_ROOT, ed24_blocks
from .phase14_quantizable_operand import ring_support, safe_corr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK_SIZE = 10_240
SENSOR = (346, 260)

#: Phase 22b's ED24 figures, measured on its 16-scene [::6][:16] subsample at 8 blocks per
#: condition. Running this phase with --anchor reproduces that subsample and must land here,
#: which is what certifies the full-coverage rows are the same measurement scaled up.
ANCHORS = {"gate_theta0_on_ed24": 0.9533933675787315,
           "beta0_ring_only": 0.9535902434117546,
           "beta1_contrast": 0.9074814974059171}

ARMS = ("adopted", "magnitude", "beta0_support", "beta1_support",
        "beta0_magnitude", "beta1_magnitude")


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   gate_config: StateConfig, score_config: StateConfig,
                   max_blocks: Optional[int]) -> Optional[Dict]:
    """Per-block AUCs for both operands and both gate branches, plus the pooled scores.

    The gate key is corr(G, R_same) read from the radius-1 extraction and the operands come
    from the radius-2 one, exactly as in Phase 14. Per-block scores are retained so the
    condition's event-level AUC can be computed both raw and rank-normalised, which is the
    contrast Section 4.4 turns on: a per-block beta puts the two branches on different
    offsets, so pooling raw scores across blocks measures that offset and not the ranking.
    """

    scale_count = len(gate_config.taus_us)
    per_block: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    corrs: List[float] = []
    #: Index of each SCORED block within the condition's block sequence. Single-class blocks are
    #: skipped, so this is not range(len(per_block)) and it is what makes a paired comparison
    #: against a baseline with different file coverage possible (Phase 33).
    indices: List[int] = []
    oracle_agree: List[bool] = []
    support_means: List[float] = []
    pooled: Dict[str, List[np.ndarray]] = {"adopted": [], "magnitude": [],
                                           "adopted_rank": [], "magnitude_rank": []}
    pooled_labels: List[np.ndarray] = []
    produced = 0

    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue

        raw = bidirectional_features(events, gate_config)
        g, r_gate, _ = per_block_operands_both(raw, scale_count, gate_config.feature_dim)
        corr = safe_corr(g, r_gate)
        beta = 1.0 if corr < 0.0 else 0.0

        raw_s = bidirectional_features(events, score_config)
        _, r_same, _ = per_block_operands_both(raw_s, scale_count, score_config.feature_dim)
        support = ring_support(raw_s, score_config)
        log_support = np.log1p(support)
        support_means.append(float(support.mean()))

        branches = {
            "beta0_support": -log_support, "beta1_support": g - log_support,
            "beta0_magnitude": -r_same, "beta1_magnitude": g - r_same,
        }
        aucs = {}
        for name, score in branches.items():
            aucs[name] = roc_auc_score(labels, score) if score.std() > 0 else 0.5
            per_block[name].append(aucs[name])

        adopted = beta * g - log_support
        magnitude = beta * g - r_same
        per_block["adopted"].append(aucs["beta1_support" if beta else "beta0_support"])
        per_block["magnitude"].append(aucs["beta1_magnitude" if beta else "beta0_magnitude"])

        # Would a label-aware oracle have picked the same branch? Section 4.7's gate-quality
        # measure, which the block-size sweep also reports.
        oracle_beta = 1.0 if aucs["beta1_support"] >= aucs["beta0_support"] else 0.0
        oracle_agree.append(bool(oracle_beta == beta))
        corrs.append(corr)
        indices.append(produced - 1)

        pooled["adopted"].append(adopted)
        pooled["magnitude"].append(magnitude)
        pooled["adopted_rank"].append(rankdata(adopted) / adopted.size)
        pooled["magnitude_rank"].append(rankdata(magnitude) / magnitude.size)
        pooled_labels.append(labels)

    if not per_block["adopted"]:
        return None

    labels_all = np.concatenate(pooled_labels)
    event_level = {}
    if labels_all.min() != labels_all.max():
        for key, chunks in pooled.items():
            event_level[key] = float(roc_auc_score(labels_all, np.concatenate(chunks)))

    return {
        "blocks": len(per_block["adopted"]),
        "block_indices": indices,
        "per_block": {arm: [float(v) for v in per_block[arm]] for arm in ARMS},
        "block_mean": {arm: float(np.mean(per_block[arm])) for arm in ARMS},
        "event_level": event_level,
        "gate": {
            "open_rate": float(np.mean(np.asarray(corrs) < 0.0)),
            "oracle_agreement": float(np.mean(oracle_agree)),
            "mean_corr": float(np.mean(corrs)),
        },
        "mean_support": float(np.mean(support_means)),
    }


def bootstrap_ci(values: Sequence[float], replicates: int = 10_000,
                 seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(values, dtype=float),
                       size=(replicates, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(conditions: Dict[str, Dict]) -> Dict:
    """Pool over conditions, and price the operand and the gate the way the paper does."""

    def column(arm: str, key: str = "block_mean") -> List[float]:
        return [c[key][arm] for c in conditions.values() if arm in c[key]]

    overall = {arm: float(np.mean(column(arm))) for arm in ARMS}
    gains = np.asarray(column("adopted")) - np.asarray(column("magnitude"))
    gate_v0 = np.asarray(column("adopted")) - np.asarray(column("beta0_support"))
    gate_v1 = np.asarray(column("adopted")) - np.asarray(column("beta1_support"))

    def priced(deltas: np.ndarray, seed: int) -> Dict:
        lo, hi = bootstrap_ci(deltas, seed=seed)
        return {"mean": float(deltas.mean()), "ci95": [lo, hi],
                "conditions_won": int((deltas > 0).sum()),
                "conditions_total": int(deltas.size),
                "smallest": float(deltas.min()), "largest": float(deltas.max())}

    event = {arm: float(np.mean(column(arm, "event_level")))
             for arm in ("adopted", "magnitude", "adopted_rank", "magnitude_rank")
             if column(arm, "event_level")}
    event_gain = (np.asarray(column("adopted", "event_level"))
                  - np.asarray(column("magnitude", "event_level")))
    rank_gain = (np.asarray(column("adopted_rank", "event_level"))
                 - np.asarray(column("magnitude_rank", "event_level")))

    return {
        "conditions": len(conditions),
        "blocks": int(sum(c["blocks"] for c in conditions.values())),
        "block_mean": overall,
        "event_level": event,
        "operand_gain_block_mean": priced(gains, seed=1),
        "operand_gain_event_level": priced(event_gain, seed=2),
        "operand_gain_rank_normalised": priced(rank_gain, seed=3),
        "gate_value_vs_beta0": priced(gate_v0, seed=4),
        "gate_value_vs_beta1": priced(gate_v1, seed=5),
        "gate": {
            "open_rate": float(np.mean([c["gate"]["open_rate"] for c in conditions.values()])),
            "oracle_agreement": float(
                np.mean([c["gate"]["oracle_agreement"] for c in conditions.values()])),
        },
        "mean_support": float(np.mean([c["mean_support"] for c in conditions.values()])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=None,
                        help="limit to the first N scenes (smoke tests)")
    parser.add_argument("--max-blocks", type=int, default=None,
                        help="per scene x level; default is no cap, unlike Phase 4/22b's 8")
    parser.add_argument("--levels", nargs="*", default=ED24_LEVELS,
                        help=f"noise levels; default is Phase 4's {ED24_LEVELS}")
    parser.add_argument("--anchor", action="store_true",
                        help="reproduce Phase 22b's 16-scene/8-block subsample and check it")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase32_ed24_full.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    gate_config = StateConfig(width=SENSOR[0], height=SENSOR[1], taus_us=(250.0,), radius=1)
    score_config = StateConfig(width=SENSOR[0], height=SENSOR[1], taus_us=(250.0,), radius=2)

    all_scenes = sorted(p.name for p in ED24_ROOT.iterdir() if p.is_dir())
    if args.anchor:
        scenes, max_blocks = all_scenes[::6][:16], 8
        print("anchor mode: Phase 22b's subsample, 16 scenes x 8 blocks", flush=True)
    else:
        scenes, max_blocks = all_scenes, args.max_blocks
        if args.scenes is not None:
            scenes = scenes[:args.scenes]

    conditions: Dict[str, Dict] = {}
    for index, scene in enumerate(scenes, start=1):
        for level in args.levels:
            result = scan_condition(ed24_blocks(scene, level, BLOCK_SIZE, max_blocks),
                                    gate_config, score_config, max_blocks)
            if result is not None:
                conditions[f"{scene}_{level}"] = result
        done = summarise(conditions)
        print(f"[{index:3d}/{len(scenes)}] {scene:16s} "
              f"adopted={done['block_mean']['adopted']:.4f} "
              f"magnitude={done['block_mean']['magnitude']:.4f} "
              f"won={done['operand_gain_block_mean']['conditions_won']}"
              f"/{done['operand_gain_block_mean']['conditions_total']} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    summary = summarise(conditions)
    per_level: Dict[str, Dict[str, float]] = {}
    for level in args.levels:
        subset = {k: v for k, v in conditions.items() if k.endswith(f"_{level}")}
        if subset:
            per_level[level] = summarise(subset)["block_mean"]

    payload = {
        "phase": "32 ED24 as a third labelled benchmark, full scene coverage",
        "score": "s = beta*G - log(1 + S_r2); comparator s = beta*G - R_r2",
        "gate_key": "corr(G, R_same) at radius 1, theta = 0 -- frozen, not refitted here",
        "prior_contact": ("ED24 was used by Phase 4 and Phase 22b to check theta = 0 against "
                          "an ED24-fitted threshold, on one side only; no constant was changed "
                          "as a result. ED24 is therefore a third noise model and a third "
                          "cluster, NOT a second held-out set. E-MLB remains the only benchmark "
                          "nothing in this paper has been selected or checked on."),
        "scenes": scenes,
        "levels": list(args.levels),
        "block_size": BLOCK_SIZE,
        "max_blocks_per_condition": max_blocks,
        "conditions": conditions,
        "per_level": per_level,
        "summary": summary,
    }

    if args.anchor:
        # Phase 22b's `endpoints` are the two branches of the SUPPORT operand -- its scan
        # subtracts log1p(ring_support) in both -- despite the "ring_only" name, which is
        # inherited from Phase 4 where the subtrahend really was the magnitude.
        measured = {"gate_theta0_on_ed24": summary["block_mean"]["adopted"],
                    "beta0_ring_only": summary["block_mean"]["beta0_support"],
                    "beta1_contrast": summary["block_mean"]["beta1_support"]}
        payload["anchor_check"] = {
            **{f"{k}_measured": v for k, v in measured.items()},
            **{f"{k}_expected": v for k, v in ANCHORS.items()},
            **{f"{k}_abs_error": abs(measured[k] - ANCHORS[k]) for k in ANCHORS},
        }
        print("\nanchor check:", json.dumps(payload["anchor_check"], indent=2), flush=True)

    payload["wall_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nsummary: {json.dumps(summary, indent=2)}")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
