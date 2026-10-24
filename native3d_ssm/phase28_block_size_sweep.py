"""Phase 28: is the block size load-bearing, and is 10,240 special?

Section 3.1 of the manuscript asserted that the gate statistic "discriminates regimes only
at this scale" and cited no sweep. The supporting evidence in Section 3.4 is a single
contrast -- pooled corr(G, R_same) = -0.0138 against within-block -0.1869 -- which shows
that a whole-recording window is worse than a block, but says nothing about the shape of
the curve in between, and nothing about whether 10,240 is a maximum or an arbitrary point
on a plateau. This phase measures the curve.

Everything except the block size is frozen and reused verbatim from Phase 19: the same 20
labelled conditions, the same block iterators, the same gate key corr(G, R_same) at
radius 1, the same adopted score s = beta*G - log(1 + S_r2), float64 throughout. Only
`block_size` moves. Nothing is trained, fitted or selected.

Per (dataset, block size) we report

    gated / beta0 / beta1   block-mean AUC of the adopted score under the three beta modes
    oracle                  block-mean AUC taking the better beta per block, using labels;
                            a ceiling, not a claimable configuration
    frac_blocks_beta1       how often the gate opens
    mean_corr               mean within-block corr(G, R_same), the gate statistic itself
    gate_oracle_agreement   fraction of blocks where the gate picks the better beta

**What is and is not comparable across N.** `frac_blocks_beta1`, `mean_corr` and
`gate_oracle_agreement` are per-block properties of the gate statistic and are directly
comparable across block sizes: they answer the scale-dependence question. The AUC columns
are NOT a clean accuracy series, for two reasons stated here rather than discovered later.
A block-mean AUC over N-event blocks is a different statistic for every N -- larger blocks
mean fewer, harder ranking problems -- and the within-block bidirectional pass is
block-local, so a larger N grants the extractor more lookahead. AUC differences across N
therefore confound three effects; AUC differences *between beta modes at the same N* do
not, and those are the paired quantities the gate discussion needs.

Audit: at N = 10,240 every column must reproduce Phase 19's `log_support|*` condition
means (DND21 gated 0.9917, DVSCLEAN gated 0.9706) and its gate opening rates (0.9456 and
0.0121). `audit` reports the deviations; they are the precondition for using any of this.

Read-only. Writes a NEW artifact and touches no existing one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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

#: Geometric sweep centred on the adopted 10,240, spanning a factor of 64.
BLOCK_SIZES: Tuple[int, ...] = (1280, 2560, 5120, 10240, 20480, 40960, 81920)
ADOPTED_BLOCK_SIZE = 10240

#: Phase 19 `log_support|*` condition means and gate opening rates at N = 10,240.
PHASE19: Dict[str, Dict[str, float]] = {
    "DND21": {"gated": 0.991685, "beta0": 0.954854, "beta1": 0.992743,
              "frac_blocks_beta1": 0.945584},
    "DVSCLEAN": {"gated": 0.970582, "beta0": 0.970642, "beta1": 0.954441,
                 "frac_blocks_beta1": 0.012116},
}

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260812


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   gate_config: StateConfig, score_config: StateConfig) -> Dict:
    """Per block: the gate statistic, and the adopted score's AUC under each beta mode.

    Column conventions and the operand follow `phase14_quantizable_operand`, so the score
    is bit-for-bit the one behind Table 1: G comes from the radius-1 extraction (it takes
    no spatial argument) and the subtrahend is log1p of the radius-2 support count.
    """

    scale_count = len(gate_config.taus_us)
    corr: List[float] = []
    auc0: List[float] = []
    auc1: List[float] = []
    beta_gated: List[float] = []
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
        score0, score1 = -subtrahend, g - subtrahend
        if score0.std() <= 0 or score1.std() <= 0:
            continue

        scored += 1
        corr.append(rho)
        auc0.append(roc_auc_score(labels, score0))
        auc1.append(roc_auc_score(labels, score1))
        beta_gated.append(1.0 if rho < 0.0 else 0.0)

    if not scored:
        return {"blocks": produced, "scored_blocks": 0}

    a0 = np.asarray(auc0)
    a1 = np.asarray(auc1)
    beta = np.asarray(beta_gated)
    oracle_beta = (a1 > a0).astype(np.float64)

    return {
        "blocks": produced,
        "scored_blocks": scored,
        "events": events_seen,
        "gated": float(np.mean(np.where(beta > 0.5, a1, a0))),
        "beta0": float(np.mean(a0)),
        "beta1": float(np.mean(a1)),
        "oracle": float(np.mean(np.maximum(a0, a1))),
        "frac_blocks_beta1": float(np.mean(beta)),
        "mean_corr": float(np.mean(corr)),
        "frac_blocks_corr_negative": float(np.mean(np.asarray(corr) < 0.0)),
        "gate_oracle_agreement": float(np.mean(beta == oracle_beta)),
    }


COLUMNS = ("gated", "beta0", "beta1", "oracle", "frac_blocks_beta1", "mean_corr",
           "frac_blocks_corr_negative", "gate_oracle_agreement")


def aggregate(conditions: Dict[str, Dict]) -> Dict[str, float]:
    """Unweighted mean over conditions, matching Phase 19's `aggregate`."""

    usable = [c for c in conditions.values() if c.get("scored_blocks")]
    if not usable:
        return {}
    out = {name: float(np.mean([c[name] for c in usable])) for name in COLUMNS}
    out["scored_blocks"] = int(sum(c["scored_blocks"] for c in usable))
    out["conditions"] = len(usable)
    return out


def paired_interval(conditions: Dict[str, Dict], left: str, right: str,
                    rng: np.random.Generator) -> Dict[str, float]:
    """Percentile bootstrap over conditions for the paired difference `left - right`.

    The unit of analysis is the condition, as everywhere else in this paper. Section 4.2
    of the manuscript notes that the 20 conditions cluster into 2 datasets, so this
    interval inherits the same effective-sample-size ceiling and is reported for magnitude
    rather than for inference.
    """

    diffs = np.asarray([c[left] - c[right]
                        for c in conditions.values() if c.get("scored_blocks")])
    if diffs.size == 0:
        return {}
    draws = rng.choice(diffs, size=(BOOTSTRAP_REPLICATES, diffs.size), replace=True)
    means = draws.mean(axis=1)
    return {
        "point": float(diffs.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n_conditions": int(diffs.size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=list(BLOCK_SIZES))
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=None,
                        help="debug only; None reproduces Phase 19's full pass")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase28_block_size_sweep.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def configs(width: int, height: int) -> Tuple[StateConfig, StateConfig]:
        gate = StateConfig(width=width, height=height, taus_us=(250.0,), radius=1)
        score = StateConfig(width=width, height=height, taus_us=(250.0,),
                            radius=args.score_radius)
        return gate, score

    dvs_gate, dvs_score = configs(1280, 720)
    dnd_gate, dnd_score = configs(346, 260)

    sweep: Dict[str, Dict] = {}
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

        sweep[str(block_size)] = {
            "per_dataset": per_dataset,
            "overall_20": overall,
            "gate_value_vs_beta0": paired_interval(merged, "gated", "beta0", rng),
            "gate_value_vs_beta1": paired_interval(merged, "gated", "beta1", rng),
            "oracle_headroom": paired_interval(merged, "oracle", "gated", rng),
            "conditions": merged,
            "wall_seconds": time.perf_counter() - step,
        }
        print(f"N={block_size:6d}  "
              f"gated={overall['gated']:.4f}  beta0={overall['beta0']:.4f}  "
              f"beta1={overall['beta1']:.4f}  oracle={overall['oracle']:.4f}  "
              f"open(DND21)={per_dataset['DND21']['frac_blocks_beta1']:.4f}  "
              f"open(DVSCLEAN)={per_dataset['DVSCLEAN']['frac_blocks_beta1']:.4f}  "
              f"agree={overall['gate_oracle_agreement']:.4f}  "
              f"[{time.perf_counter() - step:.0f}s]", flush=True)

    payload: Dict = {
        "phase": "28 block-size sweep for the gate statistic",
        "question": ("Section 3.1 claimed the gate statistic discriminates regimes only at "
                     "N = 10,240 and cited no sweep. Is 10,240 a maximum, a plateau point, "
                     "or arbitrary?"),
        "score": "s = beta*G - log(1 + S_r2)",
        "gate_key": "corr(G, R_same) at radius 1 -- unchanged from Phase 19",
        "frozen": ("everything except block_size: 20 labelled conditions, gate radius 1, "
                   "score radius 2, taus_us=[250.0], float64"),
        "comparability": ("frac_blocks_beta1, mean_corr and gate_oracle_agreement are "
                          "per-block properties of the gate statistic and are comparable "
                          "across N. The AUC columns are not a clean accuracy series: "
                          "block-mean AUC is a different statistic at every N, and the "
                          "block-local bidirectional pass grants more lookahead as N grows. "
                          "Only beta-mode differences at fixed N are unconfounded."),
        "unit_of_analysis": "condition (n = 20); the 20 cluster into 2 datasets",
        "block_sizes": list(args.block_sizes),
        "adopted_block_size": ADOPTED_BLOCK_SIZE,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "method": "percentile, resampling conditions"},
        "sweep": sweep,
    }

    key = str(ADOPTED_BLOCK_SIZE)
    if key in sweep:
        payload["audit"] = {
            "note": ("At the adopted block size this phase is Phase 19's log_support row, "
                     "so every deviation below must be ~0."),
            **{f"{ds}_{col}_deviation": sweep[key]["per_dataset"][ds][col] - expected
               for ds, cols in PHASE19.items() for col, expected in cols.items()},
        }
        print("\nAudit:", json.dumps(payload["audit"], indent=2), flush=True)
    else:
        payload["audit"] = {"note": f"N = {ADOPTED_BLOCK_SIZE} not in the sweep; not audited."}

    payload["wall_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
