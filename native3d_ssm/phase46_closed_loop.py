"""Phase 46: does excluding rejected events from the state change the filter?

Section 3.1 of the manuscript states that scoring is open-loop -- every event in a
block updates the state at unit weight, and retention is applied to the finished
score vector with no path back into the state. The consequence is that the
statistics the filter scores against are computed on the unfiltered stream: the
background activity being rejected also contributes to G and to ring occupancy.

That was written as a disclosure. This phase turns it into a measurement, because
"we never tried the obvious alternative" is not a defence a reviewer has to accept.

The alternative, one refinement iteration:

    pass A   unit weights                  -> scores -> retention mask (no labels)
    pass B   weight 0 for rejected events  -> scores -> AUC

`NativeStateBank` already supports fractional update weights, and a zero weight
leaves both the accumulator and its reference timestamp untouched, so a rejected
event becomes invisible to every later read in both directions. The mask is
derived from the filter's own pass-A scores; no label is consulted at any point.

THE CONFOUND THIS PHASE EXISTS TO BREAK. Withholding the rejected events changes
two things at once: *which* events build the state (noise excluded) and *how many*
(at r = 0.28, 72% fewer). Ring occupancy must fall under the second alone, so a
closed-loop arm on its own cannot distinguish "excluding noise hurts" from "a state
built from a quarter of the events is sparser". Every selective arm therefore has a
density-matched random control that withholds the same COUNT of events per block,
chosen uniformly at random. The selectivity effect is the gap between the two, not
the drop from `open`.

Arms, all on identical blocks so every comparison is exactly paired:

    open           the adopted filter, s = beta*G - log(1 + S_r2)
    closed@28      one refinement iteration, mask at the frozen r = 0.28
    random@28      density control: same count withheld, chosen at random
    closed@50      the same at r = 0.50
    random@50      its density control
    oracle         withhold the TRUE noise, using labels
    random@oracle  density control at the oracle's per-block retention

`oracle` uses labels and is a diagnostic, not a deployable configuration. It exists
to test the premise directly: if excluding every genuine noise event does not beat
its own density control, then no score-based refinement rule can be blamed for the
closed-loop failure, and open-loop scoring is vindicated rather than merely excused.

Support is also broken out by true class, because "the refinement strips the ring
for signal and noise alike" is a claim about class-conditional occupancy and has to
be measured rather than inferred from the aggregate.

The open arm must reproduce Phase 14's `log_support|float` figures, which is what
certifies this harness measures the same thing the headline does. Read-only;
nothing is trained.

Unit of analysis: the 20 labelled development conditions, paired within condition.
Those 20 are nested in 2 datasets, so the condition-level interval understates the
correlation between conditions drawn from the same recordings; the two dataset means
are reported alongside it rather than replaced by it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon
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
WORKSPACE = PROJECT_ROOT.parent
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK_SIZE = 10_240
FROZEN_RETENTION = 0.28
#: (arm, density-matched random control). The control withholds the same count.
SELECTIVE = (("closed@28", "random@28"), ("closed@50", "random@50"),
             ("oracle", "random@oracle"))
ARMS = ("open", "closed@28", "random@28", "closed@50", "random@50",
        "oracle", "random@oracle")

#: Phase 14 `log_support|float`, the arm this phase's `open` column must reproduce.
ANCHORS = {"DND21": 0.9916848287458562, "DVSCLEAN": 0.9705824757446344}


def keep_mask(scores: np.ndarray, retention: float) -> np.ndarray:
    """Lowest-scoring `retention` fraction survives. Label 1 = noise scores high."""

    n_keep = max(1, int(round(len(scores) * retention)))
    keep = np.zeros(len(scores), dtype=bool)
    keep[np.argsort(scores, kind="stable")[:n_keep]] = True
    return keep


def random_mask(n_events: int, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    """Density control: withhold the same COUNT, chosen uniformly at random."""

    keep = np.zeros(n_events, dtype=bool)
    keep[rng.choice(n_events, size=n_keep, replace=False)] = True
    return keep


def score_block(events: np.ndarray, gate_config: StateConfig,
                score_config: StateConfig,
                weights: Optional[np.ndarray]) -> Tuple[np.ndarray, float, float,
                                                        np.ndarray]:
    """One full scoring pass. Returns (scores, corr, beta, support).

    Identical to the adopted path in `phase32_ed24_full.scan_condition` except that
    the update weights are threaded through, which is the only thing this phase varies.
    """

    scale_count = len(gate_config.taus_us)
    raw_g = bidirectional_features(events, gate_config, weights)
    g, r_gate, _ = per_block_operands_both(raw_g, scale_count, gate_config.feature_dim)
    corr = safe_corr(g, r_gate)
    beta = 1.0 if corr < 0.0 else 0.0

    raw_s = bidirectional_features(events, score_config, weights)
    support = ring_support(raw_s, score_config)
    return beta * g - np.log1p(support), corr, beta, support


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   gate_config: StateConfig, score_config: StateConfig,
                   max_blocks: Optional[int], seed: int) -> Optional[Dict]:
    """Per-block AUC for every arm, with each selective arm density-matched."""

    rng = np.random.default_rng(seed)
    per_block: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    #: Mean support overall and split by TRUE class. The class split is what decides
    #: whether the refinement strips the ring "for signal and noise alike" or
    #: preferentially removes the neighbours of one class.
    support_mean: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    support_zero: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    support_signal: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    support_noise: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    gate_flips = {arm: 0 for arm, _ in SELECTIVE}
    signal_fraction: List[float] = []
    scored = 0

    def record(arm: str, scores: np.ndarray, support: np.ndarray,
               labels: np.ndarray) -> None:
        per_block[arm].append(roc_auc_score(labels, scores))
        support_mean[arm].append(float(support.mean()))
        support_zero[arm].append(float((support == 0).mean()))
        support_signal[arm].append(float(support[labels == 0].mean()))
        support_noise[arm].append(float(support[labels == 1].mean()))

    for events, labels in block_iter:
        if max_blocks is not None and scored >= max_blocks:
            break
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue

        s_open, _, beta_open, supp_open = score_block(events, gate_config,
                                                      score_config, None)
        record("open", s_open, supp_open, labels)
        signal_fraction.append(float((labels == 0).mean()))

        #: (selective arm, its keep-mask). The oracle keeps exactly the true signal,
        #: so its density is the block's signal fraction rather than a chosen r.
        selections = [
            ("closed@28", keep_mask(s_open, FROZEN_RETENTION)),
            ("closed@50", keep_mask(s_open, 0.50)),
            ("oracle", labels == 0),
        ]
        for arm, keep in selections:
            control = dict(SELECTIVE)[arm]
            for name, mask in ((arm, keep),
                               (control, random_mask(len(events),
                                                     max(1, int(keep.sum())), rng))):
                scores, _, beta, support = score_block(
                    events, gate_config, score_config, mask.astype(np.float32))
                record(name, scores, support, labels)
                if name == arm and beta != beta_open:
                    gate_flips[arm] += 1
        scored += 1

    if not scored:
        return None
    out: Dict = {"blocks": scored, "signal_fraction": float(np.mean(signal_fraction))}
    for arm in ARMS:
        out[arm] = float(np.mean(per_block[arm]))
        out[f"support_mean|{arm}"] = float(np.mean(support_mean[arm]))
        out[f"support_zero_fraction|{arm}"] = float(np.mean(support_zero[arm]))
        out[f"support_signal|{arm}"] = float(np.mean(support_signal[arm]))
        out[f"support_noise|{arm}"] = float(np.mean(support_noise[arm]))
    for arm, count in gate_flips.items():
        out[f"gate_flip_fraction|{arm}"] = count / scored
    return out


def paired_stats(deltas: np.ndarray, rng: np.random.Generator,
                 replicates: int) -> Dict:
    """Paired summary over conditions. Wilcoxon because n = 20 and normality is untested."""

    boot = np.array([rng.choice(deltas, len(deltas), replace=True).mean()
                     for _ in range(replicates)])
    stat = {
        "n_conditions": int(len(deltas)),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "conditions_improved": int((deltas > 0).sum()),
        "conditions_unchanged": int((deltas == 0).sum()),
        "ci95_mean_delta": [float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5))],
    }
    if np.any(deltas != 0):
        result = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
        stat["wilcoxon_p"] = float(result.pvalue)
    else:
        stat["wilcoxon_p"] = None
    return stat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=WORKSPACE / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=WORKSPACE / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase46_closed_loop.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    def configs(w: int, h: int) -> Tuple[StateConfig, StateConfig]:
        return (StateConfig(width=w, height=h, taus_us=(250.0,), radius=1),
                StateConfig(width=w, height=h, taus_us=(250.0,), radius=2))

    dvs_gate, dvs_score = configs(1280, 720)
    dnd_gate, dnd_score = configs(346, 260)

    conditions: Dict[str, Dict[str, Dict]] = {"DND21": {}, "DVSCLEAN": {}}

    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            result = scan_condition(
                dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE,
                                args.max_blocks),
                dvs_gate, dvs_score, args.max_blocks, args.seed)
            if result is None:
                continue
            conditions["DVSCLEAN"][name] = result
            print(f"DVSCLEAN {name}: open={result['open']:.4f} "
                  f"closed@28={result['closed@28']:.4f} "
                  f"random@28={result['random@28']:.4f} "
                  f"oracle={result['oracle']:.4f} "
                  f"random@oracle={result['random@oracle']:.4f}", flush=True)

    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            result = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE,
                             args.max_blocks),
                dnd_gate, dnd_score, args.max_blocks, args.seed)
            if result is None:
                continue
            conditions["DND21"][name] = result
            print(f"DND21 {name}: open={result['open']:.4f} "
                  f"closed@28={result['closed@28']:.4f} "
                  f"random@28={result['random@28']:.4f} "
                  f"oracle={result['oracle']:.4f} "
                  f"random@oracle={result['random@oracle']:.4f}", flush=True)

    _columns = ["signal_fraction", *ARMS,
                *(f"support_mean|{a}" for a in ARMS),
                *(f"support_zero_fraction|{a}" for a in ARMS),
                *(f"support_signal|{a}" for a in ARMS),
                *(f"support_noise|{a}" for a in ARMS),
                *(f"gate_flip_fraction|{a}" for a, _ in SELECTIVE)]
    dataset_means = {d: {col: float(np.mean([c[col] for c in group.values()]))
                         for col in _columns}
                     for d, group in conditions.items() if group}
    pooled = {col: float(np.mean([dataset_means[d][col] for d in dataset_means]))
              for col in _columns}

    flat = [c for group in conditions.values() for c in group.values()]
    rng = np.random.default_rng(args.seed)

    def contrast(left: str, right: str) -> Dict:
        deltas = np.array([c[left] - c[right] for c in flat])
        stat = paired_stats(deltas, rng, args.bootstrap_replicates)
        stat["per_dataset_mean_delta"] = {
            d: dataset_means[d][left] - dataset_means[d][right] for d in dataset_means}
        return stat

    comparisons: Dict[str, Dict] = {}
    #: Cost against the adopted filter. Descriptive only: it mixes the effect of
    #: withholding noise with the effect of withholding most of the block.
    for arm, _control in SELECTIVE:
        comparisons[f"{arm}_minus_open"] = contrast(arm, "open")
    #: The contrast that isolates selectivity. Each selective arm against a random
    #: mask that withholds the same COUNT, so density is held fixed and only the
    #: choice of which events to withhold differs.
    for arm, control in SELECTIVE:
        comparisons[f"{arm}_minus_{control}"] = contrast(arm, control)
    #: Pure density: what withholding that many events costs when the choice carries
    #: no information at all.
    for _arm, control in SELECTIVE:
        comparisons[f"{control}_minus_open"] = contrast(control, "open")

    #: Between-condition association between the AUC change and each candidate
    #: mechanism. These exist to be checked, not to be quoted as an explanation: the
    #: 20 conditions sit in 2 datasets and gate-flip rate separates almost perfectly
    #: by dataset, so a strong coefficient here is an n = 2 contrast wearing n = 20
    #: clothes. The saturation coefficient is reported because its SIGN is the point.
    mechanism = {}
    for arm, _control in SELECTIVE:
        delta_auc = np.array([c[arm] - c["open"] for c in flat])
        flips = np.array([c[f"gate_flip_fraction|{arm}"] for c in flat])
        delta_zero = np.array([c[f"support_zero_fraction|{arm}"]
                               - c["support_zero_fraction|open"] for c in flat])
        mechanism[arm] = {
            "corr_delta_auc_vs_gate_flip":
                float(np.corrcoef(delta_auc, flips)[0, 1]) if flips.std() > 0 else None,
            "corr_delta_auc_vs_zero_support_increase":
                float(np.corrcoef(delta_auc, delta_zero)[0, 1]),
            "per_dataset_delta_auc": {
                d: float(np.mean([c[arm] - c["open"] for c in group.values()]))
                for d, group in conditions.items() if group},
            "per_dataset_zero_support_increase": {
                d: float(np.mean([c[f"support_zero_fraction|{arm}"]
                                  - c["support_zero_fraction|open"]
                                  for c in group.values()]))
                for d, group in conditions.items() if group},
        }

    anchor_drift = {d: abs(dataset_means[d]["open"] - ANCHORS[d])
                    for d in dataset_means if d in ANCHORS}

    payload = {
        "phase": "46 closed-loop state update",
        "question": ("Section 3.1 says the filter scores open-loop, so the noise it "
                     "rejects still builds the state it scores against. Does one "
                     "refinement iteration that excludes rejected events change the "
                     "measured separation?"),
        "design": ("Pass A scores at unit weight. Its scores select a keep-mask at the "
                   "stated retention, using no labels. Pass B re-extracts with weight 0 "
                   "on rejected events and re-scores. AUC is over all events in both "
                   "arms, so the comparison is exactly paired within block."),
        "unit_of_analysis": ("Condition. The 20 conditions are nested in 2 datasets, so "
                             "the condition-level interval understates the correlation "
                             "between conditions drawn from the same recording set."),
        "frozen_retention": FROZEN_RETENTION,
        "anchor_reference": ANCHORS,
        "anchor_drift": anchor_drift,
        "conditions": conditions,
        "dataset_means": dataset_means,
        "pooled_20": pooled,
        "comparisons": comparisons,
        "mechanism_association": mechanism,
        "mechanism_caveat": ("Neither candidate supports an attribution. The saturation "
                             "coefficient has the wrong sign between conditions, and "
                             "gate-flip rate separates almost perfectly by dataset, so "
                             "with 2 datasets its coefficient is an n = 2 contrast."),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\npooled_20: {json.dumps(pooled, indent=2)}")
    print(f"anchor drift vs Phase 14: {anchor_drift}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
