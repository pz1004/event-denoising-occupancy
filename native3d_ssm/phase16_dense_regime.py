"""Phase 16: why do the dense-Night E-MLB cells lose, and is it fixable on dev?

**The gap.** After Phase 15 the Track-B deficit to reproduced EDformer is entirely two cells:
N-ND16 (-0.0940) and N-ND64 (-0.0828). Every other cell wins.

**Why this cannot be worked on directly.** Those are E-MLB cells, and E-MLB is the Track-B
*test* set. Tuning anything against them yields a metric-oracle, not a result (constraint in
status.md Part I section 3), and would void the blind validation of Phases 9/10/14b/15.

**The dev analogue.** The two losing cells are the two with the smallest mean corr(G,R)
(+0.0613, +0.0542) and the ranges are disjoint from the winning cells (>= +0.0706). ED24 --
labeled, same 346x260 sensor, and never used to fit anything -- reproduces that split:

    ED24 @ 0.1-2.0 Hz/px : corr ~ +0.098   (like the cells we win)
    ED24 @ 4.0 Hz/px     : corr ~ +0.048   (like the cells we lose)

So the dense low-corr regime *can* be studied with labels, on data that is not the test set.
Caveat worth stating plainly: with only 8 E-MLB cells, mean corr (+0.707), raw MESR (+0.708)
and adaptive retention (+0.739) all correlate about equally with the deficit and are mutually
confounded, so "low corr" is a hypothesis, not an established cause.

**Protecting the holdout.** ED24 has 100 scenes. They are split by index parity:
`scenes[0::2]` for development here, `scenes[1::2]` reserved as an untouched blind half. This
phase reads ONLY the development half.

**What is measured, per noise level**, so the three candidate explanations separate:

  hard_gate     the adopted score, beta = 1 iff corr(G,R) < 0
  oracle_beta   best beta on a grid, per condition -- the ceiling any gate could reach
  soft_gate     parameter-free: beta = P(true corr < 0) using Pearson's analytic standard
                error se = (1 - r^2)/sqrt(n-1). No fitted constant.
  corr spread   std of per-block corr; Phase 5b finding 4 says spread, not mean, predicts
                where the theta = 0 decision degrades.

Reading:
  oracle >> hard_gate at high noise only  -> the GATE is the problem (soft gate may help)
  oracle ~= hard_gate, both fall          -> the OPERAND is the problem
  neither falls                           -> ranking is fine; the E-MLB deficit is
                                             retention/MESR-specific, not ranking quality

Read-only. Nothing is trained; no threshold is fitted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .data import iter_labeled_text
from .evaluate_adaptive_beta import per_block_operands_both, ring_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ED24_ROOT = PROJECT_ROOT / "ECCV2024_datasets/ED24"
BLOCK_SIZE = 10_240
BETA_GRID = np.round(np.arange(0.0, 1.201, 0.1), 3)
LEVELS = ("0.5", "1.0", "2.0", "3.0", "4.0")


def ed24_blocks(scene: str, level: str, max_blocks: Optional[int]
                ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    path = ED24_ROOT / scene / f"{scene}_{level}.csv"
    produced = 0
    for values in iter_labeled_text(path, BLOCK_SIZE):
        if len(values) != BLOCK_SIZE:
            break
        if max_blocks is not None and produced >= max_blocks:
            break
        yield values[:, :4], values[:, 4]
        produced += 1


def soft_beta(corr: float, n: int) -> float:
    """P(true correlation < 0) under Pearson's analytic standard error. Parameter-free."""

    if n < 3:
        return 1.0 if corr < 0.0 else 0.0
    se = (1.0 - corr ** 2) / np.sqrt(n - 1)
    if se <= 0:
        return 1.0 if corr < 0.0 else 0.0
    return float(norm.cdf(-corr / se))


def scan_level(scenes: List[str], level: str, gate_cfg: StateConfig,
               score_cfg: StateConfig, max_blocks: Optional[int]) -> Dict:
    corrs: List[float] = []
    hard: List[float] = []
    soft: List[float] = []
    grid: List[np.ndarray] = []
    betas_soft: List[float] = []
    scale_count = len(gate_cfg.taus_us)

    for scene in scenes:
        for events, labels in ed24_blocks(scene, level, max_blocks):
            labels = np.asarray(labels)
            if labels.min() == labels.max():
                continue
            raw = bidirectional_features(events, gate_cfg)
            g, r_gate, _ = per_block_operands_both(raw, scale_count, gate_cfg.feature_dim)
            raw_s = bidirectional_features(events, score_cfg)
            support = ring_support(raw_s, scale_count, score_cfg.feature_dim, score_cfg.radius)
            log_support = np.log1p(support)

            corr = (float(np.corrcoef(g, r_gate)[0, 1])
                    if g.std() > 0 and r_gate.std() > 0 else 0.0)
            corrs.append(corr)
            hard.append(roc_auc_score(labels, (1.0 if corr < 0.0 else 0.0) * g - log_support))
            b = soft_beta(corr, len(events))
            betas_soft.append(b)
            soft.append(roc_auc_score(labels, b * g - log_support))
            grid.append(np.array([roc_auc_score(labels, beta * g - log_support)
                                  for beta in BETA_GRID]))

    if not corrs:
        return {"blocks": 0}
    grid_mean = np.vstack(grid).mean(axis=0)
    best = int(np.argmax(grid_mean))
    return {
        "blocks": len(corrs),
        "mean_corr": float(np.mean(corrs)),
        "std_corr": float(np.std(corrs)),
        "frac_blocks_corr_negative": float(np.mean(np.asarray(corrs) < 0)),
        "auc_hard_gate": float(np.mean(hard)),
        "auc_soft_gate": float(np.mean(soft)),
        "mean_soft_beta": float(np.mean(betas_soft)),
        "auc_oracle_beta": float(grid_mean[best]),
        "oracle_beta": float(BETA_GRID[best]),
        "auc_beta0": float(grid_mean[0]),
        "gate_headroom": float(grid_mean[best] - np.mean(hard)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=20,
                        help="How many development-half scenes to use.")
    parser.add_argument("--max-blocks", type=int, default=4,
                        help="Blocks per scene x level.")
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--half", choices=("dev", "hold"), default="dev",
                        help="dev = scenes[0::2]; hold = scenes[1::2], the reserved blind half.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase16_dense_regime.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    all_scenes = sorted(p.name for p in ED24_ROOT.iterdir() if p.is_dir())
    half = all_scenes[0::2] if args.half == "dev" else all_scenes[1::2]
    scenes = half[:args.scenes]

    gate_cfg = StateConfig(width=346, height=260, taus_us=(250.0,), radius=1)
    score_cfg = StateConfig(width=346, height=260, taus_us=(250.0,), radius=args.score_radius)

    levels: Dict[str, Dict] = {}
    for level in LEVELS:
        levels[level] = scan_level(scenes, level, gate_cfg, score_cfg, args.max_blocks)
        c = levels[level]
        print(f"level {level:>4} Hz/px: blocks={c['blocks']:4d} "
              f"corr={c['mean_corr']:+.4f}+-{c['std_corr']:.4f} "
              f"neg={c['frac_blocks_corr_negative']:.3f} | "
              f"hard={c['auc_hard_gate']:.4f} soft={c['auc_soft_gate']:.4f} "
              f"oracle={c['auc_oracle_beta']:.4f} (b*={c['oracle_beta']}) "
              f"headroom={c['gate_headroom']:+.4f}", flush=True)

    payload = {
        "phase": "16 dense low-corr regime on ED24 (dev analogue of N-ND16/N-ND64)",
        "score": "s = beta*G - log1p(ring_support), score radius 2, gate key radius 1",
        "split": {"half_used": args.half, "scenes": scenes,
                  "note": "scenes[0::2] = dev, scenes[1::2] = reserved blind half"},
        "soft_gate": ("beta = Phi(-corr / se), se = (1-corr^2)/sqrt(n-1); parameter-free, "
                      "no fitted threshold"),
        "how_to_read": ("oracle >> hard at high noise only -> gate is the problem; "
                        "oracle ~= hard and both fall -> operand is the problem; "
                        "neither falls -> ranking is fine and the E-MLB deficit is "
                        "retention/MESR-specific"),
        "levels": levels,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
