"""Phase 30: what the rejected retention rules cost per block.

Appendix D rejects the two-component Gaussian mixture as a retention rule. On the
superseded operand that rejection was carried by accuracy; on the adopted support score
the mixture *ties* the adaptive rule (0.9556 against 0.9554 pooled, phase 6a), so the
rejection now rests entirely on cost and the cost has to be measured rather than asserted.

This phase times the three per-block retention estimators of `retention.py` on 10,240-event
score arrays -- the adopted block size -- under both score shapes the gate produces: the
continuous scores of the open branch, and the atom-at-the-maximum scores of the shut branch.
Read-only; nothing is fitted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

from .retention import below_top_retention, gmm2_retention, otsu_retention

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK = 10_240
REPEATS = 200
SEED = 20260813


def make_blocks(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Two score shapes: the open branch (continuous) and the shut branch (atom at max)."""

    signal = rng.normal(-2.5, 0.8, int(BLOCK * 0.45))
    noise = rng.normal(-0.6, 0.5, BLOCK - signal.size)
    continuous = np.concatenate([signal, noise])
    rng.shuffle(continuous)

    # Shut branch: s = -log(1 + S); every event with an empty ring lands on 0, the maximum.
    atom_share = 0.42
    n_atom = int(BLOCK * atom_share)
    atomic = np.concatenate([np.zeros(n_atom),
                             -np.log1p(rng.integers(1, 25, BLOCK - n_atom))])
    rng.shuffle(atomic)
    return {"open_branch_continuous": continuous, "shut_branch_atom": atomic}


def time_rule(fn, scores: np.ndarray, repeats: int) -> float:
    fn(scores)                                   # warm up import / JIT paths
    start = time.perf_counter()
    for _ in range(repeats):
        fn(scores)
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase30_retention_rule_cost.json")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    rules = {"otsu": otsu_retention, "gmm2": gmm2_retention, "below_top": below_top_retention}
    results: Dict[str, Dict[str, float]] = {}
    for name, scores in make_blocks(rng).items():
        per_rule = {r: time_rule(fn, scores, args.repeats) for r, fn in rules.items()}
        per_rule["gmm2_over_otsu"] = per_rule["gmm2"] / per_rule["otsu"]
        per_rule["gmm2_over_below_top"] = per_rule["gmm2"] / per_rule["below_top"]
        results[name] = per_rule
        print(f"{name}: otsu={per_rule['otsu']*1e6:.1f}us gmm2={per_rule['gmm2']*1e6:.1f}us "
              f"below_top={per_rule['below_top']*1e6:.1f}us ratio={per_rule['gmm2_over_otsu']:.0f}x")

    payload = {
        "phase": "30 per-block cost of the retention rules",
        "question": "how much slower is the two-component mixture than the adopted rule's two halves",
        "block_size": BLOCK,
        "repeats": args.repeats,
        "seed": SEED,
        "unit": "seconds per block, mean over repeats",
        "results": results,
        "caveat": ("Single-thread CPU on the authors' host; the ratio, not the absolute time, is "
                   "what Appendix D quotes. Score arrays are synthetic but match the two shapes "
                   "the gate produces, and the ratio is dominated by EM iterations against a "
                   "histogram pass rather than by the distribution."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
