"""Phase 14: a quantizable sparse-regime operand.

Phase 13 established that the fixed-point failure is **not** the beta gate (provably safe:
0 flips, 100% of blocks certified at Q12.8) but the ring-magnitude operand. Nonzero
`R_same` spans nine orders of magnitude (min 2.1e-9) because it is a `log1p` of an
exponentially decayed sum, and the sparse regime's discriminative information sits in that
tail. Uniform fixed point recovers it only slowly: pooled over the 20 conditions,
`phase13_fixed_point_adaptive.json` measures a deficit against float of 0.100 AUC at
Q12.8, 0.087 at Q14.10, 0.077 at Q16.12, 0.068 at Q18.14 and still 0.060 at Q20.16
(0.9114 against 0.9716), which is the widest format measured.

An earlier version of this docstring claimed "even 24 fractional bits trails float 0.9718
vs 0.9907". That was wrong twice over and the manuscript inherited it: no 24-fractional-bit
format was ever run (Q20.16 is the widest), and the pair 0.9718/0.9907 compared the
magnitude at Q20.16 on ONE condition against the *support* operand at float and radius 1.
The pooled figures above are the sourced replacement.

**Correction to an earlier proposal.** The obvious-looking fix -- reformulate around
`log_age_same` -- does not work: that operand is `log1p(same_age / tau)` for the *centre*
pixel's own age, not the ring's, and Phase 7 measured it uninformative on DVSCLEAN
(AUC 0.4996). There is no ring-age operand in the feature set.

**The candidate that does work on paper.** `dir_support_same` is the *count* of
same-polarity ring neighbours carrying any history. Three properties make it the right
target:

  1. It is the strongest DVSCLEAN operand at radius 2 (Phase 7: 0.9706 as a score).
  2. It is the only operand in the scan whose informative sign does **not** flip between
     datasets -- the property the beta gate exists to work around.
  3. It is an **integer in [0, nc]** (nc = 8 at r=1, 24 at r=2; bidirectional sums double
     that), so it is *exactly* representable in ceil(log2(2*nc+1)) bits -- 5 at r=1, 6 at
     r=2 -- with **zero quantization error** by construction.

In hardware the count needs no magnitude at all: it is a popcount over "has this neighbour
fired", i.e. a validity bit per tap, which is dramatically cheaper than a decayed-magnitude
datapath.

This phase measures, on the same blocks, whether trading a little float AUC for exact
representability is a good deal:

    current    s = beta*G - R_same                (Phase 11 best, 0.9716 float)
    support    s = beta*G - support               raw integer count
    log_supp   s = beta*G - log1p(support)        same log domain as G, bounded

each under float and every fixed-point format. G is still quantized normally; only the
subtrahend changes. Read-only; nothing is trained or fitted.
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
from .evaluate_adaptive_beta import (
    dnd21_blocks,
    dvsclean_blocks,
    per_block_operands_both,
)
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 10_240
FORMATS = {"q16.12": (16, 12), "q12.8": (12, 8), "q8.4": (8, 4), "q6.3": (6, 3)}
VARIANTS = ("current", "support", "log_support")


def quantize(values: np.ndarray, total_bits: int, fractional_bits: int) -> np.ndarray:
    scale = float(2 ** fractional_bits)
    maximum = float(2 ** total_bits - 1) / scale
    return np.round(np.clip(values, 0.0, maximum) * scale) / scale


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() <= 0 or b.std() <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ring_support(raw: np.ndarray, config: StateConfig) -> np.ndarray:
    """Count of same-polarity ring taps with history (forward + reverse), as in Phase 7."""

    scale_count = len(config.taus_us)
    width = config.feature_dim
    nc = (2 * config.radius + 1) ** 2 - 1
    base = 7 * scale_count
    cols = [base + k * scale_count for k in range(nc)]
    stack = np.stack([raw[:, c] + raw[:, width + c] for c in cols], axis=1)
    return (stack > 0).sum(axis=1).astype(np.float64)


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, score_config: Optional[StateConfig],
                   max_blocks: Optional[int] = None) -> Dict:
    cfg_s = score_config if score_config is not None else config
    scale_count = len(config.taus_us)
    auc: Dict[str, List[float]] = {}
    support_max = 0.0
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
        corr = safe_corr(g, r_gate)
        beta = 1.0 if corr < 0.0 else 0.0

        raw_s = bidirectional_features(events, cfg_s) if score_config is not None else raw
        _, r_same, _ = per_block_operands_both(raw_s, scale_count, cfg_s.feature_dim)
        support = ring_support(raw_s, cfg_s)
        support_max = max(support_max, float(support.max()))

        for fmt in ("float", *FORMATS):
            if fmt == "float":
                gq, rq, sq = g, r_same, support
            else:
                bits = FORMATS[fmt]
                gq, rq = quantize(g, *bits), quantize(r_same, *bits)
                # The count is an integer: quantizing it is exact whenever the format has
                # enough integer bits, which is the whole point of this variant.
                sq = quantize(support, *bits)
            subtrahends = {"current": rq, "support": sq, "log_support": np.log1p(sq)}
            for name, sub in subtrahends.items():
                score = beta * gq - sub
                if score.std() <= 0:
                    continue
                auc.setdefault(f"{name}|{fmt}", []).append(roc_auc_score(labels, score))

    if not auc:
        return {"blocks": 0}
    return {"blocks": produced,
            "support_max": support_max,
            "auc": {k: float(np.mean(v)) for k, v in auc.items()}}


def aggregate(conditions: Dict[str, Dict]) -> Dict[str, float]:
    usable = [c for c in conditions.values() if c.get("blocks")]
    if not usable:
        return {}
    names = set(usable[0]["auc"])
    for c in usable[1:]:
        names &= set(c["auc"])
    return {n: float(np.mean([c["auc"][n] for c in usable])) for n in sorted(names)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase14_quantizable_operand.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    def configs(w: int, h: int):
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
            a = dvsclean[name]["auc"]
            print(f"DVSCLEAN {name}: cur/float={a['current|float']:.4f} "
                  f"sup/float={a['support|float']:.4f} "
                  f"cur/q12.8={a['current|q12.8']:.4f} sup/q12.8={a['support|q12.8']:.4f}",
                  flush=True)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd_gate, dnd_score, args.max_blocks)
            a = dnd21[name]["auc"]
            print(f"DND21 {name}: cur/float={a['current|float']:.4f} "
                  f"sup/float={a['support|float']:.4f} "
                  f"cur/q12.8={a['current|q12.8']:.4f} sup/q12.8={a['support|q12.8']:.4f}",
                  flush=True)

    dv, dn = aggregate(dvsclean), aggregate(dnd21)
    overall = {k: 0.5 * (dv[k] + dn[k]) for k in dv}

    table = {}
    for variant in VARIANTS:
        table[variant] = {}
        base = overall[f"{variant}|float"]
        for fmt in ("float", *FORMATS):
            key = f"{variant}|{fmt}"
            table[variant][fmt] = {
                "DVSCLEAN": dv[key], "DND21": dn[key], "overall_20": overall[key],
                "delta_vs_own_float": overall[key] - base,
            }

    support_max = max(c["support_max"] for c in {**dvsclean, **dnd21}.values() if c.get("blocks"))
    payload = {
        "phase": "14 quantizable sparse-regime operand",
        "question": ("Phase 13 showed the ring-magnitude operand is not representable in "
                     "uniform fixed point. Is an integer ring-support count a better "
                     "hardware operand, and what does it cost in float AUC?"),
        "correction": ("log_age_same is the CENTRE pixel's age, not the ring's, and Phase 7 "
                       "measured it uninformative on DVSCLEAN (0.4996). It is not a "
                       "candidate; dir_support_same is."),
        "config": {"gate_radius": 1, "score_radius": args.score_radius,
                   "formats": {k: f"unsigned {v[0]}-bit, {v[1]} fractional" for k, v in FORMATS.items()}},
        "support_max_observed": support_max,
        "support_bits_required": int(np.ceil(np.log2(support_max + 1))),
        "summary": {"DVSCLEAN": dv, "DND21": dn, "overall_20": overall},
        "table": table,
        "conditions": {"DVSCLEAN": dvsclean, "DND21": dnd21},
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
