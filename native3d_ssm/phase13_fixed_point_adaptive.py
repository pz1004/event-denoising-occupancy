"""Phase 13: fixed-point certificate for the ADAPTIVE path.

The existing certificate (`dnd21_fixed_point.json`) covers only the **static** filter:
`analytic.analytic_contrast_score` is `G - R_same` with beta hardcoded to 1, and it
quantizes four operands. Since then the method has gained a beta gate, an opposite-polarity
ring, a retention rule and (in the causal variant) a streamed EMA correlation. None of those
are certified, so every "FPGA-ready" claim is currently stale.

**The specific risk.** theta = 0 is a *sign test* on corr(G, R_same). Quantization perturbs
corr, so a block whose true |corr| is smaller than that perturbation can have its beta
flipped. Phase 5b already showed gate fidelity collapses exactly where per-block corr
straddles zero, so this is not hypothetical -- it is the one place fixed point can do real
damage.

**What this certifies.** Per quantization format Q(total, frac):

  1. **Gate margin.** Per block, |corr_float| against the observed perturbation
     |corr_float - corr_quantized|. A block is *certified* when its margin exceeds the
     worst perturbation seen, i.e. its beta decision provably cannot flip.
  2. **Beta flip rate.** Measured directly, not inferred.
  3. **End-to-end AUC delta** under the fully quantized path (operands, gate and score).
  4. **Retention delta** -- the retention rule reads score shape, so it is certified too.

Operands are nonnegative log1p values, so unsigned saturating formats match the hardware.
State, timestamps and decay stay floating point, exactly as the original certificate scoped
it; this phase extends coverage to the adaptive decisions built on top of the operands.

Read-only. Nothing is trained or fitted.
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
    ring_support,
)
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .retention import gated_retention

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 10_240
# Sweep resolution, not range: the sparse regime's discriminative signal sits in small
# log1p ring values, so fractional bits are the binding constraint (see the phase result).
FORMATS = {"q20.16": (20, 16), "q18.14": (18, 14), "q16.12": (16, 12),
           "q14.10": (14, 10), "q12.8": (12, 8), "q8.4": (8, 4), "q6.3": (6, 3)}


def quantize(values: np.ndarray, total_bits: int, fractional_bits: int) -> np.ndarray:
    """Unsigned saturating fixed point, matching `analytic.analytic_contrast_score`."""

    scale = float(2 ** fractional_bits)
    maximum = float(2 ** total_bits - 1) / scale
    return np.round(np.clip(values, 0.0, maximum) * scale) / scale


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() <= 0 or b.std() <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, score_config: Optional[StateConfig],
                   score: str, max_blocks: Optional[int] = None) -> Dict:
    """Float reference and every fixed-point format, on the same blocks."""

    scale_count = len(config.taus_us)
    rows: Dict[str, Dict[str, List[float]]] = {
        name: {"auc": [], "retention": [], "beta": [], "corr": [], "corr_abs_err": []}
        for name in ("float", *FORMATS)
    }
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
        raw_s = None
        if score_config is not None:
            raw_s = bidirectional_features(events, score_config)
            _, r_same, r_opp = per_block_operands_both(raw_s, scale_count,
                                                       score_config.feature_dim)
        else:
            r_same, r_opp = r_gate, per_block_operands_both(
                raw, scale_count, config.feature_dim)[2]

        cfg_s = score_config if score_config is not None else config
        support = ring_support(raw_s if score_config is not None else raw,
                               scale_count, cfg_s.feature_dim, cfg_s.radius)
        corr_float = safe_corr(g, r_gate)
        for name in ("float", *FORMATS):
            bits = FORMATS.get(name)
            if name == "float":
                gq, rgq, rsq, roq = g, r_gate, r_same, r_opp
            else:
                gq, rgq, rsq, roq = (quantize(v, *bits) for v in (g, r_gate, r_same, r_opp))
            corr = safe_corr(gq, rgq)
            beta = 1.0 if corr < 0.0 else 0.0
            if score == "support":
                scores = beta * gq - np.log1p(quantize(support, *bits)
                                              if name != "float" else support)
            else:
                scores = beta * gq - (rsq if score == "old" else rsq + roq)
            rows[name]["auc"].append(roc_auc_score(labels, scores))
            rows[name]["retention"].append(gated_retention(scores, corr))
            rows[name]["beta"].append(beta)
            rows[name]["corr"].append(corr)
            rows[name]["corr_abs_err"].append(abs(corr - corr_float))

    if not rows["float"]["auc"]:
        return {"blocks": 0}

    out: Dict[str, Dict] = {"blocks": produced}
    beta_float = np.asarray(rows["float"]["beta"])
    corr_float_arr = np.asarray(rows["float"]["corr"])
    for name in ("float", *FORMATS):
        record = rows[name]
        beta_q = np.asarray(record["beta"])
        err = np.asarray(record["corr_abs_err"])
        margin = np.abs(corr_float_arr)
        worst_err = float(err.max())
        out[name] = {
            "auc": float(np.mean(record["auc"])),
            "mean_retention": float(np.mean(record["retention"])),
            "beta_flip_rate": float(np.mean(beta_q != beta_float)),
            "max_corr_abs_err": worst_err,
            "mean_corr_abs_err": float(err.mean()),
            # Certified = this block's distance from theta=0 exceeds the worst perturbation
            # observed for this format, so its gate decision provably cannot flip.
            "frac_blocks_certified": float(np.mean(margin > worst_err)),
            "min_margin": float(margin.min()),
        }
    return out


def aggregate(conditions: Dict[str, Dict], name: str, field: str) -> float:
    usable = [c for c in conditions.values() if c.get("blocks")]
    return float(np.mean([c[name][field] for c in usable])) if usable else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--score", choices=("old", "new", "support"), default="old")
    parser.add_argument("--score-radius", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase13_fixed_point_adaptive.json")
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
                dvs_gate, dvs_score, args.score, args.max_blocks)
            c = dvsclean[name]
            print(f"DVSCLEAN {name}: float={c['float']['auc']:.4f} "
                  f"q12.8={c['q12.8']['auc']:.4f} q8.4={c['q8.4']['auc']:.4f} "
                  f"flip8.4={c['q8.4']['beta_flip_rate']:.4f}", flush=True)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd_gate, dnd_score, args.score, args.max_blocks)
            c = dnd21[name]
            print(f"DND21 {name}: float={c['float']['auc']:.4f} "
                  f"q12.8={c['q12.8']['auc']:.4f} q8.4={c['q8.4']['auc']:.4f} "
                  f"flip8.4={c['q8.4']['beta_flip_rate']:.4f}", flush=True)

    everything = {**{f"DVSCLEAN/{k}": v for k, v in dvsclean.items()},
                  **{f"DND21/{k}": v for k, v in dnd21.items()}}
    summary = {}
    for name in ("float", *FORMATS):
        summary[name] = {
            "DVSCLEAN_auc": aggregate(dvsclean, name, "auc"),
            "DND21_auc": aggregate(dnd21, name, "auc"),
            "overall_20_auc": 0.5 * (aggregate(dvsclean, name, "auc")
                                     + aggregate(dnd21, name, "auc")),
            "beta_flip_rate": aggregate(everything, name, "beta_flip_rate"),
            "max_corr_abs_err": max(c[name]["max_corr_abs_err"]
                                    for c in everything.values() if c.get("blocks")),
            "frac_blocks_certified": aggregate(everything, name, "frac_blocks_certified"),
            "mean_retention": aggregate(everything, name, "mean_retention"),
        }
        summary[name]["auc_delta_vs_float"] = (summary[name]["overall_20_auc"]
                                               - summary["float"]["overall_20_auc"])

    payload = {
        "phase": "13 fixed-point certificate for the adaptive path",
        "covers": ("operand quantization + gate corr + beta decision + score + retention. "
                   "Extends dnd21_fixed_point.json, which covered only the static "
                   "beta=1 filter."),
        "not_covered": ("state storage, timestamp arithmetic and exponential decay remain "
                        "float, as in the original certificate; the streamed EMA gate "
                        "(Phase 5b causal path) is NOT certified here."),
        "config": {"score": args.score, "score_radius": args.score_radius, "gate_radius": 1,
                   "formats": {k: f"unsigned {v[0]}-bit, {v[1]} fractional, saturating"
                               for k, v in FORMATS.items()}},
        "certificate_definition": (
            "A block is certified when |corr_float| exceeds the worst |corr_float - corr_q| "
            "observed for that format, so its theta=0 sign decision provably cannot flip."),
        "conditions": {"DVSCLEAN": dvsclean, "DND21": dnd21},
        "summary": summary,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
