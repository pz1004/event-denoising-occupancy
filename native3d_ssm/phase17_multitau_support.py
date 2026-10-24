"""Phase 17: multi-tau support fusion for the dense low-corr regime.

**Where this comes from.** Phase 16 measured, on the ED24 development half (labels, not the
test set), that the dense low-corr regime -- the dev analogue of the two losing E-MLB cells --
is **not** a gate problem: oracle-beta headroom is +0.0001 at 3.0 Hz/px and **+0.0000** at
4.0 Hz/px, with oracle beta* = 0 at every level. The hard gate is already optimal there, so
the pre-registered "soft gate near theta = 0" (Step 4) cannot help. What degrades is ranking:
AUC falls 0.9578 -> 0.9303 from 1.0 to 4.0 Hz/px.

**Mechanism.** `ring_support` counts neighbours that fired within `max_age = 20 * taus[-1]`,
i.e. 5 ms for the single 250 us plane. A 10,240-event block at 4.0 Hz/px spans only ~28 ms and
its events are spatially spread, so support collapses (mean 12.8 -> 4.6, entropy 4.58 -> 3.62
bits) and loses discriminative range.

**Why not just widen the window.** `max_age_factor` is currently a fixed 20.0. Selecting a new
value by dev performance would make it a *fitted* parameter and break the zero-parameter claim.
The **pre-declared tau bank {250 us, 1, 4, 16 ms}** already provides several age scales at no
extra design freedom, and multi-tau fusion has been the last untried operand axis since Phase 3b.
Because `max_age = 20 * taus[-1]`, a 4-tau configuration extends the cap to 320 ms, and the
per-scale value `state * exp(-age / tau)` then underflows to zero for the short taus at long
ages -- so support genuinely differs per scale rather than being a single threshold repeated.

**Variants compared on the same blocks** (gate key always the radius-1, 250 us plane, so the
frozen gate is untouched):

    single_tau      adopted baseline: taus = (250 us,),                 support at that plane
    multi_scale0    taus = 4-bank, support at the 250 us plane only     (isolates the max_age change)
    multi_sum       taus = 4-bank, sum of support across the 4 planes
    multi_logsum    taus = 4-bank, sum of log1p(support) across planes

Reported per ED24 noise level so the dense regime is visible separately. Reads only
`scenes[0::2]`; `scenes[1::2]` stays reserved as an untouched blind half.

Read-only. Nothing is trained and no threshold is fitted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import per_block_operands_both
from .phase16_dense_regime import LEVELS, ed24_blocks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ED24_ROOT = PROJECT_ROOT / "ECCV2024_datasets/ED24"
TAU_BANK = (250.0, 1_000.0, 4_000.0, 16_000.0)
VARIANTS = ("single_tau", "multi_scale0", "multi_sum", "multi_logsum")


def support_per_scale(raw: np.ndarray, config: StateConfig) -> np.ndarray:
    """[events, scales] count of same-polarity ring taps with a nonzero decayed value."""

    scale_count = len(config.taus_us)
    width = config.feature_dim
    neighbor_count = (2 * config.radius + 1) ** 2 - 1
    base = 7 * scale_count
    out = np.zeros((raw.shape[0], scale_count), dtype=np.float64)
    for scale in range(scale_count):
        columns = [base + k * scale_count + scale for k in range(neighbor_count)]
        stack = np.stack([raw[:, c] + raw[:, width + c] for c in columns], axis=1)
        out[:, scale] = (stack > 0).sum(axis=1)
    return out


def scan_level(scenes: List[str], level: str, gate_cfg: StateConfig,
               single_cfg: StateConfig, multi_cfg: StateConfig,
               max_blocks: Optional[int]) -> Dict:
    auc: Dict[str, List[float]] = {v: [] for v in VARIANTS}
    stats: Dict[str, List[float]] = {"support_single": [], "support_multi_sum": []}
    scale_count_gate = len(gate_cfg.taus_us)

    for scene in scenes:
        for events, labels in ed24_blocks(scene, level, max_blocks):
            labels = np.asarray(labels)
            if labels.min() == labels.max():
                continue

            raw_gate = bidirectional_features(events, gate_cfg)
            g, r_gate, _ = per_block_operands_both(raw_gate, scale_count_gate,
                                                   gate_cfg.feature_dim)
            corr = (float(np.corrcoef(g, r_gate)[0, 1])
                    if g.std() > 0 and r_gate.std() > 0 else 0.0)
            beta = 1.0 if corr < 0.0 else 0.0

            single = support_per_scale(bidirectional_features(events, single_cfg),
                                       single_cfg)[:, 0]
            multi = support_per_scale(bidirectional_features(events, multi_cfg), multi_cfg)
            stats["support_single"].append(float(single.mean()))
            stats["support_multi_sum"].append(float(multi.sum(axis=1).mean()))

            candidates = {
                "single_tau": np.log1p(single),
                "multi_scale0": np.log1p(multi[:, 0]),
                "multi_sum": np.log1p(multi.sum(axis=1)),
                "multi_logsum": np.log1p(multi).sum(axis=1),
            }
            for name, subtrahend in candidates.items():
                score = beta * g - subtrahend
                if score.std() <= 0:
                    continue
                auc[name].append(roc_auc_score(labels, score))

    if not auc["single_tau"]:
        return {"blocks": 0}
    result = {"blocks": len(auc["single_tau"])}
    for name in VARIANTS:
        result[f"auc_{name}"] = float(np.mean(auc[name])) if auc[name] else float("nan")
    for name, values in stats.items():
        result[f"mean_{name}"] = float(np.mean(values))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--max-blocks", type=int, default=4)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--half", choices=("dev", "hold"), default="dev")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase17_multitau_support.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    all_scenes = sorted(p.name for p in ED24_ROOT.iterdir() if p.is_dir())
    scenes = (all_scenes[0::2] if args.half == "dev" else all_scenes[1::2])[:args.scenes]

    gate_cfg = StateConfig(width=346, height=260, taus_us=(250.0,), radius=1)
    single_cfg = StateConfig(width=346, height=260, taus_us=(250.0,), radius=args.radius)
    multi_cfg = StateConfig(width=346, height=260, taus_us=TAU_BANK, radius=args.radius)

    levels: Dict[str, Dict] = {}
    for level in LEVELS:
        levels[level] = scan_level(scenes, level, gate_cfg, single_cfg, multi_cfg,
                                   args.max_blocks)
        c = levels[level]
        print(f"level {level:>4}: blocks={c['blocks']:4d} "
              f"single={c['auc_single_tau']:.4f} scale0={c['auc_multi_scale0']:.4f} "
              f"sum={c['auc_multi_sum']:.4f} logsum={c['auc_multi_logsum']:.4f} "
              f"| sup {c['mean_support_single']:.2f} -> {c['mean_support_multi_sum']:.2f}",
              flush=True)

    usable = [c for c in levels.values() if c.get("blocks")]
    overall = {v: float(np.mean([c[f"auc_{v}"] for c in usable])) for v in VARIANTS}
    dense = [levels[k] for k in ("3.0", "4.0") if levels[k].get("blocks")]
    dense_only = {v: float(np.mean([c[f"auc_{v}"] for c in dense])) for v in VARIANTS}

    payload = {
        "phase": "17 multi-tau support fusion for the dense low-corr regime",
        "motivation": ("Phase 16: the dense regime has ZERO gate headroom (+0.0000 at 4.0 Hz/px, "
                       "oracle beta* = 0), so it is an operand problem. Support collapses there "
                       "because max_age = 20*taus[-1] = 5 ms is short relative to the block."),
        "no_new_parameter": ("uses the pre-declared tau bank {250us, 1, 4, 16ms}; max_age_factor "
                             "is left at its existing 20.0. Nothing is fitted."),
        "gate": "unchanged: corr(G, R_same) on the radius-1, 250 us plane, theta = 0",
        "split": {"half_used": args.half, "scenes": scenes},
        "levels": levels,
        "overall_all_levels": overall,
        "dense_levels_only_3_and_4": dense_only,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("\noverall (all levels):", json.dumps(overall, indent=2))
    print("dense levels only   :", json.dumps(dense_only, indent=2))
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
