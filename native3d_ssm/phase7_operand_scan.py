"""Phase 7: rank every operand the extractor already computes but the score discards.

Motivation. The gate is exhausted: Phase 3 measured oracle-beta at 0.9589 versus the
frozen corr gate's 0.9565, so at most 0.0024 AUC is left in *any* choice of beta. The
remaining gap to locally reproduced EDmamba (0.9885 = DVSCLEAN 0.9951 / DND21 0.9819)
is 0.032 and lives almost entirely on DVSCLEAN, where ring-only (0.9321) already
equals oracle-beta -- i.e. the *operand set*, not the gate, is the binding constraint.
Tau does not rescue it either: the Phase 0 sweep moves DVSCLEAN ring-only only
0.9297 -> 0.9342 across 250 us -> 16 ms.

Meanwhile `core.bidirectional_features` computes seven scalar groups plus 2 x 8
directional neighbours per scale, and the score `s = beta*G - R` uses exactly **two**
of them (`global_density`, `ring_sum_same`). Everything else -- `ring_sum_opposite`,
`ring_max_same/opposite`, `center_same/opposite`, `log_age_same/opposite`, and all 16
directional columns -- is computed and thrown away at zero saving.

This script measures the per-block AUC of each discarded operand on the 20 labeled
dev conditions, so enhancement work is ordered by evidence instead of intuition.

Orientation convention: AUC is reported for the operand **as computed**, oriented so
that label 1 = noise. AUC < 0.5 means the operand is informative with the opposite
sign. Informativeness is therefore |AUC - 0.5|, but **picking a sign per dataset
would be label fitting** -- exactly the trap the corr gate exists to avoid. Any
operand whose useful sign flips between datasets needs a theory-fixed or gated sign
before it can be claimed, and this script flags those explicitly.

Read-only over data on disk. Nothing is trained; no rule is fitted here.
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
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 10_240
SCALAR_GROUPS = ("center_same", "center_opposite", "ring_sum_same", "ring_sum_opposite",
                 "ring_max_same", "ring_max_opposite", "global_density")


def operand_bank(raw: np.ndarray, config: StateConfig) -> Dict[str, np.ndarray]:
    """Every operand as a bidirectional (forward + reverse) sum, per event.

    Column layout is documented in `core._extract_features`; `bidirectional_features`
    concatenates [forward | reverse], each `config.feature_dim` wide.
    """

    scale_count = len(config.taus_us)
    width = config.feature_dim
    neighbor_count = (2 * config.radius + 1) ** 2 - 1

    def both(column: int) -> np.ndarray:
        return raw[:, column] + raw[:, width + column]

    bank: Dict[str, np.ndarray] = {}
    for group_index, name in enumerate(SCALAR_GROUPS):
        # scale 0 only; multi-scale fusion is a separate axis (see --taus).
        bank[name] = both(group_index * scale_count)

    directional_base = 7 * scale_count
    for relation, offset in (("same", 0), ("opposite", neighbor_count * scale_count)):
        columns = [directional_base + offset + k * scale_count
                   for k in range(neighbor_count)]
        stack = np.stack([both(c) for c in columns], axis=1)
        bank[f"dir_max_{relation}"] = stack.max(axis=1)
        bank[f"dir_sum_{relation}"] = stack.sum(axis=1)
        bank[f"dir_std_{relation}"] = stack.std(axis=1)
        bank[f"dir_range_{relation}"] = stack.max(axis=1) - stack.min(axis=1)
        bank[f"dir_support_{relation}"] = (stack > 0).sum(axis=1).astype(np.float32)

    base = (7 + 2 * neighbor_count) * scale_count
    bank["log_age_same"] = both(base + 0)
    bank["log_age_opposite"] = both(base + 1)
    return bank


def derived_scores(bank: Dict[str, np.ndarray], corr: float) -> Dict[str, np.ndarray]:
    """Candidate scores, all oriented so noise scores high (label 1 = noise)."""

    ring = bank["ring_sum_same"]
    global_density = bank["global_density"]
    beta = 1.0 if corr < 0.0 else 0.0
    both_ring = ring + bank["ring_sum_opposite"]
    eps = 1e-6
    return {
        # The current frozen method, for reference.
        "current_corr_gate": beta * global_density - ring,
        "ring_only": -ring,
        # Cheapest candidate: use the opposite-polarity ring the extractor already has.
        "both_polarity_ring": -both_ring,
        "gated_both_polarity_ring": beta * global_density - both_ring,
        # Peak rather than sum: robust to a single hot neighbour.
        "ring_max_same_only": -bank["ring_max_same"],
        "ring_sum_plus_max": -(ring + bank["ring_max_same"]),
        # Anisotropy: a coherent edge lights up a few aligned neighbours, isotropic
        # background activity does not concentrate.
        "dir_anisotropy_same": -bank["dir_std_same"],
        "ring_times_anisotropy": -(ring * (1.0 + bank["dir_std_same"])),
        # Support count = ring occupancy, the Phase 0 quantity, as a score.
        "dir_support_same_only": -bank["dir_support_same"],
        # Recency: noise events are temporally isolated.
        "log_age_same_only": bank["log_age_same"],
        "ring_over_age": -(ring / (eps + bank["log_age_same"])),
        # Centre-surround: own-pixel history vs neighbourhood.
        "centre_minus_ring": bank["center_same"] - ring,
    }


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, max_blocks: Optional[int] = None) -> Dict:
    """Per-block AUC of every raw operand and every derived candidate score."""

    operand_auc: Dict[str, List[float]] = {}
    score_auc: Dict[str, List[float]] = {}
    corrs: List[float] = []
    produced = 0
    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue
        raw = bidirectional_features(events, config)
        bank = operand_bank(raw, config)
        scale_count = len(config.taus_us)
        g = raw[:, 6 * scale_count] + raw[:, config.feature_dim + 6 * scale_count]
        r = raw[:, 2 * scale_count] + raw[:, config.feature_dim + 2 * scale_count]
        corr = float(np.corrcoef(g, r)[0, 1]) if g.std() > 0 and r.std() > 0 else 0.0
        corrs.append(corr)
        for name, values in bank.items():
            if values.std() <= 0:
                continue
            operand_auc.setdefault(name, []).append(roc_auc_score(labels, values))
        for name, values in derived_scores(bank, corr).items():
            if not np.isfinite(values).all() or values.std() <= 0:
                continue
            score_auc.setdefault(name, []).append(roc_auc_score(labels, values))
    if not corrs:
        return {"blocks": 0}
    return {
        "blocks": produced,
        "mean_corr": float(np.mean(corrs)),
        "operand_auc": {k: float(np.mean(v)) for k, v in operand_auc.items()},
        "score_auc": {k: float(np.mean(v)) for k, v in score_auc.items()},
    }


def aggregate(conditions: Dict[str, Dict], field: str) -> Dict[str, float]:
    usable = [c for c in conditions.values() if c.get("blocks")]
    if not usable:
        return {}
    names = set(usable[0][field])
    for c in usable[1:]:
        names &= set(c[field])
    return {n: float(np.mean([c[field][n] for c in usable])) for n in sorted(names)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--radius", type=int, default=1,
                        help="Ring radius. improve_plan.md §4.2 predicts >1 helps the sparse regime.")
    parser.add_argument("--taus", nargs="+", type=float, default=[250.0])
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=["DVSCLEAN", "DND21"])
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase7_operand_scan.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    taus = tuple(args.taus)

    dvsclean: Dict[str, Dict] = {}
    if "DVSCLEAN" in args.datasets:
        config = StateConfig(width=1280, height=720, taus_us=taus, radius=args.radius)
        for seq in SEQUENCES:
            for level in (50, 100):
                name = f"{seq}_{level}"
                dvsclean[name] = scan_condition(
                    dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE, args.max_blocks),
                    config, args.max_blocks)
                print(f"DVSCLEAN {name}: blocks={dvsclean[name]['blocks']}", flush=True)

    dnd21: Dict[str, Dict] = {}
    if "DND21" in args.datasets:
        config = StateConfig(width=346, height=260, taus_us=taus, radius=args.radius)
        for rate in DND21_PAPER:
            for scene, fname in SCENE_FILES.items():
                name = f"{rate}_{scene}"
                dnd21[name] = scan_condition(
                    dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                    config, args.max_blocks)
                print(f"DND21 {name}: blocks={dnd21[name]['blocks']}", flush=True)

    everything = {**{f"DVSCLEAN/{k}": v for k, v in dvsclean.items()},
                  **{f"DND21/{k}": v for k, v in dnd21.items()}}
    summary = {}
    for field in ("operand_auc", "score_auc"):
        per_dataset = {"DVSCLEAN": aggregate(dvsclean, field),
                       "DND21": aggregate(dnd21, field),
                       "overall": aggregate(everything, field)}
        # Flag operands whose informative sign differs between the two datasets: those
        # cannot be used without a theory-fixed or gated sign.
        flips = []
        for name in per_dataset["overall"]:
            a = per_dataset["DVSCLEAN"].get(name)
            b = per_dataset["DND21"].get(name)
            if a is None or b is None:
                continue
            if (a - 0.5) * (b - 0.5) < 0 and min(abs(a - 0.5), abs(b - 0.5)) > 0.02:
                flips.append({"operand": name, "DVSCLEAN": a, "DND21": b})
        summary[field] = {**per_dataset, "sign_flips_between_datasets": flips}

    payload = {
        "phase": "7 operand scan (which discarded operands carry signal)",
        "why": ("beta is exhausted (oracle 0.9589 vs gate 0.9565) and DVSCLEAN ring-only "
                "0.9321 already equals oracle-beta, so the operand set is the binding "
                "constraint on the 0.032 gap to reproduced EDmamba (0.9885)."),
        "orientation": ("AUC for the operand as computed, label 1 = noise. AUC<0.5 means "
                        "informative with the opposite sign; informativeness = |AUC-0.5|. "
                        "Choosing a sign per dataset would be label fitting."),
        "config": {"radius": args.radius, "taus_us": list(taus), "block_size": BLOCK_SIZE,
                   "max_blocks": args.max_blocks},
        "conditions": {"DVSCLEAN": dvsclean, "DND21": dnd21},
        "summary": summary,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")
    for field in ("operand_auc", "score_auc"):
        print(f"\n=== {field} (dataset means) ===")
        rows = summary[field]["overall"]
        for name in sorted(rows, key=lambda n: -abs(rows[n] - 0.5)):
            print(f"  {name:<28} DVSCLEAN={summary[field]['DVSCLEAN'].get(name, float('nan')):.4f} "
                  f"DND21={summary[field]['DND21'].get(name, float('nan')):.4f} "
                  f"overall={rows[name]:.4f}")
        if summary[field]["sign_flips_between_datasets"]:
            print("  sign flips between datasets:",
                  [f["operand"] for f in summary[field]["sign_flips_between_datasets"]])


if __name__ == "__main__":
    main()
