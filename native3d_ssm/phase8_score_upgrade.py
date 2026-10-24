"""Phase 8: adopt and blind-validate the both-polarity ring score.

Phase 7 measured that the score currently discards an operand the extractor already
computes. Candidate replacement, one term longer:

    old:  s = beta * G - R_same
    new:  s = beta * G - (R_same + R_opposite)

Phase 7's scan put the new score at 0.9600 overall versus 0.9565, with the gain
concentrated on DVSCLEAN (+0.0109), the regime that has been the bottleneck since
Phase 0. This script promotes that from a scan row to a properly validated result by
re-running the Phase 3/4 protocol on it.

Three things need re-deriving, because Phase 7 only reported a single aggregate:

1. **The oracle-beta ceiling.** The old 0.9589 was measured over the *old* operand set
   and therefore bounds nothing once the operands change. Recomputed here per condition
   over the full beta grid.
2. **The gate key.** beta is gated on corr(G,R) < 0. With a new R, there are two
   candidate keys: corr(G, R_same) (unchanged, the frozen Phase-3 statistic) and
   corr(G, R_same + R_opposite) (consistent with the new score). A 6-block probe showed
   the sign is preserved either way but the magnitude shrinks on DND21 (-0.239 ->
   -0.190), pushing that dataset closer to theta = 0 -- which Phase 5b showed is exactly
   where gate fidelity degrades. So both keys are evaluated, and the safer one wins.
3. **Per-condition transfer.** theta = 0 is theory-fixed, so there is nothing to fit and
   no LODO is required (Phase 3+/4 settled that). What must be shown is that the blind
   theta = 0 gate on the *new* score holds across all 20 conditions, not just on average.

Emits an EDmamba-Table-1-style per-condition markdown plus JSON. Read-only over data on
disk; no model is trained and no parameter is fitted.
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
from .evaluate_adaptive_beta import BETA_GRID, dnd21_blocks, dvsclean_blocks
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 10_240
BETA_ONE = int(np.argmin(np.abs(BETA_GRID - 1.0)))

# Locally reproduced EDmamba (see the two reproducibility reports) -- the Track-A target.
EDMAMBA_REPRODUCED = {"DVSCLEAN": 0.995098, "DND21": 0.981860}


def operands(raw: np.ndarray, config: StateConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bidirectional G, same-polarity ring, opposite-polarity ring (250 us plane)."""

    scale_count = len(config.taus_us)
    width = config.feature_dim
    g = raw[:, 6 * scale_count] + raw[:, width + 6 * scale_count]
    r_same = raw[:, 2 * scale_count] + raw[:, width + 2 * scale_count]
    r_opposite = raw[:, 3 * scale_count] + raw[:, width + 3 * scale_count]
    return g, r_same, r_opposite


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() <= 0 or b.std() <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, max_blocks: Optional[int] = None) -> Dict:
    """Per-block AUC over the beta grid for both score variants, plus both gate keys."""

    records: Dict[str, List] = {
        "corr_same": [], "corr_both": [],
        "auc_old": [], "auc_new": [],          # [blocks, beta] grids
    }
    produced = 0
    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels)
        raw = bidirectional_features(events, config)
        g, r_same, r_opposite = operands(raw, config)
        r_both = r_same + r_opposite
        records["corr_same"].append(safe_corr(g, r_same))
        records["corr_both"].append(safe_corr(g, r_both))
        if labels.min() == labels.max():
            nan_row = np.full(len(BETA_GRID), np.nan)
            records["auc_old"].append(nan_row)
            records["auc_new"].append(nan_row)
            continue
        records["auc_old"].append(np.array(
            [roc_auc_score(labels, beta * g - r_same) for beta in BETA_GRID]))
        records["auc_new"].append(np.array(
            [roc_auc_score(labels, beta * g - r_both) for beta in BETA_GRID]))
    if not produced:
        return {"blocks": 0}

    corr_same = np.asarray(records["corr_same"])
    corr_both = np.asarray(records["corr_both"])
    auc_old = np.vstack(records["auc_old"])
    auc_new = np.vstack(records["auc_new"])

    def blind(auc: np.ndarray, key: np.ndarray) -> float:
        """Realized AUC under the frozen zero-parameter gate: beta = 1 iff key < 0."""

        index = np.where(key < 0.0, BETA_ONE, 0)
        return float(np.nanmean(auc[np.arange(len(index)), index]))

    mean_old = np.nanmean(auc_old, axis=0)
    mean_new = np.nanmean(auc_new, axis=0)
    return {
        "blocks": produced,
        "mean_corr_same": float(corr_same.mean()),
        "mean_corr_both": float(corr_both.mean()),
        "frac_blocks_neg_same": float(np.mean(corr_same < 0)),
        "frac_blocks_neg_both": float(np.mean(corr_both < 0)),
        # Endpoints and ceilings, per score variant.
        "old_beta0": float(mean_old[0]),
        "old_beta1": float(mean_old[BETA_ONE]),
        "old_oracle": float(np.nanmax(mean_old)),
        "old_oracle_beta": float(BETA_GRID[int(np.nanargmax(mean_old))]),
        "new_beta0": float(mean_new[0]),
        "new_beta1": float(mean_new[BETA_ONE]),
        "new_oracle": float(np.nanmax(mean_new)),
        "new_oracle_beta": float(BETA_GRID[int(np.nanargmax(mean_new))]),
        # Blind theta=0 gate: old score, and new score under each candidate key.
        "old_blind_key_same": blind(auc_old, corr_same),
        "new_blind_key_same": blind(auc_new, corr_same),
        "new_blind_key_both": blind(auc_new, corr_both),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase8_score_upgrade.json")
    parser.add_argument("--markdown", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase8_table1.md")
    return parser.parse_args()


ROWS = [
    ("Native3D old  beta=1", "old_beta1"),
    ("Native3D old  ring-only", "old_beta0"),
    ("Native3D old  blind theta=0", "old_blind_key_same"),
    ("Native3D old  oracle-beta", "old_oracle"),
    ("Native3D NEW  beta=1", "new_beta1"),
    ("Native3D NEW  both-ring only", "new_beta0"),
    ("Native3D NEW  blind, key=corr(G,Rsame)", "new_blind_key_same"),
    ("Native3D NEW  blind, key=corr(G,Rboth)", "new_blind_key_both"),
    ("Native3D NEW  oracle-beta", "new_oracle"),
]


def aggregate(conditions: Dict[str, Dict], field: str) -> float:
    usable = [c[field] for c in conditions.values() if c.get("blocks")]
    return float(np.mean(usable)) if usable else float("nan")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    dvs_config = StateConfig(width=1280, height=720, taus_us=(250.0,))
    dnd_config = StateConfig(width=346, height=260, taus_us=(250.0,))

    dvsclean: Dict[str, Dict] = {}
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            dvsclean[name] = scan_condition(
                dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE, args.max_blocks),
                dvs_config, args.max_blocks)
            c = dvsclean[name]
            print(f"DVSCLEAN {name}: blocks={c['blocks']} "
                  f"old_blind={c['old_blind_key_same']:.4f} "
                  f"new_blind={c['new_blind_key_same']:.4f} "
                  f"new_oracle={c['new_oracle']:.4f}", flush=True)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd_config, args.max_blocks)
            c = dnd21[name]
            print(f"DND21 {name}: blocks={c['blocks']} "
                  f"old_blind={c['old_blind_key_same']:.4f} "
                  f"new_blind={c['new_blind_key_same']:.4f} "
                  f"new_oracle={c['new_oracle']:.4f}", flush=True)

    everything = {**{f"DVSCLEAN/{k}": v for k, v in dvsclean.items()},
                  **{f"DND21/{k}": v for k, v in dnd21.items()}}

    summary = {}
    for label, field in ROWS:
        summary[label] = {
            "DVSCLEAN": aggregate(dvsclean, field),
            "DND21": aggregate(dnd21, field),
            "overall_20": aggregate(everything, field),
        }
    summary["EDmamba reproduced (target)"] = {
        "DVSCLEAN": EDMAMBA_REPRODUCED["DVSCLEAN"],
        "DND21": EDMAMBA_REPRODUCED["DND21"],
        "overall_20": float(np.mean(list(EDMAMBA_REPRODUCED.values()))),
    }

    verdict = {
        "old_blind_overall": summary["Native3D old  blind theta=0"]["overall_20"],
        "new_blind_key_same_overall": summary["Native3D NEW  blind, key=corr(G,Rsame)"]["overall_20"],
        "new_blind_key_both_overall": summary["Native3D NEW  blind, key=corr(G,Rboth)"]["overall_20"],
        "new_oracle_overall": summary["Native3D NEW  oracle-beta"]["overall_20"],
        "old_oracle_overall": summary["Native3D old  oracle-beta"]["overall_20"],
    }
    verdict["gain_over_old_blind"] = (verdict["new_blind_key_same_overall"]
                                      - verdict["old_blind_overall"])
    verdict["better_key"] = ("corr(G,Rsame)"
                             if verdict["new_blind_key_same_overall"]
                             >= verdict["new_blind_key_both_overall"] else "corr(G,Rboth)")
    verdict["headroom_left_in_beta"] = (verdict["new_oracle_overall"]
                                        - max(verdict["new_blind_key_same_overall"],
                                              verdict["new_blind_key_both_overall"]))
    # Per-condition regression check: does the new score ever *hurt*?
    losses = {name: c["new_blind_key_same"] - c["old_blind_key_same"]
              for name, c in everything.items() if c.get("blocks")}
    verdict["conditions_improved"] = int(sum(1 for v in losses.values() if v > 0))
    verdict["conditions_total"] = len(losses)
    verdict["worst_regression"] = min(losses.items(), key=lambda kv: kv[1])
    verdict["best_gain"] = max(losses.items(), key=lambda kv: kv[1])

    lines = ["# Phase 8 — both-polarity ring score, blind protocol\n",
             "| Method | DVSCLEAN | DND21 | Overall (20) |", "|---|---:|---:|---:|"]
    for label in list(summary):
        s = summary[label]
        lines.append(f"| {label} | {s['DVSCLEAN']:.4f} | {s['DND21']:.4f} | {s['overall_20']:.4f} |")
    markdown = "\n".join(lines)

    payload = {
        "phase": "8 both-polarity ring score upgrade",
        "old_score": "s = beta*G - R_same",
        "new_score": "s = beta*G - (R_same + R_opposite)",
        "gate": "beta = 1 iff key < 0, theta = 0 theory-fixed (nothing fitted)",
        "note": ("Phase 7 reported the new score at 0.9600 as a single aggregate; this phase "
                 "re-derives the oracle ceiling on the new operand set, tests both gate keys, "
                 "and checks every condition individually."),
        "conditions": {"DVSCLEAN": dvsclean, "DND21": dnd21},
        "summary": summary,
        "verdict": verdict,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(markdown + "\n")
    print("\n" + markdown)
    print("\nVERDICT:", json.dumps(verdict, indent=2))
    print(f"\nWrote {args.output} and {args.markdown} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
