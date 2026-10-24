"""Phase 2' experiment: rate-adaptive global weight beta for the local/global score.

Score: s_i = beta * (G_forward + G_reverse) - (R_forward + R_reverse), on the
250-us plane. beta = 0 is ring-only, beta = 1 is the current fixed contrast.

Phase 0 showed the global operand is correctly signed on dense-uniform-noise
DND21 but wrong-signed on sparse DVSCLEAN. This script tests whether a single,
frozen, label-free rule beta(rho) -- rho = per-block per-pixel event rate --
recovers DVSCLEAN without harming DND21. Three honesty levels are reported:

  1. oracle beta per condition (grid argmax with labels; a ceiling, not claimable);
  2. in-sample rho->beta law (one monotone rule fit on all conditions);
  3. leave-one-dataset-out (LODO) frozen rule, applied PER BLOCK on the held-out
     dataset from only its label-free rho -- the honest generalization number.

Read-only over data on disk. No model training; beta is a fixed function.
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
from .data import iter_labeled_text
from .evaluate_dnd21 import PAPER_AUC, SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES, iter_hdf5_full_blocks, prepare_native_events

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BETA_GRID = np.round(np.arange(0.0, 1.201, 0.1), 3)  # 0.0 .. 1.2
BETA_ONE_INDEX = int(np.argmin(np.abs(BETA_GRID - 1.0)))

# Frozen external baselines (see the two reproducibility reports).
EDMAMBA = {"DVSCLEAN": 0.995098, "DND21": 0.981860}


def per_block_operands(raw: np.ndarray, scale_count: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bidirectional 250-us global sum G and same-polarity ring sum R (log operands)."""

    g = raw[:, 6 * scale_count] + raw[:, width + 6 * scale_count]
    r = raw[:, 2 * scale_count] + raw[:, width + 2 * scale_count]
    return g, r


def per_block_operands_both(raw: np.ndarray, scale_count: int,
                            width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bidirectional G, same-polarity ring and opposite-polarity ring (250 us plane).

    Additive superset of `per_block_operands`, which is left untouched because its
    two-tuple return is what every pre-Phase-8 script and cited number depends on.
    Column convention is identical to `phase8_score_upgrade.operands`.
    """

    g = raw[:, 6 * scale_count] + raw[:, width + 6 * scale_count]
    r_same = raw[:, 2 * scale_count] + raw[:, width + 2 * scale_count]
    r_opposite = raw[:, 3 * scale_count] + raw[:, width + 3 * scale_count]
    return g, r_same, r_opposite


def ring_support(raw: np.ndarray, scale_count: int, width: int, radius: int,
                 forward_only: bool = False) -> np.ndarray:
    """Count of same-polarity ring taps carrying any history.

    An integer in [0, nc] under both passes -- the bidirectional branch ORs the forward and
    reverse halves per tap rather than adding two counts, so the bound is nc = 24 at radius 2
    either way, not 2*nc. That is what makes it exactly representable in 5 bits, the property
    Phase 14 exploits to repair the fixed-point failure the ring *magnitude* operand causes
    (Phase 13).

    `forward_only` drops the reverse pass, which is what the no-lookahead causal rows need.
    """

    neighbor_count = (2 * radius + 1) ** 2 - 1
    base = 7 * scale_count
    columns = [base + k * scale_count for k in range(neighbor_count)]
    if forward_only:
        stack = np.stack([raw[:, c] for c in columns], axis=1)
    else:
        stack = np.stack([raw[:, c] + raw[:, width + c] for c in columns], axis=1)
    return (stack > 0).sum(axis=1).astype(np.float64)


def block_rate_per_pixel(events_us: np.ndarray, width: int, height: int) -> float:
    """Label-free per-pixel event rate rho (events / s / pixel) for one block."""

    duration_us = float(events_us[-1, 0] - events_us[0, 0])
    if duration_us <= 0:
        return 0.0
    return len(events_us) / (duration_us * 1e-6) / (width * height)


def scan_condition(
    block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
    config: StateConfig,
    max_blocks: Optional[int],
) -> Dict:
    """Per block: rho, corr(G,R), and block AUC at every beta on the grid."""

    scale_count = len(config.taus_us)
    width = config.feature_dim
    rho_list: List[float] = []
    corr_list: List[float] = []
    auc_grid: List[np.ndarray] = []
    produced = 0
    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        raw = bidirectional_features(events, config)
        g, r = per_block_operands(raw, scale_count, width)
        rho_list.append(block_rate_per_pixel(events, config.width, config.height))
        if g.std() > 0 and r.std() > 0:
            corr_list.append(float(np.corrcoef(g, r)[0, 1]))
        else:
            corr_list.append(0.0)
        if labels.min() == labels.max():
            auc_grid.append(np.full(len(BETA_GRID), np.nan))
        else:
            auc_grid.append(np.array([roc_auc_score(labels, beta * g - r) for beta in BETA_GRID]))
        produced += 1
    auc = np.vstack(auc_grid)  # [blocks, beta]
    mean_over_blocks = np.nanmean(auc, axis=0)
    oracle_index = int(np.nanargmax(mean_over_blocks))
    return {
        "blocks": produced,
        "rho_mean": float(np.mean(rho_list)),
        "corr_GR_mean": float(np.mean(corr_list)),
        "auc_by_beta": mean_over_blocks,          # kept in-memory (not serialized raw)
        "per_block_rho": np.asarray(rho_list),
        "per_block_corr": np.asarray(corr_list),
        "per_block_auc": auc,
        "auc_beta0": float(mean_over_blocks[0]),
        "auc_beta1": float(mean_over_blocks[BETA_ONE_INDEX]),
        "oracle_beta": float(BETA_GRID[oracle_index]),
        "auc_oracle": float(mean_over_blocks[oracle_index]),
    }


def dvsclean_blocks(path: Path, block_size: int, max_blocks: Optional[int]):
    for block in iter_hdf5_full_blocks(path, block_size, max_blocks):
        yield prepare_native_events(block), block["label"].astype(np.int64, copy=False)


def dnd21_blocks(path: Path, block_size: int, max_blocks: Optional[int]):
    produced = 0
    for values in iter_labeled_text(path, block_size):
        if len(values) != block_size:
            break
        if max_blocks is not None and produced >= max_blocks:
            break
        yield values[:, :4], values[:, 4]
        produced += 1


def beta_rule(rho: np.ndarray, k: float, log_rho_c: float, beta_max: float = 1.0) -> np.ndarray:
    """Monotone logistic gate on log10 rho, clamped to [0, beta_max]."""

    rho = np.maximum(np.asarray(rho, dtype=np.float64), 1e-9)
    return beta_max / (1.0 + np.exp(-k * (np.log10(rho) - log_rho_c)))


def realized_auc(conditions: Dict, k: float, log_rho_c: float) -> float:
    """Mean over conditions of mean-over-blocks AUC under per-block beta=rule(rho)."""

    condition_means = []
    for cond in conditions.values():
        betas = beta_rule(cond["per_block_rho"], k, log_rho_c)
        snap = np.clip(np.round(betas / 0.1).astype(int), 0, len(BETA_GRID) - 1)
        block_auc = cond["per_block_auc"][np.arange(len(snap)), snap]
        condition_means.append(np.nanmean(block_auc))
    return float(np.mean(condition_means))


def fit_rule(conditions: Dict) -> Tuple[float, float]:
    """Grid-search (k, log_rho_c) maximizing realized AUC on the given conditions."""

    best = (-1.0, None)
    for k in (2.0, 4.0, 6.0, 8.0, 12.0):
        for log_rho_c in np.linspace(-0.6, 1.0, 17):
            score = realized_auc(conditions, k, float(log_rho_c))
            if score > best[0]:
                best = (score, (k, float(log_rho_c)))
    return best[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path, default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path, default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--block-size", type=int, default=10_240)
    parser.add_argument("--dnd21-max-blocks", type=int, default=None)
    parser.add_argument("--dvsclean-max-blocks", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase2_adaptive_beta.json",
    )
    return parser.parse_args()


def dataset_mean(conditions: Dict, key: str) -> float:
    return float(np.mean([c[key] for c in conditions.values()]))


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    dvs_config = StateConfig(width=1280, height=720, taus_us=(250.0,))
    dnd_config = StateConfig(width=346, height=260, taus_us=(250.0,))

    dvsclean = {}
    for seq in SEQUENCES:
        for lvl in (50, 100):
            path = args.dvsclean_root / f"{seq}_{lvl}.hdf5"
            dvsclean[f"{seq}_{lvl}"] = scan_condition(
                dvsclean_blocks(path, args.block_size, args.dvsclean_max_blocks),
                dvs_config, args.dvsclean_max_blocks)
            c = dvsclean[f"{seq}_{lvl}"]
            print(f"DVSCLEAN {seq}_{lvl}: rho={c['rho_mean']:.3f} beta0={c['auc_beta0']:.4f} "
                  f"beta1={c['auc_beta1']:.4f} oracle(b={c['oracle_beta']:.1f})={c['auc_oracle']:.4f}", flush=True)

    dnd21 = {}
    for rate in PAPER_AUC:
        for scene, fname in SCENE_FILES.items():
            path = args.dnd21_root / rate / fname
            dnd21[f"{rate}_{scene}"] = scan_condition(
                dnd21_blocks(path, args.block_size, args.dnd21_max_blocks),
                dnd_config, args.dnd21_max_blocks)
            c = dnd21[f"{rate}_{scene}"]
            print(f"DND21 {rate}_{scene}: rho={c['rho_mean']:.3f} beta0={c['auc_beta0']:.4f} "
                  f"beta1={c['auc_beta1']:.4f} oracle(b={c['oracle_beta']:.1f})={c['auc_oracle']:.4f}", flush=True)

    # Which key predicts oracle beta better: log10 rho or corr(G,R)?
    all_conditions = {**{f"DVSCLEAN/{k}": v for k, v in dvsclean.items()},
                      **{f"DND21/{k}": v for k, v in dnd21.items()}}
    beta_star = np.array([c["oracle_beta"] for c in all_conditions.values()])
    log_rho = np.log10([max(c["rho_mean"], 1e-9) for c in all_conditions.values()])
    corr_gr = np.array([c["corr_GR_mean"] for c in all_conditions.values()])
    key_corr = {
        "pearson_betastar_vs_log10rho": float(np.corrcoef(beta_star, log_rho)[0, 1]),
        "pearson_betastar_vs_corrGR": float(np.corrcoef(beta_star, corr_gr)[0, 1]),
    }

    # In-sample single law (fit on everything) and LODO frozen rules.
    rule_insample = fit_rule(all_conditions)
    rule_fit_on_dnd = fit_rule(dnd21)      # apply to DVSCLEAN (unseen)
    rule_fit_on_dvs = fit_rule(dvsclean)   # apply to DND21 (unseen)

    def summarize(conditions, rule):
        k, lrc = rule
        return {
            "beta0_ring_only": dataset_mean(conditions, "auc_beta0"),
            "beta1_current_contrast": dataset_mean(conditions, "auc_beta1"),
            "oracle_beta_per_condition": dataset_mean(conditions, "auc_oracle"),
            "frozen_rule_realized": realized_auc(conditions, k, lrc),
            "frozen_rule_kc": [k, lrc],
        }

    summary = {
        "beta_key_predictiveness": key_corr,
        "rule_forms": {
            "in_sample_all": {"k": rule_insample[0], "log_rho_c": rule_insample[1]},
            "lodo_fit_on_DND21": {"k": rule_fit_on_dnd[0], "log_rho_c": rule_fit_on_dnd[1]},
            "lodo_fit_on_DVSCLEAN": {"k": rule_fit_on_dvs[0], "log_rho_c": rule_fit_on_dvs[1]},
        },
        "DVSCLEAN": {
            "edmamba": EDMAMBA["DVSCLEAN"],
            "in_sample_rule": summarize(dvsclean, rule_insample),
            "lodo_rule_from_DND21": summarize(dvsclean, rule_fit_on_dnd),
        },
        "DND21": {
            "edmamba": EDMAMBA["DND21"],
            "in_sample_rule": summarize(dnd21, rule_insample),
            "lodo_rule_from_DVSCLEAN": summarize(dnd21, rule_fit_on_dvs),
        },
    }

    def serialize(conditions):
        return {name: {kk: (vv if not isinstance(vv, np.ndarray) else None)
                       for kk, vv in c.items()
                       if kk not in ("auc_by_beta", "per_block_rho", "per_block_auc")}
                for name, c in conditions.items()}

    payload = {
        "phase": "2' rate-adaptive global weight beta",
        "score": "s = beta*(G_fwd+G_rev) - (R_fwd+R_rev), 250us plane",
        "beta_grid": BETA_GRID.tolist(),
        "rho_definition": "per-block events / second / pixel (label-free)",
        "conditions": {"DVSCLEAN": serialize(dvsclean), "DND21": serialize(dnd21)},
        "summary": summary,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output.resolve()} in {payload['wall_seconds']:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
