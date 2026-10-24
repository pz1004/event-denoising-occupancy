"""Phase 5b: causal online corr(G,R) -- the actual runtime-adaptation claim.

Phases 2'-6 gate a whole 10,240-event block with a corr(G,R) computed over that
same block, so event i is gated using events after i. That is a batch statistic:
adequate for offline benchmarking, but it is not runtime adaptation and hardware
cannot buffer a block before deciding.

Here the gate statistic is streamed instead (`causal_corr`): running mean, variance
and covariance of the two operands, so beta_i uses only events <= i. State
**persists across blocks** within a recording -- the persist-vs-reset choice
`improve_plan.md` §2 (row 3) demanded be settled -- because persistence is what
makes the estimator track a stream whose noise regime drifts.

Two experiments:

1. **Fidelity on the labeled dev sets.** Does a causal gate preserve the batch
   result? Per condition, block-mean AUC (the official metric) under
   - `batch_gate`      per-block corr, the Phase 3/4 reference
   - `cumulative`      exact running corr over the whole recording, zero parameters
   - `ema_<alpha>`     exponentially weighted, effective window ~1/alpha events
   - `beta1` / `beta0` the two fixed endpoints
   plus per-event agreement with the batch gate's decision.

   Also a **fully causal** row: the same causal gate on **forward-only** operands, so
   neither the decay nor the adaptation looks ahead. Phases 0-6 all use the disclosed
   within-block bidirectional pass, so this is the first number with no lookahead at
   all.

2. **Adaptation latency across a regime change.** E-MLB (+corr, real sensor) and
   DND21 (-corr) are both 346x260, so their operand streams can be spliced into one
   non-stationary stream that genuinely crosses theta = 0. Measures how many events
   each estimator needs before the gate settles on the new regime -- the number a
   runtime-adaptation claim actually rests on. The cumulative estimator is expected
   to fail here by construction; that contrast is the point of the forgetting factor.

Read-only over data on disk. No model is trained; beta remains a fixed function of a
label-free statistic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .causal_corr import (
    DEFAULT_WARMUP,
    causal_beta,
    crossing_index,
    cumulative_corr,
    ema_corr,
)
from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks, ring_support
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 10_240
ALPHAS = (1e-4, 1e-3, 1e-2)          # effective windows ~10k / 1k / 100 events
SETTLE_HOLD = 5_000                  # events the new sign must hold to count as settled


def condition_streams(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                      config: StateConfig, max_blocks: Optional[int] = None,
                      score_config: Optional[StateConfig] = None) -> Dict:
    """Concatenate one recording's per-block operands into contiguous streams.

    `score_config`, when given, is a wider ring used only for the SCORE rings; the gate
    statistic keeps using `config` (radius 1). Phase 11 measured that the radius-2 gain is
    entirely the operand and that widening the gate key costs 0.0028 AUC.
    """

    scale_count = len(config.taus_us)
    feature_width = config.feature_dim
    g_parts, r_parts, gf_parts, rf_parts, label_parts = [], [], [], [], []
    ro_parts, rof_parts = [], []
    sr_parts, sro_parts, srf_parts, srof_parts = [], [], [], []
    sup_parts, supf_parts = [], []
    sizes: List[int] = []
    produced = 0
    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        raw = bidirectional_features(events, config)
        forward_g = raw[:, 6 * scale_count]
        forward_r = raw[:, 2 * scale_count]
        forward_ro = raw[:, 3 * scale_count]
        g_parts.append(forward_g + raw[:, feature_width + 6 * scale_count])
        r_parts.append(forward_r + raw[:, feature_width + 2 * scale_count])
        ro_parts.append(forward_ro + raw[:, feature_width + 3 * scale_count])
        gf_parts.append(forward_g)
        rf_parts.append(forward_r)
        rof_parts.append(forward_ro)
        cfg_s = score_config if score_config is not None else config
        raw_for_support = raw
        if score_config is not None:
            raw_s = bidirectional_features(events, score_config)
            raw_for_support = raw_s
            width_s = score_config.feature_dim
            s_forward_r = raw_s[:, 2 * scale_count]
            s_forward_ro = raw_s[:, 3 * scale_count]
            sr_parts.append(s_forward_r + raw_s[:, width_s + 2 * scale_count])
            sro_parts.append(s_forward_ro + raw_s[:, width_s + 3 * scale_count])
            srf_parts.append(s_forward_r)
            srof_parts.append(s_forward_ro)
        # Phase 14 operand: integer count of same-polarity ring taps with history.
        sup_parts.append(ring_support(raw_for_support, scale_count,
                                      cfg_s.feature_dim, cfg_s.radius))
        supf_parts.append(ring_support(raw_for_support, scale_count,
                                       cfg_s.feature_dim, cfg_s.radius, forward_only=True))
        label_parts.append(np.asarray(labels))
        sizes.append(len(events))
    if not sizes:
        return {"blocks": 0}
    return {
        "blocks": len(sizes),
        "sizes": np.asarray(sizes),
        "g": np.concatenate(g_parts).astype(np.float64),
        "r": np.concatenate(r_parts).astype(np.float64),
        "r_opposite": np.concatenate(ro_parts).astype(np.float64),
        "g_forward": np.concatenate(gf_parts).astype(np.float64),
        "r_forward": np.concatenate(rf_parts).astype(np.float64),
        "r_opposite_forward": np.concatenate(rof_parts).astype(np.float64),
        "labels": np.concatenate(label_parts),
        # Score-side rings (wider radius). Fall back to the gate rings when unused.
        "r_score": (np.concatenate(sr_parts).astype(np.float64) if sr_parts
                    else np.concatenate(r_parts).astype(np.float64)),
        "r_opposite_score": (np.concatenate(sro_parts).astype(np.float64) if sro_parts
                             else np.concatenate(ro_parts).astype(np.float64)),
        "r_score_forward": (np.concatenate(srf_parts).astype(np.float64) if srf_parts
                            else np.concatenate(rf_parts).astype(np.float64)),
        "r_opposite_score_forward": (np.concatenate(srof_parts).astype(np.float64) if srof_parts
                                     else np.concatenate(rof_parts).astype(np.float64)),
        "score_radius": (score_config.radius if score_config is not None else config.radius),
        "support": np.concatenate(sup_parts).astype(np.float64),
        "support_forward": np.concatenate(supf_parts).astype(np.float64),
    }


def block_slices(sizes: np.ndarray) -> List[slice]:
    edges = np.concatenate(([0], np.cumsum(sizes)))
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(len(sizes))]


def batch_beta(stream: Dict, slices: List[slice]) -> Tuple[np.ndarray, np.ndarray]:
    """Phase 3/4 reference: one corr per block, expanded to per-event beta."""

    beta = np.zeros(stream["g"].size, dtype=np.float64)
    corrs = np.zeros(len(slices), dtype=np.float64)
    for index, chunk in enumerate(slices):
        g, r = stream["g"][chunk], stream["r"][chunk]
        corr = float(np.corrcoef(g, r)[0, 1]) if g.std() > 0 and r.std() > 0 else 0.0
        corrs[index] = corr
        beta[chunk] = 1.0 if corr < 0.0 else 0.0
    return beta, corrs


def block_mean_auc(scores: np.ndarray, labels: np.ndarray,
                   slices: List[slice]) -> float:
    """Mean over blocks of within-block ROC AUC (the metric used since Phase 0)."""

    values = []
    for chunk in slices:
        block_labels = labels[chunk]
        if block_labels.min() == block_labels.max():
            continue
        values.append(roc_auc_score(block_labels, scores[chunk]))
    return float(np.mean(values)) if values else float("nan")


def evaluate_condition(stream: Dict, warmup: int, alphas=ALPHAS,
                       g_radius: int = 1) -> Dict:
    """AUC and batch-agreement for every gate variant on one recording."""

    slices = block_slices(stream["sizes"])
    g, r = stream["g"], stream["r"]
    labels = stream["labels"]
    reference_beta, block_corrs = batch_beta(stream, slices)

    rows: Dict[str, Dict[str, float]] = {}

    def record(name: str, beta: np.ndarray, g_used: np.ndarray, r_used: np.ndarray) -> None:
        rows[name] = {
            "auc": block_mean_auc(beta * g_used - r_used, labels, slices),
            "frac_events_beta1": float(np.mean(beta)),
            "agreement_with_batch": float(np.mean(beta == reference_beta)),
        }

    record("beta1_fixed", np.ones(g.size), g, r)
    record("beta0_ring_only", np.zeros(g.size), g, r)
    record("batch_gate", reference_beta, g, r)

    cumulative = cumulative_corr(g, r)
    record("causal_cumulative", causal_beta(cumulative, warmup), g, r)
    record("causal_cumulative_no_warmup", causal_beta(cumulative, 0), g, r)
    for alpha in alphas:
        record(f"causal_ema_{alpha:g}", causal_beta(ema_corr(g, r, alpha), warmup), g, r)

    # Phase 8 score: same gate (key = corr(G,R_same), frozen by 2.4), both-ring subtrahend.
    r_both = r + stream["r_opposite"]
    record("new_beta1_fixed", np.ones(g.size), g, r_both)
    record("new_beta0_ring_only", np.zeros(g.size), g, r_both)
    record("new_batch_gate", reference_beta, g, r_both)
    record("new_causal_cumulative", causal_beta(cumulative, warmup), g, r_both)
    for alpha in alphas:
        record(f"new_causal_ema_{alpha:g}", causal_beta(ema_corr(g, r, alpha), warmup), g, r_both)

    # Wider score ring (Phase 11 configuration): gate key stays on the radius-1 ring above,
    # only the subtrahend widens. Emitted only when a score_config was supplied.
    if stream.get("score_radius") not in (None, g_radius):
        r_s = stream["r_score"]
        record("rad_beta1_fixed", np.ones(g.size), g, r_s)
        record("rad_beta0_ring_only", np.zeros(g.size), g, r_s)
        record("rad_batch_gate", reference_beta, g, r_s)
        record("rad_causal_cumulative", causal_beta(cumulative, warmup), g, r_s)
        for alpha in alphas:
            record(f"rad_causal_ema_{alpha:g}",
                   causal_beta(ema_corr(g, r, alpha), warmup), g, r_s)

    # Phase 14 operand rows: s = beta*G - log1p(ring_support). Gate key unchanged.
    if "support" in stream:
        log_sup = np.log1p(stream["support"])
        record("sup_beta0_ring_only", np.zeros(g.size), g, log_sup)
        record("sup_batch_gate", reference_beta, g, log_sup)
        record("sup_causal_cumulative", causal_beta(cumulative, warmup), g, log_sup)
        for alpha in alphas:
            record(f"sup_causal_ema_{alpha:g}",
                   causal_beta(ema_corr(g, r, alpha), warmup), g, log_sup)

    # Fully causal: causal gate on forward-only operands (no bidirectional lookahead).
    g_fwd, r_fwd = stream["g_forward"], stream["r_forward"]
    forward_corr = cumulative_corr(g_fwd, r_fwd)
    rows["fully_causal_cumulative_forward"] = {
        "auc": block_mean_auc(causal_beta(forward_corr, warmup) * g_fwd - r_fwd, labels, slices),
        "frac_events_beta1": float(np.mean(causal_beta(forward_corr, warmup))),
        "agreement_with_batch": None,       # different operands: not comparable
    }
    rows["forward_only_beta0"] = {
        "auc": block_mean_auc(-r_fwd, labels, slices),
        "frac_events_beta1": 0.0,
        "agreement_with_batch": None,
    }
    # The no-lookahead-anywhere number on the Phase 8 score (replaces the stale 0.8494).
    r_both_fwd = r_fwd + stream["r_opposite_forward"]
    rows["new_fully_causal_cumulative_forward"] = {
        "auc": block_mean_auc(causal_beta(forward_corr, warmup) * g_fwd - r_both_fwd,
                              labels, slices),
        "frac_events_beta1": float(np.mean(causal_beta(forward_corr, warmup))),
        "agreement_with_batch": None,
    }
    rows["new_forward_only_beta0"] = {
        "auc": block_mean_auc(-r_both_fwd, labels, slices),
        "frac_events_beta1": 0.0,
        "agreement_with_batch": None,
    }
    if "support_forward" in stream:
        log_sup_fwd = np.log1p(stream["support_forward"])
        rows["sup_fully_causal_cumulative_forward"] = {
            "auc": block_mean_auc(causal_beta(forward_corr, warmup) * g_fwd - log_sup_fwd,
                                  labels, slices),
            "frac_events_beta1": float(np.mean(causal_beta(forward_corr, warmup))),
            "agreement_with_batch": None,
        }
        rows["sup_forward_only_beta0"] = {
            "auc": block_mean_auc(-log_sup_fwd, labels, slices),
            "frac_events_beta1": 0.0,
            "agreement_with_batch": None,
        }
    if stream.get("score_radius") not in (None, g_radius):
        r_s_fwd = stream["r_score_forward"]
        rows["rad_fully_causal_cumulative_forward"] = {
            "auc": block_mean_auc(causal_beta(forward_corr, warmup) * g_fwd - r_s_fwd,
                                  labels, slices),
            "frac_events_beta1": float(np.mean(causal_beta(forward_corr, warmup))),
            "agreement_with_batch": None,
        }
        rows["rad_forward_only_beta0"] = {
            "auc": block_mean_auc(-r_s_fwd, labels, slices),
            "frac_events_beta1": 0.0,
            "agreement_with_batch": None,
        }

    return {
        "blocks": stream["blocks"],
        "events": int(g.size),
        "mean_block_corr": float(np.mean(block_corrs)),
        "frac_blocks_beta1_batch": float(np.mean(block_corrs < 0.0)),
        "final_cumulative_corr": float(cumulative[-1]),
        "rows": rows,
    }


def emlb_operand_stream(path: Path, config: StateConfig, max_events: int) -> Optional[Dict]:
    """Operand stream for one E-MLB recording (+corr regime, same 346x260 sensor)."""

    from .phase5_emlb_mesr import read_aedat4

    events = read_aedat4(path, max_events)
    if events is None or len(events) < BLOCK_SIZE:
        return None
    ordered = np.ascontiguousarray(
        np.column_stack((events["timestamp"].astype(np.int64),
                         events["x"].astype(np.int64),
                         events["y"].astype(np.int64),
                         (events["polarity"].astype(np.int64) > 0).astype(np.int64))),
        dtype=np.int64)
    blocks = ((ordered[start:start + BLOCK_SIZE],
               np.zeros(len(ordered[start:start + BLOCK_SIZE]), dtype=np.int64))
              for start in range(0, len(ordered) - BLOCK_SIZE + 1, BLOCK_SIZE))
    return condition_streams(blocks, config)


def splice_experiment(first: Dict, second: Dict, first_name: str, second_name: str,
                      alphas=ALPHAS) -> Dict:
    """Adaptation latency after a regime change at the splice point."""

    g = np.concatenate([first["g"], second["g"]])
    r = np.concatenate([first["r"], second["r"]])
    junction = int(first["g"].size)
    target = -1 if second["mean_block_corr"] < 0 else 1

    variants = {"cumulative": cumulative_corr(g, r)}
    for alpha in alphas:
        variants[f"ema_{alpha:g}"] = ema_corr(g, r, alpha)

    out = {
        "first": first_name,
        "second": second_name,
        "junction_event": junction,
        "total_events": int(g.size),
        "first_mean_block_corr": first["mean_block_corr"],
        "second_mean_block_corr": second["mean_block_corr"],
        "target_sign": target,
        "settle_hold_events": SETTLE_HOLD,
        "variants": {},
    }
    for name, corr in variants.items():
        index = crossing_index(corr, target, offset=junction, hold=SETTLE_HOLD)
        out["variants"][name] = {
            "settled_at_event": index,
            "latency_events": (index - junction) if index >= 0 else None,
            "latency_blocks": ((index - junction) / BLOCK_SIZE) if index >= 0 else None,
            "corr_before_junction": float(corr[junction - 1]),
            "corr_at_end": float(corr[-1]),
            "settled": bool(index >= 0),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--splice-emlb", type=str, default="N-END/Bat/Bat-ND16-1.aedat4",
                        help="E-MLB recording used as the +corr half of the splice test.")
    parser.add_argument("--splice-dnd21", type=str, default="5hz/driving_mix_result.txt",
                        help="DND21 recording used as the -corr half of the splice test.")
    parser.add_argument("--splice-max-events", type=int, default=1_024_000)
    parser.add_argument("--score-radius", type=int, default=1,
                        help="Ring radius for the SCORE. Gate key stays at radius 1 "
                             "(Phase 11: widening the gate costs 0.0028 AUC).")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--max-blocks", type=int, default=None,
                        help="Per-condition block cap (debug only; default = all).")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase5b_causal_corr_phase8score.json")
    return parser.parse_args()


def aggregate_rows(conditions: Dict[str, Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    """Unweighted mean over conditions for each gate variant."""

    usable = [c for c in conditions.values() if c.get("blocks")]
    if not usable:
        return {}
    names = list(usable[0]["rows"])
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for name in names:
        out[name] = {}
        for field in ("auc", "frac_events_beta1", "agreement_with_batch"):
            values = [c["rows"][name][field] for c in usable
                      if c["rows"][name][field] is not None]
            out[name][field] = float(np.mean(values)) if values else None
    return out


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dvs_config = StateConfig(width=1280, height=720, taus_us=(250.0,))
    dnd_config = StateConfig(width=346, height=260, taus_us=(250.0,))
    dvs_score_config = dnd_score_config = None
    if args.score_radius != 1:
        dvs_score_config = StateConfig(width=1280, height=720, taus_us=(250.0,),
                                       radius=args.score_radius)
        dnd_score_config = StateConfig(width=346, height=260, taus_us=(250.0,),
                                       radius=args.score_radius)

    dvsclean: Dict[str, Dict] = {}
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            stream = condition_streams(
                dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE, args.max_blocks),
                dvs_config, args.max_blocks, dvs_score_config)
            dvsclean[name] = evaluate_condition(stream, args.warmup)
            c = dvsclean[name]
            print(f"DVSCLEAN {name}: blocks={c['blocks']} corr={c['mean_block_corr']:+.3f} "
                  f"auc batch={c['rows']['batch_gate']['auc']:.4f} "
                  f"cum={c['rows']['causal_cumulative']['auc']:.4f} "
                  f"agree={c['rows']['causal_cumulative']['agreement_with_batch']:.4f}", flush=True)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            stream = condition_streams(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd_config, args.max_blocks, dnd_score_config)
            dnd21[name] = evaluate_condition(stream, args.warmup)
            c = dnd21[name]
            print(f"DND21 {name}: blocks={c['blocks']} corr={c['mean_block_corr']:+.3f} "
                  f"auc batch={c['rows']['batch_gate']['auc']:.4f} "
                  f"cum={c['rows']['causal_cumulative']['auc']:.4f} "
                  f"agree={c['rows']['causal_cumulative']['agreement_with_batch']:.4f}", flush=True)

    # Regime-change splice: E-MLB (+corr) <-> DND21 (-corr), both 346x260.
    splices: Dict[str, Dict] = {}
    emlb_path = args.emlb_root / args.splice_emlb
    if emlb_path.exists():
        emlb_stream = emlb_operand_stream(emlb_path, dnd_config, args.splice_max_events)
        dnd_stream = condition_streams(
            dnd21_blocks(args.dnd21_root / args.splice_dnd21, BLOCK_SIZE,
                         args.splice_max_events // BLOCK_SIZE),
            dnd_config)
        if emlb_stream and emlb_stream.get("blocks"):
            emlb_eval = evaluate_condition(emlb_stream, args.warmup)
            emlb_stream["mean_block_corr"] = emlb_eval["mean_block_corr"]
            dnd_stream["mean_block_corr"] = float(np.mean(
                batch_beta(dnd_stream, block_slices(dnd_stream["sizes"]))[1]))
            splices["emlb_then_dnd21"] = splice_experiment(
                emlb_stream, dnd_stream, args.splice_emlb, args.splice_dnd21)
            splices["dnd21_then_emlb"] = splice_experiment(
                dnd_stream, emlb_stream, args.splice_dnd21, args.splice_emlb)
            for key, value in splices.items():
                print(f"\nsplice {key}: corr {value['first_mean_block_corr']:+.3f} -> "
                      f"{value['second_mean_block_corr']:+.3f} at event {value['junction_event']}")
                for name, v in value["variants"].items():
                    print(f"   {name:>12}: settled={v['settled']} "
                          f"latency={v['latency_events']} events "
                          f"({v['latency_blocks'] if v['latency_blocks'] is None else round(v['latency_blocks'], 2)} blocks)",
                          flush=True)
    else:
        print(f"splice test skipped: {emlb_path} not found", flush=True)

    payload = {
        "phase": "5b causal online corr(G,R) -- re-validated on the Phase 8 score",
        "scores": {
            "old rows (unprefixed)": "s = beta*G - R_same",
            "new rows (new_* prefix)": "s = beta*G - (R_same + R_opposite)",
            "gate key": "corr(G, R_same) for BOTH -- frozen by decision 2.4",
        },
        "gate": "beta_i = 1 iff running corr(G,R) at event i < 0 (theta = 0, theory-fixed)",
        "state": "persists across blocks within a recording (persist, not reset)",
        "cold_start": f"beta = 0 (ring-only) for the first {args.warmup} events",
        "metric": "block-mean ROC AUC, the metric used since Phase 0",
        "caveat": ("bidirectional rows keep the disclosed within-block bidirectional decay pass; "
                   "only the gate statistic is causal. The fully_causal_* row uses forward-only "
                   "operands so neither axis looks ahead."),
        "alphas": list(ALPHAS),
        "score_radius": args.score_radius,
        "gate_radius": 1,
        "conditions": {"DVSCLEAN": dvsclean, "DND21": dnd21},
        "aggregate": {
            "DVSCLEAN": aggregate_rows(dvsclean),
            "DND21": aggregate_rows(dnd21),
            "overall_20": aggregate_rows({**{f"DVSCLEAN/{k}": v for k, v in dvsclean.items()},
                                          **{f"DND21/{k}": v for k, v in dnd21.items()}}),
        },
        "splice": splices,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
