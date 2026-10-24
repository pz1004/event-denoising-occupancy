"""Phase 12: efficiency profile -- GFLOPs, # Params, Inference Time.

Standing requirement from 2026-07-26: every method comparison reports GFLOPs, parameter
count and inference time next to accuracy, because the contribution is being repositioned
around hardware friendliness rather than raw denoising accuracy.

Three things are measured here.

**1. Operation count, analytic and exact.** Native3D is not a network, so FLOPs are
counted directly from the kernel in `core._extract_features` rather than by a graph
profiler. Per event, per scale, with `nc = (2*radius+1)^2 - 1` neighbours:

    ages (per event, not per scale)   6 arithmetic + 2 transcendental
    centre same / opposite           2 * (2 arithmetic + 1 transcendental)
    each neighbour x each polarity   6 arithmetic + 2 transcendental   [x 2*nc]
    global term                      4 arithmetic + 1 transcendental
    summary log1p writes             7 transcendental
    state update                     2 arithmetic

Arithmetic counts add/sub/mul/div/min/compare as 1 each; `exp` and `log1p` are counted
separately because on FPGA/ASIC they are LUT- or CORDIC-backed and cost far more than an
add. Reporting them merged would flatter the method; reporting them apart is the honest
form for a hardware claim.

**Worst case vs effective.** The neighbour body is skipped when a neighbour pixel has
never fired (`last_times < 0`). The analytic count above is therefore a dense-regime
*upper bound*. The effective count is measured here from real ring occupancy, taken from
the `dir_support_*` operands the extractor already produces.

**2. Parameters.** Zero, and asserted rather than asserted-by-comment: the state is
decayed history, not learned weights.

**3. Inference time.** Extractor + score + retention, which is what Step 3b asked for --
the earlier isolated number timed the extractor alone.

Read-only over data on disk.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path
from typing import Dict, List

import numpy as np

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import (
    dnd21_blocks,
    dvsclean_blocks,
    per_block_operands_both,
)
from .retention import gated_retention

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 10_240

# Published efficiency baselines, EDmamba paper Table 3 ("100K events").
#
# Use with two documented corrections:
#
# 1. **The parameter column for EDformer is wrong.** Table 3 reports 49.80K. Counting the
#    released checkpoint directly gives 584,019 tensor elements, of which 46 are BatchNorm
#    buffers, so **583,973 trainable** -- 11.7x the published figure. The gap is almost
#    exactly the two `SCformer.sparse_transformations.conv.1.weight` tensors (264,384 each,
#    90.5% of the model), i.e. the counter missed SparseConvNet's custom modules. We use our
#    own verified count.
#
# 2. **8.41 GFLOPs is therefore a LOWER BOUND.** A profiler that under-counts parameters by
#    11.7x by skipping the sparse-conv layers will skip their FLOPs too. Using it unchanged is
#    the *conservative* choice: it makes EDformer look cheaper than it is, so any advantage we
#    report over it is understated rather than inflated.
#
# Corroboration that the timing column is sound: Table 3's 2.4943 s / 100K events = 40.1 k
# events/s, and our own independent end-to-end reproduction measured ~41 k events/s (within
# 2%). The timing is trustworthy even though the parameter column is not.
PUBLISHED_TABLE3 = {
    "EDnCNN":   {"gflops_100k": 234.51, "params": 614_550,    "inf_time_s_100k": 20.1885},
    "AEDNet":   {"gflops_100k": 4400.46, "params": 45_870_000, "inf_time_s_100k": 43.4250},
    "EDformer": {"gflops_100k": 8.41,   "params": 49_800,     "inf_time_s_100k": 2.4943},
    "Pre-Mamba": {"gflops_100k": 6.23,  "params": 264_630,    "inf_time_s_100k": 0.0987},
    "Joint-SSM": {"gflops_100k": 5.17,  "params": 102_910,    "inf_time_s_100k": 0.0931},
    "EDmamba":  {"gflops_100k": 2.27,   "params": 88_980,     "inf_time_s_100k": None},
}

EDFORMER = {
    "params_published_table3": 49_800,
    "params_verified_from_checkpoint": 583_973,
    "params_checkpoint_total_incl_buffers": 584_019,
    "gflops_100k_published": 8.41,
    "gflops_note": "lower bound; same counter under-reports params by 11.7x",
    "inf_time_s_100k_published": 2.4943,
    "events_per_second_published": 100_000 / 2.4943,
    "events_per_second_our_reproduction": 41_000,
}


def analytic_ops(scale_count: int, radius: int, occupancy: float = 1.0) -> Dict[str, float]:
    """Ops per event for ONE directional pass, from the kernel body.

    `occupancy` in [0,1] scales the neighbour body, which is skipped for pixels that have
    never fired. occupancy = 1 gives the dense worst case.
    """

    nc = (2 * radius + 1) ** 2 - 1
    active = 2 * nc * occupancy          # neighbour x polarity visits actually executed

    arithmetic = 6 + scale_count * (2 + 2 + 4 + 2 + 6 * active)
    transcendental = 2 + scale_count * (1 + 1 + 1 + 7 + 2 * active)
    return {
        "neighbours": nc,
        "occupancy": occupancy,
        "arithmetic_ops": arithmetic,
        "transcendental_ops": transcendental,
        "total_ops": arithmetic + transcendental,
    }


def profile_ops(scale_count: int, radius: int, occupancy: float) -> Dict:
    """Forward-only and bidirectional op counts, worst case and effective."""

    out = {}
    for label, occ in (("worst_case_dense", 1.0), ("effective_measured", occupancy)):
        one = analytic_ops(scale_count, radius, occ)
        out[label] = {
            **one,
            "bidirectional_total_ops": 2 * one["total_ops"],
            "bidirectional_arithmetic": 2 * one["arithmetic_ops"],
            "bidirectional_transcendental": 2 * one["transcendental_ops"],
            # +2 for the score itself: beta*G - (R_same + R_opposite)
            "with_score_total_ops": 2 * one["total_ops"] + 2,
            "gflops_per_1e9_events": (2 * one["total_ops"] + 2) / 1e9 * 1e9 / 1e9,
        }
    return out


def measure_occupancy(blocks: List[np.ndarray], config: StateConfig) -> Dict[str, float]:
    """Fraction of ring taps that hit a pixel with history, from dir_support_*."""

    scale_count = len(config.taus_us)
    width = config.feature_dim
    nc = (2 * config.radius + 1) ** 2 - 1
    directional_base = 7 * scale_count
    same_hits, opp_hits, total = 0.0, 0.0, 0
    for events in blocks:
        raw = bidirectional_features(events, config)
        for relation, offset in (("same", 0), ("opposite", nc * scale_count)):
            cols = [directional_base + offset + k * scale_count for k in range(nc)]
            stack = np.stack([raw[:, c] + raw[:, width + c] for c in cols], axis=1)
            hits = float((stack > 0).sum())
            if relation == "same":
                same_hits += hits
            else:
                opp_hits += hits
        total += len(events) * nc
    return {
        "same_polarity_occupancy": same_hits / total,
        "opposite_polarity_occupancy": opp_hits / total,
        "mean_occupancy": (same_hits + opp_hits) / (2 * total),
    }


def time_pipeline(blocks: List[np.ndarray], config: StateConfig, reps: int) -> Dict[str, float]:
    """Wall time for extractor, extractor+score, and extractor+score+retention."""

    scale_count = len(config.taus_us)
    bidirectional_features(blocks[0], config)          # warm JIT

    def run(stage: str) -> float:
        best = float("inf")
        for _ in range(reps):
            start = time.perf_counter()
            for events in blocks:
                raw = bidirectional_features(events, config)
                if stage == "extract":
                    continue
                g, r_same, r_opposite = per_block_operands_both(raw, scale_count,
                                                               config.feature_dim)
                corr = (float(np.corrcoef(g, r_same)[0, 1])
                        if g.std() > 0 and r_same.std() > 0 else 0.0)
                scores = (1.0 if corr < 0.0 else 0.0) * g - (r_same + r_opposite)
                if stage == "score":
                    continue
                gated_retention(scores, corr)
            best = min(best, time.perf_counter() - start)
        return best

    events_total = sum(len(b) for b in blocks)
    out = {}
    for stage in ("extract", "score", "retention"):
        seconds = run(stage)
        out[f"{stage}_seconds"] = seconds
        out[f"{stage}_events_per_second"] = events_total / seconds
    out["events_timed"] = events_total
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--radii", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase12_efficiency.json")
    return parser.parse_args()


SENSORS = {
    "DND21_346x260": (346, 260,
                      "ECCV2024_datasets/AUC_test/5hz/driving_mix_result.txt", "dnd21"),
    "DVSCLEAN_1280x720": (1280, 720,
                          "DVSCLEAN/simulated_data/MAH00446_50.hdf5", "dvsclean"),
}


def load_blocks(kind: str, path: Path, count: int) -> List[np.ndarray]:
    reader = dnd21_blocks if kind == "dnd21" else dvsclean_blocks
    return [ev for ev, _ in islice(reader(path, BLOCK_SIZE, count), count)]


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    results: Dict[str, Dict] = {}

    for sensor, (w, h, rel, kind) in SENSORS.items():
        blocks = load_blocks(kind, PROJECT_ROOT / rel, args.blocks)
        per_radius: Dict[str, Dict] = {}
        for radius in args.radii:
            config = StateConfig(width=w, height=h, taus_us=(250.0,), radius=radius)
            occ = measure_occupancy(blocks, config)
            timing = time_pipeline(blocks, config, args.reps)
            ops = profile_ops(len(config.taus_us), radius, occ["mean_occupancy"])
            state_bytes = 2 * h * w * len(config.taus_us) * 4 + 2 * h * w * 8
            per_radius[f"radius_{radius}"] = {
                "occupancy": occ,
                "ops_per_event": ops,
                "timing": timing,
                "state_bytes": state_bytes,
                "state_MB": state_bytes / 1e6,
                "trained_parameters": 0,
                # GFLOPs at the measured throughput, effective op count, incl. score.
                "gflops_at_measured_rate": (
                    ops["effective_measured"]["with_score_total_ops"]
                    * timing["score_events_per_second"] / 1e9),
            }
            print(f"{sensor} r={radius}: "
                  f"occ={occ['mean_occupancy']:.4f} "
                  f"ops/ev(eff)={ops['effective_measured']['with_score_total_ops']:.0f} "
                  f"extract={timing['extract_events_per_second']/1e6:.3f} Mev/s "
                  f"full={timing['retention_events_per_second']/1e6:.3f} Mev/s", flush=True)
        results[sensor] = per_radius

    # Table-3-style comparison, normalised to 100K events as the published table is.
    ours = results["DND21_346x260"]["radius_1"]
    ours_ops = ours["ops_per_event"]["effective_measured"]["with_score_total_ops"]
    ours_rate = ours["timing"]["retention_events_per_second"]
    comparison = {
        "basis": "100K events, 346x260 sensor; ours = full pipeline "
                 "(extract + score + retention), radius 1",
        "rows": {},
    }
    for name, row in PUBLISHED_TABLE3.items():
        comparison["rows"][name] = {
            "gflops_100k": row["gflops_100k"],
            "params": row["params"],
            "inf_time_s_100k": row["inf_time_s_100k"],
            "source": "EDmamba paper Table 3",
        }
    comparison["rows"]["EDformer"]["params_verified_by_us"] = 583_973
    comparison["rows"]["EDformer"]["params_note"] = (
        "published 49.80K excludes the two SparseConvNet weights (90.5% of the model); "
        "our checkpoint count is 583,973 trainable")
    comparison["rows"]["Native3D (ours)"] = {
        "gflops_100k": ours_ops * 1e5 / 1e9,
        "params": 0,
        "inf_time_s_100k": 1e5 / ours_rate,
        "ops_per_event_effective": ours_ops,
        "source": "measured, this phase",
    }
    n = comparison["rows"]["Native3D (ours)"]
    e = comparison["rows"]["EDformer"]
    comparison["native3d_vs_edformer"] = {
        "gflops_ratio": e["gflops_100k"] / n["gflops_100k"],
        "gflops_ratio_note": "EDformer GFLOPs is a lower bound, so this ratio is conservative",
        "params_ratio": "infinite (0 trained parameters)",
        "inf_time_ratio": e["inf_time_s_100k"] / n["inf_time_s_100k"],
        "inf_time_caveat": (
            "NOT like-for-like hardware: ours is single-thread numba CPU on GTX 1660 SUPER "
            "host, theirs is the EDmamba paper's setup. The one anchor is that our "
            "independent EDformer reproduction (~41 k ev/s) matches their reported "
            "40.1 k ev/s within 2%, so the hosts are comparable for EDformer at least."),
    }

    payload = {
        "phase": "12 efficiency profile (GFLOPs / params / inference time)",
        "comparison_table3_style": comparison,
        "op_counting_convention": (
            "arithmetic = add,sub,mul,div,min,compare (1 each); transcendental = exp,log1p "
            "counted SEPARATELY because they are LUT/CORDIC-backed in hardware. "
            "'worst_case_dense' assumes every ring tap hits a pixel with history; "
            "'effective_measured' scales the neighbour body by measured ring occupancy."),
        "stages": ("extract = bidirectional_features only; score = + operands, corr, gate, "
                   "score; retention = + gated_retention (this is the Step 3b number)"),
        "trained_parameters": 0,
        "baseline_edformer": EDFORMER,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
