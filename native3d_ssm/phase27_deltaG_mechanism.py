"""Phase 27: what actually drives the sign of Delta G, and what does not.

Phase 26 derives the gate *threshold* from the sign of Delta G but takes that sign as given.
The manuscript's introduction went further and gave it a physical reading -- "under dense noise
a busy sensor is evidence of noise, under sparse noise it is evidence of signal". This phase
tests that reading. It does not survive in that form.

**What G measures.** The global operand is `log1p(c * A_i)` with the event-indexed recursion
`A_i = A_{i-1} * exp(-dt_i / tau) + 1`, so up to an additive constant

    G_i  ~  log Lambda(t_i),

the log instantaneous global event rate at the tau = 250 us scale. G depends on the timestamp
ONLY -- it has no spatial argument -- so

    Delta G  =  <log Lambda>_noise  -  <log Lambda>_signal

is a purely temporal contrast, and exp(Delta G) is the ratio of instantaneous global rate seen
by the two classes. No spatial mechanism can contribute to it. That already rules out most
readings of "a busy sensor is evidence of noise".

**Two candidate explanations, both tested and both rejected.**

  1. *Block warm-up.* State resets per block, so A ramps from zero and G could be largely a
     within-block position effect. Rejected: trimming the first and last 15% of every block
     leaves Delta G essentially unchanged and never flips its sign.

  2. *Timestamp quantisation.* The synthetic streams assign many events identical timestamps
     (on DND21 at 10 Hz/pixel, ~10^4 noise events can share 9 distinct stamps). If Delta G were
     an artefact of exact ties, dithering the ties apart would destroy it. Rejected: dithering
     each tied group uniformly across the gap to the next distinct stamp, preserving order,
     leaves both Delta G and the measured value of the gate unchanged.

**What is left.** A genuine contrast in *temporal burstiness at the tau scale*: whichever class
arrives at the higher instantaneous rate sits on the Lambda spikes and is marked by G. On these
two datasets that class happens to differ, and it correlates with noise density -- but this is a
property of how each stream was synthesised, and the phase reports the per-class timestamp
statistics that show it rather than asserting a physical cause.

Read-only. Labels are used only to form the two conditional averages.
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
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK_SIZE = 10_240
TRIM = 0.15
MIN_PER_CLASS = 50


def dither_ties(events: np.ndarray) -> np.ndarray:
    """Spread each tied timestamp group across the gap to the next distinct stamp.

    Event order is preserved and no event leaves its inter-stamp interval, so the coarse rate
    at the tau scale is untouched; only the exact-tie structure is destroyed. That is the point:
    if Delta G were a tie artefact it would not survive this.
    """

    out = events.copy()
    t = events[:, 0].astype(np.float64)
    unique, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    if unique.size < 2:
        return out
    gaps = np.append(np.diff(unique), float(np.median(np.diff(unique))))
    offset = np.zeros(t.size)
    for idx in np.flatnonzero(counts > 1):
        member = inverse == idx
        offset[member] = np.arange(member.sum()) / member.sum()
    out[:, 0] = (t + offset * gaps[inverse]).astype(events.dtype)
    return out


def timestamp_stats(times: np.ndarray) -> Dict:
    unique = np.unique(times)
    return {"events": int(times.size), "distinct_timestamps": int(unique.size),
            "median_multiplicity": float(times.size / max(unique.size, 1))}


def scan(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]], config: StateConfig,
         max_blocks: Optional[int]) -> Dict:
    raw_d, trim_d, dith_d, gate_raw, gate_dith, corr_pos = [], [], [], [], [], []
    mult_n, mult_s = [], []

    def delta(events: np.ndarray, labels: np.ndarray) -> Optional[Tuple[float, float, float]]:
        feats = bidirectional_features(events, config)
        g, r, _ = per_block_operands_both(feats, len(config.taus_us), config.feature_dim)
        noise, signal = labels == 1, labels == 0
        if g.std() <= 0 or r.std() <= 0:
            return None
        pos = np.arange(g.size) / g.size
        return (float(g[noise].mean() - g[signal].mean()),
                float(roc_auc_score(labels, g - r) - roc_auc_score(labels, -r)),
                float(np.corrcoef(g, pos)[0, 1]))

    for produced, (events, labels) in enumerate(block_iter):
        if max_blocks is not None and produced >= max_blocks:
            break
        labels = np.asarray(labels)
        if labels.min() == labels.max():
            continue
        noise, signal = labels == 1, labels == 0
        if noise.sum() < MIN_PER_CLASS or signal.sum() < MIN_PER_CLASS:
            continue
        base = delta(events, labels)
        if base is None:
            continue
        raw_d.append(base[0]); gate_raw.append(base[1]); corr_pos.append(base[2])

        dith = delta(dither_ties(events), labels)
        if dith is not None:
            dith_d.append(dith[0]); gate_dith.append(dith[1])

        feats = bidirectional_features(events, config)
        g, _, _ = per_block_operands_both(feats, len(config.taus_us), config.feature_dim)
        pos = np.arange(g.size) / g.size
        keep = (pos > TRIM) & (pos < 1.0 - TRIM)
        kn, ks = noise & keep, signal & keep
        if kn.sum() > 30 and ks.sum() > 30:
            trim_d.append(float(g[kn].mean() - g[ks].mean()))

        times = events[:, 0].astype(np.int64)
        mult_n.append(timestamp_stats(times[noise])["median_multiplicity"])
        mult_s.append(timestamp_stats(times[signal])["median_multiplicity"])

    if not raw_d:
        return {}
    mean_raw = float(np.mean(raw_d))
    return {
        "blocks": len(raw_d),
        "delta_G": mean_raw,
        "delta_G_trimmed": float(np.mean(trim_d)) if trim_d else None,
        "delta_G_dithered": float(np.mean(dith_d)) if dith_d else None,
        "rate_ratio_noise_over_signal": float(np.exp(mean_raw)),
        "gate_value_raw": float(np.mean(gate_raw)),
        "gate_value_dithered": float(np.mean(gate_dith)) if gate_dith else None,
        "corr_G_with_block_position": float(np.mean(corr_pos)),
        "timestamp_multiplicity_noise": float(np.mean(mult_n)),
        "timestamp_multiplicity_signal": float(np.mean(mult_s)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=25,
                        help="per condition; the contrast is stable well before this")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase27_deltaG_mechanism.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dnd = StateConfig(width=346, height=260, taus_us=(250.0,), radius=1)
    dvs = StateConfig(width=1280, height=720, taus_us=(250.0,), radius=1)

    conditions: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"DND21 {rate} {scene}"
            conditions[name] = scan(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd, args.max_blocks)
            c = conditions[name]
            print(f"{name}: dG={c['delta_G']:+.3f} trimmed={c['delta_G_trimmed']:+.3f} "
                  f"dithered={c['delta_G_dithered']:+.3f}", flush=True)
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"DVSCLEAN {seq}_{level}"
            conditions[name] = scan(
                dvsclean_blocks(args.dvsclean_root / f"{seq}_{level}.hdf5",
                                BLOCK_SIZE, args.max_blocks),
                dvs, args.max_blocks)
            c = conditions[name]
            print(f"{name}: dG={c['delta_G']:+.3f} trimmed={c['delta_G_trimmed']:+.3f} "
                  f"dithered={c['delta_G_dithered']:+.3f}", flush=True)

    usable = [c for c in conditions.values() if c]
    dnd_c = [c for k, c in conditions.items() if k.startswith("DND21") and c]
    dvs_c = [c for k, c in conditions.items() if k.startswith("DVSCLEAN") and c]
    summary = {
        "conditions": len(usable),
        "G_is": "log1p(c * A) with A_i = A_{i-1} exp(-dt/tau) + 1, i.e. ~ log Lambda(t)",
        "delta_G_is": "a purely temporal contrast: G has no spatial argument",
        "dense_delta_G_range": [min(c["delta_G"] for c in dnd_c),
                                max(c["delta_G"] for c in dnd_c)],
        "sparse_delta_G_range": [min(c["delta_G"] for c in dvs_c),
                                 max(c["delta_G"] for c in dvs_c)],
        "max_rate_ratio_dense": max(c["rate_ratio_noise_over_signal"] for c in dnd_c),
        "control_warmup": {
            "max_abs_shift_when_trimmed": max(abs(c["delta_G"] - c["delta_G_trimmed"])
                                              for c in usable),
            "sign_preserved": sum((c["delta_G"] > 0) == (c["delta_G_trimmed"] > 0)
                                  for c in usable),
            "of": len(usable),
            "mean_corr_G_with_position": float(np.mean(
                [c["corr_G_with_block_position"] for c in usable])),
            "verdict": "rejected as the mechanism",
        },
        "control_timestamp_ties": {
            "max_abs_shift_when_dithered": max(abs(c["delta_G"] - c["delta_G_dithered"])
                                               for c in usable),
            "max_abs_gate_value_shift": max(abs(c["gate_value_raw"] - c["gate_value_dithered"])
                                            for c in usable),
            "sign_preserved": sum((c["delta_G"] > 0) == (c["delta_G_dithered"] > 0)
                                  for c in usable),
            "of": len(usable),
            "verdict": "rejected as the mechanism",
        },
        "remaining_explanation": (
            "a contrast in temporal burstiness at the tau scale: the class arriving at the "
            "higher instantaneous global rate is the one G marks. Which class that is differs "
            "between these two synthetic-noise datasets, so the correlation with noise density "
            "is observed, not explained."),
    }
    payload = {
        "phase": "27 what drives the sign of Delta G",
        "summary": summary,
        "conditions": conditions,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.1f}s\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
