"""Phase 40: does noise density flip the sign of Delta G? ED24 answers what two datasets could not.

Section 7 concedes that with DND21 and DVSCLEAN we cannot separate *burstiness* from *noise
density* as the cause of the sign inversion the gate keys on: the two datasets differ in both at
once, so the explanations are confounded. ED24 breaks that confound by construction. It holds the
scenes fixed and sweeps real BA noise over 21 annotated levels, 1.5-3.5V of attenuator voltage, so
density varies roughly twentyfold while everything else about the recording stays put. If density
alone sets the sign of

    Delta G  =  <G>_noise  -  <G>_signal

then Delta G must cross zero somewhere inside that sweep. If it does not, density is not sufficient,
and Section 7's confound resolves in one direction rather than staying open.

The measurement is `phase27_deltaG_mechanism.scan`, imported rather than reimplemented, so every
number here is produced by the same code path as the DND21 and DVSCLEAN figures in Appendix F --
including its two artefact controls, the 15% block-edge trim and the tie dithering. `--anchor`
re-scores one DND21 condition through this script and checks it against the stored phase 27 value;
if that does not reproduce, the ED24 rows are measuring the plumbing rather than the sweep.

Read-only. Writes a new artifact and touches none.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .core import StateConfig
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_adaptive_beta import dnd21_blocks
from .phase4_ed24_blind import ED24_ROOT, ed24_blocks
from .phase27_deltaG_mechanism import BLOCK_SIZE, scan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"

#: ED24's own annotated band. Its authors exclude below 1.5V (BA noise too sparse) and above 3.5V
#: (nearly uniform), so the sweep is meaningful only inside it.
ED24_BAND = [f"{v / 10:.1f}" for v in range(15, 36)]

#: One stored phase 27 condition, used by --anchor to prove the imported scan is unchanged.
ANCHOR_CONDITION = ("5hz", "driving", "DND21 5hz driving")

#: The per-condition cap phase 27 was run at; its artifact does not record it.
PHASE27_CAP = 25


def summarise(conditions: Dict[str, Dict]) -> Dict:
    per_level: Dict[str, Dict] = {}
    for level in ED24_BAND:
        rows = [v for k, v in conditions.items() if k.rsplit("_", 1)[1] == level and v]
        if not rows:
            continue
        deltas = np.array([r["delta_G"] for r in rows])
        per_level[level] = {
            "scenes": len(rows),
            "blocks": int(sum(r["blocks"] for r in rows)),
            "delta_G": float(deltas.mean()),
            "delta_G_min": float(deltas.min()),
            "delta_G_max": float(deltas.max()),
            "scenes_positive": int((deltas > 0).sum()),
            "delta_G_trimmed": float(np.mean([r["delta_G_trimmed"] for r in rows
                                              if r["delta_G_trimmed"] is not None])),
            "delta_G_dithered": float(np.mean([r["delta_G_dithered"] for r in rows
                                               if r["delta_G_dithered"] is not None])),
        }

    levels = sorted(per_level, key=float)
    means = [per_level[l]["delta_G"] for l in levels]
    positive = [l for l in levels if per_level[l]["delta_G"] > 0]
    return {
        "levels": levels,
        "per_level": per_level,
        "delta_G_pooled": float(np.mean(means)),
        "levels_with_positive_delta_G": positive,
        "sign_flips_within_band": bool(positive) and len(positive) != len(levels),
        "span_low_to_high": [means[0], means[-1]] if means else None,
        # Spearman-style monotonicity: does Delta G track the level at all?
        "rank_corr_with_level": float(np.corrcoef(
            np.argsort(np.argsort([float(l) for l in levels])),
            np.argsort(np.argsort(means)))[0, 1]) if len(means) > 2 else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=25,
                        help="ED24 scenes, taken as an even stride over all 100")
    parser.add_argument("--max-blocks", type=int, default=25,
                        help="per condition; matches phase 27")
    parser.add_argument("--anchor", action="store_true",
                        help="re-score one DND21 condition and check it against phase 27")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase40_ed24_deltaG.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = StateConfig(width=346, height=260, taus_us=(250.0,), radius=1)

    payload: Dict = {
        "phase": "40 Delta G across ED24's 21 annotated noise levels",
        "question": ("does noise density alone set the sign of Delta G? if it does, the sign must "
                     "cross zero inside a twentyfold density sweep at fixed scenes"),
        "measurement": "phase27_deltaG_mechanism.scan, imported unchanged",
        "band": ED24_BAND,
        "max_blocks_per_condition": args.max_blocks,
    }

    if args.anchor:
        rate, scene, stored_key = ANCHOR_CONDITION
        path = PROJECT_ROOT / "ECCV2024_datasets/AUC_test" / rate / SCENE_FILES[scene]
        stored = json.loads((ARTIFACTS / "phase27_deltaG_mechanism.json").read_text())
        # Delta G is a mean over blocks, so the anchor must use the cap phase 27 ran at rather
        # than this run's -- a different cap is a different quantity and would fail spuriously.
        cap = PHASE27_CAP
        here = scan(dnd21_blocks(path, BLOCK_SIZE, cap), config, cap)
        payload["anchor_cap"] = cap
        want = stored["conditions"][stored_key]["delta_G"]
        payload["anchor"] = {"condition": stored_key, "phase27": want,
                             "recomputed": here["delta_G"],
                             "abs_error": abs(here["delta_G"] - want)}
        print(f"anchor {stored_key}: phase27 {want:+.4f} vs recomputed {here['delta_G']:+.4f} "
              f"(|err| {payload['anchor']['abs_error']:.2e})", flush=True)
        if payload["anchor"]["abs_error"] > 1e-9:
            raise ValueError(
                "The imported scan no longer reproduces phase 27 on a stored condition, so the "
                "ED24 rows below would not be comparable to Appendix F. Fix before reading them")

    scenes = sorted(p.name for p in ED24_ROOT.iterdir() if p.is_dir())
    stride = max(1, len(scenes) // args.scenes)
    scenes = scenes[::stride][:args.scenes]
    payload["scenes"] = scenes

    conditions: Dict[str, Dict] = {}
    for index, scene in enumerate(scenes, 1):
        for level in ED24_BAND:
            if not (ED24_ROOT / scene / f"{scene}_{level}.csv").exists():
                continue
            conditions[f"{scene}_{level}"] = scan(
                ed24_blocks(scene, level, BLOCK_SIZE, args.max_blocks), config, args.max_blocks)
        done = [v for k, v in conditions.items() if k.startswith(f"{scene}_") and v]
        if done:
            span = [v["delta_G"] for v in done]
            print(f"[{index}/{len(scenes)}] {scene}: dG {min(span):+.3f} .. {max(span):+.3f} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    payload["conditions"] = conditions
    payload["summary"] = summarise(conditions)
    payload["wall_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    s = payload["summary"]
    print(f"\nWrote {args.output}")
    print(f"Delta G by level: {s['span_low_to_high'][0]:+.4f} at 1.5V -> "
          f"{s['span_low_to_high'][1]:+.4f} at 3.5V")
    print(f"levels with Delta G > 0: {s['levels_with_positive_delta_G'] or 'none'}")
    print(f"sign flips inside the band: {s['sign_flips_within_band']}")


if __name__ == "__main__":
    main()
