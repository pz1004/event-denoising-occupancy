"""Phase 56: is LED harder because it is recorded, or only because it is sparse?

THE CONFOUND. Section 4.5 says recorded background activity is a much harder problem than
the injected noise the operand was selected on, and rests that on block-mean AUC falling
from 0.9811 on the development sets to 0.7723 on LED. Those two sets of streams differ in
more than one way. LED averages 14.9% noise; the development sets are far denser. A shift in
class balance alone moves ROC-AUC, and this paper's own ED24 result shows the operand's
advantage growing with noise density, so "sparser" is a live alternative explanation for
"harder" and the claim cannot stand until it is tested.

THE TEST. Pair every development condition's noise fraction with the block-mean AUC already
published for it, and ask whether AUC tracks density across the 20. If it does, the LED drop
is at least partly a density effect and the sentence has to be weakened. If it does not, and
in particular if the conditions nearest LED's density still score near the top, then density
does not account for the drop.

Two further controls come free. DVSCLEAN is recorded at 1280x720, the same geometry as LED,
so a resolution explanation can be excluded by comparison rather than by argument. And the
lowest-density development condition gives a direct, if single, point of comparison.

WHAT IS AND IS NOT ESTABLISHED. Ruling density out is not the same as establishing that
recording is the cause; the two datasets differ in scene content, sensor, and noise process
as well. The output records the correlation and the nearest-neighbour comparison and nothing
stronger, and `n_at_led_density` is reported so the reader can see how thin that comparison
is.

Noise fractions are computed over **every** block of every condition, not a sample. The AUCs
are read from `phase19_gate_ablation.json` rather than recomputed, so this phase adds no new
scoring path and cannot disagree with the published table.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BLOCK = 10_240
ARM = "log_support|gated"

#: LED figures this phase compares against, read from phase54 rather than transcribed.
LED_ARTEFACT = ARTIFACTS / "phase54_led.json"
#: Conditions at or below this noise fraction count as "LED-like density".
LED_DENSITY_BAND = 0.30


def condition_noise_fractions(dnd21_root: Path, dvsclean_root: Path) -> Dict[str, Dict]:
    """Noise fraction per condition, over every complete block. Labels only, no scoring."""

    out: Dict[str, Dict] = {}
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            fractions = [float(np.mean(labels)) for _, labels in
                         dnd21_blocks(dnd21_root / rate / fname, BLOCK, None)]
            out[f"{rate}_{scene}"] = {"dataset": "DND21", "blocks": len(fractions),
                                      "noise_fraction": float(np.mean(fractions)),
                                      "width": 346, "height": 260}
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            fractions = [float(np.mean(labels)) for _, labels in
                         dvsclean_blocks(dvsclean_root / f"{name}.hdf5", BLOCK, None)]
            out[name] = {"dataset": "DVSCLEAN", "blocks": len(fractions),
                         "noise_fraction": float(np.mean(fractions)),
                         "width": 1280, "height": 720}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase56_density_control.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    auc_source = json.loads((ARTIFACTS / "phase19_gate_ablation.json").read_text())["conditions"]
    led = json.loads(LED_ARTEFACT.read_text())
    led_fractions = np.array([r["noise_fraction"] for r in led["per_scene"].values()])
    led_auc = led["summary"]["support"]["block_mean_auc"]

    conditions = condition_noise_fractions(args.dnd21_root, args.dvsclean_root)
    for name, record in conditions.items():
        record["block_mean_auc"] = float(auc_source[record["dataset"]][name]["auc"][ARM])

    fractions = np.array([r["noise_fraction"] for r in conditions.values()])
    aucs = np.array([r["block_mean_auc"] for r in conditions.values()])
    correlation = float(np.corrcoef(fractions, aucs)[0, 1])

    at_density = {n: r for n, r in conditions.items()
                  if r["noise_fraction"] <= LED_DENSITY_BAND}
    nearest = min(conditions.items(),
                  key=lambda kv: abs(kv[1]["noise_fraction"] - float(led_fractions.mean())))
    same_geometry = {n: r for n, r in conditions.items() if r["width"] == 1280}

    payload = {
        "phase": "56 density control for the LED gap",
        "question": ("Section 4.5 attributes a 0.9811 -> 0.7723 drop in block-mean AUC to "
                     "recorded rather than injected noise. LED is also far sparser. Does "
                     "AUC track noise fraction across the development conditions?"),
        "arm": ARM,
        "auc_source": "phase19_gate_ablation.json (published values, not recomputed)",
        "development": {
            "conditions": len(conditions),
            "noise_fraction_mean": float(fractions.mean()),
            "noise_fraction_min": float(fractions.min()),
            "noise_fraction_max": float(fractions.max()),
            "block_mean_auc_min": float(aucs.min()),
            "block_mean_auc_max": float(aucs.max()),
            "corr_noise_fraction_vs_auc": correlation,
            "per_condition": conditions,
        },
        "led": {
            "scenes": int(len(led_fractions)),
            "noise_fraction_mean": float(led_fractions.mean()),
            "noise_fraction_median": float(np.median(led_fractions)),
            "noise_fraction_min": float(led_fractions.min()),
            "noise_fraction_max": float(led_fractions.max()),
            "block_mean_auc": float(led_auc),
        },
        "density_control": {
            "band": LED_DENSITY_BAND,
            "n_at_led_density": len(at_density),
            "conditions_at_led_density": {n: {"noise_fraction": r["noise_fraction"],
                                              "block_mean_auc": r["block_mean_auc"]}
                                          for n, r in at_density.items()},
            "nearest_condition": {
                "name": nearest[0],
                "dataset": nearest[1]["dataset"],
                "noise_fraction": nearest[1]["noise_fraction"],
                "block_mean_auc": nearest[1]["block_mean_auc"],
            },
            "verdict": ("density does not account for the LED gap"
                        if aucs.min() > led_auc + 0.10
                        else "density may account for part of the LED gap"),
            "caveat": ("ruling density out is not the same as establishing that recording "
                       "is the cause: scene content, sensor and noise process differ too, "
                       f"and only {len(at_density)} development condition(s) sit inside the "
                       "density band"),
        },
        "resolution_control": {
            "note": ("DVSCLEAN is 1280x720, the same geometry as LED, so a resolution "
                     "explanation can be excluded by comparison rather than by argument"),
            "conditions_at_1280x720": len(same_geometry),
            "block_mean_auc_min": float(min(r["block_mean_auc"]
                                            for r in same_geometry.values())),
            "block_mean_auc_max": float(max(r["block_mean_auc"]
                                            for r in same_geometry.values())),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"development: {len(conditions)} conditions, noise "
          f"{fractions.min():.3f}-{fractions.max():.3f}, AUC {aucs.min():.4f}-{aucs.max():.4f}")
    print(f"corr(noise fraction, block AUC) = {correlation:+.3f}")
    print(f"LED: noise mean {led_fractions.mean():.4f} median "
          f"{np.median(led_fractions):.4f}, block AUC {led_auc:.4f}")
    print(f"nearest development condition by density: {nearest[0]} "
          f"({nearest[1]['noise_fraction']:.4f} noise) -> {nearest[1]['block_mean_auc']:.4f}")
    print(f"conditions at <= {LED_DENSITY_BAND} noise: {len(at_density)}")
    print(f"1280x720 development AUC range: "
          f"{payload['resolution_control']['block_mean_auc_min']:.4f}-"
          f"{payload['resolution_control']['block_mean_auc_max']:.4f}")
    print(f"verdict: {payload['density_control']['verdict']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
