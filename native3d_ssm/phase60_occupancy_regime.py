"""Phase 60: does LED sit in the failure regime our own occupancy model predicted?

WHAT FORCED THIS. Phase 59 puts a published rule-based filter, EBF, far ahead of us on LED:
it is not that recorded noise is hard for event denoising, it is hard for *this score*.
Section 4.5 currently says the former, and it has to stop. But a defeat with no mechanism is
a weak thing to print, and there is a specific mechanism already written down in this paper
that would explain it, made before LED was ever scored.

Supplementary S5 derives the operand's behaviour from a Binomial occupancy model and states
its own failure regime: "a nearly empty ring, where S is almost always zero and carries no
order, and a nearly full one, where it saturates." It then asserts that no benchmark used
here samples that regime -- mean support on ED24 is 9.87 of 24 -- and says testing it "requires
a benchmark that drives ring occupancy to its ends while holding the scene fixed", which we
declined to run.

LED may be exactly that benchmark, arrived at by accident. Its streams are 1280x720 and its
noise fraction is low, so a 10,240-event block spans very little time, and the radius-2 ring
of a given event may be empty for almost everything in it. If so, three things follow at
once: the S5 claim that no benchmark samples the failure regime is false and must be
corrected; the LED collapse is predicted rather than merely observed; and EBF's advantage
has an explanation, since its 32 ms window is fixed in *time* and does not shrink when the
event rate rises.

THE MEASUREMENT. Over every complete block of the 20 development conditions and of the 66 LED
scenes: the mean ring support S, the fraction of events with S = 0, the fraction at the
saturating S = 24, and the wall duration the block spans. The support statistic is the
published extractor's, not a reimplementation. ED24's 9.87 is read from Phase 32's artefact
so the three benchmarks are on one axis.

WHAT WOULD FALSIFY THE EXPLANATION. If LED's mean support sits in the same range as
development's, the empty-ring account is wrong and the LED gap needs a different one. This
phase is written to be able to say that.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import ring_support
from .phase54_led import HEIGHT as LED_HEIGHT
from .phase54_led import LED_ROOT, TAU_US, WIDTH as LED_WIDTH, iter_scene
from .phase57_ebf_labelled import BLOCK, GEOMETRY, conditions, load_condition
from .ebf import EBF_TAU_US

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
SCORE_RADIUS = 2
#: The saturating value of the radius-2 ring support: (2*2+1)^2 - 1.
SUPPORT_MAX = 24


def block_profile(events: np.ndarray, width: int, height: int) -> Dict[str, float]:
    """Support occupancy and wall duration over every complete block of one stream."""

    config = StateConfig(width=width, height=height, taus_us=(TAU_US,), radius=SCORE_RADIUS)
    means: List[float] = []
    zeros: List[float] = []
    full: List[float] = []
    spans: List[float] = []
    for start in range(0, len(events) - BLOCK + 1, BLOCK):
        chunk = events[start:start + BLOCK]
        raw = bidirectional_features(chunk, config)
        support = ring_support(raw, len(config.taus_us), config.feature_dim, config.radius)
        means.append(float(support.mean()))
        zeros.append(float((support == 0).mean()))
        full.append(float((support == SUPPORT_MAX).mean()))
        spans.append(float(chunk[-1, 0] - chunk[0, 0]))
    if not means:
        return {}
    return {
        "blocks": len(means),
        "mean_support": float(np.mean(means)),
        "fraction_support_zero": float(np.mean(zeros)),
        "fraction_support_saturated": float(np.mean(full)),
        "block_span_us_median": float(np.median(spans)),
        "block_span_ms_median": float(np.median(spans) / 1000.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--led-root", type=Path, default=LED_ROOT)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase60_occupancy_regime.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    # Read the ED24 anchor first. It is one dictionary lookup, and doing it after the
    # scan would mean a missing key costs the whole run. 9.87 is the annotated-band
    # figure Supplementary S5 quotes; the full-set figure is carried alongside it so
    # the comparison cannot be made against whichever is more convenient.
    ed24_band = json.loads((ARTIFACTS / "phase36_ed24_inrange_ours.json").read_text())
    ed24_full = json.loads((ARTIFACTS / "phase32_ed24_full.json").read_text())
    ed24_support = float(ed24_band["summary"]["mean_support"])
    ed24_support_full = float(ed24_full["summary"]["mean_support"])

    development: Dict[str, Dict] = {}
    for dataset, name in conditions():
        width, height = GEOMETRY[dataset]
        events, _labels = load_condition(dataset, name, args.dnd21_root, args.dvsclean_root)
        record = block_profile(events, width, height)
        record["dataset"] = dataset
        development[f"{dataset}/{name}"] = record
        print(f"dev  {dataset}/{name:22s} S={record['mean_support']:5.2f} "
              f"zero={record['fraction_support_zero']:.3f} "
              f"span={record['block_span_ms_median']:7.2f} ms", flush=True)

    led: Dict[str, Dict] = {}
    scenes = sorted(p.name for p in (args.led_root / "raw_event").iterdir() if p.is_dir())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    for scene in scenes:
        events, _labels, _a, _s, _r = iter_scene(args.led_root, scene)
        if len(events) < BLOCK:
            continue
        record = block_profile(events, LED_WIDTH, LED_HEIGHT)
        led[scene] = record
        print(f"LED  {scene:22s} S={record['mean_support']:5.2f} "
              f"zero={record['fraction_support_zero']:.3f} "
              f"span={record['block_span_ms_median']:7.2f} ms", flush=True)

    def agg(group: Dict[str, Dict], field: str) -> float:
        vals = [r[field] for r in group.values() if field in r]
        return float(np.mean(vals)) if vals else float("nan")

    fields = ("mean_support", "fraction_support_zero", "fraction_support_saturated",
              "block_span_ms_median")
    summary = {
        "development": {f: agg(development, f) for f in fields},
        "led": {f: agg(led, f) for f in fields},
        "ed24_mean_support_annotated_band": ed24_support,
        "ed24_mean_support_full_set": ed24_support_full,
    }
    summary["development"]["conditions"] = len(development)
    summary["led"]["scenes"] = len(led)

    led_s = summary["led"]["mean_support"]
    dev_s = summary["development"]["mean_support"]
    payload = {
        "phase": "60 occupancy regime, and whether S5 predicted the LED gap",
        "question": ("Supplementary S5 states a failure regime at a nearly empty ring and "
                     "asserts no benchmark here samples it. Does LED?"),
        "support_max": SUPPORT_MAX,
        "block_size": BLOCK,
        "summary": summary,
        "verdict": ("LED sits in the near-empty-ring regime S5 predicted; the S5 claim that "
                    "no benchmark samples it is false"
                    if led_s < 0.5 * dev_s and summary["led"]["fraction_support_zero"] > 0.30
                    else "LED does not sit in the near-empty-ring regime; the empty-ring "
                         "account of the LED gap is not supported"),
        "ebf_comparison": {
            "ebf_window_ms": EBF_TAU_US / 1000.0,
            "note": ("EBF's window is fixed in time, so it does not shrink as the event "
                     "rate rises; our block-local window is fixed in events and does."),
            "led_block_span_ms_median": summary["led"]["block_span_ms_median"],
            "development_block_span_ms_median": summary["development"]["block_span_ms_median"],
        },
        "development": development,
        "led": led,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\nmean support of {SUPPORT_MAX}: development {dev_s:.2f}, "
          f"ED24 {summary['ed24_mean_support_annotated_band']:.2f}, LED {led_s:.2f}")
    print(f"fraction of events with S = 0: development "
          f"{summary['development']['fraction_support_zero']:.3f}, "
          f"LED {summary['led']['fraction_support_zero']:.3f}")
    print(f"median block span: development "
          f"{summary['development']['block_span_ms_median']:.2f} ms, "
          f"LED {summary['led']['block_span_ms_median']:.2f} ms "
          f"(EBF's window is {EBF_TAU_US / 1000.0:.0f} ms)")
    print(f"verdict: {payload['verdict']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
