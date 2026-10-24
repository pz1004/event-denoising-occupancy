"""Phase 33: EDformer on ED24, scored on the blocks Phase 32 scored.

Phase 32 made ED24 a third labelled benchmark but left Table 9 without a competitor: the only
comparison in it is ours against our own ring-magnitude ablation. This phase supplies the learned
row, so the section reports a benchmark rather than an internal ablation.

The comparison is not neutral, and the direction of the bias is knowable rather than guessed.
`EDformer/dataset.py` selects training files by suffix,

    voltages = ['1.5', '1.6', ..., '3.5']

and `EDformer/train.py` wraps that set in a single DataLoader with no train/val split, so the
released checkpoint was fitted on every ED24 scene at suffixes 1.5-3.5. Of the six levels this
paper evaluates, exactly one -- 2.0 -- lies inside that range. The remaining five (0.1, 0.3, 0.5,
1.0, 4.0) are noise levels the checkpoint never saw, on scenes it did. So level 2.0 is a
memorisation control and the other five measure extrapolation across noise level within seen
scenes. `training_overlap` records this per level rather than leaving it to the reader.

Our own side is not clean either: Section 3.4 checked theta = 0 on these same six levels. No
constant was changed as a result, but ED24 is not held out for either method, and neither row here
should be read as a generalisation result.

Scoring follows EDformer's released `eval_auc.py` exactly -- 4,096-event slices, timestamps
normalised over the whole file, sigmoid of the logit as the noise score -- and then aggregates the
per-event scores three ways: a block mean over 10,240-event blocks, which is the statistic Phase 32
and Table 1 use and therefore the only one comparable to our row; a block mean over EDformer's
native 4,096; and one event-level AUC per condition. Reporting all three keeps the statistic
choice of Section 4.3 visible here too.

Read-only. Writes a new artifact and touches no existing one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
ED24_ROOT = PROJECT_ROOT / "ECCV2024_datasets/ED24"
EDFORMER_ROOT = PROJECT_ROOT / "EDformer"

#: The six levels Phase 4, Phase 22b and Phase 32 evaluate, kept frozen so the rows align.
ED24_LEVELS = ["0.1", "0.3", "0.5", "1.0", "2.0", "4.0"]

#: EDformer/dataset.py::ED24.voltages -- the suffixes the released checkpoint was trained on.
EDFORMER_TRAIN_SUFFIXES = [f"{v / 10:.1f}" for v in range(15, 36)]

SEQ_LEN = 4_096          # EDformer's released block size
OUR_BLOCK = 10_240       # the block this paper reports AUC over

#: EDformer Tab. 2, transcribed in tables/make_tables.py::DND21_PUBLISHED. Used by --anchor.
DND21_ANCHOR = {
    "1hz/mix_result.txt": 0.9928,   "1hz/driving_mix_result.txt": 0.9542,
    "3hz/mix_result.txt": 0.9891,   "3hz/driving_mix_result.txt": 0.9472,
    "5hz/mix_result.txt": 0.9845,   "5hz/driving_mix_result.txt": 0.9424,
    "7hz/mix_result.txt": 0.9793,   "7hz/driving_mix_result.txt": 0.9344,
    "10hz/mix_result.txt": 0.9699,  "10hz/driving_mix_result.txt": 0.9264,
}


def load_model(checkpoint: Path) -> torch.nn.Module:
    sys.path.insert(0, str(EDFORMER_ROOT))
    from model import EDformer  # noqa: E402  -- import needs the path above

    model = EDformer().cuda()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda:0"))
    model.eval()
    return model


def score_file(model: torch.nn.Module, path: Path) -> Optional[Dict]:
    """Per-event noise scores from EDformer, plus labels, for one labelled stream.

    Timestamps are normalised over the whole file and events are fed in 4,096-event slices,
    which is what `EDformer/eval_auc.py::Inference` does. Deviating from it would make the row
    ours rather than theirs.
    """

    events = pd.read_csv(path, skiprows=1, delimiter=" ", header=None).values
    slices = len(events) // SEQ_LEN
    if slices == 0:
        return None

    stamps = events[:, 0].astype(np.float64)
    span = stamps.max() - stamps.min()
    normalised = (stamps - stamps.min()) / span if span > 0 else np.zeros_like(stamps)
    stream = np.hstack([normalised.reshape(-1, 1), events[:, 1:4]])
    stream = stream[:slices * SEQ_LEN].reshape(slices, SEQ_LEN, 4)

    scores: List[np.ndarray] = []
    with torch.no_grad():
        for index in range(slices):
            batch = torch.tensor(stream[index]).reshape(1, SEQ_LEN, 4).to(torch.float32).cuda()
            scores.append(torch.sigmoid(model(batch)).cpu().numpy().ravel())

    return {"scores": np.concatenate(scores),
            "labels": events[:slices * SEQ_LEN, 4].astype(int)}


def block_mean_auc(scores: np.ndarray, labels: np.ndarray, block: int) -> Optional[float]:
    values = []
    for start in range(0, len(scores) - block + 1, block):
        chunk = labels[start:start + block]
        if chunk.min() == chunk.max():
            continue
        values.append(roc_auc_score(chunk, scores[start:start + block]))
    return float(np.mean(values)) if values else None


def event_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


LEVELS_USED: List[str] = list(ED24_LEVELS)


def summarise(conditions: Dict[str, Dict]) -> Dict:
    def pooled(key: str) -> Optional[float]:
        values = [c[key] for c in conditions.values() if c.get(key) is not None]
        return float(np.mean(values)) if values else None

    per_level: Dict[str, Dict] = {}
    for level in LEVELS_USED:
        subset = {k: v for k, v in conditions.items() if k.endswith(f"_{level}")}
        if not subset:
            continue
        per_level[level] = {
            "conditions": len(subset),
            "block_mean_10240": float(np.mean([c["block_mean_10240"] for c in subset.values()
                                               if c["block_mean_10240"] is not None])),
            "block_mean_4096": float(np.mean([c["block_mean_4096"] for c in subset.values()
                                              if c["block_mean_4096"] is not None])),
            "event_level": float(np.mean([c["event_level"] for c in subset.values()
                                          if c["event_level"] is not None])),
            "in_edformer_training": level in EDFORMER_TRAIN_SUFFIXES,
        }
    return {
        "conditions": len(conditions),
        "block_mean_10240": pooled("block_mean_10240"),
        "block_mean_4096": pooled("block_mean_4096"),
        "event_level": pooled("event_level"),
        "per_level": per_level,
    }


def run_anchor(model: torch.nn.Module) -> Dict:
    """Score DND21 through this exact path and compare against EDformer's published row.

    If the numbers below do not land near Tab. 2, the ED24 rows are measuring our plumbing
    rather than their model, and nothing downstream should be believed.
    """

    out = {}
    for name, published in DND21_ANCHOR.items():
        scored = score_file(model, PROJECT_ROOT / "ECCV2024_datasets/AUC_test" / name)
        if scored is None:
            continue
        out[name] = {
            "published": published,
            "block_mean_4096": block_mean_auc(scored["scores"], scored["labels"], SEQ_LEN),
            "block_mean_10240": block_mean_auc(scored["scores"], scored["labels"], OUR_BLOCK),
            "event_level": event_auc(scored["scores"], scored["labels"]),
        }
        print(f"  anchor {name}: published {published:.4f} -> "
              f"{out[name]['block_mean_4096']:.4f} / {out[name]['block_mean_10240']:.4f} / "
              f"{out[name]['event_level']:.4f}", flush=True)
    return out


def anchor_verdict(anchor: Dict) -> Dict:
    """Which aggregation reproduces EDformer's published DND21 row?

    Section 4.3 currently declines to sign the DND21 margin because the source paper does not
    state its statistic. This settles it by elimination: whichever column matches the published
    value across all ten conditions is the one Tab. 2 reports, and our row must be compared on
    that column rather than on the block mean the rest of this paper uses.
    """

    columns = ("event_level", "block_mean_4096", "block_mean_10240")
    out = {}
    for column in columns:
        errors = [abs(v[column] - v["published"]) for v in anchor.values()
                  if v.get(column) is not None]
        out[column] = {"max_abs_error": max(errors), "mean_abs_error": sum(errors) / len(errors),
                       "conditions": len(errors)}
    best = min(columns, key=lambda c: out[c]["max_abs_error"])
    out["identified_statistic"] = best
    out["decisive"] = out[best]["max_abs_error"] < 0.001 and all(
        out[c]["max_abs_error"] > 0.002 for c in columns if c != best)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=EDFORMER_ROOT / "pretrained_model.pth")
    parser.add_argument("--scenes", type=int, default=0,
                        help="limit scene count for a smoke run; 0 = all 100")
    parser.add_argument("--anchor", action="store_true",
                        help="also score DND21 and compare against EDformer's published row")
    parser.add_argument("--anchor-only", action="store_true",
                        help="score DND21 and stop; identifies which statistic Tab. 2 reports")
    parser.add_argument("--levels", nargs="+", default=None,
                        help="ED24 attenuator voltages to score. ED24 defines 21 noise "
                             "levels, 1.5-3.5V; the paper excludes below 1.5V (BA noise "
                             "too sparse) and above 3.5V (nearly uniform).")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase33_ed24_baselines.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    model = load_model(args.checkpoint)

    payload: Dict = {
        "phase": "33 EDformer on ED24, aggregated onto Phase 32's blocks",
        "model": "EDformer, released pretrained_model.pth",
        "protocol": (f"eval_auc.py inference: {SEQ_LEN}-event slices, whole-file timestamp "
                     f"normalisation, sigmoid(logit) as the noise score"),
        "levels": args.levels or ED24_LEVELS,
        "training_overlap": {
            "edformer_train_suffixes": [EDFORMER_TRAIN_SUFFIXES[0], EDFORMER_TRAIN_SUFFIXES[-1]],
            "source": "EDformer/dataset.py::ED24.voltages; train.py has no train/val split",
            "levels_in_training": [l for l in (args.levels or ED24_LEVELS)
                                   if l in EDFORMER_TRAIN_SUFFIXES],
            "levels_outside_training": [l for l in (args.levels or ED24_LEVELS)
                                        if l not in EDFORMER_TRAIN_SUFFIXES],
            "note": ("the checkpoint saw every scene at suffixes 1.5-3.5, so level 2.0 is a "
                     "memorisation control and the other five measure noise-level "
                     "extrapolation on seen scenes"),
        },
    }

    if args.anchor or args.anchor_only:
        print("Anchoring on DND21 (published / 4096 / 10240 / event):", flush=True)
        payload["anchor_dnd21"] = run_anchor(model)
        payload["anchor_verdict"] = anchor_verdict(payload["anchor_dnd21"])
        print("\nverdict:", json.dumps(payload["anchor_verdict"], indent=2), flush=True)

    if args.anchor_only:
        payload["wall_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.output} in {payload['wall_seconds']:.0f}s")
        return

    levels = args.levels or ED24_LEVELS
    globals()["LEVELS_USED"] = levels
    payload["levels"] = levels
    scenes = sorted(p.name for p in ED24_ROOT.iterdir() if p.is_dir())
    if args.scenes:
        scenes = scenes[:args.scenes]
    payload["scenes"] = scenes

    conditions: Dict[str, Dict] = {}
    for index, scene in enumerate(scenes, 1):
        for level in levels:
            path = ED24_ROOT / scene / f"{scene}_{level}.csv"
            if not path.exists():
                continue
            scored = score_file(model, path)
            if scored is None:
                continue
            conditions[f"{scene}_{level}"] = {
                "block_mean_10240": block_mean_auc(scored["scores"], scored["labels"], OUR_BLOCK),
                "block_mean_4096": block_mean_auc(scored["scores"], scored["labels"], SEQ_LEN),
                "event_level": event_auc(scored["scores"], scored["labels"]),
                "events": int(len(scored["scores"])),
            }
        print(f"[{index}/{len(scenes)}] {scene}: done "
              f"({time.perf_counter() - started:.0f}s elapsed)", flush=True)

    payload["conditions"] = conditions
    payload["summary"] = summarise(conditions)
    payload["wall_seconds"] = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {payload['wall_seconds']:.0f}s")
    print(json.dumps(payload["summary"]["per_level"], indent=2))


if __name__ == "__main__":
    main()
