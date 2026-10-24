"""Phase 65: BAF, STCF and EDformer on the repaired BUAA_Campus, measured rather than cited.

WHAT THIS REPLACES. Table VII's lower panel carried three baseline values transcribed from
\\cite{guo2025ebf} -- and from its Fig. 3(a) legend rather than a table, because its only
Campus table is an EBF-on-Campus tau ablation. Worse, those values are measured on
`campus1_5Hz`, one of the four sequences Phase 62 shows carries a label leak. Cited numbers
on contaminated data are the weakest evidence in this paper. This phase runs all three
ourselves, on the repaired stream, on the same events and the same aggregation as every
other row.

THE VALIDATION GATE, AND WHAT IT SAYS ABOUT EACH ROW. Before any Campus number is reported,
each method is run on DND21 driving at 1, 3 and 5 Hz/pixel and compared against the values
\\cite{guo2025ebf} Tab.~II publishes for exactly those conditions. The three outcomes differ
and the paper has to say so:

  EDformer  reproduces to about 1e-4 against its published 0.954/0.947/0.942. Since it shares
            our preprocessing, label convention and AUC call, that also validates the data
            path for the other two.
  BAF/STCF  do NOT reproduce: ours score above the published values. Two reasons, neither of
            which is a defect we can remove. The published BAF row is itself transcribed by
            that source from \\cite{jiang2024edformer} rather than measured, and our own
            Tab.~III caption already records that the two available sources disagree by up to
            0.34 AUC on these same filters. And our score-based ROC is the continuous limit of
            the `dt` sweep those papers perform over a grid, so it upper-bounds any finite
            sweep of the same rule by construction. We therefore report ours as measured rows
            and do not claim they reproduce anything.

Reporting a baseline HIGHER than its published value is the safe direction for a paper that
compares itself against it, and it is the direction our implementations land in.

Every method is truncated to a common event prefix so the rows are exactly paired: EDformer
consumes 4,096-event slices and our own score 10,240-event blocks, so the shorter of the two
governs. Read-only. Nothing is trained; EDformer's released checkpoint is used as released.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig
from .ebf import EBF_TAU_US, check_equivalence, check_properties
from .phase54_led import score_blocks
from .phase57_ebf_labelled import block_mean_auc, ebf_noise_score, load_condition
from .phase62_buaa_campus import (BUAA_ROOT, HEIGHT, SEQUENCES, TAU_US, WIDTH,
                                  load_sequence)
from .rule_filters import baf_noise_score, stcf_noise_scores
from .rule_filters import check_equivalence as rule_equivalence
from .rule_filters import check_properties as rule_properties

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
EDFORMER_PYTHON = PROJECT_ROOT.parent / ".venv-edformer-cu118/bin/python"
EDFORMER_RUNNER = PROJECT_ROOT / "tools/run_edformer.py"
EDFORMER_SLICE = 4096
BLOCK = 10_240
#: Per-unit results are cached here so an interrupted run resumes instead of restarting.
#: Each unit costs several minutes, almost all of it EDformer, and losing a completed one
#: to an unrelated interruption is pure waste.
CACHE = ARTIFACTS / "phase65_cache"

#: \cite{guo2025ebf} Tab. II, DND21 driving. The gate every method is held to.
PUBLISHED_DRIVING = {
    "BAF": {"1hz": 0.848, "3hz": 0.816, "5hz": 0.793},
    "STCF": {"1hz": 0.912, "3hz": 0.887, "5hz": 0.860},
    "EDformer": {"1hz": 0.954, "3hz": 0.947, "5hz": 0.942},
}
#: How close a re-run must be to count as reproducing the published value.
REPRODUCES_WITHIN = 0.01


def edformer_cache_path(key: str) -> Path:
    return CACHE / f"edformer_{key}.npy"


def edformer_scores(events: np.ndarray, width: int, height: int,
                    key: Optional[str] = None) -> np.ndarray:
    """Run the released checkpoint out of process, under its own pinned interpreter.

    When `key` names a cached score file, that file is used instead. EDformer is by far the
    most expensive step here and it is deterministic given the stream, so it is computed
    once per unit by `tools/precompute_edformer.py` and read back rather than re-run.
    """

    if key is not None and edformer_cache_path(key).is_file():
        return np.load(edformer_cache_path(key))
    if not EDFORMER_PYTHON.is_file():
        raise SystemExit(f"missing EDformer interpreter: {EDFORMER_PYTHON}")
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npy"
        np.savez(src, events=events, width=width, height=height)
        result = subprocess.run(
            [str(EDFORMER_PYTHON), str(EDFORMER_RUNNER),
             "--input", str(src), "--output", str(dst)],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"EDformer runner failed:\n{result.stdout}\n{result.stderr}")
        return np.load(dst)


def score_every_method(events: np.ndarray, labels: np.ndarray, width: int, height: int,
                       with_ours: bool, key: Optional[str] = None
                       ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict]:
    """All baseline scores plus, optionally, our own arms, truncated to a common prefix.

    Every score is oriented so that larger means more noise-like, which is the orientation
    `roc_auc_score(label, score)` needs given that label 1 is noise.
    """

    scores: Dict[str, np.ndarray] = {}
    meta: Dict = {}

    scores["BAF"] = baf_noise_score(events, width, height)
    stcf = stcf_noise_scores(events, width, height)
    # STCF has two knobs and the source sweeps both; k is chosen by the sweep and recorded.
    stcf_auc = {k: float(roc_auc_score(labels, v)) for k, v in stcf.items()}
    best_k = max(stcf_auc, key=stcf_auc.get)
    scores["STCF"] = stcf[best_k]
    meta["stcf_best_k"] = int(best_k)
    meta["stcf_auc_by_k"] = {str(k): v for k, v in stcf_auc.items()}

    scores["EDformer"] = edformer_scores(events, width, height, key=key)

    if with_ours:
        gate = StateConfig(width=width, height=height, taus_us=(TAU_US,), radius=1)
        score = StateConfig(width=width, height=height, taus_us=(TAU_US,), radius=2)
        ours, _betas, _rhos = score_blocks(events, gate, score)
        valid = ~np.isnan(ours["support"])
        limit = int(np.count_nonzero(valid))
        scores["RSF"] = ours["support"][:limit]
        scores["RSF_magnitude"] = ours["magnitude"][:limit]
        scores["EBF"] = ebf_noise_score(events, width, height, float(EBF_TAU_US))
    else:
        scores["EBF"] = ebf_noise_score(events, width, height, float(EBF_TAU_US))

    common = min(len(v) for v in scores.values())
    meta["events_scored"] = int(common)
    meta["events_available"] = int(len(events))
    return {k: v[:common] for k, v in scores.items()}, labels[:common], meta


def cached(key: str, compute) -> Dict:
    """Compute `key` once and keep it; later calls read the stored result."""

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    if path.is_file():
        print(f"  [cached] {key}", flush=True)
        return json.loads(path.read_text())
    value = compute()
    path.write_text(json.dumps(value, indent=2, sort_keys=True))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--campus-root", type=Path, default=BUAA_ROOT)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase65_campus_baselines.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    rule_properties()
    rule_worst = rule_equivalence()
    check_properties()
    ebf_worst, _ = check_equivalence()
    print(f"BAF == STCF(k=1) exactly ({rule_worst}); EBF kernel matches the released "
          f"code to {ebf_worst:.2e}")

    validation: Dict[str, Dict] = {}
    if not args.skip_validation:
        for rate in ("1hz", "3hz", "5hz"):
            def run(rate=rate) -> Dict:
                events, labels = load_condition("DND21", f"{rate}_driving",
                                                args.dnd21_root, args.dvsclean_root)
                scores, y, meta = score_every_method(events, labels, 346, 260,
                                                     with_ours=False,
                                                     key=f"validation_{rate}")
                row: Dict = {}
                for method in ("BAF", "STCF", "EDformer"):
                    got = float(roc_auc_score(y, scores[method]))
                    want = PUBLISHED_DRIVING[method][rate]
                    row[method] = {"ours": got, "published": want, "delta": got - want,
                                   "reproduces": bool(abs(got - want) <= REPRODUCES_WITHIN)}
                row["meta"] = meta
                return row
            row = cached(f"validation_{rate}", run)
            validation[rate] = row
            print(f"validation {rate}: " + "  ".join(
                f"{m} {row[m]['ours']:.4f} vs {row[m]['published']:.3f} "
                f"({row[m]['delta']:+.4f})" for m in ("BAF", "STCF", "EDformer")), flush=True)

    per_sequence: Dict[str, Dict] = {}
    for name in args.sequences:
        def run(name=name) -> Dict:
            events, labels, diag = load_sequence(args.campus_root / f"{name}.npy",
                                                 True, args.seed)
            scores, y, meta = score_every_method(events, labels, WIDTH, HEIGHT,
                                                 with_ours=True, key=f"campus_{name}")
            record: Dict = {"noise_fraction": diag["noise_fraction"], **meta}
            for method, value in scores.items():
                record[method] = {"event_auc": float(roc_auc_score(y, value)),
                                  "block_mean_auc": block_mean_auc(value, y)}
            return record
        record = cached(f"campus_{name}", run)
        per_sequence[name] = record
        print(f"{name:14s} " + "  ".join(
            f"{m} {record[m]['event_auc']:.4f}" for m in
            ("BAF", "STCF", "EDformer", "RSF", "RSF_magnitude", "EBF")), flush=True)

    methods = ("BAF", "STCF", "EDformer", "RSF", "RSF_magnitude", "EBF")
    summary = {m: {f: float(np.mean([r[m][f] for r in per_sequence.values()]))
                   for f in ("event_auc", "block_mean_auc")} for m in methods}
    clean = ("campus1_1Hz", "campus1_3Hz")
    summary_clean = {m: float(np.mean([per_sequence[c][m]["event_auc"]
                                       for c in clean if c in per_sequence]))
                     for m in methods}

    payload = {
        "phase": "65 BAF, STCF and EDformer measured on the repaired BUAA_Campus",
        "why": ("Table VII's baseline panel was transcribed from a figure legend and "
                "measured on a contaminated sequence; these rows replace it"),
        "stream": "repaired: polarities outside {-1,+1} redrawn, no signal event touched",
        "protocol": {"events_per_sequence": "first 3M, as the release states",
                     "geometry": [WIDTH, HEIGHT],
                     "edformer_slice": EDFORMER_SLICE,
                     "our_block": BLOCK,
                     "alignment": "all methods truncated to a common event prefix"},
        "implementations": {
            "BAF": "ours, jaer BackgroundActivityFilter; score is the age of the most "
                   "recent neighbour event, whose ROC is the dt sweep at infinite resolution",
            "STCF": "ours, jaer SpatioTemporalCorrelationFilter, polarity-agnostic; score "
                    "is the k-th smallest neighbour age, k chosen by sweep and recorded",
            "EDformer": "the released pretrained_model.pth, run under its pinned "
                        "environment with the released 4,096-event slicing",
            "baf_equals_stcf_k1": rule_worst,
            "ebf_kernel_vs_released": ebf_worst,
        },
        "validation_vs_published": validation,
        "validation_note": ("EDformer reproduces its published DND21 values, which also "
                            "validates the shared preprocessing and AUC path; BAF and STCF "
                            "score above their published values and are reported as "
                            "measured rows, not as reproductions"),
        "summary_all_six": summary,
        "summary_clean_pair": summary_clean,
        "per_sequence": per_sequence,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("\nBUAA_Campus, event-level AUC, mean over six repaired sequences:")
    for m in methods:
        print(f"  {m:14s} {summary[m]['event_auc']:.4f}   (clean pair {summary_clean[m]:.4f})")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
