"""Phase 62: BUAA_Campus, and the label leak in four of its six released sequences.

WHY THIS BENCHMARK. BUAA_Campus is EBF's own dataset, released with its paper. Running our
filter on a competitor's home benchmark, against numbers that competitor published, is the
least favourable test we can construct for ourselves, which is what makes it worth running.

WHAT WE FOUND FIRST. The released arrays carry six columns,
`[signal_flag, t_us, x, y, polarity, ?]`, and the polarity column takes values outside
{-1,+1}. It takes them **only on noise events**: signal events are always +-1, while in four
of the six noisy sequences a little over half the noise events carry 2, 3 or 4. The rule
"polarity > 1", which reads one field and does no filtering at all, then separates noise from
signal at AUC 0.80. Collapsing to the sign -- which is what EBF and our own extractor both do,
so neither exploits the extra values as such -- still leaves AUC ~0.64, because the collapse
makes noise 80% ON where signal is 50% ON.

That is a property of the release, not of either filter, and it inflates any AUC measured on
those four sequences. campus1 at 1 and 3 Hz/pixel are clean; campus1 at 5 Hz/pixel and all
three campus2 sequences are not. **The EBF paper's own published Campus numbers are measured
on campus1 at 5 Hz/pixel**, one of the affected files.

THE REPAIR is label-free: every polarity outside {-1,+1} is redrawn uniformly from {-1,+1}.
It cannot touch a signal event, because signal events never carry those values, and it
restores the noise stream to 50/50 ON. After it, both leak rules sit at chance. Every arm is
scored twice, once on the released stream and once on the repaired one, and the difference is
the leak's contribution to that arm.

PROTOCOL. The EBF paper uses the first 3 million events of each sequence, and its release
says so; we do the same. It reports event-level AUC over the whole sequence, so we report
that, and our own block-mean AUC beside it so the row can be read either way. Sensor geometry
is 346x260. Our constants are the frozen ones and EBF runs at its published s = 5,
tau = 32 ms; nothing is refitted here for either method.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .core import StateConfig
from .ebf import EBF_S, EBF_TAU_US, check_equivalence, check_properties
from .phase54_led import score_blocks
from .phase57_ebf_labelled import block_mean_auc, ebf_noise_score
from .phase60_occupancy_regime import block_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
BUAA_ROOT = PROJECT_ROOT.parent / "BUAA_Campus"
WIDTH, HEIGHT = 346, 260
BLOCK = 10_240
TAU_US = 250.0
SCORE_RADIUS, GATE_RADIUS = 2, 1
#: The EBF release states its experiments use the first 3 million events of each sequence.
MAX_EVENTS = 3_000_000
SEQUENCES = tuple(f"campus{n}_{hz}Hz" for n in (1, 2) for hz in (1, 3, 5))
#: Values a polarity field is allowed to take. Anything else is corrupt, not a polarity.
VALID_POLARITY = (-1, 1)

#: EBF paper, Table I: EBF (s=5) on Campus at 5 Hz/pixel across tau. Transcribed, for
#: context only -- it is measured on campus1_5Hz, which this phase shows is contaminated.
EBF_PAPER_CAMPUS_TABLE1 = {"16ms": 0.878, "32ms": 0.880, "64ms": 0.865, "128ms": 0.836}
#: EBF paper, Fig. 3(a) legend: four filters on Campus at 5 Hz/pixel. Figure values, not
#: a table, and on the same contaminated file.
EBF_PAPER_CAMPUS_FIG3 = {"BAF": 0.753, "STCF": 0.803, "EDformer": 0.828, "EBF": 0.880}


def load_sequence(path: Path, repair: bool, seed: int
                  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """First `MAX_EVENTS` events as `[t,x,y,p01]` with `label 1 = noise`, plus diagnostics.

    The released `signal_flag` is 1 for SIGNAL, the opposite of every other benchmark in
    this paper, so it is inverted here once and the inversion is asserted by the noise
    fraction the caller records.
    """

    raw = np.asarray(np.load(path, mmap_mode="r")[:MAX_EVENTS])
    labels = (raw[:, 0] == 0).astype(np.int64)          # 1 = noise
    polarity = raw[:, 4].astype(np.int64).copy()
    corrupt = ~np.isin(polarity, VALID_POLARITY)

    diagnostics = {
        "events": int(len(raw)),
        "noise_fraction": float(labels.mean()),
        "corrupt_polarity_fraction": float(corrupt.mean()),
        "corrupt_on_signal": int(np.count_nonzero(corrupt & (labels == 0))),
        "corrupt_on_noise": int(np.count_nonzero(corrupt & (labels == 1))),
        "auc_polarity_gt_1": float(roc_auc_score(labels, (polarity > 1).astype(float))),
    }
    if repair and corrupt.any():
        rng = np.random.default_rng(seed)
        polarity[corrupt] = rng.choice(VALID_POLARITY, size=int(corrupt.sum()))
    diagnostics["auc_polarity_sign"] = float(
        roc_auc_score(labels, (polarity > 0).astype(float)))
    diagnostics["noise_on_fraction"] = float((polarity[labels == 1] > 0).mean())
    diagnostics["signal_on_fraction"] = float((polarity[labels == 0] > 0).mean())

    events = np.ascontiguousarray(
        np.column_stack((raw[:, 1].astype(np.int64), raw[:, 2].astype(np.int64),
                         raw[:, 3].astype(np.int64), (polarity > 0).astype(np.int64))),
        dtype=np.int64)
    if np.any(np.diff(events[:, 0]) < 0):
        raise SystemExit(f"{path.name}: timestamps are not nondecreasing")
    return events, labels, diagnostics


def score_all(events: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Every arm on one stream: three of ours and EBF, on identical blocks."""

    gate = StateConfig(width=WIDTH, height=HEIGHT, taus_us=(TAU_US,), radius=GATE_RADIUS)
    score = StateConfig(width=WIDTH, height=HEIGHT, taus_us=(TAU_US,), radius=SCORE_RADIUS)
    ours, betas, _rhos = score_blocks(events, gate, score)
    scored = ~np.isnan(ours["support"])
    y = labels[scored]

    out: Dict[str, Dict[str, float]] = {}
    for arm in ("support", "magnitude", "beta0_support", "beta0_magnitude"):
        s = ours[arm][scored]
        out[arm] = {"event_auc": float(roc_auc_score(y, s)),
                    "block_mean_auc": block_mean_auc(s, y)}
    ebf = ebf_noise_score(events, WIDTH, HEIGHT, float(EBF_TAU_US))[scored]
    out["ebf"] = {"event_auc": float(roc_auc_score(y, ebf)),
                  "block_mean_auc": block_mean_auc(ebf, y)}
    out["_meta"] = {"blocks": int(len(betas)), "gate_open_rate": float(betas.mean()),
                    "scored_events": int(scored.sum())}
    # Section IV-E attributes our LED deficit to ring occupancy, which a fixed-event-count
    # block sets. BUAA_Campus is a fifth benchmark and therefore a fifth chance for that
    # account to be wrong, so the occupancy it delivers is measured here rather than
    # inferred from the AUC it produces.
    out["_occupancy"] = block_profile(events, WIDTH, HEIGHT)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BUAA_ROOT)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase62_buaa_campus.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    check_properties()
    worst, _ = check_equivalence()
    print(f"EBF kernel verified against the released code, max |diff| = {worst:.2e}, "
          f"and against four properties the released code cannot vouch for")

    results: Dict[str, Dict] = {}
    for name in args.sequences:
        path = args.root / f"{name}.npy"
        if not path.is_file():
            raise SystemExit(f"missing sequence: {path}")
        record: Dict = {}
        for treatment, repair in (("released", False), ("repaired", True)):
            events, labels, diag = load_sequence(path, repair, args.seed)
            record[treatment] = {**score_all(events, labels), "diagnostics": diag}
        record["leak"] = {
            "corrupt_polarity_fraction":
                record["released"]["diagnostics"]["corrupt_polarity_fraction"],
            "corrupt_on_signal": record["released"]["diagnostics"]["corrupt_on_signal"],
            "auc_polarity_gt_1": record["released"]["diagnostics"]["auc_polarity_gt_1"],
            "auc_polarity_sign_released":
                record["released"]["diagnostics"]["auc_polarity_sign"],
            "auc_polarity_sign_repaired":
                record["repaired"]["diagnostics"]["auc_polarity_sign"],
            "contaminated":
                record["released"]["diagnostics"]["auc_polarity_gt_1"] > 0.55,
        }
        for arm in ("support", "beta0_support", "magnitude", "beta0_magnitude", "ebf"):
            record.setdefault("leak_cost", {})[arm] = {
                k: record["released"][arm][k] - record["repaired"][arm][k]
                for k in ("event_auc", "block_mean_auc")}
        results[name] = record
        d = record["released"]["diagnostics"]
        print(f"{name:14s} noise={d['noise_fraction']:.4f} corrupt={d['corrupt_polarity_fraction']:.3f} "
              f"leakAUC={d['auc_polarity_gt_1']:.4f} | released ours={record['released']['support']['event_auc']:.4f} "
              f"EBF={record['released']['ebf']['event_auc']:.4f} | repaired "
              f"ours={record['repaired']['support']['event_auc']:.4f} "
              f"EBF={record['repaired']['ebf']['event_auc']:.4f}", flush=True)

    clean = [n for n, r in results.items() if not r["leak"]["contaminated"]]
    dirty = [n for n, r in results.items() if r["leak"]["contaminated"]]

    def mean_over(names: List[str], treatment: str, arm: str, field: str) -> float:
        vals = [results[n][treatment][arm][field] for n in names]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "clean_sequences": clean,
        "contaminated_sequences": dirty,
        **{f"{t}_{arm}_{f}": {"clean": mean_over(clean, t, arm, f),
                              "contaminated": mean_over(dirty, t, arm, f),
                              "all": mean_over(list(results), t, arm, f)}
           for t in ("released", "repaired")
           for arm in ("support", "beta0_support", "magnitude", "beta0_magnitude", "ebf")
           for f in ("event_auc", "block_mean_auc")},
    }

    payload = {
        "phase": "62 BUAA_Campus, EBF's own dataset, and its label leak",
        "dataset": {
            "source": "BUAA_Campus, released with EBF (Guo et al., IEEE TCSVT 2025)",
            "geometry": [WIDTH, HEIGHT],
            "events_per_sequence": MAX_EVENTS,
            "protocol_note": "first 3M events, as the release states its own experiments use",
            "label_convention": "released column 0 is 1 for SIGNAL; inverted here so 1 = noise",
        },
        "leak": {
            "what": ("the polarity column takes values outside {-1,+1}, and only on noise "
                     "events; 'polarity > 1' then separates the classes with no filtering"),
            "repair": ("label-free: polarities outside {-1,+1} are redrawn uniformly from "
                       "{-1,+1}; no signal event is touched, since signal never carries them"),
            "affected_sequences": dirty,
            "clean_sequences": clean,
        },
        "ebf_paper_transcribed": {
            "table1_campus_tau_sweep_5hz": EBF_PAPER_CAMPUS_TABLE1,
            "fig3a_four_filters_campus_5hz": EBF_PAPER_CAMPUS_FIG3,
            "note": ("both are measured on campus1 at 5 Hz/pixel, which this phase finds "
                     "contaminated; Fig. 3 values are figure-legend values, not a table"),
        },
        "ours": {"tau_us": TAU_US, "block": BLOCK, "score_radius": SCORE_RADIUS,
                 "gate_radius": GATE_RADIUS, "refitted_here": False},
        "ebf": {"s": EBF_S, "tau_us": EBF_TAU_US, "refitted_here": False,
                "kernel_vs_released_max_abs_diff": worst},
        "summary": summary,
        "per_sequence": results,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\nclean sequences ({len(clean)}): {', '.join(clean)}")
    print(f"contaminated   ({len(dirty)}): {', '.join(dirty)}")
    for t in ("released", "repaired"):
        print(f"{t:9s} event-level AUC, clean subset:  "
              f"ours {summary[f'{t}_support_event_auc']['clean']:.4f}  "
              f"beta0 {summary[f'{t}_beta0_support_event_auc']['clean']:.4f}  "
              f"EBF {summary[f'{t}_ebf_event_auc']['clean']:.4f}")
        print(f"{t:9s} event-level AUC, contaminated:  "
              f"ours {summary[f'{t}_support_event_auc']['contaminated']:.4f}  "
              f"beta0 {summary[f'{t}_beta0_support_event_auc']['contaminated']:.4f}  "
              f"EBF {summary[f'{t}_ebf_event_auc']['contaminated']:.4f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
