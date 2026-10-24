"""Phase 59: EBF on LED -- labelled recorded noise, the benchmark neither method was built on.

WHY HERE. LED is the only benchmark in this paper that is labelled, recorded, and untouched
by any selection we made. It is also where our own filter is weakest: block-mean AUC falls
from 0.9811 on the development sets to 0.7723 here. A drop that large invites exactly one
question -- is recorded noise hard for *event denoising*, or hard for *this* filter? Phase 56
excluded density and resolution as explanations, but a control built from our own arms cannot
answer a question about the method itself. An independent published filter can.

EBF is the right one to ask. It is rule-based like ours, it was published in this venue, and
its constants were selected on other data (Driving and Campus), so LED is held out for EBF in
the same sense it is held out for us. If EBF also collapses on LED, the difficulty is the
benchmark's. If EBF holds up, the difficulty is ours, and Section 4.5's framing is wrong.

BOTH FILTERS ARE SCORED IN ONE PASS over each scene, so the blocks are exactly paired: same
events, same labels, same partition, same single-class skip. That is what makes a per-block
paired difference and a scene-clustered interval legitimate. Our own arms are additionally
required to reproduce Phase 54's published per-scene block-mean AUC to six decimals before
anything is written; if they do not, the pairing is not against the published table and the
run refuses to produce a number.

OPERATING POINTS. EBF gets the same two as we do, translated where translation is possible.
The frozen r = 0.28 is a pure retention rule and applies unchanged. The label-free adaptive
rule is Otsu on the score histogram; ours routes through the gate sign to pick Otsu or the
top-atom statistic, and EBF has no gate, so EBF gets Otsu directly. The count-matched
reference reads labels and is a reference for both, not a policy for either.

EBF runs at its published s = 5, tau = 32 ms. Nothing is refitted on LED for either method.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import StateConfig
from .ebf import EBF_S, EBF_TAU_US, check_equivalence, check_properties
from .phase54_led import (
    ARTIFACTS,
    BLOCK,
    GATE_RADIUS,
    HEIGHT,
    LED_ROOT,
    RETENTION,
    SCORE_RADIUS,
    TAU_US,
    WIDTH,
    _rates,
    cluster_bootstrap,
    iter_scene,
    score_blocks,
    sign_flip,
)
from .phase57_ebf_labelled import ebf_noise_score
from .retention import otsu_retention

#: Our arms, carried so the EBF rows sit beside a re-derived -- not transcribed -- pairing.
OURS = ("support", "beta0_support")
#: Phase 54's published per-scene values our arms must reproduce.
OURS_ARTEFACT = "phase54_led.json"
ANCHOR_TOLERANCE = 1e-6


def ebf_operating_point(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """The frozen, the adaptive and the count-matched retention, for a gate-less filter.

    `scores` is EBF's score in this paper's orientation (larger means noise-like), so the
    same "keep the lowest" convention applies to both methods without further translation.
    """

    keep = np.zeros(len(scores), dtype=bool)
    keep_adaptive = np.zeros(len(scores), dtype=bool)
    keep_oracle = np.zeros(len(scores), dtype=bool)
    for start in range(0, len(scores) - BLOCK + 1, BLOCK):
        block = scores[start:start + BLOCK]
        order = np.argsort(block, kind="stable")
        keep[start + order[:int(np.floor(RETENTION * BLOCK))]] = True
        # EBF has no gate, so the regime switch our rule keys on does not exist for it.
        # Otsu is the branch our rule takes on a continuous score, and EBF's is continuous.
        share = otsu_retention(block)
        keep_adaptive[start + order[:int(np.floor(share * BLOCK))]] = True
        n_signal = int((labels[start:start + BLOCK] == 0).sum())
        keep_oracle[start + order[:n_signal]] = True
    scored = ~np.isnan(scores)
    labels_s = labels[scored]
    return {**_rates(keep[scored], labels_s, ""),
            **_rates(keep_adaptive[scored], labels_s, "_adaptive"),
            **_rates(keep_oracle[scored], labels_s, "_oracle_r")}


def block_aucs(scores: np.ndarray, labels: np.ndarray) -> List[float]:
    out = []
    for start in range(0, len(scores) - BLOCK + 1, BLOCK):
        y = labels[start:start + BLOCK]
        if y.min() == y.max():
            continue
        out.append(float(roc_auc_score(y, scores[start:start + BLOCK])))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LED_ROOT)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None, help="debug only")
    parser.add_argument("--replicates", type=int, default=200_000)
    parser.add_argument("--flip-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase59_ebf_led.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    check_properties()
    worst, _ = check_equivalence()
    print(f"EBF kernel verified against the released code, max |diff| = {worst:.2e}, "
          f"and against four properties the released code cannot vouch for")

    published = json.loads((ARTIFACTS / OURS_ARTEFACT).read_text())["per_scene"]

    gate_config = StateConfig(width=WIDTH, height=HEIGHT, taus_us=(TAU_US,), radius=GATE_RADIUS)
    score_config = StateConfig(width=WIDTH, height=HEIGHT, taus_us=(TAU_US,), radius=SCORE_RADIUS)

    scenes = sorted(p.name for p in (args.root / "raw_event").iterdir() if p.is_dir())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    per_scene: Dict[str, Dict] = {}
    per_block: Dict[str, Dict[str, List[float]]] = {a: {} for a in ("ebf",) + OURS}
    anchor_worst = 0.0

    for index, scene in enumerate(scenes, start=1):
        step = time.perf_counter()
        events, labels, _amb, _skip, _reord = iter_scene(args.root, scene)
        if args.max_chunks:
            limit = args.max_chunks * BLOCK
            events, labels = events[:limit], labels[:limit]
        if len(events) < BLOCK:
            continue

        ours, betas, rhos = score_blocks(events, gate_config, score_config)
        scored = ~np.isnan(ours["support"])
        y = labels[scored]
        if y.min() == y.max():
            continue

        ebf = ebf_noise_score(events, WIDTH, HEIGHT, float(EBF_TAU_US))[scored]

        record: Dict = {"blocks": int(len(betas)), "events": int(scored.sum()),
                        "noise_fraction": float(y.mean()),
                        "gate_open_rate": float(betas.mean())}

        per_block["ebf"][scene] = block_aucs(ebf, y)
        record["ebf"] = {
            "block_mean_auc": float(np.mean(per_block["ebf"][scene])),
            "event_auc": float(roc_auc_score(y, ebf)),
            "pr_auc_noise": float(average_precision_score(y, ebf)),
            "pr_auc_signal": float(average_precision_score(1 - y, -ebf)),
            **ebf_operating_point(ebf, y),
        }

        for arm in OURS:
            s = ours[arm][scored]
            per_block[arm][scene] = block_aucs(s, y)
            got = float(np.mean(per_block[arm][scene]))
            record[arm] = {"block_mean_auc": got,
                           "event_auc": float(roc_auc_score(y, s))}
            if scene in published:
                want = float(published[scene][arm]["block_mean_auc"])
                drift = abs(got - want)
                anchor_worst = max(anchor_worst, drift)
                if drift > ANCHOR_TOLERANCE:
                    raise SystemExit(
                        f"Anchor failed on {scene}/{arm}: {got:.8f} here against "
                        f"{want:.8f} published in {OURS_ARTEFACT} (drift {drift:.2e}). "
                        f"The EBF rows would not be paired against the published table.")

        per_scene[scene] = record
        print(f"[{index:3d}/{len(scenes)}] {scene:14s} blocks={record['blocks']:4d} "
              f"noise={record['noise_fraction']:.3f} "
              f"EBF={record['ebf']['block_mean_auc']:.4f} "
              f"ours={record['support']['block_mean_auc']:.4f} "
              f"beta0={record['beta0_support']['block_mean_auc']:.4f} "
              f"[{time.perf_counter() - step:.0f}s]", flush=True)

    print(f"anchor: worst per-scene drift against {OURS_ARTEFACT} = {anchor_worst:.2e}")

    def pooled(arm: str, field: str) -> float:
        vals = [r[arm][field] for r in per_scene.values()
                if field in r[arm] and not np.isnan(r[arm][field])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "ebf": {f: pooled("ebf", f) for f in
                ("block_mean_auc", "event_auc", "pr_auc_noise", "pr_auc_signal",
                 "retained_fraction", "signal_recall", "noise_rejection",
                 "balanced_accuracy", "retained_fraction_adaptive", "signal_recall_adaptive",
                 "noise_rejection_adaptive", "balanced_accuracy_adaptive",
                 "balanced_accuracy_oracle_r")},
        **{arm: {f: pooled(arm, f) for f in ("block_mean_auc", "event_auc")} for arm in OURS},
    }

    contrasts: Dict[str, Dict] = {}
    for other in OURS:
        clusters = [(s, list(np.asarray(per_block["ebf"][s]) - np.asarray(per_block[other][s])))
                    for s in per_block["ebf"] if s in per_block[other]]
        scene_level = [(s, [float(np.mean(v))]) for s, v in clusters]
        contrasts[f"ebf_minus_{other}"] = {
            "block_mean": cluster_bootstrap(clusters, args.replicates, rng),
            "scene_level": cluster_bootstrap(scene_level, args.replicates, rng),
            "sign_flip": sign_flip(scene_level, args.flip_draws, rng),
        }

    payload = {
        "phase": "59 EBF on LED",
        "baseline": {
            "method": "EBF, Guo et al., IEEE TCSVT 35(10):9889-9894, 2025",
            "implementation": "ours, verified against a literal transcription of the "
                              "released repository (github.com/shicy17/EBF)",
            "kernel_vs_released_max_abs_diff": worst,
            "s": EBF_S, "tau_us": EBF_TAU_US,
            "operating_point": "the paper's own Sec. V-A choice; nothing refitted on LED",
            "adaptive_rule": "Otsu on the score histogram; EBF has no gate to route on",
            "causality": "strictly causal; ours takes a block-local bidirectional pass",
        },
        "pairing": "both filters scored in one pass per scene: same events, labels, blocks",
        "anchor_worst_drift_vs_published": anchor_worst,
        "anchor_source": OURS_ARTEFACT,
        "scenes": len(per_scene),
        "block_size": BLOCK,
        "retention": RETENTION,
        "summary": summary,
        "contrasts": contrasts,
        "per_scene": per_scene,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\nblock-mean AUC over {len(per_scene)} scenes: "
          f"EBF {summary['ebf']['block_mean_auc']:.4f}  "
          f"ours(gated) {summary['support']['block_mean_auc']:.4f}  "
          f"ours(beta=0) {summary['beta0_support']['block_mean_auc']:.4f}")
    for name, c in contrasts.items():
        b = c["block_mean"]
        print(f"{name}: {b['mean']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] "
              f"ahead {b['clusters_ahead']} behind {b['clusters_behind']} "
              f"of {b['n_clusters']}, sign-flip p={c['sign_flip']['p_two_sided']}")
    print(f"EBF balanced accuracy: r=0.28 {summary['ebf']['balanced_accuracy']:.3f}, "
          f"adaptive {summary['ebf']['balanced_accuracy_adaptive']:.3f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
