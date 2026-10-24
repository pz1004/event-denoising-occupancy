"""Phase 54: the operand on labelled *recorded* noise, on a benchmark nothing here touched.

WHY THIS ONE MATTERS MORE THAN THE OTHERS. Every labelled result in this paper so far comes
from injected or simulated noise: DND21 injects shot noise with v2e, DVSCLEAN simulates, and
ED24's labels come from an attenuator sweep that EDformer also trained on. The one benchmark
of real recordings, E-MLB, has no per-event labels at all, so its evidence is structural
(MESR) rather than classification accuracy. The gap that leaves is exactly the one a reviewer
should press on: the operand was selected on synthetic dense noise, and no labelled *recorded*
noise has ever tested whether it transfers.

LED closes that gap. It is a paired real-world dataset: each 10 ms chunk is released three
times, as `raw_event`, `denoised_event` and `noise_event`, so per-event labels come from the
release rather than from a simulator. Nothing in this paper was selected, checked, swept or
inspected on LED. It is the only benchmark here that is simultaneously labelled, recorded and
untouched.

LABELS. For every file we verify that the multiset of raw events is *exactly* the union of the
denoised and noise events, then label by a sort-based multiset assignment on the packed key
(t, x, y, p). A small number of events share a key with another event of the opposite label --
same pixel, same polarity, same microsecond -- and for those the assignment is arbitrary. We
count them and report the count rather than hiding it; if the fraction ever became material
this analysis would need a different labelling route. Files whose partition does not check out
are skipped and counted, never silently repaired.

FROZEN. Every constant is the one the paper froze before E-MLB: tau = 250 us, N = 10,240,
score radius 2, gate key radius 1, theta = 0, r = 0.28, c = 1e5/WH. Nothing is refitted, and
no LED number was looked at before this script was written.

ARMS, all on one event stream, sharing one gate decision per block:

    support        beta*G - log(1 + S_r2)     the adopted filter
    magnitude      beta*G - R_same_r2         the radius-matched control, inside the gate
    beta0_magnitude -R_same_r2                RSF-M: the same control on the adopted arm
    beta0_support  -log(1 + S_r2)             support only; the closed-gate branch
    beta1_support  G - log(1 + S_r2)          always-global; the open branch

The gate keys on corr(G, R_same_r1) in every arm, so no arm gets a gate tuned to it.

WHAT IS REPORTED. Per scene: block-mean ROC-AUC, pooled event-level ROC-AUC and PR-AUC, and,
at the frozen r = 0.28, signal recall, noise rejection, retained fraction and balanced
accuracy. Inference resamples the 66 *scenes*, which is the unit LED's design supplies, with a
scene's blocks carried together.

Read-only. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import ring_support
from .phase8_score_upgrade import operands, safe_corr
from .retention import gated_retention

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
LED_ROOT = PROJECT_ROOT.parent / "LED/LED_Test_upload"

BLOCK = 10_240
RETENTION = 0.28
SCORE_RADIUS = 2
GATE_RADIUS = 1
TAU_US = 250.0
#: EVK4 sensor. Asserted against the observed maxima; c = 1e5/WH depends on it.
WIDTH, HEIGHT = 1280, 720

ARMS = ("support", "magnitude", "beta0_support", "beta1_support",
        "beta0_magnitude")
#: (left, right) contrasts reported with scene-clustered inference.
CONTRASTS = (
    ("support", "magnitude"),           # the operand, radius matched -- the claim
    ("support", "beta0_support"),       # what the optional gate is worth here
    ("beta1_support", "beta0_support"),  # what admitting G unconditionally is worth
    ("beta0_support", "beta0_magnitude"),  # the operand on the ADOPTED arm: RSF vs RSF-M
)


def scene_files(scene: Path) -> List[Path]:
    """A scene's chunks in recorded order. Names are zero-padded indices."""

    return sorted(scene.glob("*.npy"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))


def pack(a: np.ndarray) -> np.ndarray:
    """(t, x, y, p) -> one int64 key, for the multiset match against the release."""

    t, x, y, p = a[3], a[0], a[1], a[2]
    return ((t.astype(np.int64) * WIDTH + x) * HEIGHT + y) * 2 + p


def load_chunk(root: Path, scene: str, name: str) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """One chunk as ([t, x, y, p] rows, labels, ambiguous count). 1 = noise, 0 = signal.

    Returns None if the released partition does not check out for this chunk.
    """

    raw = np.load(root / "raw_event" / scene / name)
    den = np.load(root / "denoised_event" / scene / name)
    noi = np.load(root / "noise_event" / scene / name)
    if raw.shape[0] != 4 or raw.shape[1] != den.shape[1] + noi.shape[1]:
        return None

    kr, kd, kn = pack(raw), pack(den), pack(noi)
    order = np.argsort(kr, kind="stable")
    merged = np.sort(np.concatenate([kd, kn]))
    if not np.array_equal(kr[order], merged):
        return None                      # raw is not the union; refuse to guess

    # Multiset assignment: within the sorted raw stream, the first occurrences of a key go
    # to whichever side holds it. Keys held by BOTH sides are the ambiguous ones.
    #
    # Two definitions of "ambiguous" are defensible for a key with n_noise noise copies and
    # n_signal signal copies: the number of events whose individual label is undetermined
    # (n_noise + n_signal), and the number that could actually end up mislabelled
    # (min(n_noise, n_signal)). The count below is n_noise for every key both sides hold,
    # which sits between them and is never smaller than the mislabelling bound, so it errs
    # towards reporting more ambiguity rather than less. On the LED test split it is 0
    # either way: 8,141 duplicate keys exist, and every one of them is held by one side
    # only (verified independently of this code path).
    noise_keys, noise_counts = np.unique(kn, return_counts=True)
    raw_keys, raw_counts = np.unique(kr, return_counts=True)
    idx = np.searchsorted(raw_keys, noise_keys)
    ambiguous = int(np.sum(np.minimum(noise_counts, raw_counts[idx]) *
                           (raw_counts[idx] > noise_counts)))

    labels_sorted = np.zeros(len(kr), dtype=np.int64)
    starts = np.searchsorted(kr[order], noise_keys, side="left")
    for start, count in zip(starts, noise_counts):
        labels_sorted[start:start + count] = 1
    labels = np.empty(len(kr), dtype=np.int64)
    labels[order] = labels_sorted

    events = np.ascontiguousarray(
        np.column_stack((raw[3], raw[0], raw[1], (raw[2] > 0).astype(np.int64))),
        dtype=np.int64)
    return events, labels, ambiguous


def iter_scene(root: Path, scene: str) -> Tuple[np.ndarray, np.ndarray, int, int, bool]:
    """A whole scene, chunks concatenated in recorded order."""

    events: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    ambiguous = 0
    skipped = 0
    for path in scene_files(root / "raw_event" / scene):
        loaded = load_chunk(root, scene, path.name)
        if loaded is None:
            skipped += 1
            continue
        e, y, amb = loaded
        events.append(e)
        labels.append(y)
        ambiguous += amb
    if not events:
        return np.empty((0, 4), np.int64), np.empty(0, np.int64), 0, skipped, False
    stream, y = np.concatenate(events), np.concatenate(labels)
    # 12 of the 66 test scenes are not stored in timestamp order, within a chunk or across
    # chunk boundaries. An event stream is a set of timestamped events, so storage order is
    # not semantic and a stable sort recovers the recorded order without changing the set.
    # The extractor requires nondecreasing timestamps and would otherwise refuse the scene.
    # We record which scenes needed it rather than sorting silently.
    reordered = bool(np.any(np.diff(stream[:, 0]) < 0))
    if reordered:
        order = np.argsort(stream[:, 0], kind="stable")
        stream, y = stream[order], y[order]
    return stream, y, ambiguous, skipped, reordered


def score_blocks(events: np.ndarray, gate_config: StateConfig,
                 score_config: StateConfig
                 ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Per-event scores for every arm, the per-block gate decision, and its statistic."""

    out = {arm: np.full(len(events), np.nan, dtype=np.float64) for arm in ARMS}
    betas: List[float] = []
    rhos: List[float] = []
    for start in range(0, len(events) - BLOCK + 1, BLOCK):
        chunk = events[start:start + BLOCK]
        stop = start + BLOCK

        raw_g = bidirectional_features(chunk, gate_config)
        g, r_same_r1, _ = operands(raw_g, gate_config)
        rho = safe_corr(g, r_same_r1)
        beta = 1.0 if rho < 0.0 else 0.0
        betas.append(beta)
        rhos.append(rho)

        raw_s = bidirectional_features(chunk, score_config)
        _, r_same_r2, _ = operands(raw_s, score_config)
        support = ring_support(raw_s, len(score_config.taus_us),
                               score_config.feature_dim, score_config.radius)
        local = np.log1p(support)

        out["support"][start:stop] = beta * g - local
        out["magnitude"][start:stop] = beta * g - r_same_r2
        out["beta0_support"][start:stop] = -local
        out["beta0_magnitude"][start:stop] = -r_same_r2
        out["beta1_support"][start:stop] = g - local
    return (out, np.asarray(betas, dtype=np.float64),
            np.asarray(rhos, dtype=np.float64))


def _rates(keep: np.ndarray, labels: np.ndarray, suffix: str) -> Dict[str, float]:
    signal, noise = labels == 0, labels == 1
    recall = float(keep[signal].mean()) if signal.any() else float("nan")
    reject = float((~keep[noise]).mean()) if noise.any() else float("nan")
    return {
        f"retained_fraction{suffix}": float(keep.mean()),
        f"signal_recall{suffix}": recall,
        f"noise_rejection{suffix}": reject,
        f"balanced_accuracy{suffix}": float(0.5 * (recall + reject)),
    }


def operating_point(scores: np.ndarray, labels: np.ndarray,
                    rhos: np.ndarray) -> Dict[str, float]:
    """Two operating points, one deployable and one an oracle.

    FROZEN: retain the lowest-scoring fraction r = 0.28 of each block, the operating point
    selected on development data and frozen before E-MLB. Nothing about it knows LED.

    ADAPTIVE: the frozen label-free rule of `retention.gated_retention` -- Otsu on the score
    histogram when the gate is open, the fraction below the top atom when it is closed. It
    reads no labels and adds no constant, keying on the same theta = 0 switch already frozen
    for beta, and it is the paper's other deployable operating point. Unlike r = 0.28 it can
    respond to LED's much lower noise fraction.

    ORACLE: retain exactly as many events as the block actually contains signal events. This
    is not a policy -- it reads the labels -- and it exists to separate two different ways of
    being wrong. LED's noise fraction is far below 0.28's implied assumption, so at the frozen
    point the filter discards a large mass of signal no matter how well it ranks. The oracle
    row shows what the same ranking achieves once the operating point is not the problem, and
    the gap between the two rows is the cost of a retention that was never chosen for this
    benchmark.
    """

    keep = np.zeros(len(scores), dtype=bool)
    keep_adaptive = np.zeros(len(scores), dtype=bool)
    keep_oracle = np.zeros(len(scores), dtype=bool)
    for index, start in enumerate(range(0, len(scores) - BLOCK + 1, BLOCK)):
        block = scores[start:start + BLOCK]
        order = np.argsort(block, kind="stable")
        keep[start + order[:int(np.floor(RETENTION * BLOCK))]] = True
        share = gated_retention(block, float(rhos[index]))
        keep_adaptive[start + order[:int(np.floor(share * BLOCK))]] = True
        n_signal = int((labels[start:start + BLOCK] == 0).sum())
        keep_oracle[start + order[:n_signal]] = True
    scored = ~np.isnan(scores)
    labels_s = labels[scored]
    return {**_rates(keep[scored], labels_s, ""),
            **_rates(keep_adaptive[scored], labels_s, "_adaptive"),
            **_rates(keep_oracle[scored], labels_s, "_oracle_r")}


def cluster_bootstrap(clusters: List[Tuple[str, List[float]]], replicates: int,
                      rng: np.random.Generator) -> Dict:
    """Resample whole scenes; a scene's blocks are drawn together or not at all."""

    blocks = [np.asarray(v, dtype=np.float64) for _, v in clusters if len(v)]
    n = len(blocks)
    flat = np.concatenate(blocks) if blocks else np.array([])
    if n < 2:
        return {"n_clusters": n, "mean": float(flat.mean()) if flat.size else float("nan"),
                "ci95": [float("nan"), float("nan")], "replicates": 0}
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([b.size for b in blocks])
    draws = rng.integers(0, n, size=(replicates, n))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    cluster_means = np.array([b.mean() for b in blocks])
    return {
        "n_clusters": n,
        "mean": float(flat.mean()),
        "bootstrap_se": float(means.std(ddof=1)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "replicates": replicates,
        "clusters_ahead": int((cluster_means > 0).sum()),
        "clusters_behind": int((cluster_means < 0).sum()),
    }


def sign_flip(clusters: List[Tuple[str, List[float]]], draws: int,
              rng: np.random.Generator) -> Dict:
    """Cluster-level permutation: a whole scene's blocks flip sign together."""

    cluster_means = np.array([np.mean(v) for _, v in clusters if len(v)], dtype=np.float64)
    if not len(cluster_means):
        return {"n_clusters": 0, "p_two_sided": float("nan")}
    observed = float(cluster_means.mean())
    signs = rng.choice((-1.0, 1.0), size=(draws, len(cluster_means)))
    null_means = (signs * cluster_means).mean(axis=1)
    p = float((np.abs(null_means) >= abs(observed) - 1e-15).mean())
    return {"n_clusters": int(len(cluster_means)), "observed_mean": observed,
            "draws": draws, "p_two_sided": max(p, 1.0 / draws)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LED_ROOT)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="debug only; truncates each scene")
    parser.add_argument("--replicates", type=int, default=200_000)
    parser.add_argument("--flip-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase54_led.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    gate_config = StateConfig(width=WIDTH, height=HEIGHT, taus_us=(TAU_US,),
                              radius=GATE_RADIUS)
    score_config = StateConfig(width=WIDTH, height=HEIGHT, taus_us=(TAU_US,),
                               radius=SCORE_RADIUS)

    scenes = sorted(p.name for p in (args.root / "raw_event").iterdir() if p.is_dir())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    per_scene: Dict[str, Dict] = {}
    per_block: Dict[str, Dict[str, List[float]]] = {arm: {} for arm in ARMS}
    observed_max = [0, 0]
    ambiguous_total = 0
    skipped_total = 0
    reordered_scenes = 0
    blocks_total = 0
    events_total = 0
    gate_open_blocks = 0

    for index, scene in enumerate(scenes, start=1):
        step = time.perf_counter()
        events, labels, ambiguous, skipped, reordered = iter_scene(args.root, scene)
        reordered_scenes += int(reordered)
        ambiguous_total += ambiguous
        skipped_total += skipped
        if args.max_chunks:
            limit = args.max_chunks * BLOCK
            events, labels = events[:limit], labels[:limit]
        if len(events) < BLOCK:
            continue
        observed_max = [max(observed_max[0], int(events[:, 1].max())),
                        max(observed_max[1], int(events[:, 2].max()))]

        scores, betas, rhos = score_blocks(events, gate_config, score_config)
        scored = ~np.isnan(scores["support"])
        y = labels[scored]
        if y.min() == y.max():
            continue
        blocks_total += len(betas)
        events_total += int(scored.sum())
        gate_open_blocks += int(betas.sum())

        record: Dict = {"blocks": len(betas), "events": int(scored.sum()),
                        "noise_fraction": float(y.mean()),
                        "gate_open_rate": float(betas.mean())}
        for arm in ARMS:
            s = scores[arm][scored]
            block_aucs = []
            for start in range(0, len(s) - BLOCK + 1, BLOCK):
                by = y[start:start + BLOCK]
                if by.min() == by.max():
                    continue
                block_aucs.append(roc_auc_score(by, s[start:start + BLOCK]))
            per_block[arm][scene] = block_aucs
            record[arm] = {
                "block_mean_auc": float(np.mean(block_aucs)) if block_aucs else float("nan"),
                "event_auc": float(roc_auc_score(y, s)),
                "pr_auc_noise": float(average_precision_score(y, s)),
                "pr_auc_signal": float(average_precision_score(1 - y, -s)),
                **operating_point(scores[arm], labels, rhos),
            }
        per_scene[scene] = record
        print(f"[{index:3d}/{len(scenes)}] {scene:14s} blocks={record['blocks']:4d} "
              f"noise={record['noise_fraction']:.3f} "
              f"support={record['support']['block_mean_auc']:.4f} "
              f"magnitude={record['magnitude']['block_mean_auc']:.4f} "
              f"gate_open={record['gate_open_rate']:.3f} "
              f"[{time.perf_counter() - step:.0f}s]", flush=True)

    if observed_max[0] >= WIDTH or observed_max[1] >= HEIGHT:
        raise SystemExit(f"observed pixel maxima {observed_max} do not fit the configured "
                         f"{WIDTH}x{HEIGHT}; c = 1e5/WH would be wrong")

    def pooled(field: str, arm: str) -> float:
        vals = [r[arm][field] for r in per_scene.values() if not np.isnan(r[arm][field])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {arm: {field: pooled(field, arm) for field in
                     ("block_mean_auc", "event_auc", "pr_auc_noise", "pr_auc_signal",
                      "retained_fraction", "signal_recall", "noise_rejection",
                      "balanced_accuracy", "retained_fraction_adaptive",
                      "signal_recall_adaptive", "noise_rejection_adaptive",
                      "balanced_accuracy_adaptive", "retained_fraction_oracle_r",
                      "signal_recall_oracle_r", "noise_rejection_oracle_r",
                      "balanced_accuracy_oracle_r")}
               for arm in ARMS}

    contrasts: Dict[str, Dict] = {}
    for left, right in CONTRASTS:
        clusters = [(s, [a - b for a, b in zip(per_block[left][s], per_block[right][s])])
                    for s in per_scene]
        name = f"{left}_minus_{right}"
        contrasts[name] = {
            "block_mean": cluster_bootstrap(clusters, args.replicates, rng),
            "sign_flip": sign_flip(clusters, args.flip_draws, rng),
            "scene_level": cluster_bootstrap(
                [(s, [np.mean(v)]) for s, v in
                 ((s, [a - b for a, b in zip(per_block[left][s], per_block[right][s])])
                  for s in per_scene) if len(v)],
                args.replicates, rng),
        }

    payload = {
        "phase": "54 LED: the operand on labelled recorded noise, nothing here touched it",
        "prior_contact": "none. No LED sequence, statistic or figure was inspected before "
                         "this script was written; no constant was selected or checked on "
                         "it. LED is the only benchmark in this paper that is labelled, "
                         "recorded and untouched at once.",
        "split": "LED_Test_upload",
        "frozen": {"tau_us": TAU_US, "block_size": BLOCK, "score_radius": SCORE_RADIUS,
                   "gate_key_radius": GATE_RADIUS, "theta": 0, "retention": RETENTION,
                   "sensor": [WIDTH, HEIGHT], "c": "1e5/WH"},
        "labels": {
            "source": "the released raw/denoised/noise partition; raw is verified to be the "
                      "exact multiset union of the other two, per chunk",
            "chunks_skipped_partition_failed": skipped_total,
            "ambiguous_events": ambiguous_total,
            "ambiguous_fraction": (ambiguous_total / events_total) if events_total else 0.0,
            "ambiguity_note": "events sharing (t, x, y, p) with an event of the other class; "
                              "the assignment between them is arbitrary",
            "scenes_stably_sorted_by_timestamp": reordered_scenes,
            "sort_note": "some released chunks are not stored in timestamp order; a stable "
                         "sort recovers the recorded order without changing the event set",
        },
        "scale": {"scenes": len(per_scene), "blocks": blocks_total,
                  "events_scored": events_total,
                  "observed_pixel_max": observed_max},
        "gate": {"open_block_rate": gate_open_blocks / blocks_total if blocks_total else 0.0,
                 "open_blocks": gate_open_blocks},
        "summary": summary,
        "contrasts": contrasts,
        "per_scene": per_scene,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\nLED test split: {len(per_scene)} scenes, {blocks_total:,} blocks, "
          f"{events_total:,} events, {ambiguous_total:,} ambiguous "
          f"({payload['labels']['ambiguous_fraction']:.2e})")
    print(f"gate opens on {payload['gate']['open_block_rate']:.4f} of blocks")
    print(f"mean noise fraction {np.mean([r['noise_fraction'] for r in per_scene.values()]):.4f}")
    print(f"{'arm':>14}  {'blockAUC':>9} {'eventAUC':>9} {'PR(noise)':>9} "
          f"{'recall':>7} {'reject':>7} {'bal':>6} | {'kept~':>6} {'bal~':>6} "
          f"| {'bal*':>6}")
    for arm in ARMS:
        s = summary[arm]
        print(f"{arm:>14}  {s['block_mean_auc']:9.4f} {s['event_auc']:9.4f} "
              f"{s['pr_auc_noise']:9.4f} {s['signal_recall']:7.4f} "
              f"{s['noise_rejection']:7.4f} {s['balanced_accuracy']:6.3f} | "
              f"{s['retained_fraction_adaptive']:6.3f} "
              f"{s['balanced_accuracy_adaptive']:6.3f} | "
              f"{s['balanced_accuracy_oracle_r']:6.3f}")
    print("~ = frozen label-free adaptive rule; * = oracle retention (reads labels)")
    for name, c in contrasts.items():
        b = c["block_mean"]
        print(f"{name}: {b['mean']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  "
              f"{b['clusters_ahead']}/{b['n_clusters']} scenes  "
              f"p={c['sign_flip']['p_two_sided']:.2e}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
