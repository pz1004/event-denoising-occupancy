"""Phase 44: empirical diagnostics for the conditional occupancy model.

The model assumes that each of ``n`` ring taps is occupied independently with a
class-conditional probability, so ``S | y ~ Binomial(n, p_y)``. Magnitude weights
that occupancy pattern by amplitudes ``A_k`` whose distribution is shared by the
two classes. Under these assumptions the occupancy count is sufficient, its
likelihood ratio is monotone, and amplitude variability determines the predicted
separation penalty. Uniform fixed-point quantisation can collapse small magnitude
values while leaving the integer count exact.

Measure the quantities needed to assess these assumptions:

  ``p_noise``, ``p_signal``  class-conditional per-tap occupancy;
  ``cv_amplitude``          coefficient of variation of nonzero tap magnitudes;
  ``amplitude_auc``         class separation from mean nonzero tap amplitude,
                            expected to be near 0.5 under the shared-amplitude model;
  ``predicted_ratio``       d_M / d_S, the model's magnitude-to-count separation ratio.

These diagnostics assess model applicability; they do not establish the assumptions
for arbitrary sensor data. Nothing is trained or fitted.
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
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks
from .evaluate_dnd21 import PAPER_AUC as DND21_RATES
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase14_quantizable_operand import BLOCK_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tap_matrix(raw: np.ndarray, config: StateConfig) -> np.ndarray:
    """Per-event x per-tap same-polarity ring magnitudes, forward plus reverse."""

    scale_count = len(config.taus_us)
    width = config.feature_dim
    taps = (2 * config.radius + 1) ** 2 - 1
    base = 7 * scale_count
    columns = [base + k * scale_count for k in range(taps)]
    return np.stack([raw[:, c] + raw[:, width + c] for c in columns], axis=1)


def separation_ratio(p_noise: float, p_signal: float, cv: float) -> float:
    """d_M / d_S in the *shared-amplitude* special case: the magnitude's penalty.

    When the amplitude distribution is the same in both classes, the two statistics have
    the same class-mean difference up to the factor ``mu_A``, so the ratio is purely a
    variance effect. Writing ``v = p(1-p)`` and ``cv = sigma_A / mu_A``, the ratio is
    ``sqrt( (v_N + v_S) / ((p_N + p_S) cv^2 + v_N + v_S) ) <= 1``, with equality only when
    the amplitude is degenerate. This is the clean statement; ``deflection`` below is the
    general one, which does not assume the amplitude is uninformative.
    """

    v_noise = p_noise * (1.0 - p_noise)
    v_signal = p_signal * (1.0 - p_signal)
    occupancy_variance = v_noise + v_signal
    nuisance = (p_noise + p_signal) * cv * cv
    if occupancy_variance + nuisance <= 0:
        return float("nan")
    return float(np.sqrt(occupancy_variance / (occupancy_variance + nuisance)))


def deflection(mean_noise: float, var_noise: float,
               mean_signal: float, var_signal: float) -> float:
    """Standardised class separation |dmu| / sqrt(mean variance), the usual proxy for a
    threshold rule's discriminability. Monotone in AUC for equal-variance Gaussians and
    used here only to *order* two statistics on the same data."""

    pooled = 0.5 * (var_noise + var_signal)
    if pooled <= 0:
        return float("nan")
    return float(abs(mean_noise - mean_signal) / np.sqrt(pooled))


def model_moments(taps: int, p: float, amplitude_mean: float,
                  amplitude_var: float) -> Tuple[float, float, float, float]:
    """Class-conditional moments of the count and the magnitude under the model.

    With ``n`` taps each occupied independently with probability ``p``, and an occupied
    tap carrying an amplitude of mean ``mu`` and variance ``s2`` drawn independently of
    occupancy, the compound sum ``M = sum_k B_k A_k`` has

        E[S] = n p,                 Var[S] = n p (1 - p)
        E[M] = n p mu,              Var[M] = n p ( s2 + (1 - p) mu^2 ) .

    Allowing ``mu`` and ``s2`` to differ by class is what makes this the general case: the
    magnitude then gains a mean-separation term the count does not have, and pays for it
    with the amplitude variance ``n p s2``. Which effect wins is an inequality, not a
    foregone conclusion, and that inequality is what this phase tests per condition.
    """

    count_mean = taps * p
    count_var = taps * p * (1.0 - p)
    magnitude_mean = taps * p * amplitude_mean
    magnitude_var = taps * p * (amplitude_var + (1.0 - p) * amplitude_mean ** 2)
    return count_mean, count_var, magnitude_mean, magnitude_var


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, max_blocks: Optional[int]) -> Dict:
    occupancy = {"noise": [], "signal": []}
    class_amplitudes: Dict[str, List[np.ndarray]] = {"noise": [], "signal": []}
    amplitudes: List[np.ndarray] = []
    amplitude_scores: List[np.ndarray] = []
    amplitude_labels: List[np.ndarray] = []
    count_auc: List[float] = []
    magnitude_auc: List[float] = []
    produced = 0

    for events, labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        labels = np.asarray(labels).astype(int)
        if labels.min() == labels.max():
            continue

        raw = bidirectional_features(events, config)
        taps = tap_matrix(raw, config)
        occupied = taps > 0

        # (1) class-conditional per-tap occupancy
        for name, mask in (("noise", labels == 1), ("signal", labels == 0)):
            if mask.any():
                occupancy[name].append(float(occupied[mask].mean()))

        # (2) amplitude distribution over occupied taps, pooled and per class
        values = taps[occupied]
        if values.size:
            amplitudes.append(values.astype(np.float64))
        for name, mask in (("noise", labels == 1), ("signal", labels == 0)):
            if mask.any():
                class_values = taps[mask][occupied[mask]]
                if class_values.size:
                    class_amplitudes[name].append(class_values.astype(np.float64))

        # (3) is amplitude, given occupancy, class-informative on its own?
        # Mean amplitude over an event's occupied taps; events with an empty ring are
        # excluded because they have no amplitude to speak of.
        occupied_count = occupied.sum(axis=1)
        has_ring = occupied_count > 0
        if has_ring.sum() > 1:
            mean_amplitude = np.zeros(len(taps))
            mean_amplitude[has_ring] = (
                taps.sum(axis=1)[has_ring] / occupied_count[has_ring]
            )
            sub_labels = labels[has_ring]
            if sub_labels.min() != sub_labels.max():
                amplitude_scores.append(mean_amplitude[has_ring])
                amplitude_labels.append(sub_labels)

        # (4) the two operands as standalone scores, for reference
        count = occupied.sum(axis=1).astype(np.float64)
        magnitude = taps.sum(axis=1)
        for store, values_ in ((count_auc, -count), (magnitude_auc, -magnitude)):
            if values_.std() > 0:
                store.append(roc_auc_score(labels, values_))

    if not occupancy["noise"] or not amplitudes:
        return {"blocks": 0}

    pooled_amplitudes = np.concatenate(amplitudes)
    mean_amplitude = float(pooled_amplitudes.mean())
    cv = float(pooled_amplitudes.std() / mean_amplitude) if mean_amplitude > 0 else float("nan")
    p_noise = float(np.mean(occupancy["noise"]))
    p_signal = float(np.mean(occupancy["signal"]))

    moments = {}
    for name in ("noise", "signal"):
        if class_amplitudes[name]:
            pooled = np.concatenate(class_amplitudes[name])
            moments[name] = (float(pooled.mean()), float(pooled.var()))
        else:
            moments[name] = (float("nan"), float("nan"))

    taps_count = (2 * config.radius + 1) ** 2 - 1
    count_mean_n, count_var_n, mag_mean_n, mag_var_n = model_moments(
        taps_count, p_noise, *moments["noise"])
    count_mean_s, count_var_s, mag_mean_s, mag_var_s = model_moments(
        taps_count, p_signal, *moments["signal"])
    d_count = deflection(count_mean_n, count_var_n, count_mean_s, count_var_s)
    d_magnitude = deflection(mag_mean_n, mag_var_n, mag_mean_s, mag_var_s)

    amplitude_auc = None
    if amplitude_scores:
        scores = np.concatenate(amplitude_scores)
        labels_all = np.concatenate(amplitude_labels)
        if labels_all.min() != labels_all.max() and scores.std() > 0:
            amplitude_auc = float(roc_auc_score(labels_all, scores))

    return {
        "blocks": produced,
        "p_noise": p_noise,
        "p_signal": p_signal,
        "occupancy_gap": p_noise - p_signal,
        "cv_amplitude": cv,
        "amplitude_auc": amplitude_auc,
        "predicted_separation_ratio": separation_ratio(p_noise, p_signal, cv),
        "amplitude_mean_noise": moments["noise"][0],
        "amplitude_mean_signal": moments["signal"][0],
        "amplitude_var_noise": moments["noise"][1],
        "amplitude_var_signal": moments["signal"][1],
        "deflection_count": d_count,
        "deflection_magnitude": d_magnitude,
        "model_predicts_count_wins": bool(d_count > d_magnitude),
        "count_auc": float(np.mean(count_auc)) if count_auc else None,
        "magnitude_auc": float(np.mean(magnitude_auc)) if magnitude_auc else None,
    }


def summarise(conditions: Dict[str, Dict]) -> Dict[str, float]:
    usable = [c for c in conditions.values() if c.get("blocks")]
    if not usable:
        return {}
    fields = ("p_noise", "p_signal", "occupancy_gap", "cv_amplitude", "amplitude_auc",
              "predicted_separation_ratio", "deflection_count", "deflection_magnitude",
              "count_auc", "magnitude_auc")
    out = {}
    for field in fields:
        values = [c[field] for c in usable if c.get(field) is not None]
        if values:
            out[field] = float(np.mean(values))
    counts = [c["count_auc"] for c in usable if c.get("count_auc") is not None]
    mags = [c["magnitude_auc"] for c in usable if c.get("magnitude_auc") is not None]
    if counts and len(counts) == len(mags):
        measured = np.asarray(counts) > np.asarray(mags)
        out["count_beats_magnitude_conditions"] = int(measured.sum())
        out["conditions"] = len(counts)
        predicted = np.asarray([bool(c.get("model_predicts_count_wins")) for c in usable])
        if len(predicted) == len(measured):
            out["model_predicts_count_wins_conditions"] = int(predicted.sum())
            out["model_agrees_with_measurement"] = int((predicted == measured).sum())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT
                        / "native3d_ssm/experiments/analytic_contrast/phase44_occupancy_model.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    dnd_config = StateConfig(width=346, height=260, taus_us=(250.0,), radius=args.radius)
    dvs_config = StateConfig(width=1280, height=720, taus_us=(250.0,), radius=args.radius)

    dnd21: Dict[str, Dict] = {}
    for rate in DND21_RATES:
        for scene, filename in SCENE_FILES.items():
            path = args.dnd21_root / rate / filename
            if not path.exists():
                continue
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(path, BLOCK_SIZE, args.max_blocks), dnd_config, args.max_blocks
            )
            print(json.dumps({"dataset": "DND21", "condition": name, **dnd21[name]}), flush=True)

    dvsclean: Dict[str, Dict] = {}
    for sequence in SEQUENCES:
        for level in (50, 100):
            name = f"{sequence}_{level}"
            path = args.dvsclean_root / f"{name}.hdf5"
            if not path.exists():
                continue
            dvsclean[name] = scan_condition(
                dvsclean_blocks(path, BLOCK_SIZE, args.max_blocks), dvs_config, args.max_blocks
            )
            print(json.dumps({"dataset": "DVSCLEAN", "condition": name, **dvsclean[name]}),
                  flush=True)

    payload = {
        "phase": "44 occupancy model behind the operand result",
        "question": ("Under what conditions is binary spatial occupancy preferable to "
                     "accumulated magnitude? The model's assumptions, measured."),
        "model": ("S | y ~ Binomial(n, p_y); magnitude = sum over occupied taps of an "
                  "amplitude whose distribution is shared by the two classes."),
        "radius": args.radius,
        "summary": {"DND21": summarise(dnd21), "DVSCLEAN": summarise(dvsclean),
                    "overall_20": summarise({**dnd21, **dvsclean})},
        "per_condition": {"DND21": dnd21, "DVSCLEAN": dvsclean},
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
