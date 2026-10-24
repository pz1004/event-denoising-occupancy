"""Phase 20: an analytic worst-case bound on the fixed-point gate perturbation.

Phase 13 classified a block as safe by comparing |corr_float| against the largest
|corr_float - corr_q| *observed over the same 3,708 blocks*. That is a validation, not a
certificate: it says nothing about a block whose perturbation exceeds the worst one measured.
This phase replaces it with an a-priori bound.

SETTING. Operands are quantized by `q(t) = delta * round(clip(t, 0, M) / delta)` with
delta = 2^-f and M = (2^m - 1) * 2^-f (unsigned saturating, round-to-nearest), then the gate
correlation is computed exactly from the quantized operands -- exactly what
phase13_fixed_point_adaptive.py does.

DERIVATION. Let x, y be a block's operands, P = I - (1/n) 11^T the centering projection,
u = Px, v = Py, so rho = <u,v> / (|u| |v|).

  1. No saturation => |e_i| <= delta/2 elementwise, so |e|_2 <= sqrt(n) delta / 2.
  2. P is an orthogonal projection, so |Pe|_2 <= |e|_2. With |u|_2 = sqrt(n) sigma_x,
         |Pe|_2 / |u|_2  <=  delta / (2 sigma_x)  =:  z_x
     The sqrt(n) cancels: the bound does not depend on block size.
  3. A perturbation of norm r rotates a vector by at most arcsin(r / |u|), so
     angle(u, u~) <= arcsin(z_x).
  4. Angle is a metric and cos is 1-Lipschitz, so the two rotations add:

     THEOREM. If no operand saturates and delta < 2 min(sigma_G, sigma_R):

         |rho~ - rho|  <=  arcsin(delta / (2 sigma_G)) + arcsin(delta / (2 sigma_R))  =:  B

     and B <= (pi delta / 4) (1/sigma_G + 1/sigma_R).

     CERTIFICATE. |rho| > B  =>  the theta = 0 sign decision provably cannot invert.

B depends only on the block's own sigmas and the bit width, so it is computable before
quantizing. Saturation is a precondition, not an afterthought: it is what fails at Q6.3, where
M = 63/8 = 7.875 and G exceeds it on the dense dataset.

This phase checks the theorem empirically on every block and format: it must never be violated
(soundness), and we report how loose it is (tightness) and how many blocks it certifies against
the empirical criterion it replaces.

Read-only. Nothing is trained, fitted or selected.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .core import StateConfig, bidirectional_features
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks, per_block_operands_both
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase13_fixed_point_adaptive import FORMATS, quantize, safe_corr
from .phase14_quantizable_operand import BLOCK_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analytic_bound(sigma_g: float, sigma_r: float, frac_bits: int) -> float:
    """B = arcsin(delta / 2 sigma_G) + arcsin(delta / 2 sigma_R); inf if the precondition fails."""

    delta = 2.0 ** (-frac_bits)
    z_g = delta / (2.0 * sigma_g) if sigma_g > 0 else np.inf
    z_r = delta / (2.0 * sigma_r) if sigma_r > 0 else np.inf
    if z_g >= 1.0 or z_r >= 1.0:
        return float("inf")           # delta >= 2 sigma: rotation is unconstrained
    return float(np.arcsin(z_g) + np.arcsin(z_r))


def scan_condition(block_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
                   config: StateConfig, max_blocks: Optional[int] = None) -> Dict:
    scale_count = len(config.taus_us)
    per_fmt: Dict[str, Dict[str, List[float]]] = {
        name: {"err": [], "bound": [], "rho": [], "saturated": []} for name in FORMATS
    }
    produced = 0

    for events, _labels in block_iter:
        if max_blocks is not None and produced >= max_blocks:
            break
        produced += 1
        raw = bidirectional_features(events, config)
        g, r_gate, _ = per_block_operands_both(raw, scale_count, config.feature_dim)
        if g.std() <= 0 or r_gate.std() <= 0:
            continue
        rho = safe_corr(g, r_gate)
        sig_g, sig_r = float(g.std()), float(r_gate.std())

        for name, (total_bits, frac_bits) in FORMATS.items():
            maximum = float(2 ** total_bits - 1) / float(2 ** frac_bits)
            saturated = bool(g.max() > maximum or r_gate.max() > maximum)
            gq = quantize(g, total_bits, frac_bits)
            rq = quantize(r_gate, total_bits, frac_bits)
            rho_q = safe_corr(gq, rq)
            per_fmt[name]["err"].append(abs(rho_q - rho))
            per_fmt[name]["bound"].append(analytic_bound(sig_g, sig_r, frac_bits))
            per_fmt[name]["rho"].append(rho)
            per_fmt[name]["saturated"].append(saturated)

    out: Dict[str, Dict] = {"blocks": produced, "formats": {}}
    for name, rec in per_fmt.items():
        if not rec["err"]:
            continue
        err = np.asarray(rec["err"])
        bound = np.asarray(rec["bound"])
        rho = np.abs(np.asarray(rec["rho"]))
        sat = np.asarray(rec["saturated"], dtype=bool)
        unsat = ~sat
        # Soundness is only claimed where the no-saturation precondition holds.
        violations = int(np.sum(err[unsat] > bound[unsat] + 1e-12))
        finite = unsat & np.isfinite(bound)
        out["formats"][name] = {
            "n": int(err.size),
            "n_saturated": int(sat.sum()),
            "violations_unsaturated": violations,
            "max_err": float(err.max()),
            "max_bound_unsaturated": (float(bound[finite].max()) if finite.any() else None),
            "mean_slack_ratio": (float(np.mean(bound[finite] / np.maximum(err[finite], 1e-15)))
                                 if finite.any() else None),
            "median_slack_ratio": (float(np.median(bound[finite] / np.maximum(err[finite], 1e-15)))
                                   if finite.any() else None),
            "frac_certified_analytic": float(np.mean(unsat & np.isfinite(bound) & (rho > bound))),
            "frac_certified_empirical": float(np.mean(rho > err.max())),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT
                        / "native3d_ssm/experiments/analytic_contrast/phase20_analytic_bound.json")
    return parser.parse_args()


def aggregate(conditions: Dict[str, Dict]) -> Dict[str, Dict]:
    names = FORMATS.keys()
    agg: Dict[str, Dict] = {}
    for name in names:
        cells = [c["formats"][name] for c in conditions.values()
                 if c.get("blocks") and name in c.get("formats", {})]
        if not cells:
            continue
        tot = sum(c["n"] for c in cells)
        agg[name] = {
            "n": tot,
            "n_saturated": sum(c["n_saturated"] for c in cells),
            "violations_unsaturated": sum(c["violations_unsaturated"] for c in cells),
            "max_err": max(c["max_err"] for c in cells),
            "max_bound_unsaturated": max((c["max_bound_unsaturated"] for c in cells
                                          if c["max_bound_unsaturated"] is not None), default=None),
            "mean_slack_ratio": float(np.average(
                [c["mean_slack_ratio"] for c in cells if c["mean_slack_ratio"] is not None],
                weights=[c["n"] for c in cells if c["mean_slack_ratio"] is not None])),
            "frac_certified_analytic": float(np.average(
                [c["frac_certified_analytic"] for c in cells], weights=[c["n"] for c in cells])),
            "frac_certified_empirical": float(np.average(
                [c["frac_certified_empirical"] for c in cells], weights=[c["n"] for c in cells])),
        }
    return agg


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    dvsclean: Dict[str, Dict] = {}
    dvs_cfg = StateConfig(width=1280, height=720, taus_us=(250.0,), radius=1)
    for seq in SEQUENCES:
        for level in (50, 100):
            name = f"{seq}_{level}"
            dvsclean[name] = scan_condition(
                dvsclean_blocks(args.dvsclean_root / f"{name}.hdf5", BLOCK_SIZE, args.max_blocks),
                dvs_cfg, args.max_blocks)
            print(f"DVSCLEAN {name}: blocks={dvsclean[name]['blocks']}", flush=True)

    dnd21: Dict[str, Dict] = {}
    dnd_cfg = StateConfig(width=346, height=260, taus_us=(250.0,), radius=1)
    for rate in DND21_PAPER:
        for scene, fname in SCENE_FILES.items():
            name = f"{rate}_{scene}"
            dnd21[name] = scan_condition(
                dnd21_blocks(args.dnd21_root / rate / fname, BLOCK_SIZE, args.max_blocks),
                dnd_cfg, args.max_blocks)
            print(f"DND21 {name}: blocks={dnd21[name]['blocks']}", flush=True)

    conditions = {"DVSCLEAN": dvsclean, "DND21": dnd21}
    flat = {**{f"DVSCLEAN/{k}": v for k, v in dvsclean.items()},
            **{f"DND21/{k}": v for k, v in dnd21.items()}}
    payload = {
        "phase": "20 analytic fixed-point bound on the gate correlation (REVIEW P1-4)",
        "theorem": ("|rho~ - rho| <= arcsin(delta / 2 sigma_G) + arcsin(delta / 2 sigma_R), "
                    "delta = 2^-f, valid when no operand saturates and delta < 2 min(sigma)."),
        "certificate": "|rho| > bound  =>  the theta = 0 sign decision cannot invert.",
        "replaces": ("phase13's empirical criterion, which compared |rho| against the worst "
                     "perturbation OBSERVED on the same blocks."),
        "soundness_claim": ("violations_unsaturated must be 0 for every format; it counts blocks "
                            "where the measured error exceeded the predicted bound."),
        "conditions": flat,
        "summary": aggregate(flat),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {time.perf_counter() - started:.1f}s\n")
    s = payload["summary"]
    print(f"{'fmt':8s} {'n':>6s} {'sat':>6s} {'viol':>5s} {'max_err':>9s} {'max_bound':>10s} "
          f"{'slack':>7s} {'cert_an':>8s} {'cert_emp':>9s}")
    for name in FORMATS:
        if name not in s:
            continue
        r = s[name]
        mb = r["max_bound_unsaturated"]
        print(f"{name:8s} {r['n']:6d} {r['n_saturated']:6d} {r['violations_unsaturated']:5d} "
              f"{r['max_err']:9.2e} {(mb if mb is not None else float('nan')):10.2e} "
              f"{r['mean_slack_ratio']:7.1f} {r['frac_certified_analytic']:8.4f} "
              f"{r['frac_certified_empirical']:9.4f}")


if __name__ == "__main__":
    main()
