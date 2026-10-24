"""Audit completed corrected controls and export manuscript-ready contrasts.

This reads stored per-unit metrics, checks the exact-count anchors, separates
float-precision changes from score-definition changes, and records source hashes.
It does not train or select a score/horizon on the evaluation benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from .phase69_operand_isolation import ARTIFACTS, ARMS, write_json


def read(name):
    return json.loads((ARTIFACTS / name).read_text())


def _summary(differences):
    values = np.asarray(differences, dtype=float)
    return {"units": len(values), "mean": float(values.mean()),
            "max_abs": float(np.max(np.abs(values))),
            "min": float(values.min()), "max": float(values.max()),
            "positive_units": int((values > 0).sum()),
            "negative_units": int((values < 0).sum()),
            "ties": int((values == 0).sum())}


def main():
    development = read("phase69_operand_isolation.json")
    led = read("phase70_operand_isolation_led.json")
    for artifact in (development, led):
        if artifact.get("schema_version") != 2 or not artifact.get("complete"):
            raise ValueError("audit requires complete corrected schema-2 artifacts")
    rows = development["per_unit"]
    groups = {
        "development": {name: r for name, r in rows.items() if r["benchmark"] != "BUAA_Campus"},
        "BUAA_Campus": {name: r for name, r in rows.items() if r["benchmark"] == "BUAA_Campus"},
        "LED_Tstar": led["per_scene"]["t_24302us"],
        "LED_fixed_N": led["per_scene"]["fixed_n_10240"],
    }
    primary = ("development", "BUAA_Campus", "LED_Tstar")
    flat_contrasts, finite_sensitivity, means, flat_precision = {}, {}, {}, {}
    for name, group in groups.items():
        means[name] = {arm: float(np.mean([r[arm]["block_mean_auc"] for r in group.values()]))
                       for arm in ARMS}
        flat_contrasts[name], finite_sensitivity[name] = {}, {}
        flat_precision[name] = {arm: _summary([
            r[arm]["block_mean_auc"] - r["logproduct_flat"]["block_mean_auc"]
            for r in group.values()]) for arm in ("logdir_flat", "logdir64_flat")}
        for arm in ("sum_flat", "logdir_flat"):
            delta = [r["count"]["block_mean_auc"] - r[arm]["block_mean_auc"]
                     for r in group.values()]
            flat_contrasts[name][arm] = _summary(delta)
            nearflat = arm.replace("_flat", "_nearflat")
            finite_sensitivity[name][arm] = _summary([
                r[arm]["block_mean_auc"] - r[nearflat]["block_mean_auc"]
                for r in group.values()])
    unit_differences = {arm: np.asarray([
        r["count"]["block_mean_auc"] - r[arm]["block_mean_auc"]
        for name in primary for r in groups[name].values()])
        for arm in ("sum_flat", "logdir_flat")}
    six_max = max(abs(flat_contrasts[name][arm]["mean"])
                  for name in primary for arm in unit_differences)
    individual = {arm: {**_summary(values),
                       "abs_exceeds_largest_benchmark_mean": int((np.abs(values) > six_max).sum())}
                  for arm, values in unit_differences.items()}

    # phase72 uses phase8.operands, including float32 feature export and addition.
    # These are valid single-log controls, independent of the broken isolation code.
    old_led = read("phase72_led_beta0_magnitude.json")["per_scene"]
    old_buaa = read("phase72_buaa_beta0_magnitude.json")["per_sequence"]
    precision = {}
    for metric in ("block_mean_auc", "event_auc"):
        precision["LED_fixed_N_" + metric] = _summary([
            row["logdir64_250us"][metric] - old_led[name]["beta0_magnitude"][metric]
            for name, row in groups["LED_fixed_N"].items()])
        precision["BUAA_Campus_" + metric] = _summary([
            row["logdir64_250us"][metric] - old_buaa[name.split("/", 1)[1]]["repaired"]["beta0_magnitude"][metric]
            for name, row in groups["BUAA_Campus"].items()])

    canonical_drifts = [abs(row["logdir_250us"][metric] - old_led[name]["beta0_magnitude"][metric])
                        for name, row in groups["LED_fixed_N"].items()
                        for metric in ("block_mean_auc", "event_auc")]
    canonical_drifts += [abs(row["logdir_250us"][metric] - old_buaa[name.split("/", 1)[1]]["repaired"]["beta0_magnitude"][metric])
                         for name, row in groups["BUAA_Campus"].items()
                         for metric in ("block_mean_auc", "event_auc")]
    if max(canonical_drifts) > 1e-12:
        raise AssertionError("canonical single-log controls differ from established phase8 metrics")
    source_dir = Path(__file__).resolve().parent
    hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
              for name in ("core.py", "matched_horizon.py", "phase69_operand_isolation.py",
                           "phase70_operand_isolation_led.py", "audit_matched_horizon_results.py")}
    payload = {
        "schema_version": 1,
        "protocol": "noise-positive AUC; equal unit weight after per-window AUC averaging",
        "primary_units": sum(len(groups[name]) for name in primary),
        "exact_flat_control": "integer directional/full-block ring event counts; no finite tau approximation",
        "means": means,
        "count_minus_exact_controls": flat_contrasts,
        "largest_abs_benchmark_mean_difference_six_comparisons": six_max,
        "individual_primary_differences": individual,
        "exact_minus_nearflat_tau_1e9_block_auc": finite_sensitivity,
        "split_log_minus_exact_product_flat_auc": flat_precision,
        "corrected_float64_minus_established_phase8_float32_single_log": precision,
        "canonical_single_log_max_auc_drift_vs_phase8": max(canonical_drifts),
        "precision_scope": "phase72 valid single-log reference at 250us; float32 feature-export and addition rounding only",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_sha256": hashes,
    }
    output = ARTIFACTS / "matched_horizon_validation.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
