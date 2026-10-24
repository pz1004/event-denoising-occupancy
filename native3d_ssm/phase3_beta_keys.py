"""Phase 3: compare beta keys (event-rate rho vs. corr(G,R)) and emit a Table-1.

Phase 2' showed a rate-keyed beta lifts DVSCLEAN but (a) mishandles DND21 1 Hz
Driving and (b) fails to transfer DND21->DVSCLEAN (extrapolation below the
training rho range). corr(G,R) -- the within-block Pearson correlation of the
global and ring log-operands -- separates the two regimes at ~0 and is monotone
with the oracle beta even within DND21, so it should transfer.

This script extracts all 20 conditions once, fits monotone logistic beta-rules on
each key (rho: beta increasing; corr: beta decreasing), evaluates them in-sample
and leave-one-dataset-out (LODO), and writes an EDmamba-Table-1-style per-condition
markdown plus a JSON summary and a single aggregate ROC AUC per row.

Read-only over data on disk. beta is a frozen function; no model is trained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .evaluate_adaptive_beta import (
    BETA_GRID,
    EDMAMBA,
    dnd21_blocks,
    dvsclean_blocks,
    scan_condition,
)
from .evaluate_dnd21 import PAPER_AUC as DND21_PAPER
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import PAPER_AUC as DVS_PAPER
from .evaluate_dvsclean_analytic import SEQUENCES
from .core import StateConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BETA_ONE = int(np.argmin(np.abs(BETA_GRID - 1.0)))

# EDmamba-Table-1 display order and pretty names.
DVS_ORDER = [("MAH00446", "Double-bracket"), ("MAH00456", "Double-ship"),
             ("MAH00454", "Double-airplane"), ("MAH00455", "Multi-helicopter"),
             ("MAH00465", "Multi-car")]
DND_ORDER = ["1hz", "3hz", "5hz", "7hz", "10hz"]


def gate(key: np.ndarray, k: float, center: float, decreasing: bool) -> np.ndarray:
    """Monotone logistic gate in [0, 1]. decreasing=True -> beta falls as key rises."""

    sign = 1.0 if decreasing else -1.0
    return 1.0 / (1.0 + np.exp(sign * k * (np.asarray(key, dtype=np.float64) - center)))


def realized(conditions: Dict, key_name: str, k: float, center: float, decreasing: bool) -> Dict[str, float]:
    """Per-condition realized block-mean AUC under per-block beta = gate(key)."""

    out = {}
    for name, c in conditions.items():
        key = np.log10(np.maximum(c["per_block_rho"], 1e-9)) if key_name == "rho" else c["per_block_corr"]
        betas = gate(key, k, center, decreasing)
        snap = np.clip(np.round(betas / 0.1).astype(int), 0, len(BETA_GRID) - 1)
        block_auc = c["per_block_auc"][np.arange(len(snap)), snap]
        out[name] = float(np.nanmean(block_auc))
    return out


def fit(conditions: Dict, key_name: str, decreasing: bool) -> Tuple[float, float]:
    """Grid-search (k, center) maximizing mean realized AUC on the given conditions."""

    if key_name == "rho":
        centers = np.linspace(-0.6, 1.0, 17)
        ks = (2.0, 4.0, 6.0, 8.0, 12.0)
    else:
        centers = np.linspace(-0.35, 0.20, 23)
        ks = (4.0, 8.0, 12.0, 20.0, 40.0)
    best = (-1.0, None)
    for k in ks:
        for center in centers:
            score = float(np.mean(list(realized(conditions, key_name, k, float(center), decreasing).values())))
            if score > best[0]:
                best = (score, (k, float(center)))
    return best[1]


def endpoint_aucs(conditions: Dict, index: int) -> Dict[str, float]:
    return {name: float(c["auc_by_beta"][index]) for name, c in conditions.items()}


def extract() -> Tuple[Dict, Dict]:
    dvs_cfg = StateConfig(width=1280, height=720, taus_us=(250.0,))
    dnd_cfg = StateConfig(width=346, height=260, taus_us=(250.0,))
    dvs = {}
    for seq, _ in DVS_ORDER:
        for lvl in (50, 100):
            dvs[f"{seq}_{lvl}"] = scan_condition(
                dvsclean_blocks(PROJECT_ROOT / f"DVSCLEAN/simulated_data/{seq}_{lvl}.hdf5", 10240, None),
                dvs_cfg, None)
        print(f"extracted DVSCLEAN {seq}", flush=True)
    dnd = {}
    for rate in DND_ORDER:
        for scene, fname in SCENE_FILES.items():
            dnd[f"{rate}_{scene}"] = scan_condition(
                dnd21_blocks(PROJECT_ROOT / f"ECCV2024_datasets/AUC_test/{rate}/{fname}", 10240, None),
                dnd_cfg, None)
        print(f"extracted DND21 {rate}", flush=True)
    return dvs, dnd


def build_table(dvs, dnd, rows: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """rows[label] = {'DVSCLEAN': {cond:auc}, 'DND21': {cond:auc}}."""

    lines = []
    # DND21 block
    lines.append("### DND21 (346 x 260) — per-condition AUC\n")
    header = "| Method | " + " | ".join(
        f"{r}/{s[:2].upper()}" for r in DND_ORDER for s in ("hotel-bar", "driving")) + " | Mean |"
    lines.append(header)
    lines.append("|" + "---|" * (10 + 2))
    for label, data in rows.items():
        cells = []
        vals = []
        for rate in DND_ORDER:
            for scene in ("hotel-bar", "driving"):
                v = data["DND21"].get(f"{rate}_{scene}")
                cells.append(f"{v:.4f}" if v is not None else "-")
                if v is not None:
                    vals.append(v)
        mean = f"{np.mean(vals):.4f}" if vals else "-"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | **{mean}** |")
    # DVSCLEAN block
    lines.append("\n### DVSCLEAN (1280 x 720) — per-condition AUC\n")
    header = "| Method | " + " | ".join(
        f"{name.split('-')[0][:4]}.{name.split('-')[1][:4] if '-' in name else ''} {lvl}%"
        for _, name in DVS_ORDER for lvl in (50, 100)) + " | Mean |"
    lines.append(header)
    lines.append("|" + "---|" * (10 + 2))
    for label, data in rows.items():
        cells = []
        vals = []
        for seq, _ in DVS_ORDER:
            for lvl in (50, 100):
                v = data["DVSCLEAN"].get(f"{seq}_{lvl}")
                cells.append(f"{v:.4f}" if v is not None else "-")
                if v is not None:
                    vals.append(v)
        mean = f"{np.mean(vals):.4f}" if vals else "-"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | **{mean}** |")
    return "\n".join(lines)


def row_from_map(dvs_map, dnd_map) -> Dict:
    return {"DVSCLEAN": dvs_map, "DND21": dnd_map}


def single_auc(row: Dict) -> Dict[str, float]:
    dvs = list(row["DVSCLEAN"].values())
    dnd = list(row["DND21"].values())
    return {
        "DND21_mean": float(np.mean(dnd)),
        "DVSCLEAN_mean": float(np.mean(dvs)),
        "overall_mean_20": float(np.mean(dvs + dnd)),
    }


def main() -> None:
    dvs, dnd = extract()

    # Endpoints.
    beta1 = row_from_map(endpoint_aucs(dvs, BETA_ONE), endpoint_aucs(dnd, BETA_ONE))
    beta0 = row_from_map(endpoint_aucs(dvs, 0), endpoint_aucs(dnd, 0))
    oracle = row_from_map(
        {n: float(np.nanmax(c["auc_by_beta"])) for n, c in dvs.items()},
        {n: float(np.nanmax(c["auc_by_beta"])) for n, c in dnd.items()})

    # Fit both keys, in-sample (all 20) and LODO.
    allc = {**dvs, **dnd}
    fits = {}
    realized_rows = {}
    for key_name, decreasing in (("rho", False), ("corr", True)):
        k_all, c_all = fit(allc, key_name, decreasing)
        k_dnd, c_dnd = fit(dnd, key_name, decreasing)   # -> apply to DVSCLEAN
        k_dvs, c_dvs = fit(dvs, key_name, decreasing)   # -> apply to DND21
        fits[key_name] = {"in_sample": [k_all, c_all], "fit_on_DND21": [k_dnd, c_dnd], "fit_on_DVSCLEAN": [k_dvs, c_dvs]}
        realized_rows[f"{key_name}_in_sample"] = row_from_map(
            realized(dvs, key_name, k_all, c_all, decreasing),
            realized(dnd, key_name, k_all, c_all, decreasing))
        realized_rows[f"{key_name}_LODO"] = row_from_map(
            realized(dvs, key_name, k_dnd, c_dnd, decreasing),   # DVSCLEAN via DND21-fit
            realized(dnd, key_name, k_dvs, c_dvs, decreasing))   # DND21 via DVSCLEAN-fit

    # EDmamba paper values (from codebase; format follows the attached Table 1).
    edmamba = row_from_map(
        {f"{seq}_{lvl}": DVS_PAPER[seq][lvl] for seq, _ in DVS_ORDER for lvl in (50, 100)},
        {f"{rate}_{scene}": DND21_PAPER[rate][scene] for rate in DND_ORDER for scene in ("hotel-bar", "driving")})

    table_rows = {
        "EDmamba (paper)": edmamba,
        "Native3D beta=1 (current)": beta1,
        "Native3D ring-only (beta=0)": beta0,
        "Native3D adaptive corr(G,R) LODO": realized_rows["corr_LODO"],
        "Native3D adaptive corr(G,R) in-sample": realized_rows["corr_in_sample"],
        "Native3D adaptive rho LODO": realized_rows["rho_LODO"],
        "Native3D oracle-beta (ceiling)": oracle,
    }
    table_md = build_table(dvs, dnd, table_rows)

    singles = {label: single_auc(row) for label, row in table_rows.items()}
    key_pred = {
        "pearson_betastar_vs_log10rho": float(np.corrcoef(
            [c["oracle_beta"] for c in allc.values()],
            np.log10([max(c["rho_mean"], 1e-9) for c in allc.values()]))[0, 1]),
        "pearson_betastar_vs_corrGR": float(np.corrcoef(
            [c["oracle_beta"] for c in allc.values()],
            [c["corr_GR_mean"] for c in allc.values()])[0, 1]),
    }

    out_json = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase3_beta_keys.json"
    out_json.write_text(json.dumps({
        "fits": fits,
        "beta_key_predictiveness": key_pred,
        "single_auc_per_row": singles,
        "table_rows": table_rows,
    }, indent=2, sort_keys=True) + "\n")
    out_md = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast/phase3_table1.md"
    out_md.write_text("# Phase 3 — EDmamba-Table-1-style AUC\n\n" + table_md + "\n\n## Single ROC AUC per row\n\n"
                      "| Method | DND21 mean | DVSCLEAN mean | Overall (20) |\n|---|---:|---:|---:|\n"
                      + "\n".join(f"| {label} | {s['DND21_mean']:.4f} | {s['DVSCLEAN_mean']:.4f} | {s['overall_mean_20']:.4f} |"
                                  for label, s in singles.items()) + "\n")
    print("\n" + table_md)
    print("\nSINGLE ROC AUC (means):")
    for label, s in singles.items():
        print(f"  {label:>42}: DND21={s['DND21_mean']:.4f} DVSCLEAN={s['DVSCLEAN_mean']:.4f} overall={s['overall_mean_20']:.4f}")
    print("\nkey predictiveness:", key_pred)
    print("fits:", json.dumps(fits, indent=2))
    print(f"\nWrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
