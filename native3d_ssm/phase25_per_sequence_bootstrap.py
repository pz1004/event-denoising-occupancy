"""Phase 25: replace the n = 8 E-MLB interval with a per-sequence one.

Every E-MLB interval in the manuscript resamples the eight cells, because the per-sequence
MESR values were averaged away inside the scoring loops and never stored. `phase10_emlb_blind.py`
and `reproduce_edformer_emlb.py` now retain them, keyed identically as `{scene}-{nd}-{rep}`,
so the two sides can be paired sequence by sequence.

Three estimates are produced, in decreasing order of how much they assume:

  1. **Paired per-sequence** (requires the EDformer per-sequence artifact). margin(s) =
     ours(s) - edformer(s) on the same recording. The Raw anchor cancels, so a margin on
     MESR is a margin on Delta-Raw. This is the estimate the manuscript wants.

  2. **Semi-paired** (fallback when only EDformer's *cell means* are available). Each of our
     sequences is compared against its own cell's EDformer mean, treated as a known
     constant. This narrows the interval on *our* sampling error only and leaves the
     baseline's unmodelled, so it is an upper bound on precision, not a paired test. It is
     labelled as such and must never be reported as one.

  3. **Our own Delta-Raw**, per sequence, which needs no baseline at all.

Only rep 1 can be paired: the EDformer reproduction was run on `--reps 1` (48 sequences per
cell) while our side covers reps 1-3 (144 per cell). That asymmetry is reported explicitly,
together with whether restricting our side to rep 1 moves our cell means -- if it does, the
manuscript's Table 2 is comparing means over different sequence sets.

Read-only over artifacts; nothing is rescored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
REPRO = PROJECT_ROOT / "reproduction/results"
CELLS = [(p, nd) for p in ("D-END", "N-END") for nd in ("ND00", "ND04", "ND16", "ND64")]
REPLICATES = 10_000
SEED = 20260730


def finite(values: Sequence[float]) -> np.ndarray:
    """Drop non-finite entries.

    MESR is undefined for a stream that filtering has left too sparse, and the cell means in
    every existing artifact are `np.nanmean`, so they already drop these silently. At the
    sequence level the drop has to be explicit, and for a *paired* statistic a sequence is
    usable only when BOTH sides are finite -- which is why callers intersect first and only
    then reduce.
    """

    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def bootstrap(values: Sequence[float], rng: np.random.Generator,
              replicates: int = REPLICATES) -> Dict:
    arr = finite(values)
    if arr.size < 2:
        return {"mean": float(arr.mean()) if arr.size else None, "n": int(arr.size),
                "ci95": None}
    idx = rng.integers(0, arr.size, size=(replicates, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(arr.mean()), "n": int(arr.size),
            "sd": float(arr.std(ddof=1)),
            "ci95": [float(lo), float(hi)],
            "bootstrap_se": float(means.std(ddof=1)),
            "replicates": replicates}


def stratified_bootstrap(by_cell: Dict[str, List[float]], rng: np.random.Generator,
                         replicates: int = REPLICATES) -> Dict:
    """Resample sequences *within* each cell, then average the eight cell means.

    The eight cells are two lighting conditions x four ND levels and are a fixed design, not
    a sample from a population of cells. Resampling cells (what the n = 8 interval did)
    treats them as exchangeable draws, which they are not. Resampling sequences inside each
    cell and keeping the cell structure fixed matches how the benchmark is actually built.
    """

    cells = [c for c in (finite(v) for v in by_cell.values()) if c.size > 1]
    if len(cells) < 2:
        return {}
    means = np.empty(replicates, dtype=np.float64)
    for rep in range(replicates):
        means[rep] = np.mean([c[rng.integers(0, c.size, c.size)].mean() for c in cells])
    observed = float(np.mean([c.mean() for c in cells]))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": observed, "cells": len(cells),
            "sequences_total": int(sum(c.size for c in cells)),
            "ci95": [float(lo), float(hi)],
            "bootstrap_se": float(means.std(ddof=1)),
            "replicates": replicates}


def exact_sign_permutation(margins: Sequence[float], rng: np.random.Generator,
                           max_enumerate: int = 20, draws: int = 200_000) -> Dict:
    """Paired randomisation test. Exact by enumeration when n is small, sampled when not."""

    arr = finite(margins)
    n = arr.size
    if n < 2:
        return {"observed_mean_margin": None, "p_two_sided": None, "n": int(n)}
    observed = float(arr.mean())
    if n <= max_enumerate:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        means = (signs * arr).mean(axis=1)
        exact = True
    else:
        signs = rng.integers(0, 2, size=(draws, n)) * 2 - 1
        means = (signs * arr).mean(axis=1)
        exact = False
    return {"observed_mean_margin": observed,
            "n": int(n),
            "p_two_sided": float(np.mean(np.abs(means) >= abs(observed) - 1e-15)),
            "exact": exact,
            "assignments": int(means.size)}


def load_ours(path: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict]]:
    payload = json.loads(path.read_text())
    per_sequence: Dict[str, Dict[str, float]] = {}
    cell_of: Dict[str, str] = {}
    cells: Dict[str, Dict] = {}
    for part, nd in CELLS:
        cell = payload["results"].get(part, {}).get(nd)
        if not cell or "per_sequence" not in cell:
            continue
        key = f"{part}/{nd}"
        cells[key] = {"sequences": cell["sequences"],
                      "raw_mesr": cell["raw_mesr"],
                      "support_score_fixed_mesr": cell["support_score_fixed_mesr"]}
        for seq, values in cell["per_sequence"].items():
            per_sequence[seq] = values
            cell_of[seq] = key
    return per_sequence, {"cells": cells, "cell_of": cell_of}


def load_edformer_per_sequence(path: Path) -> Optional[Dict[str, Dict[str, float]]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    out: Dict[str, Dict[str, float]] = {}
    for part, nd in CELLS:
        cell = payload["results"].get(part, {}).get(nd, {})
        for seq, values in cell.get("per_sequence", {}).items():
            out[seq] = values
    return out or None


def missingness_analysis(ours_path: Path, edf_path: Path) -> Dict:
    """Is the dropped-pair set informative? It is, and it biases the paired estimate.

    MESR is undefined when a filtered stream is too sparse to score. That is not neutral
    missing data: it is produced *by* aggressive filtering, so the pairs it removes are
    exactly the ones where the baseline discards most. Since the baseline's cell-level
    advantage on E-MLB comes from the two darkest cells -- where it retains 15.0% and 8.1%
    -- dropping those pairs deletes its best evidence and inflates our margin.

    A complete-case paired analysis is therefore NOT a neutral refinement of the cell-level
    comparison. Both aggregations are confounded, in opposite directions, and this function
    reports the size of the effect rather than picking a winner.
    """

    ours = json.loads(ours_path.read_text())["results"]
    edf = json.loads(edf_path.read_text())["results"]
    rows, retention, nan_rate = [], [], []
    for part, nd in CELLS:
        o = ours.get(part, {}).get(nd, {}).get("per_sequence", {})
        e = edf.get(part, {}).get(nd, {}).get("per_sequence", {})
        shared = sorted(set(o) & set(e))
        if not shared:
            continue
        ov = np.array([o[s]["support_score_fixed_mesr"] for s in shared], dtype=np.float64)
        ev = np.array([e[s]["edformer_mesr"] for s in shared], dtype=np.float64)
        both = np.isfinite(ov) & np.isfinite(ev)
        r = edf[part][nd].get("mean_retention")
        rate = float((~np.isfinite(ev)).mean())
        retention.append(r); nan_rate.append(rate)
        rows.append({
            "cell": f"{part}/{nd}", "sequences": len(shared),
            "dropped_ours": int((~np.isfinite(ov)).sum()),
            "dropped_baseline": int((~np.isfinite(ev)).sum()),
            "kept": int(both.sum()),
            "baseline_retention": r,
            "baseline_nan_rate": rate,
            # the two ways of aggregating, which disagree wherever drops are heavy
            "cell_mean_margin": float(np.nanmean(ov) - np.nanmean(ev)),
            "complete_case_margin": float((ov[both] - ev[both]).mean()),
        })
    corr = (float(np.corrcoef(retention, nan_rate)[0, 1])
            if len(retention) > 2 else None)
    cc = [r["complete_case_margin"] for r in rows]
    cm = [r["cell_mean_margin"] for r in rows]
    return {
        "verdict": ("INFORMATIVE MISSINGNESS -- the pooled paired estimate is biased in our "
                    "favour and must not be reported as evidence that we are ahead"),
        "mechanism": ("MESR is undefined when filtering leaves a stream too sparse, so the "
                      "baseline goes missing precisely where it filters hardest, which is "
                      "where its cell-level advantage comes from"),
        "pearson_r_baseline_retention_vs_nan_rate": corr,
        "dropped_total": int(sum(r["sequences"] - r["kept"] for r in rows)),
        "dropped_ours_total": int(sum(r["dropped_ours"] for r in rows)),
        "dropped_baseline_total": int(sum(r["dropped_baseline"] for r in rows)),
        "complete_case_cell_mean_margin": float(np.mean(cc)) if cc else None,
        "cell_mean_margin": float(np.mean(cm)) if cm else None,
        "cells_ahead_complete_case": int(sum(1 for x in cc if x > 0)),
        "cells_ahead_cell_mean": int(sum(1 for x in cm if x > 0)),
        "per_cell": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path,
                        default=ARTIFACTS / "phase24_emlb_per_sequence.json")
    parser.add_argument("--edformer-per-sequence", type=Path,
                        default=REPRO / "emlb_edformer_mesr_per_sequence.json")
    parser.add_argument("--edformer-cells", type=Path,
                        default=ARTIFACTS / "phase18_retention_matched.json")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "phase25_per_sequence_bootstrap.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(SEED)
    ours, meta = load_ours(args.ours)
    cell_of = meta["cell_of"]
    if not ours:
        raise SystemExit(f"no per-sequence values in {args.ours}; is the run still going?")

    payload: Dict = {
        "phase": "25 per-sequence E-MLB intervals",
        "source_ours": str(args.ours.relative_to(PROJECT_ROOT)),
        "sequences_ours": len(ours),
        "our_cell_means": meta["cells"],
    }

    # (3) our own Delta-Raw, needing no baseline.
    delta_by_cell: Dict[str, List[float]] = {}
    for seq, v in ours.items():
        d = v["support_score_fixed_mesr"] - v["raw_mesr"]
        delta_by_cell.setdefault(cell_of[seq], []).append(d)
    flat = [d for v in delta_by_cell.values() for d in v]
    payload["our_delta_raw"] = {
        "note": ("our MESR improvement over unfiltered input, per sequence. No baseline "
                 "involved, so this is a clean interval at the sequence level."),
        "pooled_sequences": bootstrap(flat, rng),
        "stratified_by_cell": stratified_bootstrap(delta_by_cell, rng),
    }

    # rep-1 restriction, which is the only set the baseline covers.
    rep1 = {s: v for s, v in ours.items() if s.endswith("-1")}
    payload["rep1_restriction"] = {
        "note": ("the EDformer reproduction ran --reps 1 only, so only rep 1 can ever be "
                 "paired. If our rep-1 cell means differ from our reps-1-3 cell means, then "
                 "Table 2 compares means taken over different sequence sets."),
        "sequences_rep1": len(rep1),
    }
    drift = {}
    for key in meta["cells"]:
        allv = [ours[s]["support_score_fixed_mesr"] for s in ours if cell_of[s] == key]
        onev = [ours[s]["support_score_fixed_mesr"] for s in rep1 if cell_of[s] == key]
        a, o = finite(allv), finite(onev)
        if a.size and o.size:
            # nanmean semantics, matching how every cell mean in the artifacts is computed.
            drift[key] = {"reps123_mean": float(a.mean()), "reps123_n": int(a.size),
                          "rep1_mean": float(o.mean()), "rep1_n": int(o.size),
                          "difference": float(o.mean() - a.mean())}
    payload["rep1_restriction"]["cell_mean_drift"] = drift
    if drift:
        payload["rep1_restriction"]["max_abs_drift"] = max(
            abs(d["difference"]) for d in drift.values())

    # (1) the real thing, if the baseline per-sequence artifact exists.
    edf = load_edformer_per_sequence(args.edformer_per_sequence)
    if edf:
        shared = sorted(set(ours) & set(edf))
        margins_by_cell: Dict[str, List[float]] = {}
        for seq in shared:
            m = ours[seq]["support_score_fixed_mesr"] - edf[seq]["edformer_mesr"]
            margins_by_cell.setdefault(cell_of[seq], []).append(m)
        margins = [m for v in margins_by_cell.values() for m in v]
        cells_covered = sorted({cell_of[s] for s in shared})
        payload["paired_per_sequence"] = {
            "status": ("available" if len(cells_covered) == len(CELLS)
                       else "PARTIAL -- baseline run incomplete, do NOT report as final"),
            "cells_covered": cells_covered,
            "cells_expected": len(CELLS),
            "sequences_intersecting": len(shared),
            "sequences_finite_both_sides": int(finite(margins).size),
            "sequences_dropped_nonfinite": len(margins) - int(finite(margins).size),
            "pooled": bootstrap(margins, rng),
            "stratified_by_cell": stratified_bootstrap(margins_by_cell, rng),
            "randomisation_test": exact_sign_permutation(margins, rng),
            "sequences_ahead": int(sum(1 for m in finite(margins) if m > 0)),
            "BIAS_WARNING": ("complete-case only. See `missingness`: the dropped pairs are "
                             "not missing at random, they are where the baseline filters "
                             "hardest, so this estimate is biased in our favour."),
        }
        payload["missingness"] = missingness_analysis(
            args.ours, args.edformer_per_sequence)
    else:
        # (2) the fallback, explicitly not a paired test.
        cells_json = json.loads(args.edformer_cells.read_text())
        ref = {f"{p}/{nd}": cells_json["edformer_reference"][p][nd]["edformer_mesr"]
               for p, nd in CELLS}
        semi_by_cell: Dict[str, List[float]] = {}
        for seq, v in rep1.items():
            semi_by_cell.setdefault(cell_of[seq], []).append(
                v["support_score_fixed_mesr"] - ref[cell_of[seq]])
        semi = [m for v in semi_by_cell.values() for m in v]
        payload["paired_per_sequence"] = {
            "status": "unavailable",
            "why": (f"{args.edformer_per_sequence.name} not found. EDformer inference cannot "
                    "be re-run in this environment: its model needs sparseconvnet, which is "
                    "not on PyPI and is not installed, and the recorded run used python "
                    "3.10.20 with torch 2.2.1+cu118 while this environment has python 3.13 "
                    "with torch 2.10. Rebuilding that toolchain is a prerequisite."),
        }
        payload["semi_paired_upper_bound"] = {
            "warning": ("NOT a paired test and must not be reported as one. Each of our "
                        "rep-1 sequences is differenced against its cell's EDformer MEAN, "
                        "treated as a known constant, so the baseline's sampling error is "
                        "unmodelled and this interval is narrower than the truth."),
            "sequences_considered": len(semi),
            "sequences_finite": int(finite(semi).size),
            "pooled": bootstrap(semi, rng),
            "stratified_by_cell": stratified_bootstrap(semi_by_cell, rng),
            "randomisation_test": exact_sign_permutation(semi, rng),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    d = payload["our_delta_raw"]
    print(f"\nour Delta-Raw, {d['pooled_sequences']['n']} sequences: "
          f"{d['pooled_sequences']['mean']:+.4f} "
          f"[{d['pooled_sequences']['ci95'][0]:+.4f}, {d['pooled_sequences']['ci95'][1]:+.4f}]")
    s = d["stratified_by_cell"]
    if s:
        print(f"  stratified within cells:            {s['mean']:+.4f} "
              f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]")
    p = payload["paired_per_sequence"]
    if "pooled" in p:
        partial = not p["status"].startswith("available")
        tag = ("PARTIAL PAIRED (%d/%d cells) -- NOT FINAL"
               % (len(p["cells_covered"]), p["cells_expected"])) if partial else "PAIRED"
        print(f"\n{tag} margin, {p['sequences_finite_both_sides']} sequences: "
              f"{p['pooled']['mean']:+.4f} "
              f"[{p['pooled']['ci95'][0]:+.4f}, {p['pooled']['ci95'][1]:+.4f}]  "
              f"randomisation p={p['randomisation_test']['p_two_sided']:.4f}")
        st = p.get("stratified_by_cell") or {}
        if st.get("ci95"):
            print(f"  stratified within cells: {st['mean']:+.4f} "
                  f"[{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]")
    else:
        q = payload["semi_paired_upper_bound"]
        print(f"\n[fallback, NOT paired] {q['sequences_finite']} sequences: "
              f"{q['pooled']['mean']:+.4f} "
              f"[{q['pooled']['ci95'][0]:+.4f}, {q['pooled']['ci95'][1]:+.4f}]")
    if payload["rep1_restriction"].get("max_abs_drift") is not None:
        print(f"\nrep-1 vs reps-1-3 cell-mean drift, max "
              f"{payload['rep1_restriction']['max_abs_drift']:.4f}")


if __name__ == "__main__":
    main()
