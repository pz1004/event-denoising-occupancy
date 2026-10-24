"""Recompute secondary LED contrasts with equal scene weights throughout.

No event scores are changed. The input is each stored scene's mean valid-block
AUC. Point estimates, paired scene bootstrap intervals and sign-flip tests all
operate on the same vector of 66 scene differences. Common-event event-level
inference remains in Phase 74 and is the primary partition comparison.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent / "experiments/analytic_contrast"
DRAWS = 200_000
SEED = 20261022


def inference(differences: np.ndarray, seed: int, draws: int = DRAWS) -> dict:
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("finite paired scene differences required")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    exceedances = 0
    observed = float(d.mean())
    for start in range(0, draws, 5000):
        size = min(5000, draws - start)
        bootstrap[start:start + size] = d[rng.integers(len(d), size=(size, len(d)))].mean(axis=1)
    for start in range(0, draws, 5000):
        size = min(5000, draws - start)
        signs = rng.integers(0, 2, size=(size, len(d)), dtype=np.int8) * 2 - 1
        null = (signs * d).mean(axis=1)
        exceedances += int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return {
        "mean": observed,
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "bootstrap_se": float(bootstrap.std(ddof=1)),
        "p_two_sided": (exceedances + 1) / (draws + 1),
        "sign_flip_exceedances": exceedances,
        "draws": draws,
        "seed": seed,
        "scenes": len(d),
        "scenes_ahead": int(np.count_nonzero(d > 0)),
        "scenes_behind": int(np.count_nonzero(d < 0)),
        "scenes_tied": int(np.count_nonzero(d == 0)),
    }


def main() -> None:
    sources = {}
    records = {}
    tasks = []
    for filename in ("phase66_time_budget.json", "phase67_matched_causal_led.json",
                     "phase72_led_beta0_magnitude.json"):
        path = ARTIFACTS / filename
        raw = path.read_bytes()
        sources[filename] = hashlib.sha256(raw).hexdigest()
        source = json.loads(raw)
        if filename.startswith("phase66"):
            for partition, scenes in sorted(source["per_scene"].items()):
                for first, second in (("support", "ebf"), ("support", "magnitude"),
                                      ("support", "support_causal"),
                                      ("support_causal", "ebf")):
                    tasks.append((f"phase66/{partition}/{first}_minus_{second}", scenes, first, second))
        elif filename.startswith("phase67"):
            for first, second in (("occupancy_w32ms", "ebf"), ("occupancy_wTstar", "ebf"),
                                  ("occupancy_w32ms", "occupancy_wTstar")):
                tasks.append((f"phase67/{first}_minus_{second}", source["per_scene"], first, second))
        else:
            for first, second in (("beta0_support", "beta0_magnitude"),
                                  ("support", "beta0_support"),
                                  ("beta1_support", "beta0_support"),
                                  ("support", "magnitude")):
                tasks.append((f"phase72/{first}_minus_{second}", source["per_scene"], first, second))
    for index, (name, scenes, first, second) in enumerate(tasks):
        differences = {s: float(r[first]["block_mean_auc"] - r[second]["block_mean_auc"])
                       for s, r in sorted(scenes.items())}
        result = inference(np.asarray(list(differences.values())), SEED + index)
        result["per_scene_difference"] = differences
        records[name] = result
        print(f"{name}: {result['mean']:+.6f} {result['ci95']} p={result['p_two_sided']:.8g}")
    # Degenerate differences have a zero interval and p=1, not a spuriously small p.
    check = inference(np.zeros(4), SEED, draws=100)
    assert check["mean"] == 0 and check["ci95"] == [0, 0] and check["p_two_sided"] == 1
    payload = {
        "schema_version": 1,
        "estimand": "equal-scene mean of paired within-scene mean-valid-block ROC-AUC differences",
        "metric": "noise-positive ROC-AUC",
        "interval": "paired scene percentile bootstrap, 200000 draws",
        "test": "two-sided scene sign-flip, 200000 draws, plus-one Monte Carlo correction",
        "scope": "secondary partition-specific event sets; common-event primary analysis is Phase 74",
        "source_sha256": sources,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__},
        "contrasts": records,
    }
    output = ARTIFACTS / "phase76_led_equal_scene_statistics.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
