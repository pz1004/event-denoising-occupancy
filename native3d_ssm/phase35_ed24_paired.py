"""Phase 35: pair our ED24 rows against EDformer's on the blocks both sides scored.

Phase 32b and Phase 33 score the same ED24 conditions but do not cover them identically, and the
reason is EDformer's protocol rather than ours. Its released `eval_auc.py` truncates each file to a
multiple of its 4,096-event slice before scoring, so on a minority of conditions its coverage stops
one 10,240-event block short of ours. Comparing a 10-block mean against a 9-block mean would let
that truncation into the margin; with a median of 9 blocks per condition, one block is over ten per
cent of a condition mean, which is well above the precision this paper quotes.

Both sides are therefore restricted here to the blocks BOTH scored. Phase 32b emits `block_indices`
because single-class blocks are skipped, so the scored blocks are not `range(n)` and pairing on
position rather than on index would misalign every block after the first gap.

`coverage` reports how many conditions the restriction touched and what it moved, so the correction
is auditable rather than asserted. Read-only; consumes two artifacts and writes a third.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "native3d_ssm/experiments/analytic_contrast"
OUR_BLOCK = 10_240
EDFORMER_SLICE = 4_096


def edformer_block_ceiling(events_scored: int) -> int:
    """How many complete 10,240-event blocks lie inside EDformer's scored prefix."""

    return events_scored // OUR_BLOCK


def exact_randomisation_p(deltas: np.ndarray, limit: int = 20) -> Optional[float]:
    """Two-sided sign-flip p-value; exact when the sample is small enough to enumerate."""

    n = len(deltas)
    if n == 0 or n > limit:
        return None
    observed = abs(deltas.mean())
    hits = 0
    for mask in range(1 << n):
        signs = np.array([1.0 if mask >> i & 1 else -1.0 for i in range(n)])
        hits += abs((deltas * signs).mean()) >= observed - 1e-12
    return hits / (1 << n)


def bootstrap_ci(values: np.ndarray, replicates: int = 10_000,
                 seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path,
                        default=ARTIFACTS / "phase32b_ed24_per_block.json")
    parser.add_argument("--theirs", type=Path,
                        default=ARTIFACTS / "phase33_ed24_baselines.json")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "phase35_ed24_paired.json")
    args = parser.parse_args()

    ours = json.loads(args.ours.read_text())
    theirs = json.loads(args.theirs.read_text())
    levels = ours["levels"]

    conditions: Dict[str, Dict] = {}
    restricted = 0
    blocks_dropped = 0

    for key, our in ours["conditions"].items():
        their = theirs["conditions"].get(key)
        if their is None:
            continue
        ceiling = edformer_block_ceiling(their["events"])
        keep = [i for i, index in enumerate(our["block_indices"]) if index < ceiling]
        if not keep:
            continue
        if len(keep) != len(our["block_indices"]):
            restricted += 1
            blocks_dropped += len(our["block_indices"]) - len(keep)

        adopted = float(np.mean([our["per_block"]["adopted"][i] for i in keep]))
        magnitude = float(np.mean([our["per_block"]["magnitude"][i] for i in keep]))
        conditions[key] = {
            "level": key.rsplit("_", 1)[1],
            "blocks_paired": len(keep),
            "blocks_ours_unrestricted": len(our["block_indices"]),
            "ours_adopted": adopted,
            "ours_magnitude": magnitude,
            "ours_adopted_unrestricted": our["block_mean"]["adopted"],
            "edformer": their["block_mean_10240"],
            "margin": adopted - their["block_mean_10240"],
        }

    def aggregate(subset: Dict[str, Dict]) -> Dict:
        margins = np.array([c["margin"] for c in subset.values()])
        lo, hi = bootstrap_ci(margins)
        return {
            "conditions": len(subset),
            "ours": float(np.mean([c["ours_adopted"] for c in subset.values()])),
            "edformer": float(np.mean([c["edformer"] for c in subset.values()])),
            "margin": float(margins.mean()),
            "ci95": [lo, hi],
            "conditions_ours_ahead": int((margins > 0).sum()),
        }

    # Derived here rather than read from `theirs`, because a run launched before the metadata fix
    # records the module's default level list instead of the one it scored. The authority is
    # EDformer's own training suffix range against the levels actually present in `ours`.
    train_suffixes = {f"{v / 10:.1f}" for v in range(15, 36)}
    overlap = {
        "levels_in_training": [l for l in levels if l in train_suffixes],
        "levels_outside_training": [l for l in levels if l not in train_suffixes],
        "recorded_by_run": theirs.get("training_overlap", {}),
    }
    per_level = {}
    for level in levels:
        subset = {k: v for k, v in conditions.items() if v["level"] == level}
        if subset:
            per_level[level] = {**aggregate(subset),
                                "in_edformer_training": level in overlap["levels_in_training"]}

    seen = {k: v for k, v in conditions.items()
            if v["level"] in overlap["levels_in_training"]}
    unseen = {k: v for k, v in conditions.items()
              if v["level"] in overlap["levels_outside_training"]}

    payload = {
        "phase": "35 ED24 paired on blocks both sides scored",
        "sources": {"ours": args.ours.name, "theirs": args.theirs.name},
        "coverage": {
            "conditions_paired": len(conditions),
            "conditions_restricted": restricted,
            "blocks_dropped": blocks_dropped,
            "note": ("EDformer truncates each file to a multiple of 4,096 events before scoring; "
                     "these are the conditions where that left it one 10,240-block short of ours"),
            "our_pooled_unrestricted": float(np.mean(
                [c["ours_adopted_unrestricted"] for c in conditions.values()])),
            "our_pooled_restricted": float(np.mean(
                [c["ours_adopted"] for c in conditions.values()])),
        },
        "pooled": aggregate(conditions),
        "per_level": per_level,
        "training_overlap": overlap,
        "by_training_overlap": {
            "levels_edformer_trained_on": {**aggregate(seen), "levels":
                                           overlap["levels_in_training"]} if seen else None,
            "levels_edformer_never_saw": {**aggregate(unseen), "levels":
                                          overlap["levels_outside_training"]} if unseen else None,
        },
        "conditions": conditions,
    }
    payload["coverage"]["correction_size"] = (payload["coverage"]["our_pooled_restricted"]
                                              - payload["coverage"]["our_pooled_unrestricted"])

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print(json.dumps({k: payload[k] for k in ("coverage", "pooled", "by_training_overlap")},
                     indent=2))


if __name__ == "__main__":
    main()
