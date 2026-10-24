"""Phase 43: memory and data-movement accounting for state-based controls.

A parameter-free filter still stores and accesses per-pixel state. This ledger
accounts for that state and traffic separately from arithmetic operations and
learned weights, including sensor geometries such as 1280 by 720 pixels.

For each geometry and configuration, compute:

  * persistent state bytes by array and direction;
  * transient block-buffer bytes as implemented and in a packed layout;
  * memory reads and writes per event, at worst-case and measured ring occupancy;
  * stored bits per variable, distinct from the fixed-point certificate.

Storage and traffic follow ``StateConfig`` and ``NativeStateBank``. Ring occupancy
is measured from the extractor's ``dir_support`` operands. Nothing is trained or
fitted, and no configuration is changed.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .core import NativeStateBank, StateConfig, bidirectional_features
from .evaluate_adaptive_beta import dnd21_blocks, dvsclean_blocks
from .evaluate_dnd21 import PAPER_AUC as DND21_RATES
from .evaluate_dnd21 import SCENE_FILES
from .evaluate_dvsclean_analytic import SEQUENCES
from .phase12_efficiency import measure_occupancy as phase12_occupancy
from .phase14_quantizable_operand import BLOCK_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Implemented event layout: four int64 columns, as the extractor consumes them.
IMPLEMENTED_EVENT_BYTES = 4 * 8
#: A packed deployment layout: 32-bit microsecond timestamp, 10-bit x, 9-bit y, 1-bit p.
PACKED_EVENT_BITS = 32 + 10 + 9 + 1


def state_breakdown(config: StateConfig, directions: int = 2) -> Dict[str, int]:
    """Exact persistent state, from the same arrays the implementation allocates."""

    bank = NativeStateBank(config)
    one_direction = {
        "decayed_states_bytes": int(bank.states.nbytes),
        "last_timestamps_bytes": int(bank.last_times.nbytes),
        "global_state_bytes": int(bank.global_states.nbytes + bank.global_last_time.nbytes),
    }
    total = sum(one_direction.values())
    return {
        **{k: v * directions for k, v in one_direction.items()},
        "directions": directions,
        "total_bytes": total * directions,
        "total_mb": total * directions / 1e6,
        "bytes_per_pixel_per_polarity_per_direction": total / (2 * config.width * config.height),
    }


def traffic_per_event(config: StateConfig, occupancy: float,
                      directions: int = 2) -> Dict[str, float]:
    """Reads and writes per event, worst case and at measured ring occupancy.

    Per direction an event reads its own centre state and timestamp, the global state and
    timestamp, and one state plus one timestamp for each ring tap it visits; it writes its
    own centre state and timestamp. The neighbour body is skipped for taps that have never
    fired, which is what ``occupancy`` measures.
    """

    taps = (2 * config.radius + 1) ** 2 - 1
    scales = len(config.taus_us)
    centre_reads = 2 * scales + 1          # state per scale, plus the last timestamp
    global_reads = scales + 1
    per_tap_reads = scales + 1
    writes = scales + 1

    worst = centre_reads + global_reads + taps * per_tap_reads
    effective = centre_reads + global_reads + taps * occupancy * per_tap_reads
    return {
        "ring_taps": taps,
        "measured_ring_occupancy": occupancy,
        "reads_per_event_worst_case": worst * directions,
        "reads_per_event_at_occupancy": effective * directions,
        "writes_per_event": writes * directions,
        "bytes_read_per_event_at_occupancy": effective * directions * 6.0,
        "note": ("6 bytes is the mean of the two stored widths, float32 state and int64 "
                 "timestamp, weighted by how often each is touched"),
    }


def buffer_bytes(block: int) -> Dict[str, float]:
    packed = block * PACKED_EVENT_BITS / 8.0
    return {
        "events": block,
        "implemented_bytes": block * IMPLEMENTED_EVENT_BYTES,
        "implemented_mb": block * IMPLEMENTED_EVENT_BYTES / 1e6,
        "packed_bits_per_event": PACKED_EVENT_BITS,
        "packed_bytes": packed,
        "packed_mb": packed / 1e6,
    }


def measure_occupancy(block_iter, config: StateConfig,
                      max_blocks: Optional[int]) -> Dict[str, float]:
    """Mean fraction of ring taps carrying history.

    This delegates to Phase 12's estimator rather than defining a second one, so the
    occupancy behind this memory table is the same number behind the operation count the
    manuscript already reports. Phase 12 averages over both polarities, which is the
    convention the quoted 0.16--0.21 range uses.
    """

    blocks = [events for events, _ in islice(block_iter, max_blocks)]
    if not blocks:
        return {"blocks": 0}
    report = phase12_occupancy(blocks, config)
    return {"blocks": len(blocks), **report}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnd21-root", type=Path,
                        default=PROJECT_ROOT / "ECCV2024_datasets/AUC_test")
    parser.add_argument("--dvsclean-root", type=Path,
                        default=PROJECT_ROOT / "DVSCLEAN/simulated_data")
    parser.add_argument("--block", type=int, default=BLOCK_SIZE)
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT
                        / "native3d_ssm/experiments/analytic_contrast/phase43_memory_accounting.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    geometries = {"DND21 346x260": (346, 260), "DVSCLEAN 1280x720": (1280, 720)}
    # The occupancy behind the manuscript's operation count lives in Phase 12's artefact.
    # Read it rather than re-sample it: a fresh small sample would put a second, slightly
    # different number next to the published one for no gain.
    phase12_path = args.output.parent / "phase12_efficiency.json"
    published_occupancy: Dict[str, Dict[int, float]] = {}
    if phase12_path.exists():
        phase12 = json.loads(phase12_path.read_text())["results"]
        for key, name in (("DND21_346x260", "DND21 346x260"),
                          ("DVSCLEAN_1280x720", "DVSCLEAN 1280x720")):
            if key in phase12:
                published_occupancy[name] = {
                    r: float(phase12[key][f"radius_{r}"]["occupancy"]["mean_occupancy"])
                    for r in (1, 2)
                }
    # Occupancy is a property of the ring, so it is measured separately per radius: the
    # radius-1 ring sits closer to the centre and is occupied more often than radius 2.
    occupancy: Dict[str, Dict[int, Dict[str, float]]] = {name: {} for name in geometries}

    for radius in (1, 2):
        dnd_config = StateConfig(width=346, height=260, taus_us=(250.0,), radius=radius)
        values: List[float] = []
        for rate in DND21_RATES:
            for scene, filename in SCENE_FILES.items():
                path = args.dnd21_root / rate / filename
                if not path.exists():
                    continue
                report = measure_occupancy(
                    dnd21_blocks(path, args.block, args.max_blocks), dnd_config, args.max_blocks
                )
                if report.get("blocks"):
                    values.append(report["mean_occupancy"])
        occupancy["DND21 346x260"][radius] = {
            "conditions": len(values),
            "mean_occupancy": float(np.mean(values)) if values else None,
            "min_occupancy": float(np.min(values)) if values else None,
            "max_occupancy": float(np.max(values)) if values else None,
        }

        dvs_config = StateConfig(width=1280, height=720, taus_us=(250.0,), radius=radius)
        values = []
        for sequence in SEQUENCES:
            for level in (50, 100):
                path = args.dvsclean_root / f"{sequence}_{level}.hdf5"
                if not path.exists():
                    continue
                report = measure_occupancy(
                    dvsclean_blocks(path, args.block, args.max_blocks), dvs_config, args.max_blocks
                )
                if report.get("blocks"):
                    values.append(report["mean_occupancy"])
        occupancy["DVSCLEAN 1280x720"][radius] = {
            "conditions": len(values),
            "mean_occupancy": float(np.mean(values)) if values else None,
            "min_occupancy": float(np.min(values)) if values else None,
            "max_occupancy": float(np.max(values)) if values else None,
        }

    table: Dict[str, Dict] = {}
    for name, (width, height) in geometries.items():
        entry: Dict[str, Dict] = {}
        for radius in (1, 2):
            occ = published_occupancy.get(name, {}).get(
                radius, occupancy[name][radius]["mean_occupancy"]
            )
            config = StateConfig(width=width, height=height, taus_us=(250.0,), radius=radius)
            entry[f"radius{radius}"] = {
                "bidirectional": {
                    "state": state_breakdown(config, directions=2),
                    "traffic": traffic_per_event(config, occ if occ else 0.0, directions=2),
                },
                "causal": {
                    "state": state_breakdown(config, directions=1),
                    "traffic": traffic_per_event(config, occ if occ else 0.0, directions=1),
                },
            }
        entry["block_buffer"] = buffer_bytes(args.block)
        entry["measured_occupancy"] = {
            "used_for_traffic": published_occupancy.get(name),
            "source": ("phase12_efficiency.json, the same estimate behind the manuscript's "
                       "operation count" if name in published_occupancy
                       else "re-measured here; phase12 artefact absent"),
            "resampled_here": {f"radius{r}": occupancy[name][r] for r in (1, 2)},
        }
        table[name] = entry

    payload = {
        "phase": "43 end-to-end memory and data-movement accounting",
        "question": ("Zero stored weights is half a hardware ledger. What is the persistent "
                     "state, the transient buffer, and the per-event memory traffic?"),
        "stored_widths": {
            "decayed_state": "float32 (32 bit) per pixel per polarity per scale",
            "last_timestamp": "int64 (64 bit) per pixel per polarity",
            "note": ("These are what the implementation stores. The fixed-point certificate "
                     "covers the score operands and the gate decision, not these arrays; "
                     "narrowing them is untested and is not claimed."),
        },
        "state_is_independent_of_radius": True,
        "table": table,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["table"], indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
