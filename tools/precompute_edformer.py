"""Compute and cache EDformer scores for one evaluation unit.

Prepare the unit's event stream, invoke the EDformer runner once, and save scores
at the cache path consumed by ``phase65_campus_baselines``. The cache supports
separate execution of the learned baseline and the paired comparison. EDformer
uses the same stream and checkpoint when scores are computed or reused.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from native3d_ssm.phase57_ebf_labelled import load_condition          # noqa: E402
from native3d_ssm.phase62_buaa_campus import (BUAA_ROOT, HEIGHT,      # noqa: E402
                                              WIDTH, load_sequence)
from native3d_ssm.phase65_campus_baselines import (CACHE, EDFORMER_PYTHON,  # noqa: E402
                                                   EDFORMER_RUNNER,
                                                   edformer_cache_path)

# Public-release adapter: the frozen experiment modules keep their original bytes.
EDFORMER_PYTHON = Path(os.environ.get("RSF_EDFORMER_PYTHON", str(EDFORMER_PYTHON)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True,
                        help="validation_1hz | validation_3hz | validation_5hz | campus_<seq>")
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = edformer_cache_path(args.unit)
    if out.is_file():
        print(f"{args.unit}: already cached ({out})")
        return

    if args.unit.startswith("validation_"):
        rate = args.unit.split("_", 1)[1]
        events, _labels = load_condition(
            "DND21", f"{rate}_driving",
            ROOT / "ECCV2024_datasets/AUC_test", ROOT / "DVSCLEAN/simulated_data")
        width, height = 346, 260
    elif args.unit.startswith("campus_"):
        name = args.unit.split("_", 1)[1]
        events, _labels, _diag = load_sequence(BUAA_ROOT / f"{name}.npy", True, args.seed)
        width, height = WIDTH, HEIGHT
    else:
        raise SystemExit(f"unrecognised unit: {args.unit}")

    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npy"
        np.savez(src, events=events, width=width, height=height)
        result = subprocess.run(
            [str(EDFORMER_PYTHON), str(EDFORMER_RUNNER),
             "--input", str(src), "--output", str(dst)],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"EDformer runner failed:\n{result.stdout}\n{result.stderr}")
        np.save(out, np.load(dst))
    print(f"{args.unit}: cached {out} ({len(np.load(out))} events)")


if __name__ == "__main__":
    main()
