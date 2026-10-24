#!/usr/bin/env python3
"""Reproduce EDformer's E-MLB MESR row (ECCV 2024, paper Table 3) with pretrained_model.pth.

Unlike EDmamba, EDformer's evaluation path is fully released and runnable: `eval_mesr.py`
plus a vendored `cuke-emlb` metric. This script therefore changes **nothing** about the
model, the metric, or the operating point. It imports EDformer's own `model.py` and
`metrics.py` unmodified and reuses the released inference recipe verbatim:

  * blocks of `seq_len = 4096` events;
  * timestamps min-max normalised **over the whole recording** before blocking (so a
    block spans only a sub-range of [0,1]) -- note this differs from Native3D/EDmamba,
    which normalise per block;
  * raw **pixel** x, y passed through unnormalised -- the convention `dataset.py` uses at
    training time too, so the checkpoint is matched to it;
  * retention `sigmoid(f) >= 0.005 -> noise`, the hard-coded operating point in
    `eval_mesr.py` (note: 0.005, not 0.5);
  * MESR via EDformer's `metrics.EventStructuralRatio.evalEventStorePerNumber`
    (30,000-event slices), whose `_calc_esr` is the same formula Phase 5 verified
    bit-exact against official `cuke-emlb`.

Two additions, neither of which touches the model:

1. **Aggregation by ND level.** The released script groups by scene (`fclass`) and so
   cannot produce the paper's ND1/ND4/ND16/ND64 columns. We group by the `-NDxx-` field
   of the filename, which is what Table 3 reports.
2. **A Raw row per cell**, computed with EDformer's own metric on the unfiltered store.
   Raw is the cross-paper anchor: EDformer Table 3 and EDmamba Table 2 print the
   identical Raw row, and our numpy pipeline already reproduces it to 3 dp.

Reading is vectorised (`batch.numpy()` concatenation) instead of the released
`for i, e in enumerate(events)` Python loop, which would need ~345 M interpreter
iterations for the full protocol. `--verify-reader` asserts the two agree exactly on one
sequence before trusting the fast path.

Inference runs one block per forward pass (`--batch-blocks 1`), exactly as released.
Batching was tried and rejected on evidence: it is deterministic at a fixed batch size
but B=1 vs B=8 logits differ by up to 5.2e-06 (float32 reduction order), and because the
released threshold 0.005 is sharp with a dense score distribution around it, that flips
about one event per million. See the comment in `Inference.__init__`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDFORMER_ROOT = PROJECT_ROOT / "EDformer"
sys.path.insert(0, str(EDFORMER_ROOT))

TIMESTAMP_COLUMN = 0
X_COLUMN = 1
Y_COLUMN = 2
POLARITY_COLUMN = 3

ND_LEVELS = ("ND00", "ND04", "ND16", "ND64")
PARTS = ("D-END", "N-END")
SEQ_LEN = 4096
NOISE_THRESHOLD = 0.005          # as released in eval_mesr.py
SLICE = 30_000

# EDformer paper Table 3 (E-MLB columns) and the shared Raw row.
PAPER_RAW = {"D-END": {"ND00": 0.821, "ND04": 0.824, "ND16": 0.815, "ND64": 0.786},
             "N-END": {"ND00": 0.890, "ND04": 0.824, "ND16": 0.786, "ND64": 0.768}}
PAPER_EDFORMER = {"D-END": {"ND00": 0.952, "ND04": 0.955, "ND16": 0.956, "ND64": 0.942},
                  "N-END": {"ND00": 1.048, "ND04": 1.019, "ND16": 1.076, "ND64": 1.099}}


def setup_seed(seed: int) -> None:
    """Identical to eval_mesr.setup_seed (default seed there is 230086)."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_store(path: Path, max_events: Optional[int]):
    """Read an .aedat4 into a dv.EventStore, optionally truncating to a prefix."""

    import dv_processing as dv

    reader = dv.io.MonoCameraRecording(str(path))
    events = dv.EventStore()
    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is not None:
            events.add(batch)
            if max_events is not None and events.size() >= max_events:
                break
    if max_events is not None and events.size() > max_events:
        events = events.slice(0, max_events)
    return events, reader.getEventResolution()


def store_to_array(events) -> np.ndarray:
    """Vectorised equivalent of eval_mesr's per-event loop: [t, x, y, p] float64."""

    packet = events.numpy()
    array = np.empty((len(packet), 4), dtype=np.float64)
    array[:, 0] = packet["timestamp"]
    array[:, 1] = packet["x"]
    array[:, 2] = packet["y"]
    array[:, 3] = packet["polarity"]
    return array


def store_to_array_reference(events) -> np.ndarray:
    """The released loop, kept only to verify `store_to_array` agrees exactly."""

    array = np.empty((len(events), 4))
    for i, e in enumerate(events):
        array[i, 0] = e.timestamp()
        array[i, 1] = e.x()
        array[i, 2] = e.y()
        array[i, 3] = e.polarity()
    return array


def normalize_column(column: np.ndarray) -> np.ndarray:
    return (column - np.min(column)) / (np.max(column) - np.min(column))


class Inference:
    """EDformer's released inference recipe, ND-agnostic and unmodified in substance."""

    def __init__(self, model, seq_len: int = SEQ_LEN,
                 threshold: float = NOISE_THRESHOLD, batch_blocks: int = 1) -> None:
        self.model = model
        self.seq_len = seq_len
        self.threshold = threshold
        # MEASURED, do not "optimise" this back: grouping blocks into one forward pass is
        # NOT exactly equivalent. The model is deterministic at a fixed batch size
        # (B=1 twice -> maxdiff 0.0), but B=1 vs B=8 logits differ by up to 5.2e-06 from
        # float32 reduction-order differences in the batched kernels. EDformer's released
        # threshold is a very sharp 0.005 and its score distribution is dense there
        # (~17 events per 164k within 1e-6 of it), so that noise flips roughly one event
        # per million -- observed as 1139491 vs 1139492 kept events on
        # Architecture-ND00-1. Batching also bought almost nothing (8.1 s vs 8.8 s), so
        # the default is 1: the released path, bit-reproducible.
        self.batch_blocks = max(1, batch_blocks)

    def inference(self, events, event_array: np.ndarray):
        import dv_processing as dv

        result = dv.EventStore()
        min_t = float(np.min(event_array[:, TIMESTAMP_COLUMN]))
        max_t = float(np.max(event_array[:, TIMESTAMP_COLUMN]))
        prepared = np.hstack((
            normalize_column(event_array[:, TIMESTAMP_COLUMN]).reshape(-1, 1),
            event_array[:, X_COLUMN].reshape(-1, 1),
            event_array[:, Y_COLUMN].reshape(-1, 1),
            event_array[:, POLARITY_COLUMN].reshape(-1, 1)))

        num_samples = len(prepared) // self.seq_len
        if num_samples == 0:
            return result, 0
        blocks = prepared[:num_samples * self.seq_len].reshape(
            (num_samples, self.seq_len, 4))

        scale = max_t - min_t
        for start in range(0, num_samples, self.batch_blocks):
            group = blocks[start:start + self.batch_blocks]
            batch = torch.from_numpy(np.ascontiguousarray(group)).to(
                dtype=torch.float32).cuda()
            with torch.no_grad():
                logits = self.model(batch)
            keep = (torch.sigmoid(logits).squeeze(-1) < self.threshold)
            kept = batch[keep].cpu().numpy()          # row-major: block-major order
            timestamps = (kept[:, TIMESTAMP_COLUMN] * scale + min_t).astype(np.int64)
            xs = kept[:, X_COLUMN].astype(np.int64)
            ys = kept[:, Y_COLUMN].astype(np.int64)
            ps = kept[:, POLARITY_COLUMN].astype(bool)
            for i in range(len(kept)):
                result.push_back(int(timestamps[i]), int(xs[i]), int(ys[i]), bool(ps[i]))
        return result, num_samples * self.seq_len


def mean_mesr(metric, events) -> float:
    score = metric.evalEventStorePerNumber(events, SLICE)
    score = np.asarray(score, dtype=np.float64)
    usable = score[~np.isnan(score)]
    return float(np.mean(usable)) if usable.size else float("nan")


def nd_of(name: str) -> Optional[str]:
    match = re.search(r"-(ND\d{2})-", name)
    return match.group(1) if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emlb-root", type=Path, default=PROJECT_ROOT / "E-MLB")
    parser.add_argument("--model-path", type=Path,
                        default=EDFORMER_ROOT / "pretrained_model.pth")
    parser.add_argument("--max-events", type=int, default=300_000,
                        help="Per-sequence cap; 0 = uncapped. 300k matches the Native3D capped rows.")
    parser.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--nd-levels", nargs="+", default=list(ND_LEVELS))
    parser.add_argument("--threshold", type=float, default=NOISE_THRESHOLD)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=230086)
    parser.add_argument("--batch-blocks", type=int, default=1,
                        help="Blocks per forward pass. KEEP AT 1: >1 shifts logits by ~5e-6 "
                             "(float32 reduction order) and flips ~1 event per million across "
                             "the sharp 0.005 threshold. No meaningful speedup anyway.")
    parser.add_argument("--verify-batching", action="store_true",
                        help="Assert batched inference matches batch_blocks=1 on the first sequence.")
    parser.add_argument("--verify-reader", action="store_true",
                        help="Assert the vectorised reader equals the released loop on one sequence.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "reproduction/results/emlb_edformer_mesr.json")
    return parser.parse_args()


def build_model(model_path: Path) -> Tuple[torch.nn.Module, Dict]:
    from model import EDformer

    model = EDformer().cuda()
    state = torch.load(str(model_path), map_location="cuda:0")
    model.load_state_dict(state)          # strict, exactly as eval_mesr.py
    model.eval()
    return model, {
        "checkpoint": str(model_path.resolve()),
        "tensors": len(state),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters()
                                        if p.requires_grad)),
        "strict_load": True,
    }


def payload(args, results, metadata, started, repairs) -> Dict:
    return {
        "what": "EDformer (ECCV 2024) official checkpoint on E-MLB, MESR via its own vendored cuke-emlb metric",
        "released_path": ("EDformer/eval_mesr.py is complete and runnable; model.py and metrics.py "
                          "are imported unmodified. Only the aggregation (by ND level) and a Raw "
                          "row are added."),
        "protocol": {
            "seq_len": args.seq_len,
            "batch_blocks": args.batch_blocks,
            "noise_threshold": args.threshold,
            "retention_rule": f"keep iff sigmoid(f) < {args.threshold} (as released, not 0.5)",
            "timestamp_normalisation": "min-max over the whole recording, before blocking (as released)",
            "coordinates": "raw pixel x,y passed unnormalised (matches dataset.py training convention)",
            "metric": "metrics.EventStructuralRatio.evalEventStorePerNumber, 30,000-event slices",
            "max_events_per_sequence": args.max_events,
            "reps": args.reps,
            "seed": args.seed,
        },
        "model": metadata,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "paper_reference": {"raw": PAPER_RAW, "edformer": PAPER_EDFORMER},
        "results": results,
        "reader_repairs": repairs,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    torch.cuda.empty_cache()
    started = time.perf_counter()

    from metrics import EventStructuralRatio

    model, metadata = build_model(args.model_path)
    print(f"loaded EDformer: {metadata['tensors']} tensors, "
          f"{metadata['trainable_parameters']} params, strict load OK", flush=True)

    inference = Inference(model, args.seq_len, args.threshold, args.batch_blocks)
    reference_inference = Inference(model, args.seq_len, args.threshold, 1)
    batching_verified = not (args.verify_batching and args.batch_blocks > 1)
    cap: Optional[int] = args.max_events if args.max_events > 0 else None
    results: Dict[str, Dict[str, Dict]] = {}
    verified = not args.verify_reader

    for part in args.parts:
        root = args.emlb_root / part
        scenes = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
        results[part] = {}
        for nd in args.nd_levels:
            raw_scores, filtered_scores, retentions = [], [], []
            #: Per-sequence values, keyed by "{scene}-{nd}-{rep}" so they can be paired
            #: against our side sequence by sequence. The cell means below are unchanged;
            #: this only stops the per-sequence values being thrown away at nanmean, which
            #: is what previously forced every E-MLB interval down to n = 8 cells.
            per_sequence: Dict[str, Dict[str, float]] = {}
            for scene in scenes:
                for rep in args.reps:
                    path = root / scene / f"{scene}-{nd}-{rep}.aedat4"
                    if not path.exists():
                        continue
                    events, resolution = read_store(path, cap)
                    if events.size() < SLICE:
                        continue
                    array = store_to_array(events)
                    if not verified:
                        reference = store_to_array_reference(events)
                        assert np.array_equal(array, reference), \
                            "vectorised reader disagrees with the released loop"
                        print(f"reader verified exactly on {path.name} "
                              f"({len(array)} events)", flush=True)
                        verified = True
                    metric = EventStructuralRatio(resolution)
                    raw_scores.append(mean_mesr(metric, events))
                    filtered, considered = inference.inference(events, array)
                    if not batching_verified:
                        reference, _ = reference_inference.inference(events, array)
                        fa, ra = filtered.numpy(), reference.numpy()
                        assert len(fa) == len(ra) and all(
                            np.array_equal(fa[f], ra[f]) for f in fa.dtype.names), (
                            f"batched inference differs from batch_blocks=1 "
                            f"({len(fa)} vs {len(ra)} events)")
                        print(f"batching verified exactly on {path.name}: "
                              f"{len(fa)} kept events identical at batch={args.batch_blocks}",
                              flush=True)
                        batching_verified = True
                    if considered:
                        retentions.append(filtered.size() / considered)
                    filtered_scores.append(mean_mesr(metric, filtered))
                    per_sequence[f"{scene}-{nd}-{rep}"] = {
                        "raw_mesr": float(raw_scores[-1]),
                        "edformer_mesr": float(filtered_scores[-1]),
                        "retention": (float(filtered.size() / considered) if considered
                                      else float("nan")),
                    }
            cell = {
                "sequences": len(raw_scores),
                "per_sequence": per_sequence,
                "raw_mesr": float(np.nanmean(raw_scores)) if raw_scores else None,
                "edformer_mesr": float(np.nanmean(filtered_scores)) if filtered_scores else None,
                "mean_retention": float(np.mean(retentions)) if retentions else None,
                "std_retention": float(np.std(retentions)) if retentions else None,
                "paper_raw": PAPER_RAW[part][nd],
                "paper_edformer": PAPER_EDFORMER[part][nd],
            }
            results[part][nd] = cell
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(
                payload(args, results, metadata, started, 0), indent=2, sort_keys=True) + "\n")
            print(f"{part} {nd}: raw={cell['raw_mesr']:.4f} (paper {cell['paper_raw']:.3f}) "
                  f"edformer={cell['edformer_mesr']:.4f} (paper {cell['paper_edformer']:.3f}) "
                  f"retention={cell['mean_retention']:.4f} n={cell['sequences']}", flush=True)

    args.output.write_text(json.dumps(
        payload(args, results, metadata, started, 0), indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
