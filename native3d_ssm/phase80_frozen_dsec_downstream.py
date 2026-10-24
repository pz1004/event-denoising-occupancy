"""Prospectively frozen DSEC downstream transfer; no event-noise labels.

Raw unrectified events are filtered continuously before rectification and the
official DSEC voxel transform. All public forward-flow pairs in the three frozen
recordings are evaluated without model fitting or target-dependent selection.
Model inference is deliberately standard/no warm-start, at full 480x640 size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA_ROOT = WORKSPACE / "downstream/assets/dsec_frozen_20261022"
ERAFT_ROOT = WORKSPACE / "external/E-RAFT"
CHECKPOINT = WORKSPACE / "downstream/assets/eraft/mvsec_20.tar"
CHECKPOINT_SHA256 = "d9845c0b85a1e5b2d5b664f27706e9a86270afcff42cb16457aeb9c2cbdb9ba2"
PROTOCOL = ROOT / "plan/protocol_frozen_dsec_downstream_20261022.md"
PROTOCOL_SHA256 = "9ecc9eb38a546f6a0490f3758a2602419422725238495c0eb81005fdf6624dd3"
COHORT = ("thun_00_a", "zurich_city_02_a", "zurich_city_11_a")
EXPECTED_PAIRS = (41, 64, 231)
ARMS = ("raw", "occupancy", "multiplicity", "ebf", "random")
SEED = 20261022
HEIGHT, WIDTH, BINS, DELTA_US = 480, 640, 15, 100_000
OUTPUT = ROOT / "native3d_ssm/experiments/analytic_contrast/phase80_frozen_dsec_downstream.json"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values):
    a = np.asarray(values)
    digest = hashlib.sha256((str(a.dtype) + str(a.shape)).encode())
    flat = a.reshape(-1)
    for start in range(0, len(flat), 1_000_000):
        digest.update(np.ascontiguousarray(flat[start:start + 1_000_000]).tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def scalar_default(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"unsupported JSON value: {type(item).__name__}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=scalar_default) + "\n")
    temporary.replace(path)


def assert_freeze():
    if file_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen protocol changed: explicitly amend before outcome access")
    if file_hash(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("unexpected downstream checkpoint")


def download_inputs(data_root):
    """Retrieve the prospectively selected official files, with an integrity manifest."""
    from concurrent.futures import ThreadPoolExecutor
    import urllib.request
    names = ("events_left.zip", "optical_flow_forward_event.zip",
             "optical_flow_forward_timestamps.txt", "calibration.zip", "image_timestamps.txt")
    data_root.mkdir(parents=True, exist_ok=True)

    def retrieve(item):
        sequence, name = item
        destination = data_root / sequence / name
        destination.parent.mkdir(exist_ok=True)
        url = f"https://download.ifi.uzh.ch/rpg/DSEC/train/{sequence}/{sequence}_{name}"
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    stream.write(chunk)
            temporary.replace(destination)
        return {"sequence": sequence, "name": name, "url": url,
                "bytes": destination.stat().st_size, "sha256": file_hash(destination)}

    with ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(retrieve, [(s, n) for s in COHORT for n in names]))
    write_json(data_root / "download_manifest.json",
               {"retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": records})


def decode_flow_array(rgb):
    """DSEC's explicit validity channel includes valid zero-displacement pixels."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("DSEC flow requires an RGB uint16 HxWx3 image")
    if not np.isin(rgb[..., 2], (0, 1)).all():
        raise ValueError("invalid DSEC validity channel")
    valid = rgb[..., 2] == 1
    flow = (rgb[..., :2].astype(np.float32) - 32768.0) / 128.0
    flow[~valid] = 0
    return flow, valid


def pair_intervals(start_us, stop_us):
    """Use annotated current endpoints and the preceding nominal 100-ms interval."""
    start_us, stop_us = int(start_us), int(stop_us)
    if not 99_000 <= stop_us - start_us <= 101_000:
        raise ValueError("unexpected DSEC forward-flow time span")
    return ((start_us - DELTA_US, start_us), (start_us, stop_us))


def selected_interval(events, indices, start_us, stop_us):
    """Identical half-open original-time interval for every filtered condition."""
    start = int(np.searchsorted(events[:, 0], start_us, side="left"))
    stop = int(np.searchsorted(events[:, 0], stop_us, side="left"))
    if indices is None:
        return np.asarray(events[start:stop])
    a, b = np.searchsorted(indices, (start, stop), side="left")
    return np.asarray(events[indices[a:b]])


def metric_record(prediction, ground_truth, valid):
    """The mask depends solely on released flow validity, never event retention."""
    if prediction.shape != ground_truth.shape or ground_truth.shape[:2] != valid.shape:
        raise ValueError("prediction, target and mask shapes disagree")
    if not np.isfinite(prediction).all() or not np.isfinite(ground_truth).all():
        raise ValueError("nonfinite prediction or target")
    errors = np.linalg.norm(prediction.astype(np.float64) - ground_truth, axis=-1)[valid]
    if not len(errors):
        raise ValueError("no valid flow pixels")
    return {"valid_pixels": int(len(errors)), "epe_sum": float(errors.sum()),
            "epe": float(errors.mean()), "outlier_1px": int((errors > 1).sum()),
            "outlier_3px": int((errors > 3).sum()),
            "prediction_sha256": array_hash(prediction)}


def aggregate_records(records):
    total = sum(r["valid_pixels"] for r in records)
    if total <= 0:
        raise ValueError("no valid target pixels")
    return {"frames": len(records), "valid_pixels": total,
            "epe": sum(r["epe_sum"] for r in records) / total,
            "one_pixel_percent": 100 * sum(r["outlier_1px"] for r in records) / total,
            "three_pixel_percent": 100 * sum(r["outlier_3px"] for r in records) / total}


def ensure_extracted(sequence_root):
    """Extract only downloaded official archives inside their dedicated input tree."""
    for archive_name, folder in (("events_left.zip", "events_left"),
                                 ("optical_flow_forward_event.zip", "flow_forward")):
        destination = sequence_root / folder
        if (destination / ".complete").exists():
            continue
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(sequence_root / archive_name) as archive:
            for name in archive.namelist():
                candidate = (destination / name).resolve()
                if not candidate.is_relative_to(destination.resolve()):
                    raise ValueError("archive path escapes its extraction directory")
            archive.extractall(destination)
        (destination / ".complete").write_text("extracted\n")


def one_file(root, name):
    found = list(root.rglob(name))
    if len(found) != 1:
        raise ValueError(f"expected exactly one {name} below {root}, got {found}")
    return found[0]


def prepare_raw(sequence_root):
    import h5py
    import hdf5plugin  # Registers the official release's compressed HDF5 filters.
    from .phase79_tstar_cost_quality import validate_events
    ensure_extracted(sequence_root)
    prepared = sequence_root / "prepared"
    prepared.mkdir(exist_ok=True)
    raw_path = prepared / "raw_events.npy"
    metadata_path = prepared / "raw_metadata.json"
    if raw_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        cached = np.load(raw_path, mmap_mode="r")
        if array_hash(cached) != metadata["raw_payload_sha256"]:
            raise ValueError("cached raw source payload changed")
        return raw_path, metadata
    source = one_file(sequence_root / "events_left", "events.h5")
    with h5py.File(source, "r") as h5:
        count = len(h5["events/t"])
        offset = int(h5["t_offset"][()])
        events = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.int64,
                                          shape=(count, 4))
        for start in range(0, count, 1_000_000):
            stop = min(start + 1_000_000, count)
            events[start:stop, 0] = h5["events/t"][start:stop].astype(np.int64) + offset
            for column, key in enumerate(("x", "y", "p"), 1):
                events[start:stop, column] = h5["events/" + key][start:stop]
        events.flush()
    validate_events(events, WIDTH, HEIGHT)
    metadata = {"event_count": count, "t_offset_us": offset,
                "first_event_us": int(events[0, 0]), "last_event_us": int(events[-1, 0]),
                "raw_payload_sha256": array_hash(events), "raw_h5_sha256": file_hash(source),
                "coordinates": "original distorted integer sensor pixels; no crop or rectification"}
    write_json(metadata_path, metadata)
    return raw_path, metadata


def prepare_sequence(data_root, sequence, sequence_index):
    """Read no flow targets: score all original events before any task evaluation."""
    from .phase79_tstar_cost_quality import iter_filtered_windows, window_bounds
    raw_path, raw_metadata = prepare_raw(data_root / sequence)
    events = np.load(raw_path, mmap_mode="r")
    prepared = raw_path.parent
    windows = [(int(a), int(b), int(c)) for a, b, c in window_bounds(events)]
    total_keep = sum((b - a) * 4 // 5 for a, b, _ in windows)
    window_table = [{"start": a, "stop": b, "close_us": c,
                     "quota": (b - a) * 4 // 5} for a, b, c in windows]
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(len(COHORT))[sequence_index])
    results = {}
    for arm in ARMS[1:]:
        print(f"prepare {sequence} {arm} ({len(events)} raw events)", flush=True)
        indices_path = prepared / f"{arm}_indices.npy"
        index_array = np.lib.format.open_memmap(indices_path, mode="w+", dtype=np.int64,
                                              shape=(total_keep,))
        offset = 0
        started = time.perf_counter()
        if arm == "random":
            for (a, b, _), row in zip(windows, window_table):
                indices = np.sort(rng.choice(b - a, size=row["quota"], replace=False)) + a
                index_array[offset:offset + len(indices)] = indices
                offset += len(indices)
        else:
            for output, row in zip(iter_filtered_windows(events, WIDTH, HEIGHT, method=arm), window_table):
                if (output.start, output.stop, output.close_us) != (row["start"], row["stop"], row["close_us"]):
                    raise AssertionError("window boundary drift")
                if len(output.indices) != row["quota"]:
                    raise AssertionError("incorrect window quota")
                if not np.array_equal(output.emitted, events[output.indices]):
                    raise AssertionError("filter changed original payloads or output order")
                index_array[offset:offset + len(output.indices)] = output.indices
                offset += len(output.indices)
                del output
        elapsed = time.perf_counter() - started
        if offset != total_keep or np.any(np.diff(index_array) <= 0):
            raise AssertionError("incorrect retained index count/order")
        index_array.flush()
        results[arm] = {"retained_events": offset, "retained_fraction": offset / len(events),
                        "selected_indices_sha256": array_hash(index_array),
                        "index_file_sha256": file_hash(indices_path),
                        "filter_seconds_including_harness_writes": elapsed}
        del index_array
    metadata = {"sequence": sequence, "protocol_sha256": PROTOCOL_SHA256,
                "source": raw_metadata, "windows": window_table,
                "nonempty_windows": len(windows), "sum_exact_quotas": total_keep,
                "filtered": results, "random_seed": SEED, "random_sequence_spawn_key": [sequence_index],
                "timing_scope": "preparation wall time including index harness writes; not an isolated throughput benchmark",
                "phase79_sha256": file_hash(ROOT / "native3d_ssm/phase79_tstar_cost_quality.py")}
    write_json(prepared / "filter_manifest.json", metadata)
    return metadata


def load_model(device):
    import torch
    sys.path.insert(0, str(ERAFT_ROOT))
    from model.eraft import ERAFT
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in checkpoint["model"].items()}
    model = ERAFT({"subtype": "standard"}, n_first_channels=BINS)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def official_voxel(events, rectify_map):
    """Mirror Sequence.events_to_voxel_grid and VoxelGrid.convert on CPU.

    Endpoints are each retained interval's first/last event, as in upstream.
    Subpixel rectification is preserved; out-of-bounds splat contributions are
    rejected by the upstream transform, without changing event-level selection.
    Empty/one-timestamp inputs deterministically give zero tensors instead of NaN.
    """
    import torch
    sys.path.insert(0, str(ERAFT_ROOT))
    from utils.dsec_utils import VoxelGrid
    if len(events) < 2 or events[-1, 0] == events[0, 0]:
        return torch.zeros((BINS, HEIGHT, WIDTH), dtype=torch.float32)
    times = (events[:, 0] - events[0, 0]).astype(np.float32)
    times = times / times[-1]
    rectified = rectify_map[events[:, 2], events[:, 1]]
    payload = {"t": torch.from_numpy(times),
               "p": torch.from_numpy(events[:, 3].astype(np.float32)),
               "x": torch.from_numpy(rectified[:, 0].astype(np.float32)),
               "y": torch.from_numpy(rectified[:, 1].astype(np.float32))}
    return VoxelGrid((BINS, HEIGHT, WIDTH), normalize=True).convert(payload)


def evaluate_sequence(data_root, sequence, model, device):
    import h5py
    import png
    import torch
    sequence_root = data_root / sequence
    prepared = sequence_root / "prepared"
    manifest = json.loads((prepared / "filter_manifest.json").read_text())
    if manifest["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("filter manifest protocol mismatch")
    if manifest["phase79_sha256"] != file_hash(ROOT / "native3d_ssm/phase79_tstar_cost_quality.py"):
        raise ValueError("phase79 changed after downstream input preparation")
    events = np.load(prepared / "raw_events.npy", mmap_mode="r")
    if array_hash(events) != manifest["source"]["raw_payload_sha256"]:
        raise ValueError("raw source payload changed after filtering")
    selected = {"raw": None, **{a: np.load(prepared / f"{a}_indices.npy", mmap_mode="r") for a in ARMS[1:]}}
    for arm in ARMS[1:]:
        if array_hash(selected[arm]) != manifest["filtered"][arm]["selected_indices_sha256"]:
            raise ValueError("selected-index hash mismatch")
    with h5py.File(one_file(sequence_root / "events_left", "rectify_map.h5"), "r") as h5:
        rectify_map = h5["rectify_map"][:]
    if rectify_map.shape != (HEIGHT, WIDTH, 2) or not np.isfinite(rectify_map).all():
        raise ValueError("invalid subpixel rectification map")
    timestamps = np.loadtxt(sequence_root / "optical_flow_forward_timestamps.txt", delimiter=",", dtype=np.int64)
    files = sorted((sequence_root / "flow_forward").rglob("*.png"))
    if len(files) != len(timestamps):
        raise ValueError("flow image/timestamp count mismatch")
    if len(files) != EXPECTED_PAIRS[COHORT.index(sequence)]:
        raise ValueError("partial or unexpected public forward-flow cohort")
    image_timestamps = np.loadtxt(sequence_root / "image_timestamps.txt", dtype=np.int64)
    for pair, path in zip(timestamps, files):
        image_index = int(path.stem)
        if image_index + 2 >= len(image_timestamps) or not np.array_equal(
                pair, image_timestamps[[image_index, image_index + 2]]):
            raise ValueError("flow PNG identifier and annotated image timestamps disagree")
    records = {arm: [] for arm in ARMS}
    exclusions, pairs = [], []
    inference_seconds = dict.fromkeys(ARMS, 0.0)
    voxel_seconds = dict.fromkeys(ARMS, 0.0)
    for pair_index, ((start_us, stop_us), flow_path) in enumerate(zip(timestamps, files)):
        intervals = pair_intervals(start_us, stop_us)
        if intervals[0][0] < events[0, 0] or intervals[-1][1] > events[-1, 0]:
            exclusions.append({"pair_index": pair_index, "reason": "raw time coverage", "intervals": intervals})
            continue
        image_width, image_height, rows, png_info = png.Reader(filename=str(flow_path)).read()
        if png_info["bitdepth"] != 16 or png_info["planes"] != 3:
            raise ValueError("released flow PNG must be 16-bit RGB")
        encoded_rgb = np.asarray(list(rows), dtype=np.uint16).reshape(image_height, image_width, 3)
        flow, valid = decode_flow_array(encoded_rgb)
        if flow.shape != (HEIGHT, WIDTH, 2):
            raise ValueError("flow geometry mismatch")
        if not valid.any():
            exclusions.append({"pair_index": pair_index, "reason": "no released valid flow pixels"})
            continue
        pair = {"pair_index": pair_index, "flow_file": flow_path.name,
                "start_us": int(start_us), "stop_us": int(stop_us), "intervals": intervals,
                "valid_pixels": int(valid.sum()), "valid_mask_sha256": array_hash(valid),
                "ground_truth_sha256": array_hash(flow), "arms": {}}
        for arm in ARMS:
            started = time.perf_counter()
            inputs, counts, hashes, degenerate = [], [], [], []
            for left, right in intervals:
                sample = selected_interval(events, selected[arm], left, right)
                counts.append(len(sample))
                hashes.append(array_hash(sample))
                degenerate.append(bool(len(sample) < 2 or sample[-1, 0] == sample[0, 0]))
                inputs.append(official_voxel(sample, rectify_map))
            voxel_seconds[arm] += time.perf_counter() - started
            old, new = [x.unsqueeze(0).to(device) for x in inputs]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                _, prediction_list = model(old, new, iters=12, flow_init=None)
                predicted = prediction_list[-1][0].permute(1, 2, 0).cpu().numpy()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_seconds[arm] += time.perf_counter() - started
            record = {"pair_index": pair_index, **metric_record(predicted, flow, valid)}
            records[arm].append(record)
            pair["arms"][arm] = {"event_counts": counts, "event_payload_sha256": hashes,
                                 "zero_grid_fallback": degenerate,
                                 "voxel_sha256": [array_hash(x.numpy()) for x in inputs], **record}
            del old, new, prediction_list, predicted, inputs
        pairs.append(pair)
        if pair_index % 20 == 0:
            print(f"evaluate {sequence}: pair {pair_index + 1}/{len(files)}", flush=True)
    summaries = {arm: {**aggregate_records(records[arm]), "model_seconds": inference_seconds[arm],
                       "voxel_seconds": voxel_seconds[arm]} for arm in ARMS}
    contrasts = {arm: summaries["occupancy"]["epe"] - summaries[arm]["epe"] for arm in ARMS if arm != "occupancy"}
    result = {"sequence": sequence, "filter_manifest": manifest, "available_pairs": len(files),
              "evaluated_pairs": len(pairs), "exclusions": exclusions, "pairs": pairs,
              "metrics": summaries, "rsf_minus_reference_epe": contrasts,
              "zero_grid_fallback_counts": {
                  arm: sum(sum(p["arms"][arm]["zero_grid_fallback"]) for p in pairs) for arm in ARMS},
              "rectify_map_sha256": array_hash(rectify_map)}
    write_json(prepared / "outcomes.json", result)
    return result


def summarize(sequences):
    metrics = {}
    for arm in ARMS:
        metrics[arm] = {key: float(np.mean([s["metrics"][arm][key] for s in sequences]))
                        for key in ("epe", "one_pixel_percent", "three_pixel_percent")}
    return {"equal_recording_metrics": metrics,
            "rsf_minus_reference_epe": {arm: metrics["occupancy"]["epe"] - metrics[arm]["epe"]
                                         for arm in ARMS if arm != "occupancy"},
            "recordings": len(sequences), "forward_pairs": sum(s["evaluated_pairs"] for s in sequences),
            "inference_scope": "descriptive three-recording mean; no population confidence interval or significance test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("download", "prepare", "evaluate", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert_freeze()
    if args.stage == "download":
        download_inputs(args.data_root)
        return
    if args.stage in ("prepare", "all"):
        from .phase79_tstar_cost_quality import iter_filtered_windows
        warmup = np.array([[0, 2, 2, 0], [1, 3, 2, 0], [2, 2, 3, 1]], dtype=np.int64)
        for arm in ("occupancy", "multiplicity", "ebf"):
            for item in iter_filtered_windows(warmup, WIDTH, HEIGHT, method=arm):
                del item
        for index, sequence in enumerate(COHORT):
            prepare_sequence(args.data_root, sequence, index)
    if args.stage in ("evaluate", "all"):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = load_model(args.device)
        start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sequences = [evaluate_sequence(args.data_root, s, model, args.device) for s in COHORT]
        code_files = [Path(__file__), ROOT / "native3d_ssm/phase79_tstar_cost_quality.py",
                      ROOT / "native3d_ssm/phase75_integer_count_cost.py",
                      ERAFT_ROOT / "loader/loader_dsec.py", ERAFT_ROOT / "utils/dsec_utils.py",
                      ERAFT_ROOT / "model/eraft.py"]
        code_files += sorted(set((ERAFT_ROOT / "model").glob("*.py")) - set(code_files))
        payload = {"schema_version": 1, "complete": True, "protocol_sha256": PROTOCOL_SHA256,
                   "outcome_evaluation_started_utc": start_utc,
                   "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "cohort": list(COHORT), "conditions": list(ARMS), "seed": SEED,
                   "checkpoint_sha256": CHECKPOINT_SHA256,
                   "upstream_eraft_revision": "c58ce0524ea0ebfa9849991caafb547f44fe9bfd",
                   "model_training_data": "MVSEC outdoor_day_2; no DSEC training or fine-tuning",
                   "execution_notes": "The first recording was repeated after a diagnostic NumPy-boolean JSON serialization failure; no task outcome was inspected or used to change the fixed protocol.",
                   "model": {"name": "E-RAFT", "checkpoint": "mvsec_20.tar", "iterations": 12,
                             "warm_start": False, "height": HEIGHT, "width": WIDTH, "voxel_bins": BINS,
                             "previous_interval_us": DELTA_US,
                             "current_interval": "annotated forward-flow start/end (~100 ms)",
                             "voxel_endpoint_normalization": "retained first/last event, exactly as upstream DSEC",
                             "empty_or_single_timestamp_interval": "zero voxel grid",
                             "rectification": "subpixel trilinear splats; out-of-bounds contributions discarded",
                             "validity": "released flow validity channel; no event mask, crop, or hood exclusion"},
                   "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                                   "platform": platform.platform(), "device": args.device,
                                   "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None},
                   "source_code_sha256": {str(p.relative_to(WORKSPACE)): file_hash(p) for p in code_files},
                   "downloads": json.loads((args.data_root / "download_manifest.json").read_text()),
                   "sequences": sequences, "summary": summarize(sequences)}
        write_json(args.output, payload)
        print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
