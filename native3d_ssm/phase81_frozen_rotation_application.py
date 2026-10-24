"""Frozen event-count-dependent rotational-motion application experiment.

The application uses a calibrated, exact SO(3) contrast-maximization adapter.
Events are backwarped to a common interval start using the positive camera-body
rotation convention: R_WC(t) = R_WC(0) Exp(omega * t). No ground-truth pose or
inertial measurement is an optimizer input. The complete data protocol must be
frozen before the real recordings are downloaded or evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

WIDTH, HEIGHT = 240, 180
WINDOW_US = 20_000
BLUR_SIGMA = 1.0
OMEGA_BOUND = 35.0
MAXITER, MAXFUN = 20, 60
FTOL, GTOL = 1e-9, 1e-5
STRIDE_US = 100_000
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "downstream/assets/davis_rotation_frozen_20261023"
PROTOCOL = ROOT / "plan/protocol_frozen_rotation_application_20261023.md"
PROTOCOL_SHA256 = "fe0ff7cdff47d07404aa8b13e083fe79d5205f19e896629c98ef56c89a2ecb64"
TIMESTAMP_ADAPTER = ROOT / "plan/protocol_rotation_timestamp_adapter_20261023.md"
TIMESTAMP_ADAPTER_SHA256 = "839652013d86c09b3bf1b6d519c023f0f1406ab41c6f5721adaa49108d2e7e8a"
OUTPUT = ROOT / "native3d_ssm/experiments/analytic_contrast/phase81_frozen_rotation_application.json"
COHORT = ("shapes_rotation", "poster_rotation", "boxes_rotation")
EXPECTED_COUNTS = (23_126_288, 169_350_136, 185_688_947)
EXPECTED_ZIP_BYTES = (157_446_920, 833_672_602, 893_172_313)
ARMS = ("raw", "occupancy", "multiplicity", "ebf", "random")
RANDOM_SEED = 20261023
PHASE79_SHA256 = "abb20e3e7395ab0a1e22045284a55e647cdeb3c4d5e8dc7931ed4d4e132c308a"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values):
    values = np.asarray(values)
    digest = hashlib.sha256((str(values.dtype)+str(values.shape)).encode())
    flat = values.reshape(-1)
    for start in range(0, len(flat), 1_000_000):
        digest.update(np.ascontiguousarray(flat[start:start+1_000_000]).tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"unsupported JSON value: {type(item).__name__}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=scalar)+"\n")
    temporary.replace(path)


def assert_freeze():
    if file_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("scientific protocol has not been frozen or has changed")
    if file_hash(TIMESTAMP_ADAPTER) != TIMESTAMP_ADAPTER_SHA256:
        raise ValueError("pre-outcome timestamp adapter clarification changed")
    if file_hash(ROOT / "native3d_ssm/phase79_tstar_cost_quality.py") != PHASE79_SHA256:
        raise ValueError("frozen phase79 implementation changed")


def canonical_microseconds(seconds):
    """Apply the frozen common-clock adapter with both precision guards."""
    seconds = np.asarray(seconds, dtype=np.float64)
    if not np.isfinite(seconds).all():
        raise ValueError("nonfinite source timestamp")
    microseconds = np.rint(seconds*1e6).astype(np.int64)
    float_error_us = float(np.max(np.abs(seconds*1e6-microseconds)))
    integer_ns = np.rint(seconds*1e9).astype(np.int64)
    exact_error_ns = int(np.max(np.abs(integer_ns-1000*microseconds)))
    if exact_error_ns > 1 or float_error_us > .00101:
        raise ValueError("source exceeds frozen 1ns timestamp-adapter bound")
    return microseconds, float_error_us, exact_error_ns


def validate_source_metadata(metadata, sequence_index):
    if (metadata["sequence"] != COHORT[sequence_index] or
            metadata["event_count"] != EXPECTED_COUNTS[sequence_index] or
            metadata["maximum_integer_ns_rounding_error"] > 1 or
            metadata["maximum_float_decode_rounding_error_us"] > .00101):
        raise ValueError("source audit does not meet frozen cohort/precision contract")
    if (metadata.get("timestamp_adapter_approval_pending", True) or
            metadata.get("protocol_sha256") != PROTOCOL_SHA256 or
            metadata.get("timestamp_adapter_sha256") != TIMESTAMP_ADAPTER_SHA256):
        raise ValueError("source metadata is not bound to both frozen protocols")


def download_inputs(data_root=DATA_ROOT):
    """Download only the prospectively named official archives."""
    from concurrent.futures import ThreadPoolExecutor
    import urllib.request
    assert_freeze()
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    def retrieve(index):
        sequence = COHORT[index]
        path = data_root / (sequence+".zip")
        url = f"https://files.ifi.uzh.ch/rpg/datasets/davis/{sequence}.zip"
        if not path.exists():
            temporary = path.with_suffix(".zip.partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                for block in iter(lambda: response.read(4*1024*1024), b""):
                    stream.write(block)
            temporary.replace(path)
        if path.stat().st_size != EXPECTED_ZIP_BYTES[index]:
            raise ValueError(f"unexpected public archive byte count: {sequence}")
        print(f"downloaded {sequence}: {path.stat().st_size} bytes", flush=True)
        return {"sequence": sequence, "url": url, "bytes": path.stat().st_size, "sha256": file_hash(path)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(retrieve, range(len(COHORT))))
    manifest = {"retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                "protocol_sha256": PROTOCOL_SHA256, "files": records}
    write_json(data_root / "download_manifest.json", manifest)
    return manifest


def prepare_recording(data_root, sequence_index, audit_only=False):
    """Decode the released text using the frozen nearest-microsecond adapter."""
    import itertools
    import zipfile
    from .phase79_tstar_cost_quality import validate_events
    assert_freeze()
    sequence = COHORT[sequence_index]
    directory = Path(data_root)/sequence
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory/"raw_events.npy"
    metadata_path = directory/("source_audit_pending.json" if audit_only else "source_metadata.json")
    pending_path = directory/"source_audit_pending.json"
    cached_metadata = metadata_path if metadata_path.exists() else pending_path
    archive_path = Path(data_root)/(sequence+".zip")
    if raw_path.exists() and cached_metadata.exists():
        metadata = json.loads(cached_metadata.read_text())
        raw = np.load(raw_path,mmap_mode="r")
        if array_hash(raw) != metadata["raw_payload_sha256"]:
            raise ValueError("cached canonical source payload changed")
        if (len(raw) != EXPECTED_COUNTS[sequence_index] or
                int(raw[0,0]) != metadata["first_event_us"] or int(raw[-1,0]) != metadata["last_event_us"]):
            raise ValueError("cached source count/coverage changed")
        for path,field in ((archive_path,"archive_sha256"),(directory/"calib.txt","calibration_sha256"),
                           (directory/"groundtruth.txt","groundtruth_sha256")):
            if file_hash(path) != metadata[field]:
                raise ValueError(f"cached source component changed: {path.name}")
        validate_events(raw,WIDTH,HEIGHT)
        build_bearing_map(np.loadtxt(directory/"calib.txt").ravel())
        if not audit_only:
            metadata.update(protocol_sha256=PROTOCOL_SHA256,timestamp_adapter_sha256=TIMESTAMP_ADAPTER_SHA256,
                            timestamp_adapter_approval_pending=False,
                            timestamp_adapter="nearest integer microseconds; exact residual<=1ns, floating residual<=.00101us")
            validate_source_metadata(metadata,sequence_index)
            write_json(metadata_path,metadata)
        return metadata
    count = EXPECTED_COUNTS[sequence_index]
    with zipfile.ZipFile(archive_path) as archive:
        def member(basename):
            found = [n for n in archive.namelist() if Path(n).name==basename]
            if len(found)!=1:
                raise ValueError(f"expected one {basename} in archive")
            return found[0]
        events_member = member("events.txt")
        raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.int64, shape=(count,4))
        offset, polarities, text_digest = 0, set(), hashlib.sha256()
        maximum_rounding_error_us = 0.0
        maximum_integer_ns_rounding_error = 0
        with archive.open(events_member) as stream:
            while True:
                lines = list(itertools.islice(stream, 500_000))
                if not lines:
                    break
                block = b"".join(lines)
                text_digest.update(block)
                parsed = np.fromstring(block.decode("ascii"),sep=" ").reshape(-1,4)
                stop = offset+len(parsed)
                if stop > count or not np.isfinite(parsed).all():
                    raise ValueError("unexpected raw source count/values")
                microseconds, rounding_error, integer_ns_error = canonical_microseconds(parsed[:,0])
                maximum_rounding_error_us = max(maximum_rounding_error_us,rounding_error)
                maximum_integer_ns_rounding_error = max(maximum_integer_ns_rounding_error,
                                                        integer_ns_error)
                if not np.array_equal(parsed[:,1:],np.rint(parsed[:,1:])):
                    raise ValueError("source event fields are not integer-valued")
                observed_polarities = set(np.unique(parsed[:,3]).tolist())
                polarities.update(observed_polarities)
                if not polarities.issubset({0.,1.}) and not polarities.issubset({-1.,1.}):
                    raise ValueError("inconsistent source polarity convention")
                raw[offset:stop,0] = microseconds.astype(np.int64)
                raw[offset:stop,1:3] = parsed[:,1:3].astype(np.int64)
                raw[offset:stop,3] = (parsed[:,3]>0).astype(np.int64)
                offset = stop
                if offset % 10_000_000 == 0:
                    print(f"prepare {sequence}: {offset}/{count} events",flush=True)
        if offset!=count:
            raise ValueError(f"published count mismatch: {offset} != {count}")
        raw.flush()
        for name in ("calib.txt","groundtruth.txt"):
            (directory/name).write_bytes(archive.read(member(name)))
        validate_events(raw,WIDTH,HEIGHT)
        calibration = np.loadtxt(directory/"calib.txt").ravel()
        build_bearing_map(calibration)
        metadata = {"sequence": sequence,"event_count": count,"first_event_us": int(raw[0,0]),
                    "last_event_us": int(raw[-1,0]),"raw_payload_sha256": array_hash(raw),
                    "events_txt_sha256": text_digest.hexdigest(),"archive_sha256": file_hash(archive_path),
                    "calibration_sha256": file_hash(directory/"calib.txt"),
                    "groundtruth_sha256": file_hash(directory/"groundtruth.txt"),
                    "source_polarity_values": sorted(polarities),
                    "canonical_polarity": "positive=1; zero or negative=0",
                    "maximum_float_decode_rounding_error_us": maximum_rounding_error_us,
                    "maximum_integer_ns_rounding_error": maximum_integer_ns_rounding_error,
                    "timestamp_adapter_approval_pending": audit_only,
                    "protocol_sha256": PROTOCOL_SHA256,"timestamp_adapter_sha256": TIMESTAMP_ADAPTER_SHA256,
                    "timestamp_adapter": "nearest integer microseconds; exact residual<=1ns, floating residual<=.00101us",
                    "coordinates": "original distorted integer240x180 sensor coordinates"}
        if not audit_only:
            validate_source_metadata(metadata,sequence_index)
        write_json(metadata_path,metadata)
    return metadata


def distort_normalized(x, y, calibration):
    k1, k2, p1, p2, k3 = calibration[4:]
    r2 = x*x+y*y
    radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    return (x*radial+2*p1*x*y+p2*(r2+2*x*x),
            y*radial+p1*(r2+2*y*y)+2*p2*x*y)


def build_bearing_map(calibration, width=WIDTH, height=HEIGHT):
    """Fixed20-step Newton inverse of released Brown–Conrady distortion."""
    calibration = np.asarray(calibration, dtype=np.float64)
    if calibration.shape != (9,) or not np.isfinite(calibration).all() or np.any(calibration[:2] <= 0):
        raise ValueError("invalid released Brown-Conrady calibration")
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = calibration
    yy, xx = np.mgrid[:height, :width]
    xd, yd = (xx-cx)/fx, (yy-cy)/fy
    x, y = xd.copy(), yd.copy()
    for _ in range(20):
        r2 = x*x+y*y
        radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        radial_prime = k1+2*k2*r2+3*k3*r2*r2
        predicted_x, predicted_y = distort_normalized(x,y,calibration)
        a = radial+2*x*x*radial_prime+2*p1*y+6*p2*x
        b = 2*x*y*radial_prime+2*p1*x+2*p2*y
        d = radial+2*y*y*radial_prime+6*p1*y+2*p2*x
        determinant = a*d-b*b
        if np.any(np.abs(determinant) < 1e-14):
            raise ValueError("singular Brown-Conrady inverse")
        ex, ey = predicted_x-xd, predicted_y-yd
        x -= (d*ex-b*ey)/determinant
        y -= (a*ey-b*ex)/determinant
    check_x, check_y = distort_normalized(x,y,calibration)
    residual = max(float(np.max(np.abs(check_x-xd))), float(np.max(np.abs(check_y-yd))))
    if not np.isfinite(residual) or residual > 1e-9:
        raise ValueError(f"Brown-Conrady inverse residual {residual}")
    return np.ascontiguousarray(np.stack((x,y,np.ones_like(x)), axis=-1))


def application_intervals(events):
    """Every source-covered aperture at the frozen absolute 10-Hz cadence."""
    if not len(events):
        return []
    first, last = int(events[0,0]), int(events[-1,0])
    return [(start, start+WINDOW_US) for start in range(first, last-WINDOW_US+1, STRIDE_US)]


def random_seed(sequence_index):
    return np.random.SeedSequence(RANDOM_SEED).spawn(len(COHORT))[sequence_index]


def filtered_windows(events, arm, sequence_index):
    """The three score filters are the frozen phase79 generator unchanged."""
    from .phase79_tstar_cost_quality import iter_filtered_windows, validate_events, window_bounds, WindowOutput
    if arm != "random":
        yield from iter_filtered_windows(events, WIDTH, HEIGHT, method=arm)
        return
    validate_events(events, WIDTH, HEIGHT)
    rng = np.random.default_rng(random_seed(sequence_index))
    for start, stop, close in window_bounds(events):
        n = stop-start
        quota = 4*n//5
        keep = np.ones(n, dtype=np.bool_)
        keep[rng.choice(n, size=n-quota, replace=False)] = False
        indices = np.flatnonzero(keep)+start
        emitted = events[indices]  # Ordered, independent original-payload copy.
        yield WindowOutput(start, stop, close, emitted, indices, None, None, 0.)
        del emitted, indices, keep


def application_packets(events, arm, sequence_index, intervals, statistics, audit=False):
    """Consume the whole source, collecting only fixed application apertures.

    The consumer must exhaust this generator: filtering continues through the
    final tail even after the last motion update. Auditing is a separate pass.
    """
    from .phase79_tstar_cost_quality import validate_events
    statistics.update({"source_events": len(events), "retained_events": 0,
                       "application_events": 0, "filter_windows": 0})
    if arm == "raw":
        validate_events(events, WIDTH, HEIGHT)
        statistics["retained_events"] = len(events)
        for start, stop in intervals:
            a, b = np.searchsorted(events[:,0], (start,stop), side="left")
            sample = events[a:b]  # Identity may use a view.
            statistics["application_events"] += len(sample)
            yield start, stop, sample
        return
    next_interval, pieces = 0, []
    digest = hashlib.sha256() if audit else None
    windows = [] if audit else None
    for output in filtered_windows(events, arm, sequence_index):
        statistics["retained_events"] += len(output.emitted)
        statistics["filter_windows"] += 1
        if audit:
            if not np.array_equal(output.emitted, events[output.indices]):
                raise AssertionError("filter altered original event payloads")
            if len(output.indices)>1 and np.any(np.diff(output.indices)<=0):
                raise AssertionError("filter output is not in original arrival order")
            digest.update(output.indices.tobytes())
            windows.append({"start": output.start, "stop": output.stop, "close_us": output.close_us,
                            "retained": len(output.emitted)})
        while next_interval < len(intervals) and intervals[next_interval][1] <= output.close_us:
            start, stop = intervals[next_interval]
            a, b = np.searchsorted(output.emitted[:,0], (start,stop), side="left")
            if b > a:
                pieces.append(output.emitted[a:b].copy())
            sample = (pieces[0] if len(pieces)==1 else np.concatenate(pieces) if pieces
                      else np.empty((0,4), np.int64))
            statistics["application_events"] += len(sample)
            yield start, stop, sample
            del sample
            pieces = []
            next_interval += 1
        if next_interval < len(intervals):
            start, stop = intervals[next_interval]
            if start < output.close_us:
                a, b = np.searchsorted(output.emitted[:,0], (start,stop), side="left")
                if b > a:
                    pieces.append(output.emitted[a:b].copy())
        del output
    if next_interval != len(intervals) or pieces:
        raise AssertionError("full source did not close every common application aperture")
    if audit:
        statistics["ordered_selected_indices_bytes_sha256"] = digest.hexdigest()
        statistics["windows"] = windows


def pose_targets(groundtruth, intervals):
    """Interpolation affects external evaluation only, never application inputs."""
    groundtruth = np.asarray(groundtruth, dtype=np.float64)
    if groundtruth.ndim != 2 or groundtruth.shape[1] != 8 or not np.isfinite(groundtruth).all():
        raise ValueError("invalid released motion-capture pose rows")
    ts = groundtruth[:,0]
    if np.any(np.diff(ts) <= 0):
        raise ValueError("pose timestamps must be strictly increasing")
    norm = np.linalg.norm(groundtruth[:,4:8], axis=1)
    if np.any(norm < 1e-12):
        raise ValueError("invalid pose quaternion")
    rotations = Rotation.from_quat(groundtruth[:,4:8]/norm[:,None])
    interpolate = Slerp(ts, rotations)
    targets, exclusions = [], []
    for index, (start, stop) in enumerate(intervals):
        endpoints = np.array((start,stop),np.float64)/1e6
        reason = None
        brackets = []
        bracket_times = []
        for endpoint in endpoints:
            right = int(np.searchsorted(ts,endpoint,side="left"))
            if endpoint < ts[0] or endpoint > ts[-1]:
                bracket_times.append(None)
            elif right < len(ts) and ts[right]==endpoint:
                bracket_times.append([float(ts[right]),float(ts[right])])
            else:
                bracket_times.append([float(ts[right-1]),float(ts[right])])
        if endpoints[0] < ts[0] or endpoints[1] > ts[-1]:
            reason = "outside released mocap time coverage"
        else:
            for endpoint in endpoints:
                right = int(np.searchsorted(ts, endpoint, side="left"))
                if right < len(ts) and ts[right] == endpoint:
                    brackets.append({"times_s": [float(ts[right]),float(ts[right])],
                                     "quaternions_xyzw": [rotations[right].as_quat().tolist()]*2})
                    continue
                if right == 0 or right == len(ts) or ts[right]-ts[right-1] > .020+1e-12:
                    reason = "mocap interpolation bracket exceeds20ms"
                else:
                    brackets.append({"times_s": [float(ts[right-1]),float(ts[right])],
                                     "quaternions_xyzw": rotations[[right-1,right]].as_quat().tolist()})
        if reason:
            targets.append(None)
            exclusions.append({"update_index": index, "reason": reason,
                               "endpoint_times_s": endpoints.tolist(),
                               "mocap_range_s": [float(ts[0]),float(ts[-1])],
                               "adjacent_bracket_times_s": bracket_times})
        else:
            pair = interpolate(endpoints)
            targets.append({"endpoint_quaternions_xyzw": pair.as_quat().tolist(),
                            "relative_quaternion_xyzw": (pair[0].inv()*pair[1]).as_quat().tolist(),
                            "interpolation_brackets": brackets})
    return targets, exclusions


@njit(cache=True)
def rodrigues_point_and_jacobian(point, omega, dt):
    """Return Exp(omega*dt) point and its exact derivative w.r.t. omega.

    Stable Taylor coefficients cover the zero-motion case. The derivative is
    derived from Rodrigues' polynomial, not a first-order rotation model.
    """
    v = omega * dt
    q = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    if q < 1e-8:
        a = 1.0 - q / 6.0 + q * q / 120.0 - q * q * q / 5040.0
        b = 0.5 - q / 24.0 + q * q / 720.0 - q * q * q / 40320.0
        da_dq = -1.0 / 6.0 + q / 60.0 - q * q / 1680.0
        db_dq = -1.0 / 24.0 + q / 360.0 - q * q / 13440.0
    else:
        theta = np.sqrt(q)
        sn, cs = np.sin(theta), np.cos(theta)
        a = sn / theta
        b = (1.0 - cs) / q
        da_dq = (theta * cs - sn) / (2.0 * q * theta)
        db_dq = (theta * sn - 2.0 * (1.0 - cs)) / (2.0 * q * q)
    cross1 = np.array((v[1]*point[2] - v[2]*point[1],
                       v[2]*point[0] - v[0]*point[2],
                       v[0]*point[1] - v[1]*point[0]))
    cross2 = np.array((v[1]*cross1[2] - v[2]*cross1[1],
                       v[2]*cross1[0] - v[0]*cross1[2],
                       v[0]*cross1[1] - v[1]*cross1[0]))
    warped = point + a * cross1 + b * cross2
    jacobian = np.empty((3, 3), np.float64)
    for j in range(3):
        if j == 0:
            ecp = np.array((0.0, -point[2], point[1]))
            ecc = np.array((0.0, -cross1[2], cross1[1]))
        elif j == 1:
            ecp = np.array((point[2], 0.0, -point[0]))
            ecc = np.array((cross1[2], 0.0, -cross1[0]))
        else:
            ecp = np.array((-point[1], point[0], 0.0))
            ecc = np.array((-cross1[1], cross1[0], 0.0))
        vcross_ecp = np.array((v[1]*ecp[2]-v[2]*ecp[1],
                              v[2]*ecp[0]-v[0]*ecp[2],
                              v[0]*ecp[1]-v[1]*ecp[0]))
        dq = 2.0 * dt * v[j]
        jacobian[:, j] = (a * dt * ecp + b * dt * (ecc + vcross_ecp)
                          + da_dq * dq * cross1 + db_dq * dq * cross2)
    return warped, jacobian


@njit(cache=True)
def iwe_and_derivative(bearings, times_s, omega, intrinsics, width, height):
    """Polarity-free bilinear splatting in the fixed output sensor rectangle.

    Every in-bounds neighbor receives its vote; missing neighbors contribute
    zero. Neither contrast nor stopping tolerance is renormalized by the number
    of events that remain inside the image after warping.
    """
    image = np.zeros((height, width), np.float64)
    derivative = np.zeros((3, height, width), np.float64)
    fx, fy, cx, cy = intrinsics
    positive_depth_events, contributed_events = 0, 0
    for k in range(len(times_s)):
        # Scalar expansion avoids allocating small arrays for every event.
        # These are the same exact Rodrigues coefficients as the public helper.
        px, py, pz = bearings[k, 0], bearings[k, 1], bearings[k, 2]
        dt = times_s[k]
        v0, v1, v2 = omega[0]*dt, omega[1]*dt, omega[2]*dt
        q = v0*v0 + v1*v1 + v2*v2
        if q < 1e-8:
            a = 1.0-q/6.0+q*q/120.0-q*q*q/5040.0
            b = 0.5-q/24.0+q*q/720.0-q*q*q/40320.0
            aq = -1.0/6.0+q/60.0-q*q/1680.0
            bq = -1.0/24.0+q/360.0-q*q/13440.0
        else:
            theta = np.sqrt(q)
            sn, cs = np.sin(theta), np.cos(theta)
            a, b = sn/theta, (1.0-cs)/q
            aq = (theta*cs-sn)/(2.0*q*theta)
            bq = (theta*sn-2.0*(1.0-cs))/(2.0*q*q)
        c0, c1, c2 = v1*pz-v2*py, v2*px-v0*pz, v0*py-v1*px
        d0, d1, d2 = v1*c2-v2*c1, v2*c0-v0*c2, v0*c1-v1*c0
        point0, point1, point2 = px+a*c0+b*d0, py+a*c1+b*d1, pz+a*c2+b*d2
        if point2 <= 0.0:
            continue
        positive_depth_events += 1
        invz = 1.0 / point2
        x, y = point0 * invz, point1 * invz
        u, v = fx*x + cx, fy*y + cy
        i, j = int(np.floor(u)), int(np.floor(v))
        if i < -1 or i >= width or j < -1 or j >= height:
            continue
        contributed_events += 1
        dx, dy = u-i, v-j
        weights = ((1.0-dx)*(1.0-dy), dx*(1.0-dy), (1.0-dx)*dy, dx*dy)
        us, vs = (i, i+1, i, i+1), (j, j, j+1, j+1)
        gu, gv = (-(1.0-dy), 1.0-dy, -dy, dy), (-(1.0-dx), -dx, 1.0-dx, dx)
        for corner in range(4):
            xx, yy = us[corner], vs[corner]
            if 0 <= xx < width and 0 <= yy < height:
                image[yy, xx] += weights[corner]
        for axis in range(3):
            if axis == 0:
                ec0, ec1, ec2 = 0.0, -pz, py
                vc0, vc1, vc2 = v1*py+v2*pz, -c2-v0*py, c1-v0*pz
                dq = 2.0*dt*v0
            elif axis == 1:
                ec0, ec1, ec2 = pz, 0.0, -px
                vc0, vc1, vc2 = c2-v1*px, v2*pz+v0*px, -c0-v1*pz
                dq = 2.0*dt*v1
            else:
                ec0, ec1, ec2 = -py, px, 0.0
                vc0, vc1, vc2 = -c1-v2*px, c0-v2*py, v0*px+v1*py
                dq = 2.0*dt*v2
            jp0 = a*dt*ec0+b*dt*vc0+aq*dq*c0+bq*dq*d0
            jp1 = a*dt*ec1+b*dt*vc1+aq*dq*c1+bq*dq*d1
            jp2 = a*dt*ec2+b*dt*vc2+aq*dq*c2+bq*dq*d2
            du = fx*invz*(jp0-x*jp2)
            dv = fy*invz*(jp1-y*jp2)
            for corner in range(4):
                xx, yy = us[corner], vs[corner]
                if 0 <= xx < width and 0 <= yy < height:
                    derivative[axis, yy, xx] += gu[corner]*du + gv[corner]*dv
    return image, derivative, positive_depth_events, contributed_events


class ContrastObjective:
    """Negative normalized IWE variance and its analytic gradient."""

    def __init__(self, bearings, times_s, intrinsics, width=WIDTH, height=HEIGHT):
        self.bearings = np.ascontiguousarray(bearings, dtype=np.float64)
        self.times_s = np.ascontiguousarray(times_s, dtype=np.float64)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        self.width, self.height = int(width), int(height)
        if self.bearings.shape != (len(self.times_s), 3):
            raise ValueError("bearing/time shapes disagree")
        if (self.intrinsics.shape != (4,) or not np.isfinite(self.intrinsics).all()
                or np.any(self.intrinsics[:2] <= 0)):
            raise ValueError("invalid pinhole intrinsics")
        if not np.isfinite(self.bearings).all() or not np.isfinite(self.times_s).all():
            raise ValueError("nonfinite estimator input")
        if np.any(self.times_s < 0) or np.any(self.times_s >= WINDOW_US / 1e6):
            raise ValueError("events must lie in the fixed half-open 20-ms interval")
        self.calls = 0
        self.last_positive_depth_events = 0
        self.last_contributed_events = 0

    def __call__(self, omega):
        self.calls += 1
        if len(self.times_s) == 0:
            return 0.0, np.zeros(3, np.float64)
        image, derivatives, positive, contributed = iwe_and_derivative(
            self.bearings, self.times_s, np.asarray(omega, dtype=np.float64),
            self.intrinsics, self.width, self.height)
        self.last_positive_depth_events, self.last_contributed_events = positive, contributed
        scale = (self.width*self.height) / len(self.times_s)
        image = gaussian_filter(image, BLUR_SIGMA, mode="constant", cval=0.0) * scale
        centered = image - image.mean()
        gradient = np.empty(3, np.float64)
        for axis in range(3):
            d = gaussian_filter(derivatives[axis], BLUR_SIGMA, mode="constant", cval=0.0) * scale
            # sum(centered) is zero, so subtracting the derivative mean is redundant.
            gradient[axis] = -2.0 * np.mean(centered*d)
        return -float(np.mean(centered*centered)), gradient


@dataclass
class Estimate:
    omega: np.ndarray
    success: bool
    status: int
    message: str
    iterations: int
    evaluations: int
    objective: float
    elapsed_seconds: float


def estimate_rotation(bearings, times_s, intrinsics, initial_omega):
    """Run identical bounded optimization; caller owns the arm-specific history."""
    initial = np.asarray(initial_omega, dtype=np.float64)
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError("invalid starting angular velocity")
    initial = np.clip(initial, -OMEGA_BOUND, OMEGA_BOUND)
    started = time.perf_counter()
    objective = ContrastObjective(bearings, times_s, intrinsics)
    if len(times_s) < 2:
        return Estimate(initial.copy(), False, -1, "fewer than two input events: hold previous estimate",
                        0, 0, 0.0, time.perf_counter()-started)
    if np.max(times_s) == np.min(times_s):
        return Estimate(initial.copy(), False, -3, "all input timestamps identical: hold previous estimate",
                        0, 0, 0.0, time.perf_counter()-started)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(-OMEGA_BOUND, OMEGA_BOUND)]*3,
                      options={"maxiter": MAXITER, "maxfun": MAXFUN,
                               "ftol": FTOL, "gtol": GTOL})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        return Estimate(initial.copy(), False, -2, "nonfinite optimizer result: hold previous estimate",
                        int(result.nit), objective.calls, 0.0, time.perf_counter()-started)
    return Estimate(result.x.copy(), bool(result.success), int(result.status), str(result.message),
                    int(result.nit), objective.calls, float(result.fun), time.perf_counter()-started)


def relative_rotation_error_degrees(estimated_omega, duration_s, start_rotation, stop_rotation):
    """Geodesic endpoint error, with both rotations mapping camera to world."""
    truth = start_rotation.inv() * stop_rotation
    predicted = Rotation.from_rotvec(np.asarray(estimated_omega)*duration_s)
    return float(np.rad2deg((truth.inv()*predicted).magnitude()))


def synthetic_events(count=100_000, omega=(2., -3., 5.), seed=20261023, quantize=True):
    """Synthetic static edge rays; never reads any prospective recording."""
    rng = np.random.default_rng(seed)
    intrinsics = np.array((199., 199., 120., 90.))
    times = np.sort(rng.uniform(0., WINDOW_US/1e6, count))
    u, v = rng.uniform(35, 205, count), rng.uniform(35, 145, count)
    vertical = np.arange(count) % 2 == 0
    u[vertical] = rng.choice((50., 90., 150., 190.), vertical.sum())
    v[~vertical] = rng.choice((40., 70., 110., 140.), (~vertical).sum())
    reference = np.column_stack(((u-120.)/199., (v-90.)/199., np.ones(count)))
    observed = Rotation.from_rotvec(-times[:, None]*np.asarray(omega)).apply(reference)
    pixels = observed[:, :2] / observed[:, 2, None] * 199. + np.array((120., 90.))
    if quantize:
        pixels = np.round(pixels)
    bearings = np.column_stack(((pixels[:, 0]-120.)/199., (pixels[:, 1]-90.)/199., np.ones(count)))
    return bearings, times, intrinsics


def synthetic_benchmark():
    bearings, times_s, intrinsics = synthetic_events()
    objective = ContrastObjective(bearings, times_s, intrinsics)
    objective(np.zeros(3))  # Exclude JIT compilation from the timing below.
    repeated = []
    for _ in range(5):
        started = time.perf_counter()
        objective(np.array((1., -2., 4.)))
        repeated.append(time.perf_counter()-started)
    fits = []
    for truth in (np.array((2., -3., 5.)), np.array((5., -7., 10.))):
        bearing, ts, calibration = synthetic_events(omega=truth)
        result = estimate_rotation(bearing, ts, calibration, np.zeros(3))
        fits.append({"truth_rad_s": truth.tolist(), "estimate_rad_s": result.omega.tolist(),
                     "error_deg_s": float(np.linalg.norm(result.omega-truth)*180/np.pi),
                     "seconds": result.elapsed_seconds, "evaluations": result.evaluations,
                     "iterations": result.iterations, "success": result.success,
                     "message": result.message})
    return {"synthetic_events": len(times_s), "objective_gradient_median_seconds": float(np.median(repeated)),
            "fits": fits}


def dispatcher_signatures():
    import sys
    from numba.core.registry import CPUDispatcher
    snapshot = {}
    for name,module in sorted(sys.modules.items()):
        owns_runner = (module is not None and getattr(module,"__file__",None) is not None
                       and Path(module.__file__).resolve()==Path(__file__).resolve())
        if module is not None and (name.startswith("native3d_ssm.") or owns_runner):
            key_prefix = "native3d_ssm.phase81_frozen_rotation_application[__main__]" if owns_runner and name=="__main__" else name
            for attribute,value in sorted(vars(module).items()):
                if isinstance(value,CPUDispatcher):
                    key = key_prefix+":"+attribute
                    # Numba cache loading can import a second copy of a -m runner.
                    # Separate alias keys detect compilation in either dispatcher.
                    snapshot[key] = sorted(str(s) for s in value.signatures)
    return snapshot


def source_code_hashes():
    import sys
    hashes = {}
    for name,module in sorted(sys.modules.items()):
        if name.startswith("native3d_ssm") and module is not None:
            path = getattr(module,"__file__",None)
            if path and Path(path).suffix==".py" and Path(path).is_relative_to(ROOT):
                hashes[str(Path(path).relative_to(ROOT))] = file_hash(path)
    hashes[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    return hashes


def warm_synthetic_kernels():
    """Compile small and uint32-count paths without prospective recording input."""
    from .phase79_tstar_cost_quality import iter_filtered_windows
    for n in (100,70_000):
        i = np.arange(n,dtype=np.int64)
        events = np.column_stack((i//10,i%WIDTH,(i//WIDTH)%HEIGHT,i%2))
        for arm in ("occupancy","multiplicity","ebf"):
            for output in iter_filtered_windows(events,WIDTH,HEIGHT,method=arm):
                del output
    calibration = np.array((199.,201.,120.,90.,-.25,.04,.003,-.002,.001))
    build_bearing_map(calibration)
    rodrigues_point_and_jacobian(np.array((.1,.2,1.)),np.zeros(3),.01)
    bearings,times,intrinsics = synthetic_events(1000)
    estimate_rotation(bearings,times,intrinsics,np.zeros(3))
    return dispatcher_signatures()


def audit_application_inputs(events,arm,sequence_index):
    stats,rows = {},[]
    intervals = application_intervals(events)
    for index,(start,stop,sample) in enumerate(application_packets(events,arm,sequence_index,intervals,stats,audit=True)):
        rows.append({"update_index": index,"start_us": start,"stop_us": stop,
                     "input_events": len(sample),"input_payload_sha256": array_hash(sample)})
        del sample
    stats["updates"] = rows
    return stats


def timed_application(events,calibration,arm,sequence_index):
    """The whole resident-source-to-estimates path, including full filter tails."""
    signatures_before = dispatcher_signatures()
    started = time.perf_counter()
    bearing_map = build_bearing_map(calibration)
    lookup_build_seconds = time.perf_counter()-started
    intervals = application_intervals(events)
    n = len(intervals)
    output = np.empty((n,3),np.float64)
    initial = np.empty((n,3),np.float64)
    counts = np.empty(n,np.int64)
    statuses = np.empty(n,np.int64)
    iterations = np.empty(n,np.int64)
    evaluations = np.empty(n,np.int64)
    objectives = np.empty(n,np.float64)
    successes = np.empty(n,np.bool_)
    messages = []
    statistics = {}
    packets = iter(application_packets(events,arm,sequence_index,intervals,statistics))
    previous = np.zeros(3,np.float64)
    collection_seconds,preparation_seconds,optimizer_seconds = 0.,0.,0.
    index = 0
    while True:
        stage_start = time.perf_counter()
        try:
            start,stop,sample = next(packets)
        except StopIteration:
            # next() above consumes all remaining source events, including tail.
            collection_seconds += time.perf_counter()-stage_start
            break
        collection_seconds += time.perf_counter()-stage_start
        stage_start = time.perf_counter()
        bearings = bearing_map[sample[:,2],sample[:,1]]
        times = (sample[:,0]-start).astype(np.float64)/1e6
        preparation_seconds += time.perf_counter()-stage_start
        initial[index] = previous
        result = estimate_rotation(bearings,times,calibration[:4],previous)
        optimizer_seconds += result.elapsed_seconds
        output[index] = result.omega
        previous = output[index].copy()
        counts[index],statuses[index] = len(sample),result.status
        iterations[index],evaluations[index] = result.iterations,result.evaluations
        objectives[index],successes[index] = result.objective,result.success
        messages.append(result.message)
        index += 1
        del sample,bearings,times,result
    elapsed = time.perf_counter()-started
    signatures_after = dispatcher_signatures()
    if signatures_after!=signatures_before:
        raise RuntimeError("new JIT signature during formal timing: trial invalid")
    if index!=n:
        raise AssertionError("application did not emit every common update")
    if n and not np.array_equal(initial[1:],output[:-1]):
        raise AssertionError("arm warm-start history drift")
    stage_sum = lookup_build_seconds+collection_seconds+preparation_seconds+optimizer_seconds
    return {"arm": arm,"total_seconds": elapsed,"statistics": statistics,
            "stage_seconds": {"bearing_map_build": lookup_build_seconds,
                              "source_validation_filter_copy_collection": collection_seconds,
                              "bearing_lookup_and_relative_timestamps": preparation_seconds,
                              "estimator": optimizer_seconds,"other_included_overhead": elapsed-stage_sum},
            "output_sha256": array_hash(output),"input_counts_sha256": array_hash(counts),
            "kernel_signatures_before": signatures_before,"kernel_signatures_after": signatures_after,
            "output_omega_rad_s": output.tolist(),"initial_omega_rad_s": initial.tolist(),
            "input_counts": counts.tolist(),"success": successes.tolist(),"status": statuses.tolist(),
            "iterations": iterations.tolist(),"evaluations": evaluations.tolist(),
            "objective": objectives.tolist(),"messages": messages,
            "function_budget_overshoots": int(np.count_nonzero(evaluations>MAXFUN))}


def recording_quality(trial,targets,intervals):
    rows,errors = [],[]
    for index,((start,stop),target) in enumerate(zip(intervals,targets)):
        omega = np.array(trial["output_omega_rad_s"][index])
        error = None
        if target is not None:
            truth = Rotation.from_quat(target["relative_quaternion_xyzw"])
            predicted = Rotation.from_rotvec(omega*((stop-start)/1e6))
            error = float(np.rad2deg((truth.inv()*predicted).magnitude()))
            errors.append(error)
        rows.append({"update_index": index,"start_us": start,"stop_us": stop,
                     "omega_rad_s": omega.tolist(),"initial_omega_rad_s": trial["initial_omega_rad_s"][index],
                     "input_events": trial["input_counts"][index],"success": trial["success"][index],
                     "status": trial["status"][index],"iterations": trial["iterations"][index],
                     "evaluations": trial["evaluations"][index],"error_degrees": error,
                     "target_eligible": target is not None})
    errors = np.asarray(errors)
    if not len(errors):
        raise ValueError("recording has no independently evaluable mocap targets")
    statuses = np.array(trial["status"])
    return {"updates": rows,"evaluated_updates": len(errors),"application_updates": len(rows),
            "mean_error_degrees": float(errors.mean()),"rmse_error_degrees": float(np.sqrt(np.mean(errors**2))),
            "median_error_degrees": float(np.median(errors)),"p95_error_degrees": float(np.percentile(errors,95)),
            "mean_error_rate_degrees_per_second": float(errors.mean()/.02),
            "solver_non_success_count": int(len(rows)-np.count_nonzero(trial["success"])),
            "fallback_counts": {str(k): int(np.count_nonzero(statuses==k)) for k in (-1,-2,-3)},
            "objective_evaluations": int(sum(trial["evaluations"])),
            "optimizer_iterations": int(sum(trial["iterations"]))}


def run_recording(data_root,sequence_index,implementation):
    sequence = COHORT[sequence_index]
    directory = Path(data_root)/sequence
    metadata = json.loads((directory/"source_metadata.json").read_text())
    validate_source_metadata(metadata,sequence_index)
    # Full source load/hash is outside the timer; all primary runs use RAM arrays.
    events = np.load(directory/"raw_events.npy")
    if not events.flags.c_contiguous or array_hash(events)!=metadata["raw_payload_sha256"]:
        raise ValueError("resident source layout/payload mismatch")
    calibration = np.loadtxt(directory/"calib.txt").ravel()
    if file_hash(directory/"calib.txt")!=metadata["calibration_sha256"]:
        raise ValueError("calibration payload changed")
    if file_hash(directory/"groundtruth.txt")!=metadata["groundtruth_sha256"]:
        raise ValueError("motion-capture payload changed")
    intervals = application_intervals(events)
    targets,exclusions = pose_targets(np.loadtxt(directory/"groundtruth.txt"),intervals)
    print(f"audit {sequence}: {len(events)} source events, {len(intervals)} application updates",flush=True)
    audits = {arm:audit_application_inputs(events,arm,sequence_index) for arm in ARMS}
    if dispatcher_signatures()!=implementation["synthetic_warm_signatures"]:
        raise RuntimeError("real-input audit introduced a signature missing from synthetic warmup")
    trials,orders,quality = {arm:[] for arm in ARMS},[],{}
    for repeat in range(3):
        shift = (sequence_index+repeat)%len(ARMS)
        order = list(ARMS[shift:]+ARMS[:shift])
        orders.append(order)
        for arm in order:
            print(f"timing {sequence} repeat{repeat+1}/3 {arm}: start",flush=True)
            result = timed_application(events,calibration,arm,sequence_index)
            result["repeat_index"] = repeat
            for key in ("source_events","retained_events","application_events","filter_windows"):
                if result["statistics"][key]!=audits[arm][key]:
                    raise AssertionError(f"audit/timed source count mismatch: {arm}/{key}")
            if result["input_counts"] != [row["input_events"] for row in audits[arm]["updates"]]:
                raise AssertionError("audited/timed application inputs disagree")
            if trials[arm] and result["output_sha256"]!=trials[arm][0]["output_sha256"]:
                raise AssertionError("repeated predictions differ; no best repeat can be selected")
            trials[arm].append(result)
            if repeat==0:
                quality[arm] = recording_quality(result,targets,intervals)
            print(f"timing {sequence} repeat{repeat+1}/3 {arm}: {result['total_seconds']:.3f}s",flush=True)
        if source_code_hashes()!=implementation["source_code_sha256"]:
            raise ValueError("implementation changed after scientific execution began")
    timing = {}
    for arm in ARMS:
        values = [r["total_seconds"] for r in trials[arm]]
        timing[arm] = {"seconds": values,"median_seconds": float(np.median(values)),
                       "min_seconds": min(values),"max_seconds": max(values)}
    result = {"sequence": sequence,"source": metadata,"calibration": calibration.tolist(),
              "intervals_us": intervals,"target_exclusions": exclusions,"targets": targets,
              "input_audits": audits,"trial_order": orders,"trials": trials,"quality": quality,"timing": timing,
              "implementation_sha256": implementation["source_code_sha256"],"complete": True}
    write_json(directory/"application_results.json",result)
    print(f"completed {sequence}: "+json.dumps({a:quality[a]["mean_error_degrees"] for a in ARMS}),flush=True)
    return result


def summarize_recordings(recordings):
    metrics = {}
    for arm in ARMS:
        metrics[arm] = {"equal_recording_mean_error_degrees": float(np.mean([r["quality"][arm]["mean_error_degrees"] for r in recordings])),
                        "sum_recording_median_seconds": sum(r["timing"][arm]["median_seconds"] for r in recordings),
                        "evaluated_updates": sum(r["quality"][arm]["evaluated_updates"] for r in recordings),
                        "application_updates": sum(r["quality"][arm]["application_updates"] for r in recordings),
                        "source_events": sum(r["input_audits"][arm]["source_events"] for r in recordings),
                        "retained_events": sum(r["input_audits"][arm]["retained_events"] for r in recordings),
                        "application_events": sum(r["input_audits"][arm]["application_events"] for r in recordings),
                        "objective_evaluations": sum(r["quality"][arm]["objective_evaluations"] for r in recordings)}
    for arm in ARMS:
        metrics[arm]["speed_ratio_raw_over_arm"] = metrics["raw"]["sum_recording_median_seconds"]/metrics[arm]["sum_recording_median_seconds"]
        metrics[arm]["observed_pareto_vs_raw"] = (metrics[arm]["equal_recording_mean_error_degrees"]<=metrics["raw"]["equal_recording_mean_error_degrees"]
                                                  and metrics[arm]["sum_recording_median_seconds"]<metrics["raw"]["sum_recording_median_seconds"])
    return {"metrics": metrics,"rsf_minus_reference_mean_error_degrees": {
                arm:metrics["occupancy"]["equal_recording_mean_error_degrees"]-metrics[arm]["equal_recording_mean_error_degrees"]
                for arm in ARMS if arm!="occupancy"},
            "rsf_better_quality_than_matched_random": metrics["occupancy"]["equal_recording_mean_error_degrees"]<metrics["random"]["equal_recording_mean_error_degrees"],
            "inference_scope": "descriptive three-recording fixed10Hz application; no significance, energy, memory or live-deadline inference"}


def main():
    import argparse
    import platform
    import sys
    import scipy
    from threadpoolctl import threadpool_limits,threadpool_info
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("synthetic","preflight","download","prepare","evaluate"),default="synthetic")
    parser.add_argument("--data-root",type=Path,default=DATA_ROOT)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args = parser.parse_args()
    if args.stage=="synthetic":
        print(json.dumps(synthetic_benchmark(),indent=2))
        return
    if args.stage=="preflight":
        with threadpool_limits(limits=1):
            before = warm_synthetic_kernels()
            runner_key = "native3d_ssm.phase81_frozen_rotation_application[__main__]"
            if not all(before[runner_key+":"+name] for name in ("iwe_and_derivative","rodrigues_point_and_jacobian")):
                raise AssertionError("entry-point synthetic kernels were not warmed")
            i = np.arange(2500,dtype=np.int64)
            events = np.column_stack((i*100,i%100+70,(i//100)%60+50,i%2))
            calibration = np.array((199.,201.,120.,90.,-.25,.04,.003,-.002,.001))
            for arm in ARMS:
                result = timed_application(events,calibration,arm,0)
                if result["kernel_signatures_before"]!=before or result["kernel_signatures_after"]!=before:
                    raise AssertionError("entry-point full-path synthetic signature mismatch")
            print(json.dumps({"passed": True,"synthetic_only": True,"kernel_signatures": before},indent=2))
        return
    assert_freeze()
    if args.stage=="download":
        download_inputs(args.data_root)
        return
    if args.stage=="prepare":
        for index in range(len(COHORT)):
            prepare_recording(args.data_root,index)
        return
    with threadpool_limits(limits=1):
        signatures = warm_synthetic_kernels()
        implementation = {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                          "protocol_sha256": PROTOCOL_SHA256,"source_code_sha256": source_code_hashes(),
                          "timestamp_adapter_sha256": TIMESTAMP_ADAPTER_SHA256,
                          "synthetic_warm_signatures": signatures}
        write_json(args.data_root/"run_implementation_manifest.json",implementation)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        recordings = [run_recording(args.data_root,index,implementation) for index in range(len(COHORT))]
        payload = {"schema_version": 1,"complete": True,"protocol_sha256": PROTOCOL_SHA256,
                   "timestamp_adapter_sha256": TIMESTAMP_ADAPTER_SHA256,
                   "implementation": implementation,"outcome_evaluation_started_utc": started,
                   "prior_execution_attempts": json.loads((args.data_root/"prior_execution_attempts.json").read_text())
                       if (args.data_root/"prior_execution_attempts.json").exists() else [],
                   "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                   "cohort": list(COHORT),"conditions": list(ARMS),"random_seed": RANDOM_SEED,
                   "model": {"name": "fixed-time exact-SO3 contrast-maximization adapter","training": "none",
                             "window_us": WINDOW_US,"stride_us": STRIDE_US,"blur_sigma": BLUR_SIGMA,
                             "blur_mode": "constant zero","blur_truncate": 4.,"omega_component_bound_rad_s": OMEGA_BOUND,
                             "maxiter": MAXITER,"maxfun": MAXFUN,"ftol": FTOL,"gtol": GTOL,
                             "inverse_distortion_steps": 20,"warm_start": "own previous estimate; initial zero"},
                   "environment": {"python": sys.version,"numpy": np.__version__,"scipy": scipy.__version__,
                                   "platform": platform.platform(),"threadpools": threadpool_info()},
                   "timing_scope": "resident original source validation through complete continuous filter tail and motion output; includes lookup build, copies, collection, preparation and optimizer; excludes download/load,JIT,target interpolation,metrics,hashing,JSON",
                   "downloads": json.loads((args.data_root/"download_manifest.json").read_text()),
                   "recordings": recordings,"summary": summarize_recordings(recordings)}
        write_json(args.output,payload)
        print(json.dumps(payload["summary"],indent=2),flush=True)


if __name__ == "__main__":
    main()
