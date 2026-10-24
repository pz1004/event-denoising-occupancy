"""Streaming readers for ED24/DND text files and raw AEDAT 3.1 data."""

from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np
import pandas as pd


LABELED_COLUMNS = ("timestamp", "x", "y", "polarity", "label")


def iter_labeled_text(path: Path, chunk_size: int) -> Iterator[np.ndarray]:
    """Yield ``[t,x,y,p,label]`` int64 chunks from ED24 or DND text files."""

    for frame in pd.read_csv(
        Path(path), sep=r"\s+", dtype=np.int64, chunksize=chunk_size, engine="c"
    ):
        if tuple(frame.columns) != LABELED_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}: {tuple(frame.columns)}")
        yield np.ascontiguousarray(frame.to_numpy(copy=False), dtype=np.int64)


def iter_aedat31(path: Path) -> Iterator[np.ndarray]:
    """Decode raw IBM DVS Gesture AEDAT 3.1 polarity-event packets.

    Each yielded array has columns ``timestamp_us, x, y, polarity``.  This is a
    direct raw decoder and does not depend on Tonic or its cached representation.
    """

    path = Path(path)
    event_dtype = np.dtype([("data", "<u4"), ("timestamp", "<u4")])
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Missing AEDAT end-of-header marker in {path}")
            if line.startswith(b"#!END-HEADER"):
                break

        while True:
            header = stream.read(28)
            if not header:
                break
            if len(header) != 28:
                raise ValueError(f"Truncated AEDAT packet header in {path}")
            (
                event_type,
                _source,
                event_size,
                _timestamp_offset,
                timestamp_overflow,
                event_capacity,
                event_number,
                _event_valid,
            ) = struct.unpack("<HHIIIIII", header)
            payload = stream.read(event_capacity * event_size)
            if len(payload) != event_capacity * event_size:
                raise ValueError(f"Truncated AEDAT packet payload in {path}")
            if event_type != 1:
                continue
            if event_size != 8:
                raise ValueError(f"Unsupported polarity event size {event_size}")
            records = np.frombuffer(payload, dtype=event_dtype, count=event_number)
            data = records["data"]
            valid = (data & 1) != 0
            data = data[valid]
            timestamps = records["timestamp"][valid].astype(np.int64)
            timestamps += int(timestamp_overflow) << 31
            x = ((data >> 17) & 0x1FFF).astype(np.int64)
            y = ((data >> 2) & 0x1FFF).astype(np.int64)
            polarity = ((data >> 1) & 1).astype(np.int64)
            yield np.column_stack((timestamps, x, y, polarity))


def read_gesture_annotations(path: Path) -> List[Dict[str, int]]:
    annotations = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            annotations.append(
                {
                    "class": int(row["class"]),
                    "startTime_usec": int(row["startTime_usec"]),
                    "endTime_usec": int(row["endTime_usec"]),
                }
            )
    return annotations


def read_trial_list(path: Path) -> List[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
