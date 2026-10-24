"""Native causal multi-timescale state extraction for event streams.

For every pixel and polarity, the filter stores a bank of exponentially
decaying states.  An incoming event reads the *pre-event* state of its center
and spatial ring, produces features, and only then updates its own state.  The
ordering makes the implementation causal and prevents an event from providing
support to itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - the Python fallback is functional.
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorate(function):
            return function

        return decorate


@dataclass(frozen=True)
class StateConfig:
    """Shape and decay constants of the streaming state bank.

    Timestamps and ``taus_us`` must use microseconds.  The default constants
    cover sub-millisecond support through slower 16 ms motion continuity.
    """

    width: int = 346
    height: int = 260
    taus_us: Tuple[float, ...] = (250.0, 1_000.0, 4_000.0, 16_000.0)
    radius: int = 1
    max_age_factor: float = 20.0
    global_reference_pixels: float = 100_000.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Sensor width and height must be positive")
        if self.radius < 1:
            raise ValueError("radius must be at least one")
        if not self.taus_us or any(tau <= 0 for tau in self.taus_us):
            raise ValueError("All decay constants must be positive")
        if tuple(sorted(self.taus_us)) != self.taus_us:
            raise ValueError("taus_us must be sorted from fast to slow")

    @property
    def feature_dim(self) -> int:
        neighbor_count = (2 * self.radius + 1) ** 2 - 1
        return (7 + 2 * neighbor_count) * len(self.taus_us) + 3

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["taus_us"] = list(self.taus_us)
        return result

    @classmethod
    def from_dict(cls, values: Dict) -> "StateConfig":
        values = dict(values)
        values["taus_us"] = tuple(float(value) for value in values["taus_us"])
        return cls(**values)


def feature_names(taus_us: Iterable[float], radius: int = 1) -> Tuple[str, ...]:
    """Return the stable feature order used by the extractor and checkpoint."""

    taus = tuple(float(value) for value in taus_us)
    groups = (
        "center_same",
        "center_opposite",
        "ring_sum_same",
        "ring_sum_opposite",
        "ring_max_same",
        "ring_max_opposite",
        "global_density",
    )
    names = [f"{group}_{tau:g}us" for group in groups for tau in taus]
    offsets = [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx != 0 or dy != 0
    ]
    for relation in ("same", "opposite"):
        for dx, dy in offsets:
            names.extend(
                f"neighbor_{relation}_dx{dx:+d}_dy{dy:+d}_{tau:g}us" for tau in taus
            )
    names.extend(("log_age_same", "log_age_opposite", "polarity"))
    return tuple(names)


@njit(cache=True)
def _extract_features(
    events: np.ndarray,
    update_weights: np.ndarray,
    states: np.ndarray,
    last_times: np.ndarray,
    global_states: np.ndarray,
    global_last_time: np.ndarray,
    taus: np.ndarray,
    radius: int,
    max_age: float,
    global_scale: float,
) -> np.ndarray:
    event_count = events.shape[0]
    scale_count = taus.shape[0]
    neighbor_count = (2 * radius + 1) ** 2 - 1
    height = states.shape[1]
    width = states.shape[2]
    base_feature_count = (7 + 2 * neighbor_count) * scale_count
    features = np.zeros((event_count, base_feature_count + 3), dtype=np.float32)

    # Feature layout for one event (sc = scale_count, nc = neighbor_count).
    # Every block is scale-minor, i.e. the ``scale_index`` runs fastest, which
    # is exactly the order produced by ``feature_names``:
    #   [0*sc, 1*sc)  center_same          [4*sc, 5*sc)  ring_max_same
    #   [1*sc, 2*sc)  center_opposite      [5*sc, 6*sc)  ring_max_opposite
    #   [2*sc, 3*sc)  ring_sum_same        [6*sc, 7*sc)  global_density
    #   [3*sc, 4*sc)  ring_sum_opposite
    #   [7*sc,               7*sc + nc*sc)  directional neighbors, same polarity
    #   [7*sc + nc*sc, 7*sc + 2*nc*sc)      directional neighbors, opposite
    #   [base, base + 3)  log_age_same, log_age_opposite, polarity
    # Directional blocks are neighbor-major/scale-minor; a neighbor left at zero
    # (out of bounds or never seen) already equals ``log1p(0)``.
    for index in range(event_count):
        timestamp = int(events[index, 0])
        x = int(events[index, 1])
        y = int(events[index, 2])
        polarity = int(events[index, 3])
        opposite = 1 - polarity
        update_weight = float(update_weights[index])

        same_last = last_times[polarity, y, x]
        opposite_last = last_times[opposite, y, x]
        same_age = max_age if same_last < 0 else min(float(timestamp - same_last), max_age)
        opposite_age = (
            max_age
            if opposite_last < 0
            else min(float(timestamp - opposite_last), max_age)
        )

        for scale_index in range(scale_count):
            tau = taus[scale_index]
            if same_last < 0:
                center_same = 0.0
            else:
                center_same = states[polarity, y, x, scale_index] * np.exp(
                    -same_age / tau
                )
            if opposite_last < 0:
                center_opposite = 0.0
            else:
                center_opposite = states[opposite, y, x, scale_index] * np.exp(
                    -opposite_age / tau
                )

            ring_sum_same = 0.0
            ring_sum_opposite = 0.0
            ring_max_same = 0.0
            ring_max_opposite = 0.0
            neighbor_index = 0
            for dy in range(-radius, radius + 1):
                neighbor_y = y + dy
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_x = x + dx
                    if (
                        neighbor_y < 0
                        or neighbor_y >= height
                        or neighbor_x < 0
                        or neighbor_x >= width
                    ):
                        neighbor_index += 1
                        continue

                    neighbor_last = last_times[polarity, neighbor_y, neighbor_x]
                    if neighbor_last >= 0:
                        age = min(float(timestamp - neighbor_last), max_age)
                        value = states[
                            polarity, neighbor_y, neighbor_x, scale_index
                        ] * np.exp(-age / tau)
                        ring_sum_same += value
                        if value > ring_max_same:
                            ring_max_same = value
                        features[
                            index,
                            7 * scale_count + neighbor_index * scale_count + scale_index,
                        ] = np.log1p(value)

                    neighbor_last = last_times[opposite, neighbor_y, neighbor_x]
                    if neighbor_last >= 0:
                        age = min(float(timestamp - neighbor_last), max_age)
                        value = states[
                            opposite, neighbor_y, neighbor_x, scale_index
                        ] * np.exp(-age / tau)
                        ring_sum_opposite += value
                        if value > ring_max_opposite:
                            ring_max_opposite = value
                        features[
                            index,
                            7 * scale_count
                            + neighbor_count * scale_count
                            + neighbor_index * scale_count
                            + scale_index,
                        ] = np.log1p(value)
                    neighbor_index += 1

            if global_last_time[0] < 0:
                global_pre = 0.0
            else:
                global_age = min(float(timestamp - global_last_time[0]), max_age)
                global_pre = global_states[scale_index] * np.exp(-global_age / tau)

            features[index, scale_index] = np.log1p(center_same)
            features[index, scale_count + scale_index] = np.log1p(center_opposite)
            features[index, 2 * scale_count + scale_index] = np.log1p(ring_sum_same)
            features[index, 3 * scale_count + scale_index] = np.log1p(
                ring_sum_opposite
            )
            features[index, 4 * scale_count + scale_index] = np.log1p(ring_max_same)
            features[index, 5 * scale_count + scale_index] = np.log1p(
                ring_max_opposite
            )
            features[index, 6 * scale_count + scale_index] = np.log1p(
                global_pre * global_scale
            )

            # Exact executable recurrence for the declared capped elapsed gap.
            # It agrees with the physical-time ZOH solution until the cap is
            # active. A zero weight leaves both value and reference time
            # unchanged, so a rejected event cannot support a later event.
            if update_weight > 0.0:
                states[polarity, y, x, scale_index] = center_same + update_weight
                global_states[scale_index] = global_pre + update_weight

        features[index, base_feature_count] = np.log1p(same_age / taus[0])
        features[index, base_feature_count + 1] = np.log1p(opposite_age / taus[0])
        features[index, base_feature_count + 2] = float(polarity)
        if update_weight > 0.0:
            last_times[polarity, y, x] = timestamp
            global_last_time[0] = timestamp

    return features


class NativeStateBank:
    """Mutable streaming state with deterministic chunk-equivalent behavior."""

    def __init__(self, config: StateConfig):
        self.config = config
        scale_count = len(config.taus_us)
        self.states = np.zeros(
            (2, config.height, config.width, scale_count), dtype=np.float32
        )
        self.last_times = np.full(
            (2, config.height, config.width), -1, dtype=np.int64
        )
        self.global_states = np.zeros(scale_count, dtype=np.float32)
        self.global_last_time = np.full(1, -1, dtype=np.int64)
        self._taus = np.asarray(config.taus_us, dtype=np.float32)

    def reset(self) -> None:
        self.states.fill(0)
        self.last_times.fill(-1)
        self.global_states.fill(0)
        self.global_last_time.fill(-1)

    @property
    def state_bytes(self) -> int:
        return int(
            self.states.nbytes
            + self.last_times.nbytes
            + self.global_states.nbytes
            + self.global_last_time.nbytes
        )

    def extract(
        self, events: np.ndarray, update_weights: np.ndarray = None
    ) -> np.ndarray:
        """Extract causal features from sorted ``[t, x, y, p]`` events.

        State is retained across calls, so splitting the same ordered stream
        into chunks produces exactly the same features as processing it once.
        """

        events = np.asarray(events)
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError("events must have shape [N, 4] ordered as t, x, y, p")
        if events.size == 0:
            return np.empty((0, self.config.feature_dim), dtype=np.float32)
        if np.any(np.diff(events[:, 0].astype(np.int64, copy=False)) < 0):
            raise ValueError("Event timestamps must be nondecreasing")
        if self.global_last_time[0] >= 0 and events[0, 0] < self.global_last_time[0]:
            raise ValueError("A new chunk cannot precede the retained streaming state")
        x = events[:, 1]
        y = events[:, 2]
        polarity = events[:, 3]
        if np.any(x < 0) or np.any(x >= self.config.width):
            raise ValueError("x coordinate is outside the configured sensor width")
        if np.any(y < 0) or np.any(y >= self.config.height):
            raise ValueError("y coordinate is outside the configured sensor height")
        if np.any((polarity != 0) & (polarity != 1)):
            raise ValueError("Polarity must be encoded as zero or one")

        if update_weights is None:
            weights = np.ones(len(events), dtype=np.float32)
        else:
            weights = np.ascontiguousarray(update_weights, dtype=np.float32)
            if weights.shape != (len(events),):
                raise ValueError("update_weights must have shape [N]")
            if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
                raise ValueError("update_weights must be finite values in [0, 1]")

        maximum_age = self.config.max_age_factor * self.config.taus_us[-1]
        global_scale = self.config.global_reference_pixels / (
            self.config.width * self.config.height
        )
        contiguous = np.ascontiguousarray(events, dtype=np.int64)
        return _extract_features(
            contiguous,
            weights,
            self.states,
            self.last_times,
            self.global_states,
            self.global_last_time,
            self._taus,
            self.config.radius,
            maximum_age,
            global_scale,
        )


def bidirectional_features(
    events: np.ndarray,
    config: StateConfig,
    update_weights: np.ndarray = None,
) -> np.ndarray:
    """Concatenate past-only and future-only native state features.

    The reverse pass remaps timestamps to an increasing clock.  Neither pass
    lets the current event update state before its features are read.  The
    function is block-local by design and therefore has bounded buffering.
    """

    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape [N, 4] ordered as t, x, y, p")
    if len(events) == 0:
        return np.empty((0, 2 * config.feature_dim), dtype=np.float32)
    if update_weights is None:
        weights = np.ones(len(events), dtype=np.float32)
    else:
        weights = np.ascontiguousarray(update_weights, dtype=np.float32)
        if weights.shape != (len(events),):
            raise ValueError("update_weights must have shape [N]")
    forward = NativeStateBank(config).extract(events, weights)
    reverse = np.ascontiguousarray(events[::-1], dtype=np.int64)
    reverse[:, 0] = int(events[-1, 0]) - reverse[:, 0]
    backward_reversed = NativeStateBank(config).extract(reverse, weights[::-1])
    return np.concatenate((forward, backward_reversed[::-1]), axis=1)
