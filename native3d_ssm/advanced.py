"""Motion-conditioned and split-state features for Native3D-SSM v2.

The advanced extractor is deliberately two-stage.  A frozen bootstrap model
scores raw bidirectional state features.  Its signal confidence gates a second
state bank and a velocity-bank time-surface.  Labels are never consumed by the
extractor, so accepted-state features are available under the same conditions
during training and inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .core import StateConfig, bidirectional_features, feature_names

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorate(function):
            return function

        return decorate


DIRECTIONS = np.asarray(
    (
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ),
    dtype=np.int64,
)


@dataclass(frozen=True)
class MotionConfig:
    """Discrete velocity-bank configuration in physical microseconds."""

    delays_us: Tuple[float, ...] = (250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0)
    spatial_strides: Tuple[int, ...] = (1,)
    spatial_steps: int = 3
    tolerance_fraction: float = 0.35
    tolerance_floor_us: float = 100.0

    def __post_init__(self) -> None:
        if not self.delays_us or any(value <= 0 for value in self.delays_us):
            raise ValueError("Motion delays must be positive")
        if tuple(sorted(self.delays_us)) != self.delays_us:
            raise ValueError("Motion delays must be sorted")
        if not self.spatial_strides or any(value <= 0 for value in self.spatial_strides):
            raise ValueError("Motion spatial strides must be positive")
        if tuple(sorted(set(self.spatial_strides))) != self.spatial_strides:
            raise ValueError("Motion spatial strides must be sorted and unique")
        if self.spatial_steps < 2:
            raise ValueError("At least two spatial steps are needed for coherence")
        if self.tolerance_fraction <= 0 or self.tolerance_floor_us <= 0:
            raise ValueError("Motion tolerances must be positive")

    @property
    def pass_feature_dim(self) -> int:
        # same-polarity mean, opposite-polarity mean, and consecutive-step
        # coherence for every direction/delay, plus two global maxima.
        return (
            len(DIRECTIONS)
            * len(self.spatial_strides)
            * len(self.delays_us)
            * 3
            + 2
        )

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["delays_us"] = list(self.delays_us)
        result["spatial_strides"] = list(self.spatial_strides)
        return result

    @classmethod
    def from_dict(cls, values: Dict) -> "MotionConfig":
        values = dict(values)
        values["delays_us"] = tuple(float(value) for value in values["delays_us"])
        values["spatial_strides"] = tuple(
            int(value) for value in values.get("spatial_strides", (1,))
        )
        return cls(**values)


@dataclass(frozen=True)
class AdvancedConfig:
    state: StateConfig = StateConfig()
    motion: MotionConfig = MotionConfig()
    gate_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.gate_threshold < 1.0:
            raise ValueError("gate_threshold must be between zero and one")

    @property
    def feature_dim(self) -> int:
        raw_and_accepted = 4 * self.state.feature_dim
        motion = 2 * self.motion.pass_feature_dim
        bootstrap = 4
        return raw_and_accepted + motion + bootstrap

    def to_dict(self) -> Dict:
        return {
            "state": self.state.to_dict(),
            "motion": self.motion.to_dict(),
            "gate_threshold": self.gate_threshold,
        }

    @classmethod
    def from_dict(cls, values: Dict) -> "AdvancedConfig":
        return cls(
            state=StateConfig.from_dict(values["state"]),
            motion=MotionConfig.from_dict(values["motion"]),
            gate_threshold=float(values["gate_threshold"]),
        )


@njit(cache=True)
def _motion_pass(
    events: np.ndarray,
    update_mask: np.ndarray,
    width: int,
    height: int,
    directions: np.ndarray,
    delays: np.ndarray,
    spatial_strides: np.ndarray,
    spatial_steps: int,
    tolerance_fraction: float,
    tolerance_floor: float,
) -> np.ndarray:
    last_times = np.full((2, height, width), -1, dtype=np.int64)
    direction_count = directions.shape[0]
    delay_count = delays.shape[0]
    stride_count = spatial_strides.shape[0]
    features = np.zeros(
        (events.shape[0], direction_count * stride_count * delay_count * 3 + 2),
        dtype=np.float32,
    )

    for event_index in range(events.shape[0]):
        timestamp = int(events[event_index, 0])
        x = int(events[event_index, 1])
        y = int(events[event_index, 2])
        polarity = int(events[event_index, 3])
        opposite = 1 - polarity
        best_same = 0.0
        best_coherence = 0.0

        for direction_index in range(direction_count):
            dx = int(directions[direction_index, 0])
            dy = int(directions[direction_index, 1])
            for stride_index in range(stride_count):
                stride = int(spatial_strides[stride_index])
                for delay_index in range(delay_count):
                    delay = float(delays[delay_index])
                    same_sum = 0.0
                    opposite_sum = 0.0
                    coherence_sum = 0.0
                    valid_steps = 0
                    valid_pairs = 0
                    previous_same = 0.0
                    have_previous = False

                    for step in range(1, spatial_steps + 1):
                        neighbor_x = x - step * stride * dx
                        neighbor_y = y - step * stride * dy
                        if (
                            neighbor_x < 0
                            or neighbor_x >= width
                            or neighbor_y < 0
                            or neighbor_y >= height
                        ):
                            break
                        valid_steps += 1
                        expected_age = step * delay
                        tolerance = tolerance_floor + tolerance_fraction * expected_age

                        same_last = last_times[polarity, neighbor_y, neighbor_x]
                        same_support = 0.0
                        if same_last >= 0:
                            age = float(timestamp - same_last)
                            same_support = np.exp(-abs(age - expected_age) / tolerance)
                        same_sum += same_support

                        opposite_last = last_times[opposite, neighbor_y, neighbor_x]
                        if opposite_last >= 0:
                            age = float(timestamp - opposite_last)
                            opposite_sum += np.exp(-abs(age - expected_age) / tolerance)

                        if have_previous:
                            coherence_sum += min(previous_same, same_support)
                            valid_pairs += 1
                        previous_same = same_support
                        have_previous = True

                    offset = (
                        (direction_index * stride_count + stride_index)
                        * delay_count
                        * 3
                        + delay_index * 3
                    )
                    if valid_steps > 0:
                        same_mean = same_sum / valid_steps
                        opposite_mean = opposite_sum / valid_steps
                        features[event_index, offset] = same_mean
                        features[event_index, offset + 1] = opposite_mean
                        if same_mean > best_same:
                            best_same = same_mean
                    if valid_pairs > 0:
                        coherence = coherence_sum / valid_pairs
                        features[event_index, offset + 2] = coherence
                        if coherence > best_coherence:
                            best_coherence = coherence

        summary_offset = direction_count * stride_count * delay_count * 3
        features[event_index, summary_offset] = best_same
        features[event_index, summary_offset + 1] = best_coherence
        if update_mask[event_index] > 0.0:
            last_times[polarity, y, x] = timestamp

    return features


def motion_pass_features(
    events: np.ndarray,
    update_mask: np.ndarray,
    state_config: StateConfig,
    motion_config: MotionConfig,
) -> np.ndarray:
    events = np.ascontiguousarray(events, dtype=np.int64)
    mask = np.ascontiguousarray(update_mask, dtype=np.float32)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape [N, 4]")
    if mask.shape != (len(events),):
        raise ValueError("update_mask must have shape [N]")
    if len(events) and np.any(np.diff(events[:, 0]) < 0):
        raise ValueError("Event timestamps must be nondecreasing")
    if np.any(events[:, 1] < 0) or np.any(events[:, 1] >= state_config.width):
        raise ValueError("x coordinate is outside the sensor")
    if np.any(events[:, 2] < 0) or np.any(events[:, 2] >= state_config.height):
        raise ValueError("y coordinate is outside the sensor")
    return _motion_pass(
        events,
        mask,
        state_config.width,
        state_config.height,
        DIRECTIONS,
        np.asarray(motion_config.delays_us, dtype=np.float32),
        np.asarray(motion_config.spatial_strides, dtype=np.int64),
        motion_config.spatial_steps,
        motion_config.tolerance_fraction,
        motion_config.tolerance_floor_us,
    )


def bidirectional_motion_features(
    events: np.ndarray,
    update_mask: np.ndarray,
    state_config: StateConfig,
    motion_config: MotionConfig,
) -> np.ndarray:
    events = np.ascontiguousarray(events, dtype=np.int64)
    mask = np.ascontiguousarray(update_mask, dtype=np.float32)
    if len(events) == 0:
        return np.empty((0, 2 * motion_config.pass_feature_dim), dtype=np.float32)
    forward = motion_pass_features(events, mask, state_config, motion_config)
    reverse = np.ascontiguousarray(events[::-1], dtype=np.int64)
    reverse[:, 0] = int(events[-1, 0]) - reverse[:, 0]
    backward = motion_pass_features(
        reverse, mask[::-1], state_config, motion_config
    )[::-1]
    return np.concatenate((forward, backward), axis=1)


def advanced_block_features(
    events: np.ndarray,
    bootstrap_noise_probability: np.ndarray,
    config: AdvancedConfig,
    raw_features: np.ndarray = None,
) -> np.ndarray:
    """Build raw, accepted, velocity, and bootstrap features for one block."""

    events = np.ascontiguousarray(events, dtype=np.int64)
    probability = np.ascontiguousarray(bootstrap_noise_probability, dtype=np.float32)
    if probability.shape != (len(events),):
        raise ValueError("bootstrap_noise_probability must have shape [N]")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("Bootstrap probabilities must be finite values in [0, 1]")

    gate = (probability < config.gate_threshold).astype(np.float32)
    raw = (
        bidirectional_features(events, config.state)
        if raw_features is None
        else np.ascontiguousarray(raw_features, dtype=np.float32)
    )
    if raw.shape != (len(events), 2 * config.state.feature_dim):
        raise ValueError("raw_features do not match the state configuration")
    accepted = bidirectional_features(events, config.state, gate)
    motion = bidirectional_motion_features(events, gate, config.state, config.motion)
    signal_probability = 1.0 - probability
    entropy = -(
        probability * np.log(np.maximum(probability, 1e-7))
        + signal_probability * np.log(np.maximum(signal_probability, 1e-7))
    )
    bootstrap = np.column_stack(
        (probability, signal_probability, entropy, gate)
    ).astype(np.float32)
    features = np.concatenate((raw, accepted, motion, bootstrap), axis=1)
    if features.shape[1] != config.feature_dim:
        raise RuntimeError("Advanced feature layout does not match its config")
    return features


def motion_feature_names(config: MotionConfig) -> Tuple[str, ...]:
    names = []
    for direction in ("past", "future"):
        for dx, dy in DIRECTIONS:
            for stride in config.spatial_strides:
                for delay in config.delays_us:
                    prefix = (
                        f"{direction}_motion_dx{int(dx):+d}_dy{int(dy):+d}"
                        f"_stride{stride}px_{delay:g}us"
                    )
                    names.extend(
                        (f"{prefix}_same", f"{prefix}_opposite", f"{prefix}_coherence")
                    )
        names.extend((f"{direction}_motion_best_same", f"{direction}_motion_best_coherence"))
    return tuple(names)


def advanced_feature_names(config: AdvancedConfig) -> Tuple[str, ...]:
    base = feature_names(config.state.taus_us, config.state.radius)
    names = []
    for bank in ("raw", "accepted"):
        for direction in ("past", "future"):
            names.extend(f"{bank}_{direction}_{name}" for name in base)
    names.extend(motion_feature_names(config.motion))
    names.extend(
        (
            "bootstrap_noise_probability",
            "bootstrap_signal_probability",
            "bootstrap_entropy",
            "bootstrap_accept_gate",
        )
    )
    return tuple(names)
