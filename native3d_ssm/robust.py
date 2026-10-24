"""Rate-invariant transforms and checkpoints for the robust Native3D scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .advanced_model import AdvancedEventClassifier
from .analytic import analytic_contrast_score
from .core import StateConfig, feature_names


TRANSFORM_VERSION = "raw_local_global_contrast_v1"


def robust_feature_dim(config: StateConfig) -> int:
    # Remove M rate-magnitude features from each pass, then append four
    # local/global contrasts per time scale and pass.
    scales = len(config.taus_us)
    return 2 * (config.feature_dim - scales + 4 * scales)


def rate_invariant_features(
    raw_bidirectional_features: np.ndarray, config: StateConfig
) -> np.ndarray:
    """Replace direct rate dependence with explicit local/global contrasts."""

    raw = np.ascontiguousarray(raw_bidirectional_features, dtype=np.float32)
    expected = (len(raw), 2 * config.feature_dim)
    if raw.shape != expected:
        raise ValueError(f"raw_bidirectional_features must have shape {expected}")
    scales = len(config.taus_us)
    transformed = []
    for pass_index in range(2):
        values = raw[
            :, pass_index * config.feature_dim : (pass_index + 1) * config.feature_dim
        ]
        global_indices = np.arange(6 * scales, 7 * scales)
        keep = np.ones(config.feature_dim, dtype=np.bool_)
        keep[global_indices] = False
        transformed.append(values[:, keep])
        global_activity = values[:, global_indices]
        for group in (2, 3, 4, 5):
            local_activity = values[:, group * scales : (group + 1) * scales]
            transformed.append(local_activity - global_activity)
    result = np.concatenate(transformed, axis=1).astype(np.float32, copy=False)
    if result.shape[1] != robust_feature_dim(config):
        raise RuntimeError("Robust feature layout does not match its definition")
    return result


def robust_feature_names(config: StateConfig) -> Tuple[str, ...]:
    base = feature_names(config.taus_us, config.radius)
    names = []
    for direction in ("past", "future"):
        names.extend(
            f"raw_{direction}_{name}"
            for name in base
            if not name.startswith("global_density_")
        )
        for group in (
            "ring_sum_same",
            "ring_sum_opposite",
            "ring_max_same",
            "ring_max_opposite",
        ):
            names.extend(
                f"contrast_{direction}_{group}_vs_global_{tau:g}us"
                for tau in config.taus_us
            )
    return tuple(names)


def save_robust_checkpoint(
    path: Path,
    model: AdvancedEventClassifier,
    config: StateConfig,
    metadata: Dict,
) -> None:
    if model.feature_dim != robust_feature_dim(config):
        raise ValueError("Model feature dimension does not match robust transform")
    payload = {
        "format_version": 3,
        "transform_version": TRANSFORM_VERSION,
        "state_config": config.to_dict(),
        "model": {
            "feature_dim": model.feature_dim,
            "hidden_dims": list(model.hidden_dims),
            "state_dict": model.state_dict(),
        },
        "metadata": dict(metadata),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_robust_checkpoint(
    path: Path, map_location: str = "cpu"
) -> Tuple[AdvancedEventClassifier, StateConfig, Dict]:
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format_version") != 3:
        raise ValueError("Unsupported robust checkpoint format")
    if payload.get("transform_version") != TRANSFORM_VERSION:
        raise ValueError("Unsupported robust feature transform")
    config = StateConfig.from_dict(payload["state_config"])
    specification = payload["model"]
    if int(specification["feature_dim"]) != robust_feature_dim(config):
        raise ValueError("Robust checkpoint feature dimension is inconsistent")
    model = AdvancedEventClassifier(
        specification["feature_dim"], specification["hidden_dims"]
    )
    model.load_state_dict(specification["state_dict"], strict=True)
    return model, config, dict(payload.get("metadata", {}))
